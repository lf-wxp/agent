# Web UI 实现说明与规范

> 定位：本文档是 CLI Web 前端的**实现说明与维护规范**。它描述当前实现"是什么"、
> 各模块之间的契约"是什么"，以及后续改动"应当遵守什么"。历史方案与修复过程不在此展开；
> 若实现与本文档出现偏差，应先更新本文档再改代码（代码内多处注释以 `docs/web-ui.md`
> 为引用目标，本文档即该引用所指向的内容）。

## 1. 定位与范围

- **是什么**：`agent` 的终端 CLI（`src/bin/cli`）在同一次进程运行时，额外提供一个 Web
  前端。浏览器页面是这次运行的另一种展示/交互方式，与终端**平级**、共享同一份会话状态，
  而不是另起一个独立的持久化多租户服务。
- **非目标**：不做用户账号体系、不做多租户、不对外公网开放。服务只 bind `127.0.0.1`。

## 2. 运行模式

`cli` 二进制通过 `--mode <cli|web|both>`（默认 `cli`，向后兼容）选择前端：

| 模式 | 行为 |
|---|---|
| `cli` | 仅终端 REPL，现状不变 |
| `web` | 不起终端 REPL，只起本地 axum server（后台常驻、只用浏览器交互） |
| `both` | 终端 REPL + axum server 同时运行，共享同一个 `Agent`、同一个会话状态 |

配套参数：

- `--web-port <port>`（环境变量 `AGENT_CLI_WEB_PORT`），默认固定端口。
- Bind 地址固定 `127.0.0.1`，不暴露公网。
- 静态资源目录由 `AGENT_CLI_WEB_DIST_DIR` 覆盖，默认指向 `crates/web-ui` 的 `trunk build`
  产物 `dist/`。

## 3. Workspace 与构建产物

前端需编译到 `wasm32-unknown-unknown`（用 `trunk`），与主二进制（native、tokio/axum）目标
不同，因此仓库是 Cargo workspace：

- 根 `Cargo.toml`：既是 `agent` 主 crate 的 `[package]`，又是 `[workspace]`（
  `members = ["crates/web-ui", "crates/shared"]`）。
- `crates/shared`：前后端共用的 wire 类型（`ChatEvent`/`ToolCallSummary`/`HistoryEntry` 等），
  native 端与 wasm 端都以 `path` 依赖引入，保证双端类型只写一遍、不会漂移。它**不依赖**
  `agent` crate（`agent` 会拉入原生依赖），只定义双方一致的线上形状。
- `crates/web-ui`：Leptos（CSR + `trunk`），依赖 `crates/shared`，通过 `fetch`/`EventSource`
  调用主 crate 暴露的 `/api/*` 路由。
- 产物 `dist/` 由 `src/bin/cli/web.rs` 用 `tower_http::services::ServeDir` 伺服；开发时可
  用 `trunk serve` 单独跑在另一端口，靠 loopback CORS（见第 9 节）打通跨域。

## 4. 架构总览

四个关键结论，是理解全部代码的骨架：

1. **`Agent` 天然支持并发共享**：`run_continuing`/`run_continuing_stream` 只借 `&self`，
   每次调用新建 `ExecutionContext`。终端循环与 Web handler 共享同一个 `Arc<Agent>`，无需加锁。
2. **唯一的共享可变状态是会话的读-改-写**：`FileSessionStore` 是 last-write-wins、无并发保护，
   用 `tokio::sync::Mutex<()>`（`turn_lock`）包住"读历史 → 调 Agent → 存历史"这段关键区间。
   当前是全局一把锁（单人单会话场景足够，不按 `session_id` 分锁）。
3. **跨前端的实时同步走同一条广播**：进程内一个 `tokio::sync::broadcast::Sender<ChatEvent>`
   （容量 256，`web::new_event_channel()` 创建），无论对话从终端还是浏览器发起，产生的事件
   都广播到该通道；`GET /api/stream` 是浏览器订阅它的唯一入口。终端自身也订阅它作为唯一渲染出口。
4. **审批属于会话，不属于某个前端**：危险工具审批通过 `ApprovalRegistry`（共享的待决审批表）
   发布，任何一个视图都能回答，先到先得。

## 5. 后端 HTTP 协议（`src/bin/cli/web.rs`）

`WebState` 持有 `main.rs` 构造的同一批 `Arc`/句柄：`agent`、`store`、`turn_lock`、
`session_id`、`approvals`（`Arc<ApprovalRegistry>`）、`approval_callback`、
`approvals_store`、`memory`、`events`（`broadcast::Sender<ChatEvent>`）。这些不是副本，
而是与终端循环共享的同一个对象。

### 5.1 路由契约

| 方法 | 路径 | 请求体 | 响应 | 语义 |
|---|---|---|---|---|
| `GET` | `/api/history` | — | `Vec<HistoryEntry>` | 当前会话完整转录，供新打开/刷新页面渲染 |
| `GET` | `/api/stream` | — | SSE（`event: chat`） | 常驻长连接，订阅广播；`Done`/`Error` 结束一轮，不结束连接 |
| `GET` | `/api/approvals` | — | `Vec<PendingApprovalView>` | 当前待决审批快照（最旧在前），供中途刷新页面发现遗漏的审批 |
| `GET` | `/api/suspended` | — | `Option<SuspendedRunView>`（`null` 表示无） | 当前被挂起、等待恢复的运行及其转录 |
| `POST` | `/api/chat` | `ChatRequest{input}` | `202` + `ChatAccepted{turn}` | 启动一轮对话，立即返回 turn id，不返回任何过程事件 |
| `POST` | `/api/approve/{id}` | `ApprovalDecision` | `204` / `404` | 按工具调用 id 答复审批；成功后广播 `ApprovalResolved` |

静态资源：`ServeDir` 以 `dist/` 的 `index.html` 为 fallback，`.layer(guard_loopback_host)`
只作用于 `/api/*`（静态资源是惰性的，不拦截）。

### 5.2 `POST /api/chat` 的行为细节

- **空白输入**返回 `400`（终端会跳过空行，网页端若接受会广播空 `UserMessage` 并写空转录）。
- **会话内命令**（`shared::commands::parse` 命中）在内联处理，**不进模型、不占 turn lock**，
  与终端一致。回复以 `ChatEvent::SystemNotice` 走共享流广播，所有视图可见；随后补一个立即
  `ChatEvent::Done` 以释放发起 tab 的 composer（见 8.4）。
  - `/resume`、`/discard` 是作用于 turn 的特例，由 `chat_handler` 自行分发（
    `drive_turn(TurnInput::Resume)` / `discard_suspended_run`），其余命令走
    `commands::execute`。
- **已有挂起运行**时，新输入不启动新轮次，而是广播 `SystemNotice` + 立即 `Done`。
- **普通消息**：`tokio::spawn(drive_turn(...))` 跑完整轮，HTTP 只回 `202` + `ChatAccepted`。

### 5.3 `drive_turn` 与事件广播

`drive_turn` 跑一轮 Web 发起的对话，把每个事件（含自己的 `ApprovalRequired`/`ApprovalResolved`）
广播到 `events`，而非作为 HTTP 响应返回。要点：

- 广播 `UserMessage { origin: Web }` 标志本轮开始（发起端不做本地乐观渲染，见 8.4）。
- 审批：`ApprovalChannel::Session(mpsc)` 收到 `PendingApproval` 后，**先** `register` 进
  `ApprovalRegistry`（会话级，终端也能答），**再**广播 `ApprovalRequired`。
- `AgentStreamEvent::Suspended` 先 `record_suspension` 落盘，再广播 `TurnSuspended`（携带
  run id 与 pending 列表），然后 `break`。
- 轮次结束时 `approvals.discard(&raised)` 回收本轮未答复的审批槽位。

`to_chat_events(&AgentStreamEvent, &turn)` 是 `AgentStreamEvent → ChatEvent` 的映射，
`Suspended` 除外（由 driver 处理，因为广播前需落盘拿到 run id）。

## 6. Wire 协议（`crates/shared`）

### 6.1 `ChatEvent`（`GET /api/stream` 的一帧，`data` 字段）

SSE 每帧事件名固定为 `chat`，`data` 是 `ChatEvent` 的 JSON。用
`#[serde(tag = "type", rename_all = "snake_case")]` 区分变体。

| 变体 | 字段 | 含义 / 渲染契约 |
|---|---|---|
| `UserMessage` | `text`, `origin` | 一轮开始；`origin` 指明发起端（`Terminal`/`Web`） |
| `Token` | `text` | 助手输出的一段文本（增量追加） |
| `Reasoning` | `text` | 思维链分片，先于同轮 `Token`/工具调用；默认不入转录 |
| `SystemNotice` | `text` | 进程自身输出（`/help`、清空确认等）；非会话内容，不持久化 |
| `ToolCallsStarted` | `calls: Vec<ToolCallSummary>` | 模型请求的工具调用，即将执行 |
| `ToolCallsFinished` | `results: Vec<ToolResultSummary>` | 上一批工具调用结束 |
| `ApprovalRequired` | `id`,`tool`,`arguments`,`once_only` | 危险工具待审批；用 `POST /api/approve/{id}` 答复 |
| `ApprovalResolved` | `id`,`approved` | 某审批已解决（可能来自其他 tab / 终端） |
| `TurnSuspended` | `turn`,`run`,`pending` | 轮次因审批无人应答而暂停，可 `/resume` |
| `SuspendedRunCleared` | — | 挂起运行已消失（被恢复或放弃） |
| `Done` | `turn`,`budget_exhausted`,`usage?` | 一轮正常结束；`turn` 用于归属到具体轮次 |
| `Error` | `turn?`,`message` | 一轮失败（`turn: None` 仅指编码失败的极罕见帧） |

**归属规则**：`Done`/`Error`/`TurnSuspended` 结束一轮；发起方用 `ChatAccepted::turn` 与这些
帧的 `turn` 字段匹配，区分"本 tab 的轮次结束"与流上其他轮次。

### 6.2 其他 wire 类型

- `HistoryEntry { author, timestamp, content: Vec<HistoryContentItem> }`，镜像
  `agent::agent::Event`。`HistoryContentItem` 变体：`Message{role,content}`、
  `ToolCall{id,name,arguments}`、`ToolResult{id,name,status,content}`、`Reasoning{text}`。
- `PendingApprovalView { id, tool, arguments, requested_at, once_only }`：待决审批的描述
  （不含决策通道，那是 native 侧的 `PendingApproval`）。
- `SuspendedRunView { pending, transcript }`：挂起运行，携带转录（挂起轮不入 `/api/history`，
  只能从这里取）。
- `ApprovalDecision { approved, sticky, reason? }`：`sticky`/`reason` 带 `#[serde(default)]`，
  旧 bundle 按一次性决策、无理由解析。
- `ToolStatus { Success, Error }`、`ToolCallSummary`、`ToolResultSummary`。
- `TurnUsage { input_tokens, output_tokens, cached_input_tokens, reasoning_tokens, cost? }`、
  `TurnCost { amount, currency, partial }`：轮次用量/成本；`partial` 表示有模型无定价、
  `amount` 是下界（显示为 `≥`）。
- `MessageOrigin { Terminal, Web }`。

### 6.3 命令集（`crates/shared/src/commands.rs`）

命令表是单一事实来源，终端与网页共享（`parse` 解析、`help_text` 渲染、`suggestions` 过滤）。
支持：`/help`、`/reset`、`/resume`、`/discard`、`/memory`、`/exit`（`exit` 仅终端；
`available_on_web: false` 的项不会出现在网页 `/help` 与 `/` 菜单中）。命令在轮次开始前处理，
不进模型、不占 turn lock。

## 7. 双通道审批（`src/callback/dual_approval.rs`）

`DualApprovalCallback` 是 `BeforeToolCallback` 的实现，替代纯终端场景的 `ApprovalCallback`
（后者保留给库消费者/示例）。核心原则：

- **审批属于会话，不属于前端**：`ApprovalChannel::Session(mpsc::UnboundedSender<PendingApproval>)`
  把 `PendingApproval` 发布给会话，任何视图都能通过共享的 `ApprovalRegistry` 回答；先到先得。
  `ApprovalChannel::Terminal` 是独立形态（阻塞 stdin），也是未 attach 通道时的 fallback。
- **`with_approval_channel`**（`tokio::task_local!`）给"一轮对话"绑定通道：从发起端一路带到
  该轮触发的每个工具调用的 `call` 里。
- **`ApprovalRegistry`**：`register`（登记 prompt + 决策通道）、`resolve`（首个答复生效）、
  `discard`（轮次结束回收）、`cancel_all`（中断时清空）、`pending_snapshot`（供中途加入的视图
  查询）、`any_pending`（供终端在非本轮时提示有可答审批）。
- **`ApprovalMeta { id, tool, raw_arguments, requested_at, once_only }`**：渲染 prompt 所需的
  描述（`id` 即工具调用自身 id，与 `ToolCallsStarted` 中的一致）。`raw_arguments` 展示原始
  JSON 字符串而非解析结果——malformed 参数也能如实呈现，不会变成 `null`。
- **决策语义**：`ApprovalOutcome { approved, sticky, reason }`。`sticky` 表示"本会话内对该工具
  不再询问"（按会话 scope 记忆，`/reset` 清除）；`reason` 仅在拒绝时记录，供模型读取"该改做
  什么"。拒绝结果统一加前缀 `User denied execution of <tool>:`，把拒绝理由归因于用户而非工具输出
  （否则模型会把理由当作注入内容拒绝采纳）。
- **`WhenUnanswered`**（默认 `Suspend`）：无人应答（超时/无监听/决策被丢弃）时，**不推断答案**。
  默认挂起轮次（落盘，释放 turn lock，问题保持可答），也可配 `Refuse` 直接拒绝。
  `without_standing_approval(tool)` 使该工具的"总是允许"只对本次生效（"总是拒绝"仍记忆）。

## 8. 前端实现（`crates/web-ui`）

### 8.1 状态模型

核心状态在 `ChatState`（`Clone, Copy` 的信号集合）与 App 级信号中：

- `timeline: RwSignal<Vec<TimelineItem>>`：完整对话时间线，唯一渲染数据源。
- `streaming_text: RwSignal<String>`：正在流式输出的本轮文本（独立于 timeline 的"打字气泡"）。
- `live_reasoning: RwSignal<Option<ReasoningBlock>>`：正在接收分片的思考块。
- `turn_active: RwSignal<bool>`：是否有轮次（任意来源）在运行，驱动 thinking 指示。
- `pending: RwSignal<PendingTurn>`：本 tab 自己的提交状态（`Idle` / `Submitting` / `Running(turn)`）。
- `next_id: RwSignal<u64>`：本地单调递增的渲染 key。
- App 级：`lang: RwSignal<Lang>`、`input_value`、`connection: RwSignal<ConnectionState>`、
  `menu_selected`、`menu_dismissed`。

`TimelineItem` 变体（渲染单元）：

| 变体 | 关键字段 |
|---|---|
| `Message` | `id`, `role`（`User`/`Assistant`）, `text` |
| `ToolCall` | `id`, `tool_id`, `name`, `arguments`, `expanded` |
| `ToolResult` | `id`, `tool_id`, `name`, `status`, `content`, `expanded` |
| `Approval` | `id`, `tool_id`, `tool`, `arguments`, `once_only`, `resolved` |
| `Notice` | `id`, `text` |
| `Usage` | `id`, `usage: TurnUsage` |
| `Reasoning` | `id`, `block: ReasoningBlock` |
| `Error` | `id`, `message` |
| `Suspended` | `id`, `pending: Vec<PendingApprovalView>`, `acted` |

`ConnectionState { Connecting, Open, Retrying }` 仅作 header 状态点展示，不门控发送
（`POST /api/chat` 是独立请求，与流连接状态无关）。

### 8.2 组件与渲染函数

`App` 布局：`<header>`（品牌 + `LangSwitch` + 连接状态点）、`<div class="timeline">`（
空状态 / `<For>` 遍历 `timeline` 调 `render_item` / thinking 指示 / 流式气泡）、
`<form class="composer">`（`/` 命令菜单 + textarea + 发送按钮 + 底部提示）。

| 函数 | 职责 |
|---|---|
| `render_item` | 按 `TimelineItem` 变体分发到各渲染器 |
| `render_reasoning` | 思考折叠块（**纯文本 `<pre>`，不走 markdown**） |
| `render_tool_card` | 工具调用/结果卡片（共用布局；超 220 字符默认折叠） |
| `render_suspended` | 挂起运行卡片（`/resume`、`/discard`） |
| `render_approval` | 审批弹窗（批准/拒绝/sticky/理由） |
| `render_streaming_bubble` | 流式"打字中"气泡 |
| `render_thinking_indicator` | 回答前的三点跳动指示 |
| `LangSwitch` | 头部语言选择器 |

### 8.3 启动加载顺序

`main()` 装 panic hook 后 `mount_to_body(App)`。`App` 内：

1. `provide_context(lang)`，`Effect` 同步 `<html lang>`。
2. 单个 `spawn_local` **顺序**执行 `load_history → load_pending_approvals →
   load_suspended_run`。顺序是强制的：待决审批、挂起运行都比转录新，必须后落地，否则
   prompt/挂起卡会渲染到所属对话之上。
3. `listen_stream` 建立一次常驻 `EventSource`（见 8.4），与加载并行。

### 8.4 `EventSource` 分发规则

`listen_stream` 打开 `EventSource::new("/api/stream")`，监听三个事件：

- `chat`：解析 `ChatEvent` → `apply_chat_event`；解析失败提示"页面可能旧于服务端，需 rebuild"。
- `open`：置 `ConnectionState::Open`。
- `error`：置 `Retrying`（`EventSource` 自带重连），并 `close_reasoning`。

`apply_chat_event` 的分发契约（先统一追加 `Reasoning` 分片并 return，其余帧先 `close_reasoning`）：

| 变体 | UI 更新 |
|---|---|
| `Reasoning` | 追加到打开/新建的思考块 |
| `UserMessage` | `turn_active=true`，push `Message(User)` |
| `Token` | `streaming_text` 追加 |
| `SystemNotice` | `turn_active=false`，push `Notice` |
| `ToolCallsStarted` | 每个调用 push `ToolCall` |
| `ToolCallsFinished` | 每个结果 push `ToolResult` |
| `ApprovalRequired` | `push_approval`（按 `tool_id` 去重） |
| `ApprovalResolved` | `resolve_approval` 写回对应审批的 `resolved` |
| `Done` | flush `streaming_text` 为 `Message(Assistant)`、清空、`turn_active=false`、push `Usage`、`budget_exhausted` 则 push `Error`、`turn_finished` |
| `TurnSuspended` | 同 `Done` 收尾 + `drop_unanswered_approvals` + `push_suspended` + `turn_finished` |
| `SuspendedRunCleared` | 移除挂起卡 |
| `Error` | `turn_active=false`、push `Error`、有 `turn` 则 `turn_finished` |

关键约定：**发起端不本地乐观渲染输入**——发送后等 `UserMessage` 经 `/api/stream` 回流再展示，
保证所有 tab（含发起者）渲染路径一致。

### 8.5 输入框（composer）交互规范

- **回车发送 / Shift+回车换行**，且在 `is_composing()`（IME 组合输入，如拼音/假名）时不提交。
- **`/` 命令菜单**：触发与过滤都走 `shared::commands`（与终端一致，`web=true` 过滤不可用命令）。
  - 键盘：`↑`/`↓` 移动（两端回绕）、`Enter`/`Tab` 补全（不发送）、`Esc` 关闭。
  - 鼠标：用 `mousedown` 而非 `click`（避免 textarea blur 先触发导致行消失）；悬停同步高亮。
  - 选中只替换文本，**不自动提交**（对齐终端 reedline：回车选中、再回车发送）。
  - 任何编辑重置 `menu_dismissed`、`menu_selected`。
- **自动增高**：先设 `height: auto` 收缩，再按 `scroll_height` 设回 `"<n>px"`（受 CSS `max-height`
  约束，超出后 textarea 自身滚动）。
- **发送中禁用**：`pending != Idle` 时禁用 textarea 与发送按钮，按钮图标切 spinner。
- **发送**：`send()` 清空输入、置 `Submitting`、`spawn_local(submit_chat)`；成功
  `turn_submitted(turn)`，失败 push `Error` 并回 `Idle`。

### 8.6 审批弹窗规范

- 待决审批是 `TimelineItem::Approval`，来源两路：加载时的 `GET /api/approvals`、流式的
  `ApprovalRequired`，二者经 `push_approval` 按 `tool_id` 去重。
- 四个按钮：批准 / 拒绝 / 总是允许 / 总是拒绝，映射到 `ApprovalDecision { approved, sticky }`。
  拒绝时可附 `reason`。
- `once_only=true` 时**不渲染**"总是允许"按钮（选项会被忽略，展示即是误导）；"总是拒绝"始终保留
  （记忆拒绝只会更安全）。hint 文案按 `once_only` 切换。
- `resolved` 为 `None` 显示按钮组，`Some(_)` 显示已决状态；`ApprovalResolved`（来自其他 tab /
  终端的答复）也写回 `resolved`。
- 提交中禁用所有按钮（`submitting` 局部信号）；提交成功 `resolved.set(Some(approved))`，失败写
  `failure`。

### 8.7 多语言（i18n）规范

见 `crates/web-ui/src/i18n.rs`：

- `enum Lang { Zh, En, Es }` + 封闭 `enum Key` + `t(lang, key) -> &'static str`：翻译表用
  无通配分支的 `match` 实现，**漏翻译在编译期报错**，而非运行时静默回退。
- 检测顺序：`localStorage`（用户手动选过）> `navigator.language` > `Zh`（兜底，保持加 i18n 前的
  默认行为）。
- `Lang` 存于一个 `RwSignal<Lang>`，`App` 用 `provide_context` 注入，渲染函数用
  `i18n::current_lang()`（`expect_context` 封装）取用，不作为参数一路传递。
- **硬性约束**：任何翻译文本必须写成 `{move || t(lang.get(), Key::X)}` 响应式闭包，不能提前算成
  普通值——这样切换语言时，几十轮之前就渲染好的工具卡片/审批按钮/角色标签也会热更新，无需重渲染整棵树。
- **例外**：一次性写入历史的自由文本（如"请求失败：xxx"、预算耗尽提示）只在产生那一刻按当时语言
  格式化一次，之后不随语言切换回溯翻译——与消息内容（用户输入、模型输出）一视同仁。
- 需要插值的文案走独立函数（`request_failed`、`approval_submit_failed`、`server_error_status`、
  `turn_usage` 等），而非 `Key -> &'static str`。

### 8.8 Markdown 渲染规范

见 `crates/web-ui/src/markdown.rs`：

- 入口 `render_markdown(text: &str) -> impl IntoView`，用 `pulldown-cmark`（纯 Rust，可编译到
  wasm）开启 strikethrough / tables / task lists。
- **安全边界**：走 `pulldown_cmark` 的事件流，直接把结构构建成 Leptos 视图（标题、列表、表格、
  代码块等），**不经过 `inner_html`/`dangerously_set_inner_html`**。文本是模型输出，可能被工具
  返回内容（网页、文件）塑造；构建 typed 元素意味着文本里的 `<script>` 只会按字面渲染，绝不作为
  markup 解析。`Event::Html`/`InlineHtml`（markdown 内嵌原始 HTML）同样按字面打印。
- 段内单个换行（soft break）渲染为真实 `<br>` 而非 CommonMark 默认的合并为空格——LLM 输出里的
  裸 `\n` 通常就是"换行"。
- 代码块带复制按钮（Clipboard API）。思考文本刻意走纯文本 `<pre>` 而非 markdown。

### 8.9 样式主题

`crates/web-ui/src/style.css`（暗色 "Terminal Noir" 主题），由 `include_str!("style.css")`
内联。`index.html` 声明 viewport / `theme-color` / `color-scheme: dark` 等 meta。flex 容器
溢出时子项被压扁的坑：`.timeline > * { flex-shrink: 0; }`（子项一旦 `overflow: hidden` 就被判定
自动最小尺寸为 0，父容器溢出收缩时会被压成几像素）。

## 9. 安全约束

- **只 bind `127.0.0.1`**，无鉴权：本机没有第二个用户要防，与 `LOCAL_SCOPE` 的推理一致。
- **DNS rebinding 防护**：`guard_loopback_host` 中间件要求每个 `/api/*` 请求的 `Host`
  （或 HTTP/2 的 URI authority）是回环名（`localhost`/`127.0.0.1`/`::1`），否则 `403`。
  这补上"本机无第二用户"推理覆盖不到的洞：攻击者页面可把受控域名指向 `127.0.0.1` 发起"同源"
  请求，CORS 对此无能为力（浏览器认为同源、不发预检），能暴露它的是 `Host` 头。
- **CORS 仅回环 Origin**：`loopback_cors_layer` 用 `AllowOrigin::predicate` 只放行
  `http://localhost`/`http://127.0.0.1[:port]`/`http://[::1][:port]` 等**纯 HTTP 回环** Origin
  （`https` 明确不放行，因为本前端从不走 TLS）。不做 `AllowOrigin::any`，否则本机浏览器访问的任何
  网站都能驱动这个 agent。
- **前端不信任模型输出为 HTML**：见 8.8，markdown 从不 `inner_html`。
- 不要把服务放到 `0.0.0.0` bind 或公网反代之后而不加真实鉴权：host 检查只防一种特定攻击，不是
  鉴权替代品。

## 10. 构建与开发

- 构建前端：在 `crates/web-ui` 下 `trunk build --release`（产物 `dist/`，gitignored）。
- `AGENT_CLI_WEB_DIST_DIR` 指向已有构建，或直接开发时用 `trunk serve`（其 API 调用打到本服务）。
- `web::dist_warning`：`dist/` 缺失或无 `index.html` 时启动横幅给出去向提示（因为 `ServeDir` 的
  fallback 就是 `index.html`，缺它会让前端所有路由 404）。`/api/*` 无论如何可用，这正是
  `trunk serve` 需要的降级态——但不能是静默的。
- 端口占用在 `web::bind` 阶段就报错（而非 `--mode both` 下被 detach 任务的 `JoinHandle` 吞掉）。

## 11. 硬性约束清单（改动前必读）

- 跨前端同步**只**通过 `broadcast::Sender<ChatEvent>`；新增跨视图可见的内容必须广播，不能只在
  发起端本地渲染。
- `crates/shared` 的 wire 类型是唯一契约；改字段/变体需保证 `#[serde(default)]` 向后兼容
  （旧前端 bundle 是常态，见 `ChatEvent::Done.usage`、`ApprovalRequired.once_only` 的处理方式）。
- 命令必须在 `shared::commands` 登记，否则 `/help` 与 `/` 菜单都不可见；`available_on_web: false`
  的项不会出现在网页端。
- 翻译文本必须走 `Key` 枚举 + `t`，且以 `{move || t(lang.get(), Key::X)}` 闭包形式读取；新增
  `Key` 需同步三种语言，漏翻译是编译错误。
- 渲染模型/工具输出为文本时，一律经 `render_markdown`（结构构建，不 `inner_html`）；思考文本用纯
  文本 `<pre>`。
- 审批答复只能经 `ApprovalRegistry` 共享；"总是允许"在 `once_only` 工具上不记忆、前端不展示该选项。
- `/api/*` 的新路由必须在 `guard_loopback_host` 的 `layer` 之后、语义上仍保持回环-only。
