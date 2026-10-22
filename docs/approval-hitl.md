# 工具审批（HITL）实现说明与规范

> 定位：本文档是工具审批（Human-in-the-loop，HITL）子系统的实现说明与维护规范。它描述当前实现
> "是什么"、各模块之间的契约、以及改动时"必须遵守的不变量"。历史方案、修复过程与实测记录不在此
> 展开；若实现与本文档出现偏差，应先更新本文档再改代码。Web 侧的协议与前端渲染细节见
> `docs/web-ui.md`（第 6/7/8 节），本文档聚焦审批本身的决策、挂起、恢复与持久化。

## 1. 定位与设计原则

工具审批在危险工具执行前拦住它，向人提问，并支持"暂时无法回答 → 挂起整轮 → 之后恢复"的完整
生命周期。三个贯穿一切的设计原则：

1. **审批归属 session，而非某个前端**：终端与任意数量浏览器 tab 都能应答同一个审批，首答生效
   （`ApprovalRegistry::resolve` 用 `remove` 保证）。这是与多数"审批绑定发起端"实现的关键区别。
2. **一切等待有界 + fail-closed**：超时、无人监听、决策通道被丢弃，三种情况都不猜测答案。默认
   语义是**挂起**（`WhenUnanswered::Suspend`），而非拒绝——"没人回答"不等于"人说了不"。
3. **展示用 `raw_arguments` 而非解析后的 `arguments`**：解析失败会显示成 `null`，让人批准一个
   看不见的调用比不问更糟。

## 2. 架构总览

审批子系统横跨四层，各层职责与文件如下：

| 层 | 文件 | 职责 |
|---|---|---|
| 决策模型 | `src/agent/callback.rs` | `ToolCallView` / `ToolCallDecision` / 三个 callback trait |
| 双通道审批 | `src/callback/dual_approval.rs` | `DualApprovalCallback` + `ApprovalRegistry` + 通道路由 |
| 挂起/恢复 | `src/agent/runtime/mod.rs` | `AgentRunState` / `StopReason` / `resume*` / `RunCheckpoint` |
| 持久化 | `src/agent/approval_store.rs` | `ApprovalStore` / `FileApprovalStore` |
| 版本指纹 | `src/agent/fingerprint.rs` | `RunFingerprint` |
| 接入 | `src/bin/cli/{main,web}.rs` + `crates/shared` + `crates/web-ui` | 驱动、路由、命令、前端卡片 |

核心数据流：

1. `Agent::execute_tool_calls` 在每个工具调用前跑 `BeforeToolCallback` 链。
2. `DualApprovalCallback` 命中危险工具 → 经 `ApprovalChannel` 发布 `PendingApproval` 到
   `ApprovalRegistry` → 广播 `ChatEvent::ApprovalRequired` → 等待人答复。
3. 答复（或超时）决定 `ToolCallDecision`：`Proceed`（放行）/ `ShortCircuit`（拒绝，记结果）/
   `Suspend`（挂起）。
4. `Suspend` 使本轮以 `AgentRunState` 落盘（`StopReason::AwaitingDecision`），turn 结束、释放
   turn lock；之后经 `resume` 或 `/resume` 命令恢复、重新询问。

## 3. 审批决策模型（`src/agent/callback.rs`）

- `ToolCallView { tool_call_id, name, arguments, raw_arguments }`：回调裁决的调用视图。
  `arguments` 是解析后的 `Value`（解析失败填 `Value::Null`），`raw_arguments` 是工具将拿到的
  原始字符串。
- `ToolCallDecision` 三态：
  - `Proceed`：放行（或交下一个回调）。
  - `ShortCircuit(ToolResultStatus, String)`：跳过真实工具，以给定内容记为结果。
  - `Suspend`：当前无法决策，停止并交还调用方（不运行工具、无副作用，因此可安全恢复重问）。
- 三个 callback trait：`BeforeToolCallback`（执行前拦截/挂起）、`AfterToolCallback`（执行后改写
  结果）、`BeforeLlmCallback`（每轮请求前改写 `LlmRequest`，不得改 transcript）。
- 回调按注册顺序执行，首个非 `Proceed` 结束链。

**设计边界（不要违背）**：审批状态**不**在 `ContentItem` 上表达。`ContentItem::ToolCall` 是
LLM 请求的直接来源，OpenAI 协议要求每个 `tool_call` 必须有配对 `tool` message，引入 `Pending`
会破坏该不变量；transcript 是 append-only + 全量覆盖写，而审批状态是**可变**的；`ToolResultStatus`
描述执行结果，与"执行前闸门"是正交域。因此审批状态作为**独立旁路实体**持久化，用
`tool_call_id` 与 transcript 关联。

## 4. 双通道审批（`src/callback/dual_approval.rs`）

`DualApprovalCallback` 是 `BeforeToolCallback` 的实现。纯终端场景的 `ApprovalCallback` 保留给
库消费者/示例，未被删除。

### 4.1 审批粒度与 fail-closed

策略表 `rules: HashMap<String, ApprovalRule>`：

- `Always`：该工具任何调用都审批（`new(tools)` 构造这个，等价旧行为）。
- `When(Arc<dyn Fn(&ToolCallView) -> bool>)`：按参数判定，配 `ApprovalRule::when(closure)` 语法糖；
  `with_rule(tool, rule)` 追加/覆盖单条。

**fail-closed 前置规则**（`unreadable_arguments`，在调用谓词**之前**判定，命中即强制审批、完全不
询问谓词）：`raw_arguments` 空/全空白（`Empty`）、JSON 解析失败（`Unparsable`，含
`NaN`/`Infinity`/`-Infinity`，serde_json 直接拒绝）、合法 JSON 但非 object（`NotAnObject`）。
这不仅是保守：谓词典型写法是"路径不在 `/tmp` 下就审批"，畸形载荷会让它读到的每个字段都缺失，
最廉价的绕过就是发一个坏 JSON。`ToolCallView.arguments` 在解析失败时被填为 `Null`，所以
"模型真的传了 `null`"与"解析失败"必须回看 `raw_arguments` 区分。

### 4.2 元数据与决策通道分离

- `ApprovalMeta { id, tool, raw_arguments, requested_at, once_only }`：渲染 prompt 所需的描述。
  `id` 即工具调用自身 id（与 `ToolCallsStarted` 中的一致）。`once_only` 表示"总是允许"对该工具
  只对本次生效。
- `PendingApproval { meta, decision: oneshot::Sender<ApprovalOutcome> }`：发布给会话的完整请求。
- `ApprovalRegistry`：会话级在途审批表，`register` / `resolve`（首答生效，remove 语义）/
  `discard`（轮次结束回收）/ `cancel_all`（中断时清空，唤醒等待侧读成"无人应答"）/
  `pending_snapshot`（供中途加入的视图查询，按 `requested_at` 升序、id 兜底）/ `any_pending`。

元数据与决策通道分开存的理由：决策通道是单次使用、不可克隆、不可序列化；描述要能被反复渲染
（包括中途刷新进来的 tab），且快照读取不应触碰决策通道。

### 4.3 通道路由

`ApprovalChannel::{ Terminal, Session(mpsc::UnboundedSender<PendingApproval>) }`，
`with_approval_channel` 用 `tokio::task_local!` 给"一轮对话"绑定通道（从发起端带到该轮每个
`call`）。未 attach 时回退 `Terminal`（纯终端/库消费者行为不变）。

### 4.4 决策语义

- `ApprovalOutcome { approved, sticky, reason }`：`sticky` 表示"本会话内对该工具不再询问"
  （按会话 scope 记忆，`/reset` 清除）；`reason` 仅拒绝时记录。
- 粘性决策存 `ContinuityCache<HashMap<String, StickyDecision>>`，key 用
  `ExecutionContext::continuity_key()`（天然隔离 terminal/web scope），`put_with` 合并而非替换
  （同会话两个工具各有记忆）。命中记忆在发布事件**之前**短路，不会出现"显示一个已决定的审批"。
  超时**不**被记忆（超时是"没有答案"，不是决策）。
- `WhenUnanswered::{ Refuse, Suspend }`，默认 `Suspend`（挂起并释放 turn lock）。
- `without_standing_approval(tool)`：该工具的"总是允许"只对本次生效（"总是拒绝"仍记忆）。
- 终端/Web 的作用域词汇由 `parse_approval_answer` 一处解析（`y`/`n`/`a`/`d` 及长写法、可选
  `: 理由`），两个 console 读取点共用；返回 `Option` 让调用方决定"无法识别"的含义（console
  fail-closed 拒绝，REPL 拦下重述）。

## 5. 挂起与恢复（`src/agent/runtime/mod.rs`）

### 5.1 三态决策的部分完成

`ToolCallDecision::Suspend` 使一轮不再是全有全无：`ToolRoundOutcome { completed, suspended }` 两个
列表可同时非空——一轮内调用并发执行，其中一个挂起时，兄弟调用可能已产生副作用（结果保留在
`completed`），只有 `completed` 写进 transcript，挂起的调用只保留已有的 `ToolCall` 记录、不发明
结果。

### 5.2 状态与原因

- `SuspendedToolCall { tool_call_id, name, raw_arguments }`：存**原始参数字符串**而非解析后的
  `Value`，保证恢复时工具拿到与审批时文本一致的 payload。
- `StopReason`：
  - `AwaitingDecision`（默认）：hook 无法决策，这些调用**保证一个都没跑**，可安全重问执行。
  - `RoundInFlight`：写入时轮次仍在飞行，每个调用是否已生效**未知**，禁止 resume 重跑。
- `GiveUp::{ Unanswered, ByUser }`：区分"没人应答"与"用户明确放弃"，二者生成的占位结果措辞不同。
- `AgentRunState { fingerprint, suspended, budget_exhausted, reason, context }`：可序列化的挂起
  状态。`context` 是 `pub(crate)`，只有 `resume`/`abandon` 两条路径能取出，且都会先修复 mid-turn
  transcript。

### 5.3 入口函数

| 入口 | 语义 |
|---|---|
| `run_continuing_resumable` / `run_continuing_stream_resumable` | 挂起时返回 `AgentOutcome::Suspended(state)` / 流式末尾 `AgentStreamEvent::Suspended` |
| `run_continuing` / `run_continuing_stream`（原有） | 行为不变：遇到挂起补记为"未答复"并继续 |
| `resume` / `resume_stream` | 恢复；先 `unresumable()` + fingerprint 校验，任一失败即报错/`yield Err` |

`resume*` 的决策参数是 `HashMap<String, ResumedDecision>`（按 `tool_call_id` 键控），
`ResumedDecision::{ Approved, Refused(String) }`。**只对部分调用给条目、其余省略即"部分答复"**，
未给条目的调用继续挂起。`resume_stream` 常被以空 map 调用（交互前端），让未决调用重新走
before-hook 重新提问——"恢复"和"重新询问"是同一个操作。

`execute_tool_calls` 的 `Suspend` 分支查 `resumed`：`Approved` 继续走完后续 hook 链再执行工具
（后置守卫仍有否决权）；`Refused(reason)` 记该 reason 为结果；`None` 仍挂起。

### 5.4 `abandon` 与 `unresumable`

- `AgentRunState::unresumable() -> Option<String>`：`AwaitingDecision → None`；`RoundInFlight →
  Some(解释)`。把"绝不复跑这些调用"这一不变量下沉到 state 本身，而非留给存储方/调用方。
- `AgentRunState::abandon(give_up) -> ExecutionContext`：用占位结果关闭所有 pending 调用 →
  修复 transcript → 返回可继续对话的 context。措辞由 `unanswered_results(give_up)` 按
  `(StopReason, GiveUp)` 组合决定。

### 5.5 `RunCheckpoint`（写前日志）

`RunCheckpoint { save(&AgentRunState), clear(&ExecutionContext) }`，`save` 写入的 state 标记为
`StopReason::RoundInFlight`。调用时机：**每个工具轮次执行前 save、轮次结束后 clear**；挂起时
**保留** checkpoint（挂起记录落到同一 store 并表达更强语义，替换它）。

覆盖可捕获信号处理不到的场景（`SIGKILL`/panic/断电）：进程消失时盘上有 `RoundInFlight` 记录，
下次启动收尾为"是否生效未知"。恢复语义选"收尾为未知"而非"重跑整轮"——重复副作用比报告未知更糟。

## 6. 持久化（`src/agent/approval_store.rs`）

- `ApprovalStore` trait：`put` / `take`（一次性，取出即删）/ `peek`（只读视图）/ `list` /
  `remove`。
- `FileApprovalStore`：每 `(scope, run_id)` 一个 JSON 文件。落盘约定与 `FileSessionStore` 一致
  （base64 编码文件名、临时文件 + 原子 rename、目录 `0700`）。
- `SuspendedRunView { pending, reason, events }` + `awaits_decision()`：展示用的视图（不返回状态
  本身，保证 `take` 的一次性语义不被展示路径破坏）。`awaits_decision()` 是判定"是否可展示为
  等待审批/可恢复"的唯一入口，`RoundInFlight` 不算（它只可能是正在跑的轮次）。
- **与 `SessionStore` 分开**：二者生命周期相反（历史要留，挂起只活到有人回答为止），共用会导致
  `/reset` 顺手丢弃待批准的 run，或清理会话留下孤儿 run。

## 7. 版本指纹（`src/agent/fingerprint.rs`）

`RunFingerprint { state_version, model, instructions_digest, tools }`：挂起的 run 由哪个 agent 配置
产生。恢复前 `mismatch(current)` 校验，任一不符拒绝恢复并报明原因（state 格式 / 模型 / 指令 /
工具缺失），而非静默跑出错误结果。要点：

- `instructions_digest` 用 crate 内固定的 `fnv1a64_hex`（`DefaultHasher` 跨 Rust 版本不稳定），
  prompt 只存摘要不存原文（明文落盘）。
- `tools` 存全名（好报"哪个工具没了"）并排序（注册顺序不是身份的一部分）。
- 新增工具不使挂起失效；`state_version` 在序列化形状不兼容变更时 bump。

## 8. CLI 与 Web 接入（`src/bin/cli/{main,web}.rs`）

### 8.1 共享驱动

`run_turn_stream` / `resume_turn_stream` 在 `main.rs`，被 `drive_terminal_turn`（终端）与
`web::drive_turn` 共用。`resume_turn_stream` 的序列：取 `turn_lock` → `take` 出 run → 依次校验
`unresumable()` 与 fingerprint（失败都 `put` 回去并报错，不销毁状态）→ 广播
`SuspendedRunCleared` → `resume_stream(state, 空决策)`。

`record_suspension` 是挂起版 `record_turn`（只写 `ApprovalStore`，**不**写 session transcript）；
`discard_suspended_run` 先 `cancel_all()`（避免排队等 approval timeout）再 `take` → 广播
`SuspendedRunCleared` → 用 `unanswered_results(GiveUp::ByUser)` 广播 `ToolCallsFinished` →
`abandon(GiveUp::ByUser)` → `record_turn`；`suspended_run_notice` 用 `peek` + `awaits_decision()`
过滤，只有真挂起才阻断新 turn。

### 8.2 启动恢复与信号处理

- `recover_interrupted_round`：`peek` 发现 `RoundInFlight` 残留 → `take` → `abandon(Unanswered)`
  → `record_turn`，自动收尾为"是否生效未知"并写入历史。它在信号处理器安装之前、Web 分支之前
  执行，确保服务器开始服务前残留已被关闭。
- `spawn_interrupt_handler`：处理 `SIGINT`(130)/`SIGHUP`(129)/`SIGTERM`(143)（非 Unix 仅 `ctrl_c`）。
  有 pending 时 `approvals.cancel_all()` 触发优雅挂起（等待侧读成"无人应答"，走普通挂起路径落盘），
  拿 `turn_lock` 等待落盘（`INTERRUPT_GRACE` 5 秒为上限，期间再收信号立即退出），退出码 `128+signum`。
- `announce_suspended_run`：启动横幅提示挂起（仅 terminal；Web 靠 `GET /api/suspended`）。

### 8.3 Web 路由与命令

- `GET /api/approvals`：待决审批快照（`PendingApprovalView` 列表）。
- `GET /api/suspended`：挂起运行（`Option<shared::SuspendedRunView>`，native 视图转 wire 视图，
  `awaits_decision()` 过滤）。
- `POST /api/approve/{id}`：答复审批。
- `/resume` / `/discard` 是 `shared::commands` 里的命令，走 `POST /api/chat`（与终端输入同路），
  由 `chat_handler` 分发到 `drive_turn(TurnInput::Resume)` / `discard_suspended_run`。

## 9. 前端渲染（`crates/web-ui`）

- 待决审批是 `TimelineItem::Approval`，来源两路（挂载时 `GET /api/approvals`、实时
  `ApprovalRequired`），按 `tool_id` 去重；四个按钮映射 `ApprovalDecision { approved, sticky }`，
  `once_only` 时隐藏"总是允许"，拒绝可附 `reason`。
- 挂起是 `TimelineItem::Suspended`，只列待批准调用、**不提供**批准/拒绝按钮（回答发生在 run
  重启之后）；resume/discard 按钮提交 `/resume` `/discard` 命令。
- 事件消费：`TurnSuspended` 结束本轮并显示挂起卡；`SuspendedRunCleared` 移除挂起卡。细节见
  `docs/web-ui.md` 第 6/8 节。

## 10. 安全与归因

- **任何注入 transcript 的人类意图必须可归因**。拒绝结果统一加前缀
  `User denied execution of <tool>: <reason>`——裸理由与工具自身输出不可区分，对齐良好的模型会
  （正确地）把它当注入拒绝采纳。拒绝理由来源优先级：单次 `reason` > `with_rejection_formatter`
  > 内置默认；空白理由被丢弃；粘性拒绝连带记住理由。
- 挂起的 run 含 transcript（可能带敏感工具参数），明文落盘，靠 `0700` 目录防护（与
  `FileSessionStore` 同等级别）。
- 序列化 state 不落 secret：指纹只存 prompt 摘要。

## 11. 不变量清单（改动前逐条核对）

1. **tool_call 与 tool_result 必须配对**：任何落盘的中间态都要满足（`abandon`/`unanswered_results`
   负责补齐；`drive` 在轮次未完成时 `return` 而非继续，保证带空洞的 transcript 不会发到模型）。
2. **首答生效语义不能退化**：`ApprovalRegistry::resolve` 的 remove 语义是其实现基础。
3. **审批超时 + 全路径 fail-closed 必须保留**：这是相对业界的优势，不能在重构中丢掉。
4. **`oneshot::Sender` 不可序列化**：内存 registry 与持久记录分离，持久层只存 `ApprovalMeta`。
5. **`tool_call_id` 必须持久化复用**：不能依赖模型重新生成（重放会变）。
6. **`RoundInFlight` 禁止 resume**：`unresumable()` 下沉到 state；`Agent::resume*` 自己也兜底
   （agent 校验是兜底，调用方先查以保证被拒时不销毁状态）。
7. **恢复的那一轮也必须 checkpoint**：`take` 已移除盘上状态，重新询问期间崩溃会彻底丢失。
8. **checkpoint 与挂起共享存储、以 `reason` 区分**：checkpoint 是轮次前的预写，挂起是轮次后的
   确定性记录，后者替换前者。
9. **恢复在服务器开始服务前完成**：信号处理器与 `recover_interrupted_round` 都排在 Web 分支之前，
   它们属于会话，不属于某个前端。
10. **`take` 是一次性、`peek` 是只读**：展示路径不能变成第二条恢复路径。
11. **一个会话同时只能有一个挂起 run**：存储的 run 持有未入历史的分支，新 turn 会分叉。
12. **`/reset` 与 `--fresh` 清除粘性决策 + 挂起 run**：两者都属于"忘记这段对话"的语义，不该由
    每个前端各自记得。

## 12. 范围边界（明确不做）

### 12.1 `edit`（改写参数后放行）—— 推迟

现有"带理由拒绝"（`n: 路径应该是 /tmp/x`）已让模型自行改参数重试，覆盖大部分场景，`edit` 的
成本当前不划算。若未来要做，已核实的影响面（规范，不是历史分析）：

- 给 `ToolCallDecision` 加变体**几乎无成本**：四个 before-hook 只返回它、从不 `match` 它，全仓
  唯一穷尽 `match` 在 `execute_tool_calls`；难点从来不在返回类型。
- **真正的难点是 transcript 一致性**：`record_tool_calls` 在审批之前就把原始参数写进历史，
  `build_messages` 又原样发回模型——改写后两者不一致，模型会看到"自己的原始参数 + 一个成功结果"
  并认定原始参数已执行。这是正确性问题，不是审计问题。
- **倾向方案**：保留 `record_tool_calls` 照旧，决策阶段收集 `HashMap<call_id, edited_args>`，在
  `execute_tool_calls` 末尾拿到可变借用时就地回填，且**只回填真正执行的调用**（被拒绝的记原始、
  被挂起的由 `SuspendedToolCall` 带走改写值）。执行顺序与并发度不变。
- **已知坑**：给历史追加"修正项"是最差选项——仓库有三处遍历 `ToolCall` 且对同 id 重复项分别取
  "最早/最后/第一个"、语义互相冲突，`build_messages` 会把同一 `tool_call_id` 在一个请求里发两
  次，多数 provider 直接拒绝。
- 额外约束：`SuspendedToolCall.raw_arguments` 必须携带改写值（否则恢复时改写静默丢失）；`edit`
  与 sticky 互斥；未批准的改写不保留。

### 12.2 其他不做项

- **MCP Elicitation 接入**：协议侧能力，现有 `PendingApproval` + session 广播已是可用载体，等
  稳定后再评估。
- **Durable Execution 引擎**：挂起/恢复是自包含改动，无需外部基建；真需要"挂起数天、跨部署"时
  再考虑 Restate。
- **按 `session_id` 细粒度锁**：与本子系统无关，当前全局一把 `turn_lock`（见 `docs/web-ui.md`
  第 4 节）。
