# CLI Web 化改造方案

> 状态：已实现（第 6 节的 8 步全部完成）。本文档记录设计结论与最终架构，作为后续维护/扩展的参考；如有偏差应先更新本文档再改代码。

## 1. 背景与目标

- 现状：`agent` 提供一个交互式终端 CLI（`src/bin/cli.rs`）和一套独立部署的多租户 HTTP 服务（`src/api/` + `src/bin/server.rs`）。两者是同一个 `Agent` 核心的两个平行前端，彼此没有依赖关系。
- 需求：**自用**场景下，希望能在浏览器里使用同一份 Agent 能力，且不是"另起一个独立的持久化多租户服务"，而是**同一次 `cli` 进程运行时，额外提供一个 Web 前端**，浏览器页面是这次运行的另一种展示/交互方式，与终端平级、共享同一份会话状态。
- 非目标：不做用户账号体系、不做多租户、不对外公网开放（默认只 bind `127.0.0.1`）。

## 2. 使用模式

`cli` 二进制新增 `--mode <cli|web|both>`（默认 `cli`，完全向后兼容）：

| 模式 | 行为 |
|---|---|
| `cli` | 现状不变，仅终端 REPL |
| `web` | 不起终端 REPL，只起本地 axum server（适合后台常驻、只用浏览器交互） |
| `both` | 终端 REPL + axum server 同时跑，共享同一个 `Arc<Agent>`、同一个 `FileSessionStore`、同一个 `session_id` |

配套 flag/环境变量：
- `--web-port <port>`（或 `AGENT_CLI_WEB_PORT`，风格对齐 `config.rs` 现有约定），默认给一个固定端口。
- Bind 地址固定 `127.0.0.1`，不暴露公网。

## 3. 核心架构结论

### 3.1 `Agent` 本身天然支持并发共享

`Agent`（`src/agent/runtime/mod.rs`）的 `run_continuing`/`run_continuing_stream` 只借用 `&self`，每次调用都新建 `ExecutionContext`，符合其文档注释"own fresh ExecutionContext so concurrent runs never share state"。**终端循环和 Web handler 可以共享同一个 `Arc<Agent>`，不需要为它加锁。**

### 3.2 唯一的共享可变状态：`FileSessionStore` 的读-改-写

终端和网页若同时各发一句话，可能出现"后保存覆盖先保存"（`FileSessionStore` 本身是 last-write-wins，无并发保护）。需要一个 `tokio::sync::Mutex<()>`（或按 `session_id` 分锁）包住"读历史 → 调 Agent → 存历史"这段关键区间，终端循环和 Web handler 共享同一把锁的 `Arc`。

### 3.3 抽取共享的"跑一轮对话"逻辑

把 `cli.rs` 主循环里"取历史 → `run_continuing`/`run_continuing_stream` → 存历史"这段（现状约第 331–370 行）抽成一个独立异步函数，终端循环和 Web handler 都调用它，保证两个前端"跑一轮"的语义完全一致。

### 3.4 危险工具审批：终端 + 网页双通道

现有 `ApprovalCallback`（`src/callback/approval.rs`）**仅供交互式终端**：阻塞 stdin，无超时，文档已明确"用在请求 handler 里会挂起线程"。网页场景需要一个新的、双通道的审批回调（不是替换现有 `ApprovalCallback`，那个继续保留给纯终端/其他库消费者用，例如 `examples/callback_approval.rs`）：

- 用 `tokio::task_local!` 标记"当前这一轮对话从哪个通道发起"：`Terminal` 或 `Web(mpsc::UnboundedSender<ApprovalEvent>)`。在跑一轮对话前设置好这个 task-local。
- **终端分支**：复用现状的阻塞 stdin `y/n` 逻辑。
- **网页分支**：往该次请求专属的 SSE 通道推 `ApprovalRequired { id, tool, arguments }` 事件，然后 `await` 一个 `oneshot::Receiver<bool>`；新增 `POST /approve/{id}` 接口，浏览器点击"批准/拒绝"后 `send` 这个 `oneshot`，回调才继续/短路返回。
- `both` 模式下，终端发起的对话仍在终端确认，网页发起的对话在网页确认——互不干扰，两个通道都真正"能够决策"。

### 3.5 详细过程展示（网页端）

现有 `AgentStreamEvent`（`runtime/mod.rs`）只有 `Token`/`Done`，看不到工具调用细节。需要扩展：

- 新增 `ToolCallsStarted { calls: Vec<{ id, name, arguments }> }` 和 `ToolCallsFinished { results: Vec<{ id, name, status, content }> }`，在 `run_continuing_stream` 循环里、`execute_tool_calls` 调用前后各 `yield` 一次。粒度是"批次级"（一轮里的所有并发工具调用），不改 `execute_tool_calls` 内部的并发执行逻辑，改动可控。
- 浏览器端的 SSE 流实际是两路事件的合并：`run_continuing_stream` 产生的 `Token`/`ToolCallsStarted`/`ToolCallsFinished`/`Done`，以及审批回调独立推送的 `ApprovalRequired`/`ApprovalResolved`——在 `/chat` handler 里用 `futures::stream::select` 合并成一条 SSE 输出。

## 4. Workspace 与前端技术选型（Leptos）

**结论：需要转成 Cargo workspace。** Leptos 前端要编译到 `wasm32-unknown-unknown`（用 `trunk`），跟主二进制（native, tokio/axum）是完全不同的编译目标，不可能共用同一份依赖树。

- 根 `Cargo.toml` 保留现有 `[package]`（`agent` 主 crate 目录不变），追加 `[workspace]`，`members = ["crates/web-ui", "crates/shared"]`（根目录既是 package 又是 workspace，Cargo 原生支持这种布局，`src/`、`examples/`、`Cargo.lock` 都不用挪）。
- `crates/shared`：前后端都要用的类型（`ApprovalEvent`/`ToolCallSummary`/SSE 消息体等 `serde` 结构），native 端和 wasm 端都 `path` 依赖它——这样双端类型只写一遍，不会漂移，这是"统一语言"能拿到的实际好处。
- `crates/web-ui`：Leptos（CSR + `trunk`），依赖 `crates/shared`，通过 `fetch`/`EventSource` 调用主 crate 暴露的 `/chat`（SSE）、`/history`、`/approve/{id}`。
- 主 crate 的 axum server 用 `tower_http::services::ServeDir` 把 `crates/web-ui` 的 `trunk build` 产物（`dist/`）当静态资源伺服；开发时也可以 `trunk serve` 单独跑在另一个端口，靠 `tower-http` 的 `cors` feature 打通跨域。

## 5. 现有多租户 HTTP 代码清理清单

自用场景不需要多租户/鉴权/幂等这套基础设施，全部清理：

**删除文件：**
- `src/api/tenant.rs`（`TenantRegistry`）
- `src/api/handlers.rs`（`AuthenticatedTenant` + `/v1/agent/run`）
- `src/api/idempotency.rs`（幂等缓存）
- `src/api/dto.rs`（`RunRequest`/`RunResponse`）
- `src/api/mod.rs`、`src/api/error.rs`
- `src/bin/server.rs`（独立 HTTP 服务二进制）
- `tenants.example.json`、`tenants.json`

**跟着清理：**
- `src/agent/session.rs` 里的 `MemorySessionStore`（`server.rs` 删除后无人使用；`SessionStore` trait 与 `FileSessionStore` 保留）
- `src/agent/mod.rs` 第 12 行 `pub use session::{FileSessionStore, MemorySessionStore, SessionStore};` 去掉 `MemorySessionStore`
- `src/lib.rs` 第 5 行 `pub mod api;` 移除
- `src/config.rs` 中：`ENV_HTTP_ADDR`/`http_addr()`、`ENV_TENANTS_CONFIG_PATH`/`tenants_config_path()`、`ENV_SESSION_TTL_SECS`/`session_ttl()`、`ENV_IDEMPOTENCY_TTL_SECS`/`idempotency_ttl()` 及对应常量全部删除（`AGENT_CLI_SESSION_DIR`、`MCP_CONFIG_PATH` 等 CLI 用的保留）
- `README.md`：删除"🏢 多租户"/"🌐 HTTP 服务" feature 行、`tenants.example.json` 相关 Quick Start 步骤、`curl localhost:8080/v1/agent/run` 示例、环境变量表对应行、"作为库使用"里提到 `MemorySessionStore` 的措辞、项目结构图里的 `api/` 行；替换为新的 Web 模式说明
- `Cargo.toml`：`axum`/`tower-http` 保留（新方案仍需要），不新增数据库/认证依赖

**保留不动：**
- `examples/multi_tenant.rs`（用的是 `Provider` 本身做并发隔离演示，与 `api::tenant` 无关）
- `src/callback/approval.rs`、`src/callback/path_guard.rs`（终端场景继续用）

## 6. 实施步骤（按优先级）— 完成情况

1. ✅ 清理旧的多租户 HTTP 代码（`src/api/*`、`bin/server.rs`、`tenants.json`、`config.rs` 相关项、`README.md`）
2. ✅ 搭建 Cargo workspace 骨架（根 `Cargo.toml` 追加 `[workspace]`）+ `crates/shared`（SSE/HTTP 线上类型）+ `crates/web-ui`（Leptos + `trunk`）
3. ✅ 抽取 `src/bin/cli/main.rs` 的共享 `run_turn_stream`/`run_turn` 函数 + `turn_lock`（`tokio::sync::Mutex<()>`）
4. ✅ 扩展 `AgentStreamEvent`：新增 `ToolCallsStarted`/`ToolCallsFinished`（`src/agent/runtime/mod.rs`）
5. ✅ 实现终端 + 网页双通道审批回调：`src/callback/dual_approval.rs`（`DualApprovalCallback` + `ApprovalChannel` + `tokio::task_local!`），替代 `cli` 二进制里原来的 `ApprovalCallback`（后者仍保留供其他消费者/示例使用）
6. ✅ 新增 Web 路由：`src/bin/cli/web.rs`（`POST /api/chat`、`GET /api/history`、`GET /api/stream`、`POST /api/approve/{id}`）+ `main.rs` 新增 `--mode <cli|web|both>`/`--web-port`
7. ✅ Leptos 前端（`crates/web-ui/src/main.rs`）：对话气泡 + 持久 `EventSource` 消费 `/api/stream` + 工具调用/结果时间线 + 审批弹窗（批准/拒绝，支持多标签页同步 `ApprovalResolved`）
8. ✅ 更新 `README.md`（新增「🌐 Web UI」小节、配置项、项目结构）与本文档

## 7. 最终架构要点（供后续维护参考）

- **入口**：`src/bin/cli/main.rs`（终端 REPL + `--mode`/`--web-port` 解析、`Arc<Agent>`/`Arc<FileSessionStore>`/`Arc<AsyncMutex<()>>`/`broadcast::Sender<ChatEvent>` 的构造与在两个前端间共享）+ `src/bin/cli/web.rs`（axum 路由，`agent`/`store`/`turn_lock`/`events` 字段与 `main.rs` 共享同一批 `Arc`/`broadcast::Sender` clone）。
- **协议**：`crates/shared`（`ChatEvent`/`ToolCallSummary`/`ToolResultSummary`/`HistoryEntry`/`ApprovalDecision` 等），native 与 wasm 两侧都以 path 依赖引入，双端类型不会漂移。
- **审批**：`src/callback/dual_approval.rs`，`tokio::task_local!` 按"这一轮对话由谁发起"路由到终端 `y/n` 阻塞或浏览器 `PendingWebApproval` 异步通道。
- **实时同步**（见第 8 节修复）：`web::WebState::events`（`tokio::sync::broadcast::Sender<ChatEvent>`），终端循环与 `web::drive_turn` 都往同一个广播里推事件；`GET /api/stream` 是浏览器订阅这个广播的唯一入口，页面加载时用原生 `EventSource` 建立一次持久连接（不是每次发消息才建连接）。
- **前端**：`crates/web-ui`（Leptos CSR + `trunk`），产物 `dist/` 由 `src/bin/cli/web.rs` 用 `tower_http::services::ServeDir` 伺服，`AGENT_CLI_WEB_DIST_DIR` 可覆盖路径。

## 8. 修复记录：`--mode both` 下终端对话未同步到 Web UI

**问题**：最初的实现里，浏览器页面只有加载时的一次性 `GET /api/history`；发消息时才临时建一条 `POST /api/chat` 的 SSE 响应，且只有发起请求的那个标签页自己能读到。终端里输入的对话完全没有推送通道能到浏览器，浏览器也不会在加载时建立任何持久连接。

**根因**：缺一个不区分来源、持续推送给所有浏览器标签页的广播通道。

**修复**：
- `shared::ChatEvent` 新增 `UserMessage { text }` 变体，在一轮对话**开始时**广播（而不是由发起端本地乐观渲染），保证所有标签页（无论是否是发起者）看到的输入来源一致。
- `web::WebState` 新增 `events: broadcast::Sender<ChatEvent>` 字段（`web::new_event_channel()` 创建，容量 256），新增路由 `GET /api/stream`：订阅该广播，转发为 SSE（`event: chat`）。
- `POST /api/chat`（`chat_handler`）不再自己返回一条 SSE 流：改为 `tokio::spawn` 一个 `drive_turn` 任务跑完整轮对话并把每个事件（含 `ApprovalRequired`/`ApprovalResolved`）广播出去，HTTP 响应本身只返回 `202 Accepted`。
- `main.rs` 的终端循环现在持有同一个 `broadcast::Sender<ChatEvent>`（`web_events_tx`），在跑每一轮对话时（流式与 `--no-stream` 两种路径）都调用 `web::to_chat_events`（改为按引用转换，供终端循环和 `web.rs` 共用）把 `AgentStreamEvent` 转成 `ChatEvent` 广播出去。
- 前端 `crates/web-ui`：移除了原来"通过 `fetch` 读取 `POST /api/chat` 响应体、手写按 `\n\n` 分帧解析 SSE"的实现（`EventSource` 不支持带 body 的 POST，这是当初的权宜之计）；改为页面加载时用原生 `EventSource` 打开一次 `GET /api/stream` 长连接（`listen_stream`），`POST /api/chat` 简化为只负责提交输入，不再读响应体；发消息不再本地乐观渲染，等服务端广播的 `UserMessage` 回流展示，所有标签页（包括发起的那个）渲染路径完全一致。

**这次修复只打通了"终端 → 网页"一个方向，见第 9 节。**

## 9. 修复记录：Web 中的对话未同步到终端

**问题**：第 8 节修好了浏览器能看到终端里打的字，但反过来——在网页发消息，终端里毫无反应——仍然是坏的。

**根因**：终端的 REPL 循环只在"自己发起"一轮对话时直接 `print!`/`println!`，从来没有订阅过那条广播通道；`web::drive_turn` 产生的事件只有 `web::stream_handler`（喂给浏览器）在消费。也就是说广播通道本身在第 8 节就已经建好了，但终端侧一直没有接上去。

**修复**：
- `shared::ChatEvent::UserMessage` 新增 `origin: MessageOrigin`（`Terminal` / `Web`）字段：终端渲染时需要知道这条消息是不是自己刚打的字（`reedline` 已经在输入时把它显示出来了，不需要再打印一遍），网页侧则不区分来源，统一渲染。
- `src/bin/cli/main.rs`：`main` 里在起 REPL 循环之前，`tokio::spawn` 一个 `print_chat_events` 任务，`subscribe()` 同一个 `web_events_tx`，作为终端唯一的渲染出口；原来在循环内直接 `print!`/`match AgentStreamEvent` 的那段逻辑整段删除，两个分支（流式/`--no-stream`）现在只管把事件转换、广播出去，不再自己打印。
- `print_chat_events`/`print_chat_event`：对 `UserMessage { origin: Web, .. }` 打印 `[web] <text>`，对 `Terminal` 来源不重复打印；`Token`/`ToolCallsStarted`/`ToolCallsFinished`/`Done`/`Error` 照常打印；`ApprovalRequired`（网页发起的）额外打印一行提示"正在等浏览器决策"（终端发起的危险操作确认走的是 `DualApprovalCallback::prompt_terminal` 自己的阻塞式 `y/n`，不经过这里，不会重复）。
- 效果：现在终端和网页共享**同一条**广播、**同一个**渲染语义——无论从哪一端发起对话，两端都能看到完整过程。

## 10. 待确认/后续可选项

- 是否需要按 `session_id` 做细粒度锁（当前默认全局一把锁，单人单会话场景足够）
- Web 静态资源在生产模式下内嵌进二进制（`rust-embed` 之类）还是运行时从 `dist/` 目录读取，暂定运行时读取，简单够用


