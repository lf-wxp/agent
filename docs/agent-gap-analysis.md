# Agent 能力差距分析与执行计划

> 分析日期：2026-09-28。基于 `main` 分支当时的代码。
> 用途：记录本框架作为通用 agent 还缺哪些能力，并按优先级给出执行项。每完成一项，就更新文末的「执行项」表。

## 1. 结论

这是一个**单 agent 框架，后端只有 OpenAI 兼容接口这一种**。已经做得比较完整的部分：

- 工具调用循环，包括并行工具调用、流式输出和结构化输出；
- 人工审批：能挂起、落盘，恢复前会校验指纹，每轮开始前写 checkpoint；
- 上下文预算：compaction、summarization、eviction 三段式流水线；
- MCP 客户端，支持 stdio 和 Streamable HTTP；
- 会话持久化，以及本地 Web UI。

缺口主要在四个方面：**agent 能主动执行的能力**、**运行时控制**、**多模型支持**、**多 agent 与记忆**。

## 2. 现状对照

| 能力 | 状态 | 说明 / 证据 |
|---|---|---|
| 工具循环、并行工具调用、流式输出 | ✅ | `Agent::drive` / `drive_stream`、`execute_tool_calls`（`join_all`） |
| 结构化输出 | ✅ | `src/llm/structured.rs`、`src/agent/runtime/structured.rs`（`final_answer` 工具，或 `response_format`） |
| 人工审批 / 挂起恢复 / checkpoint | ✅ | `DualApprovalCallback`、`ApprovalStore`、`RunFingerprint`、`RunCheckpoint`；审批时改写参数（`edit`）未做 |
| 上下文压缩、摘要、截断 | ✅ | `src/callback/context_optimizer/`；按模型选择 tokenizer（OpenAI 的新模型用 `o200k_base`，其余用 `cl100k_base` 估算）；摘要默认关闭 |
| MCP 客户端 | ⚠️ | 支持 stdio 和 Streamable HTTP；不支持 SSE；不支持 resources、prompts、sampling、elicitation；不支持 `list_changed` 通知；不会断线重连 |
| 会话持久化 | ⚠️ | 只有 `FileSessionStore` 这一种后端 |
| 写入和执行类工具 | ❌ | 内置工具只能读、删、解压，没有 `write_file`、`edit_file`、shell、`grep` |
| 取消、超时 | ❌ | 没有 `CancellationToken`，也没有整轮超时；单个工具没有超时（MCP `call_tool` 可能一直挂住）；工具并发没有上限；文件工具用同步 `std::fs`，会阻塞 async 运行时 |
| 多厂商支持 | ⚠️ | 已有 `ChatBackend` trait（`Provider::with_backend`），内置 OpenAI 兼容实现；还没有原生的 Anthropic / Gemini 适配（可走它们的 OpenAI 兼容端点） |
| Reasoning / thinking | ✅ | 读取 `reasoning_content` / `reasoning`（`AgentResult::reasoning`、`AgentStreamEvent::Reasoning`），支持 `reasoning_effort`，可选回传（`LLM_REPLAY_REASONING`，DeepSeek 思考模式带 tools 时需要）；推理 token 数尚未统计 |
| 采样参数 | ✅ | `ModelSettings`：`max_tokens`（可选 `max_completion_tokens` 字段）、`temperature`、`top_p`、`reasoning_effort` |
| 多模态输入 | ⚠️ | 用户消息只能是 `String`；MCP 返回的图片会被丢弃（`read_image` 已改走 `Provider`） |
| 规划 / 反思 | ⚠️ | 有 `update_plan` 计划工具（`src/tools/plan/`）；没有反思 / 自检步骤 |
| 多 agent | ⚠️ | 有 agent-as-tool（`AgentTool`，`src/agent/sub_agent.rs`）；handoff、workflow / graph 编排没有 |
| 长期记忆 / RAG | ✅ | SQLite 持久化向量库（`knowledge_base::store`，增量索引、复用已算过的向量）、`retrieve` 工具、跨会话记忆（`memory`：remember / forget / recall + 每轮自动召回）；检索是精确余弦，不是 ANN；没有混合检索和重排 |
| 可观测性 | ✅ | GenAI 约定的 span（`invoke_agent` / `chat` / `execute_tool`），`--features otel` 导出 OTLP；按模型统计用量（含缓存、推理 token），子 agent / 工具 / 摘要的消耗计入所在运行；按价格表估算成本。没有 metrics |
| 类型化错误 | ✅ | `agent::Error`（thiserror，`#[non_exhaustive]`，按调用方能怎么处理分 11 类），公开 API 和可实现的 trait 全部返回 `agent::Result`；库内部不再用 `anyhow`，二进制、示例、集成测试仍用它 |
| Cargo features | ✅ | `cli` / `mcp` / `sqlite` / `zip` / `otel`，默认除 `otel` 外全开；`default-features = false` 时依赖从约 260 个降到约 180 个；CI 检查每个 feature 单独开启和全部关闭 |
| 工具定义体验 | ✅ | `Tool` 只依赖自有的 `ToolDefinition`，与 `async-openai` 解耦；`FnTool` 用参数结构体加 async 闭包定义工具；注册时校验工具名。没有 `#[tool]` 过程宏（讨论后选了泛型构造器） |
| 服务化 | ⚠️ | Web UI 只监听 127.0.0.1，没有鉴权，只支持单用户 |
| 测试 | ⚠️ | CI 跑全部 workspace 成员和示例里的测试，共 900 个（全部 feature：892 个 `--all-targets` + 8 个 doctest；无 feature 时 689 个）；没有真实 LLM 的端到端回归测试（第 14 项）；`llm::complete` 没有单元测试；没有 `web-ui` 的浏览器端测试 |

## 3. 缺口详述（按优先级）

### P0：通用 agent 的基本能力

1. **写入和执行类工具**：需要 `write_file`、`edit_file`（按 search-replace 方式修改）、`run_command`（shell）、`grep`，并接入现有的 `WorkspaceGuardCallback`（工作区沙箱）和 `DualApprovalCallback`（审批）。
2. **取消和超时**：
   - 库层面加 `CancellationToken`；
   - 单个工具统一加超时；
   - 限制同一轮内工具的并发数；
   - 文件 IO 改为 `spawn_blocking` 或 `tokio::fs`。
3. **多模型抽象和模型参数**：
   - 抽象出 Provider trait，支持 Anthropic、Gemini、Ollama；
   - 支持 reasoning；
   - 可以配置 `temperature`、`top_p`、`max_tokens`；
   - 用一张模型能力表代替按关键字判断。
4. **多模态**：（→ 执行项第 13 项）
   - 用户消息支持图片；
   - `read_image` 改为走 `Provider`；
   - MCP 返回的图片透传给模型。

### P1：进阶能力

5. **规划和自我修正**：todo / plan 工具，可选的反思步骤。
6. **多 agent**：先做 agent-as-tool（把 `Agent` 包装成 `Tool`），再考虑 handoff 和 workflow 编排。
7. **长期记忆和 RAG**：持久化向量库（例如 SQLite 或 LanceDB）、`retrieve` 工具、跨会话记忆。
8. **可观测性**：用 `#[instrument]` 加 span，接入 OpenTelemetry，统计成本。
9. **补全 MCP**：SSE 传输、resources 和 prompts、`list_changed`、断线重连；把 Agent 本身作为 MCP Server 对外提供。（→ 执行项第 8、9 项）
10. **审批时改写参数（`edit`）**：方案见 `docs/approval-hitl.md` 12.1。（→ 执行项第 12 项）

### P2：工程化

- 用 `thiserror` 定义类型化错误；
- 用 Cargo features 按需裁剪依赖；
- 公开工具宏，或提供 `#[tool]` 过程宏；
- 让 `Tool` 和 `async-openai` 解耦；
- token 计数按模型选择 tokenizer；
- LLM 重试时读取 `Retry-After`；（→ 执行项第 10 项）
- 提供服务化的 HTTP API 和鉴权；（→ 执行项第 11 项）
- 增加端到端回归评测；（→ 执行项第 14 项）
- 修正 README 中测试数量不一致的问题：徽章写 418，正文写 528。

## 4. 执行项

按建议顺序逐项实现，**每完成一项停下来审批**，再决定下一步。

| # | 执行项 | 范围 | 状态 |
|---|---|---|---|
| 1 | 写入和执行类工具 | `write_file`、`edit_file`、`grep`、`run_command`；接入工作区沙箱、默认审批和上下文 compaction | ✅ 已完成 |
| 2 | 工具超时和取消 | 单个工具统一超时、`CancellationToken`、工具并发上限、文件 IO 不再阻塞运行时 | ✅ 已完成 |
| 3 | LLM Provider 抽象和模型参数 | Provider trait、reasoning、采样参数与 `max_tokens` 可配置、修复 `read_image` 绕过 `Provider` 的问题 | ✅ 已完成 |
| 4 | 子 agent 与规划 | agent-as-tool、todo / plan 工具 | ✅ 已完成 |
| 5 | 可观测性 | span 埋点、接入 OTel、成本统计 | ✅ 已完成 |
| 6 | 持久化检索和长期记忆 | 持久化向量库、`retrieve` 工具、跨会话记忆 | ✅ 已完成 |
| 7a | 工程化：Cargo features | `cli` / `mcp` / `zip` 三个 feature，默认全开；CI 检查各种组合 | ✅ 已完成 |
| 7b | 工程化：工具定义体验 | `Tool` 与 `async-openai` 解耦（自有 `ToolDefinition`）；`FnTool` 泛型构造器（结构体参数 + async 闭包） | ✅ 已完成 |
| 7c | 工程化：类型化错误 | 用 `thiserror` 定义 `agent::Error`，公开 API 全面改为返回 `agent::Result<T>` | ✅ 已完成 |
| 7d | 工程化：小修 | tokenizer 按模型选择（gpt-4o / o 系列用 `o200k_base`）；README 测试数不再写死 | ✅ 已完成，待审批 |

### 后续计划（第 8–15 项）

第 8–11 项在第 7 项讨论时确认，第 12–15 项是 2026-10-22 补充的（原来散落在第 2、3 节和 7c 的遗留里）。都暂未排期。下面每项都列了现状、子任务、开始前要确认的问题和验收标准，用来追踪进度。

| # | 执行项 | 范围 | 规模 | 依赖 | 状态 |
|---|---|---|---|---|---|
| 8 | 补全 MCP | 断线重连、`list_changed` 通知、resources / prompts、SSE 传输（待定） | 中–大 | 无 | ⏳ 待开始 |
| 9 | Agent 作为 MCP Server | 把一个 `Agent` 作为 MCP 工具暴露给其他客户端 | 中 | HTTP 传输的鉴权与第 11 项共用 | ⏳ 待开始 |
| 10 | LLM 重试读取 `Retry-After` | 自己实现聊天和 embedding 的 HTTP 层，按服务端要求的时间等待 | 小–中 | 无 | ⏳ 待开始 |
| 11 | 服务化 HTTP API 与鉴权 | 多用户、带鉴权的对外 API；Web UI 不再只能本机访问 | 大 | 无；第 9 项的 HTTP 传输复用它的鉴权 | ⏳ 待开始 |
| 12 | 审批时改写参数（`edit`） | 审批人修改工具参数后放行，对话记录与实际执行的参数保持一致 | 中 | 无 | ⏳ 待开始 |
| 13 | 多模态输入 | 用户消息带图片；工具（含 MCP）返回的图片交给模型 | 大 | 与第 8 项都改 MCP 结果的处理，先做的一方定接口 | ⏳ 待开始 |
| 14 | 真实 LLM 的端到端回归评测 | 一组固定任务，用真实模型定期跑，按通过率判断是否退化 | 中 | 需要 API key 和预算 | ⏳ 待开始 |
| 15 | 补全 CI 覆盖 | CI 补上示例里的测试、`shared` / `web-ui` 的测试和 wasm 端 clippy | 小 | 无 | ✅ 已完成（15.5 可选项未做） |

**建议顺序**：15 → 14 → 10 → 8 → 12 → 13 → 9 → 11。

- 第 15 项最小，补上之后后面每一项都受益；做的时候发现 CI 漏掉的不止示例（见第 15 项现状）。
- 第 14 项给后面几项一张用真实模型兜底的安全网：第 10 项要换掉 HTTP 层，第 13 项要改消息类型，都只靠假服务测试不够放心。它需要先定 API key 和预算，定不下来就往后挪。
- 第 10 项改动集中在两个后端，和其他项互不依赖，直接改善被限流时的表现。
- 第 8 项只改现有的 MCP 客户端。
- 第 12 项方案已有（见 `docs/approval-hitl.md` 12.1），改动集中在审批链路。
- 第 13 项最好排在第 8 项之后：两者都要改 MCP 返回内容的处理。
- 第 9 项要先决定「非交互场景下需要审批的工具怎么办」，第 11 项也要用到这个决定。
- 第 11 项最大；它的鉴权方案定下后，第 9 项的 HTTP 传输可以直接复用。

**状态约定**：⏳ 待开始 → 🚧 进行中 → ✅ 已完成，待审批 → ✅ 已完成。

- 开始一项时，把上表的状态改为 🚧，并先和你确认该项的「开始前要确认」。
- 每完成一个子任务，就勾选对应的复选框。
- 做完后在下方「执行记录」补一节，并同步更新第 2 节「现状对照」。

#### 第 8 项：补全 MCP

**现状**
- `src/tools/mcp/client.rs`：
  - `McpConnection::spawn` 走 stdio（`TokioChildProcess`），`McpConnection::connect` 走 Streamable HTTP；
  - 只用到 `list_all_tools` 和 `call_tool`；
  - 客户端处理器是 `()`，服务端发来的通知（工具列表变化、日志、进度）全部被忽略。
- `src/tools/mcp/config.rs`：`mcp.json` 里写 `"type": "sse"` 时直接报错。
- **rmcp 3.5.1 已不提供旧版 HTTP+SSE 客户端传输**（`src/transport/` 下只有 Streamable HTTP 客户端），要支持 SSE 只能自己实现。MCP 规范从 2025-03-26 版起已用 Streamable HTTP 取代 HTTP+SSE。
- **工具集是静态的**：
  - 工具列表只在 `ToolRegistry::with_mcp_over` 连接时取一次；
  - `Agent` 持有不可变的 `Arc<ToolRegistry>`；
  - 运行指纹包含工具名：挂起的运行恢复时，新增工具没有影响，少了工具会被拒绝。
- **不会重连**：子进程退出或 HTTP 会话失效后，对应的 `McpTool` 每次调用都会失败，直到进程重启。

**子任务**
- [ ] 8.1 断线重连：
  - 调用因连接断开而失败时，按原配置重连再重试一次；
  - 能检测到 stdio 子进程已退出；
  - 重连次数和退避有上限，重连失败时返回 `Error::Mcp`。
- [ ] 8.2 `list_changed`：实现 `ClientHandler::on_tool_list_changed`，收到后重新 `list_all_tools`，让新的工具集生效（生效方式见「开始前要确认」）。
- [ ] 8.3 resources：`list_resources` / `read_resource`。
- [ ] 8.4 prompts：`list_prompts` / `get_prompt`，在 CLI 里给一个入口（例如 `/prompt`）。
- [ ] 8.5 SSE 传输：决定做不做；做的话自己实现客户端（GET 建立事件流，向服务端给出的端点 POST 消息）。
- [ ] 8.6（可选）sampling / elicitation：处理服务端反向发来的「调用模型」「向用户要输入」请求；elicitation 可以接到现有的审批界面。
- [ ] 8.7 文档、`examples/mcp_*` 和测试同步更新。

**开始前要确认**
- 工具列表变化后怎么生效：
  - 让 `Agent` 的工具集可以热替换；
  - 或者只在下一次运行时生效。
  - 两种做法都要考虑和运行指纹的关系。
- resources 给谁用：
  - 做成工具交给模型（例如 `<server>__read_resource`）；
  - 或者只作为 API 给调用方读取。
- SSE 值不值得做：旧协议的服务器在减少，自己实现还要长期维护。

**验收标准**
- 用 `examples/shared` 的 demo server 验证：
  - 杀掉子进程后，下一次调用自动重连并成功；
  - 服务端增删工具后，下一轮请求里的工具定义随之变化；
  - resources 和 prompts 能列出并读取。
- 6 种 feature 组合的 clippy 和测试全部通过；关掉 `mcp` feature 时编译不受影响。

**不在本项范围**：MCP 返回的图片透传给模型，放到多模态一起做（见第 2 节「多模态输入」）。

#### 第 9 项：Agent 作为 MCP Server

**现状**
- `AgentTool`（`src/agent/sub_agent.rs`）已经能把 `Agent` 包装成工具，但只能给同一进程里的另一个 agent 用：参数是一个 `task`，每次调用都是一次全新的运行。
- rmcp 的服务端部分（`server`、`macros`、`transport-io`、Streamable HTTP 服务端）只在 `[dev-dependencies]` 里，给 `examples/mcp_server.rs`、`examples/mcp_http_server.rs` 和 `examples/shared/demo_server.rs` 用。
- **审批模型不匹配**：需要审批时运行会挂起（`AgentOutcome::Suspended`），而 MCP 的 `tools/call` 是一次请求对应一次响应。

**子任务**
- [ ] 9.1 新增 feature（暂名 `mcp-server`），把 rmcp 的服务端 feature 从 dev-dependencies 移进来；不开这个 feature 时，库的依赖不变。
- [ ] 9.2 `AgentMcpServer`（暂名）实现 `ServerHandler`：
  - 每个 `Agent` 暴露成一个 MCP 工具，参数和 `AgentTool` 一致；
  - `tools/call` 执行一次运行，把回答作为文本返回；
  - 失败时设置 `is_error`。
- [ ] 9.3 取消：把 MCP 的 `notifications/cancelled` 接到运行的 `CancellationToken`。
- [ ] 9.4 进度：把 `AgentStreamEvent`（工具调用、计划更新）转成 `notifications/progress`。
- [ ] 9.5 审批：按「开始前要确认」定下的方式处理需要审批的工具。
- [ ] 9.6 传输：
  - stdio 必做；
  - Streamable HTTP 需要鉴权：可以等第 11 项定下鉴权后再做，也可以先做最简单的 bearer token。
- [ ] 9.7 可执行入口（例如 `cli --mcp-server`，或单独的示例 / 二进制），能直接配置进 Claude Desktop、Cursor 等 MCP 客户端。
- [ ] 9.8 文档和测试。

**开始前要确认**
- 需要审批的工具怎么办，三选一：
  - 默认拒绝；
  - 用 elicitation 让 MCP 客户端那边的用户确认；
  - 只暴露不需要审批的工具集。
- 会话：每次调用都是新运行（和 `AgentTool` 一致），还是支持带会话 ID 的连续对话。
- 暴露粒度：只暴露 agent 本身，还是把它的工具也一起暴露出去。
- 用量：每次调用的 token 用量要不要随结果返回（例如放在 `_meta` 里）。

**验收标准**
- 用本 crate 自己的 MCP 客户端（`McpConnection::spawn`）连上它：
  - `tools/list` 能看到 agent 工具；
  - `tools/call` 能拿到回答；
  - 客户端取消后，服务端的运行被取消。
- 在一个真实的 MCP 客户端里手动验证一次。
- 不开 `mcp-server` feature 时，默认构建的依赖数不变。

#### 第 10 项：LLM 重试读取 `Retry-After`

**现状**
- `src/llm/retry.rs`：`with_retry` 用 `backon` 做指数退避，完全不看 `Retry-After`（文档里写了这一限制）。
- **根本原因**：`async-openai` 的 `ApiErrorResponse` 只保留状态码和错误体，响应头在错误到达我们之前就被丢掉了。用到它的地方有两处：
  - `OpenAiCompatible`（`src/llm/backend.rs`），通过 `create_byot` / `create_stream_byot`；
  - `OpenAiEmbedder`（`src/knowledge_base/embed.rs`）。
- **有利条件**：
  - `backon` 1.6 有 `adjust(|err, default_delay| ...)`，可以按错误修改下一次的等待时间；
  - `ProviderError` 是自有类型（7c），可以直接加字段；
  - `ChatRequest::to_body`、`parse_response`、`parse_chunk` 已经基于原始 JSON，自己发 HTTP 不需要重写解析；
  - `crate::http::client()` 已有共享的 reqwest 客户端。

**子任务**
- [ ] 10.1 `ProviderError` 增加 `retry_after: Option<Duration>`，以及 `with_retry_after` / `retry_after()`。
- [ ] 10.2 `with_retry` 用 `adjust`：
  - 有 `retry_after` 时，等待 `max(retry_after, 默认退避)`；
  - 超过上限时不再重试，直接返回错误。
- [ ] 10.3 聊天请求改用 reqwest 自己发：
  - 实现 `OpenAiCompatible::create` / `create_stream`，流式部分自己解析 SSE；
  - 从响应头读 `Retry-After`，支持秒数和 HTTP 日期两种格式；
  - 部分服务（如 Azure OpenAI）还会发 `retry-after-ms`，一并支持。
- [ ] 10.4 embedding 请求同样处理。
- [ ] 10.5 行为保持不变：
  - 错误分类（状态码 → 能否重试）不变；
  - `OpenAIConfig` 的配置方式不变（base URL、key、组织、项目、自定义头）；
  - `OPENAI_*` 环境变量照常生效。
- [ ] 10.6 文档（去掉 `retry.rs` 里的「已知限制」）和测试。

**开始前要确认**
- 自己实现 HTTP 层，还是给 `async-openai` 提 PR、让它保留响应头：
  - 自己实现工作量更大，但不受上游节奏影响；
  - 提 PR 改动小，但要等上游合并和发版。
- 自己实现之后，`async-openai` 基本只剩请求 / 响应的类型定义，要不要顺便减少对它的依赖。
- 等待上限设多少（例如 60 秒）；超过上限时是直接失败，还是按上限等待后再试。

**验收标准**
- 本地假服务返回 `429` 加 `Retry-After: 2`：第二次请求在约 2 秒后才发出。
- `Retry-After` 超过上限时不再重试。
- 流式和非流式、聊天和 embedding 四种情况都有测试。
- 错误分类、reasoning 字段、缓存 token 归一化相关的现有测试全部通过，发出的请求体不变。

#### 第 11 项：服务化 HTTP API 与鉴权

**现状**
- Web 服务（`src/bin/cli/web.rs`）只是 CLI 的一部分：
  - 只绑定 `127.0.0.1`，靠 CORS 和 Host 头检查防 DNS rebinding，**没有任何鉴权**；
  - 路由（`/api/chat`、`/api/history`、`/api/stream`、`/api/approvals`、`/api/approve/{id}`、`/api/suspended` 等）都面向单个会话（`--session`）和单个作用域（`LOCAL_SCOPE = "local"`）；
  - 用全局的 `turn_lock` 串行处理每一轮。
- **库里已为多用户留好接口**：
  - `SessionStore`、`ApprovalStore`、`MemoryStore` 都按 `scope` 隔离；
  - `Provider` 可以按租户各建一个，凭证和并发预算互相隔离（`examples/multi_tenant.rs`）；
  - `Agent::run_structured_raw` 接受运行时传入的 JSON Schema。
- 会话存储只有文件实现（`FileSessionStore`），多个实例之间无法共享。

**子任务**
- [ ] 11.1 设计 API：
  - 资源和路由覆盖会话、消息、流式事件（SSE）、审批、挂起的运行、结构化输出；
  - 决定请求 / 响应类型放在哪里（是否复用 `crates/shared`）。
- [ ] 11.2 鉴权：
  - API key / bearer token 映射到用户（即 `scope`）；
  - key 的存储和吊销；
  - 每个请求只能访问自己 scope 下的会话、审批和记忆。
- [ ] 11.3 独立的服务入口：新二进制（暂名 `server`）加对应 feature，不和 CLI 的本机 Web UI 混在一起；CLI 的本机模式保持不变。
- [ ] 11.4 多用户并发：
  - 每个会话一把锁，取代全局 `turn_lock`；
  - 每个用户的并发数和速率限制。
- [ ] 11.5 部署相关：
  - 监听地址可配置；
  - TLS，或明确要求部署在反向代理之后；
  - CORS 白名单、请求体大小限制、健康检查接口。
- [ ] 11.6 工具安全：服务端以什么权限、在哪个工作区执行 `run_command` 和文件写入类工具。
- [ ] 11.7（视需要）共享存储：基于 SQLite 或 Postgres 的 `SessionStore` / `ApprovalStore`。
- [ ] 11.8（后续）Web UI 能连接远程服务（登录或填写 token）。
- [ ] 11.9 文档、部署说明、端到端测试。

**开始前要确认**
- 多用户下文件和 shell 工具怎么隔离：每个用户一个工作区目录，还是服务端默认关闭这些工具。
- 鉴权做到什么程度：静态 API key 列表，还是接入 OIDC 或外部身份系统。
- 要不要支持多实例部署：这决定共享存储（11.7）是否必须做。

**验收标准**
- 两个用户用各自的 key，只能看到自己的会话、审批和记忆。
- 不带 key 或 key 错误时返回 `401`。
- 挂起 → 审批 → 恢复、以及取消的完整流程都能通过 HTTP 跑通。
- 有基于本地假模型服务的端到端测试。

#### 第 12 项：审批时改写参数（`edit`）

**现状**
- 审批只有「批准」和「带理由拒绝」两种结果：
  - 钩子返回 `ToolCallDecision`（`Proceed` / `ShortCircuit` / 挂起），恢复时是 `ResumedDecision`（`Approved` / `Refused(String)`）；
  - Web 端协议 `ApprovalDecision`（`crates/shared`）只有 `approved`、`sticky`、`reason` 三个字段。
- 现在要改参数，只能拒绝并在理由里写清楚（例如 `n: 路径应该是 /tmp/x`），让模型自己重新调用一次，多花一轮。
- 设计已经分析过（`docs/approval-hitl.md` 12.1，当时因为「带理由拒绝」已够用而推迟），结论：
  - 给决定类型加一个变体几乎没有成本，难点在**对话记录的一致性**；
  - `record_tool_calls` 在审批之前就把原始参数写进了记录，`build_messages` 又会原样发回模型。如果执行的是改写后的参数，模型会看到「自己的原始参数 + 一个成功结果」，以为原始参数已经执行过。这是正确性问题。
  - 往记录里追加一条「更正」是最差的做法：同一个 `tool_call_id` 会在一个请求里出现两次，多数 provider 直接拒绝。

**子任务**
- [ ] 12.1 决定类型：`ToolCallDecision` 和 `ResumedDecision` 各加一个带新参数的变体（暂名 `Edited(Value)`）。
- [ ] 12.2 回填记录：按 12.1 的倾向方案，审批阶段收集 `call_id → 新参数`，在 `execute_tool_calls` 末尾**只为真正执行了的调用**回填记录；被拒绝的保留原始参数。
- [ ] 12.3 挂起与恢复：`SuspendedToolCall.raw_arguments` 带上改写后的值，否则恢复时改写会悄悄丢失；没被批准的改写不保留。
- [ ] 12.4 校验：新参数必须是 JSON object；是否按工具的 schema 校验见「开始前要确认」；校验失败时让审批人重新填写，而不是直接执行。
- [ ] 12.5 `edit` 与「总是允许」（sticky）互斥：一次改写只针对这一次调用。
- [ ] 12.6 归因：模型要能知道参数被人改过（例如工具结果前加一句「参数已由用户修改为 …」），和拒绝理由的 `User denied …` 前缀同一个思路（见 `docs/approval-hitl.md` 第 10 节）。
- [ ] 12.7 前端：
  - 终端：在审批提示里加一个「编辑」选项，用预填好原参数的输入行修改；
  - Web UI：审批卡片里可编辑的 JSON 文本框；
  - `ApprovalDecision` 加 `arguments: Option<Value>`，带 `#[serde(default)]`，旧版前端照常能用。
- [ ] 12.8 文档（`docs/approval-hitl.md` 12.1 改为已实现、README）和测试。

**开始前要确认**
- 归因的写法：只在工具结果里注明，还是同时改写记录里的调用参数（推荐两者都做：记录里是实际执行的参数，结果里说明是用户改的）。
- 新参数要不要按工具的 JSON Schema 严格校验：要的话需要引入 JSON Schema 校验库。
- MCP 工具和 `AgentTool` 是否同样允许改写。

**验收标准**
- 改写后执行的调用：对话记录、发回模型的请求、落盘的会话三处都是改写后的参数，结果里注明了修改。
- 被拒绝的调用保留原始参数。
- 挂起 → 在 Web UI 里改参数 → 恢复，执行的是改写后的值。
- 同一轮里有多个并发调用、只改其中一个时，其他调用不受影响。

#### 第 13 项：多模态输入

**现状**
- 消息和工具结果都只有文本：`ContentItem::Message { content: String }`、`ContentItem::ToolResult { content: String }`，`Tool::execute` 返回 `crate::Result<String>`，`Agent::run` 只接收 `&str`。
- MCP 工具返回的图片被替换成 `[image content omitted]`（`render_content`，`src/tools/mcp/client.rs`），音频和内嵌资源同样被丢弃。
- `read_image` 的做法是绕开：单独发一次视觉请求，只把文字描述放进对话（`src/tools/read_image/`）。
- **协议限制**：Chat Completions 的 tool 消息只能放文本（`ChatCompletionRequestToolMessageContentPart` 只有 `Text`），图片只能放在 user 消息里（`ImageUrl`）。所以工具返回的图片要另外附一条 user 消息。
- 没有模型能力表：不知道当前模型是否支持图片，发过去不支持的话只能靠 provider 报错。
- 会话以 JSON 文件保存，图片若以 base64 直接写进去，文件会迅速变大。

**子任务**
- [ ] 13.1 内容类型：消息内容支持「文本 + 图片」多段，图片可以是 URL 或数据（data URL / 字节 + MIME 类型）。旧的纯文本记录要能继续读取。
- [ ] 13.2 输入入口：`Agent` 提供接收多段内容的运行方法（`run` / `run_continuing` 的多模态版本），原来接收 `&str` 的方法保持不变。
- [ ] 13.3 发送：组装请求时把图片转成 `image_url` 片段；工具返回的图片放进紧跟在 tool 消息之后的 user 消息，并说明来自哪个工具调用。
- [ ] 13.4 工具返回图片：扩展 `Tool` 的返回值（或新增可选方法），MCP 工具把 `ContentBlock::Image` 交出来，不再丢弃。
- [ ] 13.5 存储：图片按内容哈希存成单独的文件，会话记录里只存引用；`/reset` 和删除会话时一起清理。
- [ ] 13.6 上下文预算：图片按固定成本估算 token，压缩和裁剪时能把旧图片换成占位文字。
- [ ] 13.7 模型能力：配置项或模型表说明是否支持图片；不支持时给出明确错误，或自动退回 `read_image` 的做法。
- [ ] 13.8 前端：
  - Web UI：上传、粘贴图片，消息里显示缩略图，历史记录里能看到；
  - 终端：用命令或路径语法附加本地图片（例如 `/image path`）。
- [ ] 13.9 文档和测试。

**开始前要确认**
- `ContentItem` 怎么改：
  - 给 `Message` 增加可选的附件字段，旧数据不受影响；
  - 或者把 `content` 改成多段内容，改动更彻底，但所有读 `content` 的地方都要跟着改。
- `Tool` 的返回值怎么扩展：改 `execute` 的返回类型（不兼容变更），还是新增一个带默认实现的方法。
- 图片存在哪里：会话目录下按哈希存文件，还是交给调用方提供的存储。
- 支持哪些格式和大小上限；音频、PDF 是否一起做（建议先只做图片）。

**验收标准**
- 用户发图片提问，请求里带上了 `image_url`，回答基于图片内容（本地假服务校验请求格式，再用真实视觉模型手动验证一次）。
- MCP demo server 返回图片的工具，图片出现在下一次请求里，而不是 `[image content omitted]`。
- 带图片的会话保存后重启能完整恢复，会话文件里没有 base64 图片数据。
- 旧版本写出的会话文件照常能读。

#### 第 14 项：真实 LLM 的端到端回归评测

**现状**
- 所有测试都用假的模型：库内用 `ScriptedBackend`（`src/llm/testing.rs`），集成测试（`tests/`）用本地 HTTP 假服务。它们验证的是我们发出的请求和对响应的处理，不能说明换了提示词、工具描述或 HTTP 层之后，真实模型是否还能完成任务。
- 现有的 `gaia` 二进制能用真实模型跑 GAIA 基准并打印正确率，但：
  - 数据集是需要申请权限的 HF 数据集（`HF_TOKEN`），题目难、耗时长、费用高；
  - 只打印结果，不和上一次比较，也不作为失败条件。
- CI 没有任何需要 API key 的任务。
- 覆盖不均：`llm::complete` 没有单元测试，`llm::stream` 只有 4 个。

**子任务**
- [ ] 14.1 任务集：在仓库里放一组固定的小任务（例如 `evals/` 目录），覆盖：
  - 纯问答、单次和多次工具调用、并行工具调用；
  - 结构化输出（两种模式）、流式输出、子 agent、`update_plan`；
  - 挂起 → 审批 → 恢复、取消；
  - 上下文压缩（长对话）、长期记忆的保存与召回、`retrieve`。
- [ ] 14.2 判定：每个任务用程序判断结果，例如答案包含某个值、调用了某个工具、结构化输出能解析、结果文件内容正确；尽量不用另一个模型打分。
- [ ] 14.3 运行器：新二进制或 `cargo make eval`：
  - 并发和总费用有上限；
  - 每个任务可跑多次，按通过率判断，减少模型随机性带来的误报；
  - 输出 JSON 结果（通过率、耗时、token 用量和估算费用，复用第 5 项的用量统计）。
- [ ] 14.4 和上一次比较：保存基线，通过率下降或费用明显上升时报告。
- [ ] 14.5 CI：手动触发和定时（例如每晚）的 workflow，API key 放在仓库 secrets 里，不在 PR 上自动跑。
- [ ] 14.6 （可选）多模型矩阵：同一组任务在 2–3 个模型上跑（例如 DeepSeek、GPT-4o-mini、一个本地模型）。
- [ ] 14.7 顺便补上 `llm::complete` 和 `llm::stream` 的单元测试。
- [ ] 14.8 文档：怎么在本地跑、怎么加任务、怎么更新基线。

**开始前要确认**
- 用哪个 provider 和模型、每次运行的费用上限是多少。
- 多久跑一次：每晚、每次合并到 `main`，还是只手动触发。
- 判定「退化」的标准：通过率低于固定阈值，还是比基线下降超过一定比例。
- 是否把 GAIA 的一个小子集纳入（需要 `HF_TOKEN`，题目更难更贵）。

**验收标准**
- `cargo make eval`（或对应命令）在本地用一个 API key 能跑完全部任务，输出通过率和费用。
- 故意改坏一处（例如删掉某个工具的描述），评测能报告退化。
- CI 里的评测 workflow 能手动触发，且没配 secrets 时直接跳过而不是失败。

#### 第 15 项：补全 CI 覆盖

**现状**（`.github/workflows/ci.yml` 有 `fmt`、`clippy`、`test`、`features`、`test-minimal` 五个任务）
- **示例里的测试没人跑**：`cargo test` 会编译示例，但不会编译其中的 `#[cfg(test)]` 模块；`examples/rag.rs` 里有 5 个测试，只有 `cargo test --examples` 才会跑（7c 时就是这样漏掉一处的）。
- **`shared` 和 `web-ui` 的测试没人跑**：CI 的 `cargo test --all-features` 只跑根 crate（852 个），加上 `--workspace` 是 885 个，差的 33 个是这两个 crate 的。
- **wasm 端 clippy 不在 CI 里**：`Makefile.toml` 的 `ci` 任务包含 `clippy-wasm`，GitHub Actions 里却没有对应任务。
- **README 有一处说法不对**：README 说「用 `cargo make test` 或 `cargo test --workspace` 跑全部测试」，但 `Makefile.toml` 设了 `default_to_workspace = false`，`cargo make test` 实际只跑根 crate。

**子任务**
- [x] 15.1 测试任务改为同时覆盖示例和整个 workspace（例如 `cargo test --workspace --all-features --all-targets`，再单独跑 doctest；或者在 `[[example]] rag` 上设 `test = true`）。
- [x] 15.2 新增 wasm 任务：安装 `wasm32-unknown-unknown` target，跑和 `cargo make clippy-wasm` 相同的检查。
- [x] 15.3 `Makefile.toml` 的 `test` 任务同步修改，让本地和 CI 跑的是同一组。
- [x] 15.4 修正 README 里关于 `cargo make test` 覆盖范围的说法。
- [ ] 15.5（未做）（可选）前端构建检查：在 CI 里跑一次 `trunk build`，发现 `crates/web-ui` 编译不过的问题。

**开始前要确认**
- 示例测试用 `--examples` 还是 `test = true`：前者对以后新加的示例自动生效，后者只对标记过的示例生效。
- 要不要做 15.5：需要在 CI 里安装 `trunk`，会多花一些时间。

**验收标准**
- 在示例的测试里故意写一个失败的断言，CI 会失败。
- 在 `crates/shared` 里故意写一个失败的测试、或在 `crates/web-ui` 里留一个 clippy 警告，CI 会失败。
- `cargo make ci` 和 GitHub Actions 跑的检查一致。

### 执行记录

#### 第 1 项：写入和执行类工具

- 新增了 4 个内置工具（在 `src/tools/` 下）：
  - `write_file`：写入或覆盖文件，会自动创建父目录。
  - `edit_file`：按精确字符串替换。`old_string` 必须在文件中只出现一次，否则需要显式设置 `replace_all`。
  - `grep`：按正则搜索文件内容。支持 `include` 文件名通配；会跳过隐藏目录、`target`、`node_modules`、二进制文件和大文件；结果条数有上限。
  - `run_command`：通过 `sh -c`（Windows 上是 `cmd /C`）执行命令。
    - stdin 置空，避免抢占 CLI 终端的输入；
    - 有超时，默认 120 秒，最长 600 秒；超时后在 Unix 上会杀掉整个进程组；
    - 输出过长时保留开头和结尾，截掉中间部分；
    - 可选参数 `workdir`。
- `WorkspaceGuardCallback` 现在也会检查这些新工具的路径参数：`write_file` / `edit_file` 的 `file_path`、`grep` 的 `path`、`run_command` 的 `workdir`。
  - 注意：`run_command` 的命令字符串本身无法做路径校验，只能靠审批把关。
- CLI 默认需要审批的工具从 `delete_file` 扩大为 `delete_file`、`write_file`、`edit_file`、`run_command`。
- `Compaction` 默认会压缩 `grep` 的历史结果（它是只读的、可以重跑）。写入和执行类工具明确排除在外，并有测试保证。
- `gaia` 基准评测只用一组固定的工具（`calculator`、`web_search`、`list_files`、`read_file`、`unzip_file`、`read_image`），不含写入、执行和删除类工具：评测无人值守，没有人可以审批。这不是原来 `builtin()` 的完整集合（原来还含 `delete_file`），评测条件的差别仅此一项。
- 代码审查后的调整见 `docs/code-review-tools-and-cancellation-2026-10-19.md`。其中改变了对外行为的几处：
  - `ToolRegistry::builtin()` 不再含 `write_file`、`edit_file`、`run_command`；完整集合改为 `builtin_all()`，CLI 使用后者。
  - 同一轮里有副作用的工具按调用顺序执行（`Tool::concurrency_safe`，默认 `false`）。

#### 第 2 项：工具超时和取消

- **单个工具超时**：
  - 新增 `tools::tool::execute_with_timeout`。`ToolRegistry::execute`（无状态路径）和 `Agent::execute_tool_calls` 都通过它执行工具。
  - 默认上限由 `AGENT_TOOL_TIMEOUT_SECS` 控制，默认 300 秒；也可以用 `Agent::with_tool_timeout` 设置。
  - 工具可以覆盖 `Tool::timeout()` 声明自己的上限。`run_command` 设为 610 秒，比它内部的 600 秒上限多留 10 秒余量。
  - 超时后，该调用作为错误结果交给模型，整轮继续。
- **工具并发上限**：
  - 默认值由 `AGENT_TOOL_MAX_CONCURRENCY` 控制，默认 8；也可以用 `Agent::with_tool_concurrency` 设置。
  - 用 `Semaphore` 实现，只在工具真正执行期间持有名额，等待审批的调用不占名额。结果仍按调用顺序返回。
  - 无状态的 `tool_loop` 使用同一个上限。
- **取消**：
  - **传入 token**：新建运行用 `Conversation::with_cancellation(CancellationToken)`；恢复挂起的运行用 `AgentRunState::with_cancellation`。token 存在 `ExecutionContext.cancellation` 里，不参与序列化。
  - **取消后的行为**：
    - 正在进行的模型请求立即放弃。流式输出是逐个 chunk 检查的，所以回答说到一半也能停下。
    - 正在执行的工具被中断，结果记为"已取消，可能已部分生效"。
    - 还没开始执行的调用直接记为"已取消，未执行"，也不会再弹出审批。
  - **结束方式**：运行以 `agent::Cancelled` 错误结束，里面带着截至此刻的完整记录，调用方可以 downcast 取出后落盘。
  - **已知限制**：
    - 已经在等待人工审批的 before-hook 不会被打断，它依赖自己的审批超时。强行丢弃会让各个前端上的审批弹窗悬空。
    - 结构化输出的入口（`run_structured*`）目前没有地方传入 token。
    - 摘要等 before-LLM 钩子现在会随取消而中止（在各自的 `.await` 点让出）；一段不让出的同步 CPU 计算无法被打断。
- **不阻塞运行时**：`simple_tool!` 宏生成的同步工具（`read_file`、`list_files`、`delete_file`、`unzip_file`）改为放到 `spawn_blocking` 上执行。
  - 超时只能让调用方停止等待，已经在阻塞线程池里运行的同步工具会在后台继续跑完；因此对有副作用的工具，超时提示不会说"已放弃"，而是提醒先检查当前状态再重试。`grep` 例外：它在遍历中检查取消标志，调用被放弃后会很快停下。
- **CLI 未改动**：Ctrl-C 仍按原来的方式把这一轮挂起后退出。

#### 第 3 项：LLM Provider 抽象和模型参数

- **后端抽象**（`src/llm/backend.rs`）：
  - 新增 `ChatBackend` trait，只有 `create` 和 `create_stream` 两个方法。它和上层之间传递的仍是 OpenAI Chat Completions 的请求和响应类型：几乎所有厂商都直接支持这种格式，或提供兼容端点。参数是 `ChatRequest`（标准请求，加上要回传的推理内容），没有可回传内容时用 `.into()` 从标准请求转换。
  - 内置实现是 `OpenAiCompatible`，即原来的 `async-openai` 客户端。
  - `Provider` 改为持有 `Arc<dyn ChatBackend>`，并新增 `Provider::with_backend(backend, 并发上限)`。`Provider::new(OpenAIConfig, n)` 保持不变。
  - 所有模型调用统一走 `Provider::chat` / `chat_stream`，这两个方法在调用期间占用该租户的并发名额。
  - 原来的 `Provider::client()` 已删除，因为自定义后端没有 `async-openai` 客户端可以返回。
  - 自定义后端可以用 `llm::retry::Permanent::wrap` 标记“重试也没用”的错误，`is_transient` 会识别，避免白白走完整条退避曲线。
  - 响应结构解析失败时，错误里带上截断到 300 字符的原始响应体。
- **推理内容**：
  - 给 `async-openai` 开启了 `byot` feature，用来读取原始 JSON。`reasoning_content`（DeepSeek、Qwen、Kimi、vLLM）或 `reasoning`（OpenRouter、Ollama）中有哪个就取哪个，然后再按标准结构解析。
  - 非流式运行通过 `AgentResult::reasoning` 返回最终回答那一轮的推理；中间工具轮次的推理只记在 debug 日志里。
  - 流式运行在回答之前推送 `AgentStreamEvent::Reasoning` 事件。
  - 默认不写入 transcript，也不会发回给模型。
  - **回传开关** `ModelSettings::replay_reasoning`（`LLM_REPLAY_REASONING`，默认关闭）。依据 DeepSeek 官方文档：思考模式下请求带 `tools` 时，历史 assistant 消息必须带回 `reasoning_content`（同一次运行的工具轮次和之后的轮次都要），否则返回 400；而 OpenAI 等提供商不接受这个字段，所以不能默认开启。
  - 开启后：`Agent` 在 transcript 里记录新增的 `ContentItem::Reasoning`（位于它所属消息之前），`build_messages` 把它挂到后面的 assistant 消息上，`OpenAiCompatible` 把它写成该消息的 `reasoning_content`；`tool_loop::run` 和流式 `llm::stream` 在一次调用内的各轮之间同样回传。非流式、流式、结构化的两条路线都覆盖。
  - 为保证推理内容不和所属消息分开：`find_safe_start` 和驱逐的头部边界不会把二者切开；摘要的文本里不包含推理；孤立的（所属消息已被裁掉的）推理在构建请求时丢弃，不会挂到别的消息上；关闭开关后，transcript 里已有的推理也不再发送。
  - Web 历史接口返回 `Reasoning` 项（`HistoryContentItem::Reasoning`），页面刷新后以折叠状态显示；只有开启回传时 transcript 里才有。
  - 无状态的 `tool_loop::Completion` 也新增了 `reasoning` 字段。
  - 注意：开启 `byot` 后，原始的流式调用不会再自动设置 `stream: true`，现在由 `OpenAiCompatible` 自己设置。`tests/openai_backend.rs` 有端到端测试覆盖这一点。
- **模型参数**（`src/llm/settings.rs`）：
  - 新增 `ModelSettings`，包含 `max_tokens`、`token_limit_field`、`temperature`、`top_p`、`reasoning_effort`、`replay_reasoning` 六项。
  - 默认值来自环境变量 `LLM_MAX_TOKENS`、`LLM_TOKEN_LIMIT_FIELD`、`LLM_TEMPERATURE`、`LLM_TOP_P`、`LLM_REASONING_EFFORT`、`LLM_REPLAY_REASONING`。取值无效（无法解析、为 0、越界、不认识的名字）时打 warn 日志并忽略，不会让进程报错。`ModelSettings::from_config()` 每个进程只读取一次并缓存，所以警告也只出现一次；需要在 `.env` 加载之后再调用。
  - 采样参数没有设置时，请求里就不带这个字段，因为部分推理模型一看到 `temperature` 就报错。
  - `token_limit_field` 用来让 OpenAI 推理模型改用 `max_completion_tokens` 字段。
  - `request_builder` 和 `tool_loop::run` 的参数从 `max_tokens: u32` 改为 `&ModelSettings`，这是对外 API 的变更。
  - 在 `Agent` 上用 `with_model_settings` / `model_settings` 设置和读取。
  - 结构化输出的两条路线都用 `ModelSettings::at_least` 把上限至少提到该路线的预算（`final_answer` 工具路线是 `DEFAULT_MAX_TOKENS`，`response_format` 路线是各模式自己的预算），但不会把用户配置的更大上限调低。截断错误里报告的是实际发出的上限值。
  - `ModelSettings::cap_only()` 只保留输出上限和承载它的字段，用于模型不是这套设置所针对的那一个的请求（摘要、`read_image`）。
  - `DEFAULT_MAX_TOKENS` 移到 `llm::settings`，`llm::client` 里仍然 re-export，旧路径可用，也消除了 `settings` 与 `client` 的循环引用。
- **`read_image` 改走 `Provider`**：
  - 改为手写的 `ReadImageTool`。`ReadImageTool::default()` 在执行时才取 `Provider::shared()`，`ReadImageTool::new(provider)` 则使用指定的租户。
  - 视觉请求现在会经过租户限流和重试，只带 `ModelSettings` 的输出上限（`cap_only`），不带 `temperature`、`top_p`、`reasoning_effort`：模型名由 LLM 在调用时指定，未必支持这些字段。文件读取改为 `tokio::fs`。
  - `ToolRegistry` 的 `builtin` / `builtin_all` / `select` 里的 `read_image` 用的是 `Provider::shared()`，不是所在 `Agent` 的 provider；多租户调用方需要自己用 `ReadImageTool::new(provider)` 替换（已写进 `ToolRegistry::builtin` 的文档）。
- **摘要请求**：改为走 `request_builder`，用 `cap_only()` 加上自己的 512 上限，所以 `LLM_TOKEN_LIMIT_FIELD=max_completion_tokens` 时不会再被提供商拒绝。
- **公开 API 变更（版本升到 0.3.0）**：`ChatBackend` 方法参数改为 `ChatRequest`；`Provider::chat` / `chat_stream` 同样；`request_builder` 返回 `RequestDraft` 并接受 `impl Into<WireMessages>`；`ContentItem` 新增 `Reasoning` 变体，对它做穷举匹配的外部代码需要补一个分支；`AgentResult` 和 `Completion` 标了 `#[non_exhaustive]`。`AgentStreamEvent` 刻意没有加：`to_chat_events` 依赖编译器对变体做穷举检查。
- **未做 / 遗留**：
  - 没有原生的 Anthropic / Gemini 适配器，接口已经留好。
  - 没有模型能力表，`tool_choice` 和 `json_schema` 的支持仍按模型名关键字判断。
  - 推理 token 数没有计入 `TokenUsage`，留给第 5 项。
  - `StructuredAgentResult` 没有 `reasoning` 字段。

#### 第 3 项补充：在 CLI 和 Web UI 显示推理过程

- **线上协议**（`crates/shared`）：新增 `ChatEvent::Reasoning { text }` 和 `HistoryContentItem::Reasoning { text }`。`to_chat_events` 把 `AgentStreamEvent::Reasoning` 映射为前者。`--no-stream` 模式下，最终回答那一轮的推理会在答案之前单独广播。
- **终端**：
  - `print_chat_events` 的渲染逻辑抽成纯函数 `TerminalView::render`，它记录当前是否处在一段推理里。
  - 每段推理只在开头打一次 `[thinking]` 标签，之后的每个 chunk 用 ANSI 暗色（SGR 2/22）单独包裹。
  - 推理结束后，下一帧另起一行开始；如果这一帧本身就以换行开头，就不再额外加空行。
  - 设置了 `NO_COLOR` 时不输出颜色。
- **Web UI**：
  - 新增时间线项 `TimelineItem::Reasoning`。推理流入时展开、左边框为琥珀色、显示「思考中…」；同一轮的推理 chunk 都追加到同一张卡片里。
  - 收到任何非推理帧（第一个 token、工具调用、本轮结束等）时，卡片结束并自动折叠，可以点击展开。
  - 推理流入期间隐藏原来的三点「处理中」动画，页面滚动也会跟随推理文本。
  - 从历史加载的推理默认折叠。界面文案支持中、英、西三种语言。
- **验证**：
  - 单测：终端渲染 5 个（标签只出现一次、自动换行、每轮单独标签、颜色开关、其他帧输出不变）、`to_chat_events` 映射、历史映射、`shared` 的序列化往返。
  - 端到端：用本地 SSE 假服务模拟 DeepSeek 风格的推理模型，两轮推理中间夹一次 `calculator` 调用。
    - Web 端用浏览器检查了推理中、折叠后、展开后三种状态；页面没有报"unrecognized event"错误。
    - 终端在伪终端里实际运行了一次，确认了输出排版和暗色转义序列是成对的。
- **限制**：
  - 默认（不开回传）推理不进入 transcript，刷新页面后看不到之前轮次的推理。
  - 中途打开的标签页只能看到之后流入的推理。
  - 推理按纯文本显示，不渲染 markdown。

#### 第 4 项：子 agent 与规划

- **子 agent（`AgentTool`，`src/agent/sub_agent.rs`）**：
  - `AgentTool::new(name, description, agent)` 把一个 `Agent` 包装成 `Tool`，参数只有一个 `task`。父 agent 的模型调用它，等于把一个自成一体的任务交出去；子 agent 跑完自己的工具循环后，只把最终回答作为工具结果交回。
  - 每次调用都是一次全新的 `Agent::run`，子 agent 看不到父 agent 的对话，两次调用之间也互不记忆。`task` 参数的 schema 说明里提醒模型把需要的信息都写进去。
  - 名称在构造时校验（1–64 个 ASCII 字母、数字、`_`、`-`，即提供商接受的工具名），描述不能为空。
  - **并发**：只有子 agent 的工具全部是 `concurrency_safe` 时，它才是；有一个带副作用或行为未知的工具（例如 MCP 工具），整个委派就按顺序执行。
  - **超时与取消**：默认沿用父 agent 的工具超时，可以用 `with_timeout` 放宽。父 agent 取消或超时时，子 agent 的运行被直接丢弃，正在进行的模型请求和工具一起中止。
  - **审批与沙箱**：子 agent 自己的钩子只作用于它自己的工具调用，父 agent 的沙箱和审批不会继承。`AgentTool::new` 发现子 agent 有非只读工具又没有任何 before-tool 钩子时会打 warn（新增 `Agent::has_before_tool_callbacks()`）。没人回应的审批按 `Agent::run` 的方式记为未回应后继续，不会让父 agent 挂起。
  - **限制是子 agent 自己的**：父 agent 的轮次预算把整次委派算作一轮，子 agent 按它自己的 `with_max_steps` 运行，并发委派会成倍放大，需要调用方自己约束。
  - **轮次预算用尽**：回答末尾追加一句说明，提醒父 agent 这个回答可能不完整。
  - **错误**：子 agent 失败时，错误信息带上完整原因链（父 agent 的模型只看得到最外层那条消息），但截断到 1000 字符，避免把整段响应体带进父 agent 的上下文。
  - **token 用量**：工具拿不到父 agent 的 `ExecutionContext`，所以子 agent 的用量不计入父 agent 的 `usage`，而是在工具实例上累计（`AgentTool::usage()`），每次调用另打一条 info 日志。只统计成功完成的调用：失败或被丢弃的运行连同 transcript 一起丢了，用量也就丢了。
  - **不会成环**：`Agent` 的工具集在构造时固定，`AgentTool` 只能包装已经存在的 agent，所以构造不出循环；嵌套深度由调用方决定。
  - 新增 `Agent::toolbox()` 和 `ToolRegistry::iter()` 两个访问方法。
- **计划工具（`update_plan`，`src/tools/plan/`）**：
  - 参数是完整的计划（`plan: [{step, status}]`，状态为 `pending` / `in_progress` / `completed`）和可选的 `explanation`。每次调用都替换上一份计划，空列表表示清空。
  - **无状态**：工具本身不保存任何东西，当前计划就是对话记录里最后一次调用的参数。所以一个实例可以被多个 agent、多个会话共用，计划也能跟着对话记录一起落盘、挂起恢复。
  - **校验**：步骤描述不能为空，最多 50 步，单步不超过 200 字符，`explanation` 不超过 500 字符，`in_progress` 最多一个。不符合时返回错误，让模型自己改正，不做静默截断。清空计划时带的 `explanation` 会保留在返回里。
  - 返回渲染好的清单，例如 `Plan updated (1/3 completed):` 加上 `[x]` / `[>]` / `[ ]` 各行。
  - 已加入 `ToolRegistry::builtin()` 和 `select`，CLI 默认可用；GAIA 评测用的是固定工具列表，不受影响。工作区沙箱的分类测试里把它列为不涉及路径的工具。
  - 上下文压缩不处理它：结果只是参数的回显，而通常的提示语（"再调用一次即可取回"）对它来说等于让模型修改计划。
  - **裁剪的边界**：请求副本按 token 预算裁剪时，很久没更新的计划可能不在模型视野内。工具描述要求每次状态变化都重发计划，使最新副本靠近末尾；一个步骤本身就占满预算的情形仍可能丢失，对话记录和前端里仍有。
  - **终端显示**：不再打印 `update_plan` 那一行原始 JSON 调用，改为把结果打印成缩进的 `[plan]` 清单；校验失败时和其他工具一样显示 `-> Error`。Web UI 不变，结果卡片里就是这份清单。
- **测试辅助**：把 `model_tests.rs` 里的脚本化假后端移到 `src/llm/testing.rs`（仅测试编译），子 agent 测试和运行时测试共用。
- **示例**：`examples/sub_agent.rs`，主 agent 用 `update_plan` 规划，并把读代码的工作交给只读的 explorer 子 agent。
- **验证**：
  - 子 agent 14 个单测：任务委派、父 agent 端到端委派（确认父 agent 的对话没有传给子 agent）、预算用尽提示、用量累计、空任务、失败原因、原因链截断、并发判定、名称校验、超时、工具定义，以及两个取消 / 超时会丢弃子 agent 模型请求的测试和一个无钩子副作用工具的识别测试。
  - `update_plan` 13 个单测（含长度上限、清空保留原因），另有一个经脚本化模型走完整 agent 运行的端到端测试；终端渲染 2 个。
  - 没有用真实模型跑过示例，也没有在 CLI 里实际跑过 `update_plan`。
- **未做 / 遗留**：
  - 子 agent 的中间过程（它的 token、工具调用）不会出现在父 agent 的流式事件里，前端只能看到一次工具调用和它的结果。
  - 没有 handoff（把对话控制权转交给另一个 agent），也没有 workflow / graph 编排。
  - 没有反思 / 自检步骤。
  - 子 agent 的用量没有合并进父 agent 的 `usage`，失败运行的用量也没有统计，留给第 5 项。（第 5 项已解决。）
  - 加入 `update_plan` 改变了 `builtin()` 的工具列表，但不会让升级前挂起的运行失效：`RunFingerprint::mismatch` 只检查记录里的工具是否还在，新增的工具不算差异（见 `an_added_tool_still_matches`）。

#### 第 5 项：可观测性

- **span**（`src/llm/spans.rs`、`src/agent/runtime/observe.rs`）：
  - 按 OpenTelemetry GenAI 语义约定命名，一共三种：
    - `invoke_agent`：一次 `Agent` 运行；
    - `chat`：一次模型请求；
    - `execute_tool`：一次工具调用，包括审批钩子。
  - `chat` span 加在 `Provider::chat` / `chat_stream` 上，所以本 crate 发出的每个请求都有，重试的每次尝试各算一个。
    - 记录模型、响应 id、结束原因（使用 API 原名，如 `tool_calls`）、输入 / 输出 / 缓存 / 推理 token。
    - 出错时设置 `otel.status_code=error`，`error.type` 取 HTTP 状态码或错误类别。
    - 流式请求的 span 持续到流结束，用量取自最后一个 chunk。
  - `invoke_agent` 覆盖普通、流式、可恢复、恢复后的运行和两条结构化路线。
    - 记录结果（done / suspended / cancelled / error）、轮次和这次运行的 token 用量。
    - 新增 `Agent::with_name`，用作 `gen_ai.agent.name`。
  - 嵌套关系：子 agent 的 `invoke_agent` 挂在父 agent 的 `execute_tool` 下面。
  - token 计数以 `i64` 记录。`tracing-opentelemetry` 会把 `u64` 导出成字符串，导出端无法求和。
- **OTel 导出**（`--features otel`，可选依赖）：
  - 依赖版本：`opentelemetry` / `opentelemetry_sdk` / `opentelemetry-otlp` 0.33，`tracing-opentelemetry` 0.34。
  - 走 OTLP/HTTP protobuf，用 blocking reqwest 客户端，配合 SDK 在独立线程里运行的批处理器；不引入 gRPC 依赖。
  - 只有设置了 `OTEL_EXPORTER_OTLP_ENDPOINT`（或 `..._TRACES_ENDPOINT`）时才启用；`OTEL_SDK_DISABLED=true` 可以关闭。
  - 服务名默认是 `agent`，可以用 `OTEL_SERVICE_NAME` / `OTEL_RESOURCE_ATTRIBUTES` 覆盖。
  - 导出端固定接收 INFO 及以上级别，不受 `RUST_LOG` 影响。
  - `telemetry::init()` 返回一个守卫，析构时 flush 最后一批数据；`std::process::exit` 的路径（CLI 信号处理）显式调用 `telemetry::shutdown()`，最多等 2 秒。
  - 控制台日志的 span 前缀只保留识别性字段（agent / tool / model 名），导出专用字段（状态码、token 数、id）不上屏。
- **用量统计**：
  - `TokenUsage` 新增 `cached_prompt_tokens` 和 `reasoning_tokens` 两个字段，读取旧数据时默认为 0。
  - `ExecutionContext` 新增 `usage_by_model`，按模型拆分用量；序列化时为空则省略，旧数据可以正常读取。
  - **记账机制**（`src/agent/usage.rs`）：每次运行设置一个 task-local 账本。
    - `charge(model, usage)` 把用量记到当前运行上；子运行的账本挂在父账本上，每笔都同时向上传递。
    - 结果：子 agent 的消耗按它自己的模型计入父运行，失败或被中止的委派也照算；不会重复计数。
    - 已经接入的地方：agent 自己的请求、`AgentTool`（通过嵌套账本）、`read_image`、摘要请求、无状态的 `tool_loop`。
    - 在 `spawn` 出去的其他 task 上调用 `charge` 不会生效，返回 `false`。
  - `AgentTool::usage()` 保留，单独给出子 agent 自己的那一份。
- **成本**（`src/llm/pricing.rs`）：
  - `LLM_PRICING_FILE` 指向一个 JSON 价格表（每百万 token 的 `input` / `cached_input` / `output` 价格）。代码里不内置价格，因为价格会变，而且因提供商和合同而异。
  - 价格表格式严格校验：字段名拼错、价格为负都会报错；文件读不了时只打 warn，不影响运行。
  - `PriceTable::estimate` 按模型计算；有模型缺价格时结果标记为部分估算（下限）；一个模型都没有价格时返回 `None`，不会显示成 0。
- **显示**：
  - `ChatEvent::Done` 新增可选字段 `usage: Option<TurnUsage>`，包含 token 数和可选的成本；新旧版本前后端互相兼容。
  - 终端每轮结束时打印一行暗色的 `[usage] 2,700 in (1,000 cached) · 60 out · ≈ 0.0044 USD`。
  - Web UI 在回答下方显示同样的一行小字，跟随语言切换（中、英、西）。
- **验证**：
  - 单测覆盖：账本（嵌套、被丢弃的子运行、同一 task 上的并发运行）、用量字段和旧数据兼容、价格计算和文件校验、错误分类、结束原因名称、终端和 i18n 的格式化。
  - 运行时级别的测试：
    - 父 agent 委派给子 agent 时，父运行按模型拆分的用量（普通、流式、委派失败三种情况）；
    - 一次委派运行产生的完整 span 树及各字段；
    - 失败请求的 span 状态。
  - 端到端：用本地假 LLM 服务和假 OTLP 接收端，以 `--features otel` 跑真实 CLI，解码导出的 protobuf。
    - 确认 span 树、服务名、整数类型的 token 属性、`tool_calls` 结束原因都正确，终端 usage 行和成本也正确。
    - Web UI 在浏览器里检查了 usage 行和中英切换，控制台没有报错。
- **未做 / 遗留**：
  - 没有 OTel metrics（计数器、直方图），只有 trace。
  - span 默认不记录提示词和回答内容，GenAI 约定里的这部分是可选的，而且涉及隐私。
  - Web 历史记录不保存每轮用量，页面刷新后之前轮次的 usage 行不再显示。
  - 结构化入口和一次性的 `llm::*` 辅助函数在运行之外调用时，用量不会记到任何地方，只出现在 `chat` span 上。
  - 恢复（`resume`）后的第一次重试与之后的循环共用一个 `invoke_agent` span；`run_continuing` 里每次重新进入 `drive` 各算一个 span，所以一次挂起-恢复的运行在 trace 里可能有多段。

#### 第 6 项：持久化检索和长期记忆

- **方案选择**（与你确认过）：SQLite（`rusqlite` 0.40，`bundled`，不依赖系统里装的 SQLite）；CLI 记忆默认关闭，`--memory` 开启；写入记忆不需要审批，但每次写入都是可见的工具调用，并且可以用 `/memory` 查看。
- **Embedding 接口**（`src/knowledge_base/embed.rs`）：
  - 新增 `Embedder` trait（`model()` + `embed(texts)`），内置实现 `OpenAiEmbedder`，复用聊天模型的端点与凭据。
  - 现在每个请求都有重试、`embeddings` span，用量按 embedding 模型计入所在的运行（第 5 项的账本）。
  - 原有的 `vector_search` 保留，`SearchCompressorCallback` 不受影响；`vector_search` 把查询和文本块合并成一次请求。`embed_texts` / `embed_text` 两个自由函数已由 `OpenAiEmbedder` 取代，保留为 `#[deprecated]` 的薄包装以免破坏下游（见 `CHANGELOG.md`）。
- **向量库**（`src/knowledge_base/store.rs`，`VectorStore`）：
  - 一张 `documents` 表，按集合（collection）和来源（source）组织；`put_source` 在一个事务里替换某个来源的全部文本块，检索时看不到新旧混杂的中间状态。
  - **复用向量**：写入前按「文本哈希 + 模型」查找库里已有的向量（哈希只用来找候选，最终比对原文），只为没见过的文本调用 embedder，同一批里重复的文本只算一次。`IndexReport` 报告计算了多少、复用了多少。
  - **模型隔离**：每个向量记录生成它的模型，检索只比较当前模型的向量；换了模型后旧文档不会被删除，但在重建索引前检索不到，并打一条 warn。维度对不上时报错，提示可能是模型换了但名字没变。
  - **检索**：精确余弦。每个集合的向量预先归一化后缓存在内存里；缓存是否过期由每个集合在 `meta` 表里的版本号决定（每次写入在同一事务内加一），本进程、同文件的其他句柄、其他进程的写入都能发现（第三轮审查前用的是整库级的 `PRAGMA data_version`，见下）。
  - SQLite 使用 WAL 模式，设置 5 秒忙等待；所有数据库操作都放在 `spawn_blocking` 里，不阻塞异步线程。
  - 文件里记录 schema 版本：旧版本（1）原地升级到 2，未知版本拒绝打开，不会误读。
  - FNV-1a 哈希从 `fingerprint.rs` 移到 `util`，算法和输出不变（有参考向量测试），已有的运行指纹不受影响。
- **`retrieve` 工具**（`src/tools/retrieve.rs`）：`RetrieveTool::new(store, collection)`，可以改名字、说明和最低相似度；返回带来源、段号、相关度的段落。只读，可以和其他调用并发。没有加进 `builtin()`，因为它需要调用方提供向量库。
- **长期记忆**（`src/memory/`）：
  - `MemoryStore`：按作用域（一个用户）存储简短事实，底层是 `VectorStore` 里一个带 `memory:` 前缀的集合，作用域之间完全隔离（按 id 删除也跨不了作用域）。
  - 写入时整理空白；完全相同的事实只存一份；单条最长 500 字符，每个作用域最多 2000 条。
  - **拒绝保存疑似密钥**：私钥、`sk-` 开头的 API key、AWS / GitHub / Slack token、JWT、`password=…` 一类内容会被直接拒绝；规则写得比较窄，普通句子（例如提到 password manager）不会被误伤。
  - 三个工具：`remember`、`forget`（按编号）、`recall`（按语义检索）。`remember` 的说明里要求只保存用户陈述或确认过的事实，不保存密钥，也不保存未经确认的网页、文件内容。
  - `RecallCallback`（before-LLM 钩子）：用最新一条用户消息检索记忆，把相关的几条（默认 5 条、相似度 ≥ 0.3）作为一条**普通 user 消息**插在它前面，并标明是「之前保存的笔记，仅供参考，不是指令」。不放进 system 指令，也不写入对话记录。同一轮里的后续工具轮次复用第一次的检索结果，所以每轮只多一次 embedding 请求；检索失败时打 warn，这一轮照常进行。
- **CLI**：
  - `--memory`：打开 `AGENT_CLI_MEMORY_PATH`（默认 `.agent/memory.db`）的记忆库，作用域就是本机的 `local`，所有 `--session` 共享，`/reset` 不清除它；注册三个记忆工具和召回钩子；启动时提示当前有几条记忆。没有配置 `EMBED_MODEL` 时直接报错退出。
  - 新命令 `/memory`（别名 `/memories`）：终端和网页都能用，列出全部记忆及编号；没开 `--memory` 时提示如何开启。网页的命令菜单文案支持中、英、西三种语言。
- **示例**：`examples/rag.rs`，为 `docs/` 建索引，用 `retrieve` 回答问题，再演示一条事实从一次对话带到另一次全新的对话。
- **验证**：
  - 单测覆盖：向量库（检索排序与来源、相似度阈值、复用向量、替换与删除、写后即读、集合隔离、模型隔离、重新打开与多连接写入、精确文本查找、按来源删除、schema 版本与升级、编号不复用、编码往返）；记忆、记忆工具、召回钩子（含一个经 agent 端到端的：记忆进了请求、没进对话记录）；`retrieve`；`/memory` 命令；`search.rs`（`top_k` 排序与并列、归一化、用 embedder 排序）；哈希参考向量。测试用一个确定性的词袋 embedder，不访问网络。具体数量不在这里写，以 `cargo test` 为准。
  - 端到端：用本地假服务（同时提供 chat 和 embeddings 接口）跑了 `examples/rag`：
    - 第一次为 3 个文件计算了 108 个文本块的向量，第二次 0 个重新计算、108 个复用；
    - `retrieve` 返回了带来源和相关度的段落；
    - 第一次对话用 `remember` 存下事实，第二次全新对话里它被自动注入，注入的消息角色是 `user`。
  - 在伪终端里跑了真实 CLI：会话 `alpha` 里保存，`/memory` 列出；新会话 `beta` 里自动召回；不带 `--memory` 时 `/memory` 给出开启提示；没配 `EMBED_MODEL` 时报错退出。
  - 前端已重新构建。
- **未做 / 遗留**：
  - 检索是精确计算，不是近似最近邻（ANN）；几十万条以上的语料应该换专门的向量数据库。没有关键词 / BM25 混合检索，也没有重排。
  - 分块仍是固定字符长度，没有按段落、标题或代码结构切分。
  - 记忆没有自动去重合并、过期或摘要：意思相同但措辞不同的事实会存成两条。
  - 记忆以明文保存在本地 SQLite 文件里，没有加密（只靠 `0600` 权限）。
  - 自动召回只用最新一条用户消息做查询，不看更早的上下文。
  - 防提示注入靠工具说明、可见的工具调用、「参考资料 / 可能不可信」标注，以及 `forget` 的默认审批；模型仍可能把网页里的内容存成记忆。需要更严格时，可以把 `remember` 加进 `--dangerous-tools` 走审批；但这个参数会替换默认的审批列表，所以要连同默认的 `delete_file,write_file,edit_file,run_command,forget` 一起写上。
  - 开着 `--memory` 时挂起的轮次，关掉 `--memory` 后恢复会被拒绝：记录里的 `remember` 等工具不在了（新增工具不影响恢复，减少工具会）。
- **审查后修订**（上面几段保留为当时的实现记录）：
  - 记忆库默认**按工作区**隔离（相对 `--workspace` 解析），不是"整机一个"；要跨工作区共享就把 `AGENT_CLI_MEMORY_PATH` 指向绝对路径。库文件新建时权限为 `0600`。
  - `forget` 进入 CLI 默认审批列表，且不给"总是允许"；`remember` 仍免审批。召回消息里补了"可能来自不可信来源"的提示。
  - 换 `EMBED_MODEL` 不再是死局：`VectorStore::reindex` 原地重新嵌入（保留编号）。**不会在启动时自动进行**（第二轮审查改为必须显式 `--memory --reindex-memory`，因为它会覆盖旧向量）；`remember` 命中旧模型记录时只重新嵌入那一条。
  - 召回缓存改为按存储修订号失效，`remember` / `forget` 之后同样的提问会重新检索；检索失败与超时（3 秒）在这一轮内缓存为空，不会每个工具轮次都重试一遍。
  - 凭证规则补了中文字段名、`Bearer`、GitLab token，并把"值要像密钥"（够长或含数字）作为第二道判断，`token: bucket` 之类不再误伤。
  - `rusqlite` 进了默认开启的 `sqlite` feature，库使用者可以关掉整棵 SQLite 依赖树。
- **第三轮审查修订**（上面各段仍是当时的实现记录，与本段冲突处以本段为准）：
  - 文档与实现对齐：不再有"启动时自动重新嵌入"；缓存失效改为按集合的版本号。
  - 记忆编号永不复用（`AUTOINCREMENT`，旧库首次打开时原地升级并保留编号）；`forget` 必须带上那条记忆的原文，对不上就不删，审批提示因此能看到将要删除的内容。
  - `remember` 在用过可能带第三方文字的工具（网页、文件、命令、MCP）的那一轮会被拒绝（`RememberGuard`），直到用户确认；`forget` 的"总是允许"选项在提示里去掉（它本来就不会被记住）。
  - 快照重建分页读取、同一集合同一时刻只重建一次；批量嵌入同时受条数和字节数限制；凭证规则补了 `DB_PASSWORD=` 一类带前缀的字段、Stripe/Google/npm 密钥与带口令的 URL。

#### 第 7a 项：Cargo features 裁剪

- **拆分**：第 7 项按讨论拆成 7a–7d 四个子项，逐个审批；补全 MCP、Agent 作为 MCP Server、`Retry-After`、服务化 API 四块列为后续的第 8–11 项（见上表）。
- **新增三个 feature**，和原有的 `sqlite`、`otel` 一起：
  - `cli`：`axum`、`tower-http`、`reedline`、`crossterm`、与前端共用的 `shared` 类型，以及 tokio 的 `net` / `signal`。库本身从不使用它们，现在只有 `cli` 二进制会编译这些依赖。`cli` 会同时开启 `mcp` 和 `sqlite`。
  - `mcp`：`tools::mcp`、`ToolRegistry::with_mcp` / `with_mcp_over`、`callback::mcp_guard`。`rmcp` 现在只开客户端相关的 feature；示例里用到的服务端部分（`server`、`macros`、`transport-io`、Streamable HTTP 服务端）移到了 `[dev-dependencies]`，按 resolver 2 的规则不会进入普通构建。
  - `zip`：`unzip_file` 工具。关掉后 `builtin` / `builtin_all` 里没有它，`select` 遇到这个名字报「unknown tool」。
- **默认全开**（`cli`、`mcp`、`sqlite`、`zip`），所以 `cargo build`、`cargo run --bin cli` 的行为和以前一样。
- **依赖数**（`cargo tree` 去重后的运行时依赖）：
  - 改动前默认 265 个；改动后默认 262 个（`rmcp` 少了服务端部分）；
  - `default-features = false` 时 180 个（改动前同样设置是 259 个）；
  - 单独开 `mcp` 189 个、`sqlite` 186 个、`zip` 220 个、`cli` 223 个。
- **目标声明 `required-features`**，不满足时 Cargo 直接跳过，而不是编译失败：
  - `cli` 二进制需要 `cli`；`gaia` 二进制需要 `zip`（评测的工具集是固定的，包括 `unzip_file`；如果它随 feature 悄悄变少，两次跑分就没法比较）；
  - `rag` 示例需要 `sqlite`，`file_tools` 需要 `zip`，`mcp_agent` / `mcp_config` / `mcp_http_agent` / `mcp_inspect` 需要 `mcp`。`mcp_server` / `mcp_http_server` 只用 dev-dependency 里的 `rmcp`，不需要 feature。
- **代码改动**：
  - `tools::file_upzip`、`tools::mcp`、`callback::mcp_guard` 三个模块按 feature 编译；`registry.rs` 中相关的导入、注册和 `select` 分支加了 `cfg`。
  - `WorkspaceGuardCallback::guarded_fields` / `is_write` 从 if-else 链改成 `match`，`unzip_file` 那一支加 `cfg`，其他判断逻辑不变。
  - 测试：`unzip_file` 相关的用例加 `cfg`；「路径不存在时按字面解析」的用例改用 `write_file`，这样两种配置下都覆盖到；新增一个只在关闭 `zip` 时运行的测试，确认三个入口都拿不到 `unzip_file`。compaction 的排除列表测试改为直接写工具名，两种配置下都检查。
- **CI 与本地检查**：
  - `ci.yml` 新增 `features` 任务（矩阵：不开任何 feature，以及 `cli` / `mcp` / `otel` / `sqlite` / `zip` 各自单独开启，跑 `clippy -D warnings`）和 `test-minimal` 任务（不开 feature 跑测试）。
  - `Makefile.toml` 新增同样内容的 `check-features`，并加入 `ci` 和 `dev`，保证本地和 CI 跑的是同一组检查。
- **验证**：上面 6 种组合加上默认和 `--all-features`，clippy 都没有警告；默认与 `--all-features` 下库内 741 个测试、CLI 53 个通过，关闭全部 feature 时库内 623 个测试通过；wasm 端 clippy 无警告；CLI 冒烟（`--list`）正常。
- **遗留**：
  - 不开 feature 生成文档时有 8 处内部链接指向被关掉的模块（3 处是这次的 `mcp`，5 处是之前的 `sqlite`），文档任务用的是 `--all-features`，不受影响。
  - `zip` 依赖本身仍开着 bzip2、zstd、lzma、AES 等全部压缩格式，可以只留 `deflate` 再省几十个依赖，但那会让部分压缩包解不开，所以没动。
  - 原来自己设置了 `default-features = false`（为了去掉 `sqlite`）的下游，现在也会同时失去 `mcp` 和 `zip`，需要显式加回来；CHANGELOG 里已经写明。

#### 第 7b 项：工具定义体验

- **`ToolDefinition`**（`src/tools/tool.rs`）：工具对模型的描述，只有名称、描述、JSON Schema 三个字段，与任何提供商的请求格式无关。
  - `Tool::definition()` 改为返回 `ToolDefinition`，并且不会再失败（原来返回 `anyhow::Result<ChatCompletionTools>`，失败只可能来自 `async-openai` 的 builder）。
  - `ToolRegistry::definitions()` 返回 `&[ToolDefinition]`。
  - 只在 `llm::client::request_builder` 组装请求时转成 chat completions 的 `{"type": "function", "function": {...}}`；新增测试锁定转换后的 JSON，与改动前发出的完全一致。
  - `final_answer` 合成工具（结构化输出用）同样改为 `ToolDefinition`；`final_answer_tool_from_schema` 不再返回 `Result`。
  - 改完后 `src/tools` 里只剩 `read_image` 用到 `async-openai`，那是它自己发视觉请求，不属于工具抽象。
- **工具名校验**：新增 `validate_tool_name` 和 `MAX_TOOL_NAME_CHARS`（64），`ToolRegistry::add` 对每个工具都做校验。名字不合法时在注册时报错并指出是哪个工具，而不是之后每个请求都被 API 拒绝、看不出原因。内置工具、MCP 工具（客户端已做清洗）和子 agent 工具（构造时已校验）都满足要求。`examples/mcp_inspect` 里重复的名称长度检查随之去掉。
- **`FnTool`**（`src/tools/fn_tool.rs`）：一个参数结构体加一个 async 闭包就是一个工具。
  - 参数 schema 由 `schema_of::<A>()` 生成，字段上的文档注释就是模型看到的参数说明。
  - 闭包返回 `crate::Result<R>`（当时是 `anyhow::Result<R>`，7c 统一改成 `agent::Result`），`R: Into<String>`；`Err` 和其他工具一样作为错误结果交给模型。没有把错误类型做成泛型：那样在闭包里用 `?` 时 Rust 推断不出错误类型，每个闭包都要手写类型标注。
  - 参数解析失败时，serde 给出的原因（缺字段、类型不对）直接写进错误信息本身——模型只看得到最外层那条消息，只写「参数无效」它无从改正。空参数按 `{}` 处理，有些模型对无参工具会发空串。
  - `FnTool::new` 返回 `Result`（与 `AgentTool::new` 一致）：工具名不合法，或参数类型的 schema 不是 object（例如用了单元结构体 `struct NoArgs;`，它只能从 `null` 反序列化，模型发来的 `{}` 永远解析失败）时，在构造时就报错。
  - `.with_timeout(..)` / `.read_only()` 对应 `Tool::timeout` / `Tool::concurrency_safe`，默认与其他工具一样不并发、使用调用方的超时。
  - 模块文档里的示例是会编译的 doctest。
- **文档**：README（中英文）新增「自定义工具」小节；CHANGELOG 记录了新增项和不兼容变更。
- **验证**：
  - 新增 `FnTool` 9 个测试（schema 来源、正常调用、参数错误的提示、空参数、处理函数报错、超时与只读、构造时校验、闭包状态、经 agent 端到端调用并核对发出的工具定义和工具结果）、工具定义与名称校验 2 个、请求里工具 JSON 的格式 1 个。
  - 全部 feature 时库内 753 个、CLI 53 个通过；不开 feature 时库内 635 个通过；doctest 通过；6 种 feature 组合和 wasm 端 clippy 都没有警告；文档没有失效链接。
- **未做 / 遗留**：
  - 没有 `#[tool]` 过程宏，内部的 `simple_tool!` 宏仍不公开（已有 `FnTool` 覆盖同样的用途）。
  - 内置工具没有改写成 `FnTool`：它们有各自的名称常量、手写 schema 或额外的状态，改写收益不大。
  - `Tool::execute` 仍然接收原始 JSON 字符串、返回 `anyhow::Result<String>`（错误类型留到 7c 一起处理，现为 `crate::Result<String>`）。
  - 对 `Tool` 的实现方、`ToolRegistry::definitions()` 和 `request_builder` 的调用方来说是不兼容变更（见 CHANGELOG），发给模型的请求内容不变。

#### 第 7c 项：类型化错误

- **方案**（与你确认过）：单一的 `agent::Error` 枚举；调用方实现的 trait 方法（`Tool::execute`、`ChatBackend::create` / `create_stream`、`Embedder::embed`）也全部改为返回 `agent::Result`；库内部不再使用 `anyhow`；一次做完再审批。
- **`agent::Error`**（`src/error.rs`，thiserror 2.0，原本就在依赖树里）：`#[non_exhaustive]`，按调用方能怎么处理分 11 类：
  - `Cancelled`（带截至此刻的记录）、`BudgetExhausted`（包着真正的失败）、`TruncatedOutput`（带输出上限）：调用方要匹配并读取其中数据的三类；
  - `Provider(ProviderError)`：带 HTTP 状态码和「重试是否可能成功」；
  - `InvalidInput`、`Config`、`Tool`、`Mcp`、`Storage`、`Io`、`Other`：信息加可选的原因。
  - `{err}` 只打印最外层信息（工具出错时模型看到的就是这一句，和以前一致），`{err:#}` 打印完整原因链；`source()` 走的是同一条链，所以转成 `anyhow` 后打印也不会重复。
  - `Error::context` 在前面加一层说明但保留原有类别，例如 `Provider` 错误加了几层说明后，仍然能判断是否可重试。
  - `Error::downcast_ref` 沿原因链查找具体类型，例如 `Io` 背后的 `std::io::Error`、向量库背后的 `rusqlite::Error`。
  - 构造方式：`Error::tool(..)` 等构造函数；`ResultExt::wrap_err(Error::tool, "正在做什么")` 包装其他库的错误；`err_as` 只换类别、不加说明；`?` 可以直接转换 `std::io::Error`、`OpenAIError`（按状态码分类）、`rusqlite::Error`。
  - 库内部用 `bail!` / `ensure!` 两个宏（`ensure!(cond, invalid_input, "...")`），不对外公开。
- **重试判断**（`llm::retry::is_transient`）改为按类别判断，不再 downcast：
  - `Provider` 看它自己标的可否重试（`async-openai` 的错误仍按原规则分类：429 / 408 / 409 / 5xx / 网络错误 / 响应解析失败可重试，其他 4xx 不可重试）；
  - `InvalidInput`、`Config`、`Cancelled`、`TruncatedOutput`、`BudgetExhausted` 不重试；其余仍然重试，与以前「认不出的错误照样重试」一致。
  - 删除了 `llm::retry::Permanent`，自定义后端改用 `ProviderError::transient` / `permanent` 表达。
  - GAIA 求解器原来手动排除截断和预算用尽两种情况，现在 `is_transient` 本身就会排除，直接传它即可。
- **类别的选取**：迁移脚本按文件给出默认类别，之后逐处复查调整。例如工具名、schema 名、空 prompt、chunk 参数、记忆内容不合规归为 `InvalidInput`；缺少 `TAVILY_API_KEY` / `HF_TOKEN` / `EMBED_MODEL`、`mcp.json` 写错、工作区目录不存在归为 `Config`；模型返回空内容、没有 choices、结构化输出解析失败、embedding 数量或维度不对归为 `ProviderError::transient`。`wrap_err` 只包装其他库的错误，不会把本 crate 的 `Provider` 错误改成别的类别。
- **相关类型**：`Cancelled`、`BudgetExhausted`、`TruncatedOutput` 移到 `agent::error`，原路径 re-export。`BudgetExhausted` 从「附加在错误上的标记」改为「包着真正失败的错误」。trace 的 `error.type` 对非模型服务的错误改为类别名（`tool`、`storage` 等），原来是 `_OTHER`。
- **调用方的变化**：
  - 二进制、示例、集成测试仍用 `anyhow`，`?` 照常转换。只有 CLI 里由 agent 产出的流的元素类型改成 `agent::Result`。
  - 判断取消从 `downcast_ref::<Cancelled>()` 改为匹配 `Err(agent::Error::Cancelled(c))`，README 中的示例已同步修改。
- **规模**：71 个文件，新增约 1500 行、删除约 900 行；库代码（除 `error.rs` 里讲与 anyhow 互通的文档和测试）中 `anyhow` 出现次数为 0。
- **验证**：
  - `error.rs` 9 个单测（单层与完整链的打印、I/O 错误透明转换与 downcast、经 anyhow 打印不重复、`Provider` 加说明后仍保留状态码和可否重试、OpenAI 错误按状态码分类、三种特殊类别不被 `context` / `with_source` 改变、宏、`kind`、`Send + Sync + 'static`），另有 4 个 doctest；`retry` 8 个单测改为按类别判断；`FnTool` 新增一个「闭包里用 `wrap_err` 包装其他库错误」的测试。
  - 全部 feature：库内 763 个、CLI 53 个、集成测试和 doctest 全部通过；不开 feature：库内 645 个通过；示例里的测试 5 个通过。
  - 6 种 feature 组合、全部 feature 和 wasm 端的 clippy（`-D warnings`）都没有警告。
  - 端到端：用返回 401 的本地假服务跑 `agent` 二进制，1 秒内失败、只发了 1 次请求，打印 `401 Unauthorized invalid_request_error: Incorrect API key provided (code: invalid_api_key)`；指向无人监听的端口时按退避重试两次后失败，`Caused by` 打印出完整原因链。
- **未做 / 遗留**：
  - **对下游是大范围的不兼容变更**：所有返回类型、三个 trait 的签名、`Permanent` 被删除、`is_transient` 的参数类型都变了；CHANGELOG 里逐项写了迁移方法。
  - CI 跑的 `cargo test` 不会编译示例里的 `#[cfg(test)]` 模块（这次 `examples/rag.rs` 里的一个测试替身就是靠 `cargo test --examples` 才发现的），可以考虑把它加进 CI。
  - 没有给 `Tool` 错误再细分（例如「参数无效」和「执行失败」），模型看到的都只是文字。

#### 第 7d 项：小修

- **tokenizer 按模型选择**（`src/callback/context_optimizer/tokens.rs`）：
  - 新增 `Encoding`（`Cl100kBase` / `O200kBase`，`#[non_exhaustive]`）。`Encoding::for_model` 直接用 `tiktoken-rs` 自带的模型表判断：GPT-4o、GPT-4.1、GPT-5、`o` 系列、`gpt-oss` 用 `o200k_base`，其余（GPT-4、GPT-3.5，以及 DeepSeek、Qwen、Claude 等其他提供商的模型）用 `cl100k_base`。以后 OpenAI 出新模型，升级 `tiktoken-rs` 即可，不用改这里。
  - 模型名不区分大小写，忽略提供商前缀（OpenRouter 的 `openai/gpt-4o` 当作 `gpt-4o`）和微调模型的 `ft:` 包装。`o200k_harmony` 与 `o200k_base` 词表相同、只差特殊 token（这里不计），所以按 `o200k_base` 算。
  - `ContextOptimizer` 新增 `for_model(model)` / `with_encoding(..)` / `encoding()`；`Ledger` 记住自己的编码，`refresh` / `rebuild` 都用它重新计数，各阶段的计数单位一致。
  - `Agent::new` 安装的默认优化器按 agent 的模型选编码。手动构造的 `ContextOptimizer::new(n)`、`tokens` 里的自由函数和 `eviction::evict_middle` 仍用 `cl100k_base`，和以前一样。
  - 「字节数 ≥ token 数」这个上界对两种编码都成立（每个 BPE token 至少一个字节），跳过分词的快速判断不受影响，测试对两种编码都做了验证。
- **README 不再写死测试数**：中英文 README 顶部写死的 `tests-418 passing` 徽章换成 GitHub Actions 的 CI 状态徽章（`lf-wxp/agent` 的 `ci.yml`）；正文里原来的 528 已经不在了，README 中不再有具体测试数量。
- **最小示例的守卫**：中英文 README 的最小示例由 `telemetry::init()?;` 改为 `let _telemetry = telemetry::init()?;`，并注明守卫要持有到程序结束。把示例代码抽成临时 example 编译过，没有警告（之前会有 `must_use` 警告），用完已删除。
- **文档**：README「上下文管理」一节补充了按模型计数的说明；CHANGELOG 记录了新增 API 和行为变化。
- **验证**：
  - 新增 6 个测试：OpenAI 新模型映射到 `o200k_base`（13 个模型名，含大小写、前缀、`ft:`）；其他模型回退 `cl100k_base`；两种编码对同一段中文计数不同且默认是 `cl100k_base`；`Ledger` 在 `refresh` / `rebuild` 后仍用自己的编码；同一段中文对话按 `o200k_base` 计正好在预算内、优化器不动它，按 `cl100k_base` 计则会被裁剪；未知模型保持默认编码。上界测试扩展到两种编码。
  - 全部 feature：库内 769 个、CLI 53 个、集成测试、doctest 通过；不开 feature：库内 651 个通过；示例里的测试 5 个通过；6 种 feature 组合、全部 feature 和 wasm 端的 clippy 都没有警告；文档没有失效链接。
- **未做 / 遗留**：
  - **行为变化**：对 GPT-4o 等模型，同一段对话现在计得更少（中文尤其明显），裁剪会比以前晚一些开始；`LLM_MAX_HISTORY_TOKENS` 的含义没变，仍然是请求大小的软上限。
  - ~~没有为 Agent 级别的默认安装单独写测试~~：审核后补了 `Agent::context_encoding` 与对应测试（`gpt-4o` / `gpt-4` / 未知模型三种）。安装的优化器本身仍是 trait object，取不出编码，所以这个访问器报告的是构造时记下的值。
  - 其他提供商的模型仍用 `cl100k_base` 估算，没有接入它们各自的 tokenizer。
  - `examples/vector_search.rs`、`examples/callback_search_compressor.rs` 演示分块时仍直接用 `cl100k_base`，没改。
  - 新徽章依赖仓库公开且 CI 在 `main` 上跑过；仓库若是私有的，徽章不会显示状态。
  - 7c 遗留的「CI 没有跑 `cargo test --examples`」仍未处理。

#### 审核修复（2026-10-23）

对本次全部未提交改动的审核见 [code-review-uncommitted-2026-10-23.md](code-review-uncommitted-2026-10-23.md)。除 7a–7d 的记录外，审核还发现并修复了下面这些，都属于同一批改动：

- `ResultExt::{wrap_err, wrap_err_with, err_as}` 会把本 crate 的 `Error` 重新归类（`Cancelled` 变 `Tool`、不可重试的 `Provider` 变得可重试），现在只对其他 crate 的错误生效。
- CI 的 feature 矩阵补了 `cargo check --lib --bins`：`--all-targets` 会把 dev-dependencies 的 feature 合并进来，缺 `cfg` 的库代码在它下面照样能编译。
- 工具名规则统一到 `MAX_TOOL_NAME_CHARS` / `validate_tool_name`，不再有两份常量和一份手写规则；`FnTool` 与 `AgentTool` 都要求描述非空。
- `with_mcp_over` 的「坏工具只记日志并跳过」抽成 `add_skipping` 并有测试；`ToolRegistry::add` 拒绝非法名字也有测试。
- embedder 契约违约（数量不符、空向量、长度不一致）由 `transient` 改为 `permanent`：同一输入重试只会失败得一样。
- `file_read` / `file_upzip` 的裸 `?` 补了路径上下文，模型能看到是哪个文件失败。

#### 第 15 项：补全 CI 覆盖

- **实测发现的漏洞**（改之前，全部 feature）：

  | 命令 | 跑到的测试 |
  |---|---|
  | CI 原来的 `cargo test --all-features` | 860 |
  | 加 `--workspace` | 893（多出 `shared` 27、`web-ui` 6） |
  | `cargo make test` 原来的命令 | 同 CI，860 |
  | 本次改后：`--workspace --all-features --all-targets` + `--doc` | 898（再多出 `examples/rag.rs` 的 5 个） |

  - `shared` 和 `web-ui` 是 workspace 成员，裸 `cargo test` 不碰它们。
  - 示例里的 `#[cfg(test)]` 模块，裸 `cargo test` 只编译、不运行（实测：不加 `--examples` 时看不到 `examples/rag.rs` 的测试）。
  - wasm 端 clippy 在 `cargo make ci` 里有，GitHub Actions 里没有。
  - 原来 `Makefile.toml` 里的注释说 `web-ui`「没有测试、也不能在宿主上编译」，与实际不符：它在宿主上能编译，6 个测试原生就能跑，不需要 wasm 运行器。注释已改。
- **改动**：
  - `ci.yml` 的 `test` 任务改为两条命令：`cargo test --workspace --all-features --all-targets --no-fail-fast` 和 `cargo test --workspace --all-features --doc --no-fail-fast`（`--all-targets` 不含 doctest，所以要分开）。
  - 新增 `clippy-wasm` 任务：安装 `wasm32-unknown-unknown`，跑 `cargo clippy -p web-ui -p shared --target wasm32-unknown-unknown --all-targets -- -D warnings`，与 `Makefile.toml` 里的 `clippy-wasm` 同一条命令。
  - `Makefile.toml` 的 `test` 任务改成同样的两条命令，本地 `cargo make test` 和 CI 跑的一致。
  - README 中英文：任务表里 `cargo make test` 的说明改为实际覆盖范围；「裸 `cargo test` 只覆盖根 package」那条陷阱补上「也不运行示例里的测试」。
- **`--no-fail-fast` 是实测后加的**：同时在 `examples/rag.rs` 和 `crates/shared` 里各埋一个失败测试，默认模式在 `rag` 处就停了，`shared` 的失败要等前一个修好才会暴露；加 `--no-fail-fast` 一次报出两处（`error: 2 targets failed`）。
- **`Makefile.toml` 的两条命令不能串在 `set -e` 下**（初审后修）：第一条失败就退出，doctest 一个都不跑，正好抵消 `--no-fail-fast` 的意义，也让它与 CI 的两个独立 step 不一致。现在两条都跑、失败按位或汇总到退出码。
- **初审后的数字变化**：上表和下面「验收」里的 898 / 687 是第 15 项落地时的实测。随后的审查修复新增了 2 个 `config` 单测（旧变量告警），所以当前是 900（892 + 8）和 689；概览表（第 2 节「测试」）已按当前值更新。审查修复的完整清单见 `docs/change-review.md`。
- **验收**（均在本地实测，改动随即还原）：
  - 示例测试里埋一个失败断言：原来的 `cargo test --all-features` 没抓到，新命令抓到；
  - `crates/shared` 里埋一个失败测试：新命令抓到；
  - `crates/web-ui` 里埋 clippy 警告（`len() == 0`、多余的 `vec!`）：wasm 命令报错退出；
  - `cargo make test` 实际跑了 898 个测试，含 `examples/rag.rs`、`shared`、`web-ui` 和两个 crate 的 doctest；
  - 其余检查不变：fmt、全 feature clippy、6 种 feature 组合、无 feature 的 687 个测试；`ci.yml` 能被 YAML 解析，六个任务：`fmt`、`clippy`、`test`、`clippy-wasm`、`features`、`test-minimal`。
- **没改的**：
  - `check-features` 和 CI 的 `test-minimal` 保持原样。无 feature 时 `--all-targets` 不会多出测试（`rag` 需要 `sqlite`，本来就被跳过），实测前后都是 687 个。
  - 15.5（CI 里跑 `trunk build`）没做：需要在 CI 里安装 `trunk`，多花时间，而 wasm clippy 已经覆盖了编译错误。要做的话再说。
  - 没有 `web-ui` 的浏览器端测试：现有 6 个测试在宿主上跑，测的是翻译表和命令表这类纯逻辑；渲染、事件流这些要 `wasm-bindgen-test` 加浏览器才能测，本机虽然装了 `wasm-bindgen-test-runner`，CI 里还没配。
  - 新 workflow 本身没有在 GitHub 上跑过：这里只验证了 YAML 能解析、各条命令在本地等价运行。第一次推送后要看一下 `clippy-wasm` 任务（`targets:` 的写法和 rust-cache 的 key）是否一次通过。
