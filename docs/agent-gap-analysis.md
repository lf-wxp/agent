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
| 上下文压缩、摘要、截断 | ✅ | `src/callback/context_optimizer/`；tokenizer 固定为 `cl100k_base`；摘要默认关闭 |
| MCP 客户端 | ⚠️ | 支持 stdio 和 Streamable HTTP；不支持 SSE；不支持 resources、prompts、sampling、elicitation；不支持 `list_changed` 通知；不会断线重连 |
| 会话持久化 | ⚠️ | 只有 `FileSessionStore` 这一种后端 |
| 写入和执行类工具 | ❌ | 内置工具只能读、删、解压，没有 `write_file`、`edit_file`、shell、`grep` |
| 取消、超时 | ❌ | 没有 `CancellationToken`，也没有整轮超时；单个工具没有超时（MCP `call_tool` 可能一直挂住）；工具并发没有上限；文件工具用同步 `std::fs`，会阻塞 async 运行时 |
| 多厂商支持 | ⚠️ | 已有 `ChatBackend` trait（`Provider::with_backend`），内置 OpenAI 兼容实现；还没有原生的 Anthropic / Gemini 适配（可走它们的 OpenAI 兼容端点） |
| Reasoning / thinking | ✅ | 读取 `reasoning_content` / `reasoning`（`AgentResult::reasoning`、`AgentStreamEvent::Reasoning`），支持 `reasoning_effort`，可选回传（`LLM_REPLAY_REASONING`，DeepSeek 思考模式带 tools 时需要）；推理 token 数尚未统计 |
| 采样参数 | ✅ | `ModelSettings`：`max_tokens`（可选 `max_completion_tokens` 字段）、`temperature`、`top_p`、`reasoning_effort` |
| 多模态输入 | ⚠️ | 用户消息只能是 `String`；MCP 返回的图片会被丢弃（`read_image` 已改走 `Provider`） |
| 规划 / 反思 | ❌ | 只有 ReAct 式的循环，没有 todo 工具，也没有自检步骤 |
| 多 agent | ❌ | 子 agent、agent-as-tool、handoff、workflow / graph 都没有 |
| 长期记忆 / RAG | ⚠️ | `knowledge_base` 只在内存中暴力检索，每次都重新计算 embedding；没有 `retrieve` 工具，也没有跨会话记忆 |
| 可观测性 | ⚠️ | 只有 fmt 日志；没有 span 和 OpenTelemetry；`TokenUsage` 不计算成本 |
| 类型化错误 | ❌ | 全部使用 `anyhow` |
| Cargo features | ❌ | 没有 feature，MCP、web、gaia、CLI 相关依赖都会强制编译 |
| 工具定义体验 | ⚠️ | `simple_tool!` 只在 crate 内可见，没有 `#[tool]` 过程宏；`Tool` trait 和 `async-openai` 耦合 |
| 服务化 | ⚠️ | Web UI 只监听 127.0.0.1，没有鉴权，只支持单用户 |
| 测试 | ⚠️ | 约 530 个单测和集成测试；没有真实 LLM 的端到端回归测试；`stream.rs`、`complete.rs`、`embed.rs` 基本没有测试覆盖 |

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
4. **多模态**：
   - 用户消息支持图片；
   - `read_image` 改为走 `Provider`；
   - MCP 返回的图片透传给模型。

### P1：进阶能力

5. **规划和自我修正**：todo / plan 工具，可选的反思步骤。
6. **多 agent**：先做 agent-as-tool（把 `Agent` 包装成 `Tool`），再考虑 handoff 和 workflow 编排。
7. **长期记忆和 RAG**：持久化向量库（例如 SQLite 或 LanceDB）、`retrieve` 工具、跨会话记忆。
8. **可观测性**：用 `#[instrument]` 加 span，接入 OpenTelemetry，统计成本。
9. **补全 MCP**：SSE 传输、resources 和 prompts、`list_changed`、断线重连；把 Agent 本身作为 MCP Server 对外提供。
10. **审批时改写参数（`edit`）**：方案见 `docs/approval-hitl-plan.md` 4.4。

### P2：工程化

- 用 `thiserror` 定义类型化错误；
- 用 Cargo features 按需裁剪依赖；
- 公开工具宏，或提供 `#[tool]` 过程宏；
- 让 `Tool` 和 `async-openai` 解耦；
- token 计数按模型选择 tokenizer；
- LLM 重试时读取 `Retry-After`；
- 提供服务化的 HTTP API 和鉴权；
- 增加端到端回归评测；
- 修正 README 中测试数量不一致的问题：徽章写 418，正文写 528。

## 4. 执行项

按建议顺序逐项实现，**每完成一项停下来审批**，再决定下一步。

| # | 执行项 | 范围 | 状态 |
|---|---|---|---|
| 1 | 写入和执行类工具 | `write_file`、`edit_file`、`grep`、`run_command`；接入工作区沙箱、默认审批和上下文 compaction | ✅ 已完成 |
| 2 | 工具超时和取消 | 单个工具统一超时、`CancellationToken`、工具并发上限、文件 IO 不再阻塞运行时 | ✅ 已完成 |
| 3 | LLM Provider 抽象和模型参数 | Provider trait、reasoning、采样参数与 `max_tokens` 可配置、修复 `read_image` 绕过 `Provider` 的问题 | ✅ 已完成，待审批 |
| 4 | 子 agent 与规划 | agent-as-tool、todo / plan 工具 | ⏳ 待开始 |
| 5 | 可观测性 | span 埋点、接入 OTel、成本统计 | ⏳ 待开始 |
| 6 | 持久化检索和长期记忆 | 持久化向量库、`retrieve` 工具、跨会话记忆 | ⏳ 待开始 |
| 7 | 工程化 | `thiserror`、Cargo features、MCP 补全、公开工具宏等（可以按需再拆分） | ⏳ 待开始 |

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
  - Web 历史接口不返回 `Reasoning` 项。
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
  - Web UI 和终端还不显示推理内容：`AgentStreamEvent::Reasoning` 在 `to_chat_events` 中映射为空，要显示需要新增 `ChatEvent` 并重新构建前端。
  - 推理 token 数没有计入 `TokenUsage`，留给第 5 项。
  - `StructuredAgentResult` 没有 `reasoning` 字段。
