# 工具审批（HITL）能力增强与持久化方案

> 状态：**设计阶段，尚未实现**。第 6 节是实施步骤清单，完成时逐项标 ✅ 并补「修复/实现记录」小节，风格对齐 `web-ui-plan.md`。
>
> 本文档记录设计结论与取舍理由。如实现与本文档有偏差，应先更新本文档再改代码。

## 1. 背景

### 1.1 现状能力

`cli` 二进制目前的审批链路（由 `web-ui-plan.md` 第 3.4 / 6.5 节引入）：

| 组件 | 位置 | 职责 |
|---|---|---|
| `BeforeToolCallback` | `src/agent/callback.rs` | 工具执行前拦截，返回 `Some((status, content))` 即短路 |
| `DualApprovalCallback` | `src/callback/dual_approval.rs` | 命中危险工具名时向 session 广播审批请求并等待决策 |
| `ApprovalRegistry` | 同上 | session 级在途审批表，首答生效 |
| `ApprovalChannel` + `with_approval_channel` | 同上 | `tokio::task_local!` 按 turn 路由到终端或 session 广播 |
| `ChatEvent::ApprovalRequired` / `ApprovalResolved` | `crates/shared/src/lib.rs` | 前后端线上协议 |
| `POST /api/approve/{id}` | `src/bin/cli/web.rs` | 浏览器应答入口 |
| `resolve_pending_approval` | `src/bin/cli/main.rs` | 终端在 `You>` 处应答其他视图raise的审批 |

已经做对、**后续改动必须保留**的设计：

- **审批归属 session 而非单个前端**：终端与任意数量浏览器 tab 都能应答，首答生效（`ApprovalRegistry::resolve` 用 remove 保证）。
- **一切等待有界 + fail-closed**：超时、无人监听、决策通道被丢弃，三种情况全部默认拒绝。理由见 `src/config.rs` 中 `approval_timeout()` 的注释——turn 持有 session 的 turn lock，无人应答的审批会拖死所有视图。
- **展示用 `raw_arguments` 而非解析后的 `arguments`**：解析失败会显示成 `null`，让人批准看不见的调用比不问更糟。
- **turn 结束清理**：三条驱动路径（`web::drive_turn`、`drive_terminal_turn`、`publish_approvals`）都调用 `approvals.discard(&raised)`。

### 1.2 已确认的缺口

| # | 缺口 | 影响 |
|---|---|---|
| G1 | 在途审批**没有可查询的数据表示**，只以 `oneshot::Sender` 形式存在 | 刷新浏览器 tab 后该 tab 永远看不到当前审批（`broadcast` 不回放历史事件，`/api/history` 不含审批项），只能干等超时 |
| G2 | 审批粒度只到**工具名精确匹配** | 无法表达「删 `/tmp` 下的不问、删其他的要问」 |
| G3 | 无粘性决策 | 同一工具连续调用 N 次要答 N 次 |
| G4 | 响应模型只有**批准/拒绝**两种 | 无法改写参数后放行；拒绝理由硬编码为 `User denied execution of {tool}` |
| G5 | 审批状态不持久化 | 进程重启后在途审批彻底丢失 |
| G6 | turn 不可恢复 | 即使审批状态落盘，`run_continuing` 的执行进度仍在栈上 |

其中 **G1 是明确的 bug**（turn 还活着、审批仍然有效，只是查不到），其余是能力缺失。

### 1.3 中断语义现状

| 中断方式 | 审批能否接续 | 原因 |
|---|---|---|
| 超时 | 否，已成终局拒绝 | fail-closed 写入 transcript，turn 正常结束并落盘 |
| kill 进程 | 否，整个 turn 都丢失 | `record_turn` 只在 `Done` 事件/结果返回后调用 |
| 刷新浏览器 tab | 服务端仍有效，但该 tab 看不到 | G1 |
| 刷新 tab 后改用终端应答 | **能** | `resolve_pending_approval` 查的是 `any_pending()` 实时状态 |

注意 `Ctrl-C` 在 `reedline` 层被转成空行（`read_line` 的 `Ok(_) => Ok((Some(String::new()), editor))`），**不能中断正在跑的 turn**，所以「审批中途按 Ctrl-C 取消」这个操作当前并不存在。

## 2. 关键设计判断：不在 `ContentItem` 上表达审批状态

一个直觉的做法是给 `ContentItem::ToolCall` 加 `state: Pending | Approved | Denied`。**这条路会撞上两个硬约束，不要走。**

### 2.1 约束一：`ContentItem` 是 LLM 请求的直接来源

`LlmRequest::new` 把 events 摊平成 `contents`，`build_messages` 再逐项转成 API message。带 `Pending` 的 `ToolCall` 仍然会被渲染成 `assistant.tool_calls`：

```896:907:src/agent/runtime/mod.rs
        ContentItem::ToolCall {
          tool_call_id,
          name,
          arguments,
        } => {
          let tool_call = ChatCompletionMessageToolCalls::Function(ChatCompletionMessageToolCall {
            id: tool_call_id,
            function: FunctionCall {
              name,
              arguments: arguments.to_string(),
            },
          });
```

而 OpenAI 协议要求每个 `tool_call` 必须有配对的 `tool` message，否则请求直接失败。当前这个不变量靠「每个 call 必然在同一轮产生 result」免费保证，引入 `Pending` 就打破了。

注意 `safety.rs` 只守了反方向（result 不能没有 call）：

```4:7:src/callback/context_optimizer/safety.rs
//! Every strategy in [`super`] that drops items from the middle or front of a
//! conversation has the same obligation: a [`ContentItem::ToolResult`] may never survive
//! without the [`ContentItem::ToolCall`] that produced it. Most providers reject such a
//! request outright, so an over-eager trim turns into a hard API error rather than a
//! slightly worse answer.
```

「call 不能没有 result」没有任何东西兜着。

### 2.2 约束二：transcript 是 append-only + 全量覆盖写

```5:7:src/agent/llm_request.rs
//! [`ExecutionContext::events`](crate::agent::ExecutionContext::events) is the
//! authoritative, append-only transcript and is never
//! modified through this path.
```

审批状态是**可变**的（Requested → Approved/Denied）。落盘是 `store.save(scope, id, 全量 Vec<Event>)` + 整文件 rename + last-write-wins。把可变状态塞进不可变记录，等于要重做存储层。

### 2.3 `ToolResultStatus` 同样不该扩

它描述**工具执行的结果**，审批是**执行前的闸门**，两个正交的域。当前拒绝复用 `Error` 是有意的：模型只需看到一条普通 Error，完全不需要知道「审批」这个概念存在。

### 2.4 已存在的隐性问题：中断时 transcript 处于非法中间态

`record_tool_calls` 在 `execute_tool_calls` **之前**就把本轮所有 `ToolCall` 写进了 transcript：

```705:710:src/agent/runtime/mod.rs
    context.add_event(Event::new(
      context.execution_id.clone(),
      "agent",
      call_items.clone(),
    ));
    call_items
  }
```

所以审批挂起期间，`context.events` 里已经有一批 `ToolCall` 而没有对应的 `ToolResult`——**2.1 那个协议不变量此刻已经是破的**。目前之所以看不出问题，只是因为这种中间态永远不会落盘（`record_turn` 只在 turn 完成时调用）。

这条对 P5 是硬性约束：**任何要落盘的中间态，都必须先给每个未决 `ToolCall` 补一条 `ToolResult`**，否则恢复后的第一次 LLM 请求就会被 provider 拒。

### 2.5 结论

审批状态作为**独立的旁路实体**持久化，用 `tool_call_id` 与 transcript 关联，两边结构都不用改。这个判断与业界一致——LangGraph 把 interrupt 写进 checkpoint 的 **pending writes** 区（专用 `INTERRUPT` 通道键）而非业务 state，正是同一个选择。

## 3. 业界对标

### 3.1 OpenAI Agents SDK（形态最接近，主要参考）

| 机制 | 做法 |
|---|---|
| 声明 | `needs_approval=True` 或 `async fn(ctx, params, call_id) -> bool` |
| 暴露待审 | `RunResult.interruptions` → `ToolApprovalItem { tool_name, arguments, agent.name }` |
| 持久化 | `result.to_state()` → `state.to_string()` / `to_json()` |
| 恢复 | `RunState.from_json(...)` → `state.approve/reject(item)` → `Runner.run(agent, state)` |
| 粘性决策 | `always_approve` / `always_reject`，按 call ID 或工具身份作用于本 run |
| 部分决议 | `interruptions` 可只批准一部分，重跑后已决的继续、未决的再次暂停 |
| 拒绝文案 | `RunConfig.tool_error_formatter`（run 级兜底）+ `state.reject(rejection_message=...)`（单次覆盖） |

两个值得直接抄的点：

**fail-closed 被做成了明文规则**：参数缺失、空白、JSON 错误、合法 JSON 但非对象、含 `NaN`/`Infinity` —— 这些情况下**根本不调用判定函数，直接要求审批**。

**`RunState` 可序列化的前提是 run 状态本质上是数据**（消息列表 + 待决审批 + usage），不是 async 栈。这一点对本项目是好消息，见 4.5。

安全提示（官方文档明确）：序列化的 state 含 app context，不要在里面放 secret；`include_tracing_api_key=True` 会把 key 写进载荷。

### 3.2 LangGraph（暂停/恢复语义最成熟）

- `interrupt()` 抛 `GraphInterrupt`，写入 checkpoint 的 pending writes 区，不污染业务 state。
- **恢复时节点整体重跑**，`interrupt()` 先查 task scratchpad，命中就 return 而不抛。代价：**`interrupt()` 之前的副作用会重复执行**，官方建议审批节点与工具执行节点必须拆开。
- interrupt ID 由 checkpoint namespace + 任务内计数器**确定性推导**，保证重放时同一调用点得到同一 ID。
- 四种响应契约：`accept` / `edit`（改参数后放行）/ `response`（回灌 ToolMessage 让 LLM 改策略）/ `ignore`。
- **不带审批超时**，需应用层扫描挂起 thread 用 `update_state` 清除。

### 3.3 MCP Elicitation（协议层补位）

MCP 规范已有 `elicitation`：server 可在工具执行中途经 client 向用户征询结构化数据（JSON Schema 校验）。form 模式在 `2025-06-18`，URL 模式在 `2025-11-25`。

本项目本身是 MCP client（`ToolRegistry::with_mcp`、`McpGuardCallback`），当前审批是纯 client 侧决定拦不拦；Elicitation 让 server 能主动发起询问。现有 `PendingApproval` + session 广播机制正好是接这个的载体。**本轮不做，列为后续可选项。**

### 3.4 Durable Execution（对应 G6，本轮不做）

分层认知：LangGraph checkpointer 管 agent 内部状态图，Durable Execution 管整段业务工作流的执行容器，两者可叠加。

| 引擎 | 等待原语 | 部署 | 适配度 |
|---|---|---|---|
| Temporal | Signal | 重（PG + ES + Cassandra） | 低 |
| Inngest | `waitForEvent` | 核心调度走 SaaS | 低（本项目是本地 CLI） |
| DBOS | `recv()` | Postgres 一张表 | 中 |
| **Restate** | **Awakeable** | **Rust 单二进制** | **高** |

Replay 模型的确定性要求：LLM 输出不确定会破坏重放，必须把 LLM 调用放进不要求确定性的 Activity。

### 3.5 对标小结

**本项目已领先的**：内建 `approval_timeout` + 全路径 fail-closed（LangGraph 需自己实现）；审批归属 session 而非发起端（多数实现绑定单前端）。

**要补的**：G1 可查询性（对应 `interruptions` / `get_state().interrupts`）、G2 谓词粒度、G3 粘性决策、G4 四种响应、G5/G6 持久化与恢复。

## 4. 方案设计

### 4.1 P1 — 待审批可查询（修 G1）

业界不把这当问题，是因为 `RunResult.interruptions` 天然可查。本项目的症结是审批只以 channel 形式存在：

```86:88:src/callback/dual_approval.rs
pub struct ApprovalRegistry {
  pending: Mutex<HashMap<String, oneshot::Sender<bool>>>,
}
```

**改动**：

1. 新增可序列化的 `ApprovalMeta { id, tool, raw_arguments, requested_at }`（放 `dual_approval.rs`）。
2. `pending` 改为 `HashMap<String, (ApprovalMeta, oneshot::Sender<bool>)>`；`register(id, meta, decision)`。
3. 新增 `pending_snapshot() -> Vec<ApprovalMeta>`，按 `requested_at` 排序。
4. `crates/shared` 新增 `PendingApproval { id, tool, arguments, requested_at }` 与 `GET /api/approvals` 的响应体。
5. `src/bin/cli/web.rs` 新增 `GET /api/approvals` 路由（走同一个 `guard_loopback_host` 中间件）。
6. `crates/web-ui` 挂载时与 `/api/history` 并行拉取，把结果渲染成与 `ApprovalRequired` 完全相同的卡片（复用 `render_approval`）。

**顺手偿还的设计债**：`drive_terminal_turn` 现在为了留住展示元数据，塞了个占位 channel 进队列：

```709:714:src/bin/cli/main.rs
        queued.push_back(PendingApproval {
          // `decision` now lives in the registry; the queue only needs what it takes to
          // render the prompt, so a placeholder channel stands in for it.
          decision: tokio::sync::oneshot::channel().0,
          ..pending
        });
```

`queued` 改存 `ApprovalMeta` 后这个 hack 直接消失。这正是「元数据与决策通道应当分开存」的证据。

**注意命名冲突**：`dual_approval::PendingApproval`（含 `oneshot::Sender`，native 专用）与要新增的 `shared::PendingApproval`（线上类型）同名。建议把后者命名为 `PendingApprovalView`，或把前者改名 `ApprovalRequest`。

### 4.2 P2 — 审批粒度从工具名升级为谓词（修 G2）

当前是 `HashSet<String>` 精确匹配（大小写敏感，有测试钉住 `delete` 不匹配 `delete_file`）。

**改动**：`DualApprovalCallback` 的策略表改为

```rust
pub enum ApprovalRule {
  /// 该工具的任何调用都要审批（等价于当前行为）
  Always,
  /// 按解析后的参数动态判定。返回 true 才审批。
  When(Arc<dyn Fn(&ToolCallView<'_>) -> bool + Send + Sync>),
}
// dangerous_tools: HashMap<String, ApprovalRule>
```

保留 `new(impl IntoIterator<Item = impl Into<String>>)` 作为 `Always` 的糖，现有调用点与测试不用改。

**fail-closed 前置规则**（在调用谓词之前判定，命中即强制审批，不询问谓词）：

| 条件 | 判定依据 |
|---|---|
| `raw_arguments` 为空或全空白 | 直接检查字符串 |
| JSON 解析失败 | `arguments == Value::Null` **且** `raw_arguments.trim() != "null"` |
| 合法 JSON 但不是 object | `!arguments.is_object()` |
| 含 `NaN` / `Infinity` / `-Infinity` | `serde_json` 默认已拒（解析失败，归入上一条），加测试钉住 |

第二条那个区分是必要的：`ToolCallView.arguments` 在解析失败时被填为 `Value::Null`——

```768:768:src/agent/runtime/mod.rs
          let parsed_arguments: Value = serde_json::from_str(arguments).unwrap_or(Value::Null);
```

——所以「模型真的传了 `null`」和「解析失败」在 `arguments` 上无法区分，必须回看 `raw_arguments`。

### 4.3 P3 — 粘性决策（修 G3）

**复用现成设施**：`ContinuityCache<T>`（`src/agent/context.rs:169-237`）就是为「跨 turn 的 per-conversation 有界缓存」设计的，其文档注释说明的正是这个场景——回调被并发 run 共享且永远不知道某个 run 何时结束，所以累积的状态必须有界且按会话隔离。

**改动**：

1. `ApprovalDecision`（`crates/shared`）加 `scope: DecisionScope { Once, AlwaysForTool }`。
2. `DualApprovalCallback` 持 `Mutex<ContinuityCache<HashMap<String, bool>>>`，key 用 `ExecutionContext::continuity_key()`，内层 key 用工具名。
3. `call` 开头先查粘性决策：命中 `true` 直接返回 `None`（放行），命中 `false` 直接返回拒绝，都不发起询问。
4. `/reset` 与 `--fresh` 要清除对应 `continuity_key` 的粘性决策（`clear_session` 里加一步），否则「忘记这段对话」和「记住我批准过」会矛盾。

**作用域选择**：OpenAI 的语义是「本次 run 剩余时间内」。本项目选 **per-conversation**（`continuity_key` 已折入 scope，天然隔离终端/web），因为 CLI 的一轮 turn 往往很短，per-turn 粘性几乎没有价值。代价是需要上面第 4 条的显式清除。

### 4.4 P4 — 响应模型扩到四种（修 G4）

| 模式 | 语义 | 实现成本 |
|---|---|---|
| `accept` | 原样放行 | 已有 |
| `ignore` | 拒绝并终止该调用 | 已有 |
| `response` | 带自定义理由拒绝，让模型据此改策略 | **低** |
| `edit` | 人改写参数后放行 | **高，见下** |

**`response`（建议本轮做）**：`ApprovalDecision` 加 `rejection_message: Option<String>`，`DualApprovalCallback` 拒绝时用它替换硬编码文案：

```324:327:src/callback/dual_approval.rs
      Some((
        ToolResultStatus::Error,
        format!("User denied execution of {}", tool_call.name),
      ))
```

同时加一个 run 级兜底（对标 `RunConfig.tool_error_formatter`）：`DualApprovalCallback::with_rejection_formatter(Arc<dyn Fn(&ToolCallView) -> String>)`。优先级：单次 `rejection_message` > formatter > 当前默认文案。

**`edit`（建议单列，不在本轮）**，因为它要动 `BeforeToolCallback` 的返回类型：

```31:38:src/agent/callback.rs
#[async_trait::async_trait]
pub trait BeforeToolCallback: Send + Sync {
  async fn call(
    &self,
    context: &ExecutionContext,
    tool_call: ToolCallView<'_>,
  ) -> Option<(ToolResultStatus, String)>;
}
```

`Option<(..)>` 只能表达「放行」或「短路」，没有「改写参数后放行」。需要：

```rust
pub enum ToolCallDecision {
  Proceed,                                        // 原 None
  ShortCircuit(ToolResultStatus, String),         // 原 Some((..))
  ProceedWith { raw_arguments: String },          // 新增
}
```

影响面：4 个实现（`approval`、`dual_approval`、`path_guard`、`mcp_guard`）+ `execute_tool_calls` 调用点 + 全部相关测试。可以加 `impl From<Option<(ToolResultStatus, String)>> for ToolCallDecision` 降低迁移成本，但 trait 签名本身是 breaking change。

**还有一个语义问题必须先定**：`record_tool_calls` 已经在审批之前把原始参数写进 transcript（见 2.4）。`edit` 之后，transcript 记的是模型的原始参数，实际执行的是改写后的参数，两者不一致。三个选项：

| 选项 | 代价 |
|---|---|
| 接受不一致，transcript 记原始 | 审计时看不出实际执行了什么 |
| 把 `record_tool_calls` 推迟到审批之后 | 改执行顺序，且 `ToolCallsStarted` 流式事件依赖它，浏览器会晚一步看到调用 |
| 审批后追加一条 `ToolCall` 修正项 | 破坏「一个 call 一条记录」，`find_safe_start` 的 `call_index` 逻辑要跟着改 |

倾向第二个，但需要单独评估对流式时序的影响。**本轮不决策。**

**另一条硬约束（9.5 实测得出）**：`edit` 改写后的参数同样是「注入 transcript 的人类意图」，必须可归因。若只是静默替换参数，模型看到的是一个它没发出过的调用；若在结果里说明改动，则说明文字必须标注来源，否则模型会（正确地）把它当作经工具输出夹带的指令而拒绝配合。设计 `edit` 时须一并给出归因方案。

### 4.5 P5 — 跨进程恢复（修 G5/G6）

**修正一个早期判断**：曾认为 async 栈不可序列化，所以只能做「丢弃并重放」。但 OpenAI SDK 做到了真恢复，因为它的 run 状态本质是数据。本项目的 `ExecutionContext` 已经很接近：

```99:114:src/agent/context.rs
#[derive(Debug)]
pub struct ExecutionContext {
  /// Identifies this one run. A fresh value per
  /// [`crate::agent::Agent::run_continuing`] call, including successive turns of the same
  /// conversation — use [`Self::conversation_id`] to correlate those.
  pub execution_id: String,
  /// Identity of the conversation this run continues, when the caller supplied one (see
  /// [`Conversation`]). `None` for a one-shot [`crate::agent::Agent::run`].
  pub conversation_id: Option<String>,
  /// Namespace [`Self::conversation_id`] is unique within; see [`Conversation::scope`].
  pub conversation_scope: Option<String>,
  pub events: Vec<Event>,
  pub current_step: u32,
  pub final_result: Option<String>,
  pub usage: TokenUsage,
}
```

`events` / `current_step` / `usage` / `final_result` 全是可序列化数据。真正卡住的只有一处：`execute_tool_calls` 里同一轮的并发调用没有「哪些已完成、哪些待审批」的中间表示，整批要么全成要么全丢。

**改动**：

1. `ExecutionContext`、`TokenUsage` 加 `Serialize`/`Deserialize`。
2. `execute_tool_calls` 内部改为可分批：记录 `completed: Vec<ContentItem>` 与 `awaiting: Vec<(FunctionCall, ApprovalMeta)>`。
3. 新增 `AgentRunState { context, completed, awaiting, fingerprint }`（可序列化）。
4. `run_continuing` 的返回改为 `AgentOutcome::Done(AgentResult) | AgentOutcome::AwaitingApproval(AgentRunState)`。
5. 新增 `ApprovalStore` trait + `FileApprovalStore`，与 `SessionStore` 平行；目录权限对齐 `FileSessionStore` 的 `0700` 处理（`restrict_dir_permissions`）。
6. 新增 `Agent::resume(state, decisions) -> AgentOutcome`。
7. **落盘前补齐协议不变量**：给每个 `awaiting` 的 `ToolCall` 写一条占位 `ToolResult`（`status: Error`，content 形如 `Awaiting user approval`），理由见 2.4。旁路的 `ApprovalStore` 记录标记它「非终局、可重放」。
8. 恢复后用 `BeforeLlmCallback` 把这些占位 result 从本轮 `LlmRequest.contents` 里滤掉——这正是该 hook 的设计用途（改本轮请求而不动 transcript）。

**必须复用存下来的 `tool_call_id`，不能重新问模型**：id 是模型生成的，重放会变，而 `ApprovalStore` 记录和 transcript 都以它为关联键。对比 LangGraph 用「checkpoint namespace + 任务内计数器」确定性推导 interrupt ID，本项目靠持久化原 id 达到同样效果。

### 4.6 P6 — 版本标记（P5 的必需配套）

OpenAI 文档明确建议：长时间挂起的审批要在序列化 state 旁存版本标记，反序列化时路由到匹配代码路径，否则 prompt / 工具定义 / 模型变更后恢复出来的东西不兼容。

**改动**：`AgentRunState.fingerprint` 记录 `{ instructions_hash, tool_names_hash, model, state_version }`。`resume` 时不匹配则拒绝恢复，向用户报明原因并把该审批标记为 `Expired`，而不是静默跑出错误结果。

**从第一天就要带上**，后补很痛（已落盘的旧 state 没有指纹可校验）。

### 4.7 本轮不做

| 项 | 原因 |
|---|---|
| MCP Elicitation 接入 | 协议侧能力，等 P1–P4 稳定后再评估；现有广播机制已是可用载体 |
| 引入 Durable Execution 引擎 | P5 是自包含改动，不需要外部基建。真需要「挂起数天、跨部署」时再考虑 Restate（唯一形态匹配：Rust 单二进制 + Awakeable） |
| 按 `session_id` 细粒度锁 | 与本方案无关，见 `web-ui-plan.md` 第 11 节 |

## 5. 不变量清单（改动时逐条核对）

1. **tool_call 与 tool_result 必须配对**，任何落盘的中间态都要满足（2.1 / 2.4）。
2. **`find_safe_start` 只守单向**，若引入未决 call，需要对称的过滤或在 `build_messages` 跳过无 result 的 call。
3. **重放不得重复副作用**：同轮已完成的工具调用不能再跑一次（LangGraph 踩过的坑）。
4. **`tool_call_id` 必须持久化复用**，不能依赖模型重新生成（4.5）。
5. **审批超时 + 全路径 fail-closed 必须保留**，这是本项目相对业界的优势，不能在重构中丢掉。
6. **`oneshot::Sender` 不可序列化**：内存 registry 与持久记录必须分离，持久层只存 `ApprovalMeta`。
7. **序列化内容含 `raw_arguments`**，可能带敏感数据；对齐 `FileSessionStore` 的「plaintext on disk + 0700 目录」处理与文档警示。
8. **首答生效语义不能退化**：`ApprovalRegistry::resolve` 的 remove 语义是它的实现基础。
9. **任何注入 transcript 的人类意图必须可归因**（9.5 实测）：拒绝理由、`edit` 后的参数说明等,都必须带明确的来源标注。裸文本与工具自身输出不可区分,对齐良好的模型会正确地拒绝执行它。

## 6. 实施步骤

分四个阶段。**P1–P4a 相互独立**，可单独合并、单独验证；P5/P6 依赖前四项且是 breaking change。

### 阶段一：可查询性（修 G1，明确 bug）— 已完成，见第 8 节

1. ✅ `ApprovalMeta` + `ApprovalRegistry` 携带元数据 + `pending_snapshot()`；`register` 签名变更；消除 `drive_terminal_turn` 的占位 channel hack
2. ✅ `crates/shared` 加线上类型（命名为 `PendingApprovalView`，避开与 `dual_approval::PendingApproval` 冲突）+ `GET /api/approvals` 路由
3. ✅ `crates/web-ui` 挂载时拉取并复用审批卡片渲染；`push_approval` 去重处理与实时流的竞态

### 阶段二：策略能力（修 G2/G3/G4a）— 已完成，见第 9 节

4. ✅ `ApprovalRule::{Always, When}` + fail-closed 前置规则（4.2 表格逐条加测试）
5. ✅ 粘性决策：`ApprovalOutcome { approved, sticky }` + 复用 `ContinuityCache` + `/reset`/`--fresh` 清除
6. ✅ 自定义拒绝文案：`ApprovalOutcome.reason` + `with_rejection_formatter`，三级优先级（**含一处实测发现的修正，见 9.5**）

### 阶段三：持久化与恢复（修 G5/G6）— 已完成，见第 11 节

7. ✅ `ExecutionContext`/`TokenUsage` 加 serde；`AgentRunState` + `fingerprint`（P6 同步落地）
8. ✅ `execute_tool_calls` 支持部分完成（`completed` / `suspended`）
9. ✅ `AgentOutcome` + `Agent::resume` / `resume_stream`
10. ✅ `ApprovalStore` + `FileApprovalStore`（0700 + 原子写，对齐 `FileSessionStore`）
11. ✅ CLI + Web 接入：启动时提示未决审批；两端共用 `/resume` `/discard`

### 阶段四：可选增强 — `edit` 已决定暂不做，见第 13 节

12. ⏸ `edit` 模式（影响面已核实，结论是**推迟**：现有的「带理由拒绝」已覆盖大部分场景）
13. ⬜ MCP Elicitation 接入评估

### 收尾

14. ✅ 更新 `README.md`（审批行为、挂起与恢复、新命令与配置项）与本文档（实现记录见 11 / 12 节）

## 7. 待确认

- P3 粘性决策的作用域定为 per-conversation（4.3），若实际用起来觉得过宽，退化为 per-turn 只需换 key。
- P4 `edit` 的 transcript 一致性方案（4.4 三选一），倾向「推迟 `record_tool_calls`」，但要先评估对 `ToolCallsStarted` 流式时序的影响。
- P5 的 `ApprovalStore` 是否与 `FileSessionStore` 共用目录。倾向共用（同一份 `AGENT_CLI_SESSION_DIR`，不同文件名前缀），少一个配置项。

## 8. 实现记录：阶段一（审批可查询）

**问题**：turn 挂起等待审批期间刷新浏览器 tab，该 tab 再也看不到审批提示，只能等超时。而服务端此刻审批仍然有效、终端仍能应答。

**根因**：审批只以 `oneshot::Sender` 的形式存在于 registry，没有任何可查询的数据表示。`ApprovalRequired` 仅通过 `broadcast` 实时推送（不回放），`GET /api/history` 返回的 transcript 又不含审批项——两个信息源都覆盖不到「turn 进行中加入的视图」。

**改动**：

- `src/callback/dual_approval.rs`：拆出 `ApprovalMeta { id, tool, raw_arguments, requested_at }`，`PendingApproval` 变为 `{ meta, decision }`。`ApprovalRegistry` 的 map 值改为 `(ApprovalMeta, oneshot::Sender<bool>)`，`register(meta, decision)` 一个参数完成登记。
- 新增 `pending_snapshot() -> Vec<ApprovalMeta>`，按 `requested_at` 升序、id 兜底。`any_pending()` 同步改为返回**最早**的一条（原先是 `HashMap` 迭代顺序），与 snapshot 共用 `compare_by_age`——两个读取口必须同序，否则终端会显示一条、应答另一条。
- `crates/shared`：新增 `PendingApprovalView`。刻意与 native 的 `PendingApproval` 区分命名，后者持有 agent 正阻塞其上的决策通道，既不可序列化也在进程外无意义。
- `src/bin/cli/web.rs`：新增 `GET /api/approvals`（走同一个 `guard_loopback_host`）。三条驱动路径（`web::drive_turn`、`drive_terminal_turn`、`publish_approvals`）统一调整为**先 register 再广播**，否则收到事件立刻查询的视图可能看到空列表。
- `crates/web-ui`：新增 `load_pending_approvals`，与 `load_history` **在同一个 task 内串行**执行——审批按定义比整段 transcript 更新，两个独立 task 的完成顺序不定，会把审批卡片渲染到历史上方。
- 新增 `ChatState::push_approval`，按 `tool_id` 去重。这是让快照拉取与实时流能并存的关键：同一条审批可能从两个路径抵达，谁先到取决于请求时序，任一方都不能假定自己是首次引入。

**顺带偿还的设计债**：`drive_terminal_turn` 原先为了在队列里留住展示元数据，塞了一个占位 `oneshot::channel().0`。元数据与决策通道分离后该 hack 自然消失，`queued` 直接存 `ApprovalMeta`。

**验证**：`cargo test` 全绿（359 + 28 + 21 + 9 + 2），`cargo clippy --all-targets` 与 wasm target clippy 均零警告。新增 7 个测试覆盖 snapshot 描述完整性、resolve 后离开快照、两个读取口的同序性与 id 兜底。

**尚未做**：真实浏览器端的刷新回归（需 `trunk build` 后用 Playwright 实测「turn 等待审批时刷新 tab 仍能看到并应答」）。→ **已在第 10 节完成**。

## 9. 实现记录：阶段二（审批策略能力）

### 9.1 谓词粒度 + fail-closed（G2）

`dangerous_tools: HashSet<String>` 改为 `rules: HashMap<String, ApprovalRule>`：

- `ApprovalRule::Always` —— 等价原行为，`new(tools)` 仍然构造这个，现有调用点与测试不变。
- `ApprovalRule::When(Arc<dyn Fn(&ToolCallView) -> bool>)` —— 按参数判定，配 `ApprovalRule::when(closure)` 语法糖避免调用方写 `Arc`。
- `with_rule(tool, rule)` 追加/覆盖单条。

**fail-closed 落地为 `unreadable_arguments` 前置检查**，命中即当作 `Always` 处理、**完全不询问谓词**。四类：`Empty`（空/全空白）、`Unparsable`（JSON 解析失败，含 `NaN`/`Infinity`/`-Infinity`——serde_json 直接拒绝这些非标准常量）、`NotAnObject`（合法 JSON 但是 null/数组/标量）。

必要性不只是保守：谓词典型写法是「路径不在 `/tmp` 下就要审批」，畸形载荷让它读到的每个字段都缺失，于是最廉价的绕过方式就是发一个坏 JSON。测试用计数器断言谓词**调用次数为 0**，而非只断言结果。

实现上有个必须注意的点：`ToolCallView.arguments` 在解析失败时被填成 `Value::Null`（见 `runtime/mod.rs` 的 `unwrap_or(Value::Null)`），所以「模型真的传了 `null`」和「解析失败」在它上面无法区分，必须回看 `raw_arguments`。两者都不可读但归因不同，否则日志会把畸形载荷说成「模型传了 null」。

### 9.2 粘性决策（G3）

决策通道的载荷从 `bool` 改为 `ApprovalOutcome { approved, sticky }`——「同意」和「同意且别再问」是两个不同的答案，只有应答的人知道是哪个。连带改动：`ApprovalRegistry::resolve(id, outcome)`、`shared::ApprovalDecision` 加 `sticky`（带 `#[serde(default)]`，让旧的 wasm bundle 仍能反序列化为一次性决策）。

存储直接复用 `ContinuityCache<HashMap<String, bool>>`，key 用 `ExecutionContext::continuity_key()`。选它而不是普通 map 是因为其文档描述的正是这个场景：回调被并发 run 共享且永不知道某个 run 何时结束，累积状态必须有界且按会话隔离。`put_with` 的 merge 参数用来**合并而非替换**——同一会话里两个工具可以各自有记忆。

`remember` 只在 `outcome.sticky` 时调用，且**超时路径显式构造 `once(false)`**：超时是「没有答案」，把它记成长期拒绝等于凭空发明一条指令，还会让该工具在本会话余下时间静默失效。有专门测试钉这一点。

命中记忆时**在发布任何事件之前就短路**，所以不会出现「前端显示一个已决定的审批」或「已决定的审批超时」。

### 9.3 `/reset` 清除记忆

这一项牵出两处结构调整：

1. `main.rs` 原先直接把 `DualApprovalCallback` 塞进 `Arc::new(...)` 交给 agent，之后无法再取回。改为先建 `Arc<DualApprovalCallback>` 留一份，注册时显式 `as Arc<dyn BeforeToolCallback>` 转换——同一份分配既是 agent 的 hook，也是本 binary 清除记忆的句柄。`WebState` 同样持有它，使浏览器提交的 `/reset` 也能清。
2. `continuity_key` 的构造逻辑从 `ExecutionContext` 方法里抽出为 `pub fn continuity_key_for(scope, id)`，方法改为调用它。原因：reset 发生在两个 turn **之间**，此时没有 context，却必须定位到 turn 们用过的那个 key。在调用点重新拼一遍会把 NUL 连接规则放到两处，一旦漂移就是「本该被清除的状态活过了 reset」——这种 bug 不会报错，只会静默地把标准权限带过用户画的边界。

`clear_session` 从 `main.rs` 移到 `commands.rs` 并由 `/reset` 与 `--fresh` 共用，签名含 `Option<&DualApprovalCallback>`（`None` 表示无任何 gated 工具）。

### 9.4 交互扩展

终端与 web 的作用域词汇统一由 `parse_approval_answer` 一处解析（`y`/`n`/`a`/`d` 及其长写法）。两个 console 读取点——`prompt_terminal` 与 CLI REPL 的 `read_approval_line`——共用它；两套解析必然漂移，而「一个键在这儿认、在那儿不认」两边都像 bug。

它返回 `Option` 而非带默认值，好让调用方自己决定「无法识别」意味着什么：console 提示 fail-closed 拒绝，而 REPL 把该行拦下并重述问题（那里一个错字远比拒绝意图更可能）。

Web UI 加了「总是允许 / 总是拒绝」两个视觉次级按钮 + 一行说明，i18n 三语同步。次级是有意的：这两个答案会让提示不再出现，不该和只管眼前这一次的决定平起平坐。

**验证**：`cargo test` 全绿（377 + 29 + 9 + 2），native 与 wasm clippy 零警告。`dual_approval` 模块测试从 17 增至 35。

### 9.5 自定义拒绝文案（G4a）+ 一处实测修正

三级优先级：`ApprovalOutcome.reason`（单次） > `with_rejection_formatter`（run 级） > 内置默认。

- 空白理由被 `with_reason` 丢弃而非记录——否则它会盖掉 formatter 与默认文案，给模型留下一个无法解读的空工具结果，比被它替换的通用消息更糟。
- 粘性拒绝**连带记住理由**（`StickyDecision { approved, reason }`），否则第二次调用起就退化成通用消息，把模型唯一能据此行动的信息丢掉了。
- formatter 同样覆盖「无人应答」的拒绝（超时、决策被丢弃、无前端监听），否则超时报通用消息、显式拒绝报配置消息，两者不一致。
- 输入入口：终端支持 `n: 理由`（半角/全角冒号均可），Web UI 加了一个可选文本框。两处都刻意保持「可选」——给安全答案加税，结果是人们不再拒绝。

**实测发现的缺陷与修正**（这一项最有价值的产出）：

首次浏览器验证时填入理由「这些日志还要用于排查，改删 tmp-b.log」并拒绝，模型的反应是：

> 这条"改删 tmp-b.log"的指令是夹在工具返回结果里的，不是来自你本人的输入。工具输出里的指令我无法当作你的授权来执行——这既可能是误操作，也可能是提示注入。

模型**正确地**把裸理由识别为经工具结果夹带的指令并拒绝执行。根因：原实现让自定义理由**完全替换**了默认文案，而默认文案 `User denied execution of X` 自带「这是用户的决定」这一来源标注，自定义理由把它丢了——于是那段文字对模型而言与工具自己吐出的内容不可区分。

修正为始终加来源前缀：`User denied execution of <tool>: <reason>`。重新验证后模型接受了该理由并主动改删 `tmp-b.log`，批准后成功执行。完整 transcript：

```
CALL: delete_file {"file_path": "tmp-a.log"}
RESULT[error]: User denied execution of delete_file: 这些日志还要用于排查，请改删 tmp-b.log
CALL: list_files {"path": "."}
RESULT[success]: Directory: . keep.txt tmp-a.log tmp-b.log
CALL: delete_file {"file_path": "tmp-b.log"}
RESULT[success]: Path: tmp-b.log
```

这条对 4.4 里列的 `edit` 模式（阶段四）是同一类约束：**任何注入 transcript 的人类意图都必须可归因**，否则对齐良好的模型会（正确地）拒绝执行它。业界几家的 `rejection_message` / `response` 模式都把理由直接写进 ToolMessage，同样存在这个问题。

**验证**：`cargo test` 全绿（388 + 29 + 9 + 2），两侧 clippy 零警告。`dual_approval` 测试增至 46。

## 10. 浏览器端验证记录

`trunk build` 后以 `--mode web` 启动真实实例（DeepSeek provider，`tmp/` 工作区，`AGENT_APPROVAL_TIMEOUT_SECS=900`），用 Playwright 实测。

| 场景 | 结果 |
|---|---|
| **G1 核心**：turn 等待审批时刷新 tab | ✅ 审批卡片完整恢复（工具名、参数、按钮、提示） |
| 刷新时刻 `/api/history` 内容 | ✅ 返回 `0 entries`，实证了 G1 根因——transcript 覆盖不到 |
| `/api/approvals` 快照 | ✅ 返回 id/tool/arguments/requested_at |
| 省略 `sticky` 字段的旧版请求 | ✅ 404（未知 id）而非 422，serde 默认值生效 |
| 四按钮布局（1280px） | ✅ 2×2，主按钮实心、sticky 次级 |
| 窄屏（340px） | ✅ 降级单列，4 行无溢出，sticky 保持 34px |
| 待决状态下切换语言 | ✅ 四个按钮 + 提示 + 占位符全部热更新 |
| 粘性批准 → 第二次同工具调用 | ✅ 审批队列始终为空，文件直接删除 |
| `/reset` → 再次调用 | ✅ 审批被重新 raise，记忆已清除 |
| 一次性拒绝 | ✅ 文件保留，transcript 记录拒绝 |
| 带理由拒绝 | ✅ 理由写入 transcript；**并暴露了 9.5 的缺陷** |
| 修正后带理由拒绝 | ✅ 模型接受并改道执行 |

所有临时文件与进程已清理，工作区已复原。

## 11. 实现记录：阶段三（持久化与恢复）

**问题**：审批超时后 turn 被当作「用户拒绝」继续跑下去。人没回答不等于人说了不，且这个虚构不是免费的——模型会针对一个没人提出的反对意见继续推理若干轮。

### 11.1 三态决策与部分完成

`ToolCallDecision` 加 `Suspend`。一轮里的调用是并发的，所以一轮不再是全有全无：`ToolRoundOutcome { completed, suspended }`。这是后面所有事情的基础——恢复时只重试被挂起的调用，兄弟调用的结果已在 transcript 里，重跑整轮会把它们的副作用做第二遍。

`SuspendedToolCall` 存**原始参数字符串**而非解析后的 `Value`。重新序列化会让工具拿到与被批准时文本上不同的载荷，也会抹掉「模型发了 `null`」和「模型的载荷没解析成功」的区别——而后者正是审批路径依赖的（不可解析的参数必须原样展示给人看，显示成 `null` 比不问还糟）。

### 11.2 恢复：决策只替换挂起

`ResumedDecision::Approved` **继续往下走 hook 链**，而不是跳过它。最初写成「有决策就跳过 before-hooks」，随即发现这会让人的一句 yes 顺带关掉工作区沙箱。人回答的是被问到的那个问题，不是「解除所有防护」。有测试钉住：在审批 hook 之后注册的守卫，对已批准的调用仍然有否决权。

没有决策的调用会**再次挂起**而非报错——回答一轮里的一部分是受支持的半步。

`resume` 不传决策也是合法的，而且是交互式前端的常态：未答复的调用重新走 hook 链，挂起它的东西会再问一次。对前端而言「恢复」和「重新询问」是同一个操作，用已有的审批卡片渲染即可，不需要为「回答一个已存储的审批」单独开一条路径。

### 11.3 占位 `ToolResult` 的过滤 hook：最终不需要

原计划用 `BeforeLlmCallback` 在恢复时滤掉占位项。重新检查控制流后发现不需要：`drive` 在轮次未完成时是 `return` 而非继续循环，两条回来的路（`resume` 答复、`run_continuing` 补记未答复）都会先把配对补齐。**带空洞的 transcript 没有机会被发到模型**。

不变量由控制流保证比由 hook 保证更可靠——hook 会被下一个入口忘记注册，而「没有别处能发请求」是结构性的。已写入 `drive` 的文档，并有测试钉住答复前后的配对状态。

占位补齐只留在 `AgentRunState::abandon()` 里，那正是唯一需要它的地方：放弃时要落盘一份可继续的 transcript。

### 11.4 additive 而非 breaking

第 6 节原计划让 `run_continuing` 改签名。实际做成了新增入口：

| 原有（行为不变） | 新增（可恢复） |
|---|---|
| `run_continuing` | `run_continuing_resumable` |
| `run_continuing_stream` | `run_continuing_stream_resumable` |
| — | `resume` / `resume_stream` |

原有入口遇到挂起时补记为「未答复」并继续，对一个要交出成品的入口来说这是诚实的。`WhenUnanswered::Suspend` 因此可以安全地做默认值：不能恢复的入口会自动降级到与 `Refuse` 相同的落点。

### 11.5 `ApprovalStore` 与 `SessionStore` 分开

不共用目录（第 7 节原倾向共用，实现时推翻）。两者生命周期相反：会话历史是要留的，挂起的 run 只活到有人回答为止。共用会导致 `/reset` 顺手丢掉一个待批准的操作，或被清理的会话留下孤儿 run。

`take` 而非 `get`：已存的挂起是一次性的，留着它等于邀请同一个 run 被恢复两次——对有副作用的待决调用就是执行两遍。解析失败的状态同样取走并丢弃，因为它无法被恢复，留着只会一直报告一个没人能回答的审批。

`pending()` 是只读的一瞥，且**只返回描述、永不返回状态**，这样展示路径不可能变成第二条恢复路径。

落盘约定完全照抄 `FileSessionStore`（编码文件名、临时文件 + 原子 rename、目录 0700）。两者在同一个安装里并排存在，不一致是给运维挖坑；且每条约定在这里同样是必要的——run id 来自调用方，可能长得像 `../../escaped`。

### 11.6 两端同源

`run_turn_stream` / `resume_turn_stream` / `record_suspension` / `discard_suspended_run` / `suspended_run_notice` 都在 `main.rs`，CLI 与 Web 共用。前端差异只在渲染：

- `/resume` `/discard` 进 `shared::commands` 表，两端同时获得命令与 `/` 菜单补全
- Web 的按钮走 `POST /api/chat` 发这两个命令，与终端输入完全同路——顺带让恢复的 turn 也广播出去，另一端能看到
- `GET /api/suspended` 之于挂起的 run，等同 `GET /api/approvals` 之于实时审批：晚到的 tab 否则什么都看不到（既没有答案，也刻意没写进 transcript）

**一个会话同时只能有一个挂起的 run**，这是真实约束不是偷懒：存储的 run 持有尚未进入会话存储的历史，此时开新 turn 会从残缺的 transcript 分叉，之后恢复就会把两段分歧的历史拼在一起。所以有挂起时新输入被拒绝，并指明两条出路。

`/reset` 与 `--fresh` 连带清除挂起的 run——run 持有的正是刚被清空的那段对话。

### 11.7 UI 上的区分

挂起卡片刻意比实时审批卡片安静：后者 pulse，因为**此刻**有东西被它阻塞着；前者静止，因为它会一直等下去。同一个 amber 色系表示相关，无动画表示不争夺注意力。

挂起卡片**不提供批准/拒绝按钮**，只列出待批准的调用。回答发生在 run 重启之后——那时提示会作为一张普通的、背后有活的 agent 的审批卡片回来。在已存储的 run 上提供决策按钮，等于暗示一条已经不存在的决策通道。

### 11.8 验证

`cargo make ci` 全绿：438 lib + 29 shared + 9 cli + 2 doctest，两侧 clippy 零警告。

新增测试覆盖：决策只替换挂起（含「后置守卫仍有否决权」）、部分答复、指纹不匹配拒绝恢复、状态 JSON round-trip、配对不变量的答复前后、`abandon` 补齐、流式恢复的五条路径、存储的路径穿越与键歧义。

**尚未做**：真实浏览器端的挂起/恢复回归（需 `trunk build` 后用 Playwright 实测「超时挂起 → 重启进程 → tab 看到卡片 → 恢复 → 审批卡片回来 → 批准 → 工具执行」整条链路）。

## 12. 浏览器与终端端验证记录（阶段三）

`trunk build` 后以真实实例实测（DeepSeek provider，`tmp/hitl` 工作区，`AGENT_APPROVAL_TIMEOUT_SECS=20` 触发挂起）。终端侧用 Python `pty` 驱动，并代答 reedline 的光标位置查询（裸 pty 无人应答会超时）。

### 12.1 通过的场景

| 场景 | 结果 |
|---|---|
| 超时挂起 | ✅ 卡片出现，`/api/approvals` 转空，`/api/history` 为 0 条（实证未污染历史） |
| 落盘状态结构 | ✅ 0700、指纹带真实模型名/digest/工具集、原始参数字符串保真 |
| 落盘 transcript | ✅ 4 事件，`delete_file` 的 ToolCall **故意无配对结果**；`current_step: 1`（挂起轮未计预算） |
| 跨进程重启 | ✅ 新 tab 经 `/api/suspended` 发现遗留挂起并渲染卡片 |
| 恢复 → 审批重新 raise | ✅ 同一 `tool_call_id` 重新进入 registry |
| 批准 → 工具执行 | ✅ 文件删除、挂起存储清空、历史写入且**完全配对** |
| `/discard` | ✅ 文件保留、占位结果写入、完全配对 |
| 放弃后继续对话 | ✅ 模型正确转述占位语义：「删除操作因未获得批准而没有执行」 |
| 终端启动提示 | ✅ `⏸ 本会话有一轮在等待审批时被暂停（待批准: delete_file）` |
| 终端 `/resume` → `y` | ✅ 控制台重新询问、批准、执行；**同时验证跨前端**（Web 产生挂起，另一进程的终端恢复） |
| `/resumey` 这类带后缀输入 | ✅ 不被当作命令，落到挂起提示（`parse` 的「不接受参数」规则） |
| `/` 菜单 | ✅ `/resume` 带译文出现在补全菜单 |

### 12.2 实测发现并修复的 5 个缺陷

实测的价值主要在这一节——这 5 个全部是单测覆盖不到的跨组件生命周期问题。

**① 挂起后旧审批卡片仍带活按钮**（已确认 `POST /api/approve/{id}` 返回 404）

turn 结束时 registry 条目被 `discard`，但卡片还在。更严重的是第二重后果：恢复后重新 raise 的提示**会被去重吃掉**——`push_approval` 以 `tool_call_id` 去重，而恢复保留原 id，于是新提示被当作旧卡片的重复丢弃，页面上只剩一张死卡片。

修复：`TurnSuspended` 到达时移除 `pending` 中各调用对应的审批卡片。同轮已答复的兄弟调用保留（那是历史记录）。

**② 挂起期间发新消息，卡片消失**

`clear_suspended()` 挂在 `UserMessage` 上，理由是「turn 开始意味着 run 被取走」。但被拒绝的新 turn、以及任何 in-chat 命令，都会广播 `UserMessage` 而并未取走任何东西。结果：提示叫用户 `/resume`，按钮却没了。

修复：新增 `ChatEvent::SuspendedRunCleared`，由服务端在 `take` 成功的那一刻广播（`resume_turn_stream` 与 `discard_suspended_run` 各一处）。UI 只在这一个事件上清卡片，不再从「turn 开始」反推。

**③ 指纹不匹配会销毁挂起的 run**（数据丢失级）

`ApprovalStore::take` 是不可逆点，而 `Agent::resume_stream` 的指纹校验发生在它之后且已按值拿走 state。实测：换模型重启后 `/resume`，文件确实没被删（安全目标达成），但 `/api/suspended` 转空——待批准的操作被静默丢弃。

修复：`resume_turn_stream` 在 `take` 之后、消费之前先校验，不匹配则**原样 `put` 回去**并报错。`turn_lock` 全程持有，take/put-back 对其他 turn 原子。校验和比对确认文件字节级未变。`Agent::resume`/`resume_stream` 的文档补上这个陷阱。

**④ 恢复失败后卡片按钮永久消失**

`acted` 在点击时置位，但 `POST /api/chat` 返回 202 不等于恢复成功——指纹不匹配是**异步**失败的。于是页面显示「该轮仍保留，可在恢复原配置后继续」，旁边却是一张没有按钮的卡片。

修复：`Error` 到达时复位 `acted`。成功的恢复已被 `SuspendedRunCleared` 移除卡片，所以复位是无害的 no-op。

**⑤ 并发恢复导致 composer 卡死**

两个视图同时 `/resume` 时，后到的 `take` 返回 `None`，`resume_turn_stream` 直接结束空流——没有任何终止事件，提交方 tab 的输入框永久禁用。

修复：这种情况也 yield 一个错误。前端需要「某个终止事件」来释放它在提交时禁用的输入框，空流不是。

### 12.3 已知限制 — 已全部处理，见第 14 节

四条里三条已修，一条重新归类为设计选择。

所有临时文件与进程已清理，工作区已复原。

## 13. 阶段四决策：`edit` 推迟

核对代码后重新评估了 4.4，**结论是推迟**。4.4 写于阶段三之前，其中两个前提已经变了，而实际影响面与当时的记载有出入。

### 13.1 前提变化

| 4.4 的说法 | 核实后的事实 |
|---|---|
| 加 `ProceedWith` 要改「4 个实现 + 调用点 + 全部测试」 | `ToolCallDecision` 已存在（阶段三加 `Suspend` 时引入）。四个 before-hook 只**返回**它、从不 `match` 它，全仓唯一的穷尽 `match` 在 `execute_tool_calls` 一处；测试用的是非穷尽的 `matches!`。**加变体几乎无成本** |
| `response`（自定义拒绝文案）待做 | 阶段二已完成 |
| — | 新增 suspend/resume，带来三条 4.4 没有的约束（13.4） |

所以**难点从来不在返回类型**，4.4 把成本估在了错误的地方。真正的难点是 transcript 一致性。

### 13.2 三个选项的实际代价

**选项 1（接受不一致）比 4.4 说的更糟**——不只是审计问题，是**正确性问题**。`build_messages` 会把 transcript 里的 `ToolCall.arguments` 原样发回给模型。人把 `delete_file{path:"/"}` 改成 `{path:"/tmp/x"}` 后，模型下一轮看到「自己发的原始参数 + 一个成功结果」，于是认定 `/` 已被删除，后续推理全建立在错误前提上。

（相关事实：参数不可解析时 transcript 已经记 `null` 而工具拿到原串，所以「transcript 与执行不一致」不是新问题。但那是降级失真，这是主动改写。）

**选项 3（追加修正项）是最差的，不是次差。** 4.4 只提到 `find_safe_start`，实际有三处遍历 `ToolCall` 且语义互相冲突：

| 位置 | 写法 | 同 id 时谁胜 |
|---|---|---|
| `find_safe_start` | `.or_insert(index)` | 最早 |
| `compaction::apply` | `.insert(...)` | 最后 |
| `search_compressor::extract_query` | `.find_map` | 第一个 |

而且 `build_messages` 会把两条都发出去，**同一个 `tool_call_id` 在一个请求里出现两次**，多数 provider 直接拒绝。

**选项 2（推迟 `record_tool_calls`）** 有个 4.4 没提到的好处和一个没提到的代价。好处：会消除未配对 `ToolCall` 的中间态，2.4 / `drive` 的控制流论证 / `abandon()` 的占位补齐可以一起简化。代价：并发期间只有 `&ExecutionContext`，要在决策后记录必须把 `execute_tool_calls` 拆成两阶段，于是同轮的安全调用要等待待审调用的决策——真实的延迟回归。（它也带来更强的保证：一轮对审批是原子的。）

### 13.3 倾向方案（4.4 没列的选项 4）

**提前记录 + 末尾回填**：`record_tool_calls` 照旧，决策阶段收集 `HashMap<call_id, edited_args>`，在 `execute_tool_calls` 末尾拿回可变借用（本来就要 `add_event`）时就地回填。

关键在于阶段三已经确立了「轮内中间态从不落盘、从不发给模型」的结构性保证（`drive` 在轮次未完成时 `return` 而非继续循环），回填前的窗口正好被它兜住——与未配对 `ToolCall` 是同一类中间态，同一条控制流负责。

规则：**只回填真正执行了的调用**。被短路的记原始（模型确实发了它），被挂起的不回填、改写值由 `SuspendedToolCall` 带走。

成本约为选项 2 的三分之一，无正确性问题，不动执行顺序与并发度。

### 13.4 阶段三引入的三条新约束（做 `edit` 时必须一并处理）

1. **`SuspendedToolCall.raw_arguments` 必须携带改写值。** 它现在读 `function_call.function.arguments`（原始）。若 hook A 改写、hook B 挂起，改写会在恢复时**静默丢失**。
2. **`edit` 与 sticky 互斥。**「总是允许 + 这些参数」没有意义——下次调用参数不同。需要在 `ApprovalDecision` 上加校验。
3. **未批准的改写不保留。** 人改了参数但没批准就离开，挂起应存原始值：未批准的改写不是决策。

另外 9.5 的归因约束在这里有个反直觉的结论：选项 2/4 下**静默替换反而最自洽**，因为没有额外文本注入，模型看到的就是一个参数不同的调用。是否额外告知模型「用户改了参数」是独立决策；若告知，那段说明必须带来源标注。

### 13.5 为什么推迟

现有的「带理由拒绝」已覆盖大部分场景：`n: 路径应该是 /tmp/x 而不是 /` 会让模型自己改参数重试。路径长一步，但不需要任何新机制，也不引入上述任何问题。

`edit` 是四种响应模式里使用频率最低的一种，在出现「模型反复改不对、只能人来填」的真实场景之前，它的成本买不到相应的价值。结论与影响面已同步到 `README.md` 的 Roadmap。

## 14. 处理 12.3 的四条已知限制

逐条核实后：三条可修且已修，一条不该修。

### 14.1 穿插的 assistant 文本丢失（已修）

12.3 把它记为「既有行为，与本次改动无关」——**这个归类淡化了它**。它不是流式路径独有的小瑕疵，而是**三条路径都在发生的数据丢失，每一轮工具调用都会触发**。

模型常常在同一条消息里既说话又调用工具（"我先确认文件是否存在" + `list_files`）。`drive`、`drive_stream`、`structured` 三处都只取 `tool_calls` 而丢弃 `message.content` / `assistant_text`。后果：

- 模型下一轮看到的是一个没有任何解释的工具调用，它自己说过的推理不见了
- 人回看历史同样看不到

修复是 `record_assistant_text`，在 `record_tool_calls` **之前**记录。顺序是关键：`build_messages` 为这段文字开一条 assistant 消息，随后的 `ToolCall` 通过 `messages.last_mut()` 追加到**同一条**消息上——正好还原 provider 实际发出的那条消息（content 与 tool_calls 并存）。所以这个修复不仅补回了内容，还让回放比之前更忠实。

空白文本不记录：绝大多数轮次是裸调用，空 assistant 消息只会让每个消费 transcript 的地方多一次跳过。

实测：`I'll list the current directory and read a.txt for you.` 与 `我先看一下当前目录，确认 notes.txt 是否存在。` 都进了 transcript。

### 14.2 流式路径 `usage` 恒为 0（已修）

`stream_options.include_usage = true` + 读 `chunk.usage`。有一个坑：携带 usage 的正是 `choices` 为空的那个尾部 chunk，而原循环 `let Some(choice) = chunk.choices.first() else { continue }` 会跳过它。所以读取必须放在这个 guard **之前**。

`stream_options` 设在 `complete_stream` 而非共享的 `request_builder`：它只在 `stream: true` 时合法，非流式请求带上它会被 API 直接拒绝。不支持该字段的 provider 会忽略它、不发 usage chunk——与此前行为一致，不会让能用的 provider 变得不能用。

实测：`TokenUsage { prompt_tokens: 4643, completion_tokens: 184, total_tokens: 4827 }`，跨 3 轮累加正确（此前恒为 0）。

### 14.3 重启后卡片上方没有上下文（已修）

12.3 说「决策所需信息是全的」——工具名与参数确实都在，但这话只在"审批"这个尺度上成立。人看到的是一张悬在空白页面上的卡片，不知道自己当初问了什么才导致它。

数据一直都在（`AgentRunState.context.events`），只是没往外送。改动：

- `ApprovalStore::pending()` → `peek()`，返回 `SuspendedRunView { pending, events }`。仍然只返回**视图**而非状态本身，`take()` 的一次性语义不受影响：拿着一份 transcript 无法恢复任何东西。
- `GET /api/suspended` 从 `Vec<PendingApprovalView>` 变为 `Option<SuspendedRunView>`，带上 transcript
- 前端把 `load_history` 的渲染循环抽成 `push_entries`，两处共用——挂起的 transcript 与已完成的形状完全一样（所以它就用 `HistoryEntry` 传输），另写一份 match 迟早会在"多长算太长、要不要默认折叠"这类细节上跑偏

实测：重启后页面显示「当初的提问 → 工具调用卡 → 已暂停卡片」，`GET /api/history` 仍为 `[]`（挂起依然不污染历史），恢复后无重复渲染。

终端侧未做对应改动：启动横幅已列出待批准的工具名，把整段 transcript 打到横幅里只会更吵。

### 14.4 补全菜单里 Enter 选中而非提交（不修，归类错误）

这条不是缺陷，是 README 已写明的设计选择：

> 选中后只填入而不发送，是为了让选错的命令还能改，终端与网页在这点上行为一致。

改成"单候选时 Enter 直接提交"会让行为依赖候选数量——打 `/re` 有两个候选（`/reset` / `/resume`）时是选中，打 `/discard` 只有一个时却直接执行了。对一个能清空历史的命令集来说，这种不一致比多按一次 Enter 危险得多。

12.3 把它和三个真缺陷列在一起是归类错误，已改正。

## 15. 进程非正常退出时的挂起保全

**问题**（用户报告）：CLI 弹出审批提示时终止程序，重开同一 session 提示就没了。

实测复现后发现比报告的更严重：**不只是提示没了，整轮对话连同用户的提问一起消失**——审批目录与会话目录都不存在。

### 15.1 根因

挂起状态只在**超时那一刻**才被构造。超时默认 300 秒，而用户在此之前就终止了进程，于是从未走到构造挂起的那行代码。

### 15.2 两类退出，两种机制

关键分界：**turn 是否还活着**。

| 退出方式 | turn 状态 | 可用机制 |
|---|---|---|
| `Ctrl-C` / `SIGHUP`（关窗口）/ `SIGTERM`（`kill`） | 还活着 | 信号处理 → 优雅挂起 |
| `SIGKILL` / panic / 断电 | 已消失 | **只能靠写前日志** |

第二类无解于信号处理：`ExecutionContext` 就活在那个 async 任务自己的栈上，任务没了它就没了，**事后运行的任何代码都救不回来**。这一点值得单独记下来，因为它决定了不能只做信号处理就收工。

### 15.3 机制一：信号 → 优雅挂起

`ApprovalRegistry::cancel_all()` 丢弃所有决策发送端。等待侧读到发送端关闭，本来就理解为「无人应答」——与超时**完全同一条路径**。所以不需要新的持久化逻辑，只是把超时会做的事提前触发。

三个信号都处理，而不只是 `Ctrl-C`：`SIGHUP` 是关闭终端窗口发的，`SIGTERM` 是 `kill` 和系统关机发的，三者留下的现场完全一样。退出码 `128+signum`。连按两次信号跳过等待直接退出——按第二次的人就是想立刻走。

副产品：`Ctrl-C` 成了「这个审批我待会儿再说」的快捷方式，不用干等满 300 秒。

**实现中发现的阻塞点**：`drive_terminal_turn` 在等「按回车返回输入」，而 turn 锁要等它才释放，导致信号处理器等锁超时、进程 5 秒后才退出。修法是把 `stream`（持有锁守卫）在等待键盘**之前**显式 drop——锁的释放时机应当由「turn 是否结束」决定，而不是「这个终端是否收拾干净」。

### 15.4 机制二：写前日志（`RunCheckpoint`）

每个工具轮次**开始之前**把状态写盘，轮次结束后清除。代价是每轮一次小文件写，相对于产生这一轮的模型调用可以忽略。

恢复语义是这里唯一需要想清楚的地方：

一轮里的调用是**并发**的。写盘那一刻它们都没跑，但进程消失时可能已经跑了一部分——而且**无从得知是哪些**，因为结果要等整轮 settle 才进 transcript。

于是两个选项：

| 选项 | 后果 |
|---|---|
| 恢复时重跑整轮 | 已生效的副作用做第二遍。对 `delete_file`、`send_email` 这类是灾难 |
| 收尾为「是否生效未知」 | 丢失部分工作，但不重复；诚实 |

**选了后者**：重复副作用比报告未知更糟，尤其在一个整个特性都是为了守住危险操作的系统里。

为此给 `AgentRunState` 加了 `StopReason`：

- `AwaitingDecision` —— 这些调用**确定没跑**（`Suspend` 在链中的位置保证了这点），可以安全重问
- `RoundInFlight` —— **未知**，只能收尾

`abandon()` 据此选择措辞。这不是文字游戏：告诉模型「未获批准」（意味着确定没执行）而实际可能执行了，会让它自信地重试一个已经发生的操作。实测模型收到「未知」措辞后会主动 `list_files` 去核实，而不是臆断。

`serde(default)` 让旧状态按 `AwaitingDecision` 载入——旧状态确实都是挂起，没有别的来源。

### 15.5 恢复路径上的一个坑

`resume_turn_stream` 里的 `take` 会**移除**存储的状态。如果在重新询问审批时崩溃，状态就彻底没了——比修复前还糟。所以恢复的那一轮也必须 checkpoint。

### 15.6 实测

| 场景 | 结果 |
|---|---|
| `Ctrl-C` 于审批提示 | ✅ 退出码 130，挂起落盘，文件未删 |
| 重开同一 session | ✅ 启动提示 + `/resume` + `y` → 工具执行，完整闭环 |
| `SIGKILL`（真 crash） | ✅ checkpoint 在盘上，`reason: RoundInFlight`，**用户的提问保住了** |
| 崩溃后重启 | ✅ 自动收尾并提示，transcript 完全配对 |
| 收尾后继续对话 | ✅ 模型主动核实后答「没成功——`notes.txt` 仍在」 |

### 15.7 仍然覆盖不到的

- **写盘本身被打断**：临时文件 + 原子 rename 保证要么旧要么新，不会半截。但 rename 之后、轮次结束之前断电，会留下一个 `RoundInFlight` 记录——收尾为「未知」，是正确的保守答案。
- **轮次完成与清除 checkpoint 之间崩溃**：会把已完成的调用标成「未知」。「未知」永远不是错的，只是偏保守。
- ~~**`--mode web` 下的信号**：Web 服务器任务与终端共享同一个处理器，行为一致。~~ **这句话是错的**，见 17。

## 16. `/discard` 之后 web 只剩一张孤立的 tool call

**问题**（用户报告）：resume 后放弃，web 历史里只看到 tool 的 call，没有 give up 的标识。

### 16.1 两个独立的缺陷

实测先分清了是持久化还是展示的问题：`GET /api/history` 在 discard 后**确实**包含占位 `tool_result`。所以持久化是对的，问题在别处——而且是两个。

**① 占位结果从未广播。** `discard_suspended_run` 只发了 `SuspendedRunCleared`（撤掉暂停卡片）和一条 `SystemNotice`。而正在看的那个 tab，时间线上早就有这张 tool call 卡（turn 第一次跑的时候来的），此后**永远等不到它的结果**——广播不重放，所以除非刷新页面，它就一直停在"像还在跑"的样子。这正是用户描述的现象。

修复：把 `abandon()` 要写进历史的那批占位结果，同时以 `ChatEvent::ToolCallsFinished` 广播出去。为此把构造占位的逻辑从 `abandon()` 里拆成 `unanswered_results()`——**文案只能有一处来源**，两份拷贝迟早会偏（`StopReason` 的措辞已经证明过一次）。有测试钉住「广播的和落盘的是同一批」。

**② `/discard` 记的是超时的措辞。** 这个更深，是顺着用户「没有 give up 标识」这句话查出来的：

`/discard` 是**用户明确的决定**，但它走 `abandon()` 时用的是「waiting for approval that never arrived」——那是**没人应答**的措辞。两件事完全不同，而 transcript 当时分不出来。

这违反了 9.5 定下的规则：**注入 transcript 的人类意图必须可归因**。拒绝会记成 `User denied execution of X: 理由`，前缀把它标记为运行者本人的决定；而放弃却记成了一句关于环境的陈述，读起来像"系统超时了"。

于是加了 `GiveUp { Unanswered, ByUser }`，与 `StopReason` 组合：

| `StopReason` | `GiveUp` | 记录 |
|---|---|---|
| `RoundInFlight` | 任意 | 「被中断，是否生效未知」 |
| `AwaitingDecision` | `ByUser` | 「**User gave up on X** instead of deciding, so it was not run. Do not retry it without being asked to.」 |
| `AwaitingDecision` | `Unanswered` | 「waiting for approval that never arrived」 |

`RoundInFlight` 优先于谁放弃：谁按了什么，不改变这个调用到底有没有生效，而后者是更重要的事实。

这同时也是给模型的正确信息：「没人应答」意味着可以等个更好的时机，「用户放弃了」意味着**别做这件事**。把后者报成前者，会丢掉唯一真正给出的指令。

### 16.2 为什么不是加一个新的 UI 标识

考虑过在 web 上把「已放弃」渲染成一张专门的卡片。没有这么做：transcript 同时是模型的输入和历史的渲染源，而 `tool_result` 已经是「这个调用怎么结束的」的既定位置——拒绝就是这么记的。再加一类只有 UI 认得的条目，会让同一件事有两种表示，且模型看不到那一种。

修好归因之后，历史里那张结果卡的文字本身就是 give up 标识，且在**刷新后依然在**——而 `SystemNotice` 不进 transcript，刷新即消失，本来就不能承担这个职责。

### 16.3 实测

| 场景 | 结果 |
|---|---|
| 放弃前的时间线 | ✅ 复现原始缺陷：只有 `Call delete_file`，无结果 |
| 点 Give up（**不刷新**） | ✅ 结果卡当场出现：`User gave up on delete_file instead of deciding, so it was not run.` |
| 刷新后 | ✅ 标识依然在（`SystemNotice` 如预期消失），`/api/suspended` 为 `null` |
| 模型的理解 | ✅ 答「还在——因为刚才的删除操作没有实际执行」，且**没有重试删除** |

一个测试环境的教训：第一次实测拿到的是一堆莫名其妙的重复条目，查日志才发现 `Address already in use` ——之前某条 `pkill` 被取消，旧进程（旧二进制 + 累积状态）仍占着端口，实际验证的是旧代码。改用新端口起干净实例后结论才可信。**日志里的绑定失败必须先看**，否则会对着错误的进程解释现象。

## 17. `--mode web` 根本没装信号处理器

**问题**（用户报告）：纯 web 场景下收到信号时没有终端可打印提示。

查下去发现 15.7 那句「Web 服务器任务与终端共享同一个处理器，行为一致」**是错的**，而缺一条提示是三个缺陷里最轻的那个。

### 17.1 一行 `return` 造成的三重失效

`main` 里这三件事都排在终端 REPL 之前、但在 `if run_web { ... return }` **之后**：

```
if run_web { ...; if !run_terminal { handle.await?; return Ok(()); } }   // ← --mode web 从这里返回
...
recover_interrupted_round(...)   // 从未执行
spawn_interrupt_handler(...)     // 从未执行
announce_suspended_run(...)      // 从未执行
```

于是纯 web 模式下：

1. **信号处理器从未安装**。`SIGTERM` 走默认处置直接杀掉进程，那一轮来不及停泊。原本应该得到一个干净、可 `/resume` 的 `AwaitingDecision` 挂起，实际只剩 checkpoint 留下的 `RoundInFlight`——即「结果未知、不可恢复」。**功能降级，而非仅少一条提示。**
2. **崩溃收尾从未运行**。那个 `RoundInFlight` 记录**每次重启都活下来**，因为没人收尾。
3. **`/resume` 会重放副作用**。前两条叠加后，web 的 `/resume` 照单全收这个不可恢复状态、重新发起审批——批准后 `delete_file` **真的又执行了一遍**。而这正是 `RoundInFlight` 存在的唯一目的所要禁止的（它自己的文档写着「must not be resumed by re-running」）。

实测逐条确认过，包括第 3 条里文件被第二次删掉。

### 17.2 修复

**把恢复和信号处理器移到 `if run_web` 之前。** 它们属于会话，不属于某个前端。恢复必须在服务器开始接请求之前完成，否则浏览器可能在那个窗口里摸到还没收尾的状态。

`announce_suspended_run` 留在原处：那是终端专属的一次性打印，而浏览器有自己的常驻入口（加载时 `GET /api/suspended` 喂出来的暂停卡片），并不缺这一条。

**给 `/resume` 加防护，并把不变量下沉到 `AgentRunState::unresumable()`。** 修好启动顺序后，`/resume` 实际已经摸不到 `RoundInFlight` 了——但「绝不重跑这些调用」是正确性不变量，不该靠两行启动代码的先后来保障。所以 `Agent::resume` / `resume_stream` 自己也挡一道（与 `RunFingerprint::mismatch` 完全同构：agent 做兜底，调用方先查以保证被拒绝时不销毁状态），换个前端不必重新发现这个坑。

CLI 侧调 `unresumable()` 而不是自己比对 `reason`——两处各写一遍条件迟早会偏。

被拒时**放回**而不是静默收尾：`/resume` 要求的是继续，悄悄改写历史并不比拒绝更不意外。出路是 `/discard`，它记的仍是「结果未知」的措辞。

### 17.3 那条提示最后怎么处理的

没有为浏览器新加推送。理由和 16.2 一致：收尾会把「被中断、是否生效未知」写进 **history**，浏览器加载时那张 Failed 卡就是标识，且**刷新后依然在**。而收尾发生在启动时，广播通道那会儿一个订阅者都没有，推了也没人收。

信号处理器的提示仍走 `term_write`——纯 web 模式下它进日志，这本身就是有用的（实测日志里能看到「⏸ 收到终止信号」）。没有在退出前广播：进程正在消失，SSE 能否 flush 是赢不了的竞态，半条消息比没有更糟。真正的答案是那份落盘记录——下次启动时暂停卡片就在那儿。

### 17.4 实测

| 场景 | 修复前 | 修复后 |
|---|---|---|
| `--mode web` 收 `SIGTERM`（审批待决） | `RoundInFlight`，不可恢复 | ✅ `AwaitingDecision`，文件未动 |
| 重启后 `/resume` → 批准 | — | ✅ 重新询问 → 执行 → 完整闭环 |
| `--mode web` 被 `SIGKILL` 后重启 | 状态永久残留 | ✅ 自动收尾，`/api/suspended` 为 `null`，历史留「未知」记录 |
| 对 `RoundInFlight` 执行 `/resume` | **副作用被重放（文件真被删）** | ✅ 拒绝并指向 `/discard`，状态放回，文件未动 |
| 随后 `/discard` | — | ✅ 状态清空，记录「结果未知」 |
| 日志提示 | 无 | ✅ 「⏸ 收到终止信号」/「⚠️ 上次运行被强制中断」 |

## 18. Give up 之后卡在「agent 正在思考」

**问题**（用户报告，带截图）：纯 web 模式下 give up 后出现一个一直转的 agent loading 状态。

截图里有个关键细节：**输入框是可用的**，而转圈还在。活着的一轮期间输入框是禁用的，所以这两件事同时成立说明——从这个 tab 的角度看没有轮次在跑，但 `turn_active` 被置起后再没被放下。顺着这条线查出两个缺陷，它们是**复合**的。

### 18.1 `/discard` 排在它要放弃的那一轮后面

`discard_suspended_run` 第一件事就是 `turn_lock.lock().await`。而停在审批上的那一轮**整个等待期间都持着这把锁**。于是 `/discard` 排队等在了**它正要放弃的那一轮**后面，直到 `approval_timeout`（默认 **300 秒**）走完。

前端的转圈正是这段时间的表现：命令的回显是一个普通 `UserMessage`，和开启一轮的那种无从区分，所以指示器已经亮起；而负责放下它的 `SystemNotice` 要等这个函数返回才发。

修复是 `ApprovalRegistry::cancel_all()`，在取锁**之前**调用——和信号处理器用的是同一招（它的文档已经写明这个用法）：丢掉决策发送端，等待侧就读成「没人应答」，于是那一轮走正常路径挂起并落盘，只是比超时更早。锁随之释放，而要放弃的那份挂起已经在盘上等着被取——每个驱动器都在释放锁之前先落盘。

没有待决审批时它是空操作，也就是最常见的情形：本来就已挂起、没有轮次在跑。

实测：**300 秒 → 0 秒**。

### 18.2 活着轮次的 checkpoint 被当成了挂起

但 18.1 还留了个问题：输入框在活着的一轮期间是禁用的，**那用户是怎么打出 `/discard` 的？**

查下去发现 `/api/suspended` 会把 `RoundInFlight` 记录当成挂起上报。而那是 checkpoint——写在轮次**开始之前**的（启动时已收尾掉上个进程留下的，所以运行期出现的只可能属于**正在跑**的轮次）。

后果：任何在轮次进行中加载的页面，会**同时**看到活着的 Approve/Deny 卡，和一张针对同一个调用的「⏸ Paused, waiting for approval」卡，带着 Carry on / Give up 按钮。两个按钮都兑现不了自己的承诺——`/resume` 对这种状态会被拒（见 17.2），`Give up` 则去抢那一轮正持着的锁，即 18.1。

这就是那条路径：刷新页面 → 拿到假的暂停卡 → 点 Give up → 转圈 300 秒。

判定下沉到 `SuspendedRunView::awaits_decision()`，两个展示路径（web 的 `/api/suspended`、终端的 `suspended_run_notice`）共用。放在 view 上而不是各自比对 `reason`，是因为这个错误从外部看不出来，且终端侧同样中招：它会在一轮正常运行时拦住下一条消息，还指向一个本身会被拒绝的 `/resume`。

### 18.3 实测

| 场景 | 修复前 | 修复后 |
|---|---|---|
| 审批活着时 `/discard` | **阻塞 300s**（转圈不停） | ✅ 0s 返回，记录仍是「用户放弃」，文件未动 |
| 审批活着时加载页面 | 假暂停卡 + Carry on/Give up | ✅ 只有 Approve/Deny |
| 点假卡上的 Give up | 转圈 300s | ✅ <500ms，通知到达，转圈消失 |
| 真挂起后加载页面 | — | ✅ 仍正常显示暂停卡（未过度过滤） |

一个测试手法上的教训：中途我直接给 Leptos 受控 `textarea` 赋 `.value` 再按 Enter，以为提交了 `/discard`，结果信号没更新、什么都没提交，我却对着一个**正常**的转圈（那一轮确实在跑）找了半天原因。回头看时间线里最后那行是 `AGENT` 指示器本身、而不是 `/discard` 回显，才发现搞错了。**断言之前先确认自己的输入真的进去了。**
