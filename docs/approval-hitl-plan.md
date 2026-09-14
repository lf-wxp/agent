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

### 阶段四：可选增强

12. ⬜ `edit` 模式（需先决策 4.4 表格里的三个选项之一，含 `ToolCallDecision` 的 breaking change）
13. ⬜ MCP Elicitation 接入评估

### 收尾

14. ⬜ 更新 `README.md`（审批相关配置项与行为说明）与本文档（补实现记录小节）

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
