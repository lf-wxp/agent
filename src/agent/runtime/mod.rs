//! [`Agent`]: drives a model through a tool-calling loop while recording every step into
//! an [`ExecutionContext`].
//!
//! This differs from [`crate::llm::tool_loop::run`], which only returns the last choice:
//! here the whole transcript (user input, tool calls, tool results, final answer) is kept
//! as [`Event`]s, so a caller can inspect or persist what happened at each step. The model
//! call itself is not reimplemented, and neither is the round-budget behavior: both go
//! through the same shared client, request builder and [`disable_tools`] as `tool_loop`,
//! so a fix made there (retries, concurrency limits, degradation) does not have to be
//! ported by hand.
//!
//! [`Agent::run_stream`] / [`Agent::run_continuing_stream`] are the streaming counterparts
//! of [`Agent::run`] / [`Agent::run_continuing`]: same loop, round budget and event
//! recording, but assistant text is forwarded as it is produced (see
//! [`crate::llm::stream`] for the same idea without event recording).
//!
//! Multi-turn conversations are supported via [`Agent::run_continuing`], which seeds a
//! call with a prior turn's events instead of starting from an empty transcript. `Agent`
//! itself stays a stateless function of "prior events + new input": it does not own a
//! session/storage concept, that lives one layer up (see [`crate::agent::session`] for one
//! way a caller can persist history across turns). It does, however, cap how much of
//! that history it will actually send per call: [`Agent::new`] registers a
//! [`crate::callback::context_optimizer::ContextOptimizer`] by default, since an
//! unbounded session could otherwise grow past the model's context window.
//!
//! Structured output ([`Agent::run_structured`] / [`Agent::run_structured_raw`]) is
//! implemented in the [`structured`] submodule: it is a large enough sub-problem (model
//! capability dispatch, two different termination mechanisms, schema-as-type vs
//! schema-as-data) to deserve its own file, but stays part of this same `impl Agent` —
//! Rust allows an inherent impl to be split across modules, and both halves need the
//! same private fields and the same event-recording/request-building helpers defined
//! here.

mod structured;

use std::{collections::HashMap, sync::Arc};

use async_openai::types::chat::{
  ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls,
  ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
  ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestToolMessageArgs,
  ChatCompletionRequestUserMessageArgs, ChatCompletionResponseStream,
  CreateChatCompletionRequestArgs, CreateChatCompletionResponse, FunctionCall,
};
use async_stream::stream;
use futures::{Stream, StreamExt, future::join_all};
use serde_json::Value;

use crate::{
  agent::{
    callback::{
      AfterToolCallback, BeforeLlmCallback, BeforeToolCallback, ToolCallDecision, ToolCallView,
    },
    llm_request::LlmRequest,
  },
  callback::context_optimizer::ContextOptimizer,
  config,
  llm::{
    client::{DEFAULT_MAX_TOKENS, first_choice, request_builder},
    provider::Provider,
    retry::{is_transient, with_retry},
    tool_calls::ToolCallAccumulator,
    tool_loop::disable_tools,
  },
  tools::ToolRegistry,
  util::truncate_chars,
};

use super::{
  context::{Conversation, ExecutionContext},
  event::{ContentItem, Event, ToolResultStatus},
  fingerprint::RunFingerprint,
};

pub use structured::StructuredAgentResult;

/// Chars of tool arguments/results kept in debug logs, to avoid dumping large or
/// sensitive payloads (e.g. fetched web content, credentials echoed by a misbehaving
/// MCP server) into the log stream in full.
const TOOL_LOG_PREVIEW_CHARS: usize = 500;

/// Outcome of [`Agent::run`].
#[derive(Debug)]
pub struct AgentResult {
  pub output: String,
  pub context: ExecutionContext,
  /// `true` when the round budget ran out and this answer was produced with tools
  /// disabled, i.e. built from partial information. Mirrors
  /// [`crate::llm::tool_loop::Completion::budget_exhausted`].
  pub budget_exhausted: bool,
}

/// One increment produced by [`Agent::run_stream`] / [`Agent::run_continuing_stream`].
///
/// A caller interested only in the text (e.g. forwarding straight to an SSE response) can
/// match on [`Self::Token`] and ignore everything else until the stream ends; a caller that
/// needs to persist the turn (e.g. [`crate::agent::session::SessionStore`]) reads
/// [`Self::Done`]'s `context` — the same fields as [`AgentResult`], just delivered as the
/// stream's last item instead of a return value. A caller that wants to show the tool
/// calls a round makes as they happen (e.g. a web UI's process timeline) reads
/// [`Self::ToolCallsStarted`]/[`Self::ToolCallsFinished`] instead of waiting for `Done`
/// and reconstructing them from its `context.events`.
#[derive(Debug)]
pub enum AgentStreamEvent {
  /// A chunk of assistant text, forwarded as soon as the model emits it.
  Token(String),
  /// The model requested these tool calls and they are about to run, forwarded just
  /// before [`Agent::execute_tool_calls`] starts them. Every item is a
  /// [`ContentItem::ToolCall`] — one per call in this round, in the order the model
  /// requested them, the same items [`Agent::record_tool_calls`] adds to the transcript.
  ///
  /// This is round-level, not per-individual-call: calls in the same round run
  /// concurrently (see [`Agent::execute_tool_calls`]'s docs), so there is no meaningful
  /// per-call "started" instant to report separately from the round's.
  ToolCallsStarted(Vec<ContentItem>),
  /// The tool calls from the immediately preceding [`Self::ToolCallsStarted`] have all
  /// finished, forwarded right after [`Agent::execute_tool_calls`] returns. Every item is
  /// a [`ContentItem::ToolResult`], in the same order as the [`Self::ToolCallsStarted`]
  /// it answers.
  ToolCallsFinished(Vec<ContentItem>),
  /// The run finished. Always the last item; nothing follows it.
  Done {
    output: String,
    context: ExecutionContext,
    /// See [`AgentResult::budget_exhausted`].
    budget_exhausted: bool,
  },
}

/// One tool call that was stopped before it ran, awaiting a decision that could not be
/// made in the moment (see [`ToolCallDecision::Suspend`]).
///
/// Serializable, and holding the raw argument string rather than a parsed
/// [`Value`]: this is what a resumed run re-executes from, and the raw form is both what
/// the tool is actually handed and what a human was shown when asked to approve it.
/// Re-serializing a parsed value would hand the tool a textually different payload than
/// the one that was approved, and would erase the difference between "the model sent
/// `null`" and "the model's payload did not parse" — a distinction the approval path
/// relies on (see [`crate::callback::dual_approval`]).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SuspendedToolCall {
  pub tool_call_id: String,
  pub name: String,
  pub raw_arguments: String,
}

/// What one round of tool execution produced.
///
/// Split into two lists because a round is no longer all-or-nothing: calls in a round run
/// concurrently, so a decision to suspend one of them arrives while its siblings are
/// still in flight — and those siblings have already started doing whatever they do.
/// Discarding their results to report the suspension would mean re-running them later,
/// duplicating every side effect they had; reporting only the results and dropping the
/// suspension would silently execute a call nobody approved.
#[derive(Debug, Default)]
pub struct ToolRoundOutcome {
  /// [`ContentItem::ToolResult`] for every call that reached a conclusion, in call order.
  pub completed: Vec<ContentItem>,
  /// Calls that were suspended, in call order. Empty in the ordinary case.
  pub suspended: Vec<SuspendedToolCall>,
}

/// A human's answer to a call that was suspended, supplied to [`Agent::resume`].
///
/// Stands in for the decision a [`BeforeToolCallback`] could not make at the time — and
/// *only* for that. It replaces a [`ToolCallDecision::Suspend`], never a denial and never
/// a `Proceed`, so a hook that rules on the call for some other reason still rules on it:
/// a workspace sandbox does not stop applying because a human approved the call.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ResumedDecision {
  /// Carry on down the hook chain and, if nothing else objects, run the tool.
  Approved,
  /// Record `reason` in place of the call. Owns its message rather than deriving one,
  /// because the refusal was decided elsewhere — by a front-end that already knows how
  /// to word it (see [`crate::callback::dual_approval`]).
  Refused(String),
}

/// A run that stopped to ask a human, in a form that outlives the process it started in.
///
/// # The context inside is deliberately mid-turn
///
/// It holds the [`ContentItem::ToolCall`] for every suspended call with no matching
/// [`ContentItem::ToolResult`] — a shape most providers reject outright. That is the
/// honest representation (no decision has been made, so no result exists), but it means
/// this context must not be sent to a model or filed as conversation history as-is. The
/// two ways out both repair it: [`Agent::resume`] answers the calls, and
/// [`Self::abandon`] closes them out as unanswered.
///
/// Persisting it is fine, and is the point — the invariant is about what is *sent*, not
/// about what is stored.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct AgentRunState {
  /// What this run was produced by; checked before it is resumed. See [`RunFingerprint`].
  pub fingerprint: RunFingerprint,
  /// Calls waiting on a decision, in the order the model requested them.
  pub suspended: Vec<SuspendedToolCall>,
  /// See [`AgentResult::budget_exhausted`]. Carried across the suspension so a run
  /// resumed after its budget ran out still reports having been cut short.
  pub budget_exhausted: bool,
  /// Not `pub`: the two supported ways to get an owned context out of here repair it
  /// first ([`Agent::resume`], [`Self::abandon`]), and handing it over raw would make it
  /// easy to file a mid-turn transcript as history. `pub(crate)` rather than private so
  /// [`crate::agent::approval_store`] can construct one in its own tests.
  pub(crate) context: ExecutionContext,
}

impl AgentRunState {
  /// Give up on the pending decisions and return the transcript, repaired.
  ///
  /// For a caller that has decided not to wait any longer — the human is gone, the
  /// approval expired, the operator cancelled it. Every suspended call gets a result
  /// saying it was never answered, which is both true and what makes the transcript
  /// sendable again, so the conversation can carry on from here in a later turn instead
  /// of being stuck.
  pub fn abandon(mut self) -> ExecutionContext {
    let items: Vec<ContentItem> = self
      .suspended
      .iter()
      .map(|call| ContentItem::ToolResult {
        tool_call_id: call.tool_call_id.clone(),
        name: call.name.clone(),
        status: ToolResultStatus::Error,
        content: format!(
          "Tool execution stopped: {} was waiting for approval that never arrived.",
          call.name
        ),
      })
      .collect();
    self
      .context
      .add_event(Event::new(self.context.execution_id.clone(), "tool", items));
    self.context
  }

  /// The transcript so far, for a caller that wants to show what has happened while the
  /// decision is outstanding.
  ///
  /// Read-only on purpose: see the type's docs for why this context is not something to
  /// hand to a model or a session store directly.
  pub fn context(&self) -> &ExecutionContext {
    &self.context
  }
}

/// How a resumable run ended: with an answer, or with a question.
#[derive(Debug)]
pub enum AgentOutcome {
  Done(AgentResult),
  /// The run stopped before a tool call it could not decide. Answer the calls in
  /// [`AgentRunState::suspended`] and hand the state to [`Agent::resume`], or give up on
  /// them with [`AgentRunState::abandon`].
  Suspended(AgentRunState),
}

/// Rebuild the API-shaped calls for a set of suspended ones, so they can go back through
/// [`Agent::execute_tool_calls`] unchanged.
///
/// Lossless because [`SuspendedToolCall`] keeps the raw argument string: the tool is
/// handed textually what the model sent and what a human was shown, not a re-serialized
/// approximation of it.
fn rebuild_tool_calls(suspended: &[SuspendedToolCall]) -> Vec<ChatCompletionMessageToolCalls> {
  suspended
    .iter()
    .map(|call| {
      ChatCompletionMessageToolCalls::Function(ChatCompletionMessageToolCall {
        id: call.tool_call_id.clone(),
        function: FunctionCall {
          name: call.name.clone(),
          arguments: call.raw_arguments.clone(),
        },
      })
    })
    .collect()
}

/// Drives a model to a final answer, executing tool calls as it requests them.
///
/// A value rather than a free function: `provider`, `model`, `instructions` and the tool
/// set are fixed for the lifetime of the agent, while each [`Self::run`] call gets its
/// own fresh [`ExecutionContext`] so concurrent runs never share state.
pub struct Agent {
  provider: Provider,
  model: String,
  instructions: Option<String>,
  toolbox: Arc<ToolRegistry>,
  max_steps: u32,
  /// Hooks run before each tool call, in registration order; see
  /// [`Self::with_before_tool_callback`].
  before_tool_callbacks: Vec<Arc<dyn BeforeToolCallback>>,
  /// Hooks run after each tool call, in registration order; see
  /// [`Self::with_after_tool_callback`].
  after_tool_callbacks: Vec<Arc<dyn AfterToolCallback>>,
  /// Hooks run before each LLM request, in registration order; see
  /// [`Self::with_before_llm_callback`]. [`Self::new`] seeds this with a
  /// [`ContextOptimizer`] — the history token budget belongs to that callback, not
  /// to `Agent`.
  before_llm_callbacks: Vec<Arc<dyn BeforeLlmCallback>>,
}

impl Agent {
  /// Rounds default to [`config::max_tool_rounds`], the same budget `tool_loop` uses, so
  /// the two do not silently drift apart; override with [`Self::with_max_steps`] when an
  /// agent needs a different budget than the rest of the process.
  ///
  /// `provider` is the tenant this agent talks to — its credentials and concurrency
  /// budget (see [`Provider::acquire`]) are used for every request the agent makes. Pass
  /// [`Provider::shared`] for the single-tenant default, or a tenant-specific
  /// [`Provider::new`] to isolate this agent's traffic (rate limit, API key, base URL)
  /// from other tenants running in the same process.
  ///
  /// A [`ContextOptimizer`] carrying [`config::max_history_tokens`] is registered as
  /// the first before-LLM hook, so an unbounded session cannot grow past the model's
  /// context window by default. Use [`Self::clear_before_llm_callbacks`] to change or
  /// drop it.
  pub fn new(
    provider: Provider,
    model: impl Into<String>,
    instructions: Option<impl Into<String>>,
    toolbox: Arc<ToolRegistry>,
  ) -> Self {
    Self {
      provider,
      model: model.into(),
      instructions: instructions.map(Into::into),
      toolbox,
      max_steps: config::max_tool_rounds() as u32,
      before_tool_callbacks: Vec::new(),
      after_tool_callbacks: Vec::new(),
      before_llm_callbacks: vec![Arc::new(
        ContextOptimizer::new(config::max_history_tokens()),
      )],
    }
  }

  /// Rounds of tool execution allowed before further rounds fall back to a
  /// tools-disabled request, instead of looping forever on a model that never stops
  /// calling tools. See [`Self::run`] / [`Self::run_structured`].
  #[must_use]
  pub fn with_max_steps(mut self, max_steps: u32) -> Self {
    self.max_steps = max_steps;
    self
  }

  /// This agent's identity, for checking whether a run suspended earlier can still be
  /// resumed by it. See [`RunFingerprint`].
  ///
  /// Derived rather than stored: the fields it covers are fixed for the agent's lifetime,
  /// so there is no state to keep in sync — and a cached copy is exactly the thing that
  /// would go stale and start approving resumes it should refuse.
  pub fn fingerprint(&self) -> RunFingerprint {
    RunFingerprint::new(
      &self.model,
      self.instructions.as_deref(),
      self.toolbox.names(),
    )
  }

  /// Register a hook invoked before each tool call ([`Self::execute_tool_calls`]).
  /// Callbacks run in registration order, one at a time: as soon as one returns
  /// `Some((status, content))`, the chain stops there — the real tool is not executed,
  /// remaining before-hooks are not run, and that pair is recorded as the result instead
  /// (the callback picks `status` itself, since a short-circuit is not always a failure,
  /// e.g. [`ToolResultStatus::Error`] for a permission check vs. [`ToolResultStatus::Success`]
  /// for a cache hit that substitutes a ready-made answer). If every callback returns
  /// `None`, the call proceeds to the real tool as normal.
  ///
  /// Can be called more than once to register several independent hooks (e.g. an audit
  /// log that never denies anything, plus a permission check that might) — each call adds
  /// one, it does not replace the others.
  #[must_use]
  pub fn with_before_tool_callback(mut self, callback: Arc<dyn BeforeToolCallback>) -> Self {
    self.before_tool_callbacks.push(callback);
    self
  }

  /// Register a hook invoked after each tool call completes ([`Self::execute_tool_calls`]).
  /// Callbacks run in registration order, each seeing the `(status, content)` left by the
  /// one before it; whenever one returns `Some((status, content))`, that pair becomes the
  /// input to the next callback and, once the chain finishes, the recorded result — e.g.
  /// redacting sensitive content in one hook, then compressing what is left in another.
  /// A callback returning `None` leaves the current `(status, content)` untouched for the
  /// next one.
  ///
  /// Can be called more than once to register several independent hooks — each call adds
  /// one, it does not replace the others.
  ///
  /// Calls short-circuited by [`Self::with_before_tool_callback`] never reach any of
  /// these, since their result did not come from a tool; a hook that has to see every
  /// recorded result has to be registered on both ends.
  #[must_use]
  pub fn with_after_tool_callback(mut self, callback: Arc<dyn AfterToolCallback>) -> Self {
    self.after_tool_callbacks.push(callback);
    self
  }

  /// Register a hook invoked before each LLM request, once the conversation has been
  /// flattened into an [`LlmRequest`] but before it becomes API messages
  /// ([`Self::prepare_llm_request`]). Use it to trim, compress, or enrich what goes out
  /// this round — injecting a dynamic system instruction, summarizing old turns,
  /// splicing in retrieved context — without touching [`ExecutionContext::events`],
  /// which stays the authoritative transcript.
  ///
  /// Callbacks run in registration order, each seeing the previous one's edits, appended
  /// after the ones already registered — including the default [`ContextOptimizer`]
  /// from [`Self::new`]. A hook that *adds* content therefore runs after that trim and is
  /// not covered by its budget; a hook that needs the last word on size has to enforce
  /// its own.
  ///
  /// Can be called more than once to register several independent hooks — each call adds
  /// one, it does not replace the others. To replace the default trim, clear the chain
  /// first with [`Self::clear_before_llm_callbacks`].
  #[must_use]
  pub fn with_before_llm_callback(mut self, callback: Arc<dyn BeforeLlmCallback>) -> Self {
    self.before_llm_callbacks.push(callback);
    self
  }

  /// Drop every before-LLM hook registered so far, **including** the default
  /// [`ContextOptimizer`] installed by [`Self::new`].
  ///
  /// This is how the history token budget is changed: it is a property of the callback,
  /// not of the agent, so adjusting it means installing a differently-configured one.
  ///
  /// ```no_run
  /// # use std::{collections::HashMap, sync::Arc};
  /// use agent::agent::Agent;
  /// use agent::callback::context_optimizer::ContextOptimizer;
  /// # use agent::llm::provider::Provider;
  /// # use agent::tools::ToolRegistry;
  /// #
  /// # let toolbox = Arc::new(ToolRegistry::empty());
  /// let agent = Agent::new(Provider::shared().clone(), "gpt-4o", Option::<String>::None, toolbox)
  ///   .clear_before_llm_callbacks()
  ///   .with_before_llm_callback(Arc::new(ContextOptimizer::new(32_000)));
  /// ```
  ///
  /// Clearing without registering anything else disables trimming entirely: every round
  /// then sends the full transcript, which is only safe when the caller bounds it some
  /// other way.
  #[must_use]
  pub fn clear_before_llm_callbacks(mut self) -> Self {
    self.before_llm_callbacks.clear();
    self
  }

  /// Whether the current round may still call the real tools, and whether reaching this
  /// point means the round budget just ran out (only meaningful when there are tools to
  /// spend a budget on).
  ///
  /// Extracted because [`Self::run`] and every structured route in [`structured`] all
  /// make and log this same decision once per round; keeping one copy means the "budget
  /// spent -> warn" policy cannot drift between them.
  fn round_budget(&self, context: &ExecutionContext) -> (bool, bool) {
    let tools_allowed = tool_rounds_remaining(context.current_step, self.max_steps);
    let budget_exhausted = !tools_allowed && !self.toolbox.is_empty();
    if budget_exhausted {
      tracing::warn!(
        max_steps = self.max_steps,
        "tool round budget spent; forcing a final answer"
      );
    }
    (tools_allowed, budget_exhausted)
  }

  /// Run to a plain-text final answer, starting a brand-new conversation.
  ///
  /// Once the round budget([`Self::with_max_steps`]) is spent, one last request goes out
  /// with tools disabled (see [`disable_tools`]) so a long task still returns an answer
  /// built from the results already gathered, instead of losing all of that work. That
  /// case is flagged through [`AgentResult::budget_exhausted`].
  ///
  /// Equivalent to [`Self::run_continuing`] with an empty history; use that instead to
  /// send a follow-up turn in an existing conversation.
  pub async fn run(&self, user_input: &str) -> anyhow::Result<AgentResult> {
    self.run_continuing(Vec::new(), user_input).await
  }

  /// Run to a plain-text final answer, continuing a conversation whose prior turns are
  /// `conversation`.
  ///
  /// Accepts a bare `Vec<Event>` — normally a previous call's
  /// `AgentResult::context.events` — or a [`Conversation`] carrying the caller's own
  /// identifier for the exchange. Keep that history on the caller's side (in memory, a
  /// database, an HTTP session store, ...) between calls and hand it back here for the
  /// next turn, so the model sees everything so far. This is what makes multi-turn
  /// conversations possible without `Agent` itself owning any session/storage concept: it
  /// stays a pure function of "prior events + new input" (see [`crate::agent::session`]
  /// for one way to manage that storage across calls).
  ///
  /// Passing a [`Conversation`] rather than a plain `Vec` additionally lets hooks
  /// accumulate work across turns — see [`Conversation::id`] and
  /// [`ExecutionContext::continuity_key`].
  ///
  /// The round budget ([`Self::with_max_steps`]) resets every call — `current_step`
  /// starts back at zero — so a long conversation is never penalized for rounds already
  /// spent on earlier turns; only this turn's own tool calls count against it.
  pub async fn run_continuing(
    &self,
    conversation: impl Into<Conversation>,
    user_input: &str,
  ) -> anyhow::Result<AgentResult> {
    let context = self.seed_context(conversation.into(), user_input);
    let mut outcome = self.drive(context).await?;

    // This entry point hands its caller a finished result, so there is nowhere to resume
    // to: a suspension is closed out as unanswered and the loop carries on from there.
    // Re-entering `drive` rather than looping inside it keeps the resumable path free of
    // a "can I suspend?" flag — see `run_continuing_resumable`.
    loop {
      match outcome {
        AgentOutcome::Done(result) => return Ok(result),
        AgentOutcome::Suspended(state) => {
          let budget_exhausted = state.budget_exhausted;
          let suspended = state.suspended.clone();
          let mut context = state.context;
          self.record_unanswered(&mut context, &suspended);
          // The round is over now that every call has a result, which is what
          // `increment_step` accounts for; `drive` leaves it alone precisely because a
          // suspended round is not finished.
          context.increment_step();
          outcome = self.drive(context).await?;
          if let AgentOutcome::Done(mut result) = outcome {
            result.budget_exhausted |= budget_exhausted;
            return Ok(result);
          }
        }
      }
    }
  }

  /// Like [`Self::run_continuing`], but stops and hands back a resumable state when a
  /// tool call cannot be decided, instead of recording it as unanswered.
  ///
  /// For a caller that has somewhere to come back from — a stored session, a web request
  /// that will be followed by another. See [`AgentOutcome`] and [`Self::resume`].
  pub async fn run_continuing_resumable(
    &self,
    conversation: impl Into<Conversation>,
    user_input: &str,
  ) -> anyhow::Result<AgentOutcome> {
    let context = self.seed_context(conversation.into(), user_input);
    self.drive(context).await
  }

  /// Carry on a run that stopped to ask, now that `decisions` answer what it asked.
  ///
  /// Only the suspended calls are re-attempted; their siblings from that round already
  /// ran and their results are in the transcript. That is the whole reason a round
  /// reports partial completion (see [`ToolRoundOutcome`]) — re-running the round wholesale
  /// would repeat every side effect those siblings had.
  ///
  /// `decisions` is keyed by [`SuspendedToolCall::tool_call_id`]. A call with no entry
  /// stays suspended and comes back in the returned state, so answering some of a round's
  /// questions and not others is a supported half-step rather than an error.
  ///
  /// # Errors
  ///
  /// If this agent is not the one the run was suspended from — a different model, edited
  /// instructions, a tool that no longer exists. See [`RunFingerprint`] for why that is
  /// refused rather than attempted.
  pub async fn resume(
    &self,
    state: AgentRunState,
    decisions: &HashMap<String, ResumedDecision>,
  ) -> anyhow::Result<AgentOutcome> {
    let current = self.fingerprint();
    if let Some(reason) = state.fingerprint.mismatch(&current) {
      anyhow::bail!("cannot resume this run: {reason}");
    }

    let mut outcome = self
      .drive_from_round(
        state.context,
        &state.suspended,
        decisions,
        state.budget_exhausted,
      )
      .await?;

    // A suspension carried over: some call still has no answer. Report it with the
    // budget flag intact rather than losing that the run was already cut short.
    match &mut outcome {
      AgentOutcome::Suspended(carried) => carried.budget_exhausted |= state.budget_exhausted,
      AgentOutcome::Done(result) => result.budget_exhausted |= state.budget_exhausted,
    }
    Ok(outcome)
  }

  /// Re-attempt `pending` with `decisions` in hand, then carry on with the ordinary loop.
  async fn drive_from_round(
    &self,
    mut context: ExecutionContext,
    pending: &[SuspendedToolCall],
    decisions: &HashMap<String, ResumedDecision>,
    budget_exhausted: bool,
  ) -> anyhow::Result<AgentOutcome> {
    let calls = rebuild_tool_calls(pending);
    let round = self
      .execute_tool_calls(&mut context, &calls, decisions)
      .await;

    if !round.suspended.is_empty() {
      return Ok(AgentOutcome::Suspended(AgentRunState {
        fingerprint: self.fingerprint(),
        suspended: round.suspended,
        budget_exhausted,
        context,
      }));
    }

    // Every call in the round now has a result, so the round is spent.
    context.increment_step();
    self.drive(context).await
  }

  /// The tool-calling loop itself, shared by every plain-text entry point.
  ///
  /// Returns [`AgentOutcome::Suspended`] the moment a round cannot finish, leaving
  /// `current_step` untouched: a suspended round is not a spent one, and charging the
  /// budget for it would mean a run that waits on a human gets fewer rounds than one that
  /// does not.
  ///
  /// # Why no placeholder repair happens here
  ///
  /// Between [`Self::record_tool_calls`] and a round's results, the transcript holds
  /// calls with no results — a shape providers reject. This loop never issues a request
  /// from that gap: it returns instead of looping while a round is incomplete, and the
  /// two ways back in both close the gap first ([`Self::resume`] answers the calls,
  /// [`Self::run_continuing`] records them as unanswered). So the invariant is kept by
  /// control flow rather than by filtering a bad request afterwards — which is worth
  /// preferring, because a filter can be forgotten by the next entry point while this
  /// cannot: there is nowhere else to issue a request from.
  async fn drive(&self, mut context: ExecutionContext) -> anyhow::Result<AgentOutcome> {
    loop {
      let (tools_allowed, budget_exhausted) = self.round_budget(&context);

      let llm_request = self.prepare_llm_request(&context).await;
      let messages = self.build_messages(llm_request)?;
      let mut builder = request_builder(
        &self.model,
        messages,
        DEFAULT_MAX_TOKENS,
        self.toolbox.definitions(),
      );
      if !tools_allowed {
        disable_tools(&mut builder, self.toolbox.definitions());
      }

      let response = self.complete(&builder).await?;
      self.record_usage(&mut context, &response);
      let message = first_choice(response)?.message;

      // Some providers send `Some(vec![])` rather than `None` for "no tool calls".
      let Some(tool_calls) = message.tool_calls.filter(|calls| !calls.is_empty()) else {
        let content = message
          .content
          .ok_or_else(|| anyhow::anyhow!("No content in final response"))?;
        self.record_final_answer(&mut context, &content);
        return Ok(AgentOutcome::Done(AgentResult {
          output: content,
          context,
          budget_exhausted,
        }));
      };

      // The model ignored the disabled tools. Returning its message beats looping
      // forever; the caller still sees whatever content came back.
      if !tools_allowed {
        tracing::warn!("model requested tools after they were disabled");
        let content = message.content.unwrap_or_default();
        self.record_final_answer(&mut context, &content);
        return Ok(AgentOutcome::Done(AgentResult {
          output: content,
          context,
          budget_exhausted: true,
        }));
      }

      self.record_tool_calls(&mut context, &tool_calls);
      let round = self
        .execute_tool_calls(&mut context, &tool_calls, &HashMap::new())
        .await;

      if !round.suspended.is_empty() {
        return Ok(AgentOutcome::Suspended(AgentRunState {
          fingerprint: self.fingerprint(),
          suspended: round.suspended,
          budget_exhausted,
          context,
        }));
      }

      context.increment_step();
    }
  }

  /// Streaming counterpart of [`Self::run`]: same tool-calling loop and round budget, but
  /// assistant text is forwarded to the caller as soon as the model emits it, instead of
  /// only once the whole run finishes. Equivalent to [`Self::run_continuing_stream`] with
  /// an empty history.
  pub fn run_stream<'a>(
    &'a self,
    user_input: &'a str,
  ) -> impl Stream<Item = anyhow::Result<AgentStreamEvent>> + 'a {
    self.run_continuing_stream(Vec::new(), user_input)
  }

  /// Streaming counterpart of [`Self::run_continuing`].
  ///
  /// Tool-call fragments are reassembled by [`ToolCallAccumulator`], the same way
  /// [`crate::llm::stream`] does it, but tools are executed through
  /// [`Self::execute_tool_calls`] rather than [`crate::tools::ToolRegistry::execute`], so
  /// every round is recorded into this run's [`ExecutionContext`] exactly as
  /// [`Self::run_continuing`] does (assistant text that accompanies a tool call is
  /// forwarded to the caller but, like [`Self::run_continuing`], not persisted into the
  /// transcript — only the call itself is), so a caller switching between the two gets the
  /// same transcript shape either way.
  ///
  /// Text is forwarded as [`AgentStreamEvent::Token`]s as soon as it arrives; the final
  /// [`AgentStreamEvent::Done`] carries the same fields as [`AgentResult`] and is always the
  /// last item, so a caller can stream tokens to a client while still waiting for `Done` to
  /// get the context to persist (e.g. into [`crate::agent::session::SessionStore`]).
  pub fn run_continuing_stream<'a>(
    &'a self,
    conversation: impl Into<Conversation>,
    user_input: &'a str,
  ) -> impl Stream<Item = anyhow::Result<AgentStreamEvent>> + 'a {
    // Converted before the generator so the returned stream owns a plain `Conversation`
    // and borrows nothing from the caller's argument.
    let conversation = conversation.into();
    stream! {
      let mut context = self.seed_context(conversation, user_input);

      loop {
        let (tools_allowed, budget_exhausted) = self.round_budget(&context);

        let llm_request = self.prepare_llm_request(&context).await;
        let messages = self.build_messages(llm_request)?;
        let mut builder = request_builder(
          &self.model,
          messages,
          DEFAULT_MAX_TOKENS,
          self.toolbox.definitions(),
        );
        if !tools_allowed {
          disable_tools(&mut builder, self.toolbox.definitions());
        }

        let mut chunks = self.complete_stream(&builder).await?;

        let mut accumulator = ToolCallAccumulator::default();
        // Forwarded token by token as it arrives; also kept so the final answer (or the
        // fallback content once tools are disabled) can be recorded as one piece.
        let mut assistant_text = String::new();

        while let Some(chunk) = chunks.next().await {
          let chunk = match chunk {
            Ok(chunk) => chunk,
            // Abort instead of continuing: after a transport error the remaining
            // fragments would assemble into a truncated tool call.
            Err(err) => {
              yield Err(err.into());
              return;
            }
          };

          let Some(choice) = chunk.choices.first() else {
            continue;
          };

          if let Some(fragments) = &choice.delta.tool_calls {
            accumulator.push(fragments);
          }

          // Heartbeat / role-declaration chunks have empty delta.content; skip to avoid
          // flooding downstream with empty tokens.
          if let Some(text) = &choice.delta.content
            && !text.is_empty()
          {
            assistant_text.push_str(text);
            yield Ok(AgentStreamEvent::Token(text.clone()));
          }
        }

        // No tool calls means the model produced its final answer.
        if accumulator.is_empty() {
          self.record_final_answer(&mut context, &assistant_text);
          yield Ok(AgentStreamEvent::Done {
            output: assistant_text,
            context,
            budget_exhausted,
          });
          return;
        }

        // The model ignored the disabled tools. Ending here beats looping forever; the
        // caller has already seen whatever text came back as `Token`s.
        if !tools_allowed {
          tracing::warn!("model requested tools after they were disabled");
          self.record_final_answer(&mut context, &assistant_text);
          yield Ok(AgentStreamEvent::Done {
            output: assistant_text,
            context,
            budget_exhausted: true,
          });
          return;
        }

        let tool_calls = match accumulator.finish() {
          Ok(tool_calls) => tool_calls,
          Err(err) => {
            yield Err(err);
            return;
          }
        };

        let started_items = self.record_tool_calls(&mut context, &tool_calls);
        yield Ok(AgentStreamEvent::ToolCallsStarted(started_items));

        let round = self
        .execute_tool_calls(&mut context, &tool_calls, &HashMap::new())
        .await;
        let mut finished_items = round.completed;
        // This entry point cannot suspend: its caller receives a stream, not a resumable
        // handle. See `record_unanswered`.
        if !round.suspended.is_empty() {
          finished_items.extend(self.record_unanswered(&mut context, &round.suspended));
        }
        yield Ok(AgentStreamEvent::ToolCallsFinished(finished_items));

        context.increment_step();
      }
    }
  }

  /// Issue one non-streaming chat completion, retrying transient failures with backoff.
  ///
  /// Every request-issuing method on `Agent` — the plain loop above and every structured
  /// route in [`structured`] — goes through this one call site instead of repeating the
  /// "acquire a permit, build the request, retry on failure" boilerplate; a fix made here
  /// (e.g. a different retry policy) applies to all of them at once. `builder` is taken by
  /// reference and rebuilt (`builder.build()`) on every attempt, since a retried request
  /// must be constructed fresh each time rather than reusing a value already consumed.
  ///
  /// [`is_transient`] decides what is worth another attempt. A rejected key, an unknown
  /// model or a request the provider considers malformed fails *identically* every time,
  /// so retrying one only spends the attempt budget and delays the real error reaching
  /// the caller by the whole backoff curve.
  async fn complete(
    &self,
    builder: &CreateChatCompletionRequestArgs,
  ) -> anyhow::Result<CreateChatCompletionResponse> {
    with_retry(
      || async {
        let _permit = self.provider.acquire().await?;
        let response = self
          .provider
          .client()
          .chat()
          .create(builder.build()?)
          .await?;
        anyhow::Ok(response)
      },
      is_transient,
    )
    .await
  }

  /// Streaming counterpart of [`Self::complete`]: same permit/retry handling, but opens a
  /// stream instead of awaiting one complete response.
  async fn complete_stream(
    &self,
    builder: &CreateChatCompletionRequestArgs,
  ) -> anyhow::Result<ChatCompletionResponseStream> {
    with_retry(
      || async {
        let _permit = self.provider.acquire().await?;
        let stream = self
          .provider
          .client()
          .chat()
          .create_stream(builder.build()?)
          .await?;
        anyhow::Ok(stream)
      },
      is_transient,
    )
    .await
  }

  /// Build the starting [`ExecutionContext`] for a call: a fresh execution id and step
  /// counter (see [`Self::run_continuing`] on why the round budget resets per call), the
  /// conversation's own id carried over so hooks can correlate turns (see
  /// [`ExecutionContext::continuity_key`]), and its prior events stored as-is so the full
  /// transcript is preserved for persistence (see
  /// [`crate::agent::session::SessionStore`]). Token-budget trimming happens later, on
  /// the per-round [`LlmRequest`] copy inside [`Self::prepare_llm_request`], so
  /// [`ExecutionContext::events`] is never destructively truncated.
  fn seed_context(&self, conversation: Conversation, user_input: &str) -> ExecutionContext {
    let mut context = ExecutionContext::new();
    context.conversation_id = conversation.id;
    context.conversation_scope = conversation.scope;
    context.events = conversation.events;
    self.record_user_input(&mut context, user_input);
    context
  }

  fn record_user_input(&self, context: &mut ExecutionContext, user_input: &str) {
    context.add_event(Event::new(
      context.execution_id.clone(),
      "user",
      vec![ContentItem::Message {
        role: "user".to_owned(),
        content: user_input.to_owned(),
      }],
    ));
  }

  /// Record the assistant's final text and set [`ExecutionContext::final_result`].
  fn record_final_answer(&self, context: &mut ExecutionContext, content: &str) {
    context.add_event(Event::new(
      context.execution_id.clone(),
      "agent",
      vec![ContentItem::Message {
        role: "assistant".to_owned(),
        content: content.to_owned(),
      }],
    ));
    context.final_result = Some(content.to_owned());
  }

  fn record_usage(&self, context: &mut ExecutionContext, response: &CreateChatCompletionResponse) {
    if let Some(usage) = &response.usage {
      context.usage.add(
        usage.prompt_tokens,
        usage.completion_tokens,
        usage.total_tokens,
      );
    }
  }

  /// Build this round's [`LlmRequest`] — system prompt plus flattened transcript — and
  /// run every hook over it, in registration order, starting with the
  /// [`ContextOptimizer`] that [`Self::new`] installs by default.
  ///
  /// The system prompt is part of the request rather than prepended afterwards, so a hook
  /// that measures or rewrites the prompt sees all of it; see [`LlmRequest`].
  ///
  /// `context` is borrowed immutably: the whole point of this path is that the request is
  /// a throwaway copy and [`ExecutionContext::events`] survives intact for persistence.
  async fn prepare_llm_request(&self, context: &ExecutionContext) -> LlmRequest {
    let mut request = LlmRequest::new(self.instructions.clone(), &context.events);

    for callback in &self.before_llm_callbacks {
      callback.call(context, &mut request).await;
    }

    request
  }

  /// [`Self::prepare_llm_request`] for a caller that will append `trailer` as a trailing
  /// system message of its own (see [`structured`]'s `json_object` route, where the schema
  /// hint has to be the last thing the model reads).
  ///
  /// Such a message is part of what goes on the wire, so a hook measuring the request has
  /// to see it — otherwise every token budget in the chain undercounts by exactly the
  /// trailer's length, on the one route where that text is most likely to be a large
  /// generated schema. It is therefore pushed as an instruction *before* the hooks run and
  /// removed again afterwards, leaving the caller free to place it wherever it belongs.
  ///
  /// A hook that rewrote or dropped the trailer is respected rather than fought: removal
  /// matches on the exact text, so if it is no longer there nothing happens — it was still
  /// accounted for, which is the point.
  async fn prepare_llm_request_with_trailer(
    &self,
    context: &ExecutionContext,
    trailer: &str,
  ) -> LlmRequest {
    let mut request = LlmRequest::new(self.instructions.clone(), &context.events);
    request.push_instruction(trailer);

    for callback in &self.before_llm_callbacks {
      callback.call(context, &mut request).await;
    }

    if let Some(at) = request
      .instructions
      .iter()
      .rposition(|instruction| instruction == trailer)
    {
      request.instructions.remove(at);
    }

    request
  }

  /// Record one round's tool calls into the transcript and return the same items, so a
  /// caller that also wants to forward them live (see
  /// [`AgentStreamEvent::ToolCallsStarted`]) does not have to recompute or reparse
  /// anything already done here.
  fn record_tool_calls(
    &self,
    context: &mut ExecutionContext,
    tool_calls: &[ChatCompletionMessageToolCalls],
  ) -> Vec<ContentItem> {
    let mut call_items = Vec::new();
    for tool_call in tool_calls {
      if let ChatCompletionMessageToolCalls::Function(function_call) = tool_call {
        let arguments: Value =
          serde_json::from_str(&function_call.function.arguments).unwrap_or(Value::Null);
        call_items.push(ContentItem::ToolCall {
          tool_call_id: function_call.id.clone(),
          name: function_call.function.name.clone(),
          arguments,
        });
      }
    }
    context.add_event(Event::new(
      context.execution_id.clone(),
      "agent",
      call_items.clone(),
    ));
    call_items
  }

  /// Execute every call in one model turn.
  ///
  /// Calls run concurrently rather than one after another: [`crate::tools::Tool::execute`]
  /// no longer takes the caller's [`ExecutionContext`] at all (see that method's docs for
  /// why),
  /// so independent tool calls requested in the same turn (e.g. two `web_search` calls) do
  /// not have to pay for each other's network latency in sequence, and there is no shared
  /// state to serialize access to.
  ///
  /// [`Self::with_before_tool_callback`] hooks run first, in registration order, and the
  /// first one that does not return [`ToolCallDecision::Proceed`] ends the chain right
  /// there, before the real tool runs; [`Self::with_after_tool_callback`] hooks run last,
  /// each seeing the result left by the one before it, and may rewrite it (e.g. redact
  /// sensitive content, then compress what is left) before it is recorded. All of them are
  /// read-only borrows of `context`, so they do not conflict with running the calls
  /// concurrently.
  ///
  /// Returns both halves of the round — see [`ToolRoundOutcome`] for why a round is not
  /// all-or-nothing. The completed results are the same items added to the transcript, so
  /// a caller that also wants to forward them live (see
  /// [`AgentStreamEvent::ToolCallsFinished`]) does not have to reach back into
  /// `context.events` to find them.
  ///
  /// `resumed` carries answers obtained since a previous attempt suspended these calls;
  /// it is empty for an ordinary round. See [`ResumedDecision`] for why an answer
  /// replaces only a suspension and leaves the rest of the chain in force.
  async fn execute_tool_calls(
    &self,
    context: &mut ExecutionContext,
    tool_calls: &[ChatCompletionMessageToolCalls],
    resumed: &HashMap<String, ResumedDecision>,
  ) -> ToolRoundOutcome {
    // Reborrowed immutably: every concurrent call below may read `context` (e.g. to make
    // an allow/deny decision), while the mutable borrow needed to record the resulting
    // event is only taken back once all of them have resolved, below.
    let context_ref: &ExecutionContext = &*context;

    // Every call resolves to either a result or a suspension; `join_all` keeps them in
    // call order, which both lists below inherit.
    let settled = join_all(tool_calls.iter().filter_map(|tool_call| {
      let ChatCompletionMessageToolCalls::Function(function_call) = tool_call else {
        return None;
      };

      Some(async move {
        let function_name = &function_call.function.name;
        let arguments = &function_call.function.arguments;

        tracing::debug!(
          tool = %function_name,
          arguments = %truncate_chars(arguments, TOOL_LOG_PREVIEW_CHARS),
          "executing tool"
        );

        // Run every before-hook in registration order; the first not to let the call
        // through ends the chain. A short-circuit picks its own status, since it is not
        // always a denial (e.g. a cache hit substituting a real result is `Success`), so
        // it is not this call site's place to guess.
        for before in &self.before_tool_callbacks {
          // Both forms are handed over: parsed for callbacks that inspect a field, raw
          // for those that show the call to a human, since a payload that fails to parse
          // would otherwise be rendered as a bare `null`.
          let parsed_arguments: Value = serde_json::from_str(arguments).unwrap_or(Value::Null);
          let view = ToolCallView {
            tool_call_id: &function_call.id,
            name: function_name,
            arguments: &parsed_arguments,
            raw_arguments: arguments,
          };
          match before.call(context_ref, view).await {
            ToolCallDecision::Proceed => {}
            ToolCallDecision::ShortCircuit(status, content) => {
              tracing::debug!(
                tool = %function_name,
                status = ?status,
                "tool call short-circuited by before-tool callback"
              );
              return Err(ContentItem::ToolResult {
                tool_call_id: function_call.id.clone(),
                name: function_name.clone(),
                status,
                content,
              });
            }
            ToolCallDecision::Suspend => {
              // An answer obtained since this call last suspended stands in for the
              // decision the hook still cannot make. Only the suspension is replaced:
              // `Approved` falls through to the rest of the chain and then the tool, so a
              // sandbox or guard registered after this hook still gets its say.
              match resumed.get(&function_call.id) {
                Some(ResumedDecision::Approved) => {
                  tracing::debug!(
                    tool = %function_name,
                    "a supplied decision approved a previously suspended call"
                  );
                }
                Some(ResumedDecision::Refused(reason)) => {
                  tracing::debug!(
                    tool = %function_name,
                    "a supplied decision refused a previously suspended call"
                  );
                  return Err(ContentItem::ToolResult {
                    tool_call_id: function_call.id.clone(),
                    name: function_name.clone(),
                    status: ToolResultStatus::Error,
                    content: reason.clone(),
                  });
                }
                None => {
                  tracing::debug!(
                    tool = %function_name,
                    "tool call suspended by before-tool callback, awaiting a decision"
                  );
                  return Ok(SuspendedToolCall {
                    tool_call_id: function_call.id.clone(),
                    name: function_name.clone(),
                    raw_arguments: arguments.clone(),
                  });
                }
              }
            }
          }
        }

        let (mut status, mut content) = match self.toolbox.get(function_name) {
          Some(tool) => match tool.execute(arguments).await {
            Ok(result) => {
              tracing::debug!(
                tool = %function_name,
                result = %truncate_chars(&result, TOOL_LOG_PREVIEW_CHARS),
                "tool finished"
              );
              (ToolResultStatus::Success, result)
            }
            Err(err) => {
              let msg = format!("Tool execution error: {err}");
              tracing::warn!("{msg}");
              (ToolResultStatus::Error, msg)
            }
          },
          None => {
            let msg = format!("Tool execution error: unknown tool {function_name}");
            tracing::warn!("{msg}");
            (ToolResultStatus::Error, msg)
          }
        };

        // Run every after-hook in registration order, threading the (possibly rewritten)
        // result from one into the next, so e.g. a redaction hook and a compression hook
        // can both apply to the same result without knowing about each other.
        for after in &self.after_tool_callbacks {
          if let Some((new_status, new_content)) = after
            .call(
              context_ref,
              &function_call.id,
              function_name,
              status,
              &content,
            )
            .await
          {
            status = new_status;
            content = new_content;
          }
        }

        Err(ContentItem::ToolResult {
          tool_call_id: function_call.id.clone(),
          name: function_name.clone(),
          status,
          content,
        })
      })
    }))
    .await;

    // `Err` is the ordinary case here, not a failure: the two variants only distinguish
    // "produced a result" from "did not run", and `Result` is the shape `partition` reads.
    let (suspended, completed): (Vec<_>, Vec<_>) = settled.into_iter().partition(Result::is_ok);
    let suspended: Vec<SuspendedToolCall> = suspended.into_iter().map(Result::unwrap).collect();
    let completed: Vec<ContentItem> = completed
      .into_iter()
      .map(|item| item.expect_err("partitioned as Err above"))
      .collect();

    // Only the results are recorded. A suspended call already has its
    // `ContentItem::ToolCall` in the transcript from `record_tool_calls`, and must not
    // also get a result: it has not produced one, and inventing one here would make the
    // decision look already taken to anything reading the transcript back.
    if !completed.is_empty() {
      context.add_event(Event::new(
        context.execution_id.clone(),
        "tool",
        completed.clone(),
      ));
    }

    ToolRoundOutcome {
      completed,
      suspended,
    }
  }

  /// Record a placeholder result for every call in `suspended`, for an entry point with
  /// nowhere to resume to.
  ///
  /// A one-shot [`Self::run`], or a structured run, has no session behind it and no way
  /// to hand a pending question to a caller who could come back with an answer — so the
  /// only honest thing to report is that the call did not happen. Returns the placeholder
  /// items so a streaming caller can forward them alongside the real results.
  ///
  /// Leaving the calls unanswered instead is not an option: a
  /// [`ContentItem::ToolCall`] with no matching [`ContentItem::ToolResult`] is a
  /// conversation most providers reject outright, and `record_tool_calls` has already
  /// written the call into the transcript by this point. The transcript is briefly in
  /// exactly that invalid state while the decision is outstanding, which is tolerable
  /// only because it is never sent or persisted from there — see
  /// [`crate::callback::context_optimizer::safety`] for the same invariant enforced from
  /// the other direction.
  fn record_unanswered(
    &self,
    context: &mut ExecutionContext,
    suspended: &[SuspendedToolCall],
  ) -> Vec<ContentItem> {
    let items: Vec<ContentItem> = suspended
      .iter()
      .map(|call| {
        tracing::warn!(
          tool = %call.name,
          "a tool call needed a decision this run cannot obtain; recording it as unanswered"
        );
        ContentItem::ToolResult {
          tool_call_id: call.tool_call_id.clone(),
          name: call.name.clone(),
          status: ToolResultStatus::Error,
          content: format!(
            "Tool execution stopped: {} needs approval, which this run has no way to \
             ask for.",
            call.name
          ),
        }
      })
      .collect();

    context.add_event(Event::new(
      context.execution_id.clone(),
      "tool",
      items.clone(),
    ));
    items
  }

  /// Render a prepared [`LlmRequest`] into the message shape the API expects.
  ///
  /// Everything comes from `request`, including the system prompt: by this point hooks
  /// have had their say, and re-reading [`Self::instructions`] here would silently undo
  /// any edit they made to it.
  ///
  /// Takes the request **by value** and moves each payload into the message that will
  /// carry it. Borrowing would mean cloning every string a second time — the request is
  /// already a per-round copy of the transcript (see [`Self::prepare_llm_request`]), so
  /// on a long run with bulky tool output that second copy is pure waste, and it is
  /// paid on every round.
  fn build_messages(
    &self,
    request: LlmRequest,
  ) -> anyhow::Result<Vec<ChatCompletionRequestMessage>> {
    let LlmRequest {
      instructions,
      contents,
    } = request;
    let mut messages = Vec::with_capacity(instructions.len() + contents.len());

    for instruction in instructions {
      messages.push(
        ChatCompletionRequestSystemMessageArgs::default()
          .content(instruction)
          .build()?
          .into(),
      );
    }

    for item in contents {
      match item {
        ContentItem::Message { role, content } => {
          let message: ChatCompletionRequestMessage = if role == "user" {
            ChatCompletionRequestUserMessageArgs::default()
              .content(content)
              .build()?
              .into()
          } else {
            ChatCompletionRequestAssistantMessageArgs::default()
              .content(content)
              .build()?
              .into()
          };
          messages.push(message);
        }
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

          if let Some(ChatCompletionRequestMessage::Assistant(last)) = messages.last_mut() {
            last.tool_calls.get_or_insert_with(Vec::new).push(tool_call);
          } else {
            messages.push(
              ChatCompletionRequestAssistantMessageArgs::default()
                .tool_calls(vec![tool_call])
                .build()?
                .into(),
            );
          }
        }
        ContentItem::ToolResult {
          tool_call_id,
          content,
          ..
        } => {
          messages.push(
            ChatCompletionRequestToolMessageArgs::default()
              .tool_call_id(tool_call_id)
              .content(content)
              .build()?
              .into(),
          );
        }
      }
    }

    Ok(messages)
  }
}

/// Whether another round may still call the real tools.
///
/// Mirrors `round < max_rounds` in [`crate::llm::tool_loop::run`]: once the budget is
/// spent, the caller must fall back to a tools-disabled (or `final_answer`-only) request
/// instead of looping forever on a model that never stops calling tools.
fn tool_rounds_remaining(current_step: u32, max_steps: u32) -> bool {
  current_step < max_steps
}

/// See `runtime/tests.rs`: split out of this file because the implementation above and
/// its tests together no longer fit comfortably in one file to read through.
#[cfg(test)]
mod tests;
