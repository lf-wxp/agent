use serde_json::Value;

use crate::agent::{ExecutionContext, ToolResultStatus, llm_request::LlmRequest};

/// The call a [`BeforeToolCallback`] is being asked to rule on.
#[derive(Debug, Clone, Copy)]
pub struct ToolCallView<'a> {
  pub tool_call_id: &'a str,
  pub name: &'a str,
  /// Parsed arguments, for callbacks that decide by inspecting a field — e.g. which path
  /// a delete is aimed at. [`Value::Null`] when the model produced something that is not
  /// valid JSON, in which case `raw_arguments` is the only faithful record of what it
  /// asked for.
  pub arguments: &'a Value,
  /// Exactly the string the tool will be handed.
  ///
  /// Prefer this over `arguments` whenever the call is shown to a human: rendering the
  /// parsed value alone would print `null` for a malformed payload, i.e. ask someone to
  /// approve a call they cannot actually see.
  pub raw_arguments: &'a str,
}

/// What a [`BeforeToolCallback`] decided about a call.
///
/// Three outcomes rather than two, because "no" and "not yet" are different answers and
/// collapsing them loses the only one that can still be recovered from. A denial is
/// final: it is recorded and the model moves on. A suspension says the question is
/// legitimate but cannot be answered from here — typically because it needs a human who
/// is not currently present — and asks the caller to obtain the answer and run the call
/// again.
#[derive(Debug)]
pub enum ToolCallDecision {
  /// Let the call through to the real tool, or to the next callback in the chain.
  Proceed,
  /// Skip the real tool and record `content` as the result instead.
  ///
  /// The status is the callback's to choose, since a short-circuit is not always a
  /// failure: a permission check that rejects the call returns
  /// [`ToolResultStatus::Error`], while a cache hit that substitutes a ready-made answer
  /// for the tool's own work returns [`ToolResultStatus::Success`].
  ShortCircuit(ToolResultStatus, String),
  /// This call cannot be decided right now; stop before running it and hand it back to
  /// the caller undecided.
  ///
  /// Carries nothing: the call site already knows which call this is, and re-raising it
  /// later needs nothing the transcript does not already hold. What the caller does with
  /// it depends on whether it can resume — an entry point with somewhere to resume *to*
  /// (a persisted session) can suspend the run and come back to it, while a one-shot run
  /// has no choice but to record the call as unanswered.
  ///
  /// A suspended call has run no part of the real tool, so nothing about it has to be
  /// undone before it is asked again. That is a property of *where* this decision is
  /// made — before the tool, never during it — and it is what makes resuming safe rather
  /// than merely possible.
  Suspend,
}

impl ToolCallDecision {
  /// Short-circuit with an error result — the shape a permission check wants.
  pub fn deny(content: impl Into<String>) -> Self {
    Self::ShortCircuit(ToolResultStatus::Error, content.into())
  }

  /// Whether this decision lets the call through to the tool.
  ///
  /// The question most often asked of a decision, and the one worth a name: for anything
  /// that only cares whether the tool ran, the three variants collapse to this, and
  /// spelling it out as a `matches!` at every such site buries it.
  pub fn is_proceed(&self) -> bool {
    matches!(self, Self::Proceed)
  }
}

/// Runs before a tool is invoked and may short-circuit or postpone it.
///
/// See [`ToolCallDecision`] for what can be returned and what each outcome means.
/// Callbacks run in registration order and the first one not to return
/// [`ToolCallDecision::Proceed`] ends the chain, so a later hook never sees a call an
/// earlier one already ruled on.
#[async_trait::async_trait]
pub trait BeforeToolCallback: Send + Sync {
  async fn call(&self, context: &ExecutionContext, tool_call: ToolCallView<'_>)
  -> ToolCallDecision;
}

/// Runs after a tool call has produced a result and may rewrite it.
///
/// Returning `Some((status, content))` replaces what is recorded in the transcript and
/// sent back to the model — redacting a secret, say, or compressing a result too bulky to
/// be worth the context it occupies (see [`crate::callback::search_compressor`]).
/// Returning `None` records the tool's own result unchanged.
///
/// Not called for a call that never reached the tool — one a [`BeforeToolCallback`]
/// short-circuited or suspended: neither produced a tool result, so there is nothing to
/// post-process. A hook that has to observe *every* recorded result — an audit log, say —
/// therefore has to implement both traits.
#[async_trait::async_trait]
pub trait AfterToolCallback: Send + Sync {
  async fn call(
    &self,
    context: &ExecutionContext,
    tool_call_id: &str,
    tool_name: &str,
    status: ToolResultStatus,
    content: &str,
  ) -> Option<(ToolResultStatus, String)>;
}

/// Runs on every round, after [`LlmRequest`] is built from [`ExecutionContext::events`]
/// and before it is converted into API messages.
///
/// Implementations may read `context` (for auditing or budget decisions) and freely mutate
/// `request` — trimming [`LlmRequest::contents`], appending instructions, compressing tool
/// results, injecting retrieved snippets. All of it stays local to this round's request.
/// `context` is borrowed immutably on purpose: the transcript is the authoritative record
/// and no hook on this path may rewrite it.
///
/// Hooks run in registration order, each seeing the previous one's edits, so an ordering
/// choice matters: the default
/// [`crate::callback::context_optimizer::ContextOptimizer`] sits first in the chain
/// and therefore does not account for content a later hook adds.
#[async_trait::async_trait]
pub trait BeforeLlmCallback: Send + Sync {
  async fn call(&self, context: &ExecutionContext, request: &mut LlmRequest);
}
