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

/// Runs before a tool is invoked and may short-circuit it entirely.
///
/// Returning `Some((status, content))` skips the real tool call and records `content` as
/// its result instead — `status` is up to the implementation, since a short-circuit is
/// not always a failure: e.g. a permission check that rejects the call would return
/// [`ToolResultStatus::Error`], while a cache hit that substitutes a ready-made answer
/// for the tool's own work would return [`ToolResultStatus::Success`]. Returning `None`
/// lets the call proceed to the real tool as normal.
#[async_trait::async_trait]
pub trait BeforeToolCallback: Send + Sync {
  async fn call(
    &self,
    context: &ExecutionContext,
    tool_call: ToolCallView<'_>,
  ) -> Option<(ToolResultStatus, String)>;
}

/// Runs after a tool call has produced a result and may rewrite it.
///
/// Returning `Some((status, content))` replaces what is recorded in the transcript and
/// sent back to the model — redacting a secret, say, or compressing a result too bulky to
/// be worth the context it occupies (see [`crate::callback::search_compressor`]).
/// Returning `None` records the tool's own result unchanged.
///
/// Not called for a call a [`BeforeToolCallback`] short-circuited: that result never came
/// from a tool, so there is nothing to post-process. A hook that has to observe *every*
/// recorded result — an audit log, say — therefore has to implement both traits.
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
