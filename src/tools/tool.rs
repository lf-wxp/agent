//! The `Tool` abstraction implemented by every tool.

use async_openai::types::chat::{ChatCompletionTool, ChatCompletionTools, FunctionObjectArgs};
use serde_json::Value;

/// A tool the model can call.
///
/// Uses `async_trait` rather than a native `async fn`: the registry keeps tools as
/// `Box<dyn Tool>`, and native async fns in traits are not dyn-compatible.
#[async_trait::async_trait]
pub trait Tool: Send + Sync {
  /// Name used both in the advertised definition and for dispatch; must be unique
  /// across the registry, otherwise only the first match is ever reached.
  fn name(&self) -> &str;

  /// What the tool does, and when the model should reach for it.
  fn description(&self) -> &str;

  /// JSON Schema for the arguments, used as `function.parameters`.
  fn parameters(&self) -> Value;

  /// Run the tool against the raw JSON arguments produced by the model.
  ///
  /// Return `Err` for anything that did not work — malformed arguments, a failing
  /// request. The registry turns it into a tool message so the model can correct
  /// itself, which is why implementations never format their own error text.
  ///
  /// Takes only the raw arguments, not the caller's [`crate::agent::ExecutionContext`]:
  /// no built-in tool has ever needed the surrounding conversation to do its job, and a
  /// parameter every implementation had to name `_context` to ignore was pure
  /// speculative generality. Add it back (as a `Vec<Event>` snapshot or similar) if a
  /// concrete tool needs it — e.g. to avoid repeating a lookup already made earlier in
  /// the same turn.
  async fn execute(&self, args_json: &str) -> anyhow::Result<String>;

  /// Render the definition advertised to the model.
  ///
  /// Fails only when name / description / parameters are inconsistent with what the API
  /// accepts, which is a programming error rather than a runtime condition.
  fn definition(&self) -> anyhow::Result<ChatCompletionTools> {
    let function = FunctionObjectArgs::default()
      .name(self.name())
      .description(self.description())
      .parameters(self.parameters())
      .build()
      .map_err(|e| anyhow::anyhow!("Failed to build tool definition for {}: {e}", self.name()))?;

    Ok(ChatCompletionTools::Function(ChatCompletionTool {
      function,
    }))
  }
}
