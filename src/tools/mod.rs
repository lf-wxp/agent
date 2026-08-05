//! Tool registry: the definitions exposed to the model plus name-based dispatch.

pub mod calculator;

use std::sync::LazyLock;

use async_openai::types::chat::ChatCompletionTools;

/// Built once: definitions are static, and rendering them allocates.
static TOOLS: LazyLock<Vec<ChatCompletionTools>> =
  LazyLock::new(|| vec![calculator::definition::definition()]);

/// All tools available to the model.
pub fn tools() -> &'static [ChatCompletionTools] {
  &TOOLS
}

/// Dispatch a tool call by name.
///
/// Never fails: an unknown or failing tool must still yield a tool message, otherwise
/// the follow-up request is rejected for a `tool_call_id` without a matching result.
/// The error text goes back to the model so it can correct itself.
///
/// Kept `async` so IO-bound tools (search, fetch) can be added without touching callers.
pub async fn execute_tool(name: &str, arguments: &str) -> String {
  match name {
    calculator::NAME => calculator::execute::run(arguments),
    other => format!("Error: unknown tool `{other}`"),
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[tokio::test]
  async fn reports_unknown_tool_instead_of_failing() {
    assert!(execute_tool("nope", "{}").await.starts_with("Error:"));
  }

  #[test]
  fn exposes_every_registered_tool() {
    assert_eq!(tools().len(), 1);
  }
}
