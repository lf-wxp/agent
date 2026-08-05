//! Tool registry: the definitions exposed to the model plus name-based dispatch.

pub mod calculator;
pub mod tool;
pub mod web_search;

use std::sync::LazyLock;

use async_openai::types::chat::ChatCompletionTools;

use crate::tools::{calculator::Calculator, tool::Tool, web_search::WebSearch};

/// Every registered tool. The order here is the order advertised to the model.
///
/// Adding a tool means adding one entry: dispatch and the advertised definitions are both
/// derived from this list, so the two cannot fall out of sync.
static TOOLS: LazyLock<Vec<Box<dyn Tool>>> =
  LazyLock::new(|| vec![Box::new(Calculator), Box::new(WebSearch)]);

/// Rendered once: building definitions allocates, and they never change at runtime.
static DEFINITIONS: LazyLock<Vec<ChatCompletionTools>> = LazyLock::new(|| {
  TOOLS
    .iter()
    .map(|tool| tool.definition())
    .collect::<anyhow::Result<Vec<_>>>()
    // Definitions are constants: a failure is a programming error in a tool, not a
    // runtime condition a caller could handle.
    .expect("tool definitions must be valid")
});

/// All tools available to the model.
pub fn tools() -> &'static [ChatCompletionTools] {
  &DEFINITIONS
}

/// Dispatch a tool call by name.
///
/// Never fails: an unknown or failing tool must still yield a tool message, otherwise
/// the follow-up request is rejected for a `tool_call_id` without a matching result.
/// The error text goes back to the model so it can correct itself, which is why
/// [`Tool::execute`] returns a `Result` and the formatting happens here instead of in
/// every tool.
pub async fn execute_tool(name: &str, arguments: &str) -> String {
  // Linear scan: the registry holds a handful of tools, so a map would not pay for itself.
  let Some(tool) = TOOLS.iter().find(|tool| tool.name() == name) else {
    return format!("Error: unknown tool `{name}`");
  };

  match tool.execute(arguments).await {
    Ok(output) => output,
    Err(err) => {
      // Keep the full chain in the logs; the model only needs the summary.
      tracing::warn!("tool `{name}` failed: {err:#}");
      format!("Error: {err}")
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[tokio::test]
  async fn reports_unknown_tool_instead_of_failing() {
    assert!(execute_tool("nope", "{}").await.starts_with("Error:"));
  }

  #[tokio::test]
  async fn reports_tool_failure_as_text() {
    let output = execute_tool(calculator::NAME, "not json").await;
    assert!(output.starts_with("Error:"), "got: {output}");
  }

  #[tokio::test]
  async fn dispatches_to_the_named_tool() {
    let output = execute_tool(
      calculator::NAME,
      r#"{"operator":"add","first_number":2,"second_number":3}"#,
    )
    .await;

    assert_eq!(output, "5");
  }

  #[test]
  fn advertises_one_definition_per_tool() {
    let names: Vec<&str> = tools()
      .iter()
      .map(|tool| match tool {
        ChatCompletionTools::Function(tool) => tool.function.name.as_str(),
        ChatCompletionTools::Custom(_) => panic!("expected a function tool"),
      })
      .collect();

    assert_eq!(names, [calculator::NAME, web_search::NAME]);
  }

  #[test]
  fn tool_names_are_unique() {
    // A duplicate name would make dispatch silently reach only the first match.
    let mut names: Vec<&str> = TOOLS.iter().map(|tool| tool.name()).collect();
    let count = names.len();
    names.sort_unstable();
    names.dedup();

    assert_eq!(names.len(), count, "duplicate tool name in the registry");
  }
}
