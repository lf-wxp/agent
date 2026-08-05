use async_openai::types::chat::{ChatCompletionTool, ChatCompletionTools, FunctionObjectArgs};
use serde_json::json;

use crate::tools::web_search::{DEFAULT_MAX_RESULTS, MAX_RESULTS_LIMIT, NAME};

/// Function definition advertised to the model.
///
/// Only `query` and `max_results` are exposed. Search depth and the API key are
/// operational concerns (credits, credentials) that the model must not control, so they
/// come from [`crate::config`] instead of the schema.
pub fn definition() -> ChatCompletionTools {
  ChatCompletionTools::Function(ChatCompletionTool {
    function: FunctionObjectArgs::default()
      .name(NAME)
      .description(
        "Search the web for current information. Use this for facts that may have changed \
         recently, or that fall outside your knowledge. Returns ranked snippets, each with \
         its source URL.",
      )
      .parameters(json!({
        "type": "object",
        "properties": {
          "query": {
            "type": "string",
            "description": "Search query in natural language, e.g. `who won the 2026 world cup final`"
          },
          "max_results": {
            "type": "integer",
            "description": format!("How many results to return, 1-{MAX_RESULTS_LIMIT} (default {DEFAULT_MAX_RESULTS})"),
            "minimum": 1,
            "maximum": MAX_RESULTS_LIMIT
          }
        },
        "required": ["query"],
        "additionalProperties": false
      }))
      .build()
      // The definition is a constant: a build failure is a programming error, not a
      // runtime condition, so there is nothing for the caller to recover from.
      .expect("web_search tool definition must be valid"),
  })
}
