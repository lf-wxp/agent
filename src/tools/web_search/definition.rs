use serde_json::{Value, json};

use crate::tools::web_search::{DEFAULT_MAX_RESULTS, MAX_RESULTS_LIMIT};

/// What the tool does, shown to the model.
pub const DESCRIPTION: &str = "Search the web for current information. Use this for facts \
                               that may have changed recently, or that fall outside your \
                               knowledge. Returns ranked snippets, each with its source URL.";

/// JSON Schema for [`super::WebSearchArgs`].
///
/// Only `query` and `max_results` are exposed. Search depth and the API key are
/// operational concerns (credits, credentials) that the model must not control, so they
/// come from [`crate::config`] instead of the schema.
pub fn parameters() -> Value {
  json!({
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
  })
}
