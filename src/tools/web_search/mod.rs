//! Web search tool backed by the [Tavily](https://tavily.com) API.
//!
//! The definition and the executor share these argument types so the JSON Schema
//! advertised to the model can never drift from what the executor accepts.

pub mod definition;
pub mod execute;

use serde::Deserialize;

/// Tool name, used both in the definition and in dispatch.
pub const NAME: &str = "web_search";

/// Result count used when the model does not ask for a specific one.
pub const DEFAULT_MAX_RESULTS: u8 = 5;

/// Tavily rejects `max_results` above 20 with a 400.
pub const MAX_RESULTS_LIMIT: u8 = 20;

/// Arguments as produced by the model.
#[derive(Debug, Deserialize)]
pub struct WebSearchArgs {
  pub query: String,
  /// Optional: models routinely omit it, so a missing value must not fail validation.
  #[serde(default)]
  pub max_results: Option<u8>,
}

impl WebSearchArgs {
  /// Result count clamped into Tavily's accepted range.
  ///
  /// Clamping rather than validating: a hallucinated `max_results` should not turn a
  /// usable query into an error the model has to recover from.
  pub fn effective_max_results(&self) -> u8 {
    self
      .max_results
      .unwrap_or(DEFAULT_MAX_RESULTS)
      .clamp(1, MAX_RESULTS_LIMIT)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn parse(arguments: &str) -> WebSearchArgs {
    serde_json::from_str(arguments).unwrap()
  }

  #[test]
  fn defaults_max_results_when_omitted() {
    let args = parse(r#"{"query":"rust async book"}"#);
    assert_eq!(args.query, "rust async book");
    assert_eq!(args.effective_max_results(), DEFAULT_MAX_RESULTS);
  }

  #[test]
  fn honours_requested_max_results() {
    assert_eq!(
      parse(r#"{"query":"q","max_results":3}"#).effective_max_results(),
      3
    );
  }

  #[test]
  fn clamps_out_of_range_max_results() {
    assert_eq!(
      parse(r#"{"query":"q","max_results":99}"#).effective_max_results(),
      MAX_RESULTS_LIMIT
    );
    assert_eq!(
      parse(r#"{"query":"q","max_results":0}"#).effective_max_results(),
      1
    );
  }

  #[test]
  fn requires_query() {
    assert!(serde_json::from_str::<WebSearchArgs>(r#"{"max_results":3}"#).is_err());
  }
}
