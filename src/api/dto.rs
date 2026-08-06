//! Request/response bodies for the HTTP agent API. Kept separate from [`crate::agent`]'s
//! own types because the wire format (JSON, `camelCase`, string tool names) and the Rust
//! API (typed [`crate::tools::ToolRegistry`], `Arc`s) are different concerns that would
//! otherwise get tangled together.

use serde::{Deserialize, Serialize};

use crate::agent::TokenUsage;

/// Body of `POST /v1/agent/run`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunRequest {
  /// The task/question to hand to the agent.
  pub input: String,
  /// System prompt. Falls back to the agent having no special instructions.
  #[serde(default)]
  pub instructions: Option<String>,
  /// Overrides the tenant's [`crate::api::tenant::Tenant::default_model`] /
  /// [`crate::config::model`] for this call only.
  #[serde(default)]
  pub model: Option<String>,
  /// Tool names to make available, e.g. `["calculator", "web_search"]`. Omitted means the
  /// full built-in set; an empty list means no tools at all. MCP tools are a process-wide
  /// concern (see [`crate::tools::ToolRegistry::with_mcp`]) and cannot be selected per
  /// request in this version of the API.
  #[serde(default)]
  pub tools: Option<Vec<String>>,
  /// Overrides the tool-round budget ([`crate::config::max_tool_rounds`]) for this call.
  #[serde(default)]
  pub max_steps: Option<u32>,
  /// Client-chosen id for a multi-turn conversation (see [`crate::api::session`]). Send
  /// the same value on every turn to keep talking to the same conversation; omit it for
  /// a one-off, stateless call (the previous, and still default, behavior).
  #[serde(default)]
  pub session_id: Option<String>,
}

/// Body of a successful `POST /v1/agent/run` response.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RunResponse {
  pub output: String,
  /// `true` when the round budget ran out before the model naturally stopped calling
  /// tools, i.e. this answer was forced from partial information. See
  /// [`crate::agent::AgentResult::budget_exhausted`].
  pub budget_exhausted: bool,
  pub usage: UsageDto,
  /// Tool rounds actually spent.
  pub steps: u32,
  /// Echoes [`RunRequest::session_id`] back when one was provided, so a client can
  /// confirm which conversation this answer was appended to.
  #[serde(skip_serializing_if = "Option::is_none")]
  pub session_id: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageDto {
  pub prompt_tokens: u32,
  pub completion_tokens: u32,
  pub total_tokens: u32,
}

impl From<TokenUsage> for UsageDto {
  fn from(usage: TokenUsage) -> Self {
    Self {
      prompt_tokens: usage.prompt_tokens,
      completion_tokens: usage.completion_tokens,
      total_tokens: usage.total_tokens,
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn run_request_only_requires_input() {
    let request: RunRequest = serde_json::from_str(r#"{"input": "hi"}"#).unwrap();
    assert_eq!(request.input, "hi");
    assert!(request.instructions.is_none());
    assert!(request.tools.is_none());
    assert!(request.session_id.is_none());
  }

  #[test]
  fn run_request_accepts_a_session_id() {
    let request: RunRequest =
      serde_json::from_str(r#"{"input": "hi", "sessionId": "conversation-1"}"#).unwrap();
    assert_eq!(request.session_id, Some("conversation-1".to_owned()));
  }

  #[test]
  fn run_response_serializes_as_camel_case() {
    let response = RunResponse {
      output: "42".to_owned(),
      budget_exhausted: false,
      usage: TokenUsage::default().into(),
      steps: 1,
      session_id: None,
    };

    let json = serde_json::to_value(&response).unwrap();
    assert_eq!(json["budgetExhausted"], false);
    assert_eq!(json["usage"]["totalTokens"], 0);
    assert!(json.get("sessionId").is_none(), "omitted when absent");
  }

  #[test]
  fn run_response_echoes_the_session_id_when_present() {
    let response = RunResponse {
      output: "42".to_owned(),
      budget_exhausted: false,
      usage: TokenUsage::default().into(),
      steps: 1,
      session_id: Some("conversation-1".to_owned()),
    };

    let json = serde_json::to_value(&response).unwrap();
    assert_eq!(json["sessionId"], "conversation-1");
  }
}
