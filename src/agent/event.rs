use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentItem {
  #[serde(rename = "message")]
  Message { role: String, content: String },

  #[serde(rename = "tool_call")]
  ToolCall {
    tool_call_id: String,
    name: String,
    arguments: Value,
  },

  /// The chain of thought behind the assistant message that follows it.
  ///
  /// Recorded only when [`crate::llm::settings::ModelSettings::replay_reasoning`] is on,
  /// so the default transcript is unchanged. Providers that require it back (DeepSeek's
  /// thinking mode does, whenever the request carries tools) get it as
  /// `reasoning_content` on that assistant message; see
  /// [`crate::agent::Agent::with_model_settings`]. It belongs to the *next* item, so
  /// anything that cuts a transcript apart has to keep the two together.
  #[serde(rename = "reasoning")]
  Reasoning { text: String },

  #[serde(rename = "tool_result")]
  ToolResult {
    tool_call_id: String,
    name: String,
    status: ToolResultStatus,
    content: String,
  },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolResultStatus {
  Success,
  Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
  pub id: String,
  pub execution_id: String,
  pub timestamp: i64,
  pub author: String,
  pub content: Vec<ContentItem>,
}

impl Event {
  pub fn new(
    execution_id: impl Into<String>,
    author: impl Into<String>,
    content: Vec<ContentItem>,
  ) -> Self {
    Self {
      id: Uuid::new_v4().to_string(),
      execution_id: execution_id.into(),
      timestamp: chrono::Utc::now().timestamp(),
      author: author.into(),
      content,
    }
  }
}
