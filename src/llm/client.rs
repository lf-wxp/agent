//! Common construction layer for LLM requests: message assembly and request skeleton.
//!
//! `complete` / `stream` / `structured` differ only in "extra request parameters" and "response
//! parsing"; the common parts are unified here. The client itself now lives on
//! [`crate::llm::provider::Provider`] (per-tenant), not here.

use async_openai::types::chat::{
  ChatChoice, ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
  ChatCompletionRequestUserMessageArgs, ChatCompletionTools, CreateChatCompletionRequestArgs,
  CreateChatCompletionResponse,
};

/// Output token budget for normal conversations.
///
/// Structured / reasoning scenarios need a larger budget; see `structured::NATIVE_SCHEMA_MAX_TOKENS`.
pub const DEFAULT_MAX_TOKENS: u32 = 2048;

/// Empty parameters cause a request that is guaranteed to fail; intercept early.
pub fn ensure_valid_params(model: &str, prompt: &str) -> anyhow::Result<()> {
  anyhow::ensure!(!model.trim().is_empty(), "model must not be empty");
  anyhow::ensure!(!prompt.trim().is_empty(), "prompt must not be empty");
  Ok(())
}

/// Assemble the `[system?, user]` message sequence; a blank system is treated as not provided.
pub fn build_messages(
  system: Option<&str>,
  prompt: &str,
) -> anyhow::Result<Vec<ChatCompletionRequestMessage>> {
  let mut messages = Vec::with_capacity(2);
  if let Some(system) = system.map(str::trim).filter(|s| !s.is_empty()) {
    messages.push(
      ChatCompletionRequestSystemMessageArgs::default()
        .content(system)
        .build()?
        .into(),
    );
  }
  messages.push(
    ChatCompletionRequestUserMessageArgs::default()
      .content(prompt)
      .build()?
      .into(),
  );
  Ok(messages)
}

/// Request builder preloaded with model / messages / max_tokens; callers can keep appending params before `build()`.
///
/// An empty `tools` slice is left unset rather than sent as `"tools": []`: the API
/// requires at least one entry when the field is present, so an empty array is rejected.
pub fn request_builder(
  model: &str,
  messages: Vec<ChatCompletionRequestMessage>,
  max_tokens: u32,
  tools: &[ChatCompletionTools],
) -> CreateChatCompletionRequestArgs {
  let mut args = CreateChatCompletionRequestArgs::default();
  args.model(model).messages(messages).max_tokens(max_tokens);
  if !tools.is_empty() {
    args.tools(tools.to_vec());
  }
  args
}

/// Take the first choice from the response.
pub fn first_choice(response: CreateChatCompletionResponse) -> anyhow::Result<ChatChoice> {
  response
    .choices
    .into_iter()
    .next()
    .ok_or_else(|| anyhow::anyhow!("No choices in response"))
}

/// The full response can be long; only log metadata to help inspect usage and trace id.
pub fn log_response_meta(response: &CreateChatCompletionResponse, label: &'static str) {
  tracing::debug!(id = %response.id, usage = ?response.usage, "{label}");
}

#[cfg(test)]
mod tests {
  use serde_json::json;

  use super::*;

  /// Deserializing from JSON avoids touching `CreateChatCompletionResponse`'s
  /// `#[deprecated]` `system_fingerprint` field, which a struct literal would have to
  /// name explicitly.
  fn response_with_choices(choices: usize) -> CreateChatCompletionResponse {
    serde_json::from_value(json!({
      "id": "resp_1",
      "object": "chat.completion",
      "created": 0,
      "model": "gpt-test",
      "choices": (0..choices)
        .map(|index| json!({
          "index": index,
          "message": {"role": "assistant", "content": format!("choice {index}")},
          "finish_reason": "stop",
        }))
        .collect::<Vec<_>>(),
    }))
    .unwrap()
  }

  #[test]
  fn ensure_valid_params_accepts_non_blank_model_and_prompt() {
    assert!(ensure_valid_params("gpt-test", "hi").is_ok());
  }

  #[test]
  fn ensure_valid_params_rejects_blank_model() {
    assert!(ensure_valid_params("  ", "hi").is_err());
  }

  #[test]
  fn ensure_valid_params_rejects_blank_prompt() {
    assert!(ensure_valid_params("gpt-test", "   ").is_err());
  }

  #[test]
  fn build_messages_without_system_is_just_the_user_message() {
    let messages = build_messages(None, "hi").unwrap();
    assert_eq!(messages.len(), 1);
    assert!(matches!(messages[0], ChatCompletionRequestMessage::User(_)));
  }

  #[test]
  fn build_messages_treats_blank_system_as_absent() {
    let messages = build_messages(Some("   "), "hi").unwrap();
    assert_eq!(messages.len(), 1);
  }

  #[test]
  fn build_messages_with_system_prepends_it() {
    let messages = build_messages(Some("be nice"), "hi").unwrap();
    assert_eq!(messages.len(), 2);
    assert!(matches!(
      messages[0],
      ChatCompletionRequestMessage::System(_)
    ));
    assert!(matches!(messages[1], ChatCompletionRequestMessage::User(_)));
  }

  #[test]
  fn request_builder_omits_empty_tools_rather_than_sending_an_empty_array() {
    let request = request_builder("gpt-test", Vec::new(), 16, &[])
      .build()
      .unwrap();
    assert!(request.tools.is_none());
  }

  #[test]
  fn request_builder_carries_the_given_tools() {
    let registry = crate::tools::ToolRegistry::builtin().unwrap();
    let request = request_builder("gpt-test", Vec::new(), 16, registry.definitions())
      .build()
      .unwrap();
    let tools = request.tools.unwrap();
    assert_eq!(tools.len(), registry.definitions().len());
  }

  #[test]
  fn request_builder_sets_model_messages_and_max_tokens() {
    let messages = build_messages(None, "hi").unwrap();
    let request = request_builder("gpt-test", messages, 16, &[])
      .build()
      .unwrap();
    assert_eq!(request.model, "gpt-test");
    assert_eq!(request.messages.len(), 1);
    // `max_tokens` itself is `#[deprecated]` in favor of `max_completion_tokens` (the
    // crate's field, not this crate's choice); `request_builder` still targets it because
    // that is what every provider this agent has been run against actually honors.
    #[allow(deprecated)]
    let max_tokens = request.max_tokens;
    assert_eq!(max_tokens, Some(16));
  }

  #[test]
  fn first_choice_returns_the_first_of_several() {
    let response = response_with_choices(2);
    let choice = first_choice(response).unwrap();
    assert_eq!(choice.index, 0);
  }

  #[test]
  fn first_choice_errors_when_there_are_none() {
    let response = response_with_choices(0);
    assert!(first_choice(response).is_err());
  }

  #[test]
  fn log_response_meta_does_not_panic() {
    // No assertion beyond "does not panic": this only writes a tracing event.
    log_response_meta(&response_with_choices(1), "test");
  }
}
