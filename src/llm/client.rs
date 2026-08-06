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
