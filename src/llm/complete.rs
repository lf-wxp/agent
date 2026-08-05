use async_openai::types::chat::ChatCompletionTools;

use crate::llm::{
  client::{DEFAULT_MAX_TOKENS, build_messages, ensure_valid_params},
  tool_loop,
};

/// Plain text completion; executes tool calls when the model requests them.
///
/// The tool round budget being exhausted is only logged here (see
/// [`crate::llm::tool_loop::Completion::budget_exhausted`]); use the structured API when
/// the caller needs to react to it programmatically.
pub async fn chat_complete(
  model: &str,
  system: Option<&str>,
  prompt: &str,
  tools: &[ChatCompletionTools],
) -> anyhow::Result<String> {
  ensure_valid_params(model, prompt)?;

  let completion = tool_loop::run(
    model,
    build_messages(system, prompt)?,
    tools,
    DEFAULT_MAX_TOKENS,
    None,
  )
  .await?;

  completion
    .choice
    .message
    .content
    .filter(|content| !content.trim().is_empty())
    .ok_or_else(|| anyhow::anyhow!("No content in response"))
}
