use crate::llm::client::{
  DEFAULT_MAX_TOKENS, build_messages, client, ensure_valid_params, first_choice, log_response_meta,
  request_builder,
};

/// Plain text completion.
pub async fn chat_complete(
  model: &str,
  system: Option<&str>,
  prompt: &str,
) -> anyhow::Result<String> {
  ensure_valid_params(model, prompt)?;

  let request =
    request_builder(model, build_messages(system, prompt)?, DEFAULT_MAX_TOKENS).build()?;
  let response = client().chat().create(request).await?;
  log_response_meta(&response, "chat completion finished");

  first_choice(response)?
    .message
    .content
    .filter(|content| !content.trim().is_empty())
    .ok_or_else(|| anyhow::anyhow!("No content in response"))
}
