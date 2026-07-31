use async_openai::types::chat::{
  ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
  CreateChatCompletionRequestArgs,
};

pub async fn chat_complete(
  model: &str,
  system: Option<&str>,
  prompt: &str,
) -> anyhow::Result<String> {
  let client = async_openai::Client::new();
  let mut messages = vec![];
  if let Some(system) = system {
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
  let request = CreateChatCompletionRequestArgs::default()
    .model(model)
    .messages(messages)
    .max_tokens(2048u32)
    .build()?;
  let response = client.chat().create(request).await?;

  tracing::info!("Response {:#?}", response);

  Ok(String::new())
}
