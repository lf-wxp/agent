use std::{fs, path::Path};

use anyhow::Context;
use async_openai::types::chat::{
  ChatCompletionRequestMessageContentPartImageArgs,
  ChatCompletionRequestMessageContentPartTextArgs, ChatCompletionRequestUserMessageArgs,
  ChatCompletionRequestUserMessageContentPart::{self},
  CreateChatCompletionRequestArgs, ImageUrl,
};
use base64::Engine;

use crate::tools::read_image::ReadImageArgs;

/// Reads `file_path`, base64-encodes it into a data URL, and sends it alongside
/// `query` to `model` as a single-turn chat request; returns the model's text answer.
///
/// No size cap on the file: a large image inflates ~33% once base64-encoded and is
/// sent in full, so a very large input can mean a very large (and costly) request —
/// left to the caller to bound, same as any other tool argument coming from the model.
pub async fn run(args_json: &str) -> anyhow::Result<String> {
  let ReadImageArgs {
    file_path,
    query,
    model,
  } = serde_json::from_str::<ReadImageArgs>(args_json).context("invalid read image arguments")?;
  let path = Path::new(&file_path);
  let bytes = fs::read(path)?;

  let mime = match path.extension().and_then(|ext| ext.to_str()) {
    Some("png") => "image/png",
    Some("gif") => "image/gif",
    Some("webp") => "image/webp",
    _ => "image/jpeg",
  };

  let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
  let data_url = format!("data:{mime};base64,{encoded}");

  let message = ChatCompletionRequestUserMessageArgs::default()
    .content(vec![
      ChatCompletionRequestUserMessageContentPart::Text(
        ChatCompletionRequestMessageContentPartTextArgs::default()
          .text(query)
          .build()?,
      ),
      ChatCompletionRequestUserMessageContentPart::ImageUrl(
        ChatCompletionRequestMessageContentPartImageArgs::default()
          .image_url(ImageUrl {
            url: data_url,
            detail: None,
          })
          .build()?,
      ),
    ])
    .build()?;

  let request = CreateChatCompletionRequestArgs::default()
    .model(model)
    .messages(vec![message.into()])
    .max_tokens(1024u32)
    .build()?;

  let client = async_openai::Client::new();
  let response = client.chat().create(request).await?;

  response
    .choices
    .into_iter()
    .next()
    .and_then(|choice| choice.message.content)
    .ok_or_else(|| anyhow::anyhow!("No content in vision response"))
}

#[cfg(test)]
mod tests {
  use super::*;

  /// The model/network call is never reached: parsing fails before any request goes out.
  #[tokio::test]
  async fn rejects_malformed_json() {
    let err = run("not json").await.unwrap_err();
    assert!(err.to_string().contains("invalid read image arguments"));
  }

  /// Same reasoning: a missing file fails at `fs::read` before any network call.
  #[tokio::test]
  async fn rejects_missing_file() {
    let path =
      std::env::temp_dir().join(format!("agent-test-missing-{}.png", uuid::Uuid::new_v4()));
    let args = format!(
      r#"{{"file_path":"{}","query":"what is this?","model":"gpt-4o-mini"}}"#,
      path.display()
    );
    assert!(run(&args).await.is_err());
  }
}
