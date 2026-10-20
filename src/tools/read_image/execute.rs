use std::path::Path;

use anyhow::Context;
use async_openai::types::chat::{
  ChatCompletionRequestMessage, ChatCompletionRequestMessageContentPartImageArgs,
  ChatCompletionRequestMessageContentPartTextArgs, ChatCompletionRequestUserMessageArgs,
  ChatCompletionRequestUserMessageContentPart::{self},
  ImageUrl,
};
use base64::Engine;

use crate::{
  llm::{
    client::{first_choice, request_builder},
    provider::Provider,
    retry::{is_transient, with_retry},
    settings::ModelSettings,
  },
  tools::read_image::ReadImageArgs,
};

/// Reads `file_path`, base64-encodes it into a data URL, and sends it alongside
/// `query` to `model` through `provider` as a single-turn chat request; returns the
/// model's text answer.
///
/// The request carries the configured output cap only (see [`ModelSettings::cap_only`]),
/// and transient failures are retried like any other model call.
///
/// No size cap on the file: a large image inflates ~33% once base64-encoded and is
/// sent in full, so a very large input can mean a very large (and costly) request —
/// left to the caller to bound, same as any other tool argument coming from the model.
pub async fn run(provider: &Provider, args_json: &str) -> anyhow::Result<String> {
  let ReadImageArgs {
    file_path,
    query,
    model,
  } = serde_json::from_str::<ReadImageArgs>(args_json).context("invalid read image arguments")?;
  let path = Path::new(&file_path);
  let bytes = tokio::fs::read(path)
    .await
    .with_context(|| format!("failed to read `{file_path}`"))?;

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

  // `model` comes from the caller's arguments, not from whatever the configured settings
  // were written for, so only the output cap goes along: sampling or a reasoning effort
  // the vision model does not support would fail every call.
  let builder = request_builder(
    &model,
    vec![ChatCompletionRequestMessage::from(message)],
    &ModelSettings::from_config().cap_only(),
    &[],
  );
  let response = with_retry(
    || async { provider.chat(builder.chat_request()?).await },
    is_transient,
  )
  .await?
  .response;
  // Billed to the run that called the tool (see `crate::agent::usage`).
  if let Some(spent) = &response.usage {
    crate::agent::usage::charge(&model, spent.into());
  }

  first_choice(response)?
    .message
    .content
    .filter(|content| !content.trim().is_empty())
    .ok_or_else(|| anyhow::anyhow!("No content in vision response"))
}

#[cfg(test)]
mod tests {
  use std::sync::{Arc, Mutex};

  use serde_json::{Value, json};

  use super::*;
  use crate::llm::backend::{
    ChatBackend, ChatChunkStream, ChatRequest, ChatResponse, parse_response,
  };

  /// Records the request it was sent and answers with a fixed text.
  #[derive(Default)]
  struct VisionStub {
    seen: Mutex<Vec<Value>>,
  }

  #[async_trait::async_trait]
  impl ChatBackend for VisionStub {
    async fn create(&self, request: ChatRequest) -> anyhow::Result<ChatResponse> {
      self.seen.lock().unwrap().push(request.to_body()?);
      parse_response(json!({
        "id": "x", "object": "chat.completion", "created": 0, "model": "vision",
        "choices": [{ "index": 0, "finish_reason": "stop",
                      "message": { "role": "assistant", "content": "a red square" } }],
      }))
    }

    async fn create_stream(&self, _request: ChatRequest) -> anyhow::Result<ChatChunkStream> {
      anyhow::bail!("not used")
    }
  }

  fn stub_provider() -> (Provider, Arc<VisionStub>) {
    let stub = Arc::new(VisionStub::default());
    let provider = Provider::with_backend(Arc::clone(&stub) as Arc<dyn ChatBackend>, 1);
    (provider, stub)
  }

  /// The model/network call is never reached: parsing fails before any request goes out.
  #[tokio::test]
  async fn rejects_malformed_json() {
    let (provider, stub) = stub_provider();
    let err = run(&provider, "not json").await.unwrap_err();
    assert!(err.to_string().contains("invalid read image arguments"));
    assert!(stub.seen.lock().unwrap().is_empty());
  }

  /// Same reasoning: a missing file fails before any network call.
  #[tokio::test]
  async fn rejects_missing_file() {
    let (provider, stub) = stub_provider();
    let path =
      std::env::temp_dir().join(format!("agent-test-missing-{}.png", uuid::Uuid::new_v4()));
    let args = json!({ "file_path": path, "query": "what is this?", "model": "vision" });
    assert!(run(&provider, &args.to_string()).await.is_err());
    assert!(stub.seen.lock().unwrap().is_empty());
  }

  /// The point of routing through a `Provider`: the request reaches *that* provider's
  /// backend, carrying the image inline and the model the caller named.
  #[tokio::test]
  async fn sends_the_image_through_the_given_provider() {
    let (provider, stub) = stub_provider();
    let path = std::env::temp_dir().join(format!("agent-test-image-{}.png", uuid::Uuid::new_v4()));
    std::fs::write(&path, b"\x89PNG fake bytes").unwrap();

    let args = json!({ "file_path": path, "query": "what color?", "model": "vision" });
    let outcome = run(&provider, &args.to_string()).await;
    // Removed before anything can fail the test, so a failure does not leave it behind.
    let _ = std::fs::remove_file(&path);
    let answer = outcome.unwrap();

    assert_eq!(answer, "a red square");
    let seen = stub.seen.lock().unwrap();
    assert_eq!(seen.len(), 1);
    let request = &seen[0];
    assert_eq!(request["model"], "vision");
    // The output cap goes along; nothing the vision model might reject does.
    assert!(request.get("max_tokens").is_some(), "got: {request}");
    for absent in [
      "temperature",
      "top_p",
      "reasoning_effort",
      "reasoning_content",
    ] {
      assert!(
        request.get(absent).is_none(),
        "`{absent}` was sent: {request}"
      );
    }
    let parts = &request["messages"][0]["content"];
    assert_eq!(parts[0]["text"], "what color?");
    assert!(
      parts[1]["image_url"]["url"]
        .as_str()
        .unwrap()
        .starts_with("data:image/png;base64,"),
      "got: {parts}"
    );
  }
}
