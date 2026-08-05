use async_stream::stream;
use backon::{ExponentialBuilder, Retryable};
use futures::{Stream, StreamExt};

use crate::llm::client::{
  DEFAULT_MAX_TOKENS, build_messages, client, ensure_valid_params, request_builder,
};

/// Max retry attempts.
const MAX_RETRY_TIMES: usize = 3;

/// Streaming completion, yielding incremental text segments.
fn chat_stream(
  model: &str,
  system: Option<&str>,
  prompt: &str,
) -> impl Stream<Item = anyhow::Result<String>> {
  stream! {
    ensure_valid_params(model, prompt)?;

    let request = request_builder(model, build_messages(system, prompt)?, DEFAULT_MAX_TOKENS).build()?;
    let mut stream = client().chat().create_stream(request).await?;

    while let Some(chunk) = stream.next().await {
      match chunk {
        Ok(chunk) => {
          // Heartbeat / role-declaration chunks have empty delta.content; skip to avoid flooding downstream with empty strings.
          if let Some(choice) = chunk.choices.first()
            && let Some(text) = &choice.delta.content
            && !text.is_empty()
          {
            yield Ok(text.clone())
          }
        }
        Err(err) => yield Err(err.into()),
      }
    }
  }
}

/// Collect the full streaming output, retrying with exponential backoff on failure.
///
/// Note: a retry regenerates from scratch and discards already-emitted segments, so we accumulate
/// internally and only return on success, to avoid exposing partial content to the caller.
pub async fn chat_stream_with_retry(
  model: &str,
  system: Option<&str>,
  prompt: &str,
) -> anyhow::Result<String> {
  let op = || async {
    let stream = chat_stream(model, system, prompt);
    futures::pin_mut!(stream);

    let mut output = String::new();
    while let Some(result) = stream.next().await {
      match result {
        Ok(text) => output.push_str(&text),
        Err(err) => {
          tracing::error!("stream error: {err}");
          return Err(err);
        }
      }
    }
    Ok(output)
  };

  op.retry(ExponentialBuilder::default().with_max_times(MAX_RETRY_TIMES))
    .await
}
