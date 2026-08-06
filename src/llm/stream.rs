use async_stream::stream;
use backon::{ExponentialBuilder, Retryable};
use futures::{Stream, StreamExt};

use crate::{
  config,
  llm::{
    client::{DEFAULT_MAX_TOKENS, build_messages, ensure_valid_params, request_builder},
    provider::Provider,
    tool_calls::ToolCallAccumulator,
    tool_loop::{append_tool_results, disable_tools},
  },
  tools::ToolRegistry,
};

/// Max retry attempts.
const MAX_RETRY_TIMES: usize = 3;

/// Streaming completion, yielding incremental text segments.
///
/// Tool calls are handled transparently: a streamed call is reassembled from its fragments,
/// executed, and the conversation continues in a new stream, so the caller only ever sees
/// the text of the final answer. Text emitted *before* a tool call is forwarded as well,
/// since some models narrate what they are about to do.
///
/// Same budget policy as [`crate::llm::tool_loop::run`]: when the round budget is spent,
/// the last stream goes out with tools disabled so an answer still comes back. `provider`
/// supplies the credentials and concurrency slot for each stream request (see
/// [`Provider::acquire`]), held only while the stream for that one round is open.
fn chat_stream<'a>(
  provider: &'a Provider,
  model: &'a str,
  system: Option<&'a str>,
  prompt: &'a str,
  registry: &'a ToolRegistry,
) -> impl Stream<Item = anyhow::Result<String>> + 'a {
  stream! {
    ensure_valid_params(model, prompt)?;

    let max_rounds = config::max_tool_rounds();
    let mut messages = build_messages(system, prompt)?;
    let mut round: usize = 0;

    loop {
      let tools_allowed = round < max_rounds;

      let mut builder =
        request_builder(model, messages.clone(), DEFAULT_MAX_TOKENS, registry.definitions());
      if !tools_allowed {
        disable_tools(&mut builder, registry.definitions());
        // Only worth warning about when tools were actually taken away.
        if !registry.is_empty() {
          tracing::warn!(max_rounds, "tool round budget spent; forcing a final answer");
        }
      }
      let mut chunks = {
        let _permit = provider.acquire().await?;
        provider.client().chat().create_stream(builder.build()?).await?
      };

      let mut accumulator = ToolCallAccumulator::default();
      // Kept so the assistant message we replay carries whatever the model said.
      let mut assistant_text = String::new();

      while let Some(chunk) = chunks.next().await {
        let chunk = match chunk {
          Ok(chunk) => chunk,
          // Abort instead of continuing: after a transport error the remaining
          // fragments would assemble into a truncated tool call.
          Err(err) => {
            yield Err(err.into());
            return;
          }
        };

        let Some(choice) = chunk.choices.first() else {
          continue;
        };

        if let Some(fragments) = &choice.delta.tool_calls {
          accumulator.push(fragments);
        }

        // Heartbeat / role-declaration chunks have empty delta.content; skip to avoid
        // flooding downstream with empty strings.
        if let Some(text) = &choice.delta.content
          && !text.is_empty()
        {
          assistant_text.push_str(text);
          yield Ok(text.clone());
        }
      }

      // No tool calls means the model produced its final answer.
      if accumulator.is_empty() {
        return;
      }

      // The model ignored `tool_choice = none`; its text has already been forwarded,
      // so end the stream rather than spending another round.
      if !tools_allowed {
        tracing::warn!("model requested tools after they were disabled");
        return;
      }
      round += 1;

      let tool_calls = match accumulator.finish() {
        Ok(tool_calls) => tool_calls,
        Err(err) => {
          yield Err(err);
          return;
        }
      };

      if let Err(err) =
        append_tool_results(registry, &mut messages, tool_calls, Some(assistant_text)).await
      {
        yield Err(err);
        return;
      }
    }
  }
}

/// Collect the full streaming output, retrying with exponential backoff on failure.
///
/// Note: a retry regenerates from scratch and discards already-emitted segments, so we accumulate
/// internally and only return on success, to avoid exposing partial content to the caller.
///
/// `provider` selects which tenant's credentials and concurrency budget the request is
/// charged against; pass [`Provider::shared`] for the single-tenant default.
pub async fn chat_stream_with_retry(
  provider: &Provider,
  model: &str,
  system: Option<&str>,
  prompt: &str,
  registry: &ToolRegistry,
) -> anyhow::Result<String> {
  let op = || async {
    let stream = chat_stream(provider, model, system, prompt, registry);
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
