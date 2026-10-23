use async_stream::stream;
use futures::{Stream, StreamExt};

use crate::{
  config,
  llm::{
    client::{WireMessages, build_messages, ensure_valid_params, request_builder},
    provider::Provider,
    retry::{is_transient, with_retry},
    settings::ModelSettings,
    tool_calls::ToolCallAccumulator,
    tool_loop::{append_tool_results, disable_tools},
  },
  tools::ToolRegistry,
};

/// Streaming completion, yielding incremental text segments.
///
/// Tool calls are handled transparently: a streamed call is reassembled from its fragments,
/// executed, and the conversation continues in a new stream, so the caller only ever sees
/// the text of the final answer. Text emitted *before* a tool call is forwarded as well,
/// since some models narrate what they are about to do. Reasoning, if the model streams
/// any, is not forwarded: this helper yields answer text only (see
/// [`crate::agent::AgentStreamEvent::Reasoning`] for a stream that carries it).
///
/// Same budget policy as [`crate::llm::tool_loop::run`]: when the round budget is spent,
/// the last stream goes out with tools disabled so an answer still comes back. `provider`
/// supplies the backend and concurrency slot for each stream request (see
/// [`Provider::chat_stream`]), held only while the stream for that one round is opened.
/// `settings` applies to every request, and with
/// [`ModelSettings::replay_reasoning`] on each round's reasoning is sent back with the
/// assistant message it belongs to.
fn chat_stream<'a>(
  provider: &'a Provider,
  model: &'a str,
  system: Option<&'a str>,
  prompt: &'a str,
  registry: &'a ToolRegistry,
  settings: &'a ModelSettings,
) -> impl Stream<Item = crate::Result<String>> + 'a {
  stream! {
    ensure_valid_params(model, prompt)?;

    let max_rounds = config::max_tool_rounds();
    let mut wire = WireMessages::from(build_messages(system, prompt)?);
    let mut round: usize = 0;

    loop {
      let tools_allowed = round < max_rounds;

      let mut builder = request_builder(model, wire.clone(), settings, registry.definitions());
      if !tools_allowed {
        disable_tools(&mut builder, registry.definitions());
        // Only worth warning about when tools were actually taken away.
        if !registry.is_empty() {
          tracing::warn!(max_rounds, "tool round budget spent; forcing a final answer");
        }
      }
      // Asked for so the round's last chunk carries its usage; without it the stream has
      // no token counts, and a caller running inside an agent's run goes unbilled.
      builder.stream_options(async_openai::types::chat::ChatCompletionStreamOptions {
        include_usage: Some(true),
        include_obfuscation: None,
      });
      let mut chunks = provider.chat_stream(builder.chat_request()?).await?;

      let mut accumulator = ToolCallAccumulator::default();
      // Kept so the assistant message we replay carries whatever the model said.
      let mut assistant_text = String::new();
      // Not forwarded, but kept when the provider wants it back (see
      // `ModelSettings::replay_reasoning`).
      let mut reasoning = String::new();

      while let Some(chunk) = chunks.next().await {
        let chunk = match chunk {
          Ok(chunk) => chunk,
          // Abort instead of continuing: after a transport error the remaining
          // fragments would assemble into a truncated tool call.
          Err(err) => {
            yield Err(err);
            return;
          }
        };
        if settings.replay_reasoning
          && let Some(text) = &chunk.reasoning
        {
          reasoning.push_str(text);
        }
        let chunk = chunk.chunk;

        // The chunk that carries usage is the one with no choices, so this comes first.
        // A no-op outside an agent's run; see `crate::agent::usage`.
        if let Some(spent) = &chunk.usage {
          crate::agent::usage::charge(model, spent.into());
        }

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

      if let Err(err) = append_tool_results(
        registry,
        &mut wire,
        tool_calls,
        Some(assistant_text),
        Some(reasoning),
      )
      .await
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
) -> crate::Result<String> {
  let settings = ModelSettings::from_config();
  with_retry(
    || async {
      let stream = chat_stream(provider, model, system, prompt, registry, &settings);
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
    },
    // A stream that failed deterministically (rejected key, unknown model, malformed
    // request) reopens to the identical failure, so only transient ones are retried.
    is_transient,
  )
  .await
}

#[cfg(test)]
mod tests {
  use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
  };

  use serde_json::{Value, json};

  use super::*;
  use crate::llm::backend::{ChatBackend, ChatChunkStream, ChatRequest, ChatResponse, parse_chunk};

  /// Streams the next scripted round, and records every request body as sent.
  struct ScriptedStreams {
    rounds: Mutex<VecDeque<Vec<Value>>>,
    seen: Mutex<Vec<Value>>,
  }

  #[async_trait::async_trait]
  impl ChatBackend for ScriptedStreams {
    async fn create(&self, _request: ChatRequest) -> crate::Result<ChatResponse> {
      crate::error::bail!(other, "not used")
    }

    async fn create_stream(&self, request: ChatRequest) -> crate::Result<ChatChunkStream> {
      self.seen.lock().unwrap().push(request.to_body()?);
      let chunks = self
        .rounds
        .lock()
        .unwrap()
        .pop_front()
        .expect("the test scripted a round for every request");
      Ok(Box::pin(futures::stream::iter(
        chunks.into_iter().map(parse_chunk),
      )))
    }
  }

  fn chunk(delta: Value) -> Value {
    json!({
      "id": "chatcmpl-1",
      "object": "chat.completion.chunk",
      "created": 0,
      "model": "gpt-test",
      "choices": [{ "index": 0, "delta": delta, "finish_reason": null }],
    })
  }

  /// A tool round with streamed reasoning, then an answer; returns the request bodies.
  async fn run_two_rounds(settings: ModelSettings) -> Vec<Value> {
    let backend = Arc::new(ScriptedStreams {
      rounds: Mutex::new(VecDeque::from([
        vec![
          chunk(json!({ "role": "assistant", "reasoning_content": "add " })),
          chunk(json!({ "reasoning_content": "first" })),
          chunk(json!({ "tool_calls": [{
            "index": 0, "id": "call_1", "type": "function",
            "function": {
              "name": "calculator",
              "arguments": r#"{"operator":"add","first_number":2,"second_number":3}"#,
            },
          }] })),
        ],
        vec![chunk(json!({ "content": "5" }))],
      ])),
      seen: Mutex::new(Vec::new()),
    });
    let provider = Provider::with_backend(Arc::clone(&backend) as Arc<dyn ChatBackend>, 1);
    let registry = ToolRegistry::select(&["calculator".to_owned()]).unwrap();

    let stream = chat_stream(
      &provider,
      "gpt-test",
      None,
      "add 2 and 3",
      &registry,
      &settings,
    );
    futures::pin_mut!(stream);
    let mut answer = String::new();
    while let Some(text) = stream.next().await {
      answer.push_str(&text.unwrap());
    }
    assert_eq!(answer, "5");

    backend.seen.lock().unwrap().clone()
  }

  /// A provider that demands reasoning back gets it on the assistant message that made the
  /// tool call — streamed reasoning is not forwarded to the caller, but it is not lost.
  #[tokio::test]
  async fn replayed_reasoning_is_assembled_from_the_stream_and_sent_back() {
    let seen = run_two_rounds(ModelSettings {
      replay_reasoning: true,
      ..ModelSettings::default()
    })
    .await;

    assert_eq!(seen.len(), 2);
    let messages = seen[1]["messages"].as_array().unwrap();
    assert_eq!(messages[1]["reasoning_content"], "add first");
    assert!(messages[2].get("reasoning_content").is_none());
  }

  #[tokio::test]
  async fn reasoning_is_not_sent_back_by_default() {
    let seen = run_two_rounds(ModelSettings::default()).await;

    assert!(
      !seen[1].to_string().contains("reasoning_content"),
      "got: {}",
      seen[1]
    );
  }

  /// Without `include_usage` the stream carries no token counts at all, so every round
  /// would cost this crate's accounting nothing.
  #[tokio::test]
  async fn every_requested_stream_asks_for_its_usage() {
    let seen = run_two_rounds(ModelSettings::default()).await;

    for body in &seen {
      assert_eq!(body["stream_options"]["include_usage"], true, "got: {body}");
    }
  }

  /// The usage a streamed round does report is billed to the run the stream is polled
  /// from, like every other request (see `crate::agent::usage`).
  #[tokio::test]
  async fn a_streamed_round_is_charged_to_the_enclosing_run() {
    let backend = Arc::new(ScriptedStreams {
      rounds: Mutex::new(VecDeque::from([vec![
        chunk(json!({ "content": "5" })),
        json!({
          "id": "chatcmpl-1",
          "object": "chat.completion.chunk",
          "created": 0,
          "model": "gpt-test",
          "choices": [],
          "usage": { "prompt_tokens": 40, "completion_tokens": 2, "total_tokens": 42 },
        }),
      ]])),
      seen: Mutex::new(Vec::new()),
    });
    let provider = Provider::with_backend(Arc::clone(&backend) as Arc<dyn ChatBackend>, 1);
    let registry = ToolRegistry::empty();
    let settings = ModelSettings::default();

    let ledger = crate::agent::usage::Ledger::for_new_run();
    ledger
      .scope(async {
        let stream = chat_stream(&provider, "gpt-test", None, "hi", &registry, &settings);
        futures::pin_mut!(stream);
        while let Some(text) = stream.next().await {
          text.unwrap();
        }
      })
      .await;

    let mut context = crate::agent::ExecutionContext::new();
    ledger.settle(&mut context);
    assert_eq!(context.usage_by_model["gpt-test"].prompt_tokens, 40);
    assert_eq!(context.usage.completion_tokens, 2);
  }
}
