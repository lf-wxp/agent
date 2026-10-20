//! [`ChatBackend`]: the seam between this crate and whatever actually serves chat
//! completions.
//!
//! Everything above this line — [`crate::agent::Agent`], the stateless helpers in
//! [`crate::llm`], summarization — talks to a [`crate::llm::provider::Provider`], which
//! talks to one of these. The wire format at the seam is the OpenAI Chat Completions
//! shape (`async-openai`'s request/response types): it is the lingua franca nearly every
//! provider speaks natively or through a compatibility endpoint, so it is what an
//! adapter for one that does not (a native Anthropic or Gemini client, a local runtime,
//! a test double) translates to and from.
//!
//! [`OpenAiCompatible`] is the implementation that ships: one `async-openai` client per
//! tenant, pointed at any compatible base URL (OpenAI, DeepSeek, Qwen, vLLM, Ollama,
//! OpenRouter, Gemini's and Anthropic's compatibility endpoints, ...).
//!
//! # Reasoning
//!
//! The standard response types have no field for a model's chain of thought, but most
//! providers that expose one put it next to the content, as `reasoning_content`
//! (DeepSeek, Qwen, Kimi, SiliconFlow, vLLM) or `reasoning` (OpenRouter, newer vLLM,
//! Ollama). [`OpenAiCompatible`] therefore reads the raw JSON, lifts whichever is
//! present into [`ChatResponse::reasoning`] / [`ChatChunk::reasoning`], and only then
//! parses the standard shape.
//!
//! Whether it goes back is the provider's call, and they differ: DeepSeek's thinking mode
//! *requires* it on earlier assistant messages whenever the request carries tools (400
//! otherwise), while OpenAI and others reject or ignore the field. So by default it is
//! not sent; with
//! [`ModelSettings::replay_reasoning`](crate::llm::settings::ModelSettings::replay_reasoning)
//! on, the caller hands it back in [`ChatRequest::reasoning`] and [`OpenAiCompatible`]
//! writes it as `reasoning_content` on the assistant message it belongs to.

use std::pin::Pin;

use anyhow::Context;
use async_openai::{
  Client,
  config::OpenAIConfig,
  types::chat::{
    CreateChatCompletionRequest, CreateChatCompletionResponse, CreateChatCompletionStreamResponse,
  },
};
use futures::{Stream, StreamExt};
use serde::Deserialize;
use serde_json::Value;

use crate::util::truncate_chars;

/// Chars of a response body quoted in a "wrong shape" error: enough to see what came
/// back, not enough to dump a whole completion into a log.
const BODY_PREVIEW_CHARS: usize = 300;

/// The request field a replayed chain of thought is written to.
const REPLAY_FIELD: &str = "reasoning_content";

/// One request: the standard body, plus the reasoning to send back with earlier
/// assistant messages.
///
/// The typed assistant message has no field for reasoning, so it travels beside the
/// request instead of inside it.
#[derive(Debug, Clone)]
pub struct ChatRequest {
  pub request: CreateChatCompletionRequest,
  /// `reasoning[i]` is the chain of thought behind `request.messages[i]` (an assistant
  /// message), to be sent as `reasoning_content`. May be shorter than `messages`, in
  /// which case the rest carry none. A backend that has nowhere to put it ignores it.
  pub reasoning: Vec<Option<String>>,
}

impl From<CreateChatCompletionRequest> for ChatRequest {
  /// A request with nothing to replay.
  fn from(request: CreateChatCompletionRequest) -> Self {
    Self {
      request,
      reasoning: Vec::new(),
    }
  }
}

impl ChatRequest {
  /// The JSON body that goes on the wire: the standard request, with each replayed
  /// chain of thought added to its assistant message as `reasoning_content`.
  pub fn to_body(&self) -> anyhow::Result<Value> {
    let mut body = serde_json::to_value(&self.request)
      .context("the chat completion request could not be serialized")?;
    if let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) {
      for (message, text) in messages.iter_mut().zip(&self.reasoning) {
        let (Some(text), Some(object)) = (text, message.as_object_mut()) else {
          continue;
        };
        if object.get("role").and_then(Value::as_str) == Some("assistant") {
          object.insert(REPLAY_FIELD.to_owned(), Value::String(text.clone()));
        }
      }
    }
    Ok(body)
  }
}

/// One complete response, plus the reasoning behind its first choice when the backend
/// exposes it.
#[derive(Debug, Clone)]
pub struct ChatResponse {
  pub response: CreateChatCompletionResponse,
  /// The first choice's chain of thought. `None` for a model that does not reason, or a
  /// provider that does not return it.
  pub reasoning: Option<String>,
}

/// One streamed chunk, plus the reasoning text it carries for the first choice.
#[derive(Debug, Clone)]
pub struct ChatChunk {
  pub chunk: CreateChatCompletionStreamResponse,
  /// The next piece of the first choice's chain of thought, if this chunk carries one.
  /// Reasoning streams ahead of the answer, so a chunk carries one or the other.
  pub reasoning: Option<String>,
}

/// What [`ChatBackend::create_stream`] yields: chunks until the response is complete,
/// or an error that ends it.
pub type ChatChunkStream = Pin<Box<dyn Stream<Item = anyhow::Result<ChatChunk>> + Send>>;

/// Something that answers chat completion requests.
///
/// Implementations should surface a provider's own error type inside the
/// [`anyhow::Error`] where one exists (`?` does this), because retry decisions downcast
/// it: [`crate::llm::retry::is_transient`] recognizes
/// [`async_openai::error::OpenAIError`] and treats everything else as retryable. A
/// backend with errors of its own that retrying cannot fix (a rejected key, a malformed
/// request) marks them with [`crate::llm::retry::Permanent::wrap`], or each one costs
/// the whole backoff curve before it surfaces.
///
/// No concurrency control belongs here: that is per tenant and lives on
/// [`crate::llm::provider::Provider`], which wraps every call.
#[async_trait::async_trait]
pub trait ChatBackend: Send + Sync {
  /// One complete response to `request`. `request.request.stream` is not meaningful here
  /// and should be ignored.
  async fn create(&self, request: ChatRequest) -> anyhow::Result<ChatResponse>;

  /// Open a streamed response to `request`. Returns once the stream is open; the
  /// chunks themselves arrive through the returned stream.
  async fn create_stream(&self, request: ChatRequest) -> anyhow::Result<ChatChunkStream>;
}

/// Any server speaking the OpenAI Chat Completions API, reached through `async-openai`.
#[derive(Debug, Clone)]
pub struct OpenAiCompatible {
  client: Client<OpenAIConfig>,
}

impl OpenAiCompatible {
  /// A client for the endpoint and credentials in `config`.
  pub fn new(config: OpenAIConfig) -> Self {
    Self {
      client: Client::with_config(config),
    }
  }
}

#[async_trait::async_trait]
impl ChatBackend for OpenAiCompatible {
  async fn create(&self, request: ChatRequest) -> anyhow::Result<ChatResponse> {
    // The raw (`_byot`) variants skip `async-openai`'s own check that `stream` matches
    // the method, so it is set here instead.
    let mut body = request.to_body()?;
    if let Some(object) = body.as_object_mut() {
      object.remove("stream");
    }
    let raw: Value = self.client.chat().create_byot(body).await?;
    parse_response(raw)
  }

  async fn create_stream(&self, request: ChatRequest) -> anyhow::Result<ChatChunkStream> {
    // Same reason as above: the raw variant does not set this itself.
    let mut body = request.to_body()?;
    if let Some(object) = body.as_object_mut() {
      object.insert("stream".to_owned(), Value::Bool(true));
    }
    let raw = self
      .client
      .chat()
      .create_stream_byot::<_, Value>(body)
      .await?;
    Ok(Box::pin(raw.map(|item| {
      item.map_err(anyhow::Error::from).and_then(parse_chunk)
    })))
  }
}

/// Where providers put a choice's chain of thought, in the order they are checked.
const REASONING_FIELDS: [&str; 2] = ["reasoning_content", "reasoning"];

/// Split a raw response body into the standard response and its reasoning.
pub fn parse_response(mut raw: Value) -> anyhow::Result<ChatResponse> {
  let reasoning = reasoning_in(&raw, "message");
  normalize_usage(&mut raw);
  // Deserialized from a borrow, not by value, so the body is still there to quote when
  // it does not fit: a bare serde message ("missing field `id`") says nothing about what
  // the provider actually sent.
  let response = CreateChatCompletionResponse::deserialize(&raw).with_context(|| {
    format!(
      "the chat completion response has an unexpected shape: {}",
      body_preview(&raw)
    )
  })?;
  Ok(ChatResponse {
    response,
    reasoning,
  })
}

/// Split a raw stream chunk into the standard chunk and its reasoning delta.
pub fn parse_chunk(mut raw: Value) -> anyhow::Result<ChatChunk> {
  let reasoning = reasoning_in(&raw, "delta");
  normalize_usage(&mut raw);
  let chunk = CreateChatCompletionStreamResponse::deserialize(&raw).with_context(|| {
    format!(
      "a chat completion stream chunk has an unexpected shape: {}",
      body_preview(&raw)
    )
  })?;
  Ok(ChatChunk { chunk, reasoning })
}

/// Put the provider's own name for a prompt-cache hit where the standard shape keeps it.
///
/// DeepSeek reports it as `usage.prompt_cache_hit_tokens` — a top-level field, and the
/// only place it appears in responses from before it also sent the OpenAI-compatible
/// `prompt_tokens_details` — while the standard shape, and so `TokenUsage`, reads
/// `usage.prompt_tokens_details.cached_tokens`. Without this the hits are reported as
/// ordinary prompt tokens, i.e. the run is shown as costing more than it did.
fn normalize_usage(raw: &mut Value) {
  let Some(usage) = raw.get_mut("usage").and_then(Value::as_object_mut) else {
    return;
  };
  let Some(hits) = usage.get("prompt_cache_hit_tokens").cloned() else {
    return;
  };
  let details = usage
    .entry("prompt_tokens_details")
    .or_insert_with(|| Value::Object(Default::default()));
  if let Some(details) = details.as_object_mut() {
    details.entry("cached_tokens").or_insert(hits);
  }
}

fn body_preview(raw: &Value) -> String {
  truncate_chars(&raw.to_string(), BODY_PREVIEW_CHARS).to_owned()
}

/// The first choice's reasoning text under `part` (`message` or `delta`), if any.
///
/// Only a string counts: a provider that sends structured reasoning (an object or array)
/// is ignored rather than rendered as JSON into a field documented as text. An empty
/// string is no reasoning, which is what most providers send on chunks that carry the
/// answer instead.
fn reasoning_in(raw: &Value, part: &str) -> Option<String> {
  let part = raw.get("choices")?.get(0)?.get(part)?;
  REASONING_FIELDS
    .iter()
    .filter_map(|field| part.get(field).and_then(Value::as_str))
    .find(|text| !text.is_empty())
    .map(str::to_owned)
}

#[cfg(test)]
mod tests {
  use serde_json::json;

  use super::*;

  fn response_with(message: Value) -> Value {
    json!({
      "id": "chatcmpl-1",
      "object": "chat.completion",
      "created": 0,
      "model": "gpt-test",
      "choices": [{ "index": 0, "message": message, "finish_reason": "stop" }],
    })
  }

  fn chunk_with(delta: Value) -> Value {
    json!({
      "id": "chatcmpl-1",
      "object": "chat.completion.chunk",
      "created": 0,
      "model": "gpt-test",
      "choices": [{ "index": 0, "delta": delta, "finish_reason": null }],
    })
  }

  #[test]
  fn lifts_reasoning_content_out_of_a_response() {
    let parsed = parse_response(response_with(json!({
      "role": "assistant",
      "content": "42",
      "reasoning_content": "6 times 7",
    })))
    .unwrap();

    assert_eq!(parsed.reasoning.as_deref(), Some("6 times 7"));
    assert_eq!(
      parsed.response.choices[0].message.content.as_deref(),
      Some("42"),
      "the standard fields still parse"
    );
  }

  /// OpenRouter and newer vLLM name the field `reasoning`.
  #[test]
  fn also_reads_the_reasoning_field_name() {
    let parsed = parse_response(response_with(json!({
      "role": "assistant",
      "content": "42",
      "reasoning": "thinking",
    })))
    .unwrap();
    assert_eq!(parsed.reasoning.as_deref(), Some("thinking"));
  }

  #[test]
  fn a_plain_response_has_no_reasoning() {
    let parsed = parse_response(response_with(json!({
      "role": "assistant",
      "content": "42",
    })))
    .unwrap();
    assert!(parsed.reasoning.is_none());
  }

  /// Structured reasoning is not text; rendering it as JSON into a text field would be
  /// worse than leaving it out.
  #[test]
  fn non_string_reasoning_is_ignored() {
    let parsed = parse_response(response_with(json!({
      "role": "assistant",
      "content": "42",
      "reasoning": { "effort": "high" },
    })))
    .unwrap();
    assert!(parsed.reasoning.is_none());
  }

  #[test]
  fn lifts_a_reasoning_delta_out_of_a_chunk() {
    let parsed = parse_chunk(chunk_with(json!({ "reasoning_content": "step 1" }))).unwrap();
    assert_eq!(parsed.reasoning.as_deref(), Some("step 1"));
    assert!(parsed.chunk.choices[0].delta.content.is_none());
  }

  /// Providers send `"reasoning_content": ""` on every answer chunk once thinking ends.
  #[test]
  fn an_empty_reasoning_delta_is_no_reasoning() {
    let parsed = parse_chunk(chunk_with(json!({
      "content": "answer",
      "reasoning_content": "",
    })))
    .unwrap();
    assert!(parsed.reasoning.is_none());
    assert_eq!(
      parsed.chunk.choices[0].delta.content.as_deref(),
      Some("answer")
    );
  }

  /// The usage chunk has an empty `choices` array; it must still parse.
  #[test]
  fn a_chunk_without_choices_parses_without_reasoning() {
    let parsed = parse_chunk(json!({
      "id": "chatcmpl-1",
      "object": "chat.completion.chunk",
      "created": 0,
      "model": "gpt-test",
      "choices": [],
      "usage": { "prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3 },
    }))
    .unwrap();
    assert!(parsed.reasoning.is_none());
    assert!(parsed.chunk.usage.is_some());
  }

  /// DeepSeek answers `prompt_cache_hit_tokens` instead of the standard
  /// `prompt_tokens_details.cached_tokens`; read as-is, every hit would be billed as an
  /// ordinary prompt token.
  #[test]
  fn a_cache_hit_reported_the_deepseek_way_is_read_as_a_cached_token() {
    let mut body = response_with(json!({ "role": "assistant", "content": "42" }));
    body["usage"] = json!({
      "prompt_tokens": 100,
      "completion_tokens": 10,
      "total_tokens": 110,
      "prompt_cache_hit_tokens": 64,
      "prompt_cache_miss_tokens": 36,
    });

    let parsed = parse_response(body).unwrap();
    let usage = crate::agent::TokenUsage::from(parsed.response.usage.as_ref().unwrap());
    assert_eq!(usage.cached_prompt_tokens, 64);
    assert_eq!(usage.prompt_tokens, 100);
  }

  /// A provider that sends both is read by the standard field, not overridden by it.
  #[test]
  fn the_standard_cached_tokens_field_wins_over_the_provider_specific_one() {
    let mut body = response_with(json!({ "role": "assistant", "content": "42" }));
    body["usage"] = json!({
      "prompt_tokens": 100,
      "completion_tokens": 10,
      "total_tokens": 110,
      "prompt_tokens_details": { "cached_tokens": 20 },
      "prompt_cache_hit_tokens": 64,
    });

    let parsed = parse_response(body).unwrap();
    let usage = crate::agent::TokenUsage::from(parsed.response.usage.as_ref().unwrap());
    assert_eq!(usage.cached_prompt_tokens, 20);
  }

  /// The final chunk of a stream carries usage the same way.
  #[test]
  fn a_streamed_usage_chunk_is_normalized_too() {
    let parsed = parse_chunk(json!({
      "id": "chatcmpl-1",
      "object": "chat.completion.chunk",
      "created": 0,
      "model": "gpt-test",
      "choices": [],
      "usage": {
        "prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12,
        "prompt_cache_hit_tokens": 7,
      },
    }))
    .unwrap();

    let usage = crate::agent::TokenUsage::from(parsed.chunk.usage.as_ref().unwrap());
    assert_eq!(usage.cached_prompt_tokens, 7);
  }

  #[test]
  fn a_malformed_body_is_an_error_not_a_panic() {
    assert!(parse_response(json!({ "nope": true })).is_err());
    assert!(parse_chunk(json!("not an object")).is_err());
  }

  /// A bare serde message names a field, not what came back; the error quotes the body.
  #[test]
  fn a_malformed_body_is_quoted_in_the_error_but_not_in_full() {
    let err = parse_response(json!({ "nope": "x".repeat(10_000) })).unwrap_err();
    let message = format!("{err:#}");
    assert!(message.contains("\"nope\""), "{message}");
    assert!(message.len() < 1_000, "quoted {} bytes", message.len());
  }

  fn request_of(messages: Value) -> CreateChatCompletionRequest {
    serde_json::from_value(json!({ "model": "gpt-test", "messages": messages })).unwrap()
  }

  fn conversation() -> CreateChatCompletionRequest {
    request_of(json!([
      { "role": "user", "content": "add 2 and 3" },
      { "role": "assistant", "content": null, "tool_calls": [{
        "id": "call_1", "type": "function",
        "function": { "name": "calculator", "arguments": "{}" } }] },
      { "role": "tool", "tool_call_id": "call_1", "content": "5" },
    ]))
  }

  /// The replayed chain of thought lands on the assistant message it belongs to — by
  /// position, since the typed message has no field for it.
  #[test]
  fn replayed_reasoning_is_written_to_its_assistant_message() {
    let request = ChatRequest {
      request: conversation(),
      reasoning: vec![None, Some("I should add first".to_owned())],
    };
    let body = request.to_body().unwrap();

    let messages = body["messages"].as_array().unwrap();
    assert!(messages[0].get("reasoning_content").is_none());
    assert_eq!(messages[1]["reasoning_content"], "I should add first");
    assert!(messages[2].get("reasoning_content").is_none());
  }

  /// Nothing to replay is the default, and then the body is exactly the standard one.
  #[test]
  fn a_plain_request_body_has_no_reasoning_field() {
    let body = ChatRequest::from(conversation()).to_body().unwrap();
    assert!(!body.to_string().contains("reasoning_content"), "{body}");
  }

  /// Reasoning can only belong to an assistant message; one aimed anywhere else (a
  /// transcript edited by hand) is dropped rather than sent where no provider expects it.
  #[test]
  fn reasoning_aimed_at_a_non_assistant_message_is_dropped() {
    let request = ChatRequest {
      request: conversation(),
      reasoning: vec![Some("stray".to_owned()), None, Some("stray".to_owned())],
    };
    let body = request.to_body().unwrap();
    assert!(!body.to_string().contains("stray"), "{body}");
  }

  /// A reasoning list longer than the messages is harmless.
  #[test]
  fn surplus_reasoning_entries_are_ignored() {
    let request = ChatRequest {
      request: conversation(),
      reasoning: vec![None, None, None, Some("extra".to_owned())],
    };
    assert!(request.to_body().is_ok());
  }
}
