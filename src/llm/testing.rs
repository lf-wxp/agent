//! Test-only stand-ins for a model: a [`ScriptedBackend`] that answers from a script and
//! records what it was sent, plus helpers that build the response bodies it answers with.
//!
//! Shared so every test that needs an agent to actually *run* — the runtime's own, the
//! sub-agent tool's — drives the same fake instead of each growing its own copy.

use std::{
  collections::VecDeque,
  sync::{Arc, Mutex},
};

use serde_json::{Value, json};

use crate::llm::{
  backend::{ChatBackend, ChatChunkStream, ChatRequest, ChatResponse, parse_chunk, parse_response},
  provider::Provider,
};

/// Answers each request with the next scripted reply, and records every request body.
///
/// Panics when asked for more than was scripted: a test that makes an unexpected extra
/// request should fail loudly, not hang or answer with something made up.
#[derive(Default)]
pub struct ScriptedBackend {
  replies: Mutex<VecDeque<Value>>,
  streams: Mutex<VecDeque<Vec<Value>>>,
  seen: Mutex<Vec<Value>>,
}

impl ScriptedBackend {
  /// Non-streaming replies, one per request, in order.
  pub fn replying(replies: Vec<Value>) -> Arc<Self> {
    Arc::new(Self {
      replies: Mutex::new(replies.into()),
      ..Self::default()
    })
  }

  /// Streamed replies: each inner `Vec` is the chunks of one response.
  pub fn streaming(streams: Vec<Vec<Value>>) -> Arc<Self> {
    Arc::new(Self {
      streams: Mutex::new(streams.into()),
      ..Self::default()
    })
  }

  /// Every request body received so far, in order.
  pub fn seen(&self) -> Vec<Value> {
    self.seen.lock().unwrap().clone()
  }

  /// A provider that sends everything to this backend.
  pub fn provider(self: &Arc<Self>) -> Provider {
    Provider::with_backend(Arc::clone(self) as Arc<dyn ChatBackend>, 1)
  }

  /// Records the body exactly as it would go on the wire, replayed reasoning included.
  fn record(&self, request: &ChatRequest) {
    self.seen.lock().unwrap().push(request.to_body().unwrap());
  }
}

#[async_trait::async_trait]
impl ChatBackend for ScriptedBackend {
  async fn create(&self, request: ChatRequest) -> crate::Result<ChatResponse> {
    self.record(&request);
    let reply = self
      .replies
      .lock()
      .unwrap()
      .pop_front()
      .expect("the test scripted a reply for every request");
    parse_response(reply)
  }

  async fn create_stream(&self, request: ChatRequest) -> crate::Result<ChatChunkStream> {
    self.record(&request);
    let chunks = self
      .streams
      .lock()
      .unwrap()
      .pop_front()
      .expect("the test scripted a stream for every request");
    Ok(Box::pin(futures::stream::iter(
      chunks.into_iter().map(parse_chunk),
    )))
  }
}

/// A complete response whose only choice carries `message`.
pub fn reply(message: Value) -> Value {
  json!({
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 0,
    "model": "gpt-test",
    "choices": [{ "index": 0, "message": message, "finish_reason": "stop" }],
  })
}

/// A plain-text answer.
pub fn answer(text: &str) -> Value {
  reply(json!({ "role": "assistant", "content": text }))
}

/// A response asking for one call of `tool` with `arguments`.
pub fn tool_call(id: &str, tool: &str, arguments: Value) -> Value {
  reply(json!({
    "role": "assistant",
    "content": null,
    "tool_calls": [{
      "id": id,
      "type": "function",
      "function": { "name": tool, "arguments": arguments.to_string() },
    }],
  }))
}

/// `body` with a `usage` block, as a provider reports it.
pub fn with_usage(mut body: Value, prompt: u32, completion: u32) -> Value {
  body["usage"] = json!({
    "prompt_tokens": prompt,
    "completion_tokens": completion,
    "total_tokens": prompt + completion,
  });
  body
}

/// One streamed chunk whose only choice carries `delta`.
pub fn chunk(delta: Value) -> Value {
  json!({
    "id": "chatcmpl-1",
    "object": "chat.completion.chunk",
    "created": 0,
    "model": "gpt-test",
    "choices": [{ "index": 0, "delta": delta, "finish_reason": null }],
  })
}
