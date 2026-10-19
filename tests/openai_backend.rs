//! End-to-end coverage for [`agent::llm::backend::OpenAiCompatible`] over real HTTP: a
//! local stub speaking the chat-completions API, in both its JSON and SSE forms.
//!
//! The parts worth an HTTP round trip are the ones a scripted backend cannot see:
//! that the raw (`byot`) calls the backend uses still set `stream` correctly — they no
//! longer do it themselves — that a provider's `reasoning_content` survives the SSE
//! framing, and that a provider error still arrives as the `OpenAIError` the retry
//! policy recognizes, rather than being retried as though it were a network blip.

use std::{
  convert::Infallible,
  net::SocketAddr,
  sync::{Arc, Mutex},
};

use agent::{
  Agent, AgentStreamEvent,
  llm::{provider::Provider, settings::ModelSettings},
  tools::ToolRegistry,
};
use async_openai::config::OpenAIConfig;
use axum::{
  Json, Router,
  extract::State,
  http::StatusCode,
  response::{
    IntoResponse, Response,
    sse::{Event, Sse},
  },
  routing::post,
};
use futures::StreamExt;
use serde_json::{Value, json};

#[derive(Clone, Copy)]
enum Script {
  /// Answer like a reasoning model: `reasoning_content` next to the content.
  Reason,
  /// Reject every request with a 400, the way a provider rejects a malformed one.
  Reject,
  /// Like DeepSeek's thinking mode with tools: the first answer is a tool call with
  /// reasoning; once a tool result is in the request, the final answer.
  ToolThenAnswer,
}

#[derive(Clone)]
struct StubState {
  script: Script,
  seen: Arc<Mutex<Vec<Value>>>,
}

fn chunk(delta: Value) -> String {
  json!({
    "id": "chatcmpl-stub",
    "object": "chat.completion.chunk",
    "created": 0,
    "model": "gpt-test",
    "choices": [{ "index": 0, "delta": delta, "finish_reason": null }],
  })
  .to_string()
}

async fn completions(State(state): State<StubState>, Json(body): Json<Value>) -> Response {
  let streaming = body.get("stream").and_then(Value::as_bool) == Some(true);
  let has_tool_result = body["messages"]
    .as_array()
    .is_some_and(|messages| messages.iter().any(|message| message["role"] == "tool"));
  state.seen.lock().unwrap().push(body);

  match state.script {
    Script::ToolThenAnswer => {
      let message = if has_tool_result {
        json!({ "role": "assistant", "content": "5" })
      } else {
        json!({
          "role": "assistant",
          "content": null,
          "reasoning_content": "add first",
          "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {
              "name": "calculator",
              "arguments": r#"{"operator":"add","first_number":2,"second_number":3}"#,
            },
          }],
        })
      };
      Json(json!({
        "id": "chatcmpl-stub",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-test",
        "choices": [{ "index": 0, "message": message, "finish_reason": "stop" }],
      }))
      .into_response()
    }
    Script::Reject => (
      StatusCode::BAD_REQUEST,
      Json(json!({
        "error": {
          "message": "Unsupported parameter: 'max_tokens'",
          "type": "invalid_request_error",
          "param": "max_tokens",
          "code": "unsupported_parameter",
        },
      })),
    )
      .into_response(),
    Script::Reason if streaming => {
      let events = [
        chunk(json!({ "role": "assistant", "reasoning_content": "let me " })),
        chunk(json!({ "reasoning_content": "think" })),
        chunk(json!({ "content": "4", "reasoning_content": "" })),
        chunk(json!({ "content": "2" })),
        "[DONE]".to_owned(),
      ];
      let stream = futures::stream::iter(
        events
          .into_iter()
          .map(|data| Ok::<_, Infallible>(Event::default().data(data))),
      );
      Sse::new(stream).into_response()
    }
    Script::Reason => Json(json!({
      "id": "chatcmpl-stub",
      "object": "chat.completion",
      "created": 0,
      "model": "gpt-test",
      "choices": [{
        "index": 0,
        "message": {
          "role": "assistant",
          "content": "42",
          "reasoning_content": "let me think",
        },
        "finish_reason": "stop",
      }],
      "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 },
    }))
    .into_response(),
  }
}

/// An agent pointed at a fresh stub, plus the request bodies the stub receives.
async fn stub_agent(script: Script) -> (Agent, Arc<Mutex<Vec<Value>>>) {
  stub_agent_with(
    script,
    ToolRegistry::empty(),
    ModelSettings {
      temperature: Some(0.5),
      ..ModelSettings::default()
    },
  )
  .await
}

async fn stub_agent_with(
  script: Script,
  toolbox: ToolRegistry,
  settings: ModelSettings,
) -> (Agent, Arc<Mutex<Vec<Value>>>) {
  let seen = Arc::new(Mutex::new(Vec::new()));
  let app = Router::new()
    .route("/chat/completions", post(completions))
    .with_state(StubState {
      script,
      seen: Arc::clone(&seen),
    });

  let listener = tokio::net::TcpListener::bind::<SocketAddr>("127.0.0.1:0".parse().unwrap())
    .await
    .expect("an ephemeral port should always be available");
  let addr = listener.local_addr().expect("the socket is bound");
  tokio::spawn(async move {
    let _ = axum::serve(listener, app).await;
  });

  let config = OpenAIConfig::new()
    .with_api_key("stub")
    .with_api_base(format!("http://{addr}"));
  let agent = Agent::new(
    Provider::new(config, 1),
    "gpt-test",
    Option::<String>::None,
    Arc::new(toolbox),
  )
  .with_model_settings(settings);
  (agent, seen)
}

/// What a real provider would see: the second request, after a tool round, carries the
/// first round's chain of thought as `reasoning_content` on its assistant message — and
/// only when replay is on. Over real HTTP, because the field is written into the body in
/// `OpenAiCompatible`, which a scripted backend never reaches.
#[tokio::test]
async fn replayed_reasoning_reaches_the_wire_only_when_asked() {
  for (replay, expected) in [(true, Some("add first")), (false, None)] {
    let (agent, seen) = stub_agent_with(
      Script::ToolThenAnswer,
      ToolRegistry::select(&["calculator".to_owned()]).unwrap(),
      ModelSettings {
        replay_reasoning: replay,
        ..ModelSettings::default()
      },
    )
    .await;

    let result = agent.run("add 2 and 3").await.unwrap();

    assert_eq!(result.output, "5");
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 2);
    let assistant = &seen[1]["messages"][1];
    assert_eq!(assistant["role"], "assistant", "got: {}", seen[1]);
    assert_eq!(
      assistant.get("reasoning_content").and_then(Value::as_str),
      expected,
      "replay = {replay}, got: {}",
      seen[1]
    );
  }
}

#[tokio::test]
async fn a_plain_request_returns_the_providers_reasoning() {
  let (agent, seen) = stub_agent(Script::Reason).await;

  let result = agent.run("what is six times seven?").await.unwrap();

  assert_eq!(result.output, "42");
  assert_eq!(result.reasoning.as_deref(), Some("let me think"));
  let seen = seen.lock().unwrap();
  assert_eq!(seen.len(), 1);
  assert_ne!(
    seen[0].get("stream").and_then(Value::as_bool),
    Some(true),
    "a plain request must not ask for a stream: {}",
    seen[0]
  );
  assert_eq!(seen[0]["temperature"], 0.5, "settings reach the wire");
}

#[tokio::test]
async fn a_stream_asks_for_one_and_forwards_reasoning_from_the_sse_frames() {
  let (agent, seen) = stub_agent(Script::Reason).await;

  let stream = agent.run_stream("what is six times seven?");
  futures::pin_mut!(stream);
  let mut reasoning = String::new();
  let mut answer = String::new();
  let mut done = false;
  while let Some(event) = stream.next().await {
    match event.unwrap() {
      AgentStreamEvent::Reasoning(text) => {
        assert!(answer.is_empty(), "reasoning arrives ahead of the answer");
        reasoning.push_str(&text);
      }
      AgentStreamEvent::Token(text) => answer.push_str(&text),
      AgentStreamEvent::Done { output, .. } => {
        assert_eq!(output, "42");
        done = true;
      }
      other => panic!("unexpected event {other:?}"),
    }
  }

  assert!(done);
  assert_eq!(reasoning, "let me think");
  assert_eq!(answer, "42");
  let seen = seen.lock().unwrap();
  assert_eq!(
    seen[0]["stream"], true,
    "the raw stream call no longer sets this itself"
  );
}

/// A 4xx is deterministic, so it must surface after one attempt — which only works if
/// the error still carries the provider's status for the retry policy to read.
#[tokio::test]
async fn a_provider_rejection_is_not_retried() {
  let (agent, seen) = stub_agent(Script::Reject).await;

  let err = agent
    .run("hi")
    .await
    .expect_err("the stub rejects everything");

  assert!(
    format!("{err:#}").contains("Unsupported parameter"),
    "the provider's message reaches the caller: {err:#}"
  );
  assert_eq!(seen.lock().unwrap().len(), 1, "a 400 is not worth a retry");
}
