//! End-to-end coverage for cancelling a run through its token while something is
//! actually in flight — a model request that has not answered yet, or a tool the model
//! asked for that is still running. The in-module tests cover the already-cancelled case;
//! these need a provider that answers (or pointedly does not), so this file stands up the
//! same kind of local chat-completions stub as `summarization_provider.rs`.

use std::{
  net::SocketAddr,
  sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
  },
  time::{Duration, Instant},
};

use agent::{
  Agent, Cancelled,
  agent::{ContentItem, Conversation, ToolResultStatus},
  llm::provider::Provider,
  tools::{Tool, ToolRegistry},
};
use async_openai::config::OpenAIConfig;
use axum::{Json, Router, extract::State, routing::post};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

/// What the stub does with every request.
#[derive(Clone, Copy)]
enum Script {
  /// Never answer (well, not for a minute) — a model request stuck in flight.
  Hang,
  /// Ask for one `slow` tool call.
  CallSlowTool,
}

#[derive(Clone)]
struct StubState {
  script: Script,
  requests: Arc<AtomicUsize>,
}

async fn completions(State(state): State<StubState>, Json(_body): Json<Value>) -> Json<Value> {
  state.requests.fetch_add(1, Ordering::SeqCst);
  match state.script {
    Script::Hang => {
      tokio::time::sleep(Duration::from_secs(60)).await;
      Json(json!({}))
    }
    Script::CallSlowTool => Json(json!({
      "id": "chatcmpl-stub",
      "object": "chat.completion",
      "created": 0,
      "model": "gpt-test",
      "choices": [{
        "index": 0,
        "message": {
          "role": "assistant",
          "content": null,
          "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": { "name": "slow", "arguments": "{}" },
          }],
        },
        "finish_reason": "tool_calls",
      }],
      "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 },
    })),
  }
}

async fn stub_provider(script: Script) -> (Provider, Arc<AtomicUsize>) {
  let requests = Arc::new(AtomicUsize::new(0));
  let state = StubState {
    script,
    requests: Arc::clone(&requests),
  };
  let app = Router::new()
    .route("/chat/completions", post(completions))
    .with_state(state);

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
  (Provider::new(config, 1), requests)
}

/// Sleeps far longer than any test waits; counts how often it started and finished.
struct SlowTool {
  started: Arc<AtomicUsize>,
  finished: Arc<AtomicUsize>,
}

#[async_trait::async_trait]
impl Tool for SlowTool {
  fn name(&self) -> &str {
    "slow"
  }

  fn description(&self) -> &str {
    "sleeps"
  }

  fn parameters(&self) -> Value {
    json!({ "type": "object", "properties": {} })
  }

  async fn execute(&self, _args_json: &str) -> anyhow::Result<String> {
    self.started.fetch_add(1, Ordering::SeqCst);
    tokio::time::sleep(Duration::from_secs(60)).await;
    self.finished.fetch_add(1, Ordering::SeqCst);
    Ok("done".to_owned())
  }
}

/// A token that fires once `counter` has reached `target` — i.e. once the thing the test
/// wants to interrupt is actually under way.
///
/// Waiting for the event instead of sleeping a fixed time keeps these tests correct on a
/// slow machine, where a timer can fire before the request was even sent. The deadline
/// only bounds the wait if the event never comes, and then cancels anyway so that the
/// test fails on its own assertions instead of hanging.
fn cancel_when_reached(counter: &Arc<AtomicUsize>, target: usize) -> CancellationToken {
  let token = CancellationToken::new();
  let trigger = token.clone();
  let counter = Arc::clone(counter);
  tokio::spawn(async move {
    let deadline = Instant::now() + Duration::from_secs(20);
    while counter.load(Ordering::SeqCst) < target && Instant::now() < deadline {
      tokio::time::sleep(Duration::from_millis(5)).await;
    }
    trigger.cancel();
  });
  token
}

/// A model request that is not answering is abandoned the moment the token fires,
/// rather than holding the turn until the provider gives up.
#[tokio::test]
async fn cancelling_abandons_a_model_request_in_flight() {
  let (provider, requests) = stub_provider(Script::Hang).await;
  let agent = Agent::new(
    provider,
    "gpt-test",
    Option::<String>::None,
    Arc::new(ToolRegistry::empty()),
  );

  let started = Instant::now();
  let err = agent
    .run_continuing(
      Conversation::anonymous(Vec::new()).with_cancellation(cancel_when_reached(&requests, 1)),
      "hello",
    )
    .await
    .expect_err("the run was cancelled");

  assert!(started.elapsed() < Duration::from_secs(10));
  assert_eq!(requests.load(Ordering::SeqCst), 1, "the request was sent");
  let cancelled = err
    .downcast_ref::<Cancelled>()
    .expect("a `Cancelled` error");
  assert_eq!(cancelled.context.events.len(), 1, "only the user's turn");
}

/// A tool the model asked for is cut short, and the transcript handed back pairs its
/// call with a result saying so — which is what makes it safe to persist and continue.
#[tokio::test]
async fn cancelling_cuts_a_running_tool_short_and_keeps_the_round() {
  let (provider, requests) = stub_provider(Script::CallSlowTool).await;
  let tool_started = Arc::new(AtomicUsize::new(0));
  let finished = Arc::new(AtomicUsize::new(0));
  let mut registry = ToolRegistry::empty();
  registry
    .add(Arc::new(SlowTool {
      started: Arc::clone(&tool_started),
      finished: Arc::clone(&finished),
    }))
    .unwrap();
  let agent = Agent::new(
    provider,
    "gpt-test",
    Option::<String>::None,
    Arc::new(registry),
  );

  let started = Instant::now();
  let err = agent
    .run_continuing(
      Conversation::anonymous(Vec::new()).with_cancellation(cancel_when_reached(&tool_started, 1)),
      "do the slow thing",
    )
    .await
    .expect_err("the run was cancelled");

  assert!(started.elapsed() < Duration::from_secs(10));
  assert_eq!(
    tool_started.load(Ordering::SeqCst),
    1,
    "the tool was under way when it was cut short"
  );
  assert_eq!(
    finished.load(Ordering::SeqCst),
    0,
    "the tool never finished"
  );
  assert_eq!(
    requests.load(Ordering::SeqCst),
    1,
    "no second model request after the cancellation"
  );

  let cancelled = err
    .downcast_ref::<Cancelled>()
    .expect("a `Cancelled` error");
  let items: Vec<&ContentItem> = cancelled
    .context
    .events
    .iter()
    .flat_map(|event| &event.content)
    .collect();
  assert!(items.iter().any(
    |item| matches!(item, ContentItem::ToolCall { tool_call_id, .. } if tool_call_id == "call_1")
  ));
  let result = items
    .iter()
    .find_map(|item| match item {
      ContentItem::ToolResult {
        tool_call_id,
        status,
        content,
        ..
      } if tool_call_id == "call_1" => Some((*status, content.clone())),
      _ => None,
    })
    .expect("the cancelled call still gets a result");
  assert_eq!(result.0, ToolResultStatus::Error);
  assert!(result.1.contains("cancelled"), "got: {}", result.1);
}
