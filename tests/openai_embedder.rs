//! Coverage for [`agent::knowledge_base::embed::OpenAiEmbedder`] over real HTTP: a local
//! stub speaking the embeddings API.
//!
//! What is worth an HTTP round trip is what a scripted [`Embedder`] cannot show: that the
//! response is put back in input order whatever order the provider sent it in, that a
//! wrong number of vectors is an error rather than a silent misalignment, and that the
//! retry policy treats a provider's 5xx and 4xx differently.

use std::{
  net::SocketAddr,
  sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
  },
};

use agent::knowledge_base::embed::{Embedder, OpenAiEmbedder};
use async_openai::config::OpenAIConfig;
use axum::{Json, Router, extract::State, http::StatusCode, response::IntoResponse, routing::post};
use serde_json::{Value, json};

#[derive(Clone, Copy)]
enum Script {
  /// Answer with the vectors in *reverse* order, each carrying its real `index`.
  Shuffled,
  /// Answer with one vector too few.
  Short,
  /// Fail with a 500 the first `n` times, then answer normally.
  FlakyThenOk(usize),
  /// Reject every request with a 400.
  Reject,
}

#[derive(Clone)]
struct Stub {
  script: Script,
  requests: Arc<AtomicUsize>,
  bodies: Arc<Mutex<Vec<Value>>>,
}

async fn embeddings(State(stub): State<Stub>, Json(body): Json<Value>) -> axum::response::Response {
  let attempt = stub.requests.fetch_add(1, Ordering::SeqCst);
  let inputs: Vec<String> = body["input"]
    .as_array()
    .map(|items| {
      items
        .iter()
        .map(|item| item.as_str().unwrap_or_default().to_owned())
        .collect()
    })
    .unwrap_or_default();
  stub.bodies.lock().unwrap().push(body);

  // A vector that says which input it belongs to, so a misordering is visible.
  let vector_of = |text: &str| vec![text.len() as f32, 1.0];
  let data = |order: Vec<usize>| -> Vec<Value> {
    order
      .into_iter()
      .map(|index| {
        json!({
          "object": "embedding",
          "index": index,
          "embedding": vector_of(&inputs[index]),
        })
      })
      .collect()
  };
  let ok = |data: Vec<Value>| {
    Json(json!({
      "object": "list",
      "model": "embed-test",
      "data": data,
      "usage": { "prompt_tokens": 7, "total_tokens": 7 },
    }))
    .into_response()
  };

  match stub.script {
    Script::Shuffled => ok(data((0..inputs.len()).rev().collect())),
    Script::Short => ok(data((0..inputs.len().saturating_sub(1)).collect())),
    Script::FlakyThenOk(failures) if attempt < failures => (
      StatusCode::INTERNAL_SERVER_ERROR,
      Json(json!({ "error": { "message": "upstream hiccup", "type": "server_error" } })),
    )
      .into_response(),
    Script::FlakyThenOk(_) => ok(data((0..inputs.len()).collect())),
    Script::Reject => (
      StatusCode::BAD_REQUEST,
      Json(json!({
        "error": {
          "message": "Unsupported model",
          "type": "invalid_request_error",
          "code": "model_not_found",
        },
      })),
    )
      .into_response(),
  }
}

/// An embedder pointed at a fresh stub, plus how many requests the stub has seen.
async fn stub_embedder(
  script: Script,
) -> (OpenAiEmbedder, Arc<AtomicUsize>, Arc<Mutex<Vec<Value>>>) {
  let requests = Arc::new(AtomicUsize::new(0));
  let bodies = Arc::new(Mutex::new(Vec::new()));
  let app = Router::new()
    .route("/embeddings", post(embeddings))
    .with_state(Stub {
      script,
      requests: Arc::clone(&requests),
      bodies: Arc::clone(&bodies),
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
  (OpenAiEmbedder::new(config, "embed-test"), requests, bodies)
}

fn texts(items: &[&str]) -> Vec<String> {
  items.iter().map(|item| (*item).to_owned()).collect()
}

/// Callers match vectors to texts by position. A provider is free to send `data` in any
/// order (each entry says which input it is), so the embedder has to put them back.
#[tokio::test]
async fn vectors_come_back_in_input_order_whatever_order_the_provider_used() {
  let (embedder, _, bodies) = stub_embedder(Script::Shuffled).await;

  let vectors = embedder.embed(&texts(&["a", "bbb", "cc"])).await.unwrap();

  assert_eq!(
    vectors,
    vec![vec![1.0, 1.0], vec![3.0, 1.0], vec![2.0, 1.0]]
  );
  let bodies = bodies.lock().unwrap();
  assert_eq!(bodies[0]["model"], "embed-test");
  assert_eq!(bodies[0]["input"], json!(["a", "bbb", "cc"]));
}

/// The failure this guards against is silent: with one vector missing, every later text
/// would be matched to its neighbour's vector and search would simply be wrong.
#[tokio::test]
async fn a_short_answer_is_an_error_not_a_misalignment() {
  let (embedder, _, _) = stub_embedder(Script::Short).await;

  let err = embedder
    .embed(&texts(&["a", "b", "c"]))
    .await
    .expect_err("two vectors for three texts");

  assert!(
    format!("{err:#}").contains("2 vectors for 3 inputs"),
    "got: {err:#}"
  );
}

#[tokio::test]
async fn a_server_error_is_retried_until_it_succeeds() {
  let (embedder, requests, _) = stub_embedder(Script::FlakyThenOk(2)).await;

  let vectors = embedder.embed(&texts(&["a"])).await.unwrap();

  assert_eq!(vectors, vec![vec![1.0, 1.0]]);
  assert_eq!(
    requests.load(Ordering::SeqCst),
    3,
    "two failures, then the answer"
  );
}

/// A 4xx will fail the same way every time, so retrying only delays the real message.
#[tokio::test]
async fn a_rejected_request_is_not_retried() {
  let (embedder, requests, _) = stub_embedder(Script::Reject).await;

  let err = embedder
    .embed(&texts(&["a"]))
    .await
    .expect_err("the stub rejects everything");

  assert!(
    format!("{err:#}").contains("Unsupported model"),
    "the provider's message reaches the caller: {err:#}"
  );
  assert_eq!(
    requests.load(Ordering::SeqCst),
    1,
    "a 400 is not worth a retry"
  );
}

#[tokio::test]
async fn no_texts_means_no_request() {
  let (embedder, requests, _) = stub_embedder(Script::Shuffled).await;

  assert!(embedder.embed(&[]).await.unwrap().is_empty());
  assert_eq!(requests.load(Ordering::SeqCst), 0);
}

/// The model name is what a store tags vectors with; it must be exactly what was asked
/// for, not something derived from the response.
#[tokio::test]
async fn the_embedder_reports_the_model_it_was_built_for() {
  let (embedder, _, _) = stub_embedder(Script::Shuffled).await;
  assert_eq!(embedder.model(), "embed-test");
}
