//! A provider's `Retry-After` over real HTTP: a local stub answers 429 with the header, and
//! the retry has to wait for it — for chat and embeddings, complete and streamed.
//!
//! The wait is read off the stub's own clock (when each request arrived), so what is
//! measured is the gap the provider would see, not how long the client thinks it slept.
//!
//! These run with the retry settings at their defaults (1 s backoff, a 60 s limit): the
//! environment variables behind them (`LLM_MAX_RETRIES`, `LLM_RETRY_MAX_WAIT_SECS`) are
//! process-wide, so a test cannot set them without racing the others. The upper bounds
//! leave room for the jitter a `Retry-After` is stretched by (a tenth) and a slow machine.

use std::{
  net::SocketAddr,
  sync::{Arc, Mutex},
  time::{Duration, Instant, SystemTime},
};

use agent::{
  knowledge_base::embed::{Embedder, OpenAiEmbedder},
  llm::{
    provider::Provider,
    retry::{is_transient, with_retry},
  },
};
use async_openai::{config::OpenAIConfig, types::chat::CreateChatCompletionRequestArgs};
use axum::{
  Json, Router,
  extract::State,
  http::{HeaderName, HeaderValue, StatusCode, header::RETRY_AFTER},
  response::{IntoResponse, Response},
  routing::post,
};
use futures::StreamExt;
use serde_json::{Value, json};

/// How a request is answered: the first `rate_limited` requests get a 429 carrying
/// `headers`, every later one the normal answer.
#[derive(Clone)]
struct Stub {
  rate_limited: usize,
  headers: Vec<(HeaderName, &'static str)>,
  /// A `Retry-After` HTTP date this far ahead of the moment the 429 is *sent*. Computed
  /// then, not when the test is written: a date has whole-second resolution, so one made
  /// earlier loses whatever the client spends getting ready (building its HTTP client
  /// takes a while in a debug build) and the test would measure the machine's load.
  retry_after_date: Option<Duration>,
  arrivals: Arc<Mutex<Vec<Instant>>>,
}

impl Stub {
  fn new(rate_limited: usize, headers: Vec<(HeaderName, &'static str)>) -> Self {
    Self {
      rate_limited,
      headers,
      retry_after_date: None,
      arrivals: Arc::default(),
    }
  }

  fn with_retry_after_date(mut self, ahead: Duration) -> Self {
    self.retry_after_date = Some(ahead);
    self
  }

  /// `Some` 429 while the stub is still rate limiting, `None` once it lets requests through.
  fn gate(&self) -> Option<Response> {
    let mut arrivals = self.arrivals.lock().unwrap();
    arrivals.push(Instant::now());
    if arrivals.len() > self.rate_limited {
      return None;
    }
    let mut response = (
      StatusCode::TOO_MANY_REQUESTS,
      Json(json!({ "error": { "message": "slow down", "type": "rate_limit_error" } })),
    )
      .into_response();
    for (name, value) in &self.headers {
      response
        .headers_mut()
        .insert(name.clone(), HeaderValue::from_static(value));
    }
    if let Some(ahead) = self.retry_after_date {
      let date = httpdate::fmt_http_date(SystemTime::now() + ahead);
      response.headers_mut().insert(
        RETRY_AFTER,
        HeaderValue::from_str(&date).expect("an HTTP date is a valid header value"),
      );
    }
    Some(response)
  }

  fn requests(&self) -> usize {
    self.arrivals.lock().unwrap().len()
  }

  /// The pause between the first and second request.
  fn gap(&self) -> Duration {
    let arrivals = self.arrivals.lock().unwrap();
    arrivals[1] - arrivals[0]
  }
}

async fn chat(State(stub): State<Stub>, Json(body): Json<Value>) -> Response {
  if let Some(limited) = stub.gate() {
    return limited;
  }
  if body["stream"] == true {
    let chunk = json!({
      "id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
      "choices": [{ "index": 0, "delta": { "role": "assistant", "content": "hi" },
                    "finish_reason": null }],
    });
    let sse = format!("data: {chunk}\n\ndata: [DONE]\n\n");
    return (
      [(axum::http::header::CONTENT_TYPE, "text/event-stream")],
      sse,
    )
      .into_response();
  }
  Json(json!({
    "id": "c", "object": "chat.completion", "created": 0, "model": "m",
    "choices": [{ "index": 0, "message": { "role": "assistant", "content": "hi" },
                  "finish_reason": "stop" }],
  }))
  .into_response()
}

async fn embeddings(State(stub): State<Stub>, Json(_): Json<Value>) -> Response {
  if let Some(limited) = stub.gate() {
    return limited;
  }
  Json(json!({
    "object": "list", "model": "e",
    "data": [{ "object": "embedding", "index": 0, "embedding": [1.0, 0.0] }],
    "usage": { "prompt_tokens": 1, "total_tokens": 1 },
  }))
  .into_response()
}

/// Serve `stub` and return the config pointing at it.
async fn serve(stub: &Stub) -> OpenAIConfig {
  let app = Router::new()
    .route("/chat/completions", post(chat))
    .route("/embeddings", post(embeddings))
    .with_state(stub.clone());
  let listener = tokio::net::TcpListener::bind::<SocketAddr>("127.0.0.1:0".parse().unwrap())
    .await
    .expect("an ephemeral port should always be available");
  let addr = listener.local_addr().expect("the socket is bound");
  tokio::spawn(async move {
    let _ = axum::serve(listener, app).await;
  });
  OpenAIConfig::new()
    .with_api_key("stub")
    .with_api_base(format!("http://{addr}"))
}

fn chat_request() -> agent::llm::backend::ChatRequest {
  CreateChatCompletionRequestArgs::default()
    .model("m")
    .messages(Vec::new())
    .build()
    .unwrap()
    .into()
}

/// The default backoff is 1 s, so a gap of 2 s can only come from the header.
const ASKED: Duration = Duration::from_secs(2);

fn assert_waited(gap: Duration, asked: Duration) {
  assert!(
    gap >= asked - Duration::from_millis(100),
    "the retry came after {gap:?}, but the provider asked for {asked:?}"
  );
  assert!(
    gap < asked + asked / 10 + Duration::from_secs(5),
    "the retry came after {gap:?}: far longer than the {asked:?} asked for"
  );
}

#[tokio::test]
async fn a_complete_chat_request_waits_out_retry_after() {
  let stub = Stub::new(1, vec![(RETRY_AFTER, "2")]);
  let provider = Provider::new(serve(&stub).await, 1);

  let response = with_retry(|| provider.chat(chat_request()), is_transient)
    .await
    .unwrap();

  assert_eq!(
    response.response.choices[0].message.content.as_deref(),
    Some("hi")
  );
  assert_eq!(stub.requests(), 2);
  assert_waited(stub.gap(), ASKED);
}

/// Opening a stream is what a 429 interrupts; the header has to survive that path too.
/// Uses the millisecond header some providers send instead.
#[tokio::test]
async fn a_streamed_chat_request_waits_out_retry_after_ms() {
  let stub = Stub::new(1, vec![(HeaderName::from_static("retry-after-ms"), "2500")]);
  let provider = Provider::new(serve(&stub).await, 1);

  let stream = with_retry(|| provider.chat_stream(chat_request()), is_transient)
    .await
    .unwrap();
  let chunks: Vec<_> = stream.collect().await;

  assert_eq!(chunks.len(), 1, "the one chunk, then the end of the stream");
  assert!(chunks[0].is_ok());
  assert_eq!(stub.requests(), 2);
  assert_waited(stub.gap(), Duration::from_millis(2500));
}

/// `Retry-After` as an HTTP date, the other form RFC 9110 allows: the wait is the time left
/// until then.
///
/// The date is four seconds out and has whole-second resolution, so what is left when the
/// client reads it is between three and four. The default backoff is one second, so a gap of
/// two or more can only have come from reading the date; the margin below the three
/// seconds is there for a slow machine.
#[tokio::test]
async fn a_retry_after_http_date_is_waited_out() {
  let stub = Stub::new(1, vec![]).with_retry_after_date(Duration::from_secs(4));
  let provider = Provider::new(serve(&stub).await, 1);

  with_retry(|| provider.chat(chat_request()), is_transient)
    .await
    .unwrap();

  assert_eq!(stub.requests(), 2);
  assert!(
    stub.gap() >= Duration::from_secs(2),
    "the retry came after {:?}: the date was not waited for",
    stub.gap()
  );
  assert!(stub.gap() < Duration::from_secs(10), "{:?}", stub.gap());
}

#[tokio::test]
async fn an_embedding_request_waits_out_retry_after() {
  let stub = Stub::new(1, vec![(RETRY_AFTER, "2")]);
  let embedder = OpenAiEmbedder::new(serve(&stub).await, "e");

  let vectors = embedder.embed(&["a".to_owned()]).await.unwrap();

  assert_eq!(vectors, vec![vec![1.0, 0.0]]);
  assert_eq!(stub.requests(), 2);
  assert_waited(stub.gap(), ASKED);
}

/// A provider that wants longer than the limit is reported, not sat through: one request,
/// and an error that says so.
#[tokio::test]
async fn a_retry_after_beyond_the_limit_is_not_retried() {
  let stub = Stub::new(usize::MAX, vec![(RETRY_AFTER, "3600")]);
  let provider = Provider::new(serve(&stub).await, 1);

  let started = Instant::now();
  let err = with_retry(|| provider.chat(chat_request()), is_transient)
    .await
    .expect_err("an hour is more than anyone should wait");

  assert!(
    started.elapsed() < Duration::from_secs(2),
    "it did not wait"
  );
  assert_eq!(stub.requests(), 1);
  let provider_error = err.as_provider().expect("still a provider error");
  assert_eq!(provider_error.status(), Some(429));
  assert_eq!(
    provider_error.retry_after(),
    Some(Duration::from_secs(3600)),
    "the caller can see what was asked"
  );
  assert!(format!("{err:#}").contains("slow down"), "{err:#}");
}

/// Without the header the curve alone applies — as before.
#[tokio::test]
async fn a_429_without_retry_after_still_retries_on_the_default_backoff() {
  let stub = Stub::new(1, vec![]);
  let provider = Provider::new(serve(&stub).await, 1);

  with_retry(|| provider.chat(chat_request()), is_transient)
    .await
    .unwrap();

  assert_eq!(stub.requests(), 2);
  assert!(stub.gap() < Duration::from_secs(5));
}
