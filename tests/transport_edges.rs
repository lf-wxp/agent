//! The edges of [`agent::llm::backend::OpenAiCompatible`]'s HTTP layer over real HTTP: a
//! local stub that answers with whatever bytes a test scripts, including the answers a
//! well-behaved provider never sends (a gateway's HTML, a stream that stops halfway, an
//! event that is not JSON, a redirect to another host).
//!
//! `tests/openai_backend.rs` covers the provider speaking the API correctly and
//! `tests/retry_after.rs` the waiting; what is here is how a failure reads — whether it
//! is retryable, what the caller is told. (The class a trace files it under is private to the crate and
//! pinned by the unit tests in `llm::transport`.)

use std::{
  io,
  net::SocketAddr,
  sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
  },
};

use agent::{
  Error,
  llm::backend::{ChatBackend, ChatRequest, OpenAiCompatible},
};
use async_openai::{config::OpenAIConfig, types::chat::CreateChatCompletionRequestArgs};
use axum::{
  Router,
  body::{Body, Bytes},
  http::{StatusCode, header},
  response::{IntoResponse, Response},
  routing::post,
};
use futures::StreamExt;

/// What a route answers, byte for byte.
#[derive(Clone)]
enum Reply {
  /// Status, content type and body.
  Raw(StatusCode, &'static str, String),
  /// A 200 event stream made of these pieces, then the connection dropped mid-body.
  BrokenStream(Vec<String>),
  /// A 307 to this location.
  Redirect(String),
}

#[derive(Clone)]
struct Route {
  reply: Reply,
  hits: Arc<AtomicUsize>,
}

impl Route {
  fn new(reply: Reply) -> Self {
    Self {
      reply,
      hits: Arc::default(),
    }
  }

  fn hits(&self) -> usize {
    self.hits.load(Ordering::SeqCst)
  }
}

async fn answer(route: Route) -> Response {
  route.hits.fetch_add(1, Ordering::SeqCst);
  match route.reply {
    Reply::Raw(status, content_type, body) => {
      (status, [(header::CONTENT_TYPE, content_type)], body).into_response()
    }
    Reply::BrokenStream(pieces) => {
      // The failure comes after a pause: if it were ready in the same poll as the first
      // piece, the connection would drop before the response headers were out, and what
      // is under test — a stream that has begun and then stops — would never happen.
      let cut = futures::stream::once(async {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        Err::<Bytes, _>(io::Error::other("the connection was cut"))
      });
      let sent = futures::stream::iter(
        pieces
          .into_iter()
          .map(|piece| Ok::<_, io::Error>(Bytes::from(piece))),
      );
      (
        [(header::CONTENT_TYPE, "text/event-stream")],
        Body::from_stream(sent.chain(cut)),
      )
        .into_response()
    }
    Reply::Redirect(location) => (
      StatusCode::TEMPORARY_REDIRECT,
      [(header::LOCATION, location)],
    )
      .into_response(),
  }
}

/// Serve `routes` (path, route) on a fresh port; the address they are on.
async fn serve(routes: &[(&str, &Route)]) -> SocketAddr {
  let mut app = Router::new();
  for (path, route) in routes {
    let route = (*route).clone();
    app = app.route(
      path,
      post(move || {
        let route = route.clone();
        async move { answer(route).await }
      }),
    );
  }
  let listener = tokio::net::TcpListener::bind::<SocketAddr>("127.0.0.1:0".parse().unwrap())
    .await
    .expect("an ephemeral port should always be available");
  let addr = listener.local_addr().expect("the socket is bound");
  tokio::spawn(async move {
    let _ = axum::serve(listener, app).await;
  });
  addr
}

fn backend(addr: SocketAddr) -> OpenAiCompatible {
  OpenAiCompatible::new(
    OpenAIConfig::new()
      .with_api_key("stub")
      .with_api_base(format!("http://{addr}")),
  )
}

fn request() -> ChatRequest {
  CreateChatCompletionRequestArgs::default()
    .model("m")
    .messages(Vec::new())
    .build()
    .unwrap()
    .into()
}

/// `backend` answering with `reply` on the chat endpoint, and that route.
async fn chat_with(reply: Reply) -> (OpenAiCompatible, Route) {
  let route = Route::new(reply);
  let addr = serve(&[("/chat/completions", &route)]).await;
  (backend(addr), route)
}

fn event(data: &str) -> String {
  format!("data: {data}\n\n")
}

const CHUNK: &str = r#"{"id":"c","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"},"finish_reason":null}]}"#;

fn sse(pieces: &[String]) -> Reply {
  Reply::Raw(StatusCode::OK, "text/event-stream", pieces.concat())
}

fn provider_error(err: &Error) -> &agent::error::ProviderError {
  err.as_provider().expect("a provider error")
}

#[tokio::test]
async fn keepalives_comments_and_empty_events_are_skipped() {
  let (backend, _) = chat_with(sse(&[
    "event: keepalive\ndata: ping\n\n".to_owned(),
    ": a comment line\n\n".to_owned(),
    "event: message\ndata:\n\n".to_owned(),
    event(CHUNK),
    event("[DONE]"),
  ]))
  .await;

  let chunks: Vec<_> = backend
    .create_stream(request())
    .await
    .unwrap()
    .collect()
    .await;

  assert_eq!(chunks.len(), 1, "only the real chunk comes through");
  assert!(chunks[0].is_ok());
}

#[tokio::test]
async fn a_stream_may_end_without_a_done_marker() {
  let (backend, _) = chat_with(sse(&[event(CHUNK), event(CHUNK)])).await;

  let chunks: Vec<_> = backend
    .create_stream(request())
    .await
    .unwrap()
    .collect()
    .await;

  assert_eq!(chunks.len(), 2);
  assert!(chunks.iter().all(Result::is_ok));
}

#[tokio::test]
async fn nothing_after_the_done_marker_is_read() {
  let (backend, _) = chat_with(sse(&[event(CHUNK), event("[DONE]"), event(CHUNK)])).await;

  let chunks: Vec<_> = backend
    .create_stream(request())
    .await
    .unwrap()
    .collect()
    .await;

  assert_eq!(chunks.len(), 1);
}

/// The connection drops after one chunk: that chunk arrives, then one error saying the
/// stream failed — retryable, filed under `stream` — and then the stream is over rather
/// than reading on from a broken connection.
#[tokio::test]
async fn a_stream_cut_halfway_ends_in_one_stream_error() {
  let (backend, _) = chat_with(Reply::BrokenStream(vec![event(CHUNK)])).await;

  let mut stream = backend.create_stream(request()).await.unwrap();

  assert!(stream.next().await.expect("the first chunk").is_ok());
  let err = stream.next().await.expect("the failure").unwrap_err();
  let provider = provider_error(&err);
  assert!(provider.is_retryable());
  assert!(
    err
      .to_string()
      .starts_with("stream failed: EventStream error:")
  );
  assert!(stream.next().await.is_none(), "a broken stream is over");
}

/// An event that is not JSON is an error for that event; the events after it still come.
#[tokio::test]
async fn a_stream_event_that_is_not_json_is_an_invalid_response() {
  let (backend, _) = chat_with(sse(&[
    event("not json at all"),
    event(CHUNK),
    event("[DONE]"),
  ]))
  .await;

  let chunks: Vec<_> = backend
    .create_stream(request())
    .await
    .unwrap()
    .collect()
    .await;

  assert_eq!(chunks.len(), 2);
  let err = chunks[0].as_ref().unwrap_err();
  let provider = provider_error(err);
  assert!(provider.is_retryable());
  assert_eq!(
    err.to_string(),
    "a stream event is not valid JSON: not json at all"
  );
  assert!(chunks[1].is_ok());
}

/// Some providers report a failure inside a 200 stream, as an `{"error": ..}` event.
#[tokio::test]
async fn an_error_event_inside_a_stream_is_an_error_not_a_chunk() {
  let (backend, _) = chat_with(sse(&[
    event(r#"{"error":{"message":"overloaded","type":"server_error"}}"#),
    event("[DONE]"),
  ]))
  .await;

  let chunks: Vec<_> = backend
    .create_stream(request())
    .await
    .unwrap()
    .collect()
    .await;

  assert_eq!(chunks.len(), 1);
  let err = chunks[0].as_ref().unwrap_err();
  assert!(err.to_string().contains("unexpected shape"), "{err}");
  assert!(err.to_string().contains("overloaded"), "{err}");
}

#[tokio::test]
async fn a_success_whose_body_is_not_json_is_an_invalid_response() {
  let (backend, route) = chat_with(Reply::Raw(
    StatusCode::OK,
    "text/html",
    "<html>captive portal</html>".to_owned(),
  ))
  .await;

  let err = backend.create(request()).await.unwrap_err();

  let provider = provider_error(&err);
  assert!(provider.is_retryable());
  assert_eq!(provider.status(), None);
  assert_eq!(
    err.to_string(),
    "the response is not valid JSON: <html>captive portal</html>"
  );
  assert_eq!(route.hits(), 1);
}

#[tokio::test]
async fn a_success_of_the_wrong_shape_says_what_came_back() {
  let (backend, _) = chat_with(Reply::Raw(
    StatusCode::OK,
    "application/json",
    r#"{"unexpected":true}"#.to_owned(),
  ))
  .await;

  let err = backend.create(request()).await.unwrap_err();

  assert!(provider_error(&err).is_retryable());
  assert!(err.to_string().contains(r#"{"unexpected":true}"#), "{err}");
}

/// A gateway's page in place of the provider's JSON: retryable, with the status, and the
/// page quoted (trimmed) so the failure can be diagnosed.
#[tokio::test]
async fn a_5xx_with_an_html_body_is_retryable_and_quotes_the_page() {
  let (backend, _) = chat_with(Reply::Raw(
    StatusCode::BAD_GATEWAY,
    "text/html",
    "\n<html>502 Bad Gateway</html>\n".to_owned(),
  ))
  .await;

  let err = backend.create(request()).await.unwrap_err();

  let provider = provider_error(&err);
  assert_eq!(provider.status(), Some(502));
  assert!(provider.is_retryable());
  assert_eq!(
    err.to_string(),
    "502 Bad Gateway <html>502 Bad Gateway</html>"
  );
}

/// The same page with a 4xx status: a retry would meet it again, so it is not retried —
/// whatever the body, the status decides.
#[tokio::test]
async fn a_4xx_with_an_html_body_is_permanent() {
  for status in [
    StatusCode::BAD_REQUEST,
    StatusCode::UNAUTHORIZED,
    StatusCode::FORBIDDEN,
    StatusCode::NOT_FOUND,
  ] {
    let (backend, _) = chat_with(Reply::Raw(
      status,
      "text/html",
      "<html>nope</html>".to_owned(),
    ))
    .await;

    let err = backend.create(request()).await.unwrap_err();

    let provider = provider_error(&err);
    assert_eq!(provider.status(), Some(status.as_u16()));
    assert!(!provider.is_retryable(), "{status}");
  }
}

#[tokio::test]
async fn an_empty_error_body_is_said_to_be_empty() {
  let (backend, _) = chat_with(Reply::Raw(
    StatusCode::SERVICE_UNAVAILABLE,
    "text/plain",
    String::new(),
  ))
  .await;

  let err = backend.create(request()).await.unwrap_err();

  assert_eq!(
    err.to_string(),
    "503 Service Unavailable (no response body)"
  );
}

/// A huge error page is read only as far as its start, and quoted shortened.
#[tokio::test]
async fn a_huge_error_page_is_quoted_shortened() {
  let (backend, _) = chat_with(Reply::Raw(
    StatusCode::BAD_GATEWAY,
    "text/html",
    "x".repeat(5_000_000),
  ))
  .await;

  let err = backend.create(request()).await.unwrap_err();

  assert!(
    err.to_string().len() < 400,
    "{} bytes",
    err.to_string().len()
  );
}

/// A redirect inside the origin is followed: the request, body and all, reaches the new path.
#[tokio::test]
async fn a_redirect_within_the_origin_is_followed() {
  let target = Route::new(Reply::Raw(
    StatusCode::OK,
    "application/json",
    r#"{"id":"c","object":"chat.completion","created":0,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"moved"},"finish_reason":"stop"}]}"#
      .to_owned(),
  ));
  let moved = Route::new(Reply::Redirect("/moved".to_owned()));
  let addr = serve(&[("/chat/completions", &moved), ("/moved", &target)]).await;

  let response = backend(addr).create(request()).await.unwrap();

  assert_eq!(
    response.response.choices[0].message.content.as_deref(),
    Some("moved")
  );
  assert_eq!((moved.hits(), target.hits()), (1, 1));
}

/// A redirect to another host is not followed: the request, with the credentials and custom
/// headers it carries, never reaches it. The redirect itself is the error.
#[tokio::test]
async fn a_redirect_to_another_origin_is_not_followed() {
  let elsewhere = Route::new(Reply::Raw(
    StatusCode::OK,
    "application/json",
    "{}".to_owned(),
  ));
  let other = serve(&[("/chat/completions", &elsewhere)]).await;
  let redirecting = Route::new(Reply::Redirect(format!("http://{other}/chat/completions")));
  let addr = serve(&[("/chat/completions", &redirecting)]).await;

  let err = backend(addr).create(request()).await.unwrap_err();

  assert_eq!(elsewhere.hits(), 0, "the other origin was never asked");
  let provider = provider_error(&err);
  assert_eq!(provider.status(), Some(307));
  assert!(!provider.is_retryable());
  assert_eq!(
    err.to_string(),
    format!(
      "307 Temporary Redirect: redirected to http://{other}/chat/completions, which is not followed"
    )
  );
}

#[tokio::test]
async fn a_refused_connection_is_a_retryable_transport_error() {
  // Bind to learn a free port, then close it again.
  let addr = {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap()
  };

  let err = backend(addr).create(request()).await.unwrap_err();

  let provider = provider_error(&err);
  assert!(provider.is_retryable());
  assert!(err.to_string().starts_with("http error: "), "{err}");
}
