//! The HTTP layer for OpenAI-compatible chat and embedding requests.
//!
//! This exists because `async-openai`'s client throws the response headers away before
//! an error reaches the caller, and the one header that matters for retrying — `Retry-After`
//! — only comes with the error. Sending the request ourselves keeps the headers, so a 429
//! or 503 can say how long to wait ([`ProviderError::retry_after`]) and
//! [`crate::llm::retry::with_retry`] can honor it.
//!
//! What is still borrowed from `async-openai` is the part that has no reason to be
//! reimplemented: [`OpenAIConfig`] (base URL, key, organization, project, custom headers)
//! and the request / response types. Parsing a response into those types stays with the
//! callers ([`crate::llm::backend`]) or is asked of [`Transport::post_json`]
//! ([`crate::knowledge_base::embed`], which reads its answer straight into the type
//! rather than through an intermediate [`Value`]).
//!
//! Errors are classified the way they always were — transport failures, 408, 409, 429 and
//! 5xx are worth another attempt, any other status is not
//! ([`crate::error::is_retryable_status`]) — and carry the provider's own message, so what
//! a caller sees does not change.
//!
//! # Limits
//!
//! There is a connect timeout but no overall or read timeout, on purpose: a streamed
//! answer legitimately takes minutes, and a non-streamed one from a reasoning model can
//! send nothing for as long. A hung connection is the caller's to cancel.
//!
//! Redirects are followed only to the same server (the same host and port, or the same
//! host upgraded from `http` to `https`). A request sent on to another host would carry
//! whatever custom headers the config has (an `api-key`, say): `reqwest` strips
//! `Authorization` on a cross-host hop, not those. The redirect is then the error, and says
//! where it pointed.

use std::{
  pin::Pin,
  time::{Duration, SystemTime},
};

use async_openai::config::{Config, OpenAIConfig};
use eventsource_stream::Eventsource;
use futures::{Stream, StreamExt};
use reqwest::{
  Response, Url,
  header::{HeaderMap, LOCATION, RETRY_AFTER},
  redirect,
};
use serde::{Deserialize, de::DeserializeOwned};
use serde_json::Value;

use crate::{Error, Result, error::ProviderError, util::truncate_chars};

/// Chars of a response body quoted in an error: enough to see what came back, not enough
/// to dump a whole page or completion into a log.
pub(crate) const BODY_PREVIEW_CHARS: usize = 300;

/// Chars of a provider's own error message kept. Far more than a preview, since it is
/// what the provider chose to say (a context-length error names the numbers), but still
/// bounded.
const ERROR_MESSAGE_MAX_CHARS: usize = 2_000;

/// Bytes of an error response's body read. Only its message is used, so a provider that
/// answers a failure with a huge page is not read to the end.
const ERROR_BODY_MAX_BYTES: usize = 64 * 1024;

/// Bytes of a successful response's body read. Far above any real completion or
/// embeddings batch (2048 vectors of 3072 floats is on the order of 100 MB of JSON), so it
/// only stops an endpoint that never stops sending.
const RESPONSE_MAX_BYTES: usize = 256 * 1024 * 1024;

/// How long to wait for a connection to open. A request that has not connected by then is
/// retried as a transport failure.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Redirect hops followed, as in `reqwest`'s own default.
const MAX_REDIRECTS: usize = 10;

/// Azure OpenAI and a few gateways send the wait in milliseconds under this name, which
/// is more precise than `Retry-After`'s whole seconds.
const RETRY_AFTER_MS: &str = "retry-after-ms";

/// Decoded JSON events of a streamed response, until the provider ends it.
pub(crate) type JsonStream = Pin<Box<dyn Stream<Item = Result<Value>> + Send>>;

/// One endpoint with its credentials, reached over a shared connection pool.
#[derive(Debug, Clone)]
pub(crate) struct Transport {
  client: reqwest::Client,
  config: OpenAIConfig,
}

impl Transport {
  pub(crate) fn new(config: OpenAIConfig) -> Self {
    Self {
      client: reqwest::Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .redirect(same_server_redirects())
        .build()
        // Only fails when the TLS backend cannot be initialized: a startup-time
        // environment problem, not something a caller could recover from.
        .expect("HTTP client must be constructible"),
      config,
    }
  }

  /// POST `body` to `path` (under the configured base URL) and return the JSON answer as
  /// a `T`: a [`Value`] for a caller that inspects it, or the response type itself.
  pub(crate) async fn post_json<T: DeserializeOwned>(&self, path: &str, body: &Value) -> Result<T> {
    let response = self.send(path, body).await?;
    let (bytes, truncated) = read_body(response, RESPONSE_MAX_BYTES)
      .await
      .map_err(transport_error)?;
    if truncated {
      return Err(
        ProviderError::permanent(format!(
          "the response is longer than {RESPONSE_MAX_BYTES} bytes"
        ))
        .with_class("invalid_response")
        .into(),
      );
    }
    serde_json::from_slice(&bytes).map_err(|err| invalid_response("the response", &bytes, err))
  }

  /// POST `body` to `path` and return the answer's server-sent events, decoded as JSON.
  ///
  /// Resolves once the response headers are in — an error status therefore surfaces here,
  /// with its `Retry-After` — and the events arrive through the returned stream, which
  /// ends at the provider's `[DONE]` marker or the end of the response.
  pub(crate) async fn post_sse(&self, path: &str, body: &Value) -> Result<JsonStream> {
    let response = self.send(path, body).await?;
    let events = Box::pin(response.bytes_stream().eventsource());
    let stream = futures::stream::unfold(Some(events), |state| async move {
      let mut events = state?;
      loop {
        match events.next().await {
          None => return None,
          Some(Err(err)) => {
            let error =
              ProviderError::transient(format!("stream failed: EventStream error: {err}"))
                .with_class("stream")
                .into();
            // A broken stream has nothing more to give.
            return Some((Err(error), None));
          }
          Some(Ok(event)) => {
            if event.data == "[DONE]" {
              return None;
            }
            if event.event == "keepalive" || event.data.is_empty() {
              continue;
            }
            return Some((decode_event(&event.data), Some(events)));
          }
        }
      }
    });
    Ok(Box::pin(stream))
  }

  /// Send one POST and hand back the response if it is a success, or the classified error
  /// if it is not.
  async fn send(&self, path: &str, body: &Value) -> Result<Response> {
    let response = self
      .client
      .post(self.config.url(path))
      .headers(self.config.headers())
      .json(body)
      .send()
      .await
      .map_err(transport_error)?;
    if response.status().is_success() {
      Ok(response)
    } else {
      Err(status_error(response).await)
    }
  }
}

/// Follow a redirect only when it stays with the server it came from (see
/// [`stays_with_the_server`]). Anything else is returned as the response it is — a 3xx,
/// which is then an error like any other non-success status, naming where it pointed.
fn same_server_redirects() -> redirect::Policy {
  redirect::Policy::custom(|attempt| {
    if attempt.previous().len() > MAX_REDIRECTS {
      return attempt.error("too many redirects");
    }
    let stays = attempt
      .previous()
      .last()
      .is_some_and(|from| stays_with_the_server(from, attempt.url()));
    if stays {
      attempt.follow()
    } else {
      attempt.stop()
    }
  })
}

/// Whether a redirect from `from` to `to` stays with the same server: the same host on the
/// same port, or the same host moved from plain HTTP to HTTPS on their default ports (a
/// gateway pointing an `http://` base URL at its `https://` address). Never a downgrade, and
/// never another host: what the request carries — a key in a custom header, say — is meant
/// for the one the config names.
fn stays_with_the_server(from: &Url, to: &Url) -> bool {
  if from.host_str() != to.host_str() {
    return false;
  }
  from.origin() == to.origin()
    || (from.scheme() == "http"
      && to.scheme() == "https"
      && from.port_or_known_default() == Some(80)
      && to.port_or_known_default() == Some(443))
}

/// Most of a body's size taken on trust, from its `Content-Length`, when making room for
/// it. A server can claim any length, and the rest grows as the bytes arrive.
const PREALLOCATE_MAX_BYTES: usize = 1024 * 1024;

/// Read a body, stopping at `limit` bytes; the flag says whether there was more.
async fn read_body(
  response: Response,
  limit: usize,
) -> std::result::Result<(Vec<u8>, bool), reqwest::Error> {
  let announced = response
    .content_length()
    .map_or(0, |len| usize::try_from(len).unwrap_or(usize::MAX));
  let mut body = Vec::with_capacity(announced.min(limit).min(PREALLOCATE_MAX_BYTES));
  let mut chunks = response.bytes_stream();
  while let Some(chunk) = chunks.next().await {
    let chunk = chunk?;
    let room = limit - body.len();
    if chunk.len() > room {
      body.extend_from_slice(&chunk[..room]);
      return Ok((body, true));
    }
    body.extend_from_slice(&chunk);
  }
  Ok((body, false))
}

/// A response (or stream event) that is not what the caller can parse. Retryable — a
/// garbled answer can be a blip — and named `invalid_response` in a trace.
fn invalid_response(what: &str, body: &[u8], err: serde_json::Error) -> Error {
  let problem = if err.is_data() {
    "has an unexpected shape"
  } else {
    "is not valid JSON"
  };
  ProviderError::transient(format!("{what} {problem}: {}", preview_bytes(body)))
    .with_class("invalid_response")
    .with_source(err)
    .into()
}

fn decode_event(data: &str) -> Result<Value> {
  serde_json::from_str(data).map_err(|err| invalid_response("a stream event", data.as_bytes(), err))
}

/// A request that never got an answer: DNS, connect, TLS, a reset or timed-out connection.
fn transport_error(err: reqwest::Error) -> Error {
  ProviderError::transient(format!("http error: {err}"))
    .with_class("transport")
    .with_source(err)
    .into()
}

/// The error for a non-success response: the provider's message, the status, whether
/// another attempt could work, and how long the provider asked to wait first.
async fn status_error(response: Response) -> Error {
  let status = response.status();
  let retry_after = parse_retry_after(response.headers(), SystemTime::now());
  let location = response
    .headers()
    .get(LOCATION)
    .and_then(|value| value.to_str().ok())
    .map(preview)
    .filter(|_| status.is_redirection());
  let body = match read_body(response, ERROR_BODY_MAX_BYTES).await {
    Ok((body, _)) => body,
    Err(err) => return transport_error(err),
  };
  let message = match location {
    // Only reached for a redirect that was not followed (see `same_server_redirects`).
    Some(location) => format!("{status}: redirected to {location}, which is not followed"),
    None => format!(
      "{status} {}",
      api_error_message(&String::from_utf8_lossy(&body))
    ),
  };

  let error = ProviderError::from_status(status.as_u16(), message);
  match retry_after {
    Some(wait) => error.with_retry_after(wait),
    None => error,
  }
  .into()
}

#[derive(Deserialize)]
struct ApiErrorBody {
  error: Value,
}

#[derive(Deserialize)]
struct ApiErrorFields {
  message: Option<String>,
  r#type: Option<String>,
  param: Option<String>,
  code: Option<Value>,
}

/// The provider's own words for what went wrong: `{type}: {message} (param: ..) (code: ..)`
/// out of the standard `{"error": {...}}` body, a bare `{"error": "text"}`, or — for a
/// gateway's HTML page or an empty body — whatever came back.
fn api_error_message(body: &str) -> String {
  let from_json = serde_json::from_str::<ApiErrorBody>(body)
    .ok()
    .and_then(|parsed| match parsed.error {
      Value::String(text) => Some(text),
      other => {
        let fields = serde_json::from_value::<ApiErrorFields>(other).ok()?;
        let mut parts = Vec::new();
        if let Some(kind) = fields.r#type {
          parts.push(format!("{kind}:"));
        }
        parts.push(fields.message?);
        if let Some(param) = fields.param {
          parts.push(format!("(param: {param})"));
        }
        match fields.code {
          Some(Value::String(code)) => parts.push(format!("(code: {code})")),
          Some(Value::Number(code)) => parts.push(format!("(code: {code})")),
          _ => {}
        }
        Some(parts.join(" "))
      }
    });
  match from_json {
    Some(message) => truncate_chars(&message, ERROR_MESSAGE_MAX_CHARS).to_owned(),
    None if body.trim().is_empty() => "(no response body)".to_owned(),
    None => preview(body),
  }
}

fn preview(text: &str) -> String {
  truncate_chars(text.trim(), BODY_PREVIEW_CHARS).to_owned()
}

/// [`preview`] of raw bytes, without decoding more of them than the preview can hold
/// (a char is at most four bytes).
fn preview_bytes(bytes: &[u8]) -> String {
  let head = &bytes[..bytes.len().min(BODY_PREVIEW_CHARS * 4)];
  preview(&String::from_utf8_lossy(head))
}

/// How long the provider asked callers to wait, from its response headers.
///
/// Reads `retry-after-ms` (milliseconds) first, since it is the more precise, then
/// `Retry-After`, which is either a whole number of seconds or an HTTP date (RFC 9110).
/// A date already in the past means "now", i.e. no wait. An unreadable value is no value:
/// the caller's own backoff applies.
pub(crate) fn parse_retry_after(headers: &HeaderMap, now: SystemTime) -> Option<Duration> {
  if let Some(millis) = headers
    .get(RETRY_AFTER_MS)
    .and_then(|value| value.to_str().ok())
    .and_then(|text| text.trim().parse::<u64>().ok())
  {
    return Some(Duration::from_millis(millis));
  }
  let text = headers.get(RETRY_AFTER)?.to_str().ok()?.trim();
  if let Ok(seconds) = text.parse::<u64>() {
    return Some(Duration::from_secs(seconds));
  }
  let date = httpdate::parse_http_date(text).ok()?;
  Some(date.duration_since(now).unwrap_or(Duration::ZERO))
}

#[cfg(test)]
mod tests {
  use reqwest::header::HeaderValue;

  use super::*;

  fn headers(pairs: &[(&'static str, &str)]) -> HeaderMap {
    let mut map = HeaderMap::new();
    for (name, value) in pairs {
      map.insert(*name, HeaderValue::from_str(value).unwrap());
    }
    map
  }

  fn at(secs: u64) -> SystemTime {
    SystemTime::UNIX_EPOCH + Duration::from_secs(secs)
  }

  #[test]
  fn retry_after_seconds_are_read() {
    let wait = parse_retry_after(&headers(&[("retry-after", "7")]), at(0));
    assert_eq!(wait, Some(Duration::from_secs(7)));
    let padded = parse_retry_after(&headers(&[("retry-after", " 3 ")]), at(0));
    assert_eq!(padded, Some(Duration::from_secs(3)));
  }

  #[test]
  fn retry_after_ms_wins_over_retry_after() {
    let both = headers(&[("retry-after", "7"), ("retry-after-ms", "1500")]);
    assert_eq!(
      parse_retry_after(&both, at(0)),
      Some(Duration::from_millis(1500))
    );
  }

  /// RFC 9110's `HTTP-date` form, relative to the clock passed in.
  #[test]
  fn retry_after_http_dates_are_relative_to_now() {
    // 1994-11-06 08:49:37 GMT is 784 111 777 seconds after the epoch.
    let date = headers(&[("retry-after", "Sun, 06 Nov 1994 08:49:37 GMT")]);
    assert_eq!(
      parse_retry_after(&date, at(784_111_777 - 10)),
      Some(Duration::from_secs(10))
    );
    assert_eq!(
      parse_retry_after(&date, at(784_111_777 + 10)),
      Some(Duration::ZERO),
      "a date in the past is no wait"
    );
  }

  #[test]
  fn an_absent_or_unreadable_retry_after_is_none() {
    assert_eq!(parse_retry_after(&HeaderMap::new(), at(0)), None);
    for junk in ["soon", "-5", "1.5", ""] {
      assert_eq!(
        parse_retry_after(&headers(&[("retry-after", junk)]), at(0)),
        None,
        "{junk:?}"
      );
    }
    assert_eq!(
      parse_retry_after(&headers(&[("retry-after-ms", "x")]), at(0)),
      None
    );
  }

  /// What `async-openai` used to print, so the message a caller sees does not change.
  #[test]
  fn the_standard_error_body_is_rendered_as_before() {
    let body = r#"{"error":{"message":"Incorrect API key provided","type":"invalid_request_error","param":null,"code":"invalid_api_key"}}"#;
    assert_eq!(
      api_error_message(body),
      "invalid_request_error: Incorrect API key provided (code: invalid_api_key)"
    );
    let with_param = r#"{"error":{"message":"Unsupported parameter: 'max_tokens'","type":"invalid_request_error","param":"max_tokens","code":"unsupported_parameter"}}"#;
    assert_eq!(
      api_error_message(with_param),
      "invalid_request_error: Unsupported parameter: 'max_tokens' (param: max_tokens) \
       (code: unsupported_parameter)"
    );
  }

  #[test]
  fn other_error_shapes_still_say_what_came_back() {
    assert_eq!(api_error_message(r#"{"error":"slow down"}"#), "slow down");
    assert_eq!(
      api_error_message(r#"{"error":{"message":"m","code":429}}"#),
      "m (code: 429)"
    );
    assert_eq!(
      api_error_message("<html>502 Bad Gateway</html>"),
      "<html>502 Bad Gateway</html>"
    );
    assert_eq!(api_error_message("  "), "(no response body)");
    assert!(api_error_message(&"x".repeat(10_000)).len() <= BODY_PREVIEW_CHARS);
  }

  /// The provider's own message is kept longer than a preview, but not without bound.
  #[test]
  fn a_long_json_message_is_bounded_but_not_cut_to_a_preview() {
    let long = "y".repeat(10_000);
    let body = format!(r#"{{"error":{{"message":"{long}","type":"t"}}}}"#);
    let message = api_error_message(&body);
    assert_eq!(message.chars().count(), ERROR_MESSAGE_MAX_CHARS);
    assert!(message.starts_with("t: yyy"));

    let context = "z".repeat(500);
    let body = format!(r#"{{"error":{{"message":"{context}"}}}}"#);
    assert_eq!(api_error_message(&body), context, "500 chars are all kept");
  }

  #[test]
  fn a_preview_of_bytes_never_splits_a_char_or_over_reads() {
    let text = "é".repeat(10_000);
    let preview = preview_bytes(text.as_bytes());
    assert_eq!(preview.chars().count(), BODY_PREVIEW_CHARS);
    assert!(preview.chars().all(|c| c == 'é'));
    assert_eq!(preview_bytes(b"  hi \n"), "hi");
    assert_eq!(preview_bytes(&[0xff, 0xfe]).chars().count(), 2);
  }

  /// A response that does not parse says whether it is not JSON at all or JSON of the
  /// wrong shape, is retryable, and is named for a trace as it always was.
  #[test]
  fn an_unparseable_response_is_an_invalid_response() {
    let syntax = serde_json::from_slice::<Value>(b"<html>").unwrap_err();
    let err = invalid_response("the response", b"<html>", syntax);
    let provider = err.as_provider().expect("a provider error");
    assert_eq!(provider.class(), "invalid_response");
    assert!(provider.is_retryable());
    assert_eq!(provider.status(), None);
    assert_eq!(err.to_string(), "the response is not valid JSON: <html>");

    let shape = serde_json::from_str::<u32>("\"x\"").unwrap_err();
    let err = invalid_response("a stream event", b"\"x\"", shape);
    assert_eq!(err.as_provider().unwrap().class(), "invalid_response");
    assert_eq!(
      err.to_string(),
      "a stream event has an unexpected shape: \"x\""
    );
  }

  /// The message a caller sees for a failed connection is the one `async-openai` gave
  /// (including, in the `{:#}` form, what it has always repeated).
  #[test]
  fn a_transport_error_reads_as_it_did_through_async_openai() {
    let failure = || {
      reqwest::Client::new()
        .get("not a url")
        .build()
        .expect_err("an invalid URL fails to build")
    };
    let new = transport_error(failure());
    let old = Error::from(async_openai::error::OpenAIError::Reqwest(failure()));
    assert_eq!(new.to_string(), old.to_string());
    assert_eq!(format!("{new:#}"), format!("{old:#}"));
    let (new, old) = (new.as_provider().unwrap(), old.as_provider().unwrap());
    assert_eq!(new.class(), old.class());
    assert_eq!(new.is_retryable(), old.is_retryable());
    assert_eq!(new.status(), old.status());
  }

  #[test]
  fn a_redirect_stays_with_the_server_or_is_not_followed() {
    let follows = |from: &str, to: &str| {
      stays_with_the_server(&Url::parse(from).unwrap(), &Url::parse(to).unwrap())
    };
    assert!(follows("http://h/v1/chat", "http://h/v2/chat"));
    assert!(follows("https://h:8443/a", "https://h:8443/b"));
    assert!(
      follows("http://h/a", "https://h/a"),
      "the same host, upgraded to https"
    );
    assert!(follows("http://h:80/a", "https://h:443/a"));

    assert!(!follows("https://h/a", "http://h/a"), "never a downgrade");
    assert!(!follows("http://h/a", "http://other/a"));
    assert!(!follows("http://h/a", "http://h.evil.example/a"));
    assert!(!follows("http://h/a", "http://h:8080/a"), "another port");
    assert!(!follows("http://h/a", "https://h:8443/a"));
    assert!(!follows("http://h:8080/a", "https://h:8443/a"));
    assert!(!follows("http://h:8080/a", "https://h/a"));
  }

  fn response(body: &str) -> Response {
    Response::from(axum::http::Response::new(body.to_owned()))
  }

  #[tokio::test]
  async fn a_body_is_read_up_to_the_limit() {
    let (body, truncated) = read_body(response("0123456789"), 100).await.unwrap();
    assert_eq!(body, b"0123456789");
    assert!(!truncated);

    let (body, truncated) = read_body(response("0123456789"), 4).await.unwrap();
    assert_eq!(body, b"0123");
    assert!(truncated);

    let (body, truncated) = read_body(response("0123456789"), 10).await.unwrap();
    assert_eq!(body, b"0123456789");
    assert!(!truncated, "exactly the limit is not too long");
  }
}
