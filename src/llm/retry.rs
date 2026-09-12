//! Shared retry policy for a retryable LLM operation.
//!
//! Every call site that retries — a single model request in [`crate::llm::tool_loop::run`]
//! and [`crate::agent::runtime::Agent`], a whole stream in [`crate::llm::stream`], a whole
//! structured solve in [`crate::gaia::solver`] — goes through [`with_retry`], so the attempt
//! budget ([`crate::config::max_retries`]) and backoff curve live in one place instead of
//! being copied (and drifting) at each call site.

use std::future::Future;

use async_openai::error::OpenAIError;
use backon::{ExponentialBuilder, Retryable};
use reqwest::StatusCode;

use crate::config;

/// Whether `err` is the kind of failure another attempt could plausibly survive.
///
/// Transient by nature: transport errors, rate limiting (429), a request timeout (408),
/// a conflict (409), and any 5xx. Everything else a provider reports as a 4xx — a
/// malformed request, a rejected key, a model the tenant cannot reach — is
/// *deterministic*: the identical request fails identically, so retrying only burns the
/// attempt budget and delays surfacing the real error. Client-side validation failures
/// ([`OpenAIError::InvalidArgument`], raised while the request is still being built) are
/// deterministic for exactly the same reason.
///
/// Errors that are not an [`OpenAIError`] are treated as retryable: this predicate only
/// claims to recognize *provably* deterministic provider failures, and guessing at
/// anything else would silently turn a recoverable blip into a hard failure.
///
/// # Known limitation: `Retry-After`
///
/// A 429 or 503 usually carries a `Retry-After` header, and honoring it is strictly
/// better than guessing — backing off *less* than the server asked for is what turns one
/// rate-limit response into a sustained one. It is not honored here because it cannot be
/// read: [`async_openai::error::ApiErrorResponse`] keeps only the status code and the
/// parsed error body, discarding the response headers before the error reaches us. Fixing
/// this needs either an upstream change or a custom transport, so until then
/// [`with_retry`]'s exponential curve is the whole of the backoff policy.
pub fn is_transient(err: &anyhow::Error) -> bool {
  match err.downcast_ref::<OpenAIError>() {
    Some(OpenAIError::InvalidArgument(_)) => false,
    Some(OpenAIError::ApiError(response)) => {
      let status = response.status_code;
      status.is_server_error()
        || matches!(
          status,
          StatusCode::TOO_MANY_REQUESTS | StatusCode::REQUEST_TIMEOUT | StatusCode::CONFLICT
        )
    }
    _ => true,
  }
}

/// Retry `op` with exponential backoff, skipping failures for which `should_retry` returns
/// `false`.
///
/// `should_retry` exists for deterministic failures a retry cannot fix — e.g. a
/// `max_tokens` truncation ([`crate::llm::structured::TruncatedOutput`]) or a tool-round
/// budget exhausted attempt ([`crate::llm::tool_loop::BudgetExhausted`]) would just repeat
/// the same outcome at the same cost. Pass `|_| true` when there is nothing like that to
/// skip.
pub async fn with_retry<T, Fut>(
  op: impl FnMut() -> Fut,
  should_retry: impl Fn(&anyhow::Error) -> bool,
) -> anyhow::Result<T>
where
  Fut: Future<Output = anyhow::Result<T>>,
{
  op.retry(ExponentialBuilder::default().with_max_times(config::max_retries()))
    .when(should_retry)
    .notify(|err, dur| tracing::warn!("retrying after {dur:?}: {err}"))
    .await
}

#[cfg(test)]
mod tests {
  use std::sync::atomic::{AtomicUsize, Ordering};

  use super::*;

  #[tokio::test]
  async fn succeeds_without_retrying_on_the_first_try() {
    let attempts = AtomicUsize::new(0);
    let result = with_retry(
      || async {
        attempts.fetch_add(1, Ordering::SeqCst);
        anyhow::Ok(42)
      },
      |_| true,
    )
    .await
    .unwrap();

    assert_eq!(result, 42);
    assert_eq!(attempts.load(Ordering::SeqCst), 1);
  }

  #[tokio::test]
  async fn stops_immediately_when_should_retry_returns_false() {
    let attempts = AtomicUsize::new(0);
    let result: anyhow::Result<()> = with_retry(
      || async {
        attempts.fetch_add(1, Ordering::SeqCst);
        anyhow::bail!("deterministic failure")
      },
      |_| false,
    )
    .await;

    assert!(result.is_err());
    assert_eq!(attempts.load(Ordering::SeqCst), 1);
  }

  fn api_error(status: StatusCode) -> anyhow::Error {
    anyhow::Error::new(OpenAIError::ApiError(
      async_openai::error::ApiErrorResponse {
        status_code: status,
        api_error: async_openai::error::ApiError {
          message: "boom".to_owned(),
          r#type: None,
          param: None,
          code: None,
          // Only `status_code` drives `is_transient`; the payload fields are here
          // because the struct has no `Default`, not because they matter.
          misalignment: None,
        },
      },
    ))
  }

  #[test]
  fn rate_limiting_and_server_errors_are_worth_retrying() {
    for status in [
      StatusCode::TOO_MANY_REQUESTS,
      StatusCode::REQUEST_TIMEOUT,
      StatusCode::CONFLICT,
      StatusCode::INTERNAL_SERVER_ERROR,
      StatusCode::BAD_GATEWAY,
      StatusCode::SERVICE_UNAVAILABLE,
    ] {
      assert!(is_transient(&api_error(status)), "{status} should retry");
    }
  }

  #[test]
  fn deterministic_client_errors_are_not_retried() {
    for status in [
      StatusCode::BAD_REQUEST,
      StatusCode::UNAUTHORIZED,
      StatusCode::FORBIDDEN,
      StatusCode::NOT_FOUND,
      StatusCode::UNPROCESSABLE_ENTITY,
    ] {
      assert!(
        !is_transient(&api_error(status)),
        "{status} repeats identically, so retrying only burns the budget"
      );
    }
  }

  #[test]
  fn request_building_failures_are_not_retried() {
    let err = anyhow::Error::new(OpenAIError::InvalidArgument("bad model".to_owned()));
    assert!(!is_transient(&err));
  }

  /// The predicate only recognizes provider failures it can prove are deterministic;
  /// anything else keeps its retries rather than being downgraded to a hard failure.
  #[test]
  fn unrecognized_errors_keep_their_retries() {
    assert!(is_transient(&anyhow::anyhow!("some other failure")));
  }
}
