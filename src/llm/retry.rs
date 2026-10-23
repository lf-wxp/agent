//! Shared retry policy for a retryable LLM operation.
//!
//! Every call site that retries — a single model request in [`crate::llm::tool_loop::run`]
//! and [`crate::agent::runtime::Agent`], a whole stream in [`crate::llm::stream`], a whole
//! structured solve in [`crate::gaia::solver`] — goes through [`with_retry`], so the attempt
//! budget ([`crate::config::max_retries`]) and backoff curve live in one place instead of
//! being copied (and drifting) at each call site.

use std::future::Future;

use backon::{ExponentialBuilder, Retryable};

use crate::{Error, Result, config};

/// Whether `err` is the kind of failure another attempt could plausibly survive.
///
/// - [`Error::Provider`]: what the provider error says ([`crate::error::ProviderError::is_retryable`]).
///   For the provider that ships that is transport errors, rate limiting (429), a request
///   timeout (408), a conflict (409), any 5xx and a response that did not parse; any
///   other 4xx — a malformed request, a rejected key, a model the tenant cannot reach —
///   fails identically on every attempt. A custom backend says which of its own errors
///   are which with [`crate::error::ProviderError::transient`] /
///   [`crate::error::ProviderError::permanent`].
/// - Never: [`Error::InvalidInput`] and [`Error::Config`] (nothing about them changes
///   between attempts), [`Error::Cancelled`] (the caller asked to stop),
///   [`Error::TruncatedOutput`] and [`Error::BudgetExhausted`] (a retry spends the same
///   tokens, or the whole tool budget, to fail the same way).
/// - Everything else is retried: this predicate only rules out failures it can show are
///   deterministic, and guessing at the rest would turn a recoverable blip into a hard
///   failure.
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
pub fn is_transient(err: &Error) -> bool {
  match err {
    Error::Provider(provider) => provider.is_retryable(),
    Error::InvalidInput(_)
    | Error::Config(_)
    | Error::Cancelled(_)
    | Error::TruncatedOutput(_)
    | Error::BudgetExhausted(_) => false,
    _ => true,
  }
}

/// Retry `op` with exponential backoff, skipping failures for which `should_retry` returns
/// `false`.
///
/// Pass [`is_transient`] unless the operation has deterministic failures of its own it
/// knows about; `|_| true` retries everything.
pub async fn with_retry<T, Fut>(
  op: impl FnMut() -> Fut,
  should_retry: impl Fn(&Error) -> bool,
) -> Result<T>
where
  Fut: Future<Output = Result<T>>,
{
  op.retry(ExponentialBuilder::default().with_max_times(config::max_retries()))
    .when(should_retry)
    .notify(|err, dur| tracing::warn!("retrying after {dur:?}: {err}"))
    .await
}

#[cfg(test)]
mod tests {
  use std::sync::atomic::{AtomicUsize, Ordering};

  use async_openai::error::OpenAIError;
  use reqwest::StatusCode;

  use super::*;
  use crate::error::{BudgetExhausted, ProviderError, TruncatedOutput};

  #[tokio::test]
  async fn succeeds_without_retrying_on_the_first_try() {
    let attempts = AtomicUsize::new(0);
    let result = with_retry(
      || async {
        attempts.fetch_add(1, Ordering::SeqCst);
        Ok::<_, Error>(42)
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
    let result: Result<()> = with_retry(
      || async {
        attempts.fetch_add(1, Ordering::SeqCst);
        Err(Error::other("deterministic failure"))
      },
      |_| false,
    )
    .await;

    assert!(result.is_err());
    assert_eq!(attempts.load(Ordering::SeqCst), 1);
  }

  fn api_error(status: StatusCode) -> Error {
    Error::from(OpenAIError::ApiError(
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
    let err = Error::from(OpenAIError::InvalidArgument("bad model".to_owned()));
    assert!(!is_transient(&err));
  }

  /// The predicate only rules out failures it can show are deterministic; anything else
  /// keeps its retries rather than being downgraded to a hard failure.
  #[test]
  fn unrecognized_errors_keep_their_retries() {
    assert!(is_transient(&Error::other("some other failure")));
    assert!(is_transient(&Error::tool("a tool hiccup")));
  }

  /// A custom backend's own deterministic error opts out of retries, and keeps doing so
  /// under a message added on the way up.
  #[test]
  fn a_permanent_provider_error_is_not_retried_even_with_context() {
    let err = Error::from(ProviderError::permanent("key rejected")).context("calling the backend");
    assert!(!is_transient(&err));
    assert_eq!(format!("{err:#}"), "calling the backend: key rejected");

    let transient = Error::from(ProviderError::transient("connection reset"));
    assert!(is_transient(&transient));
  }

  /// What `gaia::solver` used to exclude by hand.
  #[test]
  fn truncation_budget_exhaustion_and_bad_input_are_not_retried() {
    assert!(!is_transient(&TruncatedOutput { limit: 16 }.into()));
    assert!(!is_transient(
      &BudgetExhausted::new(Error::other("x")).into()
    ));
    assert!(!is_transient(&Error::invalid_input("empty prompt")));
    assert!(!is_transient(&Error::config("EMBED_MODEL is not set")));
  }
}
