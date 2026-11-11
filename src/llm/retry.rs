//! Shared retry policy for a retryable LLM operation.
//!
//! Every call site that retries — a single model request in [`crate::llm::tool_loop::run`]
//! and [`crate::agent::runtime::Agent`], a whole stream in [`crate::llm::stream`], a whole
//! structured solve in [`crate::gaia::solver`] — goes through [`with_retry`], so the attempt
//! budget ([`crate::config::max_retries`]) and backoff curve live in one place instead of
//! being copied (and drifting) at each call site.

use std::{
  future::Future,
  time::{Duration, SystemTime, UNIX_EPOCH},
};

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
/// How long to wait is [`with_retry`]'s business, and it honors the provider's
/// `Retry-After` ([`crate::error::ProviderError::retry_after`]).
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

/// The most a `Retry-After` is stretched by [`jitter`], as a fraction of what was asked.
const RETRY_AFTER_JITTER: f64 = 0.1;

/// The wait before the next attempt: `backoff` (the exponential curve's next step, `None`
/// once its attempts are spent), raised to the provider's `Retry-After` when it asked for
/// longer.
///
/// Backing off *less* than the server asked for is what turns one rate-limit response into
/// a sustained one, so the curve is a floor, not a ceiling. `None` — give up — when the
/// attempts are spent, or when the provider wants more than `max_wait`: a request that
/// would sit in a retry loop for minutes is better reported than silently stalled.
///
/// A wait that comes from `Retry-After` is stretched by up to [`RETRY_AFTER_JITTER`] of
/// itself, scaled by `jitter` (in `0.0..=1.0`): callers rate limited together all hear the
/// same number, and would otherwise all come back at the same instant. It only ever adds to
/// what was asked, and `max_wait` is judged against what was asked, not the stretch.
fn next_wait(
  err: &Error,
  backoff: Option<Duration>,
  max_wait: Duration,
  jitter: f64,
) -> Option<Duration> {
  let backoff = backoff?;
  let asked = err
    .as_provider()
    .and_then(|provider| provider.retry_after());
  match asked {
    None => Some(backoff),
    Some(asked) if asked > max_wait => None,
    Some(asked) => {
      let stretch = asked.mul_f64(RETRY_AFTER_JITTER * jitter.clamp(0.0, 1.0));
      Some(backoff.max(asked.saturating_add(stretch)))
    }
  }
}

/// A value in `0.0..1.0` that differs between callers failing at the same moment: the
/// sub-second part of the clock. Telling callers apart is all it is for, not randomness of
/// any quality.
fn jitter() -> f64 {
  SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .map_or(0.0, |elapsed| f64::from(elapsed.subsec_nanos()) / 1e9)
}

/// Retry `op` with exponential backoff, skipping failures for which `should_retry` returns
/// `false`.
///
/// A provider that says how long to wait (`Retry-After`) is waited for, if that is within
/// [`config::retry_max_wait`] (`LLM_RETRY_MAX_WAIT_SECS`), plus a little jitter; a longer
/// request ends the retrying and the error is returned.
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
  let max_wait = config::retry_max_wait();
  op.retry(ExponentialBuilder::default().with_max_times(config::max_retries()))
    .when(should_retry)
    .adjust(|err, backoff| next_wait(err, backoff, max_wait, jitter()))
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

  fn rate_limited(asked: Option<u64>) -> Error {
    let error = ProviderError::transient("429").with_status(429);
    Error::from(match asked {
      Some(secs) => error.with_retry_after(Duration::from_secs(secs)),
      None => error,
    })
  }

  const LIMIT: Duration = Duration::from_secs(60);

  /// No jitter, so a wait can be compared exactly.
  const STEADY: f64 = 0.0;

  /// No `Retry-After`: the curve's own step, unchanged.
  #[test]
  fn without_retry_after_the_backoff_is_the_curves() {
    let backoff = Some(Duration::from_secs(1));
    assert_eq!(
      next_wait(&rate_limited(None), backoff, LIMIT, STEADY),
      backoff
    );
    assert_eq!(
      next_wait(&Error::other("x"), backoff, LIMIT, STEADY),
      backoff,
      "an error that is not a provider's has no header to read"
    );
  }

  /// Backing off less than asked is what makes one 429 into many, so the longer wins.
  #[test]
  fn retry_after_raises_the_wait_but_never_lowers_it() {
    let one = Some(Duration::from_secs(1));
    assert_eq!(
      next_wait(&rate_limited(Some(7)), one, LIMIT, STEADY),
      Some(Duration::from_secs(7))
    );
    let four = Some(Duration::from_secs(4));
    assert_eq!(
      next_wait(&rate_limited(Some(2)), four, LIMIT, STEADY),
      four,
      "the curve was already longer"
    );
  }

  #[test]
  fn a_wait_over_the_limit_ends_the_retrying() {
    let one = Some(Duration::from_secs(1));
    assert_eq!(next_wait(&rate_limited(Some(61)), one, LIMIT, STEADY), None);
    assert_eq!(
      next_wait(&rate_limited(Some(60)), one, LIMIT, STEADY),
      Some(Duration::from_secs(60)),
      "exactly the limit is still waited out"
    );
  }

  /// The attempts running out still ends it, whatever the header says.
  #[test]
  fn spent_attempts_stay_spent() {
    assert_eq!(next_wait(&rate_limited(Some(1)), None, LIMIT, STEADY), None);
  }

  /// `context` layers keep what the provider asked for, like they keep its status.
  #[test]
  fn retry_after_survives_added_context() {
    let err = rate_limited(Some(5)).context("calling the backend");
    assert_eq!(
      next_wait(&err, Some(Duration::from_secs(1)), LIMIT, STEADY),
      Some(Duration::from_secs(5))
    );
  }

  /// Jitter only ever adds to what was asked, never more than a tenth of it, and does not
  /// touch a wait the curve already made longer.
  #[test]
  fn jitter_stretches_a_retry_after_but_never_shortens_it() {
    let one = Some(Duration::from_secs(1));
    let asked = rate_limited(Some(10));
    assert_eq!(
      next_wait(&asked, one, LIMIT, 0.0),
      Some(Duration::from_secs(10))
    );
    assert_eq!(
      next_wait(&asked, one, LIMIT, 0.5),
      Some(Duration::from_millis(10_500))
    );
    assert_eq!(
      next_wait(&asked, one, LIMIT, 1.0),
      Some(Duration::from_secs(11))
    );
    assert_eq!(
      next_wait(&asked, one, LIMIT, 7.0),
      Some(Duration::from_secs(11)),
      "out-of-range jitter is clamped"
    );
    let long_curve = Some(Duration::from_secs(30));
    assert_eq!(next_wait(&asked, long_curve, LIMIT, 1.0), long_curve);
    assert_eq!(
      next_wait(&rate_limited(None), one, LIMIT, 1.0),
      one,
      "no Retry-After, nothing to stretch"
    );
    assert_eq!(
      next_wait(&rate_limited(Some(60)), one, LIMIT, 1.0),
      Some(Duration::from_secs(66)),
      "the limit is judged against what was asked"
    );
  }

  /// A provider can send any number, and the limit is configurable up to the largest one:
  /// stretching it must not overflow.
  #[test]
  fn an_extreme_retry_after_does_not_overflow_when_stretched() {
    let huge = Duration::from_secs(u64::MAX);
    let err = Error::from(ProviderError::transient("429").with_retry_after(huge));
    let wait = next_wait(&err, Some(Duration::from_secs(1)), huge, 1.0);
    assert!(wait.is_some_and(|wait| wait >= huge));
  }

  #[test]
  fn the_jitter_source_stays_within_its_range() {
    for _ in 0..100 {
      assert!((0.0..1.0).contains(&jitter()));
    }
  }

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
