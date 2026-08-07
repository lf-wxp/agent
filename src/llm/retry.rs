//! Shared retry policy for a retryable LLM operation.
//!
//! Every call site that retries — a single model request in [`crate::llm::tool_loop::run`]
//! and [`crate::agent::runtime::Agent`], a whole stream in [`crate::llm::stream`], a whole
//! structured solve in [`crate::gaia::solver`] — goes through [`with_retry`], so the attempt
//! budget ([`crate::config::max_retries`]) and backoff curve live in one place instead of
//! being copied (and drifting) at each call site.

use std::future::Future;

use backon::{ExponentialBuilder, Retryable};

use crate::config;

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
}
