//! Multi-tenancy unit: a [`Provider`] bundles the credentials/endpoint used to reach an
//! LLM backend together with the concurrency budget for that account, so multiple tenants
//! (different API keys, different rate limits, potentially different base URLs) can be
//! served concurrently from one process.
//!
//! Every request-issuing entry point in [`crate::llm`] and [`crate::agent`] takes a
//! `&Provider` instead of reaching for a process-wide global, and acquires a permit from
//! it for the exact duration of each model call — not the whole tool-calling loop — so a
//! task waiting on a slow tool no longer occupies a slot of the shared budget while it
//! does so, and the limit is enforced centrally instead of relying on each caller to
//! remember to do it. [`Provider::shared`] preserves the previous single-tenant,
//! environment-variable-configured behavior for callers that do not need per-tenant
//! isolation.
//!
//! # Multi-tenant usage
//!
//! ```no_run
//! use agent::llm::provider::Provider;
//! use async_openai::config::OpenAIConfig;
//!
//! // Each tenant gets its own credentials and its own concurrency budget: tenant `a`
//! // running hot never steals capacity from (or shares a rate limit with) tenant `b`.
//! let tenant_a = Provider::new(
//!   OpenAIConfig::new().with_api_key("sk-tenant-a"),
//!   5,
//! );
//! let tenant_b = Provider::new(
//!   OpenAIConfig::new()
//!     .with_api_key("sk-tenant-b")
//!     .with_api_base("https://tenant-b.example.com/v1"),
//!   1,
//! );
//! ```

use std::sync::{Arc, LazyLock};

use async_openai::{Client, config::OpenAIConfig};
use tokio::sync::{AcquireError, Semaphore, SemaphorePermit};

use crate::config;

/// Credentials, endpoint and concurrency budget for one tenant.
///
/// Cheap to clone: cloning only bumps a couple of `Arc` reference counts (the HTTP
/// client and the semaphore are both internally reference-counted), so the same
/// `Provider` can be handed to as many concurrent [`crate::agent::Agent`]s or `llm` calls
/// as needed without extra synchronization on the caller's part.
#[derive(Clone, Debug)]
pub struct Provider {
  client: Client<OpenAIConfig>,
  // `Semaphore` itself is not `Clone`; wrapping it lets `Provider` stay `Clone` while
  // every clone still shares (and is bound by) the same underlying budget.
  semaphore: Arc<Semaphore>,
}

impl Provider {
  /// Build a provider from an explicit `async-openai` config and a concurrency budget.
  ///
  /// Use this for multi-tenant setups: give each tenant its own [`OpenAIConfig`] (API
  /// key, base URL) and its own `max_concurrency`, so one tenant's traffic can never
  /// starve another's, and revoking one tenant's key never touches the others.
  /// `max_concurrency` is clamped to at least 1 (a semaphore of size 0 would deadlock
  /// every caller forever).
  pub fn new(config: OpenAIConfig, max_concurrency: usize) -> Self {
    Self {
      client: Client::with_config(config),
      semaphore: Arc::new(Semaphore::new(max_concurrency.max(1))),
    }
  }

  /// The process-wide default provider: credentials come from `async-openai`'s own
  /// environment handling (`OPENAI_API_KEY` / `OPENAI_API_BASE`), concurrency from
  /// [`config::max_concurrency`] (`LLM_MAX_CONCURRENCY`). Kept for single-tenant callers
  /// that do not need per-tenant isolation — every example and binary in this crate uses
  /// it unless it needs multiple tenants.
  pub fn shared() -> &'static Provider {
    static SHARED: LazyLock<Provider> =
      LazyLock::new(|| Provider::new(OpenAIConfig::default(), config::max_concurrency()));
    &SHARED
  }

  /// The underlying HTTP-level client for this tenant.
  pub fn client(&self) -> &Client<OpenAIConfig> {
    &self.client
  }

  /// Acquire one slot of this tenant's concurrency budget, held until the returned
  /// permit is dropped.
  pub async fn acquire(&self) -> Result<SemaphorePermit<'_>, AcquireError> {
    self.semaphore.acquire().await
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn cloning_shares_the_same_concurrency_budget() {
    let provider = Provider::new(OpenAIConfig::default(), 1);
    let clone = provider.clone();

    let _permit = provider.semaphore.try_acquire().unwrap();
    // The clone shares the same underlying semaphore, so its single slot is already spent.
    assert!(clone.semaphore.try_acquire().is_err());
  }

  #[test]
  fn zero_max_concurrency_is_clamped_to_one() {
    let provider = Provider::new(OpenAIConfig::default(), 0);
    assert!(provider.semaphore.try_acquire().is_ok());
  }

  #[test]
  fn shared_returns_the_same_instance_every_time() {
    let first = Provider::shared();
    let second = Provider::shared();
    assert!(std::ptr::eq(first, second));
  }
}
