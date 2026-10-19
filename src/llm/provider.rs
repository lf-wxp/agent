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
//!
//! # Other backends
//!
//! [`Provider::new`] talks to any OpenAI-compatible endpoint (see
//! [`crate::llm::backend::OpenAiCompatible`]). Anything else — a native client for a
//! provider that is not compatible, an in-process model, a test double — implements
//! [`ChatBackend`] and is wrapped with [`Provider::with_backend`]; everything built on a
//! `Provider` then works with it unchanged.

use std::sync::{Arc, LazyLock};

use async_openai::config::OpenAIConfig;
use tokio::sync::{AcquireError, Semaphore, SemaphorePermit};

use crate::{
  config,
  llm::backend::{ChatBackend, ChatChunkStream, ChatRequest, ChatResponse, OpenAiCompatible},
};

/// Backend and concurrency budget for one tenant.
///
/// Cheap to clone: cloning only bumps a couple of `Arc` reference counts (the backend
/// and the semaphore are both shared), so the same `Provider` can be handed to as many
/// concurrent [`crate::agent::Agent`]s or `llm` calls as needed without extra
/// synchronization on the caller's part.
#[derive(Clone)]
pub struct Provider {
  backend: Arc<dyn ChatBackend>,
  // `Semaphore` itself is not `Clone`; wrapping it lets `Provider` stay `Clone` while
  // every clone still shares (and is bound by) the same underlying budget.
  semaphore: Arc<Semaphore>,
}

impl std::fmt::Debug for Provider {
  /// The backend is a trait object with no `Debug` requirement, and may hold
  /// credentials besides; the budget is the part worth seeing.
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("Provider")
      .field("available_permits", &self.semaphore.available_permits())
      .finish_non_exhaustive()
  }
}

impl Provider {
  /// Build a provider for an OpenAI-compatible endpoint from an explicit `async-openai`
  /// config and a concurrency budget.
  ///
  /// Use this for multi-tenant setups: give each tenant its own [`OpenAIConfig`] (API
  /// key, base URL) and its own `max_concurrency`, so one tenant's traffic can never
  /// starve another's, and revoking one tenant's key never touches the others.
  /// `max_concurrency` is clamped to at least 1 (a semaphore of size 0 would deadlock
  /// every caller forever).
  pub fn new(config: OpenAIConfig, max_concurrency: usize) -> Self {
    Self::with_backend(Arc::new(OpenAiCompatible::new(config)), max_concurrency)
  }

  /// Build a provider around any [`ChatBackend`], with the same concurrency budget
  /// semantics as [`Self::new`].
  pub fn with_backend(backend: Arc<dyn ChatBackend>, max_concurrency: usize) -> Self {
    Self {
      backend,
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

  /// The backend requests are sent to. Prefer [`Self::chat`] / [`Self::chat_stream`],
  /// which also charge the call against this tenant's budget.
  pub fn backend(&self) -> &dyn ChatBackend {
    self.backend.as_ref()
  }

  /// Acquire one slot of this tenant's concurrency budget, held until the returned
  /// permit is dropped.
  pub async fn acquire(&self) -> Result<SemaphorePermit<'_>, AcquireError> {
    self.semaphore.acquire().await
  }

  /// Send one request, holding a slot of this tenant's budget until the response is in.
  ///
  /// No retry here: callers decide what is worth another attempt (see
  /// [`crate::llm::retry::with_retry`]), since only they know whether, say, a truncated
  /// answer is a failure.
  ///
  /// Takes a [`ChatRequest`] rather than the typed request so that a caller replaying
  /// reasoning cannot lose it by passing the wrong one; a request with nothing to replay
  /// converts with `.into()`.
  pub async fn chat(&self, request: ChatRequest) -> anyhow::Result<ChatResponse> {
    let _permit = self.acquire().await?;
    self.backend.create(request).await
  }

  /// Open a streamed response, holding a slot of this tenant's budget only while the
  /// stream is being opened — not while it is consumed, which can take as long as the
  /// caller spends between chunks (running tools, waiting on a human).
  pub async fn chat_stream(&self, request: ChatRequest) -> anyhow::Result<ChatChunkStream> {
    let _permit = self.acquire().await?;
    self.backend.create_stream(request).await
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

  /// Records how many budget slots were free while it was being called.
  struct BudgetProbe {
    free_during_call: std::sync::Mutex<Option<usize>>,
    semaphore: std::sync::OnceLock<Arc<Semaphore>>,
  }

  #[async_trait::async_trait]
  impl ChatBackend for BudgetProbe {
    async fn create(&self, _request: ChatRequest) -> anyhow::Result<ChatResponse> {
      let free = self.semaphore.get().map(|s| s.available_permits());
      *self.free_during_call.lock().unwrap() = free;
      crate::llm::backend::parse_response(serde_json::json!({
        "id": "x", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{ "index": 0, "message": { "role": "assistant", "content": "hi" },
                      "finish_reason": "stop" }],
      }))
    }

    async fn create_stream(&self, _request: ChatRequest) -> anyhow::Result<ChatChunkStream> {
      anyhow::bail!("not used")
    }
  }

  /// A custom backend gets every request, and is charged against the tenant's budget
  /// exactly like the built-in one: one slot while the call runs, none afterwards.
  #[tokio::test]
  async fn a_custom_backend_is_called_under_the_tenants_budget() {
    let probe = Arc::new(BudgetProbe {
      free_during_call: std::sync::Mutex::new(None),
      semaphore: std::sync::OnceLock::new(),
    });
    let provider = Provider::with_backend(Arc::clone(&probe) as Arc<dyn ChatBackend>, 2);
    probe
      .semaphore
      .set(Arc::clone(&provider.semaphore))
      .unwrap();

    let request = async_openai::types::chat::CreateChatCompletionRequestArgs::default()
      .model("m")
      .messages(Vec::new())
      .build()
      .unwrap();
    let response = provider.chat(request.into()).await.unwrap();

    assert_eq!(
      response.response.choices[0].message.content.as_deref(),
      Some("hi")
    );
    assert_eq!(*probe.free_during_call.lock().unwrap(), Some(1));
    assert_eq!(provider.semaphore.available_permits(), 2, "released after");
  }
}
