//! Idempotency support for `POST /v1/agent/run`: an `Idempotency-Key` request header
//! (Stripe's convention) lets a client safely retry a request that may or may not have
//! reached the server the first time (e.g. after a network timeout) without risking a
//! second, full agent run — which would spend tokens twice and, if any of the agent's
//! tools have side effects, could repeat those too.
//!
//! Scoped by `(tenant token, key)`, the same isolation rule as
//! [`crate::api::session::SessionStore`]: one tenant can never collide with — or read —
//! another tenant's idempotency record even if both happen to pick the same key string.
//!
//! # What this does and does not guard against
//!
//! - Two requests for the *same* key that race each other: the second one gets a `409`
//!   ("already being processed") rather than also running the agent. It does not block
//!   and wait for the first one to finish; the caller is expected to retry, which is what
//!   makes the key idempotent to begin with — no matter how many times a client retries
//!   with the same key, the agent runs at most once.
//! - A failed run releases its key immediately, so a legitimate retry after a real
//!   failure is not permanently stuck behind a `409`.
//! - Successful responses are cached for the TTL; a repeat request with the same key in
//!   that window gets the exact same response body back without re-running anything.
//! - It is in-memory and process-local, same tradeoff as [`crate::api::session::SessionStore`]'s
//!   default backend: fine for a single instance, not shared across replicas.

use std::{
  collections::HashMap,
  sync::Mutex,
  time::{Duration, Instant},
};

use serde_json::Value;

enum Entry {
  InProgress { since: Instant },
  Done { response: Value, since: Instant },
}

/// Outcome of [`IdempotencyStore::reserve`].
pub enum Reservation {
  /// No record for this key yet (or a stale one from a run that never completed): the
  /// caller should run the agent, then call [`IdempotencyStore::complete`] on success or
  /// [`IdempotencyStore::release`] on failure.
  Fresh,
  /// Already finished once: here is the exact response body to hand back, no need to
  /// run anything.
  Duplicate(Value),
  /// Another request with this key is currently running. The caller should reject this
  /// request (`409`) rather than run the agent a second time concurrently.
  InProgress,
}

/// Bearer-token-scoped idempotency cache, keyed by `(tenant token, client-supplied
/// `Idempotency-Key`)`.
pub struct IdempotencyStore {
  entries: Mutex<HashMap<String, Entry>>,
  ttl: Duration,
}

impl IdempotencyStore {
  pub fn new(ttl: Duration) -> Self {
    Self {
      entries: Mutex::new(HashMap::new()),
      ttl,
    }
  }

  /// Look up `key` and, if it is unused (or its previous record expired), atomically
  /// claim it as in-progress so a concurrent caller cannot also claim it — hence one
  /// combined "check, then claim" call through a single lock rather than two separate
  /// ones, which would race.
  pub fn reserve(&self, tenant_token: &str, key: &str) -> Reservation {
    let full_key = Self::key(tenant_token, key);
    let mut entries = self.entries.lock().unwrap();
    match entries.get(&full_key) {
      Some(Entry::Done { response, since }) if since.elapsed() <= self.ttl => {
        return Reservation::Duplicate(response.clone());
      }
      Some(Entry::InProgress { since }) if since.elapsed() <= self.ttl => {
        return Reservation::InProgress;
      }
      _ => {}
    }
    entries.insert(
      full_key,
      Entry::InProgress {
        since: Instant::now(),
      },
    );
    Reservation::Fresh
  }

  /// Record a successful outcome, so a repeat request with the same key gets it back
  /// verbatim instead of running the agent again.
  pub fn complete(&self, tenant_token: &str, key: &str, response: Value) {
    let full_key = Self::key(tenant_token, key);
    self.entries.lock().unwrap().insert(
      full_key,
      Entry::Done {
        response,
        since: Instant::now(),
      },
    );
  }

  /// Free a key after a failed run, so a legitimate retry is not stuck behind a `409`
  /// forever.
  pub fn release(&self, tenant_token: &str, key: &str) {
    let full_key = Self::key(tenant_token, key);
    self.entries.lock().unwrap().remove(&full_key);
  }

  /// Drop every record past its TTL. Meant to be called periodically (see
  /// `bin/server.rs`), same pattern as
  /// [`crate::api::session::SessionStore::sweep_expired`].
  pub fn sweep_expired(&self) {
    let ttl = self.ttl;
    self.entries.lock().unwrap().retain(|_, entry| {
      let since = match entry {
        Entry::InProgress { since } | Entry::Done { since, .. } => *since,
      };
      since.elapsed() <= ttl
    });
  }

  pub fn len(&self) -> usize {
    self.entries.lock().unwrap().len()
  }

  pub fn is_empty(&self) -> bool {
    self.entries.lock().unwrap().is_empty()
  }

  /// `\0` cannot appear in a bearer token or an HTTP header value, so this cannot
  /// collide between e.g. `("a", "bc")` and `("ab", "c")`.
  fn key(tenant_token: &str, key: &str) -> String {
    format!("{tenant_token}\0{key}")
  }
}

#[cfg(test)]
mod tests {
  use serde_json::json;

  use super::*;

  #[test]
  fn fresh_key_can_be_reserved() {
    let store = IdempotencyStore::new(Duration::from_secs(60));
    assert!(matches!(
      store.reserve("token-a", "key-1"),
      Reservation::Fresh
    ));
  }

  #[test]
  fn reserving_twice_before_completion_reports_in_progress() {
    let store = IdempotencyStore::new(Duration::from_secs(60));
    store.reserve("token-a", "key-1");
    assert!(matches!(
      store.reserve("token-a", "key-1"),
      Reservation::InProgress
    ));
  }

  #[test]
  fn completed_key_returns_the_cached_response() {
    let store = IdempotencyStore::new(Duration::from_secs(60));
    store.reserve("token-a", "key-1");
    store.complete("token-a", "key-1", json!({"output": "42"}));

    match store.reserve("token-a", "key-1") {
      Reservation::Duplicate(response) => assert_eq!(response, json!({"output": "42"})),
      _ => panic!("expected a duplicate"),
    }
  }

  #[test]
  fn released_key_can_be_reserved_again() {
    let store = IdempotencyStore::new(Duration::from_secs(60));
    store.reserve("token-a", "key-1");
    store.release("token-a", "key-1");
    assert!(matches!(
      store.reserve("token-a", "key-1"),
      Reservation::Fresh
    ));
  }

  #[test]
  fn keys_are_isolated_per_tenant_token() {
    let store = IdempotencyStore::new(Duration::from_secs(60));
    store.reserve("token-a", "key-1");
    store.complete("token-a", "key-1", json!({"output": "42"}));

    // Same key, different tenant: must be treated as fresh, not a duplicate.
    assert!(matches!(
      store.reserve("token-b", "key-1"),
      Reservation::Fresh
    ));
  }

  #[test]
  fn sweep_expired_evicts_records_past_the_ttl() {
    let store = IdempotencyStore::new(Duration::from_millis(1));
    store.reserve("token-a", "key-1");
    store.complete("token-a", "key-1", json!({"output": "42"}));
    std::thread::sleep(Duration::from_millis(20));

    store.sweep_expired();

    assert!(store.is_empty());
  }

  #[test]
  fn sweep_expired_keeps_records_still_within_the_ttl() {
    let store = IdempotencyStore::new(Duration::from_secs(60));
    store.reserve("token-a", "key-1");

    store.sweep_expired();

    assert_eq!(store.len(), 1);
  }
}
