//! Multi-turn conversation history behind the HTTP agent API (see
//! [`crate::api::handlers::run`]), behind a pluggable [`SessionStore`] trait.
//!
//! [`crate::agent::Agent`] itself has no notion of a "session" — it is a stateless
//! function of "prior events + new input" (see [`crate::agent::Agent::run_continuing`]).
//! A `SessionStore` is what turns that into an actual multi-turn conversation over HTTP:
//! a client picks its own `sessionId` (see [`crate::api::dto::RunRequest::session_id`])
//! and sends it on every turn; we hand back whatever history is stored under
//! `(tenant token, sessionId)` before the call, and save the updated transcript after.
//!
//! [`MemorySessionStore`] is the only implementation today — good enough for a single
//! process, but it does not survive a restart and is not shared across replicas if this
//! server is ever scaled horizontally. The trait exists precisely so a Redis- or
//! database-backed store can be dropped in later (`AppState.sessions: Arc<dyn
//! SessionStore>`) without touching [`crate::api::handlers::run`] at all.
//!
//! # Known limitations (of [`MemorySessionStore`], not the trait itself)
//!
//! - **Process-local**: history is lost on restart, and is not shared across replicas.
//! - **Last-write-wins**: two concurrent requests for the same `(token, sessionId)` race;
//!   whichever finishes last overwrites the other's turn. Callers that need strict
//!   ordering must serialize their own requests per session.
//! - **Unbounded turn count**: a very long conversation keeps every turn until the whole
//!   session expires — though [`crate::agent::Agent::run_continuing`] does trim the
//!   *token* budget per call (see [`crate::agent::history::trim_to_budget`]), so this
//!   mainly costs memory here, not prompt size sent to the model.

use std::{
  collections::HashMap,
  sync::Mutex,
  time::{Duration, Instant},
};

use crate::agent::Event;

/// Persists multi-turn conversation history, scoped by `(tenant token, client-chosen
/// session id)` so one tenant can never read or overwrite another tenant's conversation,
/// even if both happen to pick the same session id string.
///
/// `async_trait` rather than a native `async fn`: [`crate::api::AppState`] keeps this as
/// `Arc<dyn SessionStore>` so the backend can be swapped per deployment, and native async
/// fns in traits are not dyn-compatible (same reasoning as [`crate::tools::tool::Tool`]).
#[async_trait::async_trait]
pub trait SessionStore: Send + Sync {
  /// Prior turns stored under `(tenant_token, session_id)`, or an empty history when
  /// there is none yet (unknown or expired id) — that is not an error, it just means
  /// this is the first turn of a new conversation under that id.
  async fn history(&self, tenant_token: &str, session_id: &str) -> Vec<Event>;

  /// Persist the transcript after a turn completes, replacing whatever was stored under
  /// this id before.
  async fn save(&self, tenant_token: &str, session_id: &str, history: Vec<Event>);

  /// Drop every session idle past whatever retention policy this implementation uses.
  /// Meant to be called periodically (see `bin/server.rs`); a no-op implementation
  /// (e.g. a store backed by Redis `EXPIRE`/a database TTL index that already expires
  /// entries on its own) is a perfectly valid choice here.
  async fn sweep_expired(&self);
}

struct Session {
  history: Vec<Event>,
  last_used: Instant,
}

/// In-memory [`SessionStore`]: good enough for a single-process deployment or local
/// development; see the module docs for what it does not cover.
pub struct MemorySessionStore {
  sessions: Mutex<HashMap<String, Session>>,
  ttl: Duration,
}

impl MemorySessionStore {
  pub fn new(ttl: Duration) -> Self {
    Self {
      sessions: Mutex::new(HashMap::new()),
      ttl,
    }
  }

  /// Number of sessions currently stored, expired or not. Exposed mainly for tests and
  /// operational logging.
  pub fn len(&self) -> usize {
    self.sessions.lock().unwrap().len()
  }

  pub fn is_empty(&self) -> bool {
    self.sessions.lock().unwrap().is_empty()
  }

  fn expired(&self, session: &Session) -> bool {
    session.last_used.elapsed() > self.ttl
  }

  /// `\0` cannot appear in a bearer token or a JSON string session id, so this cannot
  /// collide between e.g. `("a", "bc")` and `("ab", "c")`.
  fn key(tenant_token: &str, session_id: &str) -> String {
    format!("{tenant_token}\0{session_id}")
  }
}

#[async_trait::async_trait]
impl SessionStore for MemorySessionStore {
  async fn history(&self, tenant_token: &str, session_id: &str) -> Vec<Event> {
    let key = Self::key(tenant_token, session_id);
    let sessions = self.sessions.lock().unwrap();
    match sessions.get(&key) {
      Some(session) if !self.expired(session) => session.history.clone(),
      _ => Vec::new(),
    }
  }

  async fn save(&self, tenant_token: &str, session_id: &str, history: Vec<Event>) {
    let key = Self::key(tenant_token, session_id);
    self.sessions.lock().unwrap().insert(
      key,
      Session {
        history,
        last_used: Instant::now(),
      },
    );
  }

  async fn sweep_expired(&self) {
    self
      .sessions
      .lock()
      .unwrap()
      .retain(|_, session| session.last_used.elapsed() <= self.ttl);
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::agent::ContentItem;

  fn sample_history(text: &str) -> Vec<Event> {
    vec![Event::new(
      "exec",
      "user",
      vec![ContentItem::Message {
        role: "user".to_owned(),
        content: text.to_owned(),
      }],
    )]
  }

  #[tokio::test]
  async fn unknown_session_returns_empty_history() {
    let store = MemorySessionStore::new(Duration::from_secs(60));
    assert!(store.history("token-a", "session-1").await.is_empty());
  }

  #[tokio::test]
  async fn save_then_history_round_trips() {
    let store = MemorySessionStore::new(Duration::from_secs(60));
    store
      .save("token-a", "session-1", sample_history("hi"))
      .await;

    let history = store.history("token-a", "session-1").await;
    assert_eq!(history.len(), 1);
  }

  #[tokio::test]
  async fn sessions_are_isolated_per_tenant_token() {
    let store = MemorySessionStore::new(Duration::from_secs(60));
    store
      .save("token-a", "session-1", sample_history("hi"))
      .await;

    // Same session id, different tenant: must not see tenant-a's history.
    assert!(store.history("token-b", "session-1").await.is_empty());
  }

  #[tokio::test]
  async fn save_replaces_the_previous_history_for_the_same_session() {
    let store = MemorySessionStore::new(Duration::from_secs(60));
    store
      .save("token-a", "session-1", sample_history("first"))
      .await;
    store
      .save(
        "token-a",
        "session-1",
        vec![
          sample_history("first")[0].clone(),
          sample_history("second")[0].clone(),
        ],
      )
      .await;

    assert_eq!(store.history("token-a", "session-1").await.len(), 2);
    assert_eq!(store.len(), 1, "one session, not two");
  }

  #[tokio::test]
  async fn sweep_expired_evicts_sessions_past_the_ttl() {
    let store = MemorySessionStore::new(Duration::from_millis(1));
    store
      .save("token-a", "session-1", sample_history("hi"))
      .await;
    std::thread::sleep(Duration::from_millis(20));

    store.sweep_expired().await;

    assert_eq!(store.len(), 0);
    assert!(store.history("token-a", "session-1").await.is_empty());
  }

  #[tokio::test]
  async fn sweep_expired_keeps_sessions_still_within_the_ttl() {
    let store = MemorySessionStore::new(Duration::from_secs(60));
    store
      .save("token-a", "session-1", sample_history("hi"))
      .await;

    store.sweep_expired().await;

    assert_eq!(store.len(), 1);
  }
}
