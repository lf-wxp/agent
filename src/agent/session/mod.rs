//! Pluggable multi-turn conversation history for [`crate::agent::Agent`], behind a
//! [`SessionStore`] trait.
//!
//! [`Agent`](crate::agent::Agent) itself has no notion of a "session" — it is a stateless
//! function of "prior events + new input" (see [`crate::agent::Agent::run_continuing`]).
//! A `SessionStore` is what turns that into an actual multi-turn conversation for a given
//! front-end: a caller picks its own `session_id` (e.g. a CLI's own `--session` flag, or a
//! web front-end's own conversation id) and sends it on every turn; we hand back whatever
//! history is stored under `(scope, session_id)` before the call, and save the updated
//! transcript after.
//!
//! `scope` isolates one caller-defined namespace from another under the same
//! `session_id` — e.g. a terminal-originated turn and a web-originated turn sharing one
//! process could use different scopes if they should never see each other's history; a
//! single-front-end caller can just pass a constant.
//!
//! This module holds only the trait; each backend lives in its own module beside it.
//! [`mod@file`] is the one shipped here: [`FileSessionStore`] persists one JSON file per
//! `(scope, session_id)` on disk, so a short-lived process — a CLI invocation, most
//! obviously — can pick a conversation back up on its *next* invocation.
//!
//! The trait exists so a Redis- or database-backed store could be dropped in later too
//! (anywhere this is held as `Arc<dyn SessionStore>`) without touching the caller at all.

pub mod file;

pub use file::{FileSessionStore, SessionSummary};

use crate::agent::Event;

/// Persists multi-turn conversation history, scoped by `(scope, client-chosen session
/// id)` so one scope can never read or overwrite another scope's conversation, even if
/// both happen to pick the same session id string.
///
/// `async_trait` rather than a native `async fn`: callers keep this as `Arc<dyn
/// SessionStore>` so the backend can be swapped per deployment, and native async fns in
/// traits are not dyn-compatible (same reasoning as [`crate::tools::tool::Tool`]).
#[async_trait::async_trait]
pub trait SessionStore: Send + Sync {
  /// Prior turns stored under `(scope, session_id)`, or an empty history when there is
  /// none yet (unknown or expired id) — that is not an error, it just means this is the
  /// first turn of a new conversation under that id.
  async fn history(&self, scope: &str, session_id: &str) -> Vec<Event>;

  /// Persist the transcript after a turn completes, replacing whatever was stored under
  /// this id before.
  async fn save(&self, scope: &str, session_id: &str, history: Vec<Event>);

  /// Drop every session idle past whatever retention policy this implementation uses.
  /// Meant to be called periodically by whatever long-running process owns this store;
  /// a no-op implementation (e.g. a store backed by Redis `EXPIRE`/a database TTL index
  /// that already expires entries on its own) is a perfectly valid choice here.
  async fn sweep_expired(&self);
}
