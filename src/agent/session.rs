//! Pluggable multi-turn conversation history for [`crate::agent::Agent`], behind a
//! [`SessionStore`] trait.
//!
//! [`Agent`](crate::agent::Agent) itself has no notion of a "session" — it is a stateless
//! function of "prior events + new input" (see [`crate::agent::Agent::run_continuing`]).
//! A `SessionStore` is what turns that into an actual multi-turn conversation for a given
//! front-end: a caller picks its own `session_id` (e.g. [`crate::api::dto::RunRequest::
//! session_id`] for the HTTP API, or a CLI's own `--session` flag) and sends it on every
//! turn; we hand back whatever history is stored under `(scope, session_id)` before the
//! call, and save the updated transcript after.
//!
//! `scope` isolates one caller-defined namespace from another under the same
//! `session_id` — the HTTP API uses the authenticated tenant's bearer token (see
//! [`crate::api::handlers::AuthenticatedTenant::token`]) so one tenant can never read or
//! overwrite another tenant's conversation; a CLI with no multi-tenant concept can just
//! pass a constant.
//!
//! Two implementations exist today:
//!
//! - [`MemorySessionStore`]: good enough for a single HTTP server process (see
//!   [`crate::api::AppState::sessions`]), but history is lost on restart and is not
//!   shared across replicas.
//! - [`FileSessionStore`]: one JSON file per `(scope, session_id)` on disk, so a
//!   short-lived process — a CLI invocation, most obviously — can pick a conversation
//!   back up on its *next* invocation, which an in-memory store cannot do at all.
//!
//! The trait exists precisely so a Redis- or database-backed store can be dropped in
//! later too (anywhere this is held as `Arc<dyn SessionStore>`) without touching the
//! caller at all.
//!
//! # Known limitations
//!
//! Both implementations here are single-machine:
//!
//! - **Not shared across replicas**: a session lives in one process's memory or on one
//!   machine's disk; a horizontally scaled HTTP server needs a Redis-/database-backed
//!   implementation instead.
//! - **Last-write-wins**: two concurrent callers for the same `(scope, session_id)` race;
//!   whichever finishes last overwrites the other's turn. Callers that need strict
//!   ordering must serialize their own requests per session.
//! - **Unbounded turn count**: a very long conversation keeps every turn until the whole
//!   session expires — though [`crate::agent::Agent::run_continuing`] does trim the
//!   *token* budget per call (see [`crate::agent::history::trim_to_budget`]), so this
//!   mainly costs memory/disk here, not prompt size sent to the model.
//! - **Plaintext on disk** ([`FileSessionStore`] only): transcripts are written as
//!   plaintext JSON. The directory is created `0700` (owner-only) on Unix so other local
//!   users cannot read them, but this is not encryption at rest — do not point it at a
//!   world-readable location, and treat the machine's disk as trusted.

use std::{
  collections::HashMap,
  path::{Path, PathBuf},
  sync::Mutex,
  time::{Duration, Instant, SystemTime},
};

use base64::Engine;

use crate::agent::Event;

/// One stored session as reported by [`FileSessionStore::list`]: enough to show a human
/// which conversations exist and let them pick one, without loading the full transcript.
#[derive(Debug, Clone)]
pub struct SessionSummary {
  pub session_id: String,
  /// Number of events in the stored transcript. Not the same as "number of chat turns"
  /// (one turn is a user message plus an assistant reply, i.e. roughly two events), but
  /// a monotonically increasing size proxy either way.
  pub turns: usize,
  /// The file's mtime — same clock [`FileSessionStore::sweep_expired`] uses to decide
  /// expiry, so "last active" here matches whatever expiry policy applies.
  pub last_used: SystemTime,
}

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

  /// `\0` cannot appear in a bearer token, a CLI-supplied scope, or a JSON string
  /// session id, so this cannot collide between e.g. `("a", "bc")` and `("ab", "c")`.
  fn key(scope: &str, session_id: &str) -> String {
    format!("{scope}\0{session_id}")
  }
}

#[async_trait::async_trait]
impl SessionStore for MemorySessionStore {
  async fn history(&self, scope: &str, session_id: &str) -> Vec<Event> {
    let key = Self::key(scope, session_id);
    let sessions = self.sessions.lock().unwrap();
    match sessions.get(&key) {
      Some(session) if !self.expired(session) => session.history.clone(),
      _ => Vec::new(),
    }
  }

  async fn save(&self, scope: &str, session_id: &str, history: Vec<Event>) {
    let key = Self::key(scope, session_id);
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

/// [`SessionStore`] that persists each `(scope, session_id)` as one JSON file under
/// `dir` — one process's [`MemorySessionStore`] loses everything on exit, which is fine
/// for a long-running HTTP server (the next request just starts a fresh conversation)
/// but wrong for a CLI, where "continue my last conversation" has to survive the process
/// exiting between invocations.
///
/// Recency for [`SessionStore::sweep_expired`] and expiry is tracked via the file's own
/// mtime (updated by every [`SessionStore::save`]) rather than an in-memory clock, so it
/// stays correct across restarts without needing a second piece of state to keep in sync
/// with the first.
///
/// A `ttl` of `None` means "never expire": [`SessionStore::history`] always reads a
/// stored file and [`SessionStore::sweep_expired`] deletes nothing. This is what the CLI
/// uses — "continue the conversation I had last week" is a normal thing to want from a
/// command-line tool, whereas the HTTP server (whose sessions are ephemeral per client)
/// passes a finite TTL so idle conversations are eventually reclaimed.
pub struct FileSessionStore {
  dir: PathBuf,
  /// `None` = never expire (see the type docs); `Some(ttl)` = expire after `ttl` of
  /// inactivity, measured from the file's mtime.
  ttl: Option<Duration>,
}

impl FileSessionStore {
  /// A store whose sessions expire after `ttl` of inactivity. `dir` is created lazily on
  /// first [`SessionStore::save`], not here — constructing a store should never fail or
  /// touch the filesystem before it is actually used.
  pub fn new(dir: impl Into<PathBuf>, ttl: Duration) -> Self {
    Self {
      dir: dir.into(),
      ttl: Some(ttl),
    }
  }

  /// A store whose sessions never expire: nothing is ever evicted by
  /// [`SessionStore::sweep_expired`], and [`SessionStore::history`] always reads whatever
  /// is on disk. Used by the CLI so a conversation can be resumed no matter how long ago
  /// its last turn was; only an explicit reset (`--fresh` / `/reset`) clears it.
  pub fn new_persistent(dir: impl Into<PathBuf>) -> Self {
    Self {
      dir: dir.into(),
      ttl: None,
    }
  }

  /// One file per `(scope, session_id)`, named from a URL-safe base64 encoding of the
  /// pair rather than the raw strings: either could contain `/`, `..`, or other
  /// characters that are meaningful to a filesystem path, and letting them straight
  /// through would risk writing outside `dir` entirely. `\0` as the separator (like
  /// [`MemorySessionStore::key`]) means `("a", "bc")` and `("ab", "c")` still encode to
  /// different strings before encoding, so they cannot collide after it either.
  fn path_for(&self, scope: &str, session_id: &str) -> PathBuf {
    let key = format!("{scope}\0{session_id}");
    let encoded = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(key);
    self.dir.join(format!("{encoded}.json"))
  }

  /// Whether the file at `path` is past this store's TTL. `Some(false)` for a store with
  /// no TTL (never expires) as long as the file exists; `None` for a missing file
  /// (unknown session, nothing to report) or one whose mtime could not be read;
  /// `Some(true)` once it is older than `ttl`.
  async fn is_expired(&self, path: &Path) -> Option<bool> {
    let metadata = tokio::fs::metadata(path).await.ok()?;
    let Some(ttl) = self.ttl else {
      return Some(false); // Never expires, but confirm the file actually exists.
    };
    let modified = metadata.modified().ok()?;
    Some(modified.elapsed().unwrap_or_default() > ttl)
  }

  /// Best-effort tighten `dir` to owner-only (`0700`) on Unix so other local users cannot
  /// read stored transcripts (see the module docs' "Plaintext on disk" note). A no-op on
  /// non-Unix targets and never fatal: failing to tighten permissions must not stop a
  /// session from being saved.
  #[cfg(unix)]
  async fn restrict_dir_permissions(&self) {
    use std::os::unix::fs::PermissionsExt;
    if let Err(err) =
      tokio::fs::set_permissions(&self.dir, std::fs::Permissions::from_mode(0o700)).await
    {
      tracing::warn!(
        dir = %self.dir.display(),
        "failed to restrict session directory permissions: {err}"
      );
    }
  }

  #[cfg(not(unix))]
  async fn restrict_dir_permissions(&self) {}

  /// Every session stored under `scope`, most recently active first — the basis for the
  /// `cli` binary's `--list`. Best-effort like [`SessionStore::sweep_expired`]: a missing
  /// directory (nothing saved yet) yields an empty list rather than an error, and any
  /// entry that is not a session file for this `scope` (wrong extension, an undecodable
  /// name, a different scope, or a leftover temp file from an interrupted
  /// [`SessionStore::save`]) is silently skipped rather than failing the whole listing.
  ///
  /// This does not consider [`Self::ttl`]: an expired-but-not-yet-swept file (only
  /// possible for a finite-TTL store; a [`Self::new_persistent`] store never expires
  /// anything) still shows up here, since "still on disk" is the property `--list` is
  /// answering, not "still resumable".
  ///
  /// Cost note: getting `turns` means reading and fully deserializing every session
  /// file in `scope`, not just its metadata — for a directory with many sessions, or
  /// unusually long individual transcripts, `list` is a full-directory read, not a cheap
  /// listing. Acceptable for `--list` (a human-triggered, one-shot CLI command run at
  /// most every few seconds), but not something to call in a hot path.
  pub async fn list(&self, scope: &str) -> Vec<SessionSummary> {
    let Ok(mut entries) = tokio::fs::read_dir(&self.dir).await else {
      return Vec::new();
    };

    let mut summaries = Vec::new();
    while let Ok(Some(entry)) = entries.next_entry().await {
      let path = entry.path();
      let Some(session_id) = self.decode_session_id(&path, scope) else {
        continue;
      };
      let Ok(metadata) = entry.metadata().await else {
        continue;
      };
      let Ok(last_used) = metadata.modified() else {
        continue;
      };
      let turns = tokio::fs::read(&path)
        .await
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Vec<Event>>(&bytes).ok())
        .map(|history| history.len())
        .unwrap_or(0);
      summaries.push(SessionSummary {
        session_id,
        turns,
        last_used,
      });
    }

    summaries.sort_by_key(|session| std::cmp::Reverse(session.last_used));
    summaries
  }

  /// Delete a single session's file, if any. `true` if a file existed and was removed;
  /// `false` for an unknown session — not an error, just nothing to do.
  pub async fn remove(&self, scope: &str, session_id: &str) -> bool {
    tokio::fs::remove_file(self.path_for(scope, session_id))
      .await
      .is_ok()
  }

  /// The inverse of [`Self::path_for`]: recovers `session_id` from `path`'s filename if
  /// it decodes to a `(scope, session_id)` pair matching `scope`. `None` for anything
  /// that is not one of this store's own session files — used by [`Self::list`] to skip
  /// everything else in `dir` (wrong extension, undecodable name, another scope, or a
  /// stray `.tmp-*` file from [`SessionStore::save`]).
  fn decode_session_id(&self, path: &Path, scope: &str) -> Option<String> {
    if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
      return None;
    }
    let stem = path.file_stem()?.to_str()?;
    let decoded = base64::engine::general_purpose::URL_SAFE_NO_PAD
      .decode(stem)
      .ok()?;
    let key = String::from_utf8(decoded).ok()?;
    let (key_scope, session_id) = key.split_once('\0')?;
    (key_scope == scope).then(|| session_id.to_owned())
  }
}

#[async_trait::async_trait]
impl SessionStore for FileSessionStore {
  async fn history(&self, scope: &str, session_id: &str) -> Vec<Event> {
    let path = self.path_for(scope, session_id);
    if self.is_expired(&path).await.unwrap_or(true) {
      return Vec::new();
    }
    let Ok(bytes) = tokio::fs::read(&path).await else {
      return Vec::new();
    };
    serde_json::from_slice(&bytes).unwrap_or_else(|err| {
      tracing::warn!(path = %path.display(), "failed to parse session file: {err}");
      Vec::new()
    })
  }

  async fn save(&self, scope: &str, session_id: &str, history: Vec<Event>) {
    if let Err(err) = tokio::fs::create_dir_all(&self.dir).await {
      tracing::warn!(
        dir = %self.dir.display(),
        "failed to create session directory: {err}"
      );
      return;
    }
    self.restrict_dir_permissions().await;

    let bytes = match serde_json::to_vec(&history) {
      Ok(bytes) => bytes,
      Err(err) => {
        tracing::warn!("failed to serialize session history: {err}");
        return;
      }
    };

    // Write to a unique temp file first, then atomically `rename` it over the target: a
    // plain `write` truncates the file before the new bytes land, so a crash mid-write
    // would leave a half-written (unparseable) transcript on disk. `rename` within the
    // same directory is atomic, so a reader ever only sees the old file or the fully
    // written new one — never a partial one. The temp name carries a fresh UUID so two
    // concurrent saves of the same session cannot clobber each other's temp file (the
    // final `rename` is still last-write-wins, as documented).
    let path = self.path_for(scope, session_id);
    let tmp = self.dir.join(format!(".tmp-{}", uuid::Uuid::new_v4()));
    if let Err(err) = tokio::fs::write(&tmp, bytes).await {
      tracing::warn!(path = %tmp.display(), "failed to write session file: {err}");
      return;
    }
    if let Err(err) = tokio::fs::rename(&tmp, &path).await {
      tracing::warn!(path = %path.display(), "failed to persist session file: {err}");
      let _ = tokio::fs::remove_file(&tmp).await;
    }
  }

  /// Best-effort: a directory that does not exist yet (nothing has been saved) is not
  /// an error, and one unreadable entry does not stop the rest from being swept.
  async fn sweep_expired(&self) {
    let Ok(mut entries) = tokio::fs::read_dir(&self.dir).await else {
      return;
    };
    while let Ok(Some(entry)) = entries.next_entry().await {
      let path = entry.path();
      if self.is_expired(&path).await.unwrap_or(false) {
        let _ = tokio::fs::remove_file(&path).await;
      }
    }
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
    assert!(store.history("scope-a", "session-1").await.is_empty());
  }

  #[tokio::test]
  async fn save_then_history_round_trips() {
    let store = MemorySessionStore::new(Duration::from_secs(60));
    store
      .save("scope-a", "session-1", sample_history("hi"))
      .await;

    let history = store.history("scope-a", "session-1").await;
    assert_eq!(history.len(), 1);
  }

  #[tokio::test]
  async fn sessions_are_isolated_per_scope() {
    let store = MemorySessionStore::new(Duration::from_secs(60));
    store
      .save("scope-a", "session-1", sample_history("hi"))
      .await;

    // Same session id, different scope: must not see scope-a's history.
    assert!(store.history("scope-b", "session-1").await.is_empty());
  }

  #[tokio::test]
  async fn save_replaces_the_previous_history_for_the_same_session() {
    let store = MemorySessionStore::new(Duration::from_secs(60));
    store
      .save("scope-a", "session-1", sample_history("first"))
      .await;
    store
      .save(
        "scope-a",
        "session-1",
        vec![
          sample_history("first")[0].clone(),
          sample_history("second")[0].clone(),
        ],
      )
      .await;

    assert_eq!(store.history("scope-a", "session-1").await.len(), 2);
    assert_eq!(store.len(), 1, "one session, not two");
  }

  #[tokio::test]
  async fn sweep_expired_evicts_sessions_past_the_ttl() {
    let store = MemorySessionStore::new(Duration::from_millis(1));
    store
      .save("scope-a", "session-1", sample_history("hi"))
      .await;
    std::thread::sleep(Duration::from_millis(20));

    store.sweep_expired().await;

    assert_eq!(store.len(), 0);
    assert!(store.history("scope-a", "session-1").await.is_empty());
  }

  #[tokio::test]
  async fn sweep_expired_keeps_sessions_still_within_the_ttl() {
    let store = MemorySessionStore::new(Duration::from_secs(60));
    store
      .save("scope-a", "session-1", sample_history("hi"))
      .await;

    store.sweep_expired().await;

    assert_eq!(store.len(), 1);
  }

  /// A freshly created directory under the OS temp dir that no other test can collide
  /// with; not cleaned up afterwards, same tradeoff as the `tools::file_list` tests this
  /// mirrors (bounded, disposable test dirs).
  fn unique_temp_dir(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
      "agent-session-test-{label}-{}",
      uuid::Uuid::new_v4()
    ))
  }

  #[tokio::test]
  async fn file_store_unknown_session_returns_empty_history() {
    let store = FileSessionStore::new(unique_temp_dir("unknown"), Duration::from_secs(60));
    assert!(store.history("scope-a", "session-1").await.is_empty());
  }

  #[tokio::test]
  async fn file_store_save_then_history_round_trips() {
    let store = FileSessionStore::new(unique_temp_dir("roundtrip"), Duration::from_secs(60));
    store
      .save("scope-a", "session-1", sample_history("hi"))
      .await;

    let history = store.history("scope-a", "session-1").await;
    assert_eq!(history.len(), 1);
  }

  #[tokio::test]
  async fn file_store_creates_its_directory_lazily_on_save() {
    let dir = unique_temp_dir("lazy-create");
    assert!(!dir.exists(), "directory must not exist before any save");

    let store = FileSessionStore::new(dir.clone(), Duration::from_secs(60));
    store
      .save("scope-a", "session-1", sample_history("hi"))
      .await;

    assert!(dir.exists());
  }

  #[tokio::test]
  async fn file_store_sessions_are_isolated_per_scope() {
    let store = FileSessionStore::new(unique_temp_dir("isolation"), Duration::from_secs(60));
    store
      .save("scope-a", "session-1", sample_history("hi"))
      .await;

    // Same session id, different scope: must not see scope-a's history.
    assert!(store.history("scope-b", "session-1").await.is_empty());
  }

  #[tokio::test]
  async fn file_store_save_replaces_the_previous_history_for_the_same_session() {
    let store = FileSessionStore::new(unique_temp_dir("replace"), Duration::from_secs(60));
    store
      .save("scope-a", "session-1", sample_history("first"))
      .await;
    store
      .save(
        "scope-a",
        "session-1",
        vec![
          sample_history("first")[0].clone(),
          sample_history("second")[0].clone(),
        ],
      )
      .await;

    assert_eq!(store.history("scope-a", "session-1").await.len(), 2);
  }

  #[tokio::test]
  async fn file_store_sweep_expired_evicts_files_past_the_ttl() {
    let store = FileSessionStore::new(unique_temp_dir("sweep-evict"), Duration::from_millis(1));
    store
      .save("scope-a", "session-1", sample_history("hi"))
      .await;
    tokio::time::sleep(Duration::from_millis(20)).await;

    store.sweep_expired().await;

    assert!(store.history("scope-a", "session-1").await.is_empty());
  }

  #[tokio::test]
  async fn file_store_sweep_expired_keeps_files_still_within_the_ttl() {
    let store = FileSessionStore::new(unique_temp_dir("sweep-keep"), Duration::from_secs(60));
    store
      .save("scope-a", "session-1", sample_history("hi"))
      .await;

    store.sweep_expired().await;

    assert_eq!(store.history("scope-a", "session-1").await.len(), 1);
  }

  /// The transcript directory must not be world-/group-readable: it holds plaintext
  /// conversation history, and on a shared machine another user could otherwise read it.
  #[cfg(unix)]
  #[tokio::test]
  async fn file_store_directory_is_owner_only_after_save() {
    use std::os::unix::fs::PermissionsExt;

    let dir = unique_temp_dir("perms");
    let store = FileSessionStore::new(dir.clone(), Duration::from_secs(60));
    store
      .save("scope-a", "session-1", sample_history("hi"))
      .await;

    let mode = tokio::fs::metadata(&dir)
      .await
      .unwrap()
      .permissions()
      .mode();
    // Only the low 9 permission bits matter; group/other bits must all be clear.
    assert_eq!(
      mode & 0o077,
      0,
      "session dir must not be group-/world-accessible"
    );
  }

  /// A persistent store must resume a conversation no matter how stale it is: a finite
  /// TTL would have evicted this by now, but `new_persistent` never expires.
  #[tokio::test]
  async fn file_store_persistent_reads_history_regardless_of_age() {
    let store = FileSessionStore::new_persistent(unique_temp_dir("persistent-read"));
    store
      .save("scope-a", "session-1", sample_history("hi"))
      .await;
    // Long enough that any nonzero TTL would treat the file as expired.
    tokio::time::sleep(Duration::from_millis(20)).await;

    assert_eq!(store.history("scope-a", "session-1").await.len(), 1);
  }

  /// `sweep_expired` on a persistent store deletes nothing, so the session survives it.
  #[tokio::test]
  async fn file_store_persistent_sweep_keeps_everything() {
    let store = FileSessionStore::new_persistent(unique_temp_dir("persistent-sweep"));
    store
      .save("scope-a", "session-1", sample_history("hi"))
      .await;
    tokio::time::sleep(Duration::from_millis(20)).await;

    store.sweep_expired().await;

    assert_eq!(store.history("scope-a", "session-1").await.len(), 1);
  }

  /// A persistent store still returns empty history for a session that was never saved:
  /// "never expires" must not be confused with "invents history for unknown ids".
  #[tokio::test]
  async fn file_store_persistent_unknown_session_returns_empty_history() {
    let store = FileSessionStore::new_persistent(unique_temp_dir("persistent-unknown"));
    assert!(store.history("scope-a", "session-1").await.is_empty());
  }

  #[tokio::test]
  async fn file_store_list_returns_every_session_for_the_scope() {
    let store = FileSessionStore::new_persistent(unique_temp_dir("list"));
    store
      .save("scope-a", "session-1", sample_history("hi"))
      .await;
    store
      .save("scope-a", "session-2", sample_history("hi"))
      .await;
    // Different scope: must not show up in scope-a's listing.
    store
      .save("scope-b", "session-3", sample_history("hi"))
      .await;

    let mut ids: Vec<_> = store
      .list("scope-a")
      .await
      .into_iter()
      .map(|session| session.session_id)
      .collect();
    ids.sort();

    assert_eq!(ids, vec!["session-1".to_owned(), "session-2".to_owned()]);
  }

  #[tokio::test]
  async fn file_store_list_reports_the_turn_count() {
    let store = FileSessionStore::new_persistent(unique_temp_dir("list-turns"));
    store
      .save(
        "scope-a",
        "session-1",
        vec![
          sample_history("first")[0].clone(),
          sample_history("second")[0].clone(),
        ],
      )
      .await;

    let sessions = store.list("scope-a").await;
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].turns, 2);
  }

  #[tokio::test]
  async fn file_store_list_is_empty_when_nothing_has_been_saved() {
    let store = FileSessionStore::new_persistent(unique_temp_dir("list-missing"));
    assert!(store.list("scope-a").await.is_empty());
  }

  #[tokio::test]
  async fn file_store_remove_deletes_the_session() {
    let store = FileSessionStore::new_persistent(unique_temp_dir("remove"));
    store
      .save("scope-a", "session-1", sample_history("hi"))
      .await;

    assert!(store.remove("scope-a", "session-1").await);
    assert!(store.history("scope-a", "session-1").await.is_empty());
    assert!(store.list("scope-a").await.is_empty());
  }

  #[tokio::test]
  async fn file_store_remove_returns_false_for_an_unknown_session() {
    let store = FileSessionStore::new_persistent(unique_temp_dir("remove-unknown"));
    assert!(!store.remove("scope-a", "session-1").await);
  }

  #[tokio::test]
  async fn file_store_remove_does_not_affect_other_scopes() {
    let store = FileSessionStore::new_persistent(unique_temp_dir("remove-scoped"));
    store
      .save("scope-a", "session-1", sample_history("hi"))
      .await;
    store
      .save("scope-b", "session-1", sample_history("hi"))
      .await;

    store.remove("scope-a", "session-1").await;

    assert!(store.history("scope-a", "session-1").await.is_empty());
    assert_eq!(store.history("scope-b", "session-1").await.len(), 1);
  }
}
