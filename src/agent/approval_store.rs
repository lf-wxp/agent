//! Persisting a run that stopped to ask a human, so the question outlives the process.
//!
//! # Why this is separate from [`crate::agent::session`]
//!
//! A session store holds *finished* turns: its unit is a completed exchange, and its
//! contents are what gets replayed to the model next time. A suspended run is neither.
//! Its transcript is mid-turn by construction — tool calls with no results, which
//! providers reject — so filing it as conversation history would produce a session that
//! cannot be continued at all.
//!
//! They also have opposite lifetimes. Session history is the thing worth keeping; a
//! suspended run is a temporary state that exists only until someone answers, and a
//! resume or an abandon should remove it. Sharing one store would mean a `/reset` that
//! clears history also silently discards a run someone was about to approve, or a swept
//! session leaving an orphaned run behind.
//!
//! So: one store per concern, and [`ApprovalStore`] is the smaller of the two — put,
//! take, list, remove.

use std::path::PathBuf;

use base64::Engine;

use crate::agent::{
  Event,
  runtime::{AgentRunState, SuspendedToolCall},
};

/// A stored run as seen by a display path: what it is waiting on, and what led up to it.
///
/// Deliberately not the run itself — see [`ApprovalStore::peek`].
#[derive(Debug, Clone)]
pub struct SuspendedRunView {
  /// Always non-empty: a run is only stored because a round could not finish.
  pub pending: Vec<SuspendedToolCall>,
  /// The turn as it stood when it stopped, including the message that started it.
  ///
  /// This is the part a session store cannot supply — a suspended turn is deliberately
  /// absent from conversation history (see the module docs), so without this a view that
  /// arrives after the fact sees a pending approval with no idea what was being asked.
  ///
  /// Mid-turn, so the same caveat as [`AgentRunState`] applies: fine to render, not
  /// something to send to a model or file as history.
  pub events: Vec<Event>,
}

/// Where a suspended run is kept while it waits to be answered.
///
/// A trait for the same reason [`crate::agent::session::SessionStore`] is one: the CLI
/// wants a file on disk, a server would want whatever its other state lives in, and the
/// runtime should need neither. Keyed by `(scope, run_id)` to match the session store's
/// two-part key, so a caller that already separates terminal-scoped from web-scoped
/// sessions can keep that separation here without inventing a second scheme.
#[async_trait::async_trait]
pub trait ApprovalStore: Send + Sync {
  /// Store `state` under `(scope, run_id)`, replacing anything already there.
  async fn put(&self, scope: &str, run_id: &str, state: &AgentRunState);

  /// Remove and return the state stored under `(scope, run_id)`.
  ///
  /// Taking rather than reading, because a stored suspended run is a single-use thing:
  /// answering it produces a new state (or a finished run) and the old one is no longer
  /// valid. Leaving it in place would invite resuming the same run twice — which for a
  /// run whose pending call has side effects means doing them twice.
  async fn take(&self, scope: &str, run_id: &str) -> Option<AgentRunState>;

  /// What is stored under `(scope, run_id)`, without consuming it. `None` when nothing
  /// is.
  ///
  /// The read-only counterpart of [`Self::take`], for showing a human what is waiting
  /// and what led up to it. Returning a [`SuspendedRunView`] rather than the state
  /// itself is what keeps this from undermining `take`'s single-use guarantee: there is
  /// nothing here that could be resumed, so a display path cannot quietly become a
  /// second way to run the same call.
  async fn peek(&self, scope: &str, run_id: &str) -> Option<SuspendedRunView>;

  /// Every run id currently suspended under `scope`, most recently stored first.
  ///
  /// For a caller that needs to tell a human what is waiting — the CLI prints this at
  /// startup, since a suspended run is invisible otherwise: it produced no answer and
  /// left nothing in the session history.
  async fn list(&self, scope: &str) -> Vec<String>;

  /// Discard the state under `(scope, run_id)` without reading it, for a caller that has
  /// already decided not to continue.
  async fn remove(&self, scope: &str, run_id: &str);
}

/// [`ApprovalStore`] that writes one JSON file per `(scope, run_id)` under `dir`.
///
/// Deliberately the same on-disk conventions as
/// [`crate::agent::session::FileSessionStore`] — base64-encoded key as the filename,
/// write-to-temp-then-rename, `0700` on the directory — because the two sit side by side
/// in the same installation and differing would be a trap for whoever operates it. See
/// that type for why each of those is the way it is.
///
/// # Known limitations
///
/// - **Plaintext on disk**: a suspended run contains the transcript so far, including
///   tool arguments awaiting approval. Same exposure as the session store, and the same
///   mitigation: owner-only directory, not encryption.
/// - **Single-machine, last-write-wins**: as with the session store.
pub struct FileApprovalStore {
  dir: PathBuf,
}

impl FileApprovalStore {
  /// `dir` is created lazily on first [`ApprovalStore::put`], so constructing a store
  /// never touches the filesystem.
  pub fn new(dir: impl Into<PathBuf>) -> Self {
    Self { dir: dir.into() }
  }

  /// See [`crate::agent::session::FileSessionStore`]'s equivalent: the raw key could
  /// contain `/` or `..`, so it is encoded rather than used as a path component, and the
  /// `\0` separator keeps `("a", "bc")` and `("ab", "c")` from colliding.
  fn path_for(&self, scope: &str, run_id: &str) -> PathBuf {
    let key = format!("{scope}\0{run_id}");
    let encoded = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(key);
    self.dir.join(format!("{encoded}.json"))
  }

  /// The `run_id` a stored file belongs to, or `None` if it is not one of `scope`'s.
  fn decode_run_id(&self, path: &std::path::Path, scope: &str) -> Option<String> {
    if path.extension()? != "json" {
      return None;
    }
    let encoded = path.file_stem()?.to_str()?;
    let decoded = base64::engine::general_purpose::URL_SAFE_NO_PAD
      .decode(encoded)
      .ok()?;
    let key = String::from_utf8(decoded).ok()?;
    let (key_scope, run_id) = key.split_once('\0')?;
    (key_scope == scope).then(|| run_id.to_owned())
  }

  /// Best-effort owner-only on the directory; never fatal, since failing to tighten
  /// permissions must not stop a run from being saved.
  #[cfg(unix)]
  async fn restrict_dir_permissions(&self) {
    use std::os::unix::fs::PermissionsExt;
    if let Err(err) =
      tokio::fs::set_permissions(&self.dir, std::fs::Permissions::from_mode(0o700)).await
    {
      tracing::warn!(
        dir = %self.dir.display(),
        "failed to restrict approval directory permissions: {err}"
      );
    }
  }

  #[cfg(not(unix))]
  async fn restrict_dir_permissions(&self) {}
}

#[async_trait::async_trait]
impl ApprovalStore for FileApprovalStore {
  async fn put(&self, scope: &str, run_id: &str, state: &AgentRunState) {
    if let Err(err) = tokio::fs::create_dir_all(&self.dir).await {
      tracing::warn!(
        dir = %self.dir.display(),
        "failed to create approval directory: {err}"
      );
      return;
    }
    self.restrict_dir_permissions().await;

    let bytes = match serde_json::to_vec(state) {
      Ok(bytes) => bytes,
      Err(err) => {
        tracing::warn!("failed to serialize a suspended run: {err}");
        return;
      }
    };

    // Temp file then atomic rename, so a crash mid-write cannot leave a half-written
    // state that would fail to parse — which for this store means a pending approval
    // quietly becoming unresumable.
    let path = self.path_for(scope, run_id);
    let tmp = self.dir.join(format!(".tmp-{}", uuid::Uuid::new_v4()));
    if let Err(err) = tokio::fs::write(&tmp, bytes).await {
      tracing::warn!(path = %tmp.display(), "failed to write approval state: {err}");
      return;
    }
    if let Err(err) = tokio::fs::rename(&tmp, &path).await {
      tracing::warn!(path = %path.display(), "failed to persist approval state: {err}");
      let _ = tokio::fs::remove_file(&tmp).await;
    }
  }

  async fn take(&self, scope: &str, run_id: &str) -> Option<AgentRunState> {
    let path = self.path_for(scope, run_id);
    let bytes = tokio::fs::read(&path).await.ok()?;

    // Removed whether or not it parses: an unparseable state cannot be resumed, so
    // leaving it would keep reporting a pending approval nobody can ever answer.
    let _ = tokio::fs::remove_file(&path).await;

    match serde_json::from_slice(&bytes) {
      Ok(state) => Some(state),
      Err(err) => {
        tracing::warn!(
          path = %path.display(),
          "discarding an unparseable suspended run: {err}"
        );
        None
      }
    }
  }

  async fn peek(&self, scope: &str, run_id: &str) -> Option<SuspendedRunView> {
    let path = self.path_for(scope, run_id);
    let bytes = tokio::fs::read(&path).await.ok()?;
    // Unlike `take`, an unparseable file is left alone: this is a read, and deleting
    // something on the way past would make merely looking at a pending approval a
    // destructive act.
    let state: AgentRunState = serde_json::from_slice(&bytes).ok()?;
    Some(SuspendedRunView {
      pending: state.suspended,
      events: state.context.events,
    })
  }

  async fn list(&self, scope: &str) -> Vec<String> {
    let Ok(mut entries) = tokio::fs::read_dir(&self.dir).await else {
      return Vec::new();
    };

    let mut found: Vec<(std::time::SystemTime, String)> = Vec::new();
    while let Ok(Some(entry)) = entries.next_entry().await {
      let path = entry.path();
      let Some(run_id) = self.decode_run_id(&path, scope) else {
        continue;
      };
      let stored_at = entry
        .metadata()
        .await
        .and_then(|metadata| metadata.modified())
        .unwrap_or(std::time::UNIX_EPOCH);
      found.push((stored_at, run_id));
    }

    // Most recent first, breaking ties on id so the order is total — two runs suspended
    // in the same round share an mtime, and an unstable listing would make the CLI
    // present them differently each time it is asked.
    found.sort_by(|(a_time, a_id), (b_time, b_id)| b_time.cmp(a_time).then_with(|| a_id.cmp(b_id)));
    found.into_iter().map(|(_, run_id)| run_id).collect()
  }

  async fn remove(&self, scope: &str, run_id: &str) {
    let _ = tokio::fs::remove_file(self.path_for(scope, run_id)).await;
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::agent::{ExecutionContext, SuspendedToolCall, fingerprint::RunFingerprint};

  fn temp_dir(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
      "agent-approval-store-{label}-{}",
      uuid::Uuid::new_v4()
    ))
  }

  fn state(tool: &str) -> AgentRunState {
    AgentRunState {
      fingerprint: RunFingerprint::new("gpt-test", None, ["delete_file"]),
      suspended: vec![SuspendedToolCall {
        tool_call_id: "call_1".to_owned(),
        name: tool.to_owned(),
        raw_arguments: r#"{"path":"a.txt"}"#.to_owned(),
      }],
      budget_exhausted: false,
      context: ExecutionContext::new(),
    }
  }

  #[tokio::test]
  async fn a_stored_run_comes_back_intact() {
    let store = FileApprovalStore::new(temp_dir("round-trip"));
    store.put("local", "run-1", &state("delete_file")).await;

    let back = store.take("local", "run-1").await.expect("stored above");

    assert_eq!(back.suspended.len(), 1);
    assert_eq!(back.suspended[0].name, "delete_file");
    assert_eq!(back.suspended[0].raw_arguments, r#"{"path":"a.txt"}"#);
  }

  /// Taking is single-use: a second take finds nothing. Resuming the same run twice
  /// would mean performing its pending call's side effects twice.
  #[tokio::test]
  async fn taking_a_run_consumes_it() {
    let store = FileApprovalStore::new(temp_dir("consume"));
    store.put("local", "run-1", &state("delete_file")).await;

    assert!(store.take("local", "run-1").await.is_some());
    assert!(store.take("local", "run-1").await.is_none());
  }

  #[tokio::test]
  async fn an_unknown_run_is_absent_rather_than_an_error() {
    let store = FileApprovalStore::new(temp_dir("unknown"));
    assert!(store.take("local", "nope").await.is_none());
    assert!(store.list("local").await.is_empty());
  }

  /// Scopes do not see each other's runs, the same way sessions do not.
  #[tokio::test]
  async fn scopes_stay_separate() {
    let store = FileApprovalStore::new(temp_dir("scopes"));
    store.put("local", "run-1", &state("delete_file")).await;

    assert!(store.take("web", "run-1").await.is_none());
    assert!(store.list("web").await.is_empty());
    assert_eq!(store.list("local").await, vec!["run-1".to_owned()]);
  }

  /// The keys are encoded rather than used as path components, so an id that looks like
  /// a path traversal is stored as an ordinary file and read back unchanged.
  #[tokio::test]
  async fn a_run_id_that_looks_like_a_path_is_handled_literally() {
    let dir = temp_dir("traversal");
    let store = FileApprovalStore::new(&dir);
    let hostile = "../../escaped";

    store.put("local", hostile, &state("delete_file")).await;

    assert!(store.take("local", hostile).await.is_some());
    // Nothing was written outside the store's own directory.
    let escaped = dir.join("../../escaped.json");
    assert!(
      !escaped.exists(),
      "a key must not escape the store directory"
    );
  }

  /// `("a", "bc")` and `("ab", "c")` must not collide, which is what the `\0` separator
  /// in the key is for.
  #[tokio::test]
  async fn keys_cannot_be_confused_across_the_scope_boundary() {
    let store = FileApprovalStore::new(temp_dir("ambiguity"));
    store.put("a", "bc", &state("first")).await;
    store.put("ab", "c", &state("second")).await;

    let first = store.take("a", "bc").await.expect("stored above");
    let second = store.take("ab", "c").await.expect("stored above");

    assert_eq!(first.suspended[0].name, "first");
    assert_eq!(second.suspended[0].name, "second");
  }

  #[tokio::test]
  async fn listing_reports_every_suspended_run_in_a_stable_order() {
    let store = FileApprovalStore::new(temp_dir("listing"));
    for id in ["run-a", "run-b", "run-c"] {
      store.put("local", id, &state("delete_file")).await;
    }

    let listed = store.list("local").await;

    assert_eq!(listed.len(), 3);
    assert_eq!(
      store.list("local").await,
      listed,
      "the same listing must come back the same way twice"
    );
  }

  /// Peeking is the display path, so it must not consume — and must carry the
  /// transcript, which is the part no other store has: a suspended turn is deliberately
  /// absent from conversation history.
  #[tokio::test]
  async fn peeking_reports_the_run_without_consuming_it() {
    let store = FileApprovalStore::new(temp_dir("peek"));
    let mut state = state("delete_file");
    state.context.add_event(Event::new(
      "exec",
      "user",
      vec![crate::agent::ContentItem::Message {
        role: "user".to_owned(),
        content: "delete a.txt".to_owned(),
      }],
    ));
    store.put("local", "run-1", &state).await;

    let first = store.peek("local", "run-1").await.expect("stored above");
    assert_eq!(first.pending.len(), 1);
    assert_eq!(first.pending[0].name, "delete_file");
    assert_eq!(
      first.events.len(),
      1,
      "the request that led to the pending call has to come back too"
    );

    assert!(
      store.peek("local", "run-1").await.is_some(),
      "peeking twice must work — unlike `take`, it is not a claim"
    );
    assert!(
      store.take("local", "run-1").await.is_some(),
      "and it must leave the run resumable"
    );
  }

  #[tokio::test]
  async fn peeking_an_unknown_run_reports_nothing() {
    let store = FileApprovalStore::new(temp_dir("peek-unknown"));
    assert!(store.peek("local", "nope").await.is_none());
  }

  /// Unlike `take`, a read must not delete: merely looking at a pending approval being
  /// destructive would be a trap, and the file may yet be recoverable by hand.
  #[tokio::test]
  async fn peeking_leaves_an_unparseable_file_alone() {
    let dir = temp_dir("peek-corrupt");
    let store = FileApprovalStore::new(&dir);
    store.put("local", "run-1", &state("delete_file")).await;
    let path = store.path_for("local", "run-1");
    tokio::fs::write(&path, b"{not json").await.unwrap();

    assert!(store.peek("local", "run-1").await.is_none());
    assert!(
      path.exists(),
      "a read must not delete what it could not parse"
    );
  }

  #[tokio::test]
  async fn removing_a_run_discards_it_without_reading_it() {
    let store = FileApprovalStore::new(temp_dir("remove"));
    store.put("local", "run-1", &state("delete_file")).await;

    store.remove("local", "run-1").await;

    assert!(store.take("local", "run-1").await.is_none());
    assert!(store.list("local").await.is_empty());
  }

  /// A half-written file cannot be resumed, so it is discarded rather than left to be
  /// reported as a pending approval nobody can answer.
  #[tokio::test]
  async fn an_unparseable_state_is_discarded() {
    let dir = temp_dir("corrupt");
    let store = FileApprovalStore::new(&dir);
    store.put("local", "run-1", &state("delete_file")).await;

    let path = store.path_for("local", "run-1");
    tokio::fs::write(&path, b"{not json").await.unwrap();

    assert!(store.take("local", "run-1").await.is_none());
    assert!(
      store.list("local").await.is_empty(),
      "it must not keep being reported as pending"
    );
  }

  /// A later `put` replaces the earlier state rather than accumulating: resuming
  /// produces a new suspension when only part of a round was answered, and the old one
  /// is no longer valid.
  #[tokio::test]
  async fn putting_again_replaces_the_previous_state() {
    let store = FileApprovalStore::new(temp_dir("replace"));
    store.put("local", "run-1", &state("first")).await;
    store.put("local", "run-1", &state("second")).await;

    let back = store.take("local", "run-1").await.expect("stored above");

    assert_eq!(back.suspended[0].name, "second");
    assert_eq!(store.list("local").await.len(), 0);
  }

  /// The directory is owner-only, since a suspended run holds the transcript so far.
  #[cfg(unix)]
  #[tokio::test]
  async fn the_directory_is_owner_only() {
    use std::os::unix::fs::PermissionsExt;

    let dir = temp_dir("perms");
    let store = FileApprovalStore::new(&dir);
    store.put("local", "run-1", &state("delete_file")).await;

    let mode = tokio::fs::metadata(&dir)
      .await
      .unwrap()
      .permissions()
      .mode();

    assert_eq!(mode & 0o777, 0o700, "got {:o}", mode & 0o777);
  }

  /// A leftover temp file from an interrupted `put` is not a run, and must not be
  /// reported as one.
  #[tokio::test]
  async fn a_leftover_temp_file_is_not_listed() {
    let dir = temp_dir("leftover");
    let store = FileApprovalStore::new(&dir);
    store.put("local", "run-1", &state("delete_file")).await;
    tokio::fs::write(dir.join(".tmp-interrupted"), b"partial")
      .await
      .unwrap();

    assert_eq!(store.list("local").await, vec!["run-1".to_owned()]);
  }
}
