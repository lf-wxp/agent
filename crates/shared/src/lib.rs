//! Types shared between the native agent process and the Leptos web UI.
//!
//! Both sides depend on this crate by path rather than duplicating these definitions:
//! a field added or renamed here shows up as a compile error on whichever side was not
//! updated, instead of a silent runtime mismatch discovered only when a request/response
//! fails to deserialize.
//!
//! None of these mirror the native `agent` crate's own types (`agent::agent::ContentItem`,
//! `Event`, ...) by depending on that crate directly — `agent` pulls in native-only
//! dependencies (a multi-threaded `tokio` runtime, `rmcp`, filesystem tools, ...) that do
//! not compile for `wasm32-unknown-unknown`. The native side (`src/bin/cli/web.rs`)
//! converts between the two; this crate only defines the wire shape both ends agree on.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Request body for `POST /api/chat`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
  pub input: String,
}

/// Response body for an accepted `POST /api/chat`. The turn itself is watched on
/// `GET /api/stream` like everyone else's (see [`ChatEvent`]); the one thing only the
/// submitter can know is *which* of the turns on that shared stream is the one it just
/// started — that is what `turn` is for, and it is the only reason this route answers
/// with a body at all. See [`ChatEvent::Done`]'s `turn`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatAccepted {
  pub turn: String,
}

/// Request body for `POST /api/approve/{id}`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ApprovalDecision {
  pub approved: bool,
}

/// Mirrors `agent::agent::ToolResultStatus`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolStatus {
  Success,
  Error,
}

/// One tool call the model requested, as shown in the process timeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallSummary {
  pub id: String,
  pub name: String,
  pub arguments: Value,
}

/// One tool call's result, as shown in the process timeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultSummary {
  pub id: String,
  pub name: String,
  pub status: ToolStatus,
  pub content: String,
}

/// Which front-end started a turn — carried on [`ChatEvent::UserMessage`] so a renderer
/// can tell whether it needs to display the input text itself, or whether the user
/// already saw it appear some other way (e.g. the terminal's own line editor already
/// echoes what was typed there, so the terminal's renderer skips re-printing a
/// `Terminal`-origin message but does print a `Web`-origin one).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageOrigin {
  Terminal,
  Web,
}

/// One frame of the `GET /api/stream` SSE broadcast (the `data` field of each
/// `event: chat` SSE message, JSON-encoded). Unlike a per-request response, this single
/// stream carries events for *every* turn this process runs — terminal-originated or
/// browser-originated — so every connected tab (and the terminal itself, via its own
/// stdout) sees the same conversation regardless of which front-end drove a given turn.
/// `Done`/`Error` ends one turn, not the connection itself; more turns may follow on the
/// same stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatEvent {
  /// A turn just started with this input — from the terminal or from a browser tab
  /// (including this one): the server broadcasts this the moment a turn begins, rather
  /// than a submitting tab optimistically rendering its own input locally, precisely so
  /// every tab (and only every tab, none of them specially) renders the same thing for
  /// the same reason. `origin` says which front-end started it — see
  /// [`MessageOrigin`]'s docs for why that matters to a renderer.
  UserMessage { text: String, origin: MessageOrigin },
  /// A chunk of assistant text.
  Token { text: String },
  /// The model requested these tool calls; they are about to run.
  ToolCallsStarted { calls: Vec<ToolCallSummary> },
  /// The immediately preceding `ToolCallsStarted` batch has finished.
  ToolCallsFinished { results: Vec<ToolResultSummary> },
  /// A dangerous tool call (see `agent::callback::dual_approval::DualApprovalCallback`)
  /// is waiting on a decision. Resolve it with `POST /api/approve/{id}`, `id` being this
  /// tool call's `id` — the same one it carries in the enclosing `ToolCallsStarted`.
  ApprovalRequired {
    id: String,
    tool: String,
    /// The raw (unparsed) JSON arguments string — see
    /// `agent::agent::callback::ToolCallView::raw_arguments`'s docs for why this is not
    /// parsed here: showing a human what they are approving should never fail just
    /// because the model produced a malformed payload.
    arguments: String,
  },
  /// `id` was just resolved (by a `POST /api/approve/{id}` call, from this browser tab
  /// or another one) with `approved`. Sent so every tab watching this turn can update
  /// its `ApprovalRequired` prompt, not just the one that submitted the decision.
  ApprovalResolved { id: String, approved: bool },
  /// The turn finished normally. `turn` is the id its front-end was given when it
  /// started it (`ChatAccepted::turn` for a browser-submitted one), and is what lets a
  /// tab tell *its own* turn ending from any of the other turns sharing this stream —
  /// several may be queued behind each other at once (see `run_turn_stream`'s
  /// `turn_lock` docs in `src/bin/cli/main.rs`), so "some turn finished" says nothing
  /// about whether the one this tab submitted did.
  Done {
    turn: String,
    budget_exhausted: bool,
  },
  /// The turn failed outrightly (as opposed to an individual tool call failing, which
  /// surfaces as a normal `ToolCallsFinished { status: Error, .. }` and does not stop
  /// the turn). Always the last frame of that turn when present.
  ///
  /// `turn` is `None` only for the rare error that belongs to no turn at all — the
  /// server failing to encode a frame it was about to send (see `to_sse_event` in
  /// `src/bin/cli/web.rs`); such a frame is worth showing, but ends nothing.
  Error {
    turn: Option<String>,
    message: String,
  },
}

/// One entry of the transcript returned by `GET /api/history`. Mirrors
/// `agent::agent::Event`'s shape (see that type's docs) rather than reusing it directly,
/// for the same reason the rest of this crate does not depend on `agent` (see the module
/// docs).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
  pub author: String,
  pub timestamp: i64,
  pub content: Vec<HistoryContentItem>,
}

/// Mirrors `agent::agent::ContentItem`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum HistoryContentItem {
  Message {
    role: String,
    content: String,
  },
  ToolCall {
    id: String,
    name: String,
    arguments: Value,
  },
  ToolResult {
    id: String,
    name: String,
    status: ToolStatus,
    content: String,
  },
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn chat_event_round_trips_through_json_with_the_tagged_shape() {
    let event = ChatEvent::ApprovalRequired {
      id: "call-1".to_owned(),
      tool: "delete_file".to_owned(),
      arguments: "{\"path\":\"notes.txt\"}".to_owned(),
    };
    let json = serde_json::to_value(&event).unwrap();
    assert_eq!(json["type"], "approval_required");
    assert_eq!(json["tool"], "delete_file");

    let back: ChatEvent = serde_json::from_value(json).unwrap();
    assert!(matches!(back, ChatEvent::ApprovalRequired { id, .. } if id == "call-1"));
  }

  #[test]
  fn history_entry_round_trips_through_json() {
    let entry = HistoryEntry {
      author: "agent".to_owned(),
      timestamp: 0,
      content: vec![HistoryContentItem::ToolResult {
        id: "call-1".to_owned(),
        name: "calculator".to_owned(),
        status: ToolStatus::Success,
        content: "42".to_owned(),
      }],
    };
    let json = serde_json::to_string(&entry).unwrap();
    let back: HistoryEntry = serde_json::from_str(&json).unwrap();
    assert_eq!(back.content.len(), 1);
  }
}
