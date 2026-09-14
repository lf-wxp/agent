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

pub mod commands;

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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalDecision {
  pub approved: bool,
  /// Apply this answer to every later call of the same tool in this conversation,
  /// instead of only to the call being decided.
  ///
  /// `#[serde(default)]` so a front-end bundle built before this field existed still
  /// deserializes — as a one-off decision, which is what it meant to send. Worth the
  /// leniency here specifically: a stale bundle is a normal state for this page (see
  /// `listen_stream`'s note on the same hazard), and the safe reading of a missing field
  /// is the narrower of the two scopes.
  #[serde(default)]
  pub sticky: bool,
  /// Why the call was refused, recorded in place of it so the model learns what to do
  /// instead of merely that it was stopped. Ignored when `approved`.
  #[serde(default)]
  pub reason: Option<String>,
}

/// One approval still awaiting a decision, as returned by `GET /api/approvals`.
///
/// Carries exactly what [`ChatEvent::ApprovalRequired`] does, because it answers the same
/// question for a view that arrived too late to have received that event: the broadcast
/// behind `GET /api/stream` only reaches subscribers present when a frame is sent and
/// never replays, so a tab opened (or reloaded) while a turn sits waiting on an approval
/// would otherwise have no way to learn of it — and, since an approval is deliberately
/// not part of the transcript, `GET /api/history` cannot cover for that either.
///
/// Named apart from the native side's `PendingApproval` on purpose: that type owns the
/// decision channel the agent is blocked on, which is neither serializable nor meaningful
/// off-process. This is only the description.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingApprovalView {
  pub id: String,
  pub tool: String,
  /// The raw (unparsed) JSON arguments string, for the same reason
  /// [`ChatEvent::ApprovalRequired`] carries it unparsed.
  pub arguments: String,
  /// Unix seconds at which the prompt was raised. Lets a client show how long something
  /// has been waiting — relevant because an unanswered approval eventually suspends the
  /// turn (see [`ChatEvent::TurnSuspended`]), so a prompt left alone does not stay live
  /// indefinitely.
  pub requested_at: i64,
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
  /// Output that came from the process itself rather than the model — a `/help` listing,
  /// a confirmation that history was cleared. Broadcast like anything else so every view
  /// sees it, including views that did not issue the command.
  ///
  /// Distinct from [`Self::Token`] on purpose: this is not part of the conversation and
  /// is never persisted to the transcript, so a renderer should style it as an aside
  /// rather than as something the model said.
  ///
  /// Only ever produced by an in-chat command, and a command is resolved *before* a turn
  /// starts (it never reaches the model and never takes the turn lock) — so a notice
  /// never arrives mid-turn, and one arriving means no turn is running. A renderer
  /// showing a "waiting for the model" affordance may rely on that to take it down:
  /// the [`Self::UserMessage`] echoing a command looks exactly like one that starts a
  /// turn, and for a command there is no [`Self::Done`] guaranteed to follow.
  SystemNotice { text: String },
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
  /// The turn stopped without finishing, because nobody decided an approval it was
  /// waiting on. It is stored and can be carried on later with `POST /api/resume/{run}`.
  ///
  /// Ends a turn the same way [`Self::Done`] does — a renderer must take down its
  /// "waiting for the model" affordance on either — but says something different about
  /// what happened: no answer was produced, and the transcript was *not* added to the
  /// conversation. The turn's user message is held inside the stored run rather than in
  /// history, so it reappears when the run is resumed rather than being lost.
  ///
  /// Deliberately not modelled as an error. Nothing went wrong; a question was asked and
  /// is still open, which is the approval mechanism working.
  TurnSuspended {
    turn: String,
    /// Identifies the stored run. What `POST /api/resume/{run}` takes, and what the
    /// terminal's `/resume` uses.
    run: String,
    /// What the run is still waiting on, so a view that missed the original
    /// [`Self::ApprovalRequired`] — or has been reloaded since — can show it without a
    /// second request.
    pending: Vec<PendingApprovalView>,
  },
  /// The suspended run is gone — taken up by a resume, or given up on.
  ///
  /// Separate from the turn that may follow, because the two are different facts and a
  /// view needs the first without waiting for the second: a resume broadcasts this the
  /// moment the stored run is claimed, which is before the model has produced anything.
  ///
  /// A renderer showing a "paused" affordance should take it down here and *only* here.
  /// Inferring it from a turn starting looks equivalent but is not: a turn refused
  /// because a run is suspended, and an in-chat command, both echo a
  /// [`Self::UserMessage`] without anything having been claimed — so treating that as
  /// the signal removes the one control the user needs, precisely when they need it.
  SuspendedRunCleared,
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
  fn pending_approval_view_round_trips_through_json() {
    let view = PendingApprovalView {
      id: "call-1".to_owned(),
      tool: "delete_file".to_owned(),
      arguments: "{\"path\":\"notes.txt\"}".to_owned(),
      requested_at: 1_700_000_000,
    };
    let json = serde_json::to_string(&view).unwrap();
    let back: PendingApprovalView = serde_json::from_str(&json).unwrap();

    assert_eq!(back.id, "call-1");
    assert_eq!(
      back.arguments, "{\"path\":\"notes.txt\"}",
      "the raw payload must survive intact — it is what a human judges the call by"
    );
    assert_eq!(back.requested_at, 1_700_000_000);
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
