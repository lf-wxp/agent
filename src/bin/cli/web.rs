//! Local web front-end for the same `Agent`/session state the terminal loop in
//! `main.rs` drives — the browser is another way to interact with *this* process, not a
//! separate deployment (see `docs/web-ui-plan.md`).
//!
//! Every turn this process runs — typed in the terminal or submitted from a browser tab
//! — is broadcast to every connected `GET /api/stream` subscriber via [`WebState::events`]
//! (see that field's docs): the browser is a live view of the shared conversation, not
//! just a way to talk to it. `POST /api/chat` only *starts* a turn (see [`chat_handler`]);
//! it does not itself stream anything back — the caller (any tab, including the one that
//! submitted it) sees the turn unfold by watching `/api/stream`, same as every other tab.
//!
//! Bound to `127.0.0.1` only (see [`serve`]'s caller in `main.rs`), and there is no
//! authentication of any kind on these routes: there is no second user on this machine
//! to keep out, the same reasoning [`super::LOCAL_SCOPE`] already relies on. Do not put
//! this behind a `0.0.0.0` bind or a public reverse proxy without adding some.

use std::{
  collections::HashMap,
  convert::Infallible,
  net::SocketAddr,
  path::Path as FsPath,
  sync::{Arc, Mutex as StdMutex},
};

use agent::{
  Agent, AgentStreamEvent,
  agent::{ContentItem, Event, ToolResultStatus},
  callback::dual_approval::{ApprovalChannel, PendingWebApproval},
  session::{FileSessionStore, SessionStore},
};
use axum::{
  Json, Router,
  extract::{Path, State},
  http::{HeaderValue, Method, StatusCode, header},
  response::sse::{Event as SseEvent, KeepAlive, Sse},
  routing::{get, post},
};
use futures::{Stream, StreamExt};
use shared::{
  ApprovalDecision, ChatAccepted, ChatEvent, ChatRequest, HistoryContentItem, HistoryEntry,
  MessageOrigin, ToolCallSummary, ToolResultSummary, ToolStatus,
};
use tokio::sync::{Mutex as AsyncMutex, broadcast, mpsc, oneshot};
use tower_http::{
  cors::{AllowOrigin, CorsLayer},
  services::{ServeDir, ServeFile},
};

use super::{LOCAL_SCOPE, run_turn_stream};

/// Capacity of [`WebState::events`]: how many not-yet-delivered frames a slow/disconnected
/// subscriber may fall behind by before [`broadcast::Sender::send`] starts overwriting
/// its oldest ones (surfacing to that subscriber as a skipped [`broadcast::error::
/// RecvError::Lagged`], handled in [`stream_handler`] by just continuing — a dropped
/// frame or two on a lagging tab is an acceptable trade-off for never blocking the turn
/// that produced them).
const EVENT_BUFFER_CAPACITY: usize = 256;

/// State shared by every route in [`serve`]'s router — and, for `agent`/`store`/
/// `turn_lock`/`events`, the exact same `Arc`/[`broadcast::Sender`] clones the terminal
/// loop in `main.rs` holds, not independent copies. See [`super::run_turn_stream`]'s
/// `turn_lock` docs for why sharing the lock (not just the underlying data) matters once
/// a terminal- and a web-originated turn can run at the same time (`--mode both`).
pub struct WebState {
  agent: Arc<Agent>,
  store: Arc<FileSessionStore>,
  turn_lock: Arc<AsyncMutex<()>>,
  session_id: String,
  /// Tool calls waiting on a browser's approve/deny decision, keyed by tool call id.
  /// [`drive_turn`] inserts an entry when a [`PendingWebApproval`] for a web-originated
  /// turn comes in; [`approve_handler`] removes it once a decision arrives. A plain
  /// [`std::sync::Mutex`] (not `tokio`'s) is enough here: every critical section
  /// touching it is a single, non-blocking `HashMap` operation, never held across an
  /// `.await`.
  pending_approvals: StdMutex<HashMap<String, oneshot::Sender<bool>>>,
  /// Every [`ChatEvent`] this process produces, from *any* turn regardless of which
  /// front-end started it — this is the one channel that makes a terminal-typed message
  /// (or one from a different browser tab) show up here. [`main.rs`]'s terminal loop
  /// holds its own clone of the exact same sender (constructed once, before this
  /// `WebState` even exists — see `main.rs`) and calls [`to_chat_events`] itself for the
  /// same reason [`drive_turn`] does below: neither side special-cases the other's
  /// origin past choosing an [`ApprovalChannel`] for its own approval prompts.
  /// [`stream_handler`] is `GET /api/stream`'s whole implementation: subscribe, forward.
  events: broadcast::Sender<ChatEvent>,
}

impl WebState {
  pub fn new(
    agent: Arc<Agent>,
    store: Arc<FileSessionStore>,
    turn_lock: Arc<AsyncMutex<()>>,
    session_id: String,
    events: broadcast::Sender<ChatEvent>,
  ) -> Self {
    Self {
      agent,
      store,
      turn_lock,
      session_id,
      pending_approvals: StdMutex::new(HashMap::new()),
      events,
    }
  }
}

/// A fresh, unsubscribed broadcast sender for [`WebState::events`]/`main.rs`'s terminal
/// loop to share. Split out of [`WebState::new`] so `main.rs` can create it *before*
/// `WebState` exists (it needs its own clone for the terminal loop regardless of
/// `--mode`, even in `--mode cli` where no `WebState` is ever built at all — see that
/// module's docs) — capacity is [`EVENT_BUFFER_CAPACITY`].
pub fn new_event_channel() -> broadcast::Sender<ChatEvent> {
  let (sender, _receiver) = broadcast::channel(EVENT_BUFFER_CAPACITY);
  sender
}

/// Build the router and serve it on `addr` until it errors or the process exits.
///
/// `dist_dir` is `crates/web-ui`'s `trunk build` output (see
/// [`agent::config::cli_web_dist_dir`]) — a missing directory is not fatal here, it just
/// means every request under it 404s and only the `/api/*` routes below work. That is a
/// deliberately usable degraded state, not just a tolerated one: it is the same thing an
/// engineer wants while developing the front-end separately via `trunk serve` (which
/// proxies its own dev server's API calls to this one instead of serving them itself).
pub async fn serve(
  state: Arc<WebState>,
  addr: SocketAddr,
  dist_dir: &FsPath,
) -> anyhow::Result<()> {
  let index_html = dist_dir.join("index.html");
  let static_files = ServeDir::new(dist_dir).fallback(ServeFile::new(index_html));

  let app = Router::new()
    .route("/api/chat", post(chat_handler))
    .route("/api/history", get(history_handler))
    .route("/api/stream", get(stream_handler))
    .route("/api/approve/{id}", post(approve_handler))
    .fallback_service(static_files)
    .layer(loopback_cors_layer())
    .with_state(state);

  let listener = tokio::net::TcpListener::bind(addr).await?;
  tracing::info!("web UI listening on http://{addr}");
  axum::serve(listener, app).await?;
  Ok(())
}

/// Lets a page served from *another* loopback port call these routes — the `trunk serve`
/// dev workflow [`serve`] describes above, for a dev server pointed straight at this one
/// rather than proxying `/api/*` through itself. Without this, the browser blocks every
/// such call, which is what the `cors` feature this crate already enables is for.
///
/// Scoped to loopback origins rather than [`AllowOrigin::any`] on purpose: these routes
/// have no authentication (see the module docs) because there is no second *user* on
/// this machine, but "any origin" would additionally hand every website this machine's
/// browser happens to visit the ability to drive this agent — a different thing
/// entirely, and not one the local-only threat model covers.
fn loopback_cors_layer() -> CorsLayer {
  CorsLayer::new()
    .allow_origin(AllowOrigin::predicate(|origin, _request| {
      is_loopback_origin(origin)
    }))
    .allow_methods([Method::GET, Method::POST])
    .allow_headers([header::CONTENT_TYPE])
}

/// Whether an `Origin` header names this machine over plain HTTP — `http://localhost`,
/// `http://127.0.0.1:8080`, `http://[::1]:8080`, … An `https` origin is deliberately not
/// matched: nothing serves this front-end over TLS, so such an origin can only be
/// something else claiming a loopback name.
fn is_loopback_origin(origin: &HeaderValue) -> bool {
  let Ok(origin) = origin.to_str() else {
    return false;
  };
  let Some(authority) = origin.strip_prefix("http://") else {
    return false;
  };
  // Split the port off, keeping in mind that an IPv6 host is bracketed (`[::1]:8080`)
  // and so cannot simply be cut at its first `:`.
  let host = match authority.strip_prefix('[') {
    Some(rest) => rest.split(']').next().unwrap_or_default(),
    None => authority.split(':').next().unwrap_or_default(),
  };
  matches!(host, "localhost" | "127.0.0.1" | "::1")
}

/// A fresh id for one turn, unique across every front-end this process drives (see
/// [`ChatEvent::Done`]'s `turn`). Assigned when a turn is *submitted*, not when it
/// actually starts running — a browser tab needs it in `POST /api/chat`'s response,
/// which is answered long before the turn reaches the front of `turn_lock`'s queue.
pub(crate) fn new_turn_id() -> String {
  uuid::Uuid::new_v4().to_string()
}

/// `POST /api/chat`: kick off one turn and return immediately (`202 Accepted`) without
/// waiting for any of it — [`drive_turn`] runs detached (`tokio::spawn`), broadcasting
/// its events to [`WebState::events`] as they occur rather than streaming them back on
/// this response. A caller (any browser tab, including this request's own) watches
/// `GET /api/stream` to see the turn unfold; the only thing this response says about the
/// turn is which one on that stream it is ([`ChatAccepted`]).
///
/// Blank input is rejected with `400` rather than accepted: the terminal loop skips an
/// empty line without starting a turn at all, and a turn started here would otherwise
/// broadcast an empty `UserMessage` and persist an empty user entry to the shared
/// transcript that every front-end then has to render.
async fn chat_handler(
  State(state): State<Arc<WebState>>,
  Json(request): Json<ChatRequest>,
) -> Result<(StatusCode, Json<ChatAccepted>), StatusCode> {
  if request.input.trim().is_empty() {
    return Err(StatusCode::BAD_REQUEST);
  }
  let turn = new_turn_id();
  tokio::spawn(drive_turn(state, turn.clone(), request.input));
  Ok((StatusCode::ACCEPTED, Json(ChatAccepted { turn })))
}

/// Runs one web-originated turn to completion, broadcasting every event it produces —
/// including its own [`PendingWebApproval`] prompts and resolutions — to
/// [`WebState::events`] rather than returning them anywhere. Spawned by [`chat_handler`]
/// and outlives that request; nothing here depends on the HTTP response still being
/// open. `turn` is the id that response already handed the submitting tab, so the
/// [`ChatEvent::Done`]/[`ChatEvent::Error`] ending this turn is recognizable as *its*
/// turn ending among every other turn sharing the same stream.
async fn drive_turn(state: Arc<WebState>, turn: String, input: String) {
  let _ = state.events.send(ChatEvent::UserMessage {
    text: input.clone(),
    origin: MessageOrigin::Web,
  });

  let (approval_tx, mut approval_rx) = mpsc::unbounded_channel::<PendingWebApproval>();
  let stream = run_turn_stream(
    &state.agent,
    &state.store,
    &state.turn_lock,
    &state.session_id,
    &input,
    ApprovalChannel::Web(approval_tx),
  );
  futures::pin_mut!(stream);

  loop {
    tokio::select! {
      pending = approval_rx.recv() => {
        let Some(pending) = pending else {
          // The turn's last `ApprovalChannel::Web` clone was dropped: not itself a
          // reason to stop this loop (`stream` below is the actual end-of-stream
          // signal), just nothing left to ever receive here again. Read it once more
          // every future iteration would just get another `None` immediately.
          continue;
        };
        let PendingWebApproval { id, tool, raw_arguments, decision } = pending;
        state.pending_approvals.lock().unwrap().insert(id.clone(), decision);
        let _ = state.events.send(ChatEvent::ApprovalRequired { id, tool, arguments: raw_arguments });
      }
      next = stream.next() => {
        match next {
          Some(Ok(event)) => {
            for chat_event in to_chat_events(&event, &turn) {
              let _ = state.events.send(chat_event);
            }
          }
          Some(Err(err)) => {
            let _ = state.events.send(ChatEvent::Error {
              turn: Some(turn.clone()),
              message: format!("{err:#}"),
            });
            break;
          }
          None => break,
        }
      }
    }
  }
}

/// `POST /api/approve/{id}`: resolve a pending [`ChatEvent::ApprovalRequired`] by tool
/// call id. `404` if `id` is unknown — already resolved, never existed, or belonged to a
/// turn that has since finished/errored (its sender was dropped, which
/// [`DualApprovalCallback`](agent::callback::dual_approval::DualApprovalCallback) treats
/// as a denial on the other end regardless of whether anyone ever calls this route for
/// it). Broadcasts [`ChatEvent::ApprovalResolved`] on success so every tab watching this
/// turn — not just the one that submitted the decision — updates its prompt.
async fn approve_handler(
  State(state): State<Arc<WebState>>,
  Path(id): Path<String>,
  Json(decision): Json<ApprovalDecision>,
) -> StatusCode {
  let sender = state.pending_approvals.lock().unwrap().remove(&id);
  match sender {
    // `send` fails only if the receiving `DualApprovalCallback::prompt_web` call already
    // gave up (e.g. its turn errored out concurrently) — nothing left to notify, and
    // this decision arrived too late to matter either way.
    Some(sender) => match sender.send(decision.approved) {
      Ok(()) => {
        let _ = state.events.send(ChatEvent::ApprovalResolved {
          id,
          approved: decision.approved,
        });
        StatusCode::NO_CONTENT
      }
      Err(_) => StatusCode::GONE,
    },
    None => StatusCode::NOT_FOUND,
  }
}

/// `GET /api/history`: the current session's full transcript, for a freshly opened (or
/// refreshed) browser tab to render before it starts watching `GET /api/stream` for
/// anything new.
async fn history_handler(State(state): State<Arc<WebState>>) -> Json<Vec<HistoryEntry>> {
  let events = state.store.history(LOCAL_SCOPE, &state.session_id).await;
  Json(events.into_iter().map(to_history_entry).collect())
}

/// `GET /api/stream`: the persistent, long-lived connection a browser tab opens once
/// (typically on page load) and keeps open for as long as the tab is open — every
/// [`ChatEvent`] any turn produces from then on, from *any* origin, arrives here. This
/// is the entire mechanism behind "a message typed in the terminal shows up in the
/// browser": nothing origin-specific happens in this handler, it is just "subscribe,
/// forward". A slow/disconnected subscriber that falls behind by more than
/// [`EVENT_BUFFER_CAPACITY`] frames silently skips the ones it missed
/// ([`broadcast::error::RecvError::Lagged`]) rather than closing the connection —
/// losing a few intermediate token chunks is preferable to dropping the whole page's
/// live view over a momentary stall.
async fn stream_handler(
  State(state): State<Arc<WebState>>,
) -> Sse<impl Stream<Item = Result<SseEvent, Infallible>>> {
  let mut receiver = state.events.subscribe();
  let stream = async_stream::stream! {
    loop {
      match receiver.recv().await {
        Ok(event) => yield to_sse_event(event),
        Err(broadcast::error::RecvError::Lagged(_)) => continue,
        Err(broadcast::error::RecvError::Closed) => break,
      }
    }
  };
  Sse::new(stream).keep_alive(KeepAlive::default())
}

fn to_history_entry(event: Event) -> HistoryEntry {
  HistoryEntry {
    author: event.author,
    timestamp: event.timestamp,
    content: event
      .content
      .into_iter()
      .map(to_history_content_item)
      .collect(),
  }
}

fn to_history_content_item(item: ContentItem) -> HistoryContentItem {
  match item {
    ContentItem::Message { role, content } => HistoryContentItem::Message { role, content },
    ContentItem::ToolCall {
      tool_call_id,
      name,
      arguments,
    } => HistoryContentItem::ToolCall {
      id: tool_call_id,
      name,
      arguments,
    },
    ContentItem::ToolResult {
      tool_call_id,
      name,
      status,
      content,
    } => HistoryContentItem::ToolResult {
      id: tool_call_id,
      name,
      status: to_tool_status(status),
      content,
    },
  }
}

/// Every [`AgentStreamEvent`] variant maps to exactly one [`ChatEvent`], except
/// [`AgentStreamEvent::ToolCallsStarted`]/[`AgentStreamEvent::ToolCallsFinished`], whose
/// [`ContentItem`]s are always [`ContentItem::ToolCall`]/[`ContentItem::ToolResult`] by
/// construction (see `Agent::record_tool_calls`/`Agent::execute_tool_calls`) — returning
/// a `Vec` here rather than a single value is just to let a well-formed input still
/// silently drop anything unexpected instead of panicking, not because more than one
/// output is normally expected.
///
/// Takes `event` by reference (cloning the handful of `String`/`Vec<ContentItem>` fields
/// it needs) rather than by value: [`super::main`]'s terminal loop needs the original
/// `event` afterwards too, to print it, and `pub(crate)` so that loop (in the parent
/// module) can call this directly instead of duplicating the conversion.
///
/// `turn` identifies the turn being converted (see [`new_turn_id`]) — carried on the
/// terminating [`ChatEvent::Done`] only, since that is the one frame a front-end has to
/// attribute to a specific turn rather than to "the conversation".
pub(crate) fn to_chat_events(event: &AgentStreamEvent, turn: &str) -> Vec<ChatEvent> {
  match event {
    AgentStreamEvent::Token(text) => vec![ChatEvent::Token { text: text.clone() }],
    AgentStreamEvent::ToolCallsStarted(items) => vec![ChatEvent::ToolCallsStarted {
      calls: items
        .iter()
        .cloned()
        .filter_map(to_tool_call_summary)
        .collect(),
    }],
    AgentStreamEvent::ToolCallsFinished(items) => vec![ChatEvent::ToolCallsFinished {
      results: items
        .iter()
        .cloned()
        .filter_map(to_tool_result_summary)
        .collect(),
    }],
    AgentStreamEvent::Done {
      budget_exhausted, ..
    } => vec![ChatEvent::Done {
      turn: turn.to_owned(),
      budget_exhausted: *budget_exhausted,
    }],
  }
}

fn to_tool_call_summary(item: ContentItem) -> Option<ToolCallSummary> {
  match item {
    ContentItem::ToolCall {
      tool_call_id,
      name,
      arguments,
    } => Some(ToolCallSummary {
      id: tool_call_id,
      name,
      arguments,
    }),
    _ => None,
  }
}

fn to_tool_result_summary(item: ContentItem) -> Option<ToolResultSummary> {
  match item {
    ContentItem::ToolResult {
      tool_call_id,
      name,
      status,
      content,
    } => Some(ToolResultSummary {
      id: tool_call_id,
      name,
      status: to_tool_status(status),
      content,
    }),
    _ => None,
  }
}

fn to_tool_status(status: ToolResultStatus) -> ToolStatus {
  match status {
    ToolResultStatus::Success => ToolStatus::Success,
    ToolResultStatus::Error => ToolStatus::Error,
  }
}

/// Encode one [`ChatEvent`] as an SSE frame. Encoding failure is not expected — every
/// field here is already `String`/`serde_json::Value` — but is handled by degrading to a
/// [`ChatEvent::Error`] frame rather than panicking or silently dropping the event, on
/// the theory that a browser mid-stream should hear about a broken frame rather than
/// just missing one with no explanation.
fn to_sse_event(event: ChatEvent) -> Result<SseEvent, Infallible> {
  let json = serde_json::to_string(&event).unwrap_or_else(|err| {
    serde_json::to_string(&ChatEvent::Error {
      // Ends no turn: whichever turn produced the unencodable frame is still running,
      // and its own `Done`/`Error` is still coming (see `ChatEvent::Error`'s docs).
      turn: None,
      message: format!("failed to encode event: {err}"),
    })
    .expect("ChatEvent::Error always encodes")
  });
  Ok(SseEvent::default().event("chat").data(json))
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn only_plain_http_loopback_origins_are_allowed() {
    for origin in [
      "http://localhost",
      "http://localhost:8080",
      "http://127.0.0.1:1420",
      "http://[::1]:8080",
    ] {
      assert!(
        is_loopback_origin(&HeaderValue::from_static(origin)),
        "{origin} is this machine and should be allowed"
      );
    }

    for origin in [
      "https://localhost",
      "http://localhost.example.com",
      "http://127.0.0.1.example.com",
      "http://example.com",
      "null",
    ] {
      assert!(
        !is_loopback_origin(&HeaderValue::from_static(origin)),
        "{origin} is not this machine and should be rejected"
      );
    }
  }
}
