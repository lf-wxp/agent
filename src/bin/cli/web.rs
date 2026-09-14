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
//! to keep out, the same reasoning [`super::LOCAL_SCOPE`] already relies on. What that
//! reasoning does *not* cover is a request that reaches this server from outside the
//! machine's own trust boundary while still looking local — DNS rebinding — so every
//! `/api/*` request additionally has to arrive under a loopback host name (see
//! [`guard_loopback_host`]). Do not put this behind a `0.0.0.0` bind or a public reverse
//! proxy without adding real authentication; the host check is a guard against one
//! specific trick, not a substitute for one.

use std::{convert::Infallible, net::SocketAddr, path::Path as FsPath, pin::Pin, sync::Arc};

use agent::{
  Agent, AgentStreamEvent,
  agent::{ApprovalStore, ContentItem, Event, FileApprovalStore, ToolResultStatus},
  callback::dual_approval::{
    ApprovalChannel, ApprovalMeta, ApprovalOutcome, ApprovalRegistry, DualApprovalCallback,
    PendingApproval,
  },
  session::{FileSessionStore, SessionStore},
};
use anyhow::Context;
use axum::{
  Json, Router,
  extract::{Path, Request, State},
  http::{HeaderValue, Method, StatusCode, header},
  middleware::{self, Next},
  response::{
    IntoResponse, Response,
    sse::{Event as SseEvent, KeepAlive, Sse},
  },
  routing::{get, post},
};
use futures::{Stream, StreamExt};
use shared::{
  ApprovalDecision, ChatAccepted, ChatEvent, ChatRequest, HistoryContentItem, HistoryEntry,
  MessageOrigin, PendingApprovalView, SuspendedRunView, ToolCallSummary, ToolResultSummary,
  ToolStatus,
};
use tokio::sync::{Mutex as AsyncMutex, broadcast, mpsc};
use tower_http::{
  cors::{AllowOrigin, CorsLayer},
  services::{ServeDir, ServeFile},
};

use super::{LOCAL_SCOPE, resume_turn_stream, run_turn_stream};

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
  /// The session's in-flight approvals — the *same* registry the terminal loop in
  /// `main.rs` holds, not a private copy. That sharing is the whole point: a prompt
  /// raised by a terminal-typed turn is answerable here, and one raised by a
  /// browser-submitted turn is answerable there. See
  /// [`agent::callback::dual_approval`]'s module docs.
  approvals: Arc<ApprovalRegistry>,
  /// The approval callback, when anything is gated — the *same* one the agent holds as a
  /// hook, so a `/reset` submitted here clears the remembered "always allow" answers that
  /// callback would otherwise keep applying. `None` under `--no-approval`/an empty
  /// `--dangerous-tools`, where nothing is gated and so nothing can have been remembered.
  approval_callback: Option<Arc<DualApprovalCallback>>,
  /// Where a turn that stopped waiting on an approval is kept — the *same* store the
  /// terminal loop holds, for the same reason the registry above is shared: a run
  /// suspended by a terminal-typed turn must be resumable from a browser tab, and the
  /// other way round.
  approvals_store: Arc<FileApprovalStore>,
  /// Every [`ChatEvent`] this process produces, from *any* turn regardless of which
  /// front-end started it — this is the one channel that makes a terminal-typed message
  /// (or one from a different browser tab) show up here. `main.rs`'s terminal loop
  /// holds its own clone of the exact same sender (constructed once, before this
  /// `WebState` even exists — see `main.rs`) and calls [`to_chat_events`] itself for the
  /// same reason [`drive_turn`] does below: neither side special-cases the other's
  /// origin past choosing an [`ApprovalChannel`] for its own approval prompts.
  /// [`stream_handler`] is `GET /api/stream`'s whole implementation: subscribe, forward.
  events: broadcast::Sender<ChatEvent>,
}

impl WebState {
  // Eight `Arc`/handle parameters, each a distinct piece of process-wide state this
  // binary already owns. A parameter struct would only move the same list one level out
  // and add a type whose sole purpose is to be destructured here.
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    agent: Arc<Agent>,
    store: Arc<FileSessionStore>,
    turn_lock: Arc<AsyncMutex<()>>,
    session_id: String,
    approvals: Arc<ApprovalRegistry>,
    approval_callback: Option<Arc<DualApprovalCallback>>,
    approvals_store: Arc<FileApprovalStore>,
    events: broadcast::Sender<ChatEvent>,
  ) -> Self {
    Self {
      agent,
      store,
      turn_lock,
      session_id,
      approvals,
      approval_callback,
      approvals_store,
      events,
    }
  }

  /// Whether this session has a turn waiting on an approval.
  async fn has_suspended_run(&self) -> bool {
    self.suspended_run().await.is_some()
  }

  /// This session's suspended run in the wire shape, or `None` if nothing is suspended.
  ///
  /// `requested_at` is stamped at read time rather than carried from when the prompt was
  /// first raised: a stored run has no live clock behind it, and reporting the original
  /// instant would have a client render "waiting for 14 hours" as though something were
  /// still counting down. Nothing is — it waits until answered.
  async fn suspended_run(&self) -> Option<SuspendedRunView> {
    let now = chrono::Utc::now().timestamp();
    let view = self
      .approvals_store
      .peek(LOCAL_SCOPE, &self.session_id)
      .await?;
    Some(SuspendedRunView {
      pending: view
        .pending
        .into_iter()
        .map(|call| PendingApprovalView {
          id: call.tool_call_id,
          tool: call.name,
          arguments: call.raw_arguments,
          requested_at: now,
        })
        .collect(),
      transcript: view.events.into_iter().map(to_history_entry).collect(),
    })
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

/// Claim `addr` up front, so that failing to (the port already being in use, most
/// often) is reported to whoever asked for the server rather than discovered later.
///
/// Split from [`serve`] because `--mode both` runs the server as a detached task whose
/// result nobody is left to inspect: binding *there* meant an "already in use" error
/// vanishing into a dropped `JoinHandle` while the startup banner had already claimed
/// the UI was up. Binding here keeps that failure on the caller's own `?`, and also
/// means the banner is only printed once the port is genuinely claimed — a browser
/// opened the instant it appears can no longer beat the listener to it.
pub async fn bind(addr: SocketAddr) -> anyhow::Result<tokio::net::TcpListener> {
  tokio::net::TcpListener::bind(addr)
    .await
    .with_context(|| format!("failed to bind the web UI to {addr}"))
}

/// Why the front-end cannot be served from `dist_dir`, if it cannot.
///
/// [`serve`] treats a missing `dist_dir` as a usable degraded state (see its docs) and
/// that is worth keeping — but it must not be a *silent* one. `dist_dir` is `trunk`
/// output and is not in version control, so a fresh clone has none of it: without this,
/// the first thing a new checkout does on `--mode web`/`both` is answer the page with a
/// bare 404 and no indication anywhere that a front-end build was the missing step.
///
/// `None` means the assets are there and nothing needs saying.
pub fn dist_warning(dist_dir: &FsPath) -> Option<String> {
  let how_to_fix = "run `trunk build --release` in `crates/web-ui` (or point \
                    AGENT_CLI_WEB_DIST_DIR at an existing build). The `/api/*` routes \
                    work regardless, which is all `trunk serve` needs.";

  if !dist_dir.is_dir() {
    return Some(format!(
      "no front-end build at `{}`, so the page itself will 404 — {how_to_fix}",
      dist_dir.display()
    ));
  }
  // Present but without an entry point: a half-finished or cleaned-out build directory.
  // Checked separately because `ServeDir`'s fallback is that very file, so its absence
  // 404s every route the front-end has, not just `/`.
  if !dist_dir.join("index.html").is_file() {
    return Some(format!(
      "`{}` has no `index.html`, so the page itself will 404 — {how_to_fix}",
      dist_dir.display()
    ));
  }
  None
}

/// Build the router and serve it on `listener` (see [`bind`]) until it errors or the
/// process exits.
///
/// `dist_dir` is `crates/web-ui`'s `trunk build` output (see
/// [`agent::config::cli_web_dist_dir`]) — a missing directory is not fatal here, it just
/// means every request under it 404s and only the `/api/*` routes below work. That is a
/// deliberately usable degraded state, not just a tolerated one: it is the same thing an
/// engineer wants while developing the front-end separately via `trunk serve` (which
/// proxies its own dev server's API calls to this one instead of serving them itself).
/// [`dist_warning`] is what keeps that state from being a silent one.
pub async fn serve(
  state: Arc<WebState>,
  listener: tokio::net::TcpListener,
  dist_dir: &FsPath,
) -> anyhow::Result<()> {
  let index_html = dist_dir.join("index.html");
  let static_files = ServeDir::new(dist_dir).fallback(ServeFile::new(index_html));

  let app = Router::new()
    .route("/api/chat", post(chat_handler))
    .route("/api/history", get(history_handler))
    .route("/api/stream", get(stream_handler))
    .route("/api/approvals", get(approvals_handler))
    .route("/api/approve/{id}", post(approve_handler))
    .route("/api/suspended", get(suspended_handler))
    // Only the API routes: a static asset is inert, and rejecting one would break the
    // `trunk serve` workflow above for no gain.
    .layer(middleware::from_fn(guard_loopback_host))
    .fallback_service(static_files)
    .layer(loopback_cors_layer())
    .with_state(state);

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
  is_loopback_host(host_of(authority))
}

/// The host part of an `authority`, with any port removed.
///
/// An IPv6 host is bracketed (`[::1]:8080`) and so cannot simply be cut at its first `:`.
fn host_of(authority: &str) -> &str {
  match authority.strip_prefix('[') {
    Some(rest) => rest.split(']').next().unwrap_or_default(),
    None => authority.split(':').next().unwrap_or_default(),
  }
}

fn is_loopback_host(host: &str) -> bool {
  matches!(host, "localhost" | "127.0.0.1" | "::1")
}

/// Reject any `/api/*` request that reached this server under a name other than
/// loopback's.
///
/// Closes the one hole the module's "there is no second user on this machine" reasoning
/// does not cover: **DNS rebinding**. A page the user visits can point a hostname it
/// controls at `127.0.0.1` and then have the browser issue same-origin requests to it.
/// CORS does not help — the browser believes those requests *are* same-origin, so no
/// preflight is made and [`loopback_cors_layer`] never gets a say. What gives the attack
/// away is the `Host` header: the browser sends the name it resolved, which is the
/// attacker's domain and never a loopback name.
///
/// A request with no host at all is allowed through. A rebound request always carries
/// one — carrying the attacker's domain is the entire mechanism — so denying the absent
/// case would reject odd-but-harmless clients without closing anything.
async fn guard_loopback_host(request: Request, next: Next) -> Response {
  let host = request
    .headers()
    .get(header::HOST)
    .and_then(|value| value.to_str().ok())
    .map(host_of)
    // HTTP/2 carries the authority in the URI rather than a `Host` header.
    .or_else(|| request.uri().host());

  match host {
    Some(host) if !is_loopback_host(host) => {
      tracing::warn!(
        %host,
        "rejected an API request that did not arrive over loopback; see guard_loopback_host"
      );
      StatusCode::FORBIDDEN.into_response()
    }
    _ => next.run(request).await,
  }
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
///
/// An in-chat command (see [`commands`]) is handled inline instead of becoming a turn —
/// the same as the terminal does with it — so it neither reaches the model nor waits on
/// the session's turn lock. The reply comes back as a [`ChatEvent::SystemNotice`] on the
/// shared stream, so every view sees it, not just the tab that typed it.
pub(crate) async fn chat_handler(
  State(state): State<Arc<WebState>>,
  Json(request): Json<ChatRequest>,
) -> Result<(StatusCode, Json<ChatAccepted>), StatusCode> {
  if request.input.trim().is_empty() {
    return Err(StatusCode::BAD_REQUEST);
  }

  if let Some(command) = shared::commands::parse(&request.input) {
    let turn = new_turn_id();

    // Turn-level commands are dispatched here rather than through `commands::execute`,
    // exactly as the terminal loop does — see `shared::commands::Command`.
    match command {
      shared::commands::Command::Resume => {
        let _ = state.events.send(ChatEvent::UserMessage {
          text: request.input.clone(),
          origin: MessageOrigin::Web,
        });
        if state.has_suspended_run().await {
          tokio::spawn(drive_turn(state, turn.clone(), TurnInput::Resume));
          return Ok((StatusCode::ACCEPTED, Json(ChatAccepted { turn })));
        }
        let _ = state.events.send(ChatEvent::SystemNotice {
          text: "没有被暂停的一轮可以继续。".to_owned(),
        });
      }
      shared::commands::Command::Discard => {
        let _ = state.events.send(ChatEvent::UserMessage {
          text: request.input.clone(),
          origin: MessageOrigin::Web,
        });
        let discarded = super::discard_suspended_run(
          &state.store,
          &state.approvals_store,
          &state.turn_lock,
          &state.session_id,
          &state.events,
        )
        .await;
        let _ = state.events.send(ChatEvent::SystemNotice {
          text: if discarded {
            "已放弃被暂停的一轮，已完成的部分保留在历史中。".to_owned()
          } else {
            "没有被暂停的一轮可以放弃。".to_owned()
          },
        });
      }
      _ => {
        super::commands::execute(
          command,
          &request.input,
          MessageOrigin::Web,
          &state.store,
          state.approval_callback.as_deref(),
          &state.session_id,
          &state.events,
        )
        .await;
        // A reset clears the suspended run with it: see the terminal loop's equivalent.
        if matches!(command, shared::commands::Command::Reset) {
          state
            .approvals_store
            .remove(super::LOCAL_SCOPE, &state.session_id)
            .await;
        }
      }
    }

    // Reported as an immediately-finished turn purely to release *this tab's* composer:
    // it went into "sending" when it posted and only a `Done` carrying the id in this
    // response frees it, so a command producing no turn at all would leave it disabled
    // for good. Nothing else needs this — a view's "thinking" indicator comes down on
    // the notice itself, which is what a command run from the terminal (no turn id, no
    // `Done`) relies on.
    let _ = state.events.send(ChatEvent::Done {
      turn: turn.clone(),
      budget_exhausted: false,
    });
    return Ok((StatusCode::ACCEPTED, Json(ChatAccepted { turn })));
  }

  // A turn cannot start while one is suspended — see `super::suspended_run_notice` for
  // why the two cannot coexist. Reported as a notice plus an immediate `Done`, the same
  // shape a command takes, so the tab's composer is released.
  if let Some(notice) = super::suspended_run_notice(&state.approvals_store, &state.session_id).await
  {
    let turn = new_turn_id();
    let _ = state.events.send(ChatEvent::UserMessage {
      text: request.input.clone(),
      origin: MessageOrigin::Web,
    });
    let _ = state.events.send(ChatEvent::SystemNotice { text: notice });
    let _ = state.events.send(ChatEvent::Done {
      turn: turn.clone(),
      budget_exhausted: false,
    });
    return Ok((StatusCode::ACCEPTED, Json(ChatAccepted { turn })));
  }

  let turn = new_turn_id();
  tokio::spawn(drive_turn(
    state,
    turn.clone(),
    TurnInput::Fresh(request.input),
  ));
  Ok((StatusCode::ACCEPTED, Json(ChatAccepted { turn })))
}

/// What a web turn is starting from — the counterpart of the terminal's own
/// `super::TurnInput`, and separate from it only because a spawned turn must own its
/// input rather than borrow it.
#[derive(Debug, Clone)]
pub(crate) enum TurnInput {
  Fresh(String),
  Resume,
}

/// Runs one web-originated turn to completion, broadcasting every event it produces —
/// including its own [`ChatEvent::ApprovalRequired`] prompts and their
/// [`ChatEvent::ApprovalResolved`] answers — to
/// [`WebState::events`] rather than returning them anywhere. Spawned by [`chat_handler`]
/// and outlives that request; nothing here depends on the HTTP response still being
/// open. `turn` is the id that response already handed the submitting tab, so the
/// [`ChatEvent::Done`]/[`ChatEvent::Error`] ending this turn is recognizable as *its*
/// turn ending among every other turn sharing the same stream.
///
/// `pub(crate)` to match [`chat_handler`] above: the terminal loop's own
/// `drive_terminal_turn` is documented as mirroring this function, and a module-private
/// one is not nameable from there — so the reference that explains the symmetry would
/// render as dead text.
pub(crate) async fn drive_turn(state: Arc<WebState>, turn: String, input: TurnInput) {
  if let TurnInput::Fresh(text) = &input {
    let _ = state.events.send(ChatEvent::UserMessage {
      text: text.clone(),
      origin: MessageOrigin::Web,
    });
  }

  let (approval_tx, mut approval_rx) = mpsc::unbounded_channel::<PendingApproval>();
  let channel = ApprovalChannel::Session(approval_tx);
  // Identical plumbing either way — see `drive_terminal_turn`, which mirrors this.
  let stream: Pin<Box<dyn Stream<Item = anyhow::Result<AgentStreamEvent>> + Send + '_>> =
    match &input {
      TurnInput::Fresh(text) => Box::pin(run_turn_stream(
        &state.agent,
        &state.store,
        &state.turn_lock,
        &state.session_id,
        text,
        channel,
      )),
      TurnInput::Resume => Box::pin(resume_turn_stream(
        &state.agent,
        &state.store,
        &state.approvals_store,
        &state.turn_lock,
        &state.session_id,
        channel,
        &state.events,
      )),
    };
  futures::pin_mut!(stream);

  // Ids this turn published, so anything still unanswered when it ends can be cleared
  // rather than lingering in the shared registry for the life of the process.
  let mut raised = Vec::new();

  // Once the approval channel is closed no front-end can ever send another approval, so
  // this branch is gated off and the loop is then driven by `stream` alone. Without the
  // gate, a closed receiver would resolve immediately on every iteration and `continue`
  // back into `select!`, spinning until `stream` also ended.
  let mut approvals_open = true;
  loop {
    tokio::select! {
      pending = approval_rx.recv(), if approvals_open => {
        match pending {
          Some(PendingApproval { meta, decision }) => {
            raised.push(meta.id.clone());
            let announcement = ChatEvent::ApprovalRequired {
              id: meta.id.clone(),
              tool: meta.tool.clone(),
              arguments: meta.raw_arguments.clone(),
            };
            // Into the *session-wide* registry, so a terminal sharing this session can
            // answer it too — not just the browser tab that submitted this turn. Done
            // before the announcement goes out, so a tab that reacts by immediately
            // asking `GET /api/approvals` cannot see an empty list.
            state.approvals.register(meta, decision);
            let _ = state.events.send(announcement);
          }
          // The turn's last `ApprovalChannel::Session` clone was dropped: nothing will
          // ever be received here again. `stream` below is still the end-of-stream
          // signal, so stop polling this branch rather than stopping the loop.
          None => approvals_open = false,
        }
      }
      next = stream.next() => {
        match next {
          // Stored before it is announced, so a tab that reacts by calling
          // `POST /api/resume` cannot get there before the run is on disk. Same
          // ordering, and the same reason, as registering an approval before
          // broadcasting it above.
          Some(Ok(AgentStreamEvent::Suspended(run_state))) => {
            let pending = super::record_suspension(
              &state.approvals_store,
              &state.session_id,
              &run_state,
            )
            .await;
            let _ = state.events.send(ChatEvent::TurnSuspended {
              turn: turn.clone(),
              run: state.session_id.clone(),
              pending,
            });
            break;
          }
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

  // Whatever is left here was never answered (it timed out, or the turn ended around
  // it); the waiting side has already given up and denied, so there is nothing to
  // notify — only slots to reclaim.
  state.approvals.discard(&raised);
}

/// `GET /api/approvals`: every approval this session is currently waiting on, oldest
/// first.
///
/// The companion to [`history_handler`] for a freshly opened or reloaded tab. Live
/// [`ChatEvent::ApprovalRequired`] frames only reach tabs that were already subscribed to
/// `GET /api/stream` when they were sent, and that broadcast does not replay; an approval
/// is also deliberately absent from the transcript, so `GET /api/history` does not show
/// one either. Without this route, reloading a tab while a turn sits waiting left that
/// tab unable to see — let alone answer — the prompt blocking it, with nothing to do but
/// watch it time out.
///
/// Reads the same registry [`approve_handler`] writes to and the terminal shares, so what
/// this returns is answerable immediately, whichever front-end raised it.
async fn approvals_handler(State(state): State<Arc<WebState>>) -> Json<Vec<PendingApprovalView>> {
  Json(
    state
      .approvals
      .pending_snapshot()
      .into_iter()
      .map(to_pending_approval_view)
      .collect(),
  )
}

fn to_pending_approval_view(meta: ApprovalMeta) -> PendingApprovalView {
  PendingApprovalView {
    id: meta.id,
    tool: meta.tool,
    arguments: meta.raw_arguments,
    requested_at: meta.requested_at,
  }
}

/// `GET /api/suspended`: this session's suspended run, or `null` if there is none.
///
/// The same role [`approvals_handler`] plays for a *live* prompt, for the case where the
/// turn is no longer running at all. A tab opened after a suspension — or after the
/// process restarted — missed the [`ChatEvent::TurnSuspended`] frame and will find
/// nothing in [`history_handler`] either, since a suspended turn is deliberately not
/// written to the session transcript. Without this route such a tab shows a
/// finished-looking conversation with no sign that anything is waiting.
///
/// Returns the turn's own transcript alongside the pending calls for the same reason the
/// route exists at all: it is the only place that has it.
async fn suspended_handler(State(state): State<Arc<WebState>>) -> Json<Option<SuspendedRunView>> {
  // Peeked rather than taken: this is a read for display, and taking would consume the
  // run that `POST /api/chat` with `/resume` is about to need.
  Json(state.suspended_run().await)
}

/// `POST /api/approve/{id}`: resolve a pending [`ChatEvent::ApprovalRequired`] by tool
/// call id — whichever front-end raised it. `404` if `id` is unknown: already answered
/// (possibly from the terminal, which shares this registry), timed out, or belonging to a
/// turn that has since finished. Broadcasts [`ChatEvent::ApprovalResolved`] on success so
/// every view — other tabs and the terminal alike — updates its prompt.
async fn approve_handler(
  State(state): State<Arc<WebState>>,
  Path(id): Path<String>,
  Json(decision): Json<ApprovalDecision>,
) -> StatusCode {
  let approved = decision.approved;
  let mut outcome = ApprovalOutcome {
    approved,
    sticky: decision.sticky,
    reason: None,
  };
  if let Some(reason) = decision.reason {
    // Via the builder rather than the field: it drops a blank reason, which would
    // otherwise override the run-wide formatter with an empty tool result.
    outcome = outcome.with_reason(reason);
  }
  if !state.approvals.resolve(&id, outcome) {
    return StatusCode::NOT_FOUND;
  }
  let _ = state
    .events
    .send(ChatEvent::ApprovalResolved { id, approved });
  StatusCode::NO_CONTENT
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
pub(crate) async fn stream_handler(
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
    // Handled by the turn driver rather than here: announcing a suspension means first
    // storing the run it can be resumed from, which is a side effect this conversion
    // has no business performing — and the resulting `ChatEvent::TurnSuspended` needs
    // the run id that storing it produces. See `drive_turn`.
    AgentStreamEvent::Suspended(_) => Vec::new(),
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

pub(crate) fn to_tool_result_summary(item: ContentItem) -> Option<ToolResultSummary> {
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

  fn temp_dir(label: &str) -> std::path::PathBuf {
    let path =
      std::env::temp_dir().join(format!("agent-web-dist-{label}-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&path).expect("temp dir");
    path
  }

  /// The fresh-clone case: `dist` is `trunk` output and gitignored, so this is what a new
  /// checkout hits. It has to say so rather than leave the page 404ing unexplained.
  #[test]
  fn a_missing_dist_dir_is_reported() {
    let missing = std::env::temp_dir().join(format!("agent-web-absent-{}", uuid::Uuid::new_v4()));
    let warning = dist_warning(&missing).expect("a missing build should warn");
    assert!(warning.contains("trunk build"), "{warning}");
    assert!(
      warning.contains(&missing.display().to_string()),
      "the warning should name the path it looked in: {warning}"
    );
  }

  /// A directory that exists but has no entry point 404s just as thoroughly, since
  /// `ServeDir`'s fallback *is* that file — so it cannot be treated as a working build.
  #[test]
  fn a_dist_dir_without_an_index_is_reported() {
    let dir = temp_dir("no-index");
    std::fs::write(dir.join("app.wasm"), b"not the entry point").expect("write");
    let warning = dist_warning(&dir).expect("a build with no index.html should warn");
    assert!(warning.contains("index.html"), "{warning}");
  }

  #[test]
  fn a_complete_dist_dir_warns_about_nothing() {
    let dir = temp_dir("complete");
    std::fs::write(dir.join("index.html"), b"<!doctype html>").expect("write");
    assert!(dist_warning(&dir).is_none());
  }

  /// A file where the directory should be is still "no build here", not a panic.
  #[test]
  fn a_file_in_place_of_the_dist_dir_is_reported() {
    let dir = temp_dir("as-file");
    let path = dir.join("dist");
    std::fs::write(&path, b"not a directory").expect("write");
    assert!(dist_warning(&path).is_some());
  }

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

  #[test]
  fn a_port_is_not_part_of_the_host() {
    assert_eq!(host_of("127.0.0.1:8080"), "127.0.0.1");
    assert_eq!(host_of("localhost"), "localhost");
    assert_eq!(host_of("[::1]:8080"), "::1");
    assert_eq!(host_of("[::1]"), "::1");
  }

  /// The DNS rebinding guard: a hostname the attacker controls resolves to `127.0.0.1`,
  /// so the request reaches this server — but the browser sends the name it looked up,
  /// which is never a loopback one.
  #[test]
  fn only_a_loopback_host_is_accepted() {
    for host in [
      "localhost",
      "localhost:8080",
      "127.0.0.1:1420",
      "[::1]:8080",
    ] {
      assert!(
        is_loopback_host(host_of(host)),
        "{host} is this machine and should be allowed"
      );
    }

    for host in [
      "evil.example.com",
      "evil.example.com:8080",
      // The prefix/suffix tricks that a `starts_with`/`contains` check would wave through.
      "localhost.example.com",
      "127.0.0.1.example.com",
      "not-localhost",
    ] {
      assert!(
        !is_loopback_host(host_of(host)),
        "{host} is a rebinding attempt and should be rejected"
      );
    }
  }
}
