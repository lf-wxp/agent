//! Leptos web UI entry point, built with `trunk` (see `index.html`).
//!
//! Talks to the native side's routes in `src/bin/cli/web.rs`:
//!
//! - `GET /api/history` once, on mount, to show the conversation already on disk.
//! - `GET /api/stream`, opened once on mount and kept open for as long as this tab is,
//!   for every live [`ChatEvent`] this process produces from then on — from *any*
//!   origin, not just this tab's own messages (see [`listen_stream`]'s docs). This is
//!   what makes a message typed in the terminal (or sent from a different browser tab)
//!   show up here without this tab ever calling `/api/chat` itself.
//! - `POST /api/chat` to submit a message. The response carries nothing about the turn
//!   beyond the id identifying it (see [`PendingTurn`]) — the turn itself is watched via
//!   `/api/stream`, same as everyone else's.
//! - `POST /api/approve/{id}` to answer an [`ChatEvent::ApprovalRequired`] prompt.

use gloo_net::http::Request;
use leptos::{ev::SubmitEvent, prelude::*};
use shared::{
  ApprovalDecision, ChatAccepted, ChatEvent, ChatRequest, HistoryContentItem, ToolStatus,
};
use wasm_bindgen::{JsCast, closure::Closure};
use wasm_bindgen_futures::spawn_local;
use web_sys::{EventSource, MessageEvent};

fn main() {
  // Routes Rust `panic!`s to the browser console with a real stack trace instead of the
  // opaque "unreachable" trap wasm panics produce by default — worth paying for in a dev
  // build; harmless in release since a panic here should not happen in normal use.
  console_error_panic_hook::set_once();
  leptos::mount::mount_to_body(App);
}

/// Who said one [`TimelineItem::Message`] — the only two authors `agent::agent::Event`
/// ever records (see `agent::agent::runtime`'s `"user"`/`"assistant"` literals).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
  User,
  Assistant,
}

/// One entry in the chat transcript as rendered, in the order it should appear.
/// `id` is a purely local, monotonically increasing render key (see
/// [`ChatState::next_id`]) — unrelated to any id the backend assigns; a tool call's
/// [`TimelineItem::ToolCall`]/[`TimelineItem::ToolResult`]/[`TimelineItem::Approval`]
/// additionally carry `tool_id`, the model-assigned tool call id, which is what ties an
/// [`TimelineItem::Approval`] back to the [`ApprovalDecision`] POSTed for it.
#[derive(Clone, Debug)]
enum TimelineItem {
  Message {
    id: u64,
    role: Role,
    text: String,
  },
  ToolCall {
    id: u64,
    tool_id: String,
    name: String,
    arguments: String,
  },
  ToolResult {
    id: u64,
    tool_id: String,
    name: String,
    status: ToolStatus,
    content: String,
  },
  Approval {
    id: u64,
    tool_id: String,
    tool: String,
    arguments: String,
    /// `None` while waiting on a decision; set from either this browser's own
    /// approve/deny button once the server has accepted it (see `render_approval`) or a
    /// [`ChatEvent::ApprovalResolved`] — the latter needed because another browser tab
    /// watching the same turn may have resolved it first.
    resolved: RwSignal<Option<bool>>,
  },
  Error {
    id: u64,
    message: String,
  },
}

impl TimelineItem {
  fn key(&self) -> u64 {
    match self {
      Self::Message { id, .. }
      | Self::ToolCall { id, .. }
      | Self::ToolResult { id, .. }
      | Self::Approval { id, .. }
      | Self::Error { id, .. } => *id,
    }
  }
}

/// This tab's own submission, tracked separately from everything else on the shared
/// stream. Turns from other tabs (and from the terminal) run through the very same
/// `/api/stream` this tab watches, and several can be queued at once behind the process-
/// wide turn lock, so "a turn just finished" is not by itself news about *this* tab's
/// turn — matching [`ChatEvent::Done`]'s `turn` against the id held here is.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
enum PendingTurn {
  /// Nothing submitted from this tab is outstanding; the composer is usable.
  #[default]
  Idle,
  /// `POST /api/chat` has gone out but has not come back yet, so this tab does not yet
  /// know its turn's id. `finished` collects every turn that ends during that window:
  /// one of them could be this tab's own, and the only way to find out is to compare
  /// them against the id once it arrives — assuming none of them was (the composer then
  /// never re-enables) or that one of them was (the bug this whole type exists to avoid)
  /// are both wrong.
  Submitting { finished: Vec<String> },
  /// This tab's turn is under way; only a `Done`/`Error` carrying this id ends it.
  Running(String),
}

/// Everything the chat page needs, grouped so it can be passed around (into
/// [`stream_chat`], into event handlers) as one `Copy` value — every field is itself a
/// signal, so cloning this struct never clones the underlying state, only the handles to
/// it.
#[derive(Clone, Copy)]
struct ChatState {
  timeline: RwSignal<Vec<TimelineItem>>,
  /// Text of the turn currently streaming in, shown as its own "typing" bubble
  /// (`render_streaming_bubble`) separate from [`Self::timeline`] until
  /// [`ChatEvent::Done`] flushes it in as a real [`TimelineItem::Message`] — a
  /// half-finished sentence has no [`TimelineItem::key`] of its own yet, and does not
  /// need one; it is always exactly the most recent thing on screen.
  streaming_text: RwSignal<String>,
  /// This tab's outstanding submission, if any — see [`PendingTurn`]. Also what the
  /// composer's disabled state reads (via [`Self::is_sending`]): "sending" means *this
  /// tab* has a turn in flight, not that the process is busy with someone's.
  pending: RwSignal<PendingTurn>,
  next_id: RwSignal<u64>,
}

impl ChatState {
  fn new() -> Self {
    Self {
      timeline: RwSignal::new(Vec::new()),
      streaming_text: RwSignal::new(String::new()),
      pending: RwSignal::new(PendingTurn::Idle),
      next_id: RwSignal::new(0),
    }
  }

  fn next_id(&self) -> u64 {
    let id = self.next_id.get_untracked();
    self.next_id.set(id + 1);
    id
  }

  fn push(&self, item: TimelineItem) {
    self.timeline.update(|items| items.push(item));
  }

  /// Whether this tab is waiting on a turn of its own (reactive: this is what disables
  /// the composer). `_untracked` is the same question asked from an event handler, where
  /// subscribing to the answer would be meaningless.
  fn is_sending(&self) -> bool {
    self.pending.with(|pending| *pending != PendingTurn::Idle)
  }

  fn is_sending_untracked(&self) -> bool {
    self
      .pending
      .with_untracked(|pending| *pending != PendingTurn::Idle)
  }

  /// `POST /api/chat` came back with `id`. Everything that finished while it was in
  /// flight was recorded rather than judged (see [`PendingTurn::Submitting`]) — so if
  /// this turn is among them, it was over before this tab even learned its name.
  fn turn_submitted(&self, id: String) {
    self.pending.update(|pending| {
      let already_finished = match pending {
        PendingTurn::Submitting { finished } => finished.contains(&id),
        _ => false,
      };
      *pending = if already_finished {
        PendingTurn::Idle
      } else {
        PendingTurn::Running(id)
      };
    });
  }

  /// Some turn on the shared stream ended — this tab's, another tab's, or the
  /// terminal's. Only the first of those frees this tab's composer.
  fn turn_finished(&self, id: &str) {
    self.pending.update(|pending| match pending {
      PendingTurn::Running(pending_id) if pending_id == id => *pending = PendingTurn::Idle,
      PendingTurn::Submitting { finished } => finished.push(id.to_owned()),
      _ => {}
    });
  }

  /// Resolve whichever [`TimelineItem::Approval`] carries `tool_id`, if any is still
  /// waiting — a no-op if it was already resolved (e.g. this browser tab's own button
  /// click already set it, and this is the corresponding [`ChatEvent::ApprovalResolved`]
  /// echoed back).
  fn resolve_approval(&self, tool_id: &str, approved: bool) {
    self.timeline.with(|items| {
      for item in items {
        if let TimelineItem::Approval {
          tool_id: this_id,
          resolved,
          ..
        } = item
          && this_id == tool_id
        {
          resolved.set(Some(approved));
        }
      }
    });
  }
}

#[component]
fn App() -> impl IntoView {
  let state = ChatState::new();
  let input_value = RwSignal::new(String::new());

  // One-shot load, not a reactive `Resource`: the initial transcript never needs to be
  // re-fetched from inside this page (every later change arrives live via `/api/stream`
  // instead — see `listen_stream`), so there is no dependency for a `Resource` to key
  // off of.
  spawn_local(load_history(state));
  // Opened once, kept open for this tab's whole lifetime — not per message (contrast
  // the old per-`POST /api/chat` stream this replaced): see the module docs for why a
  // single persistent connection is what makes cross-origin (terminal <-> browser, tab
  // <-> tab) live sync possible at all.
  listen_stream(state);

  let send = move || {
    let input = input_value.get_untracked();
    if input.trim().is_empty() || state.is_sending_untracked() {
      return;
    }
    input_value.set(String::new());
    state.pending.set(PendingTurn::Submitting {
      finished: Vec::new(),
    });
    // Deliberately not pushed to `state.timeline` here: the server broadcasts a
    // `ChatEvent::UserMessage` for this input the moment it starts the turn (see
    // `web::drive_turn`'s docs), and this tab hears that the same way every other tab
    // does, via `/api/stream` — pushing it here too would double it up locally.
    spawn_local(async move {
      match submit_chat(input).await {
        Ok(turn) => state.turn_submitted(turn),
        Err(err) => {
          state.push(TimelineItem::Error {
            id: state.next_id(),
            message: format!("请求失败：{err}"),
          });
          state.pending.set(PendingTurn::Idle);
        }
      }
    });
  };

  let on_submit = move |ev: SubmitEvent| {
    ev.prevent_default();
    send();
  };

  view! {
    <main class="app">
      <style>{CSS}</style>
      <header>
        <h1>"agent"</h1>
        <p class="subtitle">"本地 web UI · 与终端共享同一份会话"</p>
      </header>
      <div class="timeline">
        <For each=move || state.timeline.get() key=TimelineItem::key children=render_item />
        {move || {
          let text = state.streaming_text.get();
          (!text.is_empty()).then(|| render_streaming_bubble(text))
        }}
      </div>
      <form class="composer" on:submit=on_submit>
        <input
          type="text"
          placeholder="输入消息…"
          prop:value=move || input_value.get()
          prop:disabled=move || state.is_sending()
          on:input=move |ev| input_value.set(event_target_value(&ev))
        />
        <button type="submit" disabled=move || state.is_sending()>
          {move || if state.is_sending() { "运行中…" } else { "发送" }}
        </button>
      </form>
    </main>
  }
}

fn render_item(item: TimelineItem) -> impl IntoView {
  match item {
    TimelineItem::Message { role, text, .. } => {
      let class = if role == Role::User {
        "bubble user"
      } else {
        "bubble assistant"
      };
      view! { <div class=class>{text}</div> }.into_any()
    }
    TimelineItem::ToolCall {
      tool_id,
      name,
      arguments,
      ..
    } => view! {
      <div class="tool-call">
        <span class="tool-badge">"调用"</span>
        <code title=tool_id>{name}</code>
        <pre class="tool-args">{arguments}</pre>
      </div>
    }
    .into_any(),
    TimelineItem::ToolResult {
      tool_id,
      name,
      status,
      content,
      ..
    } => {
      let status_class = match status {
        ToolStatus::Success => "tool-status ok",
        ToolStatus::Error => "tool-status err",
      };
      let status_text = match status {
        ToolStatus::Success => "成功",
        ToolStatus::Error => "失败",
      };
      view! {
        <div class="tool-result">
          <span class=status_class>{status_text}</span>
          <code title=tool_id>{name}</code>
          <pre class="tool-args">{content}</pre>
        </div>
      }
      .into_any()
    }
    TimelineItem::Approval {
      tool_id,
      tool,
      arguments,
      resolved,
      ..
    } => render_approval(tool_id, tool, arguments, resolved).into_any(),
    TimelineItem::Error { message, .. } => view! { <div class="error">{message}</div> }.into_any(),
  }
}

fn render_approval(
  tool_id: String,
  tool: String,
  arguments: String,
  resolved: RwSignal<Option<bool>>,
) -> impl IntoView {
  // A decision is only *this browser's* until the server confirms having handed it to
  // the tool call waiting on it: the agent stays blocked in
  // `DualApprovalCallback::prompt_web` until then, so rendering "已批准" off the click
  // alone would claim something that has not happened — and, if the request failed,
  // never will, leaving the turn hanging with the buttons already gone. Hence: disable
  // the buttons while the request is in flight, and only write `resolved` once it has
  // succeeded, putting the buttons back (with the reason) if it has not.
  let submitting = RwSignal::new(false);
  let failure = RwSignal::new(None::<String>);

  let decide = move |approved: bool| {
    if submitting.get_untracked() {
      return;
    }
    let tool_id = tool_id.clone();
    submitting.set(true);
    failure.set(None);
    spawn_local(async move {
      match submit_approval(&tool_id, approved).await {
        Ok(()) => resolved.set(Some(approved)),
        Err(err) => failure.set(Some(format!("提交决策失败，请重试：{err}"))),
      }
      submitting.set(false);
    });
  };
  let decide_yes = decide.clone();
  let decide_no = decide;

  view! {
    <div class="approval">
      <p>
        "即将执行高危操作 "
        <code>{tool}</code>
      </p>
      <pre class="tool-args">{arguments}</pre>
      {move || match resolved.get() {
        None => {
          // Cloned inside this closure's body, not just captured by the outer `move ||`
          // once: the outer closure is `FnMut` (Leptos re-invokes it on every re-render
          // while `resolved` stays `None`), so `decide_yes`/`decide_no` themselves must
          // stay owned by it across calls — only the fresh clone made on *this*
          // invocation may be moved into the one-shot `on:click` closure below.
          let decide_yes = decide_yes.clone();
          let decide_no = decide_no.clone();
          view! {
            <div class="approval-buttons">
              <button
                class="approve"
                disabled=move || submitting.get()
                on:click=move |_| decide_yes(true)
              >
                "批准"
              </button>
              <button
                class="deny"
                disabled=move || submitting.get()
                on:click=move |_| decide_no(false)
              >
                "拒绝"
              </button>
            </div>
            {move || failure.get().map(|message| view! { <p class="error">{message}</p> })}
          }
            .into_any()
        }
        Some(true) => view! { <p class="approval-decided">"已批准"</p> }.into_any(),
        Some(false) => view! { <p class="approval-decided">"已拒绝"</p> }.into_any(),
      }}
    </div>
  }
}

fn render_streaming_bubble(text: String) -> impl IntoView {
  view! { <div class="bubble assistant streaming">{text}</div> }
}

/// `GET /api/history` once, on mount — see [`App`]'s call site.
async fn load_history(state: ChatState) {
  let response = match Request::get("/api/history").send().await {
    Ok(response) => response,
    Err(err) => {
      leptos::logging::error!("failed to load history: {err}");
      return;
    }
  };
  let entries: Vec<shared::HistoryEntry> = match response.json().await {
    Ok(entries) => entries,
    Err(err) => {
      leptos::logging::error!("failed to parse history: {err}");
      return;
    }
  };
  for entry in entries {
    for item in entry.content {
      let id = state.next_id();
      let timeline_item = match item {
        HistoryContentItem::Message { role, content } => TimelineItem::Message {
          id,
          role: if role == "user" {
            Role::User
          } else {
            Role::Assistant
          },
          text: content,
        },
        HistoryContentItem::ToolCall {
          id: tool_id,
          name,
          arguments,
        } => TimelineItem::ToolCall {
          id,
          tool_id,
          name,
          arguments: arguments.to_string(),
        },
        HistoryContentItem::ToolResult {
          id: tool_id,
          name,
          status,
          content,
        } => TimelineItem::ToolResult {
          id,
          tool_id,
          name,
          status,
          content,
        },
      };
      state.push(timeline_item);
    }
  }
}

/// `POST /api/approve/{id}` — a plain JSON request/response, no streaming involved, so
/// `gloo-net` alone is enough here.
///
/// An error *status* is an `Err` here just as much as a failed request is (see
/// [`error_status`]): a `404`/`410` means this decision reached no one — the prompt was
/// already resolved elsewhere, or its turn is gone — which is precisely what the caller
/// must not render as a decision taken.
async fn submit_approval(tool_id: &str, approved: bool) -> Result<(), String> {
  let response = Request::post(&format!("/api/approve/{tool_id}"))
    .json(&ApprovalDecision { approved })
    .map_err(|err| err.to_string())?
    .send()
    .await
    .map_err(|err| err.to_string())?;
  match error_status(&response) {
    Some(status) => Err(status),
    None => Ok(()),
  }
}

/// `POST /api/chat`: submit `input` as a new turn, returning the id the server assigned
/// it ([`shared::ChatAccepted`]). That id is the only part of the turn this response
/// carries — everything the turn actually produces arrives later via [`listen_stream`],
/// on the same stream as every other front-end's turns, which is exactly why the id is
/// needed (see [`PendingTurn`]).
async fn submit_chat(input: String) -> Result<String, String> {
  let response = Request::post("/api/chat")
    .json(&ChatRequest { input })
    .map_err(|err| err.to_string())?
    .send()
    .await
    .map_err(|err| err.to_string())?;
  if let Some(status) = error_status(&response) {
    return Err(status);
  }
  let accepted: ChatAccepted = response.json().await.map_err(|err| err.to_string())?;
  Ok(accepted.turn)
}

/// The failure message for a non-2xx response, or `None` if it was a success. `gloo-net`
/// resolves a `4xx`/`5xx` as an `Ok(Response)` like any other completed exchange, so a
/// caller that only propagates its `Err`s would treat "the server refused this" as
/// "this worked".
fn error_status(response: &gloo_net::http::Response) -> Option<String> {
  (!response.ok()).then(|| {
    format!(
      "服务端返回 {} {}",
      response.status(),
      response.status_text()
    )
  })
}

/// Opens `GET /api/stream` via the browser's native [`EventSource`] (auto-reconnecting
/// on drop, so a momentary network blip does not need any retry logic here) and applies
/// every [`ChatEvent`] it delivers to `state` for as long as this tab is open. This is
/// the entire mechanism behind "a message typed in the terminal shows up here without
/// this tab sending anything": nothing about this function is specific to messages this
/// tab itself submitted — it just listens.
///
/// The server tags every frame with the SSE event name `chat` (see `to_sse_event` in
/// `src/bin/cli/web.rs`), so this listens on `"chat"` specifically —
/// [`EventSource`]'s default, untagged-frame `message` event would never fire for these.
///
/// `source`/`on_message` are deliberately leaked (via [`Box::leak`]/[`Closure::forget`]):
/// both need to outlive this function — `source` for the whole tab session, the closure
/// for as long as `source` might still invoke it — but neither has a Rust-side owner
/// left to hold onto them once this function returns. This is the same trade-off
/// `Closure::forget` exists to make explicit; a page this small never unmounts anyway,
/// so there is no cleanup this would otherwise be skipping.
fn listen_stream(state: ChatState) {
  let source = match EventSource::new("/api/stream") {
    Ok(source) => source,
    Err(err) => {
      leptos::logging::error!("failed to open /api/stream: {err:?}");
      return;
    }
  };

  let on_message = Closure::<dyn FnMut(MessageEvent)>::new(move |event: MessageEvent| {
    let Some(data) = event.data().as_string() else {
      return;
    };
    if let Ok(chat_event) = serde_json::from_str::<ChatEvent>(&data) {
      apply_chat_event(chat_event, state);
    }
  });
  if let Err(err) =
    source.add_event_listener_with_callback("chat", on_message.as_ref().unchecked_ref())
  {
    leptos::logging::error!("failed to attach /api/stream listener: {err:?}");
    return;
  }
  on_message.forget();
  Box::leak(Box::new(source));
}

fn apply_chat_event(event: ChatEvent, state: ChatState) {
  match event {
    ChatEvent::UserMessage { text, .. } => {
      // `origin` is ignored here: unlike the terminal's own renderer (which skips
      // re-printing a `Terminal`-origin message because `reedline` already echoed it),
      // a browser tab never saw this input any other way — every `UserMessage`, from
      // any origin, is new information to this tab and rendered the same way.
      state.push(TimelineItem::Message {
        id: state.next_id(),
        role: Role::User,
        text,
      });
    }
    ChatEvent::Token { text } => {
      state
        .streaming_text
        .update(|current| current.push_str(&text));
    }
    ChatEvent::ToolCallsStarted { calls } => {
      for call in calls {
        state.push(TimelineItem::ToolCall {
          id: state.next_id(),
          tool_id: call.id,
          name: call.name,
          arguments: call.arguments.to_string(),
        });
      }
    }
    ChatEvent::ToolCallsFinished { results } => {
      for result in results {
        state.push(TimelineItem::ToolResult {
          id: state.next_id(),
          tool_id: result.id,
          name: result.name,
          status: result.status,
          content: result.content,
        });
      }
    }
    ChatEvent::ApprovalRequired {
      id,
      tool,
      arguments,
    } => {
      state.push(TimelineItem::Approval {
        id: state.next_id(),
        tool_id: id,
        tool,
        arguments,
        resolved: RwSignal::new(None),
      });
    }
    ChatEvent::ApprovalResolved { id, approved } => {
      state.resolve_approval(&id, approved);
    }
    ChatEvent::Done {
      turn,
      budget_exhausted,
    } => {
      // Flushed regardless of whose turn this was: only one turn runs at a time across
      // the whole process (see `turn_lock`'s docs in `src/bin/cli/main.rs`), so the text
      // streaming in right now belongs to whichever turn is ending here.
      let text = state.streaming_text.get_untracked();
      if !text.is_empty() {
        state.push(TimelineItem::Message {
          id: state.next_id(),
          role: Role::Assistant,
          text,
        });
      }
      state.streaming_text.set(String::new());
      if budget_exhausted {
        state.push(TimelineItem::Error {
          id: state.next_id(),
          message: "工具调用轮次预算已用尽，回答可能基于部分结果。".to_owned(),
        });
      }
      // The composer, on the other hand, is this tab's alone: turns queue up behind each
      // other, so the one ending here may be someone else's while this tab's own is
      // still waiting its turn to run.
      state.turn_finished(&turn);
    }
    ChatEvent::Error { turn, message } => {
      state.push(TimelineItem::Error {
        id: state.next_id(),
        message,
      });
      if let Some(turn) = turn {
        state.turn_finished(&turn);
      }
    }
  }
}

const CSS: &str = r#"
  :root { color-scheme: light dark; }
  body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
  .app { max-width: 720px; margin: 0 auto; padding: 1rem; display: flex; flex-direction: column; height: 100vh; box-sizing: border-box; }
  header h1 { margin: 0; font-size: 1.25rem; }
  header .subtitle { margin: 0.15rem 0 1rem; font-size: 0.8rem; opacity: 0.6; }
  .timeline { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 0.5rem; padding-bottom: 1rem; }
  .bubble { padding: 0.5rem 0.75rem; border-radius: 0.75rem; max-width: 80%; white-space: pre-wrap; word-break: break-word; }
  .bubble.user { align-self: flex-end; background: #2563eb; color: white; }
  .bubble.assistant { align-self: flex-start; background: rgba(127, 127, 127, 0.15); }
  .bubble.streaming { opacity: 0.75; }
  .tool-call, .tool-result { align-self: flex-start; font-size: 0.85rem; border-left: 3px solid #94a3b8; padding-left: 0.5rem; opacity: 0.85; }
  .tool-badge { font-weight: bold; margin-right: 0.35rem; }
  .tool-status { font-weight: bold; margin-right: 0.35rem; }
  .tool-status.ok { color: #16a34a; }
  .tool-status.err { color: #dc2626; }
  .tool-args { margin: 0.2rem 0 0; font-size: 0.8rem; white-space: pre-wrap; word-break: break-word; opacity: 0.8; }
  .approval { align-self: stretch; border: 1px solid #f59e0b; border-radius: 0.5rem; padding: 0.5rem 0.75rem; background: rgba(245, 158, 11, 0.08); }
  .approval-buttons { display: flex; gap: 0.5rem; margin-top: 0.4rem; }
  .approval-buttons .approve { background: #16a34a; color: white; }
  .approval-buttons .deny { background: #dc2626; color: white; }
  .approval-decided { font-weight: bold; margin: 0.4rem 0 0; }
  .error { align-self: stretch; color: #dc2626; font-size: 0.85rem; }
  .composer { display: flex; gap: 0.5rem; padding-top: 0.5rem; border-top: 1px solid rgba(127, 127, 127, 0.25); }
  .composer input { flex: 1; padding: 0.5rem 0.75rem; border-radius: 0.5rem; border: 1px solid rgba(127, 127, 127, 0.35); }
  .composer button { padding: 0.5rem 1rem; border-radius: 0.5rem; border: none; background: #2563eb; color: white; cursor: pointer; }
  .composer button:disabled { opacity: 0.5; cursor: default; }
  button { cursor: pointer; }
"#;
