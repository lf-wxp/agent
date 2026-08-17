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

mod i18n;
mod markdown;

use gloo_net::http::Request;
use i18n::{Key, Lang, t};
use leptos::{
  ev::{KeyboardEvent, SubmitEvent},
  html,
  prelude::*,
};
use markdown::render_markdown;
use shared::{
  ApprovalDecision, ChatAccepted, ChatEvent, ChatRequest, HistoryContentItem, ToolStatus,
};
use wasm_bindgen::{JsCast, closure::Closure};
use wasm_bindgen_futures::spawn_local;
use web_sys::{Event as DomEvent, EventSource, HtmlTextAreaElement, MessageEvent};

fn main() {
  // Routes Rust `panic!`s to the browser console with a real stack trace instead of the
  // opaque "unreachable" trap wasm panics produce by default — worth paying for in a dev
  // build; harmless in release since a panic here should not happen in normal use.
  console_error_panic_hook::set_once();
  leptos::mount::mount_to_body(App);
}

/// A tool call/result body long enough that it starts collapsed by default (see
/// [`TimelineItem::ToolCall`]/[`TimelineItem::ToolResult`]'s `expanded`) — short ones
/// (a `calculator` call, say) are more useful shown immediately than behind a tap, but a
/// multi-kilobyte `web_search` result is not, especially on a phone screen.
const AUTO_EXPAND_CHAR_LIMIT: usize = 220;

/// Who said one [`TimelineItem::Message`] — the only two authors `agent::agent::Event`
/// ever records (see `agent::agent::runtime`'s `"user"`/`"assistant"` literals).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
  User,
  Assistant,
}

/// `GET /api/stream`'s connection state, reflected in the header's status dot — purely
/// cosmetic (nothing here gates sending a message; a `POST /api/chat` that goes out
/// while this reads anything but `Open` still reaches the server just fine, since it is
/// an independent HTTP request), but a live "is this tab actually hearing about other
/// front-ends' turns right now" indicator is worth having given how much of this page's
/// whole point depends on that connection staying up.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
enum ConnectionState {
  #[default]
  Connecting,
  Open,
  /// [`EventSource`] is retrying on its own (it always does, see [`listen_stream`]'s
  /// docs) — this is not a terminal state, just what it looks like in between attempts.
  Retrying,
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
    /// Whether the argument body is shown or collapsed behind a toggle — see
    /// [`AUTO_EXPAND_CHAR_LIMIT`] for the default this starts at.
    expanded: RwSignal<bool>,
  },
  ToolResult {
    id: u64,
    tool_id: String,
    name: String,
    status: ToolStatus,
    content: String,
    expanded: RwSignal<bool>,
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
  /// Whether *some* turn — this tab's own, another tab's, or the terminal's — is
  /// currently running, purely to drive the "thinking" indicator shown before the first
  /// token of a reply arrives (see [`render_thinking_indicator`]). Sound because only
  /// one turn runs at a time across the whole process (see `turn_lock`'s docs in
  /// `src/bin/cli/main.rs`): a `UserMessage` always means *the* turn just started, a
  /// `Done`/`Error` always means *the* turn just ended, with no other turn interleaved
  /// in between to confuse this flag about.
  turn_active: RwSignal<bool>,
  /// This tab's outstanding submission, if any — see [`PendingTurn`]. Also what the
  /// composer's disabled state reads (via [`Self::is_sending`]): "sending" means *this
  /// tab* has a turn in flight, not that the process is busy with someone's.
  pending: RwSignal<PendingTurn>,
  next_id: RwSignal<u64>,
  /// The language a one-off, free-form notice (a request failure, the
  /// [`Key::BudgetExhausted`] warning, ...) should be formatted in at the moment it is
  /// pushed into [`Self::timeline`] — see `i18n`'s module docs for why that text is
  /// frozen at push time rather than kept reactive to a later language switch the way
  /// this page's own chrome (badges, buttons, placeholders) is.
  lang: RwSignal<Lang>,
}

impl ChatState {
  fn new(lang: RwSignal<Lang>) -> Self {
    Self {
      timeline: RwSignal::new(Vec::new()),
      streaming_text: RwSignal::new(String::new()),
      turn_active: RwSignal::new(false),
      pending: RwSignal::new(PendingTurn::Idle),
      next_id: RwSignal::new(0),
      lang,
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
  // Created here — the top of the component tree — and shared with every rendering
  // function below via `provide_context`/`i18n::current_lang` rather than as a
  // parameter threaded through each one's signature; see `i18n`'s module docs for the
  // reasoning and for why every translated string must be read from inside a
  // `move || ...` closure rather than as a plain value.
  let lang = RwSignal::new(Lang::detect());
  provide_context(lang);

  let state = ChatState::new(lang);
  let input_value = RwSignal::new(String::new());
  let connection = RwSignal::new(ConnectionState::default());
  let textarea_ref = NodeRef::<html::Textarea>::new();
  let timeline_ref = NodeRef::<html::Div>::new();

  // Keeps `<html lang>` in sync with the active language — screen readers and the
  // browser's own "translate this page?" heuristics both read that attribute, and
  // there is no reason for either to still see whatever `index.html` shipped with as
  // its static default once this has actually detected/switched to something else.
  Effect::new(move |_| {
    if let Some(document_element) = web_sys::window()
      .and_then(|w| w.document())
      .and_then(|d| d.document_element())
    {
      let _ = document_element.set_attribute("lang", lang.get().code());
    }
  });

  // One-shot load, not a reactive `Resource`: the initial transcript never needs to be
  // re-fetched from inside this page (every later change arrives live via `/api/stream`
  // instead — see `listen_stream`), so there is no dependency for a `Resource` to key
  // off of.
  spawn_local(load_history(state));
  // Opened once, kept open for this tab's whole lifetime — not per message (contrast
  // the old per-`POST /api/chat` stream this replaced): see the module docs for why a
  // single persistent connection is what makes cross-origin (terminal <-> browser, tab
  // <-> tab) live sync possible at all.
  listen_stream(state, connection);

  // Keeps the timeline scrolled to its newest content as it grows — a chat page whose
  // view silently stays pinned to a message from five turns ago the moment a new one
  // arrives is not a usable one. Reads both signals through `.with(|_| ())` rather than
  // `.get()` purely to avoid cloning the whole transcript (or the in-flight text) just
  // to throw the clone away — the value itself is never needed here, only "did either
  // of these change".
  Effect::new(move |_| {
    state.timeline.with(|_| ());
    state.streaming_text.with(|_| ());
    if let Some(el) = timeline_ref.get() {
      el.set_scroll_top(el.scroll_height());
    }
  });

  let reset_composer = move || {
    input_value.set(String::new());
    if let Some(el) = textarea_ref.get_untracked() {
      set_textarea_height(&el, "auto");
    }
  };

  let send = move || {
    let input = input_value.get_untracked();
    if input.trim().is_empty() || state.is_sending_untracked() {
      return;
    }
    reset_composer();
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
            message: i18n::request_failed(state.lang.get_untracked(), &err),
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

  // Enter sends, Shift+Enter inserts a newline — the usual chat-app convention — except
  // while an IME composition is in progress (`is_composing`): the Enter that confirms a
  // candidate in, say, an active Pinyin/Kana input session must never also submit the
  // message, or every such message would go out one keystroke before the user meant it
  // to.
  let on_keydown = move |ev: KeyboardEvent| {
    if ev.key() == "Enter" && !ev.shift_key() && !ev.is_composing() {
      ev.prevent_default();
      send();
    }
  };

  // Auto-grows the textarea with its content, up to the `max-height` set in CSS (beyond
  // that, the textarea itself scrolls). See [`set_textarea_height`]'s docs for why
  // `scroll_height` is measured after collapsing the height first.
  let on_input = move |ev| {
    input_value.set(event_target_value(&ev));
    if let Some(el) = textarea_ref.get_untracked() {
      set_textarea_height(&el, "auto");
      let scroll_height = el.scroll_height();
      set_textarea_height(&el, &format!("{scroll_height}px"));
    }
  };

  let timeline_is_empty = move || {
    state.timeline.with(|items| items.is_empty()) && state.streaming_text.with(String::is_empty)
  };
  let show_thinking =
    move || state.turn_active.get() && state.streaming_text.with(String::is_empty);

  view! {
    <div class="app-shell">
      <style>{CSS}</style>
      <div class="ambient-glow" aria-hidden="true"></div>
      <header class="app-header">
        <div class="brand">
          <span class="brand-mark">
            {move || t(lang.get(), Key::RoleAgent)}
            <span class="cursor" aria-hidden="true"></span>
          </span>
          <span class="brand-tag">{move || t(lang.get(), Key::BrandTag)}</span>
        </div>
        <div class="header-right">
          <LangSwitch lang=lang />
          <div class="connection" data-state=move || connection_data_attr(connection.get())>
            <span class="connection-dot"></span>
            <span class="connection-label">
              {move || connection_label(connection.get(), lang.get())}
            </span>
          </div>
        </div>
      </header>

      <div class="timeline" node_ref=timeline_ref>
        <Show when=timeline_is_empty>
          <div class="empty-state">
            <p class="empty-state-glyph">"[ ]"</p>
            <p>{move || t(lang.get(), Key::EmptyTitle)}</p>
            <p class="empty-state-hint">{move || t(lang.get(), Key::EmptyHint)}</p>
          </div>
        </Show>
        <For each=move || state.timeline.get() key=TimelineItem::key children=render_item />
        <Show when=show_thinking>{render_thinking_indicator}</Show>
        {move || {
          let text = state.streaming_text.get();
          (!text.is_empty()).then(|| render_streaming_bubble(text))
        }}
      </div>

      <form class="composer" on:submit=on_submit>
        <div class="composer-inner">
          <textarea
            class="composer-input"
            node_ref=textarea_ref
            rows="1"
            placeholder=move || t(lang.get(), Key::ComposerPlaceholder)
            prop:value=move || input_value.get()
            prop:disabled=move || state.is_sending()
            on:input=on_input
            on:keydown=on_keydown
          ></textarea>
          <button
            type="submit"
            class="send-btn"
            disabled=move || state.is_sending() || input_value.with(|v| v.trim().is_empty())
            aria-label=move || t(lang.get(), Key::SendAria)
          >
            {move || {
              if state.is_sending() {
                view! { <span class="spinner" aria-hidden="true"></span> }.into_any()
              } else {
                view! { <SendIcon /> }.into_any()
              }
            }}
          </button>
        </div>
        <p class="composer-hint">{move || t(lang.get(), Key::ComposerHint)}</p>
      </form>
    </div>
  }
}

/// The header's language picker: one small button per [`i18n::ALL_LANGS`], the active
/// one highlighted. Each button's own label is [`Lang::short_label`] (kept to two or
/// three characters so all three fit next to the connection indicator on a narrow
/// phone screen) with [`Lang::native_name`] as its `title`/`aria-label` for the full
/// name — see those methods' docs for why neither is translated into whichever language
/// is currently active.
#[component]
fn LangSwitch(lang: RwSignal<Lang>) -> impl IntoView {
  view! {
    <div class="lang-switch" role="group" aria-label="Language / 语言 / Idioma">
      {i18n::ALL_LANGS
        .map(|candidate| {
          view! {
            <button
              type="button"
              class=move || {
                if lang.get() == candidate {
                  "lang-btn active"
                } else {
                  "lang-btn"
                }
              }
              title=candidate.native_name()
              aria-label=candidate.native_name()
              aria-pressed=move || lang.get() == candidate
              on:click=move |_| {
                lang.set(candidate);
                candidate.store();
              }
            >
              {candidate.short_label()}
            </button>
          }
        })
        .collect_view()}
    </div>
  }
}

/// Sets the composer textarea's inline `height` style to `value` (either `"auto"`, to
/// collapse it back down before re-measuring, or a `"<n>px"` string). Goes through an
/// explicit `&web_sys::HtmlTextAreaElement` binding rather than calling `.style()`
/// straight off the `NodeRef`'s `HtmlElement<html::Textarea>` wrapper: that wrapper also
/// has its own `.style()` (Leptos's reactive style-attribute helper, for `style:` view
/// attributes), which shadows the raw DOM `CSSStyleDeclaration` getter this needs —
/// disambiguating with a type annotation is what picks the latter.
fn set_textarea_height(el: &HtmlTextAreaElement, value: &str) {
  // `HtmlElement::style` (path-qualified, not `el.style()`): dot-call method
  // resolution finds `tachys`'s `ElementExt::style` (Leptos's reactive style-attribute
  // helper, blanket-implemented broadly enough to match `HtmlTextAreaElement` before
  // autoderef ever reaches `web_sys::HtmlElement`'s own inherent `style`) first and
  // shadows the one this needs. Naming the type explicitly bypasses that: inherent
  // methods always win over trait methods for a type-qualified call.
  let _ = web_sys::HtmlElement::style(el).set_property("height", value);
}

#[component]
fn SendIcon() -> impl IntoView {
  view! {
    <svg
      class="send-icon"
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <path
        d="M4 12L20 4L14 20L11 13L4 12Z"
        stroke="currentColor"
        stroke-width="1.6"
        stroke-linejoin="round"
        stroke-linecap="round"
      ></path>
    </svg>
  }
}

fn connection_data_attr(state: ConnectionState) -> &'static str {
  match state {
    ConnectionState::Connecting => "connecting",
    ConnectionState::Open => "open",
    ConnectionState::Retrying => "retrying",
  }
}

fn connection_label(state: ConnectionState, lang: Lang) -> &'static str {
  let key = match state {
    ConnectionState::Connecting => Key::ConnConnecting,
    ConnectionState::Open => Key::ConnOpen,
    ConnectionState::Retrying => Key::ConnRetrying,
  };
  t(lang, key)
}

fn render_item(item: TimelineItem) -> impl IntoView {
  let lang = i18n::current_lang();
  match item {
    TimelineItem::Message { role, text, .. } => {
      let (row_class, bubble_class, role_key) = if role == Role::User {
        ("message-row from-user", "bubble user", Key::RoleYou)
      } else {
        (
          "message-row from-assistant",
          "bubble assistant",
          Key::RoleAgent,
        )
      };
      view! {
        <div class=row_class>
          <span class="role-label">{move || t(lang.get(), role_key)}</span>
          <div class=bubble_class>{render_markdown(&text)}</div>
        </div>
      }
      .into_any()
    }
    TimelineItem::ToolCall {
      tool_id,
      name,
      arguments,
      expanded,
      ..
    } => render_tool_card(tool_id, name, arguments, ToolCardKind::Call, expanded).into_any(),
    TimelineItem::ToolResult {
      tool_id,
      name,
      status,
      content,
      expanded,
      ..
    } => render_tool_card(
      tool_id,
      name,
      content,
      ToolCardKind::Result(status),
      expanded,
    )
    .into_any(),
    TimelineItem::Approval {
      tool_id,
      tool,
      arguments,
      resolved,
      ..
    } => render_approval(tool_id, tool, arguments, resolved).into_any(),
    TimelineItem::Error { message, .. } => view! {
      <div class="notice error">
        <span class="notice-icon">"!"</span>
        <span>{message}</span>
      </div>
    }
    .into_any(),
  }
}

/// What [`render_tool_card`] is rendering — a call about to run, or a finished result
/// (carrying its outcome). Sharing one renderer between the two keeps the "collapsible
/// body behind a header row" layout identical for both instead of two near-duplicate
/// implementations drifting apart over time.
enum ToolCardKind {
  Call,
  Result(ToolStatus),
}

fn render_tool_card(
  tool_id: String,
  name: String,
  body: String,
  kind: ToolCardKind,
  expanded: RwSignal<bool>,
) -> impl IntoView {
  let lang = i18n::current_lang();
  let (card_class, badge_class, badge_key) = match kind {
    ToolCardKind::Call => ("tool-card call", "tool-badge call", Key::ToolCallBadge),
    ToolCardKind::Result(ToolStatus::Success) => {
      ("tool-card result ok", "tool-badge ok", Key::ToolDoneBadge)
    }
    ToolCardKind::Result(ToolStatus::Error) => (
      "tool-card result err",
      "tool-badge err",
      Key::ToolFailedBadge,
    ),
  };
  let has_body = !body.trim().is_empty();
  let toggle = move |_| expanded.update(|value| *value = !*value);

  view! {
    <div class=card_class>
      <button
        type="button"
        class="tool-card-header"
        title=tool_id
        disabled=!has_body
        on:click=toggle
      >
        <span class=badge_class>{move || t(lang.get(), badge_key)}</span>
        <code class="tool-name">{name}</code>
        <Show when=move || has_body>
          <span class="tool-toggle">
            {move || {
              t(lang.get(), if expanded.get() { Key::ToolCollapse } else { Key::ToolExpand })
            }}
          </span>
        </Show>
      </button>
      <Show when=move || has_body && expanded.get()>
        <pre class="tool-body">{body.clone()}</pre>
      </Show>
    </div>
  }
}

fn render_approval(
  tool_id: String,
  tool: String,
  arguments: String,
  resolved: RwSignal<Option<bool>>,
) -> impl IntoView {
  let lang = i18n::current_lang();
  // A decision is only *this browser's* until the server confirms having handed it to
  // the tool call waiting on it: the agent stays blocked in
  // `DualApprovalCallback::prompt_web` until then, so rendering "approved" off the
  // click alone would claim something that has not happened — and, if the request
  // failed, never will, leaving the turn hanging with the buttons already gone. Hence:
  // disable the buttons while the request is in flight, and only write `resolved` once
  // it has succeeded, putting the buttons back (with the reason) if it has not.
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
        Err(err) => failure.set(Some(i18n::approval_submit_failed(
          lang.get_untracked(),
          &err,
        ))),
      }
      submitting.set(false);
    });
  };
  let decide_yes = decide.clone();
  let decide_no = decide;

  view! {
    <div class="approval">
      <div class="approval-head">
        <span class="approval-icon">"!"</span>
        <p>
          {move || t(lang.get(), Key::ApprovalPrompt)}
          " "
          <code>{tool}</code>
        </p>
      </div>
      <pre class="tool-body approval-args">{arguments}</pre>
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
                type="button"
                class="approve"
                disabled=move || submitting.get()
                on:click=move |_| decide_yes(true)
              >
                {move || t(lang.get(), Key::ApprovalApprove)}
              </button>
              <button
                type="button"
                class="deny"
                disabled=move || submitting.get()
                on:click=move |_| decide_no(false)
              >
                {move || t(lang.get(), Key::ApprovalDeny)}
              </button>
            </div>
            {move || {
              failure.get().map(|message| view! { <p class="notice error compact">{message}</p> })
            }}
          }
            .into_any()
        }
        Some(true) => view! {
          <p class="approval-decided approve">{move || t(lang.get(), Key::ApprovalApproved)}</p>
        }
          .into_any(),
        Some(false) => view! {
          <p class="approval-decided deny">{move || t(lang.get(), Key::ApprovalDenied)}</p>
        }
          .into_any(),
      }}
    </div>
  }
}

fn render_streaming_bubble(text: String) -> impl IntoView {
  let lang = i18n::current_lang();
  // Re-parsed as markdown on every token (this whole function re-runs whenever
  // `streaming_text` changes) rather than incrementally patched: a partial code fence
  // or unterminated `**` mid-stream renders a little oddly for a moment, but
  // `pulldown_cmark` never panics on unterminated constructs, and the flicker resolves
  // itself the instant the closing token arrives — an incremental parser would be a lot
  // more code to render the exact same steady state slightly more smoothly along the
  // way.
  view! {
    <div class="message-row from-assistant">
      <span class="role-label">{move || t(lang.get(), Key::RoleAgent)}</span>
      <div class="bubble assistant streaming">
        {render_markdown(&text)}
        <span class="type-cursor" aria-hidden="true"></span>
      </div>
    </div>
  }
}

fn render_thinking_indicator() -> impl IntoView {
  let lang = i18n::current_lang();
  view! {
    <div class="message-row from-assistant">
      <span class="role-label">{move || t(lang.get(), Key::RoleAgent)}</span>
      <div class="bubble assistant thinking" aria-label=move || t(lang.get(), Key::ThinkingAria)>
        <span class="thinking-dot"></span>
        <span class="thinking-dot"></span>
        <span class="thinking-dot"></span>
      </div>
    </div>
  }
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
        } => {
          let arguments = arguments.to_string();
          let expanded = RwSignal::new(arguments.len() <= AUTO_EXPAND_CHAR_LIMIT);
          TimelineItem::ToolCall {
            id,
            tool_id,
            name,
            arguments,
            expanded,
          }
        }
        HistoryContentItem::ToolResult {
          id: tool_id,
          name,
          status,
          content,
        } => {
          let expanded = RwSignal::new(content.len() <= AUTO_EXPAND_CHAR_LIMIT);
          TimelineItem::ToolResult {
            id,
            tool_id,
            name,
            status,
            content,
            expanded,
          }
        }
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
  // `Lang::En` here is fine even though this fails independently of it: `error_status`'s
  // message only reaches the user through [`i18n::approval_submit_failed`], which
  // re-wraps it in the *caller's* current language — this inner status text is a
  // sub-detail embedded inside that outer message, not directly user-facing English
  // text left untranslated, so it does not need `render_approval`'s own current
  // language threaded all the way down here just to immediately get wrapped again.
  let response = Request::post(&format!("/api/approve/{tool_id}"))
    .json(&ApprovalDecision { approved })
    .map_err(|err| err.to_string())?
    .send()
    .await
    .map_err(|err| err.to_string())?;
  match error_status(&response, Lang::En) {
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
  // See [`submit_approval`]'s docs for why `Lang::En` here does not skip translating
  // anything user-facing — the caller ([`App`]'s `send`) re-wraps this in
  // [`i18n::request_failed`] using its own current language before it ever reaches the
  // timeline.
  let response = Request::post("/api/chat")
    .json(&ChatRequest { input })
    .map_err(|err| err.to_string())?
    .send()
    .await
    .map_err(|err| err.to_string())?;
  if let Some(status) = error_status(&response, Lang::En) {
    return Err(status);
  }
  let accepted: ChatAccepted = response.json().await.map_err(|err| err.to_string())?;
  Ok(accepted.turn)
}

/// The failure message for a non-2xx response, or `None` if it was a success. `gloo-net`
/// resolves a `4xx`/`5xx` as an `Ok(Response)` like any other completed exchange, so a
/// caller that only propagates its `Err`s would treat "the server refused this" as
/// "this worked".
fn error_status(response: &gloo_net::http::Response, lang: Lang) -> Option<String> {
  (!response.ok())
    .then(|| i18n::server_error_status(lang, response.status(), &response.status_text()))
}

/// Opens `GET /api/stream` via the browser's native [`EventSource`] (auto-reconnecting
/// on drop, so a momentary network blip does not need any retry logic here) and applies
/// every [`ChatEvent`] it delivers to `state` for as long as this tab is open. This is
/// the entire mechanism behind "a message typed in the terminal shows up here without
/// this tab sending anything": nothing about this function is specific to messages this
/// tab itself submitted — it just listens. `connection` is updated from `onopen`/
/// `onerror` purely for the header's status dot (see [`ConnectionState`]'s docs) — it
/// does not otherwise affect anything here.
///
/// The server tags every frame with the SSE event name `chat` (see `to_sse_event` in
/// `src/bin/cli/web.rs`), so this listens on `"chat"` specifically —
/// [`EventSource`]'s default, untagged-frame `message` event would never fire for these.
///
/// `source`/the closures below are deliberately leaked (via [`Box::leak`]/
/// [`Closure::forget`]): all three need to outlive this function — `source` for the
/// whole tab session, the closures for as long as `source` might still invoke them —
/// but none has a Rust-side owner left to hold onto them once this function returns.
/// This is the same trade-off `Closure::forget` exists to make explicit; a page this
/// small never unmounts anyway, so there is no cleanup this would otherwise be skipping.
fn listen_stream(state: ChatState, connection: RwSignal<ConnectionState>) {
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
  let on_open = Closure::<dyn FnMut(DomEvent)>::new(move |_: DomEvent| {
    connection.set(ConnectionState::Open);
  });
  let on_error = Closure::<dyn FnMut(DomEvent)>::new(move |_: DomEvent| {
    connection.set(ConnectionState::Retrying);
  });

  let listeners = [
    ("chat", on_message.as_ref().unchecked_ref()),
    ("open", on_open.as_ref().unchecked_ref()),
    ("error", on_error.as_ref().unchecked_ref()),
  ];
  for (event_name, callback) in listeners {
    if let Err(err) = source.add_event_listener_with_callback(event_name, callback) {
      leptos::logging::error!("failed to attach /api/stream `{event_name}` listener: {err:?}");
    }
  }
  on_message.forget();
  on_open.forget();
  on_error.forget();
  Box::leak(Box::new(source));
}

fn apply_chat_event(event: ChatEvent, state: ChatState) {
  match event {
    ChatEvent::UserMessage { text, .. } => {
      // `origin` is ignored here: unlike the terminal's own renderer (which skips
      // re-printing a `Terminal`-origin message because `reedline` already echoed it),
      // a browser tab never saw this input any other way — every `UserMessage`, from
      // any origin, is new information to this tab and rendered the same way.
      state.turn_active.set(true);
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
        let arguments = call.arguments.to_string();
        let expanded = RwSignal::new(arguments.len() <= AUTO_EXPAND_CHAR_LIMIT);
        state.push(TimelineItem::ToolCall {
          id: state.next_id(),
          tool_id: call.id,
          name: call.name,
          arguments,
          expanded,
        });
      }
    }
    ChatEvent::ToolCallsFinished { results } => {
      for result in results {
        let expanded = RwSignal::new(result.content.len() <= AUTO_EXPAND_CHAR_LIMIT);
        state.push(TimelineItem::ToolResult {
          id: state.next_id(),
          tool_id: result.id,
          name: result.name,
          status: result.status,
          content: result.content,
          expanded,
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
      state.turn_active.set(false);
      if budget_exhausted {
        state.push(TimelineItem::Error {
          id: state.next_id(),
          message: t(state.lang.get_untracked(), Key::BudgetExhausted).to_owned(),
        });
      }
      // The composer, on the other hand, is this tab's alone: turns queue up behind each
      // other, so the one ending here may be someone else's while this tab's own is
      // still waiting its turn to run.
      state.turn_finished(&turn);
    }
    ChatEvent::Error { turn, message } => {
      state.turn_active.set(false);
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

const CSS: &str = include_str!("style.css");
