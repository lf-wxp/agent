//! [`DualApprovalCallback`]: the same "ask a human before running a dangerous tool" idea
//! as [`crate::callback::approval::ApprovalCallback`], but able to ask that human through
//! whichever front-ends happen to be attached to the session — a terminal, one or more
//! browser tabs, or both at once.
//!
//! # Approval belongs to the session, not to one front-end
//!
//! A terminal and a browser pointed at the same session are two *views* of one
//! conversation, not two deployments. So a prompt raised by a turn typed in the terminal
//! has to be answerable from the browser, and vice versa: whoever is looking gets to
//! decide. [`ApprovalChannel::Session`] is that shape — the prompt is published once, to
//! everyone, and the first decision to come back wins ([`ApprovalRegistry`] enforces the
//! "first" part).
//!
//! Routing a prompt to only the front-end that happened to start the turn would strand it
//! whenever that view walked away, and since a turn holds the session's turn lock for its
//! whole duration, a stranded prompt does not merely stall itself — it wedges every other
//! view of that session too.
//!
//! # Nothing waits forever
//!
//! Every wait is bounded (see [`DualApprovalCallback::with_timeout`]) and every way of
//! failing to get an answer — timeout, no front-end listening, a view that received the
//! prompt and vanished — denies. Fail-closed is the only safe default for a callback whose
//! entire job is gating destructive operations, and a bound is what makes "nobody
//! answered" recoverable instead of terminal.
//!
//! [`with_approval_channel`] attaches a channel to one turn: a [`tokio::task_local!`]
//! carries it from wherever the turn starts down to whichever [`BeforeToolCallback::call`]
//! it triggers, however many tool calls that turn ends up making.

use std::{
  collections::HashMap,
  future::Future,
  io::{self, Write},
  sync::Mutex,
  time::Duration,
};

use tokio::sync::{Mutex as AsyncMutex, mpsc, oneshot};

use crate::{
  agent::{
    ExecutionContext, ToolResultStatus,
    callback::{BeforeToolCallback, ToolCallView},
  },
  config,
};

tokio::task_local! {
  /// Which [`ApprovalChannel`] the turn currently executing should use for any
  /// [`DualApprovalCallback`] prompts it triggers. Not set outside of
  /// [`with_approval_channel`]; [`DualApprovalCallback::call`] falls back to
  /// [`ApprovalChannel::Terminal`] when that is the case, so a caller that never heard of
  /// the web front-end (an existing test, an example, a library consumer with no web
  /// mode) keeps behaving exactly like the plain [`crate::callback::approval::
  /// ApprovalCallback`].
  static APPROVAL_CHANNEL: ApprovalChannel;
}

/// One tool call waiting on a human decision.
///
/// Published by [`DualApprovalCallback`] to whichever [`ApprovalChannel::Session`] is
/// active for the current turn. The receiving end — the CLI's terminal turn loop, its
/// `POST /api/approve/{id}` route, or both — is responsible for showing it to whoever is
/// watching and resolving `decision` once someone decides. `id` is the tool call's own id
/// (the same one the model assigned it, and the same one a
/// [`crate::agent::AgentStreamEvent::ToolCallsStarted`] the browser already received
/// carries), so no separate id scheme is needed just for approvals.
pub struct PendingApproval {
  pub id: String,
  pub tool: String,
  pub raw_arguments: String,
  pub decision: oneshot::Sender<bool>,
}

/// The session's in-flight approvals, keyed by tool call id.
///
/// Shared by every front-end attached to a session, which is what lets any of them answer
/// any prompt. [`Self::resolve`] removes the entry it answers, so the first decision wins
/// and a second one is a no-op rather than a panic on an already-consumed sender.
///
/// A plain [`std::sync::Mutex`] is enough: every critical section is a single
/// non-blocking `HashMap` operation, never held across an `.await`.
#[derive(Default)]
pub struct ApprovalRegistry {
  pending: Mutex<HashMap<String, oneshot::Sender<bool>>>,
}

impl ApprovalRegistry {
  pub fn new() -> Self {
    Self::default()
  }

  /// Take custody of a prompt's decision channel so any front-end can resolve it later.
  pub fn register(&self, id: String, decision: oneshot::Sender<bool>) {
    self.lock().insert(id, decision);
  }

  /// Answer a pending prompt. `true` if this call is the one that decided it; `false` if
  /// it was already resolved, timed out, or never existed.
  pub fn resolve(&self, id: &str, approved: bool) -> bool {
    let Some(decision) = self.lock().remove(id) else {
      return false;
    };
    // `send` fails only when the waiting side already gave up (it timed out, or its turn
    // ended) — the decision arrived too late to matter either way.
    decision.send(approved).is_ok()
  }

  /// Drop entries without answering them, for a turn that has ended. The waiting side has
  /// already stopped waiting, so this only reclaims the map slots — without it, every
  /// abandoned prompt would linger for the life of the process.
  pub fn discard(&self, ids: &[String]) {
    let mut pending = self.lock();
    for id in ids {
      pending.remove(id);
    }
  }

  /// Whether `id` is still awaiting a decision. Lets a front-end skip prompting for
  /// something another view already answered.
  pub fn is_pending(&self, id: &str) -> bool {
    self.lock().contains_key(id)
  }

  /// Id of some prompt currently awaiting a decision, if any. Lets a front-end that is
  /// *not* running the turn — a terminal sitting at its prompt while a browser-submitted
  /// turn asks about `delete_file` — offer to answer it.
  pub fn any_pending(&self) -> Option<String> {
    self.lock().keys().next().cloned()
  }

  pub fn is_empty(&self) -> bool {
    self.lock().is_empty()
  }

  /// A poisoned mutex means a previous holder panicked while holding it. Nothing in the
  /// single-operation critical sections above can panic, so this is unreachable in
  /// practice; recovering beats turning one historical panic into a panic on every
  /// subsequent approval.
  fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<String, oneshot::Sender<bool>>> {
    self
      .pending
      .lock()
      .unwrap_or_else(|poisoned| poisoned.into_inner())
  }
}

/// Where a [`DualApprovalCallback`] should send its prompt for the turn currently
/// executing. See [`with_approval_channel`] for how one gets attached to a turn.
#[derive(Clone)]
pub enum ApprovalChannel {
  /// Prompt on the console and block on stdin. The standalone shape, equivalent to
  /// [`crate::callback::approval::ApprovalCallback`], for a caller with exactly one
  /// front-end and no session-wide broker to publish to; also the fallback when no
  /// channel was attached at all (see [`APPROVAL_CHANNEL`]'s docs).
  Terminal,
  /// Publish a [`PendingApproval`] to the whole session and wait for any front-end to
  /// answer it. This is what a caller with more than one view of a session uses — see
  /// the module docs.
  Session(mpsc::UnboundedSender<PendingApproval>),
}

/// Attach `channel` to every [`DualApprovalCallback`] prompt triggered while `fut` runs.
///
/// Meant to wrap exactly one turn — see `run_turn`/`run_turn_stream` in `bin/cli` — not
/// an individual tool call: the channel is looked up once per call from inside
/// [`DualApprovalCallback::call`], so attaching it any more granularly than "for this
/// whole turn" would not change anything.
pub async fn with_approval_channel<F: Future>(channel: ApprovalChannel, fut: F) -> F::Output {
  APPROVAL_CHANNEL.scope(channel, fut).await
}

/// Asks a human before letting any listed tool run, through whichever
/// [`ApprovalChannel`] [`with_approval_channel`] attached to the turn currently running.
/// Denying — including by timeout — records an error result in place of the call, and the
/// model carries on without it.
pub struct DualApprovalCallback {
  dangerous_tools: std::collections::HashSet<String>,
  timeout: Duration,
  // Serializes the terminal prompt/read pair below: tool calls in the same round run
  // concurrently (see `Agent::execute_tool_calls`), and without this lock two concurrent
  // dangerous calls would interleave their console prompts and could read the wrong
  // `y`/`n` answer for the wrong tool call. A session-channel prompt does not need this:
  // each call gets its own `PendingApproval` with its own `decision` channel, so several
  // can be shown (and resolved) at once without any of them reading another's answer.
  terminal_prompt_lock: AsyncMutex<()>,
}

impl DualApprovalCallback {
  pub fn new(dangerous_tools: impl IntoIterator<Item = impl Into<String>>) -> Self {
    Self {
      dangerous_tools: dangerous_tools.into_iter().map(Into::into).collect(),
      timeout: config::approval_timeout(),
      terminal_prompt_lock: AsyncMutex::new(()),
    }
  }

  /// How long to wait for a human before denying. Defaults to
  /// [`config::approval_timeout`].
  pub fn with_timeout(mut self, timeout: Duration) -> Self {
    self.timeout = timeout;
    self
  }

  /// Console prompt for the standalone [`ApprovalChannel::Terminal`] shape. Denies by
  /// default if stdin cannot be read (e.g. a non-interactive process) or nobody answers
  /// within the timeout.
  async fn prompt_terminal(&self, tool_call: &ToolCallView<'_>) -> bool {
    let _guard = self.terminal_prompt_lock.lock().await;

    eprintln!("\n⚠️  即将执行高危操作");
    eprintln!("工具: {}", tool_call.name);
    // The raw string rather than the parsed arguments: an unparseable payload would show
    // up as `null`, and approving a call whose arguments you cannot see is worse than no
    // prompt at all.
    eprintln!("参数: {}", tool_call.raw_arguments);

    let read = tokio::task::spawn_blocking(|| {
      eprint!("是否执行？(y/n): ");
      if let Err(err) = io::stderr().flush() {
        tracing::warn!("failed to flush approval prompt: {err}");
      }
      let mut input = String::new();
      if let Err(err) = io::stdin().read_line(&mut input) {
        tracing::warn!("failed to read approval answer, denying by default: {err}");
        return false;
      }
      input.trim().eq_ignore_ascii_case("y")
    });

    // A timed-out read is abandoned, not cancelled: a blocking stdin read cannot be
    // interrupted, so its thread stays parked until a line eventually arrives (and is
    // then discarded).
    //
    // That leaves a reader sitting on stdin. Harmless for a caller whose only use of the
    // console is this prompt — the stray line is simply swallowed — but a caller that
    // *also* drives a line editor must not let the two overlap: a full-screen editor puts
    // the terminal in raw mode and queries it for the cursor position, and this reader
    // would consume the reply. Such callers should use `ApprovalChannel::Session` instead,
    // which keeps the console read under their own control (see `bin/cli`).
    let approved = match tokio::time::timeout(self.timeout, read).await {
      Ok(result) => result.unwrap_or(false),
      Err(_) => {
        eprintln!("\n⏳ 审批超时，默认拒绝");
        false
      }
    };

    if approved {
      eprintln!("✅ 已批准，继续执行...\n");
    } else {
      eprintln!("❌ 已拒绝，跳过执行\n");
    }
    approved
  }

  /// Publishes a [`PendingApproval`] to the session and waits for any front-end to answer
  /// it. Denies by default if nobody is listening, if whoever received it dropped the
  /// decision without answering, or if the timeout expires first.
  async fn prompt_session(
    sender: &mpsc::UnboundedSender<PendingApproval>,
    tool_call: &ToolCallView<'_>,
    timeout: Duration,
  ) -> bool {
    let (decision_tx, decision_rx) = oneshot::channel();
    let request = PendingApproval {
      id: tool_call.tool_call_id.to_owned(),
      tool: tool_call.name.to_owned(),
      raw_arguments: tool_call.raw_arguments.to_owned(),
      decision: decision_tx,
    };

    if sender.send(request).is_err() {
      tracing::warn!("no front-end is listening for approvals, denying by default");
      return false;
    }

    match tokio::time::timeout(timeout, decision_rx).await {
      Ok(Ok(approved)) => approved,
      // Every front-end holding the decision dropped it without answering.
      Ok(Err(_)) => {
        tracing::warn!("approval was abandoned without a decision, denying by default");
        false
      }
      Err(_) => {
        tracing::warn!(
          tool = %tool_call.name,
          timeout_secs = timeout.as_secs(),
          "nobody approved in time, denying by default"
        );
        false
      }
    }
  }
}

#[async_trait::async_trait]
impl BeforeToolCallback for DualApprovalCallback {
  async fn call(
    &self,
    _context: &ExecutionContext,
    tool_call: ToolCallView<'_>,
  ) -> Option<(ToolResultStatus, String)> {
    if !self.dangerous_tools.contains(tool_call.name) {
      return None;
    }

    let channel = APPROVAL_CHANNEL
      .try_with(Clone::clone)
      .unwrap_or(ApprovalChannel::Terminal);

    let approved = match channel {
      ApprovalChannel::Terminal => self.prompt_terminal(&tool_call).await,
      ApprovalChannel::Session(sender) => {
        Self::prompt_session(&sender, &tool_call, self.timeout).await
      }
    };

    if approved {
      None
    } else {
      Some((
        ToolResultStatus::Error,
        format!("User denied execution of {}", tool_call.name),
      ))
    }
  }
}

#[cfg(test)]
mod tests {
  use std::sync::Arc;

  use serde_json::{Value, json};

  use super::*;

  fn view<'a>(name: &'a str, arguments: &'a Value) -> ToolCallView<'a> {
    ToolCallView {
      tool_call_id: "call-1",
      name,
      arguments,
      raw_arguments: "",
    }
  }

  /// Long enough that no test below reaches it by accident; the timeout path has its own
  /// test that sets a deliberately tiny one.
  fn approval() -> DualApprovalCallback {
    DualApprovalCallback::new(["delete_file"]).with_timeout(Duration::from_secs(30))
  }

  // None of the early cases below name a tool the callback treats as dangerous, so `call`
  // returns from its guard clause before ever prompting — meaning it never touches
  // stdin/stderr, nor does it need an `ApprovalChannel` attached, and cannot block the
  // test suite waiting for an answer.

  #[tokio::test]
  async fn lets_a_tool_outside_the_dangerous_list_through_untouched() {
    let context = ExecutionContext::new();
    let args = json!({ "path": "notes.txt" });

    assert!(
      approval()
        .call(&context, view("read_file", &args))
        .await
        .is_none()
    );
  }

  #[tokio::test]
  async fn matches_dangerous_tool_names_exactly_not_as_a_substring() {
    let approval = DualApprovalCallback::new(["delete"]);
    let context = ExecutionContext::new();
    let args = json!({});

    assert!(
      approval
        .call(&context, view("delete_file", &args))
        .await
        .is_none()
    );
  }

  #[tokio::test]
  async fn session_channel_approves_when_a_front_end_says_yes() {
    let context = ExecutionContext::new();
    let args = json!({ "path": "notes.txt" });
    let (tx, mut rx) = mpsc::unbounded_channel::<PendingApproval>();

    let approval = approval();
    let call_future = with_approval_channel(
      ApprovalChannel::Session(tx),
      approval.call(&context, view("delete_file", &args)),
    );

    let respond_future = async {
      let request = rx.recv().await.expect("a request should have been sent");
      assert_eq!(request.tool, "delete_file");
      request.decision.send(true).unwrap();
    };

    let (result, ()) = tokio::join!(call_future, respond_future);
    assert!(result.is_none(), "approved calls are not short-circuited");
  }

  #[tokio::test]
  async fn session_channel_denies_when_a_front_end_says_no() {
    let context = ExecutionContext::new();
    let args = json!({});
    let (tx, mut rx) = mpsc::unbounded_channel::<PendingApproval>();

    let approval = approval();
    let call_future = with_approval_channel(
      ApprovalChannel::Session(tx),
      approval.call(&context, view("delete_file", &args)),
    );

    let respond_future = async {
      let request = rx.recv().await.expect("a request should have been sent");
      request.decision.send(false).unwrap();
    };

    let (result, ()) = tokio::join!(call_future, respond_future);
    assert!(matches!(result, Some((ToolResultStatus::Error, _))));
  }

  #[tokio::test]
  async fn session_channel_denies_when_nothing_is_listening() {
    let context = ExecutionContext::new();
    let args = json!({});

    let (tx, rx) = mpsc::unbounded_channel::<PendingApproval>();
    drop(rx); // No front-end attached: `sender.send` fails immediately.

    let result = with_approval_channel(
      ApprovalChannel::Session(tx),
      approval().call(&context, view("delete_file", &args)),
    )
    .await;

    assert!(matches!(result, Some((ToolResultStatus::Error, _))));
  }

  #[tokio::test]
  async fn session_channel_denies_when_the_decision_is_dropped_unused() {
    let context = ExecutionContext::new();
    let args = json!({});
    let (tx, mut rx) = mpsc::unbounded_channel::<PendingApproval>();

    let approval = approval();
    let call_future = with_approval_channel(
      ApprovalChannel::Session(tx),
      approval.call(&context, view("delete_file", &args)),
    );

    let drop_future = async {
      // Received but never decided: dropping `decision` closes the oneshot, so the
      // waiting side resolves to `Err` rather than hanging.
      let request = rx.recv().await.expect("a request should have been sent");
      drop(request.decision);
    };

    let (result, ()) = tokio::join!(call_future, drop_future);
    assert!(matches!(result, Some((ToolResultStatus::Error, _))));
  }

  /// The case that used to wedge the whole session: a front-end receives the prompt,
  /// holds on to the decision, and nobody ever answers.
  #[tokio::test]
  async fn session_channel_denies_once_the_timeout_expires() {
    let approval =
      DualApprovalCallback::new(["delete_file"]).with_timeout(Duration::from_millis(30));
    let context = ExecutionContext::new();
    let args = json!({});
    let (tx, mut rx) = mpsc::unbounded_channel::<PendingApproval>();

    // Held, never resolved — exactly what an abandoned browser tab leaves behind.
    let hold_future = async {
      let request = rx.recv().await.expect("a request should have been sent");
      tokio::time::sleep(Duration::from_secs(30)).await;
      drop(request);
    };

    let result = tokio::select! {
      result = with_approval_channel(
        ApprovalChannel::Session(tx),
        approval.call(&context, view("delete_file", &args)),
      ) => result,
      () = hold_future => panic!("the approval should have timed out first"),
    };

    assert!(
      matches!(result, Some((ToolResultStatus::Error, _))),
      "an unanswered approval must deny rather than hang"
    );
  }

  /// The regression behind "reject in the browser, and both front-ends go silent": a
  /// decision from another view has to unblock the waiting side *immediately*, without
  /// the view that raised the prompt having to type anything. A driver that awaited its
  /// own console read inline would stop pumping the agent's event stream here, so nothing
  /// would be printed anywhere until someone pressed Enter locally.
  #[tokio::test]
  async fn another_view_can_resolve_while_the_raising_view_is_still_asking() {
    let registry = Arc::new(ApprovalRegistry::new());
    let approval = DualApprovalCallback::new(["delete_file"]).with_timeout(Duration::from_secs(30));
    let context = ExecutionContext::new();
    let args = json!({});
    let (tx, mut rx) = mpsc::unbounded_channel::<PendingApproval>();

    let registry_for_view = Arc::clone(&registry);
    let raising_view = async move {
      let pending = rx.recv().await.expect("a prompt should have been raised");
      let id = pending.id.clone();
      registry_for_view.register(id.clone(), pending.decision);
      // Stands in for a console read nobody ever answers.
      tokio::time::sleep(Duration::from_secs(30)).await;
      id
    };

    let other_view = async {
      // Wait for the prompt to be registered, then answer from "elsewhere".
      while registry.any_pending().is_none() {
        tokio::task::yield_now().await;
      }
      let id = registry.any_pending().expect("checked just above");
      assert!(registry.resolve(&id, false), "the other view decides");
    };

    let result = tokio::select! {
      result = with_approval_channel(
        ApprovalChannel::Session(tx),
        approval.call(&context, view("delete_file", &args)),
      ) => result,
      _ = async { tokio::join!(raising_view, other_view) } => {
        panic!("the approval should have resolved without a local answer")
      }
    };

    assert!(
      matches!(result, Some((ToolResultStatus::Error, _))),
      "a rejection from another view must deny the call"
    );
  }

  #[test]
  fn registry_resolves_a_registered_approval_once() {
    let registry = ApprovalRegistry::new();
    let (tx, _rx) = oneshot::channel();
    registry.register("call-1".to_owned(), tx);

    assert!(registry.resolve("call-1", true), "the first answer decides");
    assert!(
      !registry.resolve("call-1", false),
      "a second answer must be a no-op"
    );
    assert!(registry.is_empty());
  }

  #[test]
  fn registry_reports_an_unknown_approval() {
    let registry = ApprovalRegistry::new();
    assert!(!registry.resolve("nope", true));
  }

  #[tokio::test]
  async fn registry_delivers_the_decision_to_the_waiting_side() {
    let registry = ApprovalRegistry::new();
    let (tx, rx) = oneshot::channel();
    registry.register("call-1".to_owned(), tx);

    registry.resolve("call-1", true);
    assert!(rx.await.unwrap(), "the decision should arrive as sent");
  }

  #[test]
  fn registry_exposes_a_pending_id_for_another_front_end_to_answer() {
    let registry = ApprovalRegistry::new();
    assert!(registry.any_pending().is_none());

    let (tx, _rx) = oneshot::channel();
    registry.register("call-1".to_owned(), tx);
    assert_eq!(registry.any_pending().as_deref(), Some("call-1"));
  }

  #[test]
  fn registry_discards_abandoned_entries() {
    let registry = ApprovalRegistry::new();
    for id in ["call-1", "call-2"] {
      let (tx, _rx) = oneshot::channel();
      registry.register(id.to_owned(), tx);
    }

    registry.discard(&["call-1".to_owned(), "call-2".to_owned()]);
    assert!(
      registry.is_empty(),
      "a finished turn must not leave entries behind"
    );
  }
}
