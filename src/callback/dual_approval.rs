//! [`DualApprovalCallback`]: the same "ask a human before running a dangerous tool" idea
//! as [`crate::callback::approval::ApprovalCallback`], but able to ask that human through
//! either the terminal or a web front-end, chosen per turn rather than baked into the
//! callback itself.
//!
//! A single [`Agent`](crate::agent::Agent) shared by both front-ends in the same process
//! (see `docs/web-ui-plan.md`'s "both" mode) registers exactly one before-tool callback;
//! that callback cannot know by itself whether the turn it is currently running for was
//! typed in the terminal or submitted from a browser tab. [`with_approval_channel`] is
//! how the caller driving that turn tells it: a [`tokio::task_local!`] carries the answer
//! from wherever the turn starts down to whichever [`BeforeToolCallback::call`] this
//! triggers, however many tool calls that turn ends up making.

use std::{
  collections::HashSet,
  future::Future,
  io::{self, Write},
};

use tokio::sync::{Mutex, mpsc, oneshot};

use crate::agent::{
  ExecutionContext, ToolResultStatus,
  callback::{BeforeToolCallback, ToolCallView},
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

/// One tool call waiting on a human decision submitted through a web front-end.
///
/// Constructed by [`DualApprovalCallback`] and sent down whichever
/// [`ApprovalChannel::Web`] sender is active for the current turn; the receiving end (a
/// `/chat` SSE handler, see `docs/web-ui-plan.md`) is responsible for showing it to the
/// user and, once they decide, resolving `decision` — a `POST /approve/{id}` route
/// sending `true`/`false` down it is the intended shape. `id` is the tool call's own id
/// (the same one the model assigned it, and the same one a [`crate::agent::AgentStreamEvent::
/// ToolCallsStarted`] the browser already received carries), so the UI does not need a
/// separate id scheme just for approvals.
pub struct PendingWebApproval {
  pub id: String,
  pub tool: String,
  pub raw_arguments: String,
  pub decision: oneshot::Sender<bool>,
}

/// Where a [`DualApprovalCallback`] should send its prompt for the turn currently
/// executing. See [`with_approval_channel`] for how one gets attached to a turn.
#[derive(Clone)]
pub enum ApprovalChannel {
  /// Prompt on the console and block on stdin — exactly what
  /// [`crate::callback::approval::ApprovalCallback`] does; this is also the fallback
  /// used when no channel was ever attached (see [`APPROVAL_CHANNEL`]'s docs), so an
  /// unmodified terminal-only caller needs no changes to keep that behavior.
  Terminal,
  /// Send a [`PendingWebApproval`] down this sender instead of touching stdin/stderr,
  /// then wait for its `decision`.
  Web(mpsc::UnboundedSender<PendingWebApproval>),
}

/// Attach `channel` to every [`DualApprovalCallback`] prompt triggered while `fut` runs.
///
/// Meant to wrap exactly one turn — see `run_turn`/`run_turn_stream` in `bin/cli.rs` — not
/// an individual tool call: the channel is looked up once per call from inside
/// [`DualApprovalCallback::call`], so attaching it any more granularly than "for this
/// whole turn" would not change anything, and attaching it any less granularly (e.g. once
/// for the whole process) would not let two front-ends sharing one `Agent` (see
/// `docs/web-ui-plan.md`'s "both" mode) each get their own turns routed to the right
/// place.
pub async fn with_approval_channel<F: Future>(channel: ApprovalChannel, fut: F) -> F::Output {
  APPROVAL_CHANNEL.scope(channel, fut).await
}

/// Asks a human — on the console or through a web front-end, depending on which
/// [`ApprovalChannel`] [`with_approval_channel`] attached to the turn currently running —
/// before letting any listed tool run. Denying records an error result in place of the
/// call, and the model carries on without it.
pub struct DualApprovalCallback {
  dangerous_tools: HashSet<String>,
  // Serializes the terminal prompt/read pair below: tool calls in the same round run
  // concurrently (see `Agent::execute_tool_calls`), and without this lock two concurrent
  // dangerous calls would interleave their console prompts and could read the wrong
  // `y`/`n` answer for the wrong tool call. A web-channel prompt does not need this: each
  // call gets its own `PendingWebApproval` with its own `decision` channel, so the
  // browser can show (and resolve) several at once without any of them reading another's
  // answer.
  terminal_prompt_lock: Mutex<()>,
}

impl DualApprovalCallback {
  pub fn new(dangerous_tools: impl IntoIterator<Item = impl Into<String>>) -> Self {
    Self {
      dangerous_tools: dangerous_tools.into_iter().map(Into::into).collect(),
      terminal_prompt_lock: Mutex::new(()),
    }
  }

  /// Same prompt/read sequence as [`crate::callback::approval::ApprovalCallback`]: blocks
  /// on stdin with no timeout, denies by default if stdin cannot be read (e.g. a
  /// non-interactive process).
  async fn prompt_terminal(&self, tool_call: &ToolCallView<'_>) -> bool {
    let _guard = self.terminal_prompt_lock.lock().await;

    eprintln!("\n⚠️  即将执行高危操作");
    eprintln!("工具: {}", tool_call.name);
    // The raw string rather than the parsed arguments: an unparseable payload would show
    // up as `null`, and approving a call whose arguments you cannot see is worse than no
    // prompt at all.
    eprintln!("参数: {}", tool_call.raw_arguments);

    let approved = tokio::task::spawn_blocking(|| {
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
    })
    .await
    .unwrap_or(false);

    if approved {
      eprintln!("✅ 已批准，继续执行...\n");
    } else {
      eprintln!("❌ 已拒绝，跳过执行\n");
    }
    approved
  }

  /// Sends a [`PendingWebApproval`] down `sender` and waits for its `decision`. Denies by
  /// default if `sender`'s receiver is gone (the web session ended before deciding) or
  /// `decision` is dropped without ever being sent (same failure) — the same fail-closed
  /// default [`Self::prompt_terminal`] uses when stdin cannot be read.
  async fn prompt_web(
    sender: &mpsc::UnboundedSender<PendingWebApproval>,
    tool_call: &ToolCallView<'_>,
  ) -> bool {
    let (decision_tx, decision_rx) = oneshot::channel();
    let request = PendingWebApproval {
      id: tool_call.tool_call_id.to_owned(),
      tool: tool_call.name.to_owned(),
      raw_arguments: tool_call.raw_arguments.to_owned(),
      decision: decision_tx,
    };

    if sender.send(request).is_err() {
      tracing::warn!("approval channel's receiver is gone, denying by default");
      return false;
    }

    decision_rx.await.unwrap_or(false)
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
      ApprovalChannel::Web(sender) => Self::prompt_web(&sender, &tool_call).await,
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

  // None of the cases below name a tool the callback treats as dangerous, so `call`
  // returns from its guard clause before ever prompting — meaning it never touches
  // stdin/stderr, nor does it need an `ApprovalChannel` attached, and cannot block the
  // test suite waiting for an answer.

  #[tokio::test]
  async fn lets_a_tool_outside_the_dangerous_list_through_untouched() {
    let approval = DualApprovalCallback::new(["delete_file"]);
    let context = ExecutionContext::new();
    let args = json!({ "path": "notes.txt" });

    assert!(
      approval
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
  async fn web_channel_approves_when_the_receiver_says_yes() {
    let approval = DualApprovalCallback::new(["delete_file"]);
    let context = ExecutionContext::new();
    let args = json!({ "path": "notes.txt" });

    let (tx, mut rx) = mpsc::unbounded_channel::<PendingWebApproval>();

    let call_future = with_approval_channel(
      ApprovalChannel::Web(tx),
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
  async fn web_channel_denies_when_the_receiver_says_no() {
    let approval = DualApprovalCallback::new(["delete_file"]);
    let context = ExecutionContext::new();
    let args = json!({});

    let (tx, mut rx) = mpsc::unbounded_channel::<PendingWebApproval>();

    let call_future = with_approval_channel(
      ApprovalChannel::Web(tx),
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
  async fn web_channel_denies_by_default_when_the_receiver_is_dropped() {
    let approval = DualApprovalCallback::new(["delete_file"]);
    let context = ExecutionContext::new();
    let args = json!({});

    let (tx, rx) = mpsc::unbounded_channel::<PendingWebApproval>();
    drop(rx); // No web session listening: `sender.send` fails immediately.

    let result = with_approval_channel(
      ApprovalChannel::Web(tx),
      approval.call(&context, view("delete_file", &args)),
    )
    .await;

    assert!(matches!(result, Some((ToolResultStatus::Error, _))));
  }

  #[tokio::test]
  async fn web_channel_denies_by_default_when_the_decision_sender_is_dropped_unused() {
    let approval = DualApprovalCallback::new(["delete_file"]);
    let context = ExecutionContext::new();
    let args = json!({});

    let (tx, mut rx) = mpsc::unbounded_channel::<PendingWebApproval>();

    let call_future = with_approval_channel(
      ApprovalChannel::Web(tx),
      approval.call(&context, view("delete_file", &args)),
    );

    let drop_future = async {
      // Received but never decided (e.g. the browser tab closed): dropping `decision`
      // closes the oneshot, so `decision_rx.await` resolves to `Err` on the other side.
      let request = rx.recv().await.expect("a request should have been sent");
      drop(request.decision);
    };

    let (result, ()) = tokio::join!(call_future, drop_future);
    assert!(matches!(result, Some((ToolResultStatus::Error, _))));
  }
}
