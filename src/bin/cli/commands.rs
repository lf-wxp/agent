//! Carrying out an in-chat command, for whichever front-end asked.
//!
//! The command *set* — which commands exist, what they are called, how a line resolves to
//! one, and what the `/`-triggered menu offers — lives in [`shared::commands`] so the
//! browser has the same table this process does (see that module's docs). What is here is
//! the half that cannot: performing a command's side effects and telling every view about
//! them, which needs a session store and the broadcast channel.

use agent::{
  agent::{ApprovalStore, FileApprovalStore, continuity_key_for},
  callback::dual_approval::DualApprovalCallback,
  session::{FileSessionStore, SessionStore},
};
use shared::{
  ChatEvent, MessageOrigin,
  commands::{Command, help_text},
};
use tokio::sync::broadcast;

/// Forget everything a session accumulated outside its transcript.
///
/// Clearing the stored history is not by itself a reset. Two other things belong to the
/// same conversation and would otherwise outlive it:
///
/// - A remembered "always allow" for a destructive tool, which lives in the approval
///   callback. Leaving it in place would carry the single riskiest piece of session state
///   across exactly the boundary the user asked to draw.
/// - A suspended run, which holds this conversation *mid-turn*. Leaving it would let a
///   cleared session be resumed straight back into what was just cleared.
///
/// Both are done here rather than at the call sites. They were duplicated across the
/// three of them (`--fresh`, the terminal's `/reset`, the browser's `/reset`), which is
/// the shape of bug where one copy gets a fix and the others do not — and for state whose
/// whole purpose is to gate destructive operations, the failure is silent: something the
/// user asked to forget simply keeps applying.
///
/// Shared by `/reset` and `--fresh`, which mean the same thing.
///
/// `approvals` is `None` when nothing is gated (`--no-approval`, or an empty
/// `--dangerous-tools`), in which case there is no remembered decision to forget.
pub async fn clear_session(
  store: &FileSessionStore,
  approvals: Option<&DualApprovalCallback>,
  approvals_store: &FileApprovalStore,
  session_id: &str,
) {
  store.save(super::LOCAL_SCOPE, session_id, Vec::new()).await;
  if let Some(approvals) = approvals {
    approvals.forget_sticky(&continuity_key_for(Some(super::LOCAL_SCOPE), session_id));
  }
  approvals_store.remove(super::LOCAL_SCOPE, session_id).await;
}

/// Carry out `command` and broadcast everything a front-end needs in order to render it:
/// an echo of the line that asked for it, then the reply as a
/// [`ChatEvent::SystemNotice`].
///
/// Shared by the terminal loop and `POST /api/chat` so the two cannot disagree on what a
/// command does or on which events it produces. They each had their own copy of this
/// once and drifted apart exactly as you would expect: a fix applied to the web copy was
/// not applied to the terminal's, and a command typed at the terminal left every browser
/// tab showing a "thinking" indicator for a turn that was never going to start.
///
/// The echo goes out first so every view's transcript shows the question as well as the
/// answer — otherwise a `/help` run from one front-end appears in the others as an
/// explanation with nothing prompting it. `origin` says which front-end asked (a
/// renderer that already echoed the line locally, like the terminal's, uses it to avoid
/// printing it twice) and doubles as the answer to "can this view run every command?",
/// which is what `/help` filters its listing on.
///
/// [`Command::Exit`] is included for the sake of a front-end that cannot honor it, which
/// gets told so; the terminal acts on it directly instead of calling this.
// Eight parameters, seven of which are the pieces of session state a command may need to
// act on — the same list `web::WebState::new` carries for the same reason. A parameter
// struct would move the list one level out and add a type whose only job is to be
// destructured here, while the call sites (one per front-end) would still name all eight.
#[allow(clippy::too_many_arguments)]
pub async fn execute(
  command: Command,
  input: &str,
  origin: MessageOrigin,
  store: &FileSessionStore,
  approvals: Option<&DualApprovalCallback>,
  approvals_store: &FileApprovalStore,
  session_id: &str,
  events: &broadcast::Sender<ChatEvent>,
) {
  let web = matches!(origin, MessageOrigin::Web);

  let _ = events.send(ChatEvent::UserMessage {
    text: input.trim().to_owned(),
    origin,
  });

  let notice = match command {
    Command::Help => help_text(web),
    Command::Reset => {
      clear_session(store, approvals, approvals_store, session_id).await;
      format!("已清空会话 `{session_id}` 的历史记录。")
    }
    // Nothing for a browser tab to exit: the process belongs to whoever launched it, and
    // a tab closing is not a reason to end it. `/help` does not offer this on the web for
    // the same reason (`available_on_web`); someone can still type it.
    Command::Exit => "该命令仅在命令行中可用，关闭标签页即可离开。".to_owned(),
    // Turn-level, so every front-end dispatches these before reaching here — see
    // `Command`'s docs. Reaching this arm means one of them forgot to, which would
    // otherwise show up as a command that silently does nothing.
    Command::Resume | Command::Discard => {
      tracing::error!(
        ?command,
        "a turn-level command reached the side-effect path"
      );
      "该命令未被当前界面处理，请报告此问题。".to_owned()
    }
  };

  let _ = events.send(ChatEvent::SystemNotice { text: notice });
}

#[cfg(test)]
mod tests {
  use super::*;

  fn test_store() -> FileSessionStore {
    FileSessionStore::new_persistent(
      std::env::temp_dir().join(format!("agent-cli-commands-test-{}", uuid::Uuid::new_v4())),
    )
  }

  fn test_approvals_store() -> FileApprovalStore {
    FileApprovalStore::new(std::env::temp_dir().join(format!(
      "agent-cli-commands-approvals-{}",
      uuid::Uuid::new_v4()
    )))
  }

  fn sample_history() -> Vec<agent::agent::Event> {
    vec![agent::agent::Event::new(
      "exec",
      "user",
      vec![agent::agent::ContentItem::Message {
        role: "user".to_owned(),
        content: "hi".to_owned(),
      }],
    )]
  }

  /// Draining the channel is what the assertions are really about: a command has to put
  /// *both* halves of the exchange on the stream. Broadcasting only the reply leaves a
  /// view rendering an answer to nothing; broadcasting only the echo leaves one waiting
  /// for a reply that never comes, which is exactly the bug `execute` was extracted to
  /// stop the two front-ends from reintroducing one at a time.
  async fn run(command: Command, origin: MessageOrigin) -> Vec<ChatEvent> {
    let (tx, mut rx) = broadcast::channel(8);
    let store = test_store();
    let approvals_store = test_approvals_store();
    execute(
      command,
      "  /cmd  ",
      origin,
      &store,
      None,
      &approvals_store,
      "s1",
      &tx,
    )
    .await;
    drop(tx);

    let mut events = Vec::new();
    while let Ok(event) = rx.try_recv() {
      events.push(event);
    }
    events
  }

  #[tokio::test]
  async fn a_command_broadcasts_its_echo_before_its_reply() {
    let events = run(Command::Help, MessageOrigin::Terminal).await;

    assert_eq!(
      events.len(),
      2,
      "expected an echo and a reply, got {events:?}"
    );
    assert!(
      matches!(
        &events[0],
        // Trimmed: the echo is what every view renders as the question.
        ChatEvent::UserMessage { text, origin }
          if text == "/cmd" && *origin == MessageOrigin::Terminal
      ),
      "first event should echo the line that was typed, got {:?}",
      events[0]
    );
    assert!(
      matches!(&events[1], ChatEvent::SystemNotice { text } if text.contains("/help")),
      "second event should carry the reply, got {:?}",
      events[1]
    );
  }

  /// The echo carries the origin through so a front-end that already showed the line
  /// locally (the terminal, via `reedline`) can avoid printing it a second time.
  #[tokio::test]
  async fn the_echo_reports_which_front_end_asked() {
    let events = run(Command::Help, MessageOrigin::Web).await;
    assert!(matches!(
      &events[0],
      ChatEvent::UserMessage { origin, .. } if *origin == MessageOrigin::Web
    ));
  }

  #[tokio::test]
  async fn help_from_the_web_omits_terminal_only_commands() {
    let events = run(Command::Help, MessageOrigin::Web).await;
    let ChatEvent::SystemNotice { text } = &events[1] else {
      panic!("expected a notice, got {:?}", events[1]);
    };
    assert!(text.contains("/reset"));
    assert!(
      !text.contains(":q"),
      "a browser tab cannot exit the process, so it should not be offered"
    );
  }

  #[tokio::test]
  async fn reset_clears_the_stored_history() {
    let (tx, _rx) = broadcast::channel(8);
    let store = test_store();
    let approvals_store = test_approvals_store();
    store
      .save(super::super::LOCAL_SCOPE, "s1", sample_history())
      .await;

    execute(
      Command::Reset,
      "/reset",
      MessageOrigin::Web,
      &store,
      None,
      &approvals_store,
      "s1",
      &tx,
    )
    .await;

    assert!(
      store
        .history(super::super::LOCAL_SCOPE, "s1")
        .await
        .is_empty()
    );
  }

  /// A reset has to drop the suspended run as well. The stored run holds this very
  /// conversation mid-turn, so keeping it would leave `/resume` able to carry the session
  /// straight back into the history that was just cleared — and, worse, to re-raise an
  /// approval for a turn the user has already walked away from.
  ///
  /// Asserted on `clear_session` rather than on one front-end's `/reset`, because that is
  /// the point of it living there: all three entry points get this from one place.
  #[tokio::test]
  async fn reset_drops_a_suspended_run_along_with_the_history() {
    use agent::{
      AgentRunState,
      agent::{ExecutionContext, RunFingerprint},
    };

    let store = test_store();
    let approvals_store = test_approvals_store();

    // Built from its serialized shape rather than as a struct literal:
    // `AgentRunState::context` is deliberately not public, so that a mid-turn transcript
    // cannot be handed straight to a model or filed as history. JSON is the supported way
    // in from outside the crate — it is how a stored run gets loaded in the first place.
    let state: AgentRunState = serde_json::from_value(serde_json::json!({
      "fingerprint": RunFingerprint::new("gpt-test", None, ["delete_file"]),
      "suspended": [{
        "tool_call_id": "call_1",
        "name": "delete_file",
        "raw_arguments": r#"{"path":"a.txt"}"#,
      }],
      "budget_exhausted": false,
      "reason": "AwaitingDecision",
      "context": ExecutionContext::new(),
    }))
    .expect("this is the on-disk shape of a suspended run");

    approvals_store
      .put(super::super::LOCAL_SCOPE, "s1", &state)
      .await;
    assert!(
      approvals_store
        .peek(super::super::LOCAL_SCOPE, "s1")
        .await
        .is_some(),
      "the run should be stored before the reset"
    );

    clear_session(&store, None, &approvals_store, "s1").await;

    assert!(
      approvals_store
        .peek(super::super::LOCAL_SCOPE, "s1")
        .await
        .is_none(),
      "a cleared session must not leave a resumable run behind"
    );
  }

  /// A reset has to clear the remembered "always allow/deny" answers too, not just the
  /// transcript: standing permission for a destructive tool is the one piece of session
  /// state where carrying it across a reset is actively dangerous.
  #[tokio::test]
  async fn reset_forgets_remembered_approval_decisions() {
    use agent::{
      agent::{BeforeToolCallback, ExecutionContext, ToolCallDecision, ToolCallView},
      callback::dual_approval::{ApprovalChannel, ApprovalOutcome, with_approval_channel},
    };

    let store = test_store();
    let approvals = DualApprovalCallback::new(["delete_file"]);

    // A context standing in for a turn of session `s1`, keyed the way the CLI keys its
    // own — which is what `clear_session` has to address to clear anything at all.
    let mut context = ExecutionContext::new();
    context.conversation_id = Some("s1".to_owned());
    context.conversation_scope = Some(super::super::LOCAL_SCOPE.to_owned());

    let arguments = serde_json::json!({ "path": "notes.txt" });
    let call = || ToolCallView {
      tool_call_id: "call-1",
      name: "delete_file",
      arguments: &arguments,
      raw_arguments: r#"{"path":"notes.txt"}"#,
    };

    // Grant standing permission by answering the first prompt with a sticky approval.
    let (approval_tx, mut approval_rx) = tokio::sync::mpsc::unbounded_channel();
    let first = with_approval_channel(
      ApprovalChannel::Session(approval_tx),
      approvals.call(&context, call()),
    );
    let answer = async {
      let pending = approval_rx.recv().await.expect("a prompt was raised");
      let _ = pending.decision.send(ApprovalOutcome::sticky(true));
    };
    let (result, ()) = tokio::join!(first, answer);
    assert!(result.is_proceed(), "the sticky answer approved this call");

    // With it remembered, a second call needs no prompt — no channel is attached, so if
    // one were raised this would fall back to a terminal prompt and hang rather than
    // return.
    assert!(
      approvals.call(&context, call()).await.is_proceed(),
      "the remembered answer should apply without asking again"
    );

    clear_session(&store, Some(&approvals), &test_approvals_store(), "s1").await;

    // Forgotten: nothing is remembered, so the call is gated again rather than silently
    // proceeding. With no front-end listening it suspends rather than refusing — the
    // default, see `WhenUnanswered` — but either way the point holds: the tool does not
    // run on the strength of an answer given before the reset.
    let (dead_tx, dead_rx) = tokio::sync::mpsc::unbounded_channel();
    drop(dead_rx);
    let after_reset = with_approval_channel(
      ApprovalChannel::Session(dead_tx),
      approvals.call(&context, call()),
    )
    .await;
    assert!(
      !after_reset.is_proceed(),
      "after a reset the tool must be gated again, not silently allowed"
    );
    assert!(
      matches!(after_reset, ToolCallDecision::Suspend),
      "with nobody listening the question is kept, not answered"
    );
  }

  /// A front-end that cannot end the process is told so rather than silently getting
  /// nothing back — the terminal never routes `Exit` here, it acts on it directly.
  #[tokio::test]
  async fn exit_from_the_web_explains_itself() {
    let events = run(Command::Exit, MessageOrigin::Web).await;
    assert!(matches!(
      &events[1],
      ChatEvent::SystemNotice { text } if text.contains("命令行")
    ));
  }
}
