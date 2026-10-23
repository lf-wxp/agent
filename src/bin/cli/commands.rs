//! Carrying out an in-chat command, for whichever front-end asked.
//!
//! The command *set* — which commands exist, what they are called, how a line resolves to
//! one, and what the `/`-triggered menu offers — lives in [`shared::commands`] so the
//! browser has the same table this process does (see that module's docs). What is here is
//! the half that cannot: performing a command's side effects and telling every view about
//! them, which needs a session store and the broadcast channel.

use std::{fmt::Write as _, time::Duration};

use agent::{
  agent::{ApprovalStore, FileApprovalStore, continuity_key_for},
  callback::dual_approval::DualApprovalCallback,
  memory::MemoryStore,
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

/// The pieces of process-wide state a command may act on, one per front-end.
///
/// A struct rather than a parameter list: every front-end has to supply the same set, and
/// as positional arguments a new one (memory was the latest) had to be threaded through
/// three call sites and made `execute` long enough to trip `clippy::too_many_arguments`.
/// Naming them at the construction site is also what keeps two same-typed stores
/// (`store` and `approvals_store` are both path-backed) from being passed the wrong way
/// round.
pub struct CommandContext<'a> {
  /// Which front-end asked. Decides what `/help` lists, and reaches every view as the
  /// origin of the echoed line.
  pub origin: MessageOrigin,
  pub store: &'a FileSessionStore,
  /// `None` when nothing is gated (`--no-approval`, or an empty `--dangerous-tools`), in
  /// which case there is no remembered decision for `/reset` to forget.
  pub approvals: Option<&'a DualApprovalCallback>,
  pub approvals_store: &'a FileApprovalStore,
  /// `None` unless the process was started with `--memory`.
  pub memory: Option<&'a MemoryStore>,
  pub session_id: &'a str,
  /// Every [`ChatEvent`] any front-end renders.
  pub events: &'a broadcast::Sender<ChatEvent>,
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
///
/// With `memory` set to `None` (the process was not started with `--memory`), `/memory`
/// says how to turn it on rather than listing nothing.
pub async fn execute(command: Command, input: &str, context: CommandContext<'_>) {
  let CommandContext {
    origin,
    store,
    approvals,
    approvals_store,
    memory,
    session_id,
    events,
  } = context;
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
    Command::Memory => memory_listing(memory).await,
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

/// Most memories `/memory` lists before it stops listing and reports a count instead.
/// A store can hold [`agent::memory::MAX_MEMORIES`] facts of up to
/// [`agent::memory::MAX_FACT_CHARS`] each, and the whole listing goes out as one
/// [`ChatEvent::SystemNotice`] — over SSE, in one frame — so an unbounded listing would be
/// megabytes of text arriving at a terminal and a browser tab.
const MAX_LISTED: usize = 50;

/// The `/memory` reply: every stored memory with its number, or why there are none.
///
/// Memories are not session state — they outlive `/reset` on purpose, that being their
/// whole point — so this lists the store as a whole, and says how to remove one.
///
/// The most recent [`MAX_LISTED`] are shown, since those are the ones a "forget #N" is
/// likely to mean. Older ones are still counted, and still reachable through `recall`.
async fn memory_listing(memory: Option<&MemoryStore>) -> String {
  let Some(memory) = memory else {
    return "长期记忆未开启。以 --memory 启动即可开启（需要配置 EMBED_MODEL）。".to_owned();
  };
  let memories = match memory.list().await {
    Ok(memories) => memories,
    Err(err) => return format!("读取长期记忆失败：{err:#}"),
  };
  if memories.is_empty() {
    return "长期记忆为空。对话中让 agent 记住某件事，它会用 remember 工具保存。".to_owned();
  }

  let mut out = String::new();
  let _ = writeln!(out, "长期记忆（共 {} 条）：", memories.len());
  let shown = memories.len().min(MAX_LISTED);
  if memories.len() > shown {
    let _ = writeln!(
      out,
      "（下面只列最近 {shown} 条；更早的 {} 条仍可被 recall 找到。）",
      memories.len() - shown
    );
  }
  for item in &memories[memories.len() - shown..] {
    let _ = writeln!(out, "  #{:<4} {}  {}", item.id, item.date(), item.text);
  }
  match memory.stale_count().await {
    Ok(0) => {}
    Ok(stale) => {
      let _ = writeln!(
        out,
        "注意：其中 {stale} 条是用另一个 embedding 模型保存的，当前无法被召回。\
         换回原来的 EMBED_MODEL，或以 --memory --reindex-memory 启动一次重新嵌入\
         （会覆盖旧向量）。"
      );
    }
    Err(err) => {
      let _ = writeln!(out, "注意：无法确认这些记忆是否仍可召回：{err:#}");
    }
  }
  let _ = write!(out, "要删除某条，直接告诉 agent，例如「忘掉第 3 条记忆」。");
  out
}

/// What `--reindex-memory` came to.
#[derive(Debug)]
pub enum Reindexed {
  /// This many memories were re-embedded.
  Done(usize),
  /// The embedder or the store failed; the memories are as they were.
  Failed(agent::Error),
  /// `timeout` ran out first. `remaining` is how many still need re-embedding, `None` if
  /// that could not be found out either.
  ///
  /// The rewrite is one transaction at the very end, so the store is whole whichever side
  /// of it the timeout fell on — it is either all re-embedded or untouched, never half —
  /// and `remaining` says which, rather than guessing.
  TimedOut { remaining: Option<usize> },
}

/// Re-embed the memories left behind by another embedding model, giving up after
/// `timeout` — what a hung embedding endpoint would otherwise hold startup up for, since
/// the retry policy alone lets a call run far longer.
pub async fn reindex_memory(memory: &MemoryStore, timeout: Duration) -> Reindexed {
  match tokio::time::timeout(timeout, memory.reindex_stale()).await {
    Ok(Ok(reindexed)) => Reindexed::Done(reindexed),
    Ok(Err(err)) => Reindexed::Failed(err),
    Err(_) => Reindexed::TimedOut {
      remaining: memory.stale_count().await.ok(),
    },
  }
}

#[cfg(test)]
mod tests {
  use super::test_embedders::*;
  use super::*;
  use agent::knowledge_base::store::VectorStore;

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

  /// A memory store in a scratch database, over `embedder`.
  fn memory_over(embedder: impl agent::knowledge_base::embed::Embedder + 'static) -> MemoryStore {
    MemoryStore::new(
      VectorStore::open_in_memory(std::sync::Arc::new(embedder)).unwrap(),
      super::super::LOCAL_SCOPE,
    )
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
      CommandContext {
        origin,
        store: &store,
        approvals: None,
        approvals_store: &approvals_store,
        memory: None,
        session_id: "s1",
        events: &tx,
      },
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
      CommandContext {
        origin: MessageOrigin::Web,
        store: &store,
        approvals: None,
        approvals_store: &approvals_store,
        memory: None,
        session_id: "s1",
        events: &tx,
      },
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

  /// Without `--memory` there is nothing to list, and the reply says how to turn it on.
  #[tokio::test]
  async fn memory_without_the_flag_says_how_to_turn_it_on() {
    let events = run(Command::Memory, MessageOrigin::Terminal).await;
    assert!(matches!(
      &events[1],
      ChatEvent::SystemNotice { text } if text.contains("--memory")
    ));
  }

  /// With memory on, `/memory` lists every fact with the number `forget` takes.
  #[tokio::test]
  async fn memory_lists_what_is_stored_with_its_numbers() {
    let memory = memory_over(Named("words"));
    let (tx, mut rx) = broadcast::channel(8);
    let notice = |rx: &mut broadcast::Receiver<ChatEvent>| {
      let _echo = rx.try_recv().unwrap();
      match rx.try_recv().unwrap() {
        ChatEvent::SystemNotice { text } => text,
        other => panic!("expected a notice, got {other:?}"),
      }
    };
    let list = |memory: &MemoryStore| {
      let (store, approvals_store, tx, memory) = (
        test_store(),
        test_approvals_store(),
        tx.clone(),
        memory.clone(),
      );
      async move {
        execute(
          Command::Memory,
          "/memory",
          CommandContext {
            origin: MessageOrigin::Web,
            store: &store,
            approvals: None,
            approvals_store: &approvals_store,
            memory: Some(&memory),
            session_id: "s1",
            events: &tx,
          },
        )
        .await;
      }
    };

    list(&memory).await;
    assert!(notice(&mut rx).contains("为空"));

    memory.remember("Prefers metric units").await.unwrap();
    memory.remember("Works on Project Atlas").await.unwrap();
    list(&memory).await;
    let text = notice(&mut rx);
    assert!(text.contains("共 2 条"), "got: {text}");
    let first = memory.list().await.unwrap()[0].id;
    assert!(text.contains(&format!("#{first}")), "got: {text}");
    assert!(text.contains("Prefers metric units") && text.contains("Project Atlas"));
  }

  /// One SSE frame carries the whole reply, so a store full of facts must not turn into
  /// a megabyte of text: the newest are listed, the rest are counted.
  #[tokio::test]
  async fn memory_lists_only_the_most_recent_when_there_are_many() {
    let memory = memory_over(Named("words"));
    for n in 0..(MAX_LISTED + 7) {
      memory
        .remember(&format!("fact number {n} about topic"))
        .await
        .unwrap();
    }

    let text = memory_listing(Some(&memory)).await;

    assert!(
      text.contains(&format!("共 {} 条", MAX_LISTED + 7)),
      "got: {text}"
    );
    assert!(text.contains("只列最近 50 条"), "got: {text}");
    assert!(text.contains("更早的 7 条"), "got: {text}");
    assert!(
      !text.contains("fact number 0 about"),
      "the oldest is left out"
    );
    assert!(
      text.contains(&format!("fact number {} about", MAX_LISTED + 6)),
      "the newest is listed"
    );
    assert_eq!(
      text
        .lines()
        .filter(|line| line.trim_start().starts_with('#'))
        .count(),
      MAX_LISTED
    );
  }

  /// `/memory` lists a memory that recall cannot return after a model switch — saying so
  /// is what stops the listing from looking like everything is fine.
  #[tokio::test]
  async fn memory_warns_about_facts_left_by_another_embedding_model() {
    let dir = std::env::temp_dir().join(format!("agent-cli-memory-stale-{}", uuid::Uuid::new_v4()));
    let path = dir.join("memory.db");
    MemoryStore::open(
      &path,
      std::sync::Arc::new(Named("old-model")),
      super::super::LOCAL_SCOPE,
    )
    .unwrap()
    .remember("Prefers metric units")
    .await
    .unwrap();
    let memory = MemoryStore::open(
      &path,
      std::sync::Arc::new(Named("new-model")),
      super::super::LOCAL_SCOPE,
    )
    .unwrap();

    let text = memory_listing(Some(&memory)).await;
    let _ = std::fs::remove_dir_all(&dir);

    assert!(
      text.contains("Prefers metric units"),
      "still listed: {text}"
    );
    assert!(
      text.contains("1 条是用另一个 embedding 模型保存的"),
      "got: {text}"
    );
  }

  /// A file holding one memory saved under `old-model`, opened under `new`.
  async fn memory_left_by_another_model(
    new: impl agent::knowledge_base::embed::Embedder + 'static,
  ) -> (MemoryStore, std::path::PathBuf) {
    let dir = std::env::temp_dir().join(format!("agent-cli-reindex-{}", uuid::Uuid::new_v4()));
    let path = dir.join("memory.db");
    MemoryStore::open(
      &path,
      std::sync::Arc::new(Named("old-model")),
      super::super::LOCAL_SCOPE,
    )
    .unwrap()
    .remember("Prefers metric units")
    .await
    .unwrap();
    let memory =
      MemoryStore::open(&path, std::sync::Arc::new(new), super::super::LOCAL_SCOPE).unwrap();
    (memory, dir)
  }

  /// `--reindex-memory`, the path that has to work when someone changes `EMBED_MODEL`.
  #[tokio::test]
  async fn reindexing_makes_the_old_models_memories_recallable_again() {
    let (memory, dir) = memory_left_by_another_model(Named("new-model")).await;
    assert_eq!(memory.stale_count().await.unwrap(), 1);

    let outcome = reindex_memory(&memory, Duration::from_secs(5)).await;
    let _ = std::fs::remove_dir_all(&dir);

    assert!(matches!(outcome, Reindexed::Done(1)), "got {outcome:?}");
    assert_eq!(memory.stale_count().await.unwrap(), 0);
  }

  /// A failing endpoint is reported and leaves the memories as they were.
  #[tokio::test]
  async fn a_failing_endpoint_leaves_the_memories_untouched() {
    let (memory, dir) = memory_left_by_another_model(Failing("new-model")).await;

    let outcome = reindex_memory(&memory, Duration::from_secs(5)).await;
    let still_stale = memory.stale_count().await.unwrap();
    let _ = std::fs::remove_dir_all(&dir);

    let Reindexed::Failed(err) = outcome else {
      panic!("expected a failure, got {outcome:?}");
    };
    assert!(err.to_string().contains("is down"), "got: {err:#}");
    assert_eq!(still_stale, 1);
  }

  /// A hung endpoint cannot hold startup up: the timeout ends the wait, and the report says
  /// how many memories are still waiting rather than claiming nothing happened.
  #[tokio::test]
  async fn a_hung_endpoint_is_given_up_on_and_the_report_says_what_is_left() {
    let (memory, dir) = memory_left_by_another_model(Hanging("new-model")).await;

    let outcome = reindex_memory(&memory, Duration::from_millis(50)).await;
    let _ = std::fs::remove_dir_all(&dir);

    assert!(
      matches!(outcome, Reindexed::TimedOut { remaining: Some(1) }),
      "got {outcome:?}"
    );
  }
}

/// Embedders for the `/memory` and `--reindex-memory` tests: dimensions are irrelevant,
/// since these tests list and migrate rather than search.
#[cfg(test)]
mod test_embedders {
  use agent::knowledge_base::embed::Embedder;

  /// Answers every text with the same vector, under the model name it is given.
  pub struct Named(pub &'static str);

  #[async_trait::async_trait]
  impl Embedder for Named {
    fn model(&self) -> &str {
      self.0
    }
    async fn embed(&self, texts: &[String]) -> agent::Result<Vec<Vec<f32>>> {
      Ok(texts.iter().map(|_| vec![1.0, 0.0]).collect())
    }
  }

  /// An endpoint that is down.
  pub struct Failing(pub &'static str);

  #[async_trait::async_trait]
  impl Embedder for Failing {
    fn model(&self) -> &str {
      self.0
    }
    async fn embed(&self, _texts: &[String]) -> agent::Result<Vec<Vec<f32>>> {
      Err(agent::error::ProviderError::transient("the embedding endpoint is down").into())
    }
  }

  /// An endpoint that accepts the request and never answers.
  pub struct Hanging(pub &'static str);

  #[async_trait::async_trait]
  impl Embedder for Hanging {
    fn model(&self) -> &str {
      self.0
    }
    async fn embed(&self, _texts: &[String]) -> agent::Result<Vec<Vec<f32>>> {
      std::future::pending().await
    }
  }
}
