//! Carrying out an in-chat command, for whichever front-end asked.
//!
//! The command *set* — which commands exist, what they are called, how a line resolves to
//! one, and what the `/`-triggered menu offers — lives in [`shared::commands`] so the
//! browser has the same table this process does (see that module's docs). What is here is
//! the half that cannot: performing a command's side effects and telling every view about
//! them, which needs a session store and the broadcast channel.

use agent::session::{FileSessionStore, SessionStore};
use shared::{
  ChatEvent, MessageOrigin,
  commands::{Command, help_text},
};
use tokio::sync::broadcast;

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
pub async fn execute(
  command: Command,
  input: &str,
  origin: MessageOrigin,
  store: &FileSessionStore,
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
      store.save(super::LOCAL_SCOPE, session_id, Vec::new()).await;
      format!("已清空会话 `{session_id}` 的历史记录。")
    }
    // Nothing for a browser tab to exit: the process belongs to whoever launched it, and
    // a tab closing is not a reason to end it. `/help` does not offer this on the web for
    // the same reason (`available_on_web`); someone can still type it.
    Command::Exit => "该命令仅在命令行中可用，关闭标签页即可离开。".to_owned(),
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
    execute(command, "  /cmd  ", origin, &store, "s1", &tx).await;
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
    store
      .save(super::super::LOCAL_SCOPE, "s1", sample_history())
      .await;

    execute(
      Command::Reset,
      "/reset",
      MessageOrigin::Web,
      &store,
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
