//! Interactive CLI chat client: `cargo run --bin cli`.
//!
//! A terminal front-end for [`agent::Agent`], the same way `bin/server.rs` is an HTTP
//! front-end for it: same `Agent`, same tool-calling loop, just a different way for a
//! human to drive it. Every line typed is one turn; the assistant's reply streams back
//! token by token by default (see [`agent::AgentStreamEvent`]), or all at once with
//! `--no-stream`.
//!
//! Unlike the HTTP API, a CLI invocation is short-lived — the process exits when the
//! user leaves the chat, and a later invocation should be able to pick the same
//! conversation back up. That is exactly what [`agent::session::FileSessionStore`] is
//! for (see its docs, and [`agent::session::SessionStore`]'s): each turn's transcript is
//! saved to disk under `--session <id>` and reloaded on the next run.
//!
//! Tools: the built-in set plus, when [`agent::config::mcp_config_path`]
//! (`MCP_CONFIG_PATH`, default `mcp.json`) exists, every enabled MCP server it declares
//! (see [`agent::tools::ToolRegistry::with_mcp`]) — no config file is not an error, it
//! just means MCP is skipped. `--tools` opts out of both in favor of an explicit,
//! built-in-only subset (MCP tools are not nameable that way; see
//! [`agent::tools::ToolRegistry::select`]).
//!
//! `--workspace <dir>` pins the one directory this invocation operates in (default: the
//! directory `cli` was launched from, so nothing changes for anyone who never passes
//! this): every relative path the process touches afterwards — a filesystem tool
//! argument from the model, the default `mcp.json` lookup, the default
//! `.agent/sessions` — resolves against it, via a real
//! [`std::env::set_current_dir`], not just a value threaded through by convention. On
//! top of that, [`agent::callback::path_guard::WorkspaceGuardCallback`] enforces it as a
//! hard boundary for the built-in filesystem tools specifically: a call whose path
//! argument resolves outside `--workspace` (an absolute path elsewhere, a `../` escape,
//! …) is denied before it runs — before an approval prompt for it would even ask a
//! human to weigh in. `--no-sandbox` turns that enforcement off; the working directory
//! itself is still pinned either way.
//!
//! Destructive tools (`delete_file` by default) prompt for a `y`/`n` on the console
//! before running, via [`agent::callback::approval::ApprovalCallback`] — see `--dangerous-
//! tools` / `--no-approval` below. A bulky `web_search` result is compressed to the
//! passages that answer the query before it enters the transcript, via
//! [`agent::callback::search_compressor::SearchCompressorCallback`] — see
//! `--no-search-compression` below.
//!
//! ```sh
//! cargo run --bin cli                              # chat in the `default` session
//! cargo run --bin cli -- --session work            # a separate, named conversation
//! cargo run --bin cli -- --workspace ~/projects/foo  # pin to (and sandbox to) a dir
//! cargo run --bin cli -- --tools calculator        # restrict to these built-ins, no MCP
//! cargo run --bin cli -- --fresh                   # clear this session before starting
//! cargo run --bin cli -- --list                     # list every stored session
//! cargo run --bin cli -- --rm work                  # delete the `work` session
//! cargo run --bin cli -- --no-stream                # print the whole reply at once
//! cargo run --bin cli -- --dangerous-tools delete_file,demo__write_file
//! cargo run --bin cli -- --no-approval             # run every tool without prompting
//! cargo run --bin cli -- --no-search-compression   # keep web_search results uncompressed
//! cargo run --bin cli -- --no-sandbox               # let filesystem tools roam anywhere
//! cargo run --bin cli -- --no-vi-mode               # use Emacs keybindings instead
//! ```
//!
//! Line editing at the `You>` prompt is handled by [`reedline`] (the line editor behind
//! `nushell`), in `vi`'s modal editing mode by default: type in insert mode as normal,
//! `Esc` drops to normal mode for `hjkl`/`w`/`b`/`0`/`$`/`dd`/… (and `k`/`j` to walk
//! history), `i`/`a` back to insert mode — mirroring `bash`'s `set -o vi` / `zsh`'s
//! `bindkey -v`. The terminal cursor itself changes shape with the mode (a steady bar in
//! insert mode, a steady block in normal mode — via [`reedline::CursorConfig`]) so which
//! mode is active is visible without reading the typed text. `--no-vi-mode` switches to
//! `reedline`'s other mode, Emacs-style keybindings (arrow keys, `Ctrl-A`/`Ctrl-E`,
//! history via up/down, and no cursor-shape switching, since Emacs mode has no
//! insert/normal distinction to indicate) — the stock `bash`/`readline` default; nothing
//! else about the chat loop changes either way. `Ctrl-C` cancels the line currently
//! being typed (loops back to a fresh prompt) rather than killing the process; `Ctrl-D`
//! on an empty line still leaves the chat, same as before.
//!
//! In-chat commands: `/reset` clears the current session's history; `exit` / `quit`
//! (or Ctrl-D) leaves the chat.
//!
//! `--list` and `--rm <session>` are one-shot session-management commands: each prints
//! its result and exits immediately, without starting a chat or touching the configured
//! LLM provider (see [`list_sessions`], [`remove_session_command`]).
//!
//! There is no multi-tenant concept here (contrast [`agent::api::handlers::
//! AuthenticatedTenant`]): every session on this machine lives under one constant scope
//! ([`LOCAL_SCOPE`]) and is distinguished purely by `--session`. Session files live under
//! [`agent::config::cli_session_dir`] (`AGENT_CLI_SESSION_DIR`) and, unlike the HTTP
//! server's TTL'd sessions, never expire ([`FileSessionStore::new_persistent`]) — a
//! conversation from any time ago can be resumed; only `--fresh` / `/reset` clears one.

use std::{
  collections::HashMap,
  io::{self, Write},
  sync::Arc,
};

use agent::{
  Agent, AgentStreamEvent,
  agent::Event,
  callback::{
    approval::ApprovalCallback, path_guard::WorkspaceGuardCallback,
    search_compressor::SearchCompressorCallback,
  },
  config,
  llm::provider::Provider,
  session::{FileSessionStore, SessionStore},
  telemetry,
  tools::{ToolRegistry, file_delete},
};
use anyhow::Context;
use crossterm::cursor::SetCursorStyle;
use futures::StreamExt;
use reedline::{
  CursorConfig, EditMode as ReedlineEditMode, Emacs, Prompt, PromptEditMode, PromptHistorySearch,
  Reedline, Signal, Vi, default_vi_insert_keybindings, default_vi_normal_keybindings,
};

const SYSTEM_PROMPT: &str =
  "You are a helpful, general-purpose assistant running in a command-line chat session.";

/// The `You>` prompt: a fixed left-hand label, nothing on the right, and no extra
/// indicator text — [`reedline`]'s cursor shape already communicates the current vi
/// mode (see [`configure_cursor`] and the module docs), so there is nothing useful to
/// add here beyond the label itself.
struct ChatPrompt;

impl Prompt for ChatPrompt {
  fn render_prompt_left(&self) -> std::borrow::Cow<'_, str> {
    "You> ".into()
  }

  fn render_prompt_right(&self) -> std::borrow::Cow<'_, str> {
    "".into()
  }

  fn render_prompt_indicator(&self, _prompt_mode: PromptEditMode) -> std::borrow::Cow<'_, str> {
    "".into()
  }

  fn render_prompt_multiline_indicator(&self) -> std::borrow::Cow<'_, str> {
    "::: ".into()
  }

  fn render_prompt_history_search_indicator(
    &self,
    history_search: PromptHistorySearch,
  ) -> std::borrow::Cow<'_, str> {
    format!(
      "({}reverse-search: {}) ",
      if matches!(
        history_search.status,
        reedline::PromptHistorySearchStatus::Failing
      ) {
        "failed "
      } else {
        ""
      },
      history_search.term
    )
    .into()
  }
}

/// Every CLI session lives under this constant scope: a CLI has no bearer-token tenant
/// the way the HTTP API does, so there is nothing meaningful to isolate sessions by
/// besides the session id itself (see [`agent::session::SessionStore`]'s `scope`
/// parameter).
const LOCAL_SCOPE: &str = "local";

const DEFAULT_SESSION_ID: &str = "default";

/// Commands (case-insensitive) that leave the chat.
const EXIT_COMMANDS: [&str; 4] = ["exit", "quit", ":q", "/exit"];

#[tokio::main]
async fn main() -> anyhow::Result<()> {
  telemetry::init()?;

  let args = parse_args();

  // Every relative path this process touches afterwards — a filesystem tool argument
  // from the model, the default `mcp.json` lookup, the default `.agent/sessions` — is
  // resolved by the OS against the current working directory. Pinning it once, here,
  // to `--workspace` (default: wherever this was invoked from, so this is a no-op for
  // anyone who never passes the flag) is what turns "whatever directory the shell
  // happened to be in" into one deliberate, named root for the whole invocation — the
  // gap this flag exists to close (see the module docs).
  let workspace = match args.get("workspace") {
    Some(dir) => {
      std::fs::canonicalize(dir).with_context(|| format!("--workspace `{dir}` does not exist"))?
    }
    None => std::env::current_dir().context("failed to read the current directory")?,
  };
  std::env::set_current_dir(&workspace)
    .with_context(|| format!("failed to switch to workspace `{}`", workspace.display()))?;

  // Persistent (never-expiring) store: resuming "the conversation I had last week" is a
  // normal thing to want from a CLI, unlike the HTTP server's ephemeral, TTL'd sessions.
  // Nothing is ever swept; only an explicit `--fresh` / `/reset` clears a session.
  let store = FileSessionStore::new_persistent(config::cli_session_dir());

  // `--list` / `--rm` are one-shot session-management commands, handled before touching
  // the LLM provider at all: neither needs a configured `Agent` (see the module docs).
  if args.contains_key("list") {
    return list_sessions(&store).await;
  }
  if let Some(target) = args.get("rm") {
    return remove_session_command(&store, target).await;
  }

  let session_id = args
    .get("session")
    .filter(|value| !value.is_empty())
    .cloned()
    .unwrap_or_else(|| DEFAULT_SESSION_ID.to_owned());

  // `--tools` opts out of MCP entirely (see the module docs): it names an explicit
  // built-in subset, and MCP tools have no fixed names to list before they are
  // discovered from the server, so the two are mutually exclusive by construction.
  let (registry, mcp_connections) = match args.get("tools").filter(|value| !value.is_empty()) {
    Some(names) => (
      ToolRegistry::select(&names.split(',').map(str::to_owned).collect::<Vec<_>>())?,
      Vec::new(),
    ),
    None => ToolRegistry::with_mcp(config::mcp_config_path()).await?,
  };

  // `delete_file` by default: the one built-in tool that destroys data outside this
  // process. `--dangerous-tools` overrides the list (comma-separated; an empty value
  // disables prompting for everything), and `--no-approval` is shorthand for the same
  // regardless of what `--dangerous-tools` would otherwise pick.
  let dangerous_tools: Vec<String> = if args.contains_key("no-approval") {
    Vec::new()
  } else {
    match args.get("dangerous-tools") {
      Some(names) => names
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_owned)
        .collect(),
      None => vec![file_delete::NAME.to_owned()],
    }
  };

  let agent = Agent::new(
    Provider::shared().clone(),
    config::model(),
    Some(SYSTEM_PROMPT),
    Arc::new(registry),
  );
  // Registered before the approval callback below: [`Agent::with_before_tool_callback`]
  // runs hooks in registration order and stops at the first denial, so an out-of-
  // workspace call is rejected here without ever reaching a `y`/`n` prompt for it.
  // `--no-sandbox` skips this registration; the working directory is still pinned to
  // `--workspace` either way (see the module docs).
  let agent = if args.contains_key("no-sandbox") {
    agent
  } else {
    agent.with_before_tool_callback(Arc::new(WorkspaceGuardCallback::new(&workspace)?))
  };
  let agent = if dangerous_tools.is_empty() {
    agent
  } else {
    agent.with_before_tool_callback(Arc::new(ApprovalCallback::new(dangerous_tools)))
  };
  // On by default: a raw `web_search` result is mostly padding that gets re-sent to the
  // model on every subsequent tool round, so compressing it once, as it enters the
  // transcript, is a strict improvement with no user-visible downside — a failed
  // compression (e.g. the embedding provider is unreachable) just falls back to the
  // uncompressed result (see `SearchCompressorCallback`) rather than erroring the turn.
  // `--no-search-compression` is only for debugging what the model actually received.
  let agent = if args.contains_key("no-search-compression") {
    agent
  } else {
    agent.with_after_tool_callback(Arc::new(SearchCompressorCallback))
  };

  // Bare flag, like `--fresh`: `--no-stream` prints the whole reply at once instead of
  // token by token. Streaming is the default because it is the more responsive
  // interactive experience; non-streaming exists for piping output or a terminal that
  // renders partial lines badly.
  let streaming = !args.contains_key("no-stream");

  if args.contains_key("fresh") {
    clear_session(&store, &session_id).await;
  }

  // vi's modal keybindings by default (see the module docs); `--no-vi-mode` falls back
  // to `reedline`'s other mode, Emacs-style. This only affects how a line is *typed*;
  // nothing about the chat loop, history, or the model changes either way.
  let vi_mode = !args.contains_key("no-vi-mode");
  let edit_mode: Box<dyn ReedlineEditMode> = if vi_mode {
    Box::new(Vi::new(
      default_vi_insert_keybindings(),
      default_vi_normal_keybindings(),
    ))
  } else {
    Box::new(Emacs::default())
  };
  let mut editor = Reedline::create()
    .with_edit_mode(edit_mode)
    .with_cursor_config(configure_cursor());

  println!(
    "agent CLI — session `{session_id}` (model: {}) — workspace `{}`{}. Type `/reset` to \
     clear history, `exit`/`quit` to leave.\n",
    config::model(),
    workspace.display(),
    if args.contains_key("no-sandbox") {
      ""
    } else {
      " (sandboxed)"
    }
  );

  loop {
    let (line, next_editor) = read_line(editor).await?;
    editor = next_editor;
    let Some(line) = line else {
      break; // Ctrl-D / stdin closed.
    };
    let input = line.trim();

    if input.is_empty() {
      continue;
    }
    if EXIT_COMMANDS.contains(&input.to_ascii_lowercase().as_str()) {
      break;
    }
    if input.eq_ignore_ascii_case("/reset") {
      clear_session(&store, &session_id).await;
      continue;
    }

    let history = store.history(LOCAL_SCOPE, &session_id).await;

    if streaming {
      let stream = agent.run_continuing_stream(history, input);
      futures::pin_mut!(stream);

      print!("Agent> ");
      io::stdout().flush()?;

      while let Some(event) = stream.next().await {
        match event? {
          // Printed without a newline: chunks are meant to be concatenated as they arrive.
          AgentStreamEvent::Token(text) => {
            print!("{text}");
            io::stdout().flush()?;
          }
          AgentStreamEvent::Done {
            context,
            budget_exhausted,
            ..
          } => {
            println!("\n");
            record_turn(&store, &session_id, context.events, budget_exhausted).await;
          }
        }
      }
    } else {
      // `--no-stream`: one `run_continuing` call instead of the streaming counterpart, so
      // nothing is printed until the model — and every tool round it runs along the way —
      // has fully finished.
      let result = agent.run_continuing(history, input).await?;
      println!("Agent> {}\n", result.output);
      record_turn(
        &store,
        &session_id,
        result.context.events,
        result.budget_exhausted,
      )
      .await;
    }
  }

  // Drop `agent` before shutting down MCP connections: `McpConnection::shutdown` refuses
  // to run while any `Arc` clone of its underlying service is still alive, and `agent`'s
  // toolbox is the last thing still holding one (see `ToolRegistry::with_mcp`). Not doing
  // this is not a resource leak — the process is about to exit either way, and a stdio
  // server's child process dies with it — but it does let a well-behaved server clean up
  // instead of being killed out from under it.
  drop(agent);
  for connection in mcp_connections {
    if let Err(err) = connection.shutdown().await {
      tracing::warn!("failed to shut down an MCP server cleanly: {err:#}");
    }
  }

  println!("Bye!");
  Ok(())
}

/// Persist a completed turn's transcript, warning first if the round budget ran out
/// before producing it. Shared by both the streaming and `--no-stream` branches of the
/// chat loop so the two cannot drift on what "finishing a turn" means.
async fn record_turn(
  store: &FileSessionStore,
  session_id: &str,
  events: Vec<Event>,
  budget_exhausted: bool,
) {
  if budget_exhausted {
    tracing::warn!("tool round budget exhausted; answer may be based on partial work");
  }
  store.save(LOCAL_SCOPE, session_id, events).await;
}

/// Reset a session's stored history to empty, printing the same confirmation whether
/// this was triggered by `--fresh` at startup or `/reset` mid-chat (both mean exactly
/// "forget this conversation and start over").
async fn clear_session(store: &FileSessionStore, session_id: &str) {
  store.save(LOCAL_SCOPE, session_id, Vec::new()).await;
  println!("Cleared history for session `{session_id}`.\n");
}

/// `--list`: print every session stored under [`LOCAL_SCOPE`], most recently active
/// first, then exit without starting a chat.
async fn list_sessions(store: &FileSessionStore) -> anyhow::Result<()> {
  let sessions = store.list(LOCAL_SCOPE).await;
  if sessions.is_empty() {
    println!(
      "No sessions found under `{}`.",
      config::cli_session_dir().display()
    );
    return Ok(());
  }

  println!("Sessions under `{}`:", config::cli_session_dir().display());
  for session in sessions {
    let elapsed = session.last_used.elapsed().unwrap_or_default();
    println!(
      "  {:<20} {:>4} event(s)   last active {}",
      session.session_id,
      session.turns,
      format_elapsed(elapsed)
    );
  }
  Ok(())
}

/// `--rm <session>`: delete a single session's stored history, then exit without
/// starting a chat. An empty `session_id` (bare `--rm` with no value) prints usage
/// instead of trying to delete a session literally named `""`.
async fn remove_session_command(store: &FileSessionStore, session_id: &str) -> anyhow::Result<()> {
  if session_id.is_empty() {
    println!("Usage: cli --rm <session>");
    return Ok(());
  }
  if store.remove(LOCAL_SCOPE, session_id).await {
    println!("Removed session `{session_id}`.");
  } else {
    println!("No session named `{session_id}` found.");
  }
  Ok(())
}

/// Render `elapsed` as a short, human-friendly "how long ago" string (e.g. `"5m ago"`).
/// Coarsest matching unit only — good enough for a session list, not meant to be a
/// precise duration.
fn format_elapsed(elapsed: std::time::Duration) -> String {
  let secs = elapsed.as_secs();
  if secs < 60 {
    "just now".to_owned()
  } else if secs < 3_600 {
    format!("{}m ago", secs / 60)
  } else if secs < 86_400 {
    format!("{}h ago", secs / 3_600)
  } else {
    format!("{}d ago", secs / 86_400)
  }
}

/// Cursor shapes per [`reedline`] edit mode (see [`reedline::CursorConfig`]): a steady
/// bar in vi insert mode, a steady block in vi normal mode — the same convention Vim
/// itself uses, so which mode is active is visible at a glance without reading the
/// typed text. `emacs: None` leaves the cursor untouched in Emacs mode, which has no
/// insert/normal distinction to indicate with a shape change.
fn configure_cursor() -> CursorConfig {
  CursorConfig {
    vi_insert: Some(SetCursorStyle::SteadyBar),
    vi_normal: Some(SetCursorStyle::SteadyBlock),
    emacs: None,
  }
}

/// Read one line at the `You>` prompt via `editor` (Emacs or vi keybindings, depending
/// on `--no-vi-mode` — see the module docs), handing the same editor back so the caller
/// can keep its history across turns. `Ok((None, editor))` means the user asked to
/// leave the chat: `Ctrl-D` (EOF). `Ctrl-C` instead yields `Ok((Some(String::new()),
/// editor))` — the existing empty-input check in the main loop then just loops back to
/// a fresh prompt, so cancelling a line in progress does not exit the chat the way it
/// would with a raw, unhandled `SIGINT`. [`Signal`] is `#[non_exhaustive]`; any variant
/// besides the three above (e.g. a host-command passthrough) is treated the same as
/// `Ctrl-C` — nothing this CLI defines emits one, but the alternative (erroring the
/// whole process out) would be a worse failure mode if `reedline` ever added one.
///
/// The blocking read runs inside [`tokio::task::spawn_blocking`] rather than directly on
/// a runtime worker thread: `Reedline::read_line` blocks its thread indefinitely waiting
/// for a human, and doing that on a `tokio` worker thread would starve every other task
/// sharing the runtime (see [`agent::callback::approval::ApprovalCallback`] for the same
/// reasoning). Nothing else runs concurrently with this chat loop today, but staying off
/// worker threads while blocked on the user is the right default regardless. `editor` is
/// moved into the blocking closure and handed back alongside the result rather than kept
/// on the async side, since [`Reedline`] is not `Clone`. History is tracked by `editor`
/// itself on a successful line — no manual bookkeeping needed here.
async fn read_line(mut editor: Reedline) -> anyhow::Result<(Option<String>, Reedline)> {
  tokio::task::spawn_blocking(move || match editor.read_line(&ChatPrompt) {
    Ok(Signal::Success(line)) => Ok((Some(line), editor)),
    Ok(Signal::CtrlD) => Ok((None, editor)),
    Ok(_) => Ok((Some(String::new()), editor)),
    Err(err) => Err(anyhow::Error::from(err)).context("failed to read input line"),
  })
  .await?
}

/// Minimal manual `--flag value` / bare `--flag` parser (the latter recorded as an empty
/// string, e.g. `--fresh`, or a bare `--rm`). No CLI-argument crate dependency for a
/// handful of flags: every other binary in this crate already reads its configuration
/// from environment variables (see `src/config.rs`) rather than flags, so this stays
/// consistent with that rather than pulling in `clap` for something this small.
fn parse_args() -> HashMap<String, String> {
  let mut args = HashMap::new();
  let mut iter = std::env::args().skip(1).peekable();
  while let Some(arg) = iter.next() {
    let Some(key) = arg.strip_prefix("--") else {
      continue;
    };
    let value = match iter.peek() {
      Some(next) if !next.starts_with("--") => iter.next().unwrap_or_default(),
      _ => String::new(),
    };
    args.insert(key.to_owned(), value);
  }
  args
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn exit_commands_are_matched_case_insensitively() {
    for input in ["exit", "EXIT", "Quit", ":q", "/exit"] {
      assert!(
        EXIT_COMMANDS.contains(&input.to_ascii_lowercase().as_str()),
        "{input} should be recognized as an exit command"
      );
    }
  }

  #[test]
  fn format_elapsed_picks_the_coarsest_matching_unit() {
    use std::time::Duration;

    assert_eq!(format_elapsed(Duration::from_secs(5)), "just now");
    assert_eq!(format_elapsed(Duration::from_secs(59)), "just now");
    assert_eq!(format_elapsed(Duration::from_secs(120)), "2m ago");
    assert_eq!(format_elapsed(Duration::from_secs(3 * 3_600)), "3h ago");
    assert_eq!(format_elapsed(Duration::from_secs(2 * 86_400)), "2d ago");
  }
}
