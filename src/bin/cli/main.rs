//! Interactive CLI chat client: `cargo run --bin cli`.
//!
//! A terminal front-end for [`agent::Agent`]. Every line typed is one turn; the
//! assistant's reply streams back token by token by default (see
//! [`agent::AgentStreamEvent`]), or all at once with `--no-stream`.
//!
//! A CLI invocation is short-lived — the process exits when the
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
//! human to weigh in. The same flag also registers
//! [`agent::callback::mcp_guard::McpGuardCallback`], which covers MCP tools specifically
//! (unreachable for the built-in guard above, since an MCP tool's argument schema is
//! only known at runtime — see that callback's docs): it denies any MCP tool call whose
//! arguments name a well-known credential path (`~/.ssh`, `~/.aws`, ...), regardless of
//! which field carries it. Plus, every stdio MCP server this invocation spawns gets a
//! minimal environment rather than inheriting this process's own (see
//! `agent::tools::mcp::config`'s `INHERITED_ENV_VARS`), so a server cannot read this
//! process's own credentials (API keys, ...) just by being spawned — that part is not
//! gated by `--no-sandbox` at all, since there is no legitimate reason an MCP server
//! would need this agent's own secrets. `--no-sandbox` turns off the two call-time
//! guards above; the working directory is still pinned, and stdio servers still get a
//! minimal environment, either way.
//!
//! Destructive tools (`delete_file` by default) prompt for a `y`/`n` before running, via
//! [`agent::callback::dual_approval::DualApprovalCallback`] — see `--dangerous-tools` /
//! `--no-approval` below. The prompt goes to the console for a terminal-originated turn
//! and to the browser for a web-originated one (see [`mod@web`] and the `--mode` flag
//! below), decided per turn rather than baked into one binary-wide choice. A bulky
//! `web_search` result is compressed to the passages that answer the query before it
//! enters the transcript, via
//! [`agent::callback::search_compressor::SearchCompressorCallback`] — see
//! `--no-search-compression` below.
//!
//! `--mode` picks which front-end(s) this invocation drives, all sharing the exact same
//! `Agent`/session state — the browser is not a separate deployment, it is another way
//! to interact with *this* process (see `docs/web-ui-plan.md`):
//!
//! - `cli` (default): terminal only, unchanged from before this flag existed.
//! - `web`: no terminal REPL; only the local web server (see [`mod@web`]) runs, until it
//!   errors or the process is killed (e.g. `Ctrl-C`).
//! - `both`: terminal REPL and the web server at once. A turn typed in the terminal and
//!   one submitted from a browser tab both go through the same `--session`'s history,
//!   serialized against each other (see `run_turn_stream`'s `turn_lock` docs) rather
//!   than racing — and both are broadcast live to every connected browser tab via
//!   `GET /api/stream` (see [`web::WebState::events`]), so a message typed in the
//!   terminal shows up in an already-open browser tab without that tab having sent
//!   anything itself, and vice versa.
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
//! cargo run --bin cli -- --mode both                # terminal + browser at once
//! cargo run --bin cli -- --mode web --web-port 4000 # browser only, custom port
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
//! There is no multi-tenant/multi-user concept here: every session on this machine lives
//! under one constant scope ([`LOCAL_SCOPE`]) and is distinguished purely by `--session`.
//! Session files live under [`agent::config::cli_session_dir`] (`AGENT_CLI_SESSION_DIR`)
//! and never expire ([`FileSessionStore::new_persistent`]) — a conversation from any time
//! ago can be resumed; only `--fresh` / `/reset` clears one. The web server (`--mode
//! web`/`both`) binds `127.0.0.1` only and has no authentication of its own for the same
//! reason: there is no second user to keep out on this machine (see
//! `docs/web-ui-plan.md`'s "非目标" section).

mod web;

use std::{
  collections::HashMap,
  io::{self, Write},
  net::SocketAddr,
  sync::Arc,
};

use agent::{
  Agent, AgentResult, AgentStreamEvent,
  agent::Event,
  callback::{
    dual_approval::{ApprovalChannel, DualApprovalCallback, with_approval_channel},
    mcp_guard::McpGuardCallback,
    path_guard::WorkspaceGuardCallback,
    search_compressor::SearchCompressorCallback,
  },
  config,
  llm::provider::Provider,
  session::{FileSessionStore, SessionStore},
  telemetry,
  tools::{ToolRegistry, file_delete},
};
use anyhow::Context;
use async_stream::stream;
use crossterm::cursor::SetCursorStyle;
use futures::{Stream, StreamExt};
use reedline::{
  CursorConfig, EditMode as ReedlineEditMode, Emacs, Prompt, PromptEditMode, PromptHistorySearch,
  Reedline, Signal, Vi, default_vi_insert_keybindings, default_vi_normal_keybindings,
};
use shared::{ChatEvent, MessageOrigin};
use tokio::sync::{Mutex as AsyncMutex, broadcast};

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

/// Every CLI session lives under this constant scope: there is no multi-user concept
/// here, so there is nothing meaningful to isolate sessions by besides the session id
/// itself (see [`agent::session::SessionStore`]'s `scope`
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
  // normal thing to want from a CLI. Nothing is ever swept; only an explicit `--fresh` /
  // `/reset` clears a session.
  let store = FileSessionStore::new_persistent(config::cli_session_dir());

  // `--list` / `--rm` are one-shot session-management commands, handled before touching
  // the LLM provider at all: neither needs a configured `Agent` (see the module docs).
  if args.contains_key("list") {
    return list_sessions(&store).await;
  }
  if let Some(target) = args.get("rm") {
    return remove_session_command(&store, target).await;
  }

  let mode = args.get("mode").map(String::as_str).unwrap_or("cli");
  if !["cli", "web", "both"].contains(&mode) {
    anyhow::bail!("--mode must be one of `cli`, `web`, `both` (got `{mode}`)");
  }
  // Same `Agent`/session state either way — `--mode` only decides which front-end(s)
  // are actually driving turns against it this run (see the module docs).
  let run_terminal = mode != "web";
  let run_web = mode != "cli";

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
  // workspace call, or an MCP call reaching for a credential path, is rejected here
  // without ever reaching a `y`/`n` prompt for it. `--no-sandbox` skips both
  // registrations — same flag for both, since they are the same "sandbox" concept from
  // an operator's point of view, just covering two different tool populations (built-in
  // vs. MCP; see [`WorkspaceGuardCallback`]'s "Known limitations" for why one callback
  // could not cover both). The working directory is still pinned to `--workspace`
  // either way.
  let agent = if args.contains_key("no-sandbox") {
    agent
  } else {
    let agent = agent.with_before_tool_callback(Arc::new(WorkspaceGuardCallback::new(&workspace)?));
    agent.with_before_tool_callback(Arc::new(McpGuardCallback::new()))
  };
  let agent = if dangerous_tools.is_empty() {
    agent
  } else {
    agent.with_before_tool_callback(Arc::new(DualApprovalCallback::new(dangerous_tools)))
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

  // `Arc` from here on: shared as-is (not cloned into independent copies) between the
  // terminal loop below and the web server (`--mode web`/`both`, see [`mod@web`]) — one
  // `Agent`, one on-disk history, one lock guarding it, no matter how many front-ends
  // are driving turns against it this run.
  let agent = Arc::new(agent);
  let store = Arc::new(store);
  // Serializes concurrent turns against `store` for the same session (see
  // `run_turn_stream`'s docs) — needed for real once `--mode both` lets a terminal-
  // originated and a web-originated turn race on the same session.
  let turn_lock = Arc::new(AsyncMutex::new(()));

  // Created unconditionally (even in `--mode cli`, where nothing ever subscribes to
  // it): every turn this loop runs broadcasts onto it below, and gating that behind
  // `if run_web` would mean duplicating the loop body instead of just letting a
  // send with no subscribers be the harmless no-op `broadcast::Sender::send` already
  // makes it. `web::WebState` (built below, only when `run_web`) holds a clone of this
  // exact sender — see `web::new_event_channel`'s docs for why it is not built inside
  // `WebState::new` itself.
  let web_events_tx = web::new_event_channel();

  if args.contains_key("fresh") {
    clear_session(&store, &session_id).await;
  }

  if run_web {
    let port = args
      .get("web-port")
      .filter(|value| !value.is_empty())
      .and_then(|value| value.parse::<u16>().ok())
      .unwrap_or_else(config::cli_web_port);
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let state = Arc::new(web::WebState::new(
      Arc::clone(&agent),
      Arc::clone(&store),
      Arc::clone(&turn_lock),
      session_id.clone(),
      web_events_tx.clone(),
    ));
    let dist_dir = config::cli_web_dist_dir();
    println!("Web UI: http://{addr} (session `{session_id}`)\n");
    let handle = tokio::spawn(async move { web::serve(state, addr, &dist_dir).await });

    if !run_terminal {
      // `--mode web`: no REPL to keep the process alive, so this *is* the run — block
      // here until the server errors or the process is killed (e.g. `Ctrl-C`), the same
      // way any other long-running server would.
      handle.await.context("web server task panicked")??;
      return Ok(());
    }
    // `--mode both`: intentionally not awaited/stored anywhere further — dropping the
    // `JoinHandle` detaches the task (it keeps running; only *awaiting* the handle would
    // block here). It shares the exact same `Arc` clones as the REPL loop below, so it
    // needs no further wiring to participate in the same session.
  }

  // Reaching here means `run_terminal` is true (the `--mode web` branch above always
  // returns before this point) — subscribe now, before the REPL loop starts consuming
  // any input, so a web-originated turn that starts while this process is still setting
  // up cannot have its events land on the broadcast before anyone is listening for them.
  // See [`print_chat_events`] for what this prints and, more importantly, what it does
  // *not* re-print (the terminal's own `Terminal`-origin messages — already visible from
  // typing them).
  tokio::spawn(print_chat_events(web_events_tx.subscribe()));

  // Bare flag, like `--fresh`: `--no-stream` prints the whole reply at once instead of
  // token by token. Streaming is the default because it is the more responsive
  // interactive experience; non-streaming exists for piping output or a terminal that
  // renders partial lines badly.
  let streaming = !args.contains_key("no-stream");

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

    // Identifies this turn on the shared broadcast, the same way a browser-submitted
    // one is identified by the id `POST /api/chat` hands back (see `web::new_turn_id`).
    // Nothing in this loop needs it — a terminal runs one turn at a time by
    // construction — but a browser tab watching along does: it is how a tab tells its
    // own turn's `Done` from this one's.
    let turn = web::new_turn_id();

    // Broadcast before the turn even starts (see `web::WebState::events`'s docs): a
    // browser tab watching `/api/stream` — and [`print_chat_events`], watching on this
    // process's own behalf — should see the same input this loop is about to run, the
    // same instant it starts. Tagged `Terminal` so that task knows *not* to re-print
    // this particular message: `reedline` already echoed it to this same terminal as
    // it was typed.
    let _ = web_events_tx.send(ChatEvent::UserMessage {
      text: input.to_owned(),
      origin: MessageOrigin::Terminal,
    });

    // Neither branch below prints anything itself: [`print_chat_events`] (spawned once,
    // before this loop started) is the single renderer for every [`ChatEvent`] this
    // process broadcasts, this turn's included — see that function's docs for why
    // unifying rendering there (rather than also printing directly here, which is what
    // an earlier version of this loop did) is what makes a web-originated turn's output
    // show up in *this* terminal too, not just a browser tab's.
    if streaming {
      let stream = run_turn_stream(
        &agent,
        &store,
        &turn_lock,
        &session_id,
        input,
        ApprovalChannel::Terminal,
      );
      futures::pin_mut!(stream);

      while let Some(event) = stream.next().await {
        let event = event?;
        for chat_event in web::to_chat_events(&event, &turn) {
          let _ = web_events_tx.send(chat_event);
        }
      }
    } else {
      // `--no-stream`: one `run_turn` call instead of the streaming counterpart, so
      // nothing is printed until the model — and every tool round it runs along the way —
      // has fully finished.
      let result = run_turn(
        &agent,
        &store,
        &turn_lock,
        &session_id,
        input,
        ApprovalChannel::Terminal,
      )
      .await?;
      // No per-round `ToolCallsStarted`/`Finished` to broadcast here (this branch never
      // sees them at all — see `run_turn`'s docs), just the final text and completion,
      // so a browser watching along at least sees *something* for a non-streaming turn
      // instead of silence until the next streaming one.
      let _ = web_events_tx.send(ChatEvent::Token {
        text: result.output,
      });
      let _ = web_events_tx.send(ChatEvent::Done {
        turn,
        budget_exhausted: result.budget_exhausted,
      });
    }
  }

  // Drop `agent` before shutting down MCP connections: `McpConnection::shutdown` refuses
  // to run while any `Arc` clone of its underlying service is still alive, and `agent`'s
  // toolbox is the last thing still holding one (see `ToolRegistry::with_mcp`). Not doing
  // this is not a resource leak — the process is about to exit either way, and a stdio
  // server's child process dies with it — but it does let a well-behaved server clean up
  // instead of being killed out from under it. In `--mode both`, the detached web server
  // task above still holds its own `Arc` clone, so this drop alone will not bring the
  // count to zero and the shutdown attempt below is a best-effort no-op in that case —
  // acceptable for the same reason: the whole process is exiting regardless.
  drop(agent);
  for connection in mcp_connections {
    if let Err(err) = connection.shutdown().await {
      tracing::warn!("failed to shut down an MCP server cleanly: {err:#}");
    }
  }

  println!("Bye!");
  Ok(())
}

/// The single renderer for every [`ChatEvent`] this process broadcasts on
/// [`web::WebState::events`]/`web_events_tx` — spawned once, in `main`, right before
/// the REPL loop starts, and running for as long as the process does. Neither branch of
/// the loop above prints anything directly; this task is what actually puts characters
/// on this terminal, for a turn typed here *or* submitted from a browser tab, treating
/// both the same way except for [`ChatEvent::UserMessage`] (see the match arm below).
///
/// A [`broadcast::error::RecvError::Lagged`] is handled the same way
/// [`web::stream_handler`] handles it for a browser tab: skip ahead rather than treat it
/// as fatal — a terminal that missed a few intermediate token chunks because this task
/// briefly fell behind should keep printing what comes next, not stop rendering
/// entirely.
async fn print_chat_events(mut events: broadcast::Receiver<ChatEvent>) {
  loop {
    let event = match events.recv().await {
      Ok(event) => event,
      Err(broadcast::error::RecvError::Lagged(_)) => continue,
      Err(broadcast::error::RecvError::Closed) => break,
    };
    print_chat_event(event);
  }
}

fn print_chat_event(event: ChatEvent) {
  match event {
    ChatEvent::UserMessage { text, origin } => {
      match origin {
        // Already visible: `reedline` echoed it to this terminal as it was typed, and
        // re-printing it here would just duplicate it.
        MessageOrigin::Terminal => {}
        // Not otherwise visible here at all — this is the one thing this task prints
        // that a web-originated turn would not show up without.
        MessageOrigin::Web => term_write(&format!("\n[web] {text}\n")),
      }
      term_write("Agent> ");
    }
    // Chunks are meant to be concatenated as they arrive, hence no trailing newline of
    // its own here — but `text` itself can still contain one or more bare `\n`s (a
    // multi-line assistant reply is normal), which is exactly what `term_write` exists
    // to handle correctly.
    ChatEvent::Token { text } => term_write(&text),
    ChatEvent::ToolCallsStarted { calls } => {
      for call in calls {
        term_write(&format!("\n[tool] {}({})\n", call.name, call.arguments));
      }
    }
    ChatEvent::ToolCallsFinished { results } => {
      for result in results {
        term_write(&format!("[tool] {} -> {:?}\n", result.name, result.status));
      }
    }
    ChatEvent::ApprovalRequired { tool, .. } => {
      // Purely informational here: the actual decision for a web-originated call is
      // made in the browser (see `web::approve_handler`), and a terminal-originated
      // call's own `DualApprovalCallback::prompt_terminal` already prints its own
      // blocking `y`/`n` prompt directly — this print only covers the case this task
      // would otherwise stay silent about, a *web*-originated call waiting on the
      // browser.
      term_write(&format!(
        "\n[approval] `{tool}` is waiting on a decision in the browser\n"
      ));
    }
    ChatEvent::ApprovalResolved { approved, .. } => {
      term_write(&format!(
        "[approval] {}\n",
        if approved { "approved" } else { "denied" }
      ));
    }
    ChatEvent::Done { .. } => term_write("\n\n"),
    // `turn` is ignored here, unlike in a browser tab: this terminal has no per-turn UI
    // state to unwind (it prints as events arrive and blocks on its own turns), so which
    // turn an error belongs to changes nothing about how it is shown.
    ChatEvent::Error { message, .. } => term_write(&format!("\n[error] {message}\n")),
  }
}

/// Writes `text` to stdout, translating every bare `\n` into `\r\n` first, then flushes.
///
/// This is not cosmetic: [`reedline`]'s `read_line` (see [`read_line`]'s docs) puts the
/// terminal in raw mode for as long as this process is blocked inside it waiting for a
/// human to type — which, once a web-originated turn can run concurrently with that wait
/// (`--mode both`, or even `--mode web` while a human is sitting at the terminal without
/// typing anything), is exactly when [`print_chat_event`] is printing *this* task's
/// output. Raw mode disables the terminal driver's usual behavior of translating a bare
/// `\n` into a full "return to column 0, then move down one row" — so a plain
/// `println!` there only moves the cursor down while leaving it at whatever column it
/// was already at, and each subsequent line drifts one line's worth of already-printed
/// characters further to the right than the last. That is the exact "阶梯状缩进"
/// (staircase indentation) bug this function exists to prevent — and it needed fixing
/// everywhere this task writes a newline, not just between `println!` calls, since a
/// streamed [`ChatEvent::Token`] chunk can itself contain a bare `\n` (ordinary
/// multi-line assistant output) that needs the exact same treatment.
///
/// Explicit `\r\n` is always correct regardless of whether the terminal happens to be in
/// raw mode at the time or not: in normal (cooked) mode the driver's own `\n` -> `\r\n`
/// translation would have produced the same bytes anyway, so this changes nothing
/// visible there — it only matters, and only fixes anything, while raw mode is active.
fn term_write(text: &str) {
  let mut stdout = io::stdout();
  let _ = stdout.write_all(normalize_line_endings(text).as_bytes());
  let _ = stdout.flush();
}

/// Rewrites every bare `\n` in `text` to `\r\n`. Split out of [`term_write`] as a pure
/// function purely so it has something a unit test can call without capturing stdout.
///
/// A `\r` is skipped when `\n` is already preceded by one (checking the last character
/// already *written to the output*, not the corresponding position in `text` itself —
/// though for a single call the two amount to the same thing; the distinction only
/// matters across chunk boundaries between two separate `term_write` calls, e.g. text
/// already ending `...\r` immediately followed by another call starting `\n...`, which
/// is unlikely for model output but cheap to guard against here regardless) to avoid
/// ever emitting a redundant `\r\r\n` for text that already uses `\r\n` line endings.
fn normalize_line_endings(text: &str) -> String {
  let mut out = String::with_capacity(text.len());
  for c in text.chars() {
    if c == '\n' && !out.ends_with('\r') {
      out.push('\r');
    }
    out.push(c);
  }
  out
}

/// One streaming turn: load `session_id`'s prior history, run the agent, and persist the
/// updated transcript once it finishes — the same "load -> run -> save" sequence
/// [`run_turn`] does non-streaming. Shared by every front-end this process drives a turn
/// for (the terminal loop above; [`web::chat_handler`] below), so none of them can drift
/// on what "starting a turn" (which history to load) or "finishing one" (persisting it,
/// even if the round budget ran out) means.
///
/// `turn_lock` is held for the *whole* sequence, including while the caller is still
/// consuming the returned stream: [`FileSessionStore`] is last-write-wins (see its
/// docs), so two turns racing on the same `session_id` — e.g. one typed in the terminal
/// and one submitted from a browser tab at the same moment (`--mode both`) — could
/// otherwise have the second one's save silently overwrite the first one's. Holding the
/// lock for the whole turn instead serializes that race into a queue: the second turn's
/// `history` load waits until the first one's `save` has completed.
///
/// `channel` is where any [`DualApprovalCallback`] prompt this turn triggers should go —
/// [`ApprovalChannel::Terminal`] for the loop above, `ApprovalChannel::Web(..)` for
/// [`web::chat_handler`]. It is re-attached (via [`with_approval_channel`]) around each
/// individual `inner.next()` poll rather than around the whole stream once: a
/// [`tokio::task_local!`] only stays set for the duration of the future it wraps, and
/// `inner` (an `async_stream` generator, not something that itself takes a channel
/// parameter) is that future one poll at a time, not once for its entire lifetime.
fn run_turn_stream<'a>(
  agent: &'a Agent,
  store: &'a FileSessionStore,
  turn_lock: &'a AsyncMutex<()>,
  session_id: &'a str,
  input: &'a str,
  channel: ApprovalChannel,
) -> impl Stream<Item = anyhow::Result<AgentStreamEvent>> + 'a {
  stream! {
    let _guard = turn_lock.lock().await;
    let history = store.history(LOCAL_SCOPE, session_id).await;

    let inner = agent.run_continuing_stream(history, input);
    futures::pin_mut!(inner);

    while let Some(event) = with_approval_channel(channel.clone(), inner.next()).await {
      if let Ok(AgentStreamEvent::Done { context, budget_exhausted, .. }) = &event {
        record_turn(store, session_id, context.events.clone(), *budget_exhausted).await;
      }
      yield event;
    }
  }
}

/// Non-streaming counterpart of [`run_turn_stream`]: the same load/run/save sequence and
/// the same `turn_lock`/`channel` contract, but waits for the whole turn to finish
/// instead of forwarding tokens as they arrive.
async fn run_turn(
  agent: &Agent,
  store: &FileSessionStore,
  turn_lock: &AsyncMutex<()>,
  session_id: &str,
  input: &str,
  channel: ApprovalChannel,
) -> anyhow::Result<AgentResult> {
  let _guard = turn_lock.lock().await;
  let history = store.history(LOCAL_SCOPE, session_id).await;

  let result = with_approval_channel(channel, agent.run_continuing(history, input)).await?;
  record_turn(
    store,
    session_id,
    result.context.events.clone(),
    result.budget_exhausted,
  )
  .await;
  Ok(result)
}

/// Persist a completed turn's transcript, warning first if the round budget ran out
/// before producing it. Shared by both [`run_turn_stream`] and [`run_turn`] so the two
/// cannot drift on what "finishing a turn" means.
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
/// sharing the runtime (see [`DualApprovalCallback`] for the same reasoning). Staying off
/// worker threads while blocked on the user matters even more now that `--mode both`
/// can have the web server's tasks sharing this same runtime concurrently with this loop.
/// `editor` is moved into the blocking closure and handed back alongside the result
/// rather than kept on the async side, since [`Reedline`] is not `Clone`. History is
/// tracked by `editor` itself on a successful line — no manual bookkeeping needed here.
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

  #[test]
  fn normalize_line_endings_inserts_cr_before_every_bare_lf() {
    assert_eq!(
      normalize_line_endings("[web] hi\nAgent> "),
      "[web] hi\r\nAgent> "
    );
    assert_eq!(normalize_line_endings("a\nb\nc"), "a\r\nb\r\nc");
  }

  #[test]
  fn normalize_line_endings_does_not_double_an_existing_cr() {
    assert_eq!(normalize_line_endings("a\r\nb"), "a\r\nb");
  }

  #[test]
  fn normalize_line_endings_is_a_no_op_without_any_newline() {
    assert_eq!(normalize_line_endings("Agent> "), "Agent> ");
  }
}
