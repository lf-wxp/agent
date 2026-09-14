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
//! `--no-approval` below. The prompt belongs to the *session*, not to whichever front-end
//! started the turn: it is shown in the terminal and in every connected browser tab at
//! once, and the first answer from any of them decides. Answer in the terminal by typing
//! `y`/`n` — inline when this terminal is running the turn, or at the `You>` prompt when
//! a browser-submitted turn raised it. Nothing waits forever: an unanswered prompt denies
//! after [`agent::config::approval_timeout`] (`AGENT_APPROVAL_TIMEOUT_SECS`), so walking
//! away from a prompt cannot wedge the session — which matters because a turn holds the
//! session's turn lock until it finishes. A bulky
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
//! In-chat commands are defined once in [`shared::commands`] and shared by both
//! front-ends: `/help` lists them, `/reset` clears the current session's history,
//! `exit`/`quit` (or Ctrl-D) leaves the chat. A command is handled before a turn starts,
//! so it never reaches the model; its output is broadcast like anything else, so a
//! command run in one view is visible in the others. Typing `/` here opens a menu of
//! them to pick from (`Tab` reopens it, `↑`/`↓` walk it, `Enter` picks) rather than
//! requiring that they be remembered — see [`mod@completer`]; the browser's composer
//! offers the same menu over the same table.
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

mod commands;
mod completer;
mod web;

use std::{
  collections::{HashMap, VecDeque},
  io::{self, IsTerminal, Write},
  net::SocketAddr,
  sync::Arc,
};

use agent::{
  Agent, AgentResult, AgentStreamEvent,
  agent::{BeforeToolCallback, Conversation, Event},
  callback::{
    dual_approval::{
      ApprovalChannel, ApprovalMeta, ApprovalOutcome, ApprovalRegistry, DualApprovalCallback,
      PendingApproval, parse_approval_answer, with_approval_channel,
    },
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
  Reedline, Signal, Vi, default_emacs_keybindings, default_vi_insert_keybindings,
  default_vi_normal_keybindings,
};
use shared::{ChatEvent, MessageOrigin, commands as command_set};
use tokio::sync::{Mutex as AsyncMutex, broadcast, mpsc};

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
  let run_web = mode != "cli";
  // A REPL also needs somewhere to read from. `reedline` drives the terminal directly
  // (raw mode, cursor shapes) and cannot work against a pipe or a closed stdin: asked to
  // anyway, its first `read_line` fails outright ("Device not configured"), which used to
  // take the whole process — and, under `--mode both`, the web server with it — down
  // before the port it had just announced could serve anything.
  //
  // So the terminal half is only run when there is a terminal to run it on:
  //
  // - `--mode both` degrades to exactly `--mode web`, since that half was asked for and
  //   works perfectly well on its own. Announced rather than done quietly: a missing REPL
  //   is worth knowing about, and it is usually a sign this was launched from somewhere
  //   that cannot host one (a detached/background job, an IDE's run panel, a pipe).
  // - `--mode cli` has nothing left to fall back to, so it is an error, naming the mode
  //   that would have worked instead.
  let stdin_is_tty = io::stdin().is_terminal();
  if !stdin_is_tty && !run_web {
    anyhow::bail!(
      "--mode cli needs an interactive terminal, but stdin is not a TTY. Use `--mode web` \
       for a browser-only session."
    );
  }
  let run_terminal = mode != "web" && stdin_is_tty;
  if run_web && !run_terminal && mode == "both" {
    println!("Note: stdin is not a TTY, so there is no terminal prompt — serving the web UI only.");
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
  // Held onto rather than handed straight to the agent: the callback owns the "always
  // allow/deny this tool" answers a human gave (see `DualApprovalCallback`'s `sticky`
  // field), and `/reset`/`--fresh` have to be able to clear them. `Arc<DualApproval-
  // Callback>` coerces to `Arc<dyn BeforeToolCallback>` at the registration below, so one
  // allocation serves both.
  //
  // `None` when nothing is gated (`--no-approval`, or an empty `--dangerous-tools`), in
  // which case no callback is registered at all.
  let approval_callback =
    (!dangerous_tools.is_empty()).then(|| Arc::new(DualApprovalCallback::new(dangerous_tools)));
  let agent = match &approval_callback {
    // The explicit coercion is the point of holding an `Arc<DualApprovalCallback>`: the
    // same allocation is both the agent's hook and this binary's handle for clearing
    // remembered decisions.
    Some(callback) => {
      agent.with_before_tool_callback(Arc::clone(callback) as Arc<dyn BeforeToolCallback>)
    }
    None => agent,
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

  // The session's in-flight approvals, shared with the web server exactly like
  // `turn_lock` is: a terminal and a browser pointed at this session are two views of one
  // conversation, so either must be able to answer a prompt the other raised. See
  // `agent::callback::dual_approval`'s module docs.
  let approvals = Arc::new(ApprovalRegistry::new());

  // Created unconditionally (even in `--mode cli`, where nothing ever subscribes to
  // it): every turn this loop runs broadcasts onto it below, and gating that behind
  // `if run_web` would mean duplicating the loop body instead of just letting a
  // send with no subscribers be the harmless no-op `broadcast::Sender::send` already
  // makes it. `web::WebState` (built below, only when `run_web`) holds a clone of this
  // exact sender — see `web::new_event_channel`'s docs for why it is not built inside
  // `WebState::new` itself.
  let web_events_tx = web::new_event_channel();

  if args.contains_key("fresh") {
    commands::clear_session(&store, approval_callback.as_deref(), &session_id).await;
    println!("Cleared history for session `{session_id}`.\n");
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
      Arc::clone(&approvals),
      approval_callback.clone(),
      web_events_tx.clone(),
    ));
    let dist_dir = config::cli_web_dist_dir();
    // Bound here, not inside the spawned task: the port being taken is the ordinary way
    // this fails, and `--mode both` detaches that task, so binding in there meant the
    // error went nowhere while the banner below had already announced the UI was up.
    // Failing on this `?` also keeps `--mode both` from starting a terminal session that
    // silently has no browser half.
    let listener = web::bind(addr).await?;
    // Only now that the port is actually claimed — a browser opened the moment this
    // appears cannot arrive before the listener does.
    println!("Web UI: http://{addr} (session `{session_id}`)\n");
    if let Some(warning) = web::dist_warning(&dist_dir) {
      println!("Warning: {warning}\n");
    }
    let handle = tokio::spawn(async move { web::serve(state, listener, &dist_dir).await });

    if !run_terminal {
      // `--mode web`: no REPL to keep the process alive, so this *is* the run — block
      // here until the server errors or the process is killed (e.g. `Ctrl-C`), the same
      // way any other long-running server would.
      handle.await.context("web server task panicked")??;
      return Ok(());
    }
    // `--mode both`: the REPL below is what keeps this process alive, so the server runs
    // on as a task this thread never awaits. Its *outcome* is still watched, by a second
    // task — an `axum::serve` that stops mid-run leaves every browser tab dead while the
    // terminal carries on working, which is not something to find out by guessing. Only
    // reachable if the server stops early; a healthy one never resolves.
    tokio::spawn(async move {
      let report = match handle.await {
        Ok(Ok(())) => "web server stopped".to_owned(),
        Ok(Err(err)) => format!("web server stopped: {err:#}"),
        Err(err) => format!("web server panicked: {err}"),
      };
      tracing::error!("{report}");
      // Also straight to the terminal, via `term_write` for the raw-mode reasons that
      // function documents: `tracing` output is not necessarily visible here, and this
      // is exactly the kind of thing someone sitting at the `You>` prompt needs told.
      term_write(&format!("\n[web] {report}\n"));
    });
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
  //
  // Either way the command menu's keys are bound on top (see `completer`): in vi mode
  // only over the *insert* bindings, since `/` in normal mode is vi's own search.
  let vi_mode = !args.contains_key("no-vi-mode");
  let edit_mode: Box<dyn ReedlineEditMode> = if vi_mode {
    let mut insert = default_vi_insert_keybindings();
    completer::bind_menu_keys(&mut insert);
    Box::new(Vi::new(insert, default_vi_normal_keybindings()))
  } else {
    let mut emacs = default_emacs_keybindings();
    completer::bind_menu_keys(&mut emacs);
    Box::new(Emacs::new(emacs))
  };
  let mut editor = Reedline::create()
    .with_edit_mode(edit_mode)
    .with_completer(Box::new(completer::CommandCompleter))
    .with_menu(completer::command_menu())
    .with_cursor_config(configure_cursor());

  println!(
    "agent CLI — session `{session_id}` (model: {}) — workspace `{}`{}. {}.\n",
    config::model(),
    workspace.display(),
    if args.contains_key("no-sandbox") {
      ""
    } else {
      " (sandboxed)"
    },
    command_set::hint()
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
    // Commands are handled here rather than becoming a turn: none reaches the model or
    // waits on the turn lock. Their output goes out on the broadcast like everything
    // else, so a browser sharing this session sees it too — and so this terminal renders
    // it through the one renderer (`print_chat_events`) rather than printing directly.
    match command_set::parse(input) {
      Some(command_set::Command::Exit) => break,
      Some(command) => {
        commands::execute(
          command,
          input,
          MessageOrigin::Terminal,
          &store,
          approval_callback.as_deref(),
          &session_id,
          &web_events_tx,
        )
        .await;
        continue;
      }
      None => {}
    }
    // A prompt raised by another view of this session — a browser-submitted turn — can
    // be answered from here, since this terminal is idle while that turn runs. Checked
    // before anything below treats the line as a message: while something is pending, a
    // bare `y` is far more likely to be an answer than a chat turn.
    if resolve_pending_approval(&approvals, &web_events_tx, input) {
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
      drive_terminal_turn(
        &agent,
        &store,
        &turn_lock,
        &approvals,
        &session_id,
        input,
        &turn,
        &web_events_tx,
      )
      .await?;
    } else {
      // `--no-stream`: one `run_turn` call instead of the streaming counterpart, so
      // nothing is printed until the model — and every tool round it runs along the way —
      // has fully finished. Approvals still have to reach every view of the session while
      // that happens, so they are published from a task running alongside the call rather
      // than inline.
      let (approval_tx, approval_rx) = mpsc::unbounded_channel::<PendingApproval>();
      let pump = tokio::spawn(publish_approvals(
        approval_rx,
        Arc::clone(&approvals),
        web_events_tx.clone(),
      ));

      let result = run_turn(
        &agent,
        &store,
        &turn_lock,
        &session_id,
        input,
        ApprovalChannel::Session(approval_tx),
      )
      .await;

      // Dropping the turn's sender ends the pump, which then clears anything it raised.
      let raised = pump.await.unwrap_or_default();
      approvals.discard(&raised);
      let result = result?;
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

/// Run one terminal-originated streaming turn, publishing any approval it raises to the
/// whole session and offering to answer it right here.
///
/// Mirrors [`web::drive_turn`] deliberately: both front-ends raise approvals the same way
/// (to the shared [`ApprovalRegistry`], broadcast to every view) and both clear whatever
/// they raised when the turn ends. The one asymmetry is that this side can also *ask* —
/// stdin is free while a terminal turn runs, since the REPL is busy consuming this
/// stream — whereas a browser-submitted turn leaves the terminal at its `You>` prompt,
/// where [`resolve_pending_approval`] takes over instead.
///
/// The console read is a *branch* of the loop below rather than something awaited inside
/// one. Awaiting it inline would stop polling `stream` for as long as the human took to
/// answer — so a decision made in another view would unblock the agent, yet none of the
/// events that followed would be drained or printed: both front-ends would sit silent
/// until someone finally pressed Enter here.
#[allow(clippy::too_many_arguments)]
async fn drive_terminal_turn(
  agent: &Agent,
  store: &FileSessionStore,
  turn_lock: &AsyncMutex<()>,
  approvals: &Arc<ApprovalRegistry>,
  session_id: &str,
  input: &str,
  turn: &str,
  events: &broadcast::Sender<ChatEvent>,
) -> anyhow::Result<()> {
  let (approval_tx, mut approval_rx) = mpsc::unbounded_channel::<PendingApproval>();
  let stream = run_turn_stream(
    agent,
    store,
    turn_lock,
    session_id,
    input,
    ApprovalChannel::Session(approval_tx),
  );
  futures::pin_mut!(stream);

  let mut raised = Vec::new();
  let mut failure = None;
  // Tool calls in one round run concurrently, so several prompts can be outstanding at
  // once; they are asked one at a time so two console prompts cannot interleave. Only the
  // description is queued — the decision channel lives in `approvals` from the moment the
  // prompt is registered, so that any view can answer it.
  let mut queued: VecDeque<ApprovalMeta> = VecDeque::new();
  let mut asking: Option<(String, tokio::task::JoinHandle<Option<ApprovalOutcome>>)> = None;

  loop {
    // Take over the console for the next prompt that is still unanswered. The prompt
    // itself was already drawn by `print_chat_events` when the broadcast went out — this
    // only claims stdin to read the reply, and only one at a time, so two concurrent
    // prompts cannot interleave their input.
    if asking.is_none() {
      while let Some(next) = queued.pop_front() {
        if !approvals.is_pending(&next.id) {
          continue;
        }
        asking = Some((next.id, tokio::task::spawn_blocking(read_approval_line)));
        break;
      }
    }

    tokio::select! {
      pending = approval_rx.recv() => {
        let Some(PendingApproval { meta, decision }) = pending else {
          // Every sender for this turn is gone; nothing more will arrive here, but the
          // stream below is what actually ends the loop.
          continue;
        };
        raised.push(meta.id.clone());
        let _ = events.send(ChatEvent::ApprovalRequired {
          id: meta.id.clone(),
          tool: meta.tool.clone(),
          arguments: meta.raw_arguments.clone(),
        });
        // Registered before being queued: registration is what makes the prompt
        // answerable from any view (and visible to `GET /api/approvals`), whereas the
        // queue below only decides whose turn it is to read an answer off this console.
        approvals.register(meta.clone(), decision);
        queued.push_back(meta);
      }
      answered = async {
                   // `&mut JoinHandle` is itself a future, so the handle stays in place
                   // and other branches keep their claim on it across polls.
                   (&mut asking.as_mut().expect("guarded by the condition below").1).await
                 }, if asking.is_some() => {
        let (id, _) = asking.take().expect("guarded above");
        // `None` means stdin could not be read, or what was typed was not an answer;
        // leave the prompt for another view or the timeout. A stale answer (another view
        // got there first) resolves nothing.
        if let Some(outcome) = answered.unwrap_or(None) {
          // Read before the outcome is handed over: it carries a reason string, so it is
          // moved rather than copied into `resolve`.
          let approved = outcome.approved;
          if approvals.resolve(&id, outcome) {
            let _ = events.send(ChatEvent::ApprovalResolved { id, approved });
          }
        }
      }
      next = stream.next() => {
        match next {
          Some(Ok(event)) => {
            for chat_event in web::to_chat_events(&event, turn) {
              let _ = events.send(chat_event);
            }
          }
          Some(Err(err)) => {
            failure = Some(err);
            break;
          }
          None => break,
        }
      }
    }
  }

  approvals.discard(&raised);

  // A console read can still be outstanding here: another view answered the prompt first,
  // or it timed out and the turn carried on without it. Either way the answer no longer
  // matters — but the read itself cannot be cancelled, because a blocking stdin read stays
  // parked until a line actually arrives.
  //
  // So it has to be waited out rather than abandoned. Handing the terminal back to
  // `reedline` while this reader is still on stdin means two readers competing for it:
  // `reedline` puts the terminal in raw mode and asks it for the cursor position, this
  // reader consumes the reply, and `reedline` fails with "The cursor position could not be
  // read within a normal duration". Blocking here until the line lands keeps stdin
  // single-reader at all times, which is what makes returning to the prompt safe.
  if let Some((_, handle)) = asking.take() {
    term_write("\n(该审批已由其他界面处理，按回车返回输入)\n");
    let _ = handle.await;
  }

  match failure {
    Some(err) => Err(err),
    None => Ok(()),
  }
}

/// One blocking answer read from the console. `None` if stdin is closed or unreadable
/// (a non-interactive process, say), or if what was typed is not an answer at all —
/// either way the caller treats it as "no answer from here".
///
/// Unbounded on purpose: the waiting side in [`agent::callback::dual_approval`] already
/// applies [`agent::config::approval_timeout`], so bounding it here too would just race
/// two clocks.
fn read_approval_line() -> Option<ApprovalOutcome> {
  let mut input = String::new();
  match io::stdin().read_line(&mut input) {
    Ok(0) => None, // stdin closed.
    Ok(_) => parse_approval_answer(&input),
    Err(err) => {
      tracing::warn!("failed to read approval answer: {err}");
      None
    }
  }
}

/// Forward approvals raised by a non-streaming turn to the session, returning the ids it
/// raised so the caller can clear them once the turn ends.
///
/// The `--no-stream` counterpart of the `approval_rx` arm in [`drive_terminal_turn`]:
/// `run_turn` does not yield anything until it is completely finished, so without a task
/// alongside it nothing would publish an approval while it waits — the turn would block
/// on a prompt no view had been told about, until it timed out. This side does not ask on
/// the console: stdin belongs to the blocked `run_turn` call's own callback path here, so
/// the answer comes from a browser or from [`resolve_pending_approval`] afterwards.
async fn publish_approvals(
  mut approvals_rx: mpsc::UnboundedReceiver<PendingApproval>,
  approvals: Arc<ApprovalRegistry>,
  events: broadcast::Sender<ChatEvent>,
) -> Vec<String> {
  let mut raised = Vec::new();
  while let Some(PendingApproval { meta, decision }) = approvals_rx.recv().await {
    raised.push(meta.id.clone());
    let announcement = ChatEvent::ApprovalRequired {
      id: meta.id.clone(),
      tool: meta.tool.clone(),
      arguments: meta.raw_arguments.clone(),
    };
    // Registered before the announcement goes out, so a view that reacts to it by
    // immediately asking `GET /api/approvals` cannot see an empty list.
    approvals.register(meta, decision);
    let _ = events.send(announcement);
  }
  raised
}

/// Answer an approval raised by *another* view of this session — a browser-submitted turn
/// asking about `delete_file` while this terminal sits idle at `You>`.
///
/// The terminal cannot prompt inline in that situation: it is not running the turn, and
/// stdin belongs to `reedline`. So the REPL reads `y`/`n` as a decision instead of a
/// message whenever something is pending. `true` if `line` was consumed here rather than
/// being sent to the model.
///
/// Anything else typed while a prompt is outstanding is also consumed — with a reminder
/// of what is being asked. Letting it through would start a second turn that immediately
/// blocks on the turn lock the pending one still holds, so the reply would go nowhere and
/// the prompt would stay unanswered; saying so beats that silent stall.
fn resolve_pending_approval(
  approvals: &ApprovalRegistry,
  events: &broadcast::Sender<ChatEvent>,
  line: &str,
) -> bool {
  let Some(id) = approvals.any_pending() else {
    return false;
  };

  let Some(outcome) = parse_approval_answer(line) else {
    term_write(
      "\n[approval] 有待处理的审批，请输入 y（本次允许）/ n（本次拒绝）/ \
       a（本会话总是允许）/ d（本会话总是拒绝），可加「: 理由」\n",
    );
    return true;
  };

  // Read before `outcome` is moved into `resolve` — it owns an optional reason string.
  let approved = outcome.approved;
  if approvals.resolve(&id, outcome) {
    let _ = events.send(ChatEvent::ApprovalResolved { id, approved });
  }
  true
}

/// The approval prompt as this terminal shows it, for a prompt raised by *any* view of
/// the session.
///
/// One renderer so the two cases cannot drift apart: a browser-raised prompt used to get
/// a one-line "waiting on a decision in the browser" — no arguments to judge it by, and
/// no hint that it was answerable from here — while a terminal-raised one got the full
/// block. They are the same prompt and carry the same options, so they read the same.
///
/// Raw arguments rather than parsed ones: a payload that failed to parse would render as
/// `null`, and approving a call you cannot see is worse than not being asked.
fn approval_prompt(tool: &str, raw_arguments: &str) -> String {
  format!(
    "\n⚠️  即将执行高危操作\n工具: {tool}\n参数: {raw_arguments}\n\
     是否执行？(y=本次允许 / n=本次拒绝 / a=本会话总是允许 / d=本会话总是拒绝): "
  )
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
    ChatEvent::SystemNotice { text } => term_write(&format!("\n{text}\n")),
    ChatEvent::ApprovalRequired {
      tool, arguments, ..
    } => {
      // The single place an approval prompt is rendered on this terminal, whichever view
      // raised it — see `approval_prompt`. The turn driver does not print one of its own;
      // it only broadcasts, and this arm (a subscriber like any other) draws it.
      term_write(&approval_prompt(&tool, &arguments));
    }
    ChatEvent::ApprovalResolved { approved, .. } => {
      term_write(&format!(
        "\n[approval] {}\n",
        if approved {
          "✅ 已批准，继续执行..."
        } else {
          "❌ 已拒绝，跳过执行"
        }
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
/// `channel` is where any [`DualApprovalCallback`] prompt this turn triggers should go.
/// Both front-ends pass [`ApprovalChannel::Session`]: a prompt belongs to the session
/// rather than to whoever started the turn, so either view can answer it (see
/// [`drive_terminal_turn`] and [`web::drive_turn`]). It is re-attached (via
/// [`with_approval_channel`]) around each
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

    let inner = agent.run_continuing_stream(
      Conversation::new(session_id, history).with_scope(LOCAL_SCOPE),
      input,
    );
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

  let result = with_approval_channel(
    channel,
    agent.run_continuing(
      Conversation::new(session_id, history).with_scope(LOCAL_SCOPE),
      input,
    ),
  )
  .await?;
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
///
/// Built from [`CursorConfig::default`] (all modes untouched) rather than an exhaustive
/// literal: the struct also carries `hx_*` fields under `reedline`'s optional `helix`
/// feature, so which fields a literal must name depends on whether anything in the
/// dependency graph turned that feature on — and this CLI has no helix mode to style.
fn configure_cursor() -> CursorConfig {
  CursorConfig {
    vi_insert: Some(SetCursorStyle::SteadyBar),
    vi_normal: Some(SetCursorStyle::SteadyBlock),
    emacs: None,
    ..CursorConfig::default()
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
  fn format_elapsed_picks_the_coarsest_matching_unit() {
    use std::time::Duration;

    assert_eq!(format_elapsed(Duration::from_secs(5)), "just now");
    assert_eq!(format_elapsed(Duration::from_secs(59)), "just now");
    assert_eq!(format_elapsed(Duration::from_secs(120)), "2m ago");
    assert_eq!(format_elapsed(Duration::from_secs(3 * 3_600)), "3h ago");
    assert_eq!(format_elapsed(Duration::from_secs(2 * 86_400)), "2d ago");
  }

  /// Whichever view raised it, the prompt has to carry the same three things: the tool,
  /// the arguments to judge it by, and how to answer.
  #[test]
  fn the_approval_prompt_shows_the_tool_arguments_and_options() {
    let prompt = approval_prompt("delete_file", r#"{"path":"notes.txt"}"#);

    assert!(prompt.contains("delete_file"));
    assert!(
      prompt.contains(r#"{"path":"notes.txt"}"#),
      "the arguments are what the decision is made on"
    );
    // Every accepted key has to be offered, not just the one-off pair: an answer the
    // parser understands but the prompt never mentions is an answer nobody will give.
    for key in ["y", "n", "a", "d"] {
      assert!(
        prompt.contains(&format!("{key}=")),
        "the prompt must spell out `{key}`, got: {prompt}"
      );
    }
  }

  fn registry_with_one_pending() -> (
    ApprovalRegistry,
    tokio::sync::oneshot::Receiver<ApprovalOutcome>,
  ) {
    let registry = ApprovalRegistry::new();
    let (tx, rx) = tokio::sync::oneshot::channel();
    registry.register(
      ApprovalMeta {
        id: "call-1".to_owned(),
        tool: "delete_file".to_owned(),
        raw_arguments: r#"{"path":"notes.txt"}"#.to_owned(),
        requested_at: 0,
      },
      tx,
    );
    (registry, rx)
  }

  #[test]
  fn a_pending_approval_captures_every_answer_at_the_prompt() {
    for (line, expected) in [
      ("y", ApprovalOutcome::once(true)),
      ("Y", ApprovalOutcome::once(true)),
      ("yes", ApprovalOutcome::once(true)),
      ("/approve", ApprovalOutcome::once(true)),
      ("n", ApprovalOutcome::once(false)),
      ("no", ApprovalOutcome::once(false)),
      ("/deny", ApprovalOutcome::once(false)),
      // The sticky answers: the REPL has to carry the scope through, not just the
      // verdict, or "always allow" would silently degrade to a one-off yes.
      ("a", ApprovalOutcome::sticky(true)),
      ("always", ApprovalOutcome::sticky(true)),
      ("/always", ApprovalOutcome::sticky(true)),
      ("d", ApprovalOutcome::sticky(false)),
      ("never", ApprovalOutcome::sticky(false)),
      ("/never", ApprovalOutcome::sticky(false)),
    ] {
      let (registry, rx) = registry_with_one_pending();
      let (events, _keepalive) = broadcast::channel(4);

      assert!(
        resolve_pending_approval(&registry, &events, line),
        "{line} should be taken as an answer"
      );
      assert_eq!(
        rx.blocking_recv().unwrap(),
        expected,
        "{line} should resolve to {expected:?}"
      );
      assert!(registry.is_empty(), "answering clears the prompt");
    }
  }

  /// Anything else is held back rather than sent as a message: it would only start a turn
  /// that blocks on the lock the pending one still holds.
  #[test]
  fn other_input_is_held_back_while_an_approval_is_pending() {
    let (registry, _rx) = registry_with_one_pending();
    let (events, _keepalive) = broadcast::channel(4);

    assert!(resolve_pending_approval(&registry, &events, "what is 2+2?"));
    assert!(
      !registry.is_empty(),
      "the prompt is still waiting for a real answer"
    );
  }

  /// With nothing pending, `y` is an ordinary message and must reach the model.
  #[test]
  fn input_passes_through_when_no_approval_is_pending() {
    let registry = ApprovalRegistry::new();
    let (events, _keepalive) = broadcast::channel(4);

    assert!(!resolve_pending_approval(&registry, &events, "y"));
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
