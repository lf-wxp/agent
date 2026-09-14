//! The in-chat command set, defined once for every front-end.
//!
//! A command typed at the terminal and one submitted from a browser tab mean the same
//! thing, so both resolve through [`parse`] against the same [`COMMANDS`] table rather
//! than each matching its own string literals. `/help` renders that table, so a command
//! cannot exist without being discoverable — the list and the behavior are the same data.
//!
//! Commands are handled *before* a turn starts: none of them reaches the model, and none
//! takes the session's turn lock.
//!
//! This lives in `shared` rather than next to the terminal's own command handling
//! because both front-ends need the table itself, not just the right to ask the server
//! about it. The `/`-triggered menu each one pops up (see [`suggestions`]) has to appear
//! on the keystroke that opens it, which rules out fetching the list over HTTP for the
//! web UI; and having the browser filter a list the terminal filters differently is how
//! the two would drift on which commands exist. What is *not* here is carrying a command
//! out — that needs a session store and an event channel, neither of which compiles for
//! `wasm32-unknown-unknown` (see `cli::commands::execute`).

/// What a line resolved to, once [`parse`] has had a look at it.
///
/// # Two kinds of command
///
/// Most are pure side effects and are carried out by `cli::commands::execute`, which is
/// why none of them reaches the model or takes the turn lock. [`Self::Resume`] and
/// [`Self::Discard`] are the exceptions: they act on a *turn*, so each front-end
/// dispatches them itself rather than routing them through `execute` — the same
/// arrangement [`Self::Exit`] already had. They are listed here regardless, because the
/// table is also what `/help` and the `/` menu are built from, and a command nobody can
/// discover may as well not exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Command {
  /// Show [`help_text`].
  Help,
  /// Clear this session's stored history.
  Reset,
  /// Carry on a turn that stopped waiting for an approval, re-raising whatever it is
  /// waiting on. Drives a turn, so each front-end handles it itself.
  Resume,
  /// Give up on a turn that stopped waiting for an approval, committing what it had
  /// already done so the conversation can carry on. Also front-end dispatched.
  Discard,
  /// Leave the chat. Terminal-only in effect — a browser tab has no process to end, so
  /// the web front-end reports it as unsupported rather than pretending.
  Exit,
}

/// One row of the table `/help` prints.
pub struct CommandSpec {
  /// What `/help` shows. The first entry is the canonical spelling.
  pub aliases: &'static [&'static str],
  pub summary: &'static str,
  pub command: Command,
  /// Whether a browser tab can run it (see [`Command::Exit`]).
  pub available_on_web: bool,
}

/// Every in-chat command. The single source of truth for both dispatch and `/help`.
pub const COMMANDS: &[CommandSpec] = &[
  CommandSpec {
    aliases: &["/help", "/?", "/commands"],
    summary: "显示可用命令",
    command: Command::Help,
    available_on_web: true,
  },
  CommandSpec {
    aliases: &["/reset", "/clear"],
    summary: "清空当前会话的历史记录",
    command: Command::Reset,
    available_on_web: true,
  },
  CommandSpec {
    aliases: &["/resume", "/continue"],
    summary: "继续被暂停的一轮（会重新询问审批）",
    command: Command::Resume,
    available_on_web: true,
  },
  CommandSpec {
    aliases: &["/discard"],
    summary: "放弃被暂停的一轮，保留已完成的部分",
    command: Command::Discard,
    available_on_web: true,
  },
  CommandSpec {
    aliases: &["exit", "quit", ":q", "/exit"],
    summary: "退出（仅命令行；浏览器请直接关闭标签页）",
    command: Command::Exit,
    available_on_web: false,
  },
];

/// Resolve `line` to a command, or `None` if it is an ordinary message for the model.
///
/// Case-insensitive and whitespace-tolerant, matching how the aliases read to a human;
/// anything with arguments is deliberately not a command, so a message that merely starts
/// with a command word ("exit the loop early, please") still reaches the model.
pub fn parse(line: &str) -> Option<Command> {
  let normalized = line.trim().to_ascii_lowercase();
  COMMANDS
    .iter()
    .find(|spec| spec.aliases.contains(&normalized.as_str()))
    .map(|spec| spec.command)
}

/// The `/help` listing, aligned into columns.
///
/// `web` drops the commands a browser cannot run, so a tab is never shown an option that
/// would only answer "unsupported".
pub fn help_text(web: bool) -> String {
  let rows: Vec<(String, &str)> = COMMANDS
    .iter()
    .filter(|spec| spec.available_on_web || !web)
    .map(|spec| (spec.aliases.join(", "), spec.summary))
    .collect();

  let width = rows.iter().map(|(names, _)| names.len()).max().unwrap_or(0);
  let mut out = String::from("可用命令:\n");
  for (names, summary) in rows {
    out.push_str(&format!("  {names:<width$}  {summary}\n"));
  }
  out
}

/// One-line reminder for the startup banner, naming only the commands worth leading with.
pub fn hint() -> String {
  format!(
    "输入 {} 查看可用命令",
    COMMANDS
      .iter()
      .find(|spec| spec.command == Command::Help)
      .map(|spec| spec.aliases[0])
      .unwrap_or("/help")
  )
}

/// The `/`-prefixed word `line` is in the middle of typing, if that is what it is doing.
///
/// This is the whole trigger condition for the command menu, shared so that the terminal
/// and the browser cannot disagree about when one is open. `None` means "not typing a
/// command", and the menu stays shut:
///
/// - Anything not starting with `/` (a `/` mid-word is a path, not a command — `"see
///   src/main.rs"` must not open a menu).
/// - A finished word: `"/help "` has moved on to what would be an argument, and
///   `"/help foo"` is an ordinary message, since [`parse`] rejects arguments outright.
///
/// Leading whitespace is tolerated to match [`parse`], and the returned token is a slice
/// of `line`, so a caller that needs to replace it (the terminal's completer does) can
/// recover its offset as `line.len() - token.len()`.
pub fn command_token(line: &str) -> Option<&str> {
  let token = line.trim_start();
  if !token.starts_with('/') || token.ends_with(char::is_whitespace) {
    return None;
  }
  // Any interior whitespace means the word is finished and something follows it.
  (!token.contains(char::is_whitespace)).then_some(token)
}

/// One row of the `/`-triggered menu: an alias the user can pick, and what it does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommandSuggestion {
  /// The exact text picking this row should leave in the input — always a prefix match
  /// on what was typed, so the menu never replaces a line with something unrelated to it.
  pub alias: &'static str,
  pub summary: &'static str,
  pub command: Command,
}

/// Every alias `token` could still be extended into, for the menu to list.
///
/// Aliases are listed individually rather than one row per [`Command`]: `/clear` and
/// `/reset` are the same command, but a menu that answered `/cl` with a row reading
/// `/reset` would be offering to replace what was typed with text that does not contain
/// it. Every row here starts with what the user has already typed, which is also what
/// makes picking one safe to do by simple replacement.
///
/// `web` filters exactly as [`help_text`] does, so a tab is never offered a command it
/// would only be told it cannot run.
pub fn suggestions(token: &str, web: bool) -> Vec<CommandSuggestion> {
  let needle = token.trim().to_ascii_lowercase();
  let mut matches = Vec::new();
  for spec in COMMANDS {
    if web && !spec.available_on_web {
      continue;
    }
    for alias in spec.aliases {
      if alias.starts_with(&needle) {
        matches.push(CommandSuggestion {
          alias,
          summary: spec.summary,
          command: spec.command,
        });
      }
    }
  }
  matches
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn every_alias_resolves_to_its_command() {
    for spec in COMMANDS {
      for alias in spec.aliases {
        assert_eq!(
          parse(alias),
          Some(spec.command),
          "{alias} should resolve to {:?}",
          spec.command
        );
      }
    }
  }

  #[test]
  fn matching_ignores_case_and_surrounding_space() {
    assert_eq!(parse("  /RESET  "), Some(Command::Reset));
    assert_eq!(parse("QUIT"), Some(Command::Exit));
  }

  #[test]
  fn an_ordinary_message_is_not_a_command() {
    for line in ["hello", "reset the counter", "/resetx", "what is /help?"] {
      assert_eq!(parse(line), None, "{line} should reach the model");
    }
  }

  /// Aliases have to be unique, or `parse` would silently pick whichever came first.
  #[test]
  fn no_alias_is_claimed_twice() {
    let mut seen = Vec::new();
    for spec in COMMANDS {
      for alias in spec.aliases {
        assert!(!seen.contains(alias), "{alias} is claimed by two commands");
        seen.push(alias);
      }
    }
  }

  #[test]
  fn help_lists_every_command_it_offers() {
    let text = help_text(false);
    for spec in COMMANDS {
      assert!(
        text.contains(spec.aliases[0]),
        "{} is missing from /help",
        spec.aliases[0]
      );
      assert!(text.contains(spec.summary));
    }
  }

  #[test]
  fn help_hides_terminal_only_commands_from_the_web() {
    let text = help_text(true);
    assert!(text.contains("/reset"));
    assert!(
      !text.contains(":q"),
      "a browser tab cannot exit the process, so it should not be offered"
    );
  }

  #[test]
  fn the_hint_points_at_help() {
    assert!(hint().contains("/help"));
  }

  #[test]
  fn a_bare_slash_is_a_command_token() {
    assert_eq!(command_token("/"), Some("/"));
    assert_eq!(command_token("/he"), Some("/he"));
    assert_eq!(command_token("  /he"), Some("/he"));
  }

  /// The menu must not open over a path, which is the common way a `/` gets typed in an
  /// ordinary message.
  #[test]
  fn a_slash_inside_a_word_is_not_a_command_token() {
    for line in ["see src/main.rs", "src/", "a /help"] {
      assert_eq!(command_token(line), None, "{line} should not open a menu");
    }
  }

  /// Once the word is finished the menu has nothing left to offer: a command takes no
  /// arguments, so anything past the alias makes the line an ordinary message.
  #[test]
  fn a_finished_word_is_not_a_command_token() {
    assert_eq!(command_token("/help "), None);
    assert_eq!(command_token("/help me"), None);
    assert_eq!(command_token(""), None);
  }

  /// The offset a caller replacing the token needs (the terminal's completer) has to be
  /// recoverable by length alone.
  #[test]
  fn a_token_is_a_suffix_of_the_line_it_came_from() {
    let line = "   /re";
    let token = command_token(line).expect("a command token");
    assert_eq!(&line[line.len() - token.len()..], token);
  }

  #[test]
  fn a_bare_slash_suggests_every_command() {
    let aliases: Vec<_> = suggestions("/", false)
      .into_iter()
      .map(|s| s.alias)
      .collect();
    // Only the `/`-spelled aliases: the menu is opened by typing `/`, so a row that
    // does not start with one could never be reached by extending what was typed.
    assert_eq!(
      aliases,
      vec![
        "/help",
        "/?",
        "/commands",
        "/reset",
        "/clear",
        "/resume",
        "/continue",
        "/discard",
        "/exit"
      ]
    );
  }

  /// Every row must contain what was typed, or picking it would replace the line with
  /// unrelated text (see [`suggestions`]' docs).
  #[test]
  fn every_suggestion_extends_what_was_typed() {
    for token in ["/", "/c", "/re", "/help"] {
      for suggestion in suggestions(token, false) {
        assert!(
          suggestion.alias.starts_with(token),
          "{} does not extend {token}",
          suggestion.alias
        );
      }
    }
  }

  /// `/cl` reaching `/reset`'s row by its `/clear` alias is the case that motivates
  /// listing aliases separately rather than one row per command.
  #[test]
  fn an_alias_is_suggested_under_its_own_spelling() {
    let matches = suggestions("/cl", false);
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].alias, "/clear");
    assert_eq!(matches[0].command, Command::Reset);
  }

  #[test]
  fn suggestions_ignore_case() {
    assert_eq!(suggestions("/HE", false).len(), 1);
  }

  #[test]
  fn the_web_is_not_offered_terminal_only_commands() {
    let aliases: Vec<_> = suggestions("/", true)
      .into_iter()
      .map(|s| s.alias)
      .collect();
    assert!(!aliases.contains(&"/exit"));
    assert!(aliases.contains(&"/help"));
  }

  #[test]
  fn an_unknown_prefix_suggests_nothing() {
    assert!(suggestions("/zzz", false).is_empty());
  }

  /// What the menu offers and what `parse` accepts have to be the same set, or the menu
  /// could complete a line into something that then fails to resolve.
  #[test]
  fn every_suggestion_parses_back_to_its_command() {
    for suggestion in suggestions("/", false) {
      assert_eq!(parse(suggestion.alias), Some(suggestion.command));
    }
  }
}
