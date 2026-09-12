//! The `/`-triggered command menu at the `You>` prompt.
//!
//! Typing `/` pops up the list of in-chat commands ([`shared::commands`]) to pick from,
//! rather than requiring that they be remembered or that `/help` be run first. `Tab`
//! opens the same menu, `↑`/`↓` walk it, `Enter` picks the highlighted row — and the
//! browser's composer offers the same thing for the same table (see `web-ui`'s
//! `CommandMenu`).
//!
//! Two pieces make that work, and they are deliberately split: [`CommandCompleter`]
//! decides *what* to offer, while [`bind_menu_keys`] decides *when* to ask. The `/`
//! binding fires on every `/` typed anywhere in a line — including the ones in
//! `"see src/main.rs"` — so "should a menu be open at all?" cannot live in the
//! keybinding. It lives in the completer, which returns nothing outside a command
//! context ([`shared::commands::command_token`]); a menu with no rows does not show,
//! which is what keeps a `/` in ordinary prose from flashing a popup.

use reedline::{
  Completer, CompletionResult, DescriptionMode, EditCommand, IdeMenu, KeyCode, KeyModifiers,
  Keybindings, MenuBuilder, ReedlineEvent, ReedlineMenu, Span, Suggestion,
};

/// The name [`bind_menu_keys`]' events use to address the menu built by
/// [`command_menu`]. `reedline` pairs the two up by string at runtime, so a typo here is
/// a menu that silently never opens — hence one constant rather than three literals.
const MENU_NAME: &str = "command_menu";

/// Offers in-chat commands at the `You>` prompt, and nothing else.
///
/// Notably not a file/path completer: every other `/` a user types here belongs to
/// ordinary prose or a path inside a message to the model, and completing those would
/// mean guessing at which of the two a given line is. Commands are the one thing at this
/// prompt with a closed, known set of valid answers.
pub struct CommandCompleter;

impl Completer for CommandCompleter {
  /// `reedline` models a completer as possibly asynchronous, so the return type
  /// distinguishes authoritative results from in-flight ones. This completer answers
  /// from a compile-time table with no I/O, so every answer — including "nothing to
  /// offer" — is [`CompletionResult::fresh`]; it is never `Pending`/`Stale`, and an
  /// empty `Fresh` is exactly the "no menu" signal the module docs describe.
  fn complete(&mut self, line: &str, pos: usize) -> CompletionResult {
    // Only what is left of the cursor: with the cursor mid-line, the text after it is
    // not part of the word being completed, and `pos` is where the replacement has to
    // end. Snapped to a `char` boundary first — `pos` is a byte offset, and slicing a
    // multi-byte character in half (an easy thing to do at this prompt, where a line is
    // as likely to be Chinese as not) would panic rather than fail to complete.
    if !line.is_char_boundary(pos) {
      return CompletionResult::fresh(Vec::new());
    }
    let head = &line[..pos];

    // The guard the `/` keybinding cannot apply for itself (see the module docs): no
    // command context, no rows, and so no menu.
    let Some(token) = shared::commands::command_token(head) else {
      return CompletionResult::fresh(Vec::new());
    };
    // `token` is `head` minus leading whitespace, so its length locates it.
    let span = Span::new(pos - token.len(), pos);

    let rows: Vec<Suggestion> = shared::commands::suggestions(token, false)
      .into_iter()
      .map(|suggestion| Suggestion {
        value: suggestion.alias.to_owned(),
        description: Some(suggestion.summary.to_owned()),
        span,
        // No trailing space: a command takes no arguments (`shared::commands::parse`
        // rejects a line with any), so the completed line is already exactly what should
        // be submitted. Appending one would also immediately close the menu's own
        // trigger condition, leaving `"/help "` — which `parse` no longer resolves.
        append_whitespace: false,
        // Styling, a display string that differs from what gets inserted, and
        // match-highlight offsets are all left to `reedline`'s own defaults.
        ..Suggestion::default()
      })
      .collect();
    CompletionResult::fresh(rows)
  }
}

/// The popup [`CommandCompleter`]'s rows are drawn in.
///
/// [`IdeMenu`] rather than `reedline`'s columnar default because each row here has a
/// description worth reading — `/clear` and `/reset` are indistinguishable by name alone
/// — and this is the menu style that shows one per row next to its command.
/// [`DescriptionMode::PreferRight`] keeps those descriptions beside the commands when
/// the terminal is wide enough and moves them left when it is not, rather than truncating.
pub fn command_menu() -> ReedlineMenu {
  ReedlineMenu::EngineCompleter(Box::new(
    IdeMenu::default()
      .with_name(MENU_NAME)
      .with_description_mode(DescriptionMode::PreferRight),
  ))
}

/// Binds `/` and `Tab` to open the command menu.
///
/// Applied to *insert*-mode bindings only when `vi` mode is on: `/` in vi's normal mode
/// is its own search command, and a menu is not what someone pressing it there is asking
/// for. Nothing needs to be bound for `Enter`, `↑`/`↓` or `Esc` — `reedline` already
/// routes those to whichever menu is active (`Enter` accepting the highlighted row is
/// its built-in behavior), and rebinding them would break them everywhere else.
pub fn bind_menu_keys(keybindings: &mut Keybindings) {
  // Insert the `/` first, *then* open the menu: the completer needs the character in the
  // buffer to have a token to match on, and the menu has to be opened after it to see it.
  keybindings.add_binding(
    KeyModifiers::NONE,
    KeyCode::Char('/'),
    ReedlineEvent::Multiple(vec![
      ReedlineEvent::Edit(vec![EditCommand::InsertChar('/')]),
      ReedlineEvent::Menu(MENU_NAME.to_owned()),
    ]),
  );
  // `Tab` is where a shell user looks for completion regardless, and it is the way back
  // to a menu dismissed with `Esc` without deleting and retyping the `/`.
  // `UntilFound` keeps `Tab`'s usual second job: once the menu is already open, step
  // through it instead of trying to reopen it.
  keybindings.add_binding(
    KeyModifiers::NONE,
    KeyCode::Tab,
    ReedlineEvent::UntilFound(vec![
      ReedlineEvent::Menu(MENU_NAME.to_owned()),
      ReedlineEvent::MenuNext,
    ]),
  );
}

#[cfg(test)]
mod tests {
  use super::*;

  /// The rows a completion carries, which is all these tests care about — the
  /// `Fresh`/`Stale`/`Pending` distinction is a no-op for a synchronous completer (see
  /// [`CommandCompleter::complete`]).
  fn rows(line: &str, pos: usize) -> Vec<Suggestion> {
    CommandCompleter.complete(line, pos).suggestions().to_vec()
  }

  fn complete(line: &str) -> Vec<Suggestion> {
    rows(line, line.len())
  }

  #[test]
  fn a_bare_slash_offers_every_command() {
    let values: Vec<_> = complete("/").into_iter().map(|s| s.value).collect();
    assert_eq!(
      values,
      vec!["/help", "/?", "/commands", "/reset", "/clear", "/exit"]
    );
  }

  #[test]
  fn every_row_carries_its_description() {
    for suggestion in complete("/") {
      assert!(
        suggestion.description.is_some_and(|d| !d.is_empty()),
        "{} has nothing explaining it",
        suggestion.value
      );
    }
  }

  /// The case the `/` keybinding cannot rule out for itself: a path in ordinary prose
  /// must not flash a popup (see the module docs).
  #[test]
  fn a_slash_in_ordinary_prose_offers_nothing() {
    for line in ["see src/main.rs", "and/or", "/help me"] {
      assert!(complete(line).is_empty(), "{line} should not open the menu");
    }
  }

  /// Replacing the span has to leave exactly the alias — no duplicated `/`, and the
  /// leading whitespace `command_token` tolerated left alone.
  #[test]
  fn accepting_a_row_replaces_only_the_typed_token() {
    let line = "  /re";
    let suggestion = complete(line).into_iter().next().expect("a suggestion");
    let mut completed = line.to_owned();
    completed.replace_range(
      suggestion.span.start..suggestion.span.end,
      &suggestion.value,
    );
    assert_eq!(completed, "  /reset");
    assert_eq!(
      shared::commands::parse(&completed),
      Some(shared::commands::Command::Reset)
    );
  }

  /// The completion must be submittable as-is; a trailing space would stop `parse` from
  /// resolving it (see `append_whitespace`'s comment).
  #[test]
  fn a_completed_command_needs_no_trailing_space() {
    for suggestion in complete("/") {
      assert!(!suggestion.append_whitespace, "{}", suggestion.value);
    }
  }

  /// Only the text left of the cursor is being completed.
  #[test]
  fn text_after_the_cursor_is_not_part_of_the_token() {
    let line = "/re then more";
    // Cursor just after `/re`.
    let suggestions = rows(line, 3);
    assert_eq!(suggestions.len(), 1);
    assert_eq!(suggestions[0].value, "/reset");
  }

  /// `pos` is a byte offset, and this prompt takes Chinese as readily as ASCII, so a
  /// cursor position that is not a `char` boundary must not panic.
  #[test]
  fn a_position_inside_a_multibyte_character_is_not_a_panic() {
    let line = "你好";
    assert!(rows(line, 1).is_empty());
  }

  #[test]
  fn an_empty_line_offers_nothing() {
    assert!(complete("").is_empty());
  }
}
