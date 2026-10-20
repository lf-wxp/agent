//! [`Compaction`] — rewrite spent tool results in place, the cheapest stage of
//! [`super::ContextOptimizer`].
//!
//! Tool output is where context bloat overwhelmingly lives: a file read or a search hit
//! is bulky once and then re-sent verbatim on every subsequent round, long after the
//! model has extracted what it needed. This stage replaces those spent payloads with a
//! one-line note saying what was fetched and how to fetch it again.
//!
//! Three properties make it the right thing to try first:
//!
//! - **Free.** No model call, no tokenizer pass — just a walk over the items.
//! - **Structure-preserving.** Only the text inside items is rewritten — a result's
//!   `content`, and the bulky string arguments described below; no item is added or
//!   removed, so tool calls keep their results and every index into the conversation
//!   stays valid (which [`super::summarization`] relies on).
//! - **Recoverable.** The replacement text tells the model how to get the real output
//!   back, so a rewrite that turns out to have been premature costs one extra tool call
//!   rather than the answer.
//!
//! The tail is exempt: [`Compaction::keep_recent`] items at the end are left whole, so
//! the model still sees the full output of what it just did.
//!
//! # Bulky *arguments* of calls that were made
//!
//! Results are not the only large thing in a transcript. `write_file`'s `content` and
//! `edit_file`'s `old_string` / `new_string` are the file, or a large part of it, and
//! stay in the request for the rest of the conversation. Those are trimmed here as well
//! ([`BULKY_ARGUMENTS`]), once the call is no longer recent. Nothing is lost that matters
//! for the model's next step: the call's result already says what happened, and the
//! file on disk is the record of its content. The *keys* stay, so the call still
//! reads as a call to that tool with those parameters (a provider validating the history
//! against the tool's schema sees every required field), and each trimmed value says how
//! much was cut.
//!
//! This is the opposite exemption from the one below: a trimmed argument is never
//! re-fetched, so it applies to tools with side effects — the call has already happened.
//!
//! Only tools that have been *registered* are recognized. An unknown tool is left alone
//! rather than summarized by guesswork — a tool with side effects, or one whose result
//! cannot be reproduced, must not be told to re-run. [`Compaction::new`] pre-registers
//! the four built-in tools whose output is bulky, side-effect-free and cheap to fetch
//! again (`read_file`, `list_files`, `grep`, `web_search`); [`Compaction::with_tool`]
//! extends that to a caller's own tools, including ones reached over MCP.
//!
//! The remaining built-ins are left out on purpose, for three different reasons. Worth
//! spelling out, because "was never considered" and "was considered and rejected" look
//! identical from the registry:
//!
//! - [`crate::tools::file_delete`], [`crate::tools::file_upzip`],
//!   [`crate::tools::file_write`], [`crate::tools::file_edit`], [`crate::tools::shell`]
//!   — side effects. Re-running deletes, overwrites or executes again, so the note would
//!   be an instruction to redo work, not to re-fetch it.
//! - [`crate::tools::read_image`] — safe and reproducible, but not *cheap*: recovering
//!   the dropped answer costs another vision model call, and the new answer need not
//!   match the old one.
//! - [`crate::tools::calculator`] — its result is already shorter than any note
//!   describing it.
//! - [`crate::tools::plan`] (`update_plan`) — its result is a short echo of the call's own
//!   arguments, and the usual note ("call it again to get this back") would be an
//!   instruction to *change* the plan, since every call replaces it.

use std::{collections::HashMap, sync::Arc};

use serde_json::Value;

use crate::{
  agent::{ContentItem, llm_request::LlmRequest},
  tools::{file_edit, file_list, file_read, file_write, grep, web_search},
};

/// Arguments of already-made calls that are bulky enough to trim, per tool.
///
/// Only the named keys: a path or a flag is small and is what makes the call
/// recognizable. Keyed by tool name rather than by size alone so that an unregistered
/// tool's arguments are never touched, for the same reason its results are not (see the
/// module docs).
const BULKY_ARGUMENTS: &[(&str, &[&str])] = &[
  (file_write::NAME, &["content"]),
  (file_edit::NAME, &["old_string", "new_string"]),
];

/// A bulky argument is only replaced when it is longer than this many chars: the stand-in
/// text has a length of its own, so trimming something already shorter would grow it.
const MIN_TRIMMED_ARGUMENT_CHARS: usize = 256;

/// Builds the note that stands in for one tool's spent result, given the arguments the
/// call was made with.
///
/// Only the *arguments* are available on purpose. A describer is free to echo them,
/// because they were produced by the model itself and are about to be shown back to it.
/// The result payload is deliberately out of reach: it is the part that can carry
/// arbitrary fetched or user-supplied text, and interpolating it into a note the agent
/// reads as recent history is how a compaction pass would turn into an injection vector.
pub type Describe = Arc<dyn Fn(&Value) -> String + Send + Sync>;

/// Chars of a single argument echoed into a note.
///
/// Arguments come from the model, and a model routinely produces enormous ones — a
/// `query` built by pasting in a page it just fetched, a path assembled from a long
/// listing. A note is only worth writing if it is *smaller* than the payload it replaces,
/// and an uncapped echo has no such guarantee: it can make the request grow, spending the
/// whole pipeline to end up worse off. Generous enough that a real file path or search
/// query survives intact, so the note keeps the one fact it exists to carry.
const MAX_ARGUMENT_CHARS: usize = 200;

/// Value of `arguments[key]` as a string, or `"unknown"` when the call is not available
/// (a truncated transcript) or the key is missing.
///
/// Truncated to [`MAX_ARGUMENT_CHARS`], so a note can never be larger than the result it
/// replaces by more than a bounded constant.
///
/// Exposed so a caller writing a [`Describe`] for its own tool gets the same
/// "never panic, never guess, never grow without bound" handling the built-ins use.
pub fn argument(arguments: &Value, key: &str) -> String {
  preview(
    arguments
      .get(key)
      .and_then(Value::as_str)
      .unwrap_or("unknown"),
  )
}

/// `text` capped at [`MAX_ARGUMENT_CHARS`], marked when anything was cut.
///
/// Counts `char`s rather than bytes, so a multi-byte path or a Chinese query is cut at a
/// character boundary instead of panicking on one.
fn preview(text: &str) -> String {
  let mut preview: String = text.chars().take(MAX_ARGUMENT_CHARS).collect();
  if preview.len() < text.len() {
    preview.push('…');
  }
  preview
}

fn describe_file_read(arguments: &Value) -> String {
  format!(
    "File '{}' was already read. Call {} again if you need it.",
    argument(arguments, "file_path"),
    file_read::NAME
  )
}

fn describe_web_search(arguments: &Value) -> String {
  format!(
    "Search results for '{}' were already processed. Call {} again if you need them.",
    argument(arguments, "query"),
    web_search::NAME
  )
}

/// Unlike the others this one does not go through [`argument`]: `path` is optional, and a
/// model that means "look around from here" routinely omits it. Rendering that as
/// "unknown" would throw away the single fact the note exists to carry, so the omission
/// resolves to the same default the executor itself applied.
fn describe_file_list(arguments: &Value) -> String {
  let path = preview(
    arguments
      .get("path")
      .and_then(Value::as_str)
      .unwrap_or(file_list::DEFAULT_PATH),
  );

  format!(
    "Directory '{path}' was already listed. Call {} again if you need it.",
    file_list::NAME
  )
}

/// Same reasoning as [`describe_file_list`] for the optional `path`.
///
/// Also names the options that change *which* lines match (`include`,
/// `case_insensitive`): a model that re-runs the search from this note should get the
/// same matches back, not a differently filtered set.
fn describe_grep(arguments: &Value) -> String {
  let path = preview(
    arguments
      .get("path")
      .and_then(Value::as_str)
      .unwrap_or(grep::DEFAULT_PATH),
  );

  let mut options = String::new();
  if let Some(include) = arguments
    .get("include")
    .and_then(Value::as_str)
    .filter(|include| !include.is_empty())
  {
    options.push_str(&format!(", only files matching '{}'", preview(include)));
  }
  if arguments
    .get("case_insensitive")
    .and_then(Value::as_bool)
    .unwrap_or(false)
  {
    options.push_str(", case-insensitive");
  }

  format!(
    "Search for '{}' in '{path}'{options} was already run. Call {} again with the same \
     arguments if you need the matches.",
    argument(arguments, "pattern"),
    grep::NAME
  )
}

#[derive(Clone)]
pub struct Compaction {
  /// Trailing items left untouched — recent work the model is still reasoning about.
  keep_recent: usize,
  /// Tools whose spent results may be rewritten, and how to describe each.
  describers: HashMap<String, Describe>,
}

impl std::fmt::Debug for Compaction {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let mut names: Vec<&String> = self.describers.keys().collect();
    names.sort();
    f.debug_struct("Compaction")
      .field("keep_recent", &self.keep_recent)
      .field("describers", &names)
      .finish()
  }
}

impl Compaction {
  /// Rewrite spent results for the built-in reproducible tools (`read_file`,
  /// `list_files`, `grep`, `web_search`), leaving the last `keep_recent` items whole.
  pub fn new(keep_recent: usize) -> Self {
    Self::empty(keep_recent)
      .with_tool(file_read::NAME, describe_file_read)
      .with_tool(file_list::NAME, describe_file_list)
      .with_tool(grep::NAME, describe_grep)
      .with_tool(web_search::NAME, describe_web_search)
  }

  /// Recognize nothing at all. Useful as a base for a caller that wants only its own
  /// tools compacted, or to deliberately exempt a built-in.
  pub fn empty(keep_recent: usize) -> Self {
    Self {
      keep_recent,
      describers: HashMap::new(),
    }
  }

  /// Allow `name`'s spent results to be replaced by `describe`.
  ///
  /// Register a tool only when re-running it is both **safe** (no side effects) and
  /// **sufficient** (the same call reproduces what was dropped) — the note tells the
  /// model it can simply call the tool again.
  #[must_use]
  pub fn with_tool(
    mut self,
    name: impl Into<String>,
    describe: impl Fn(&Value) -> String + Send + Sync + 'static,
  ) -> Self {
    self.describers.insert(name.into(), Arc::new(describe));
    self
  }

  /// Trailing items left untouched.
  pub fn keep_recent(&self) -> usize {
    self.keep_recent
  }

  /// Rewrite every recognized, non-recent tool result in `request` in place.
  ///
  /// Returns the indices that were rewritten, so a caller tracking per-item token costs
  /// ([`super::tokens::Ledger`]) can re-measure only those instead of the whole request.
  ///
  /// Two passes rather than one. A result carries its tool's name but not the arguments
  /// it was called with, and the replacement text needs them (which file? which query?),
  /// so the calls have to be collected on the way past. Doing that during a *mutable*
  /// walk would mean cloning every tool call's arguments — including those of tools that
  /// are not registered and whose results will not be touched — which on a long
  /// conversation is an O(n) deep copy of `Value`s per round, thrown away immediately.
  /// Planning against an immutable borrow lets the map hold borrowed arguments instead,
  /// and only the (bounded) replacement strings are materialized.
  pub fn apply(&self, request: &mut LlmRequest) -> Vec<usize> {
    let protect_from = request.contents.len().saturating_sub(self.keep_recent);
    // Calls always precede their results, so a single forward pass suffices.
    let mut call_args: HashMap<&str, &Value> = HashMap::new();
    let mut planned: Vec<(usize, String)> = Vec::new();
    let mut trimmed_calls: Vec<usize> = Vec::new();

    for (index, item) in request.contents.iter().enumerate() {
      match item {
        ContentItem::ToolCall {
          tool_call_id,
          name,
          arguments,
        } => {
          call_args.insert(tool_call_id.as_str(), arguments);
          // Trimmed whether or not a result is already paired with this call: an
          // unpaired (cancelled or interrupted) write/edit is just as much "already
          // sent" as a completed one, and its payload is no longer needed either way.
          if index < protect_from && has_bulky_arguments(name, arguments) {
            trimmed_calls.push(index);
          }
        }
        ContentItem::ToolResult {
          tool_call_id,
          name,
          content,
          ..
        } if index < protect_from => {
          if let Some(describe) = self.describers.get(name.as_str()) {
            // A missing call means a truncated or malformed transcript; `argument`
            // renders that as "unknown" rather than losing the rewrite entirely.
            let arguments = call_args.get(tool_call_id.as_str()).copied();
            let replacement = describe(arguments.unwrap_or(&Value::Null));
            if *content != replacement {
              planned.push((index, replacement));
            }
          }
        }
        // A protected (recent) result, or a plain message: nothing to do either way.
        _ => {}
      }
    }

    let mut rewritten = Vec::with_capacity(planned.len() + trimmed_calls.len());
    for (index, replacement) in planned {
      if let Some(ContentItem::ToolResult { content, .. }) = request.contents.get_mut(index) {
        *content = replacement;
        rewritten.push(index);
      }
    }
    for index in trimmed_calls {
      if let Some(ContentItem::ToolCall {
        name, arguments, ..
      }) = request.contents.get_mut(index)
      {
        trim_bulky_arguments(name, arguments);
        rewritten.push(index);
      }
    }

    rewritten
  }
}

/// Whether `arguments` of a call to `tool` hold something [`trim_bulky_arguments`] would
/// shrink.
fn has_bulky_arguments(tool: &str, arguments: &Value) -> bool {
  bulky_keys(tool).iter().any(|key| {
    arguments
      .get(*key)
      .and_then(Value::as_str)
      .is_some_and(|text| text.chars().count() > MIN_TRIMMED_ARGUMENT_CHARS)
  })
}

fn bulky_keys(tool: &str) -> &'static [&'static str] {
  BULKY_ARGUMENTS
    .iter()
    .find(|(name, _)| *name == tool)
    .map_or(&[], |(_, keys)| *keys)
}

/// Replace each bulky string argument of `tool` with a note of how much it held.
///
/// Idempotent: the note is shorter than [`MIN_TRIMMED_ARGUMENT_CHARS`], so a second pass
/// finds nothing left to do.
fn trim_bulky_arguments(tool: &str, arguments: &mut Value) {
  for key in bulky_keys(tool) {
    let Some(slot) = arguments.get_mut(*key) else {
      continue;
    };
    let Some(text) = slot.as_str() else {
      continue;
    };
    let chars = text.chars().count();
    if chars > MIN_TRIMMED_ARGUMENT_CHARS {
      let lines = text.lines().count();
      *slot = Value::String(format!(
        "[{chars} chars, {lines} lines, omitted from history]"
      ));
    }
  }
}

#[cfg(test)]
mod tests {
  use serde_json::json;

  use super::*;
  use crate::{
    agent::ToolResultStatus,
    callback::context_optimizer::tokens,
    tools::{calculator, file_delete, file_edit, file_upzip, file_write, plan, read_image, shell},
  };

  fn call(id: &str, name: &str, arguments: Value) -> ContentItem {
    ContentItem::ToolCall {
      tool_call_id: id.to_owned(),
      name: name.to_owned(),
      arguments,
    }
  }

  fn result(id: &str, name: &str, content: &str) -> ContentItem {
    ContentItem::ToolResult {
      tool_call_id: id.to_owned(),
      name: name.to_owned(),
      status: ToolResultStatus::Success,
      content: content.to_owned(),
    }
  }

  fn request_of(contents: Vec<ContentItem>) -> LlmRequest {
    LlmRequest {
      instructions: Vec::new(),
      contents,
    }
  }

  fn result_contents(request: &LlmRequest) -> Vec<&str> {
    request
      .contents
      .iter()
      .filter_map(|item| match item {
        ContentItem::ToolResult { content, .. } => Some(content.as_str()),
        _ => None,
      })
      .collect()
  }

  /// One read per round, plus a trailing message so the protected window is easy to
  /// reason about.
  fn reads(rounds: usize) -> Vec<ContentItem> {
    let mut contents = Vec::new();
    for i in 0..rounds {
      contents.push(call(
        &format!("call_{i}"),
        file_read::NAME,
        json!({ "file_path": format!("src/module_{i}.rs") }),
      ));
      contents.push(result(
        &format!("call_{i}"),
        file_read::NAME,
        &"line ".repeat(300),
      ));
    }
    contents
  }

  #[test]
  fn a_spent_read_is_replaced_by_a_note_naming_the_file() {
    let mut request = request_of(reads(4));
    Compaction::new(0).apply(&mut request);

    let first = result_contents(&request)[0];
    assert!(first.contains("src/module_0.rs"), "got: {first}");
    assert!(
      first.contains(file_read::NAME),
      "the note must say how to get the real output back, got: {first}"
    );
  }

  #[test]
  fn a_spent_search_is_replaced_by_a_note_naming_the_query() {
    let mut request = request_of(vec![
      call(
        "s0",
        web_search::NAME,
        json!({ "query": "rust async book" }),
      ),
      result("s0", web_search::NAME, &"hit ".repeat(300)),
    ]);
    Compaction::new(0).apply(&mut request);

    let first = result_contents(&request)[0];
    assert!(first.contains("rust async book"), "got: {first}");
  }

  #[test]
  fn a_spent_listing_is_replaced_by_a_note_naming_the_directory() {
    let mut request = request_of(vec![
      call("l0", file_list::NAME, json!({ "path": "src/agent" })),
      result("l0", file_list::NAME, &"entry\n".repeat(300)),
    ]);
    Compaction::new(0).apply(&mut request);

    let first = result_contents(&request)[0];
    assert!(first.contains("src/agent"), "got: {first}");
    assert!(
      first.contains(file_list::NAME),
      "the note must say how to get the real output back, got: {first}"
    );
  }

  /// `path` is optional, so a call that omits it must still name the directory that was
  /// actually searched — the executor's default — rather than degrading to "unknown".
  #[test]
  fn a_spent_grep_is_replaced_by_a_note_naming_pattern_and_path() {
    let mut request = request_of(vec![
      call("g0", grep::NAME, json!({ "pattern": "fn main" })),
      result("g0", grep::NAME, &"src/main.rs:1: fn main\n".repeat(300)),
    ]);
    Compaction::new(0).apply(&mut request);

    let first = result_contents(&request)[0];
    assert!(first.contains("fn main"), "got: {first}");
    assert!(
      first.contains(&format!("'{}'", grep::DEFAULT_PATH)),
      "got: {first}"
    );
    assert!(first.contains(grep::NAME), "got: {first}");
  }

  /// A note that dropped the filters would send the model back for a different result
  /// set than the one it had.
  #[test]
  fn a_spent_grep_note_keeps_the_options_that_change_the_matches() {
    let mut request = request_of(vec![
      call(
        "g0",
        grep::NAME,
        json!({ "pattern": "todo", "include": "*.rs", "case_insensitive": true }),
      ),
      result("g0", grep::NAME, &"src/main.rs:1: TODO\n".repeat(300)),
    ]);
    Compaction::new(0).apply(&mut request);

    let first = result_contents(&request)[0];
    assert!(first.contains("*.rs"), "got: {first}");
    assert!(first.contains("case-insensitive"), "got: {first}");
  }

  /// `path` is optional, so a call that omits it must still name the directory that was
  /// actually listed — the executor's default — rather than degrading to "unknown".
  #[test]
  fn a_listing_without_an_explicit_path_names_the_default_directory() {
    let mut request = request_of(vec![
      call("l0", file_list::NAME, json!({})),
      result("l0", file_list::NAME, &"entry\n".repeat(300)),
    ]);
    Compaction::new(0).apply(&mut request);

    let first = result_contents(&request)[0];
    assert!(
      first.contains(&format!("'{}'", file_list::DEFAULT_PATH)),
      "got: {first}"
    );
    assert!(!first.contains("unknown"), "got: {first}");
  }

  #[test]
  fn compaction_shrinks_the_request_without_dropping_anything() {
    let mut request = request_of(reads(6));
    let before_items = request.contents.len();
    let before_tokens = tokens::count_request(&request);

    Compaction::new(2).apply(&mut request);

    assert_eq!(
      request.contents.len(),
      before_items,
      "rewriting in place must not add or remove items"
    );
    assert!(tokens::count_request(&request) < before_tokens);
  }

  /// The model has to see the full output of what it just did, or compaction would be
  /// deleting the very result it is about to reason over.
  #[test]
  fn the_recent_tail_is_left_whole() {
    let mut request = request_of(reads(4));
    Compaction::new(2).apply(&mut request);

    let results = result_contents(&request);
    let last = results.last().expect("a result should exist");
    assert!(
      last.starts_with("line "),
      "the newest result must survive verbatim, got: {last}"
    );
  }

  /// An unregistered tool may have side effects or an unreproducible result; telling the
  /// model to "call it again" would be wrong, so it is left alone.
  #[test]
  fn an_unknown_tool_is_left_alone() {
    let mut request = request_of(vec![
      call("d0", "delete_file", json!({ "file_path": "/tmp/x" })),
      result("d0", "delete_file", "deleted /tmp/x"),
      result("orphan", "some_other_tool", "payload"),
    ]);
    Compaction::new(0).apply(&mut request);

    assert_eq!(result_contents(&request), ["deleted /tmp/x", "payload"]);
  }

  /// A model-written file stays in every later request unless its payload is dropped, and
  /// the file on disk already is the record of it.
  #[test]
  fn a_spent_write_has_its_content_trimmed_but_keeps_its_shape() {
    let body = "line\n".repeat(400);
    let mut request = request_of(vec![
      call(
        "w0",
        file_write::NAME,
        json!({ "file_path": "src/a.rs", "content": body }),
      ),
      result(
        "w0",
        file_write::NAME,
        "Created src/a.rs (2000 bytes, 400 lines)",
      ),
    ]);
    let rewritten = Compaction::new(0).apply(&mut request);

    assert_eq!(rewritten, vec![0]);
    let ContentItem::ToolCall { arguments, .. } = &request.contents[0] else {
      panic!("the call must stay a call");
    };
    assert_eq!(arguments["file_path"], "src/a.rs", "the path is kept");
    let content = arguments["content"].as_str().unwrap();
    assert!(content.contains("2000 chars"), "got: {content}");
    assert!(content.contains("400 lines"), "got: {content}");
    // Its result is still the real one: only the payload was dropped.
    assert_eq!(
      result_contents(&request)[0],
      "Created src/a.rs (2000 bytes, 400 lines)"
    );
  }

  #[test]
  fn a_spent_edit_has_both_strings_trimmed() {
    let big = "x".repeat(1000);
    let mut request = request_of(vec![
      call(
        "e0",
        file_edit::NAME,
        json!({ "file_path": "a.rs", "old_string": big, "new_string": big }),
      ),
      result("e0", file_edit::NAME, "Replaced 1 occurrence in a.rs"),
    ]);
    Compaction::new(0).apply(&mut request);

    let ContentItem::ToolCall { arguments, .. } = &request.contents[0] else {
      panic!("the call must stay a call");
    };
    for key in ["old_string", "new_string"] {
      assert!(
        arguments[key].as_str().unwrap().contains("1000 chars"),
        "{key}: {arguments}"
      );
    }
  }

  /// The newest call is the one the model is still acting on.
  #[test]
  fn a_recent_write_keeps_its_content() {
    let body = "line\n".repeat(400);
    let mut request = request_of(vec![call(
      "w0",
      file_write::NAME,
      json!({ "file_path": "a.rs", "content": body.clone() }),
    )]);
    let rewritten = Compaction::new(5).apply(&mut request);

    assert!(rewritten.is_empty());
    let ContentItem::ToolCall { arguments, .. } = &request.contents[0] else {
      panic!("the call must stay a call");
    };
    assert_eq!(arguments["content"], body);
  }

  /// Short arguments stay: the stand-in text would not be shorter.
  #[test]
  fn a_small_write_is_left_alone() {
    let mut request = request_of(vec![
      call(
        "w0",
        file_write::NAME,
        json!({ "file_path": "a.txt", "content": "hello" }),
      ),
      result("w0", file_write::NAME, "Created a.txt"),
    ]);

    assert!(Compaction::new(0).apply(&mut request).is_empty());
  }

  /// Another tool's `content` argument is not this module's to judge.
  #[test]
  fn an_unrelated_tool_keeps_its_arguments() {
    let body = "x".repeat(1000);
    let mut request = request_of(vec![
      call("o0", "some_other_tool", json!({ "content": body.clone() })),
      result("o0", "some_other_tool", "ok"),
    ]);
    Compaction::new(0).apply(&mut request);

    let ContentItem::ToolCall { arguments, .. } = &request.contents[0] else {
      panic!("the call must stay a call");
    };
    assert_eq!(arguments["content"], body);
  }

  #[test]
  fn trimming_arguments_is_idempotent_and_shrinks_the_request() {
    let mut once = request_of(vec![
      call(
        "w0",
        file_write::NAME,
        json!({ "file_path": "a.rs", "content": "line\n".repeat(400) }),
      ),
      result("w0", file_write::NAME, "Created a.rs"),
    ]);
    let before = tokens::count_request(&once);
    Compaction::new(0).apply(&mut once);
    assert!(tokens::count_request(&once) < before);

    let mut twice = request_of(once.contents.clone());
    assert!(Compaction::new(0).apply(&mut twice).is_empty());
    assert_eq!(
      serde_json::to_string(&once.contents).unwrap(),
      serde_json::to_string(&twice.contents).unwrap()
    );
  }

  /// The built-ins this module's docs single out are excluded by decision, not by
  /// oversight. Asserting it keeps someone from "completing" the registry by hand and
  /// quietly telling the model to re-delete a file or re-pay for a vision call.
  #[test]
  fn the_default_registry_excludes_the_unsafe_and_the_expensive() {
    let compaction = Compaction::new(0);
    for name in [
      file_delete::NAME,
      file_upzip::NAME,
      file_write::NAME,
      file_edit::NAME,
      shell::NAME,
      read_image::NAME,
      calculator::NAME,
      plan::NAME,
    ] {
      assert!(
        !compaction.describers.contains_key(name),
        "`{name}` must not be compacted by default"
      );
    }
  }

  /// Running twice must not stack notes on top of notes: the second pass rewrites an
  /// already-rewritten result to the identical text.
  #[test]
  fn applying_twice_is_idempotent() {
    let mut once = request_of(reads(4));
    Compaction::new(1).apply(&mut once);
    let mut twice = request_of(once.contents.clone());
    Compaction::new(1).apply(&mut twice);

    assert_eq!(result_contents(&once), result_contents(&twice));
  }

  /// The reported indices drive a token ledger, so they have to name exactly the items
  /// whose text actually changed — no more (a no-op rewrite is not a change) and no
  /// fewer.
  #[test]
  fn apply_reports_which_items_it_rewrote() {
    let mut request = request_of(reads(4));
    let rewritten = Compaction::new(2).apply(&mut request);

    // 8 items: call/result pairs at 0..8, the last two protected by `keep_recent`.
    assert_eq!(rewritten, vec![1, 3, 5]);

    // A second pass changes nothing, so it must report nothing.
    assert!(
      Compaction::new(2).apply(&mut request).is_empty(),
      "an idempotent pass must not report phantom changes"
    );
  }

  /// A caller's own tool — including one reached over MCP — can opt into compaction
  /// without this module knowing anything about it.
  #[test]
  fn a_registered_custom_tool_is_compacted() {
    let mut request = request_of(vec![
      call("q0", "run_query", json!({ "sql": "select 1" })),
      result("q0", "run_query", &"row ".repeat(300)),
    ]);

    Compaction::empty(0)
      .with_tool("run_query", |arguments| {
        format!("Query '{}' was already run.", argument(arguments, "sql"))
      })
      .apply(&mut request);

    assert_eq!(
      result_contents(&request),
      ["Query 'select 1' was already run."]
    );
  }

  /// `empty` recognizes nothing, so even a built-in is left whole — the escape hatch for
  /// a deployment where re-reading a file is not actually cheap.
  #[test]
  fn an_empty_registry_compacts_nothing() {
    let mut request = request_of(reads(2));
    let before = result_contents(&request)
      .into_iter()
      .map(str::to_owned)
      .collect::<Vec<_>>();

    assert!(Compaction::empty(0).apply(&mut request).is_empty());
    assert_eq!(result_contents(&request), before);
  }

  /// A result whose call is missing (a truncated or malformed transcript) still gets a
  /// usable note rather than a panic.
  #[test]
  fn a_result_without_its_call_falls_back_to_unknown() {
    let mut request = request_of(vec![
      result("ghost", file_read::NAME, &"line ".repeat(300)),
      result("ghost2", file_read::NAME, "tail"),
    ]);
    Compaction::new(1).apply(&mut request);

    assert!(result_contents(&request)[0].contains("unknown"));
  }

  #[test]
  fn an_empty_request_is_handled() {
    let mut request = request_of(Vec::new());
    assert!(Compaction::new(4).apply(&mut request).is_empty());
    assert!(request.contents.is_empty());
  }

  /// `keep_recent` larger than the conversation protects all of it rather than
  /// underflowing into "protect nothing".
  #[test]
  fn a_keep_recent_larger_than_the_conversation_protects_everything() {
    let mut request = request_of(reads(2));
    let before = result_contents(&request)
      .into_iter()
      .map(str::to_owned)
      .collect::<Vec<_>>();

    Compaction::new(999).apply(&mut request);

    assert_eq!(result_contents(&request), before);
  }

  /// The point of the whole stage is that the request gets *smaller*. An argument the
  /// model made enormous — a `query` built by pasting in a page it just fetched — would
  /// otherwise be echoed whole into the note that replaces the result, and the "note"
  /// could end up larger than the payload it stands in for.
  #[test]
  fn an_enormous_argument_is_capped_rather_than_echoed_whole() {
    let huge = "q".repeat(50_000);
    let mut request = request_of(vec![
      call("s0", web_search::NAME, json!({ "query": huge })),
      result("s0", web_search::NAME, &"hit ".repeat(300)),
    ]);
    let before = tokens::count_request(&request);

    Compaction::new(0).apply(&mut request);

    let note = result_contents(&request)[0];
    assert!(note.chars().count() < 400, "the note must stay a note");
    assert!(note.contains('…'), "a cut must be visible, got: {note}");
    assert!(
      tokens::count_request(&request) < before,
      "compaction that grows the request is worse than not running at all"
    );
  }

  /// Truncation counts `char`s, so a multi-byte argument is cut at a character boundary
  /// instead of panicking on one.
  #[test]
  fn a_multi_byte_argument_is_cut_on_a_character_boundary() {
    let mut request = request_of(vec![
      call(
        "s0",
        web_search::NAME,
        json!({ "query": "中".repeat(50_000) }),
      ),
      result("s0", web_search::NAME, &"hit ".repeat(300)),
    ]);

    Compaction::new(0).apply(&mut request);

    assert!(result_contents(&request)[0].contains('…'));
  }

  /// The same cap applies to a caller's own tool, since `argument` is what they are
  /// pointed at for writing a describer.
  #[test]
  fn the_cap_applies_to_a_custom_describer_too() {
    let mut request = request_of(vec![
      call("q0", "run_query", json!({ "sql": "x".repeat(10_000) })),
      result("q0", "run_query", &"row ".repeat(300)),
    ]);

    Compaction::empty(0)
      .with_tool("run_query", |arguments| {
        format!("Query '{}' was already run.", argument(arguments, "sql"))
      })
      .apply(&mut request);

    assert!(result_contents(&request)[0].chars().count() < 400);
  }
}
