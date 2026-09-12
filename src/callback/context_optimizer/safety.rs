//! Keeping a truncated conversation structurally valid.
//!
//! Every strategy in [`super`] that drops items from the middle or front of a
//! conversation has the same obligation: a [`ContentItem::ToolResult`] may never survive
//! without the [`ContentItem::ToolCall`] that produced it. Most providers reject such a
//! request outright, so an over-eager trim turns into a hard API error rather than a
//! slightly worse answer.
//!
//! [`find_safe_start`] is the shared primitive for that: [`super::eviction`] uses it to
//! push a cut boundary back until the surviving range is structurally valid, and
//! [`super::summarization`] uses it to pick where a recap may end.

use std::collections::HashMap;

use crate::agent::ContentItem;

/// Move `start` earlier until nothing in `contents[start..]` is an orphaned tool result.
///
/// Returns the earliest index at or before `start` from which the conversation can be
/// kept without separating any tool result from its call. When a result's call cannot be
/// found anywhere (a malformed transcript), the result is left as-is rather than scanning
/// to index 0 — there is no call to reunite it with, so dropping everything would achieve
/// nothing.
///
/// Single backward pass: tool call positions are indexed once up front, then `start` is
/// walked back as violations are found. Once the scan drops below the current `start`
/// nothing further can change it, so it stops there rather than walking to 0.
pub fn find_safe_start(contents: &[ContentItem], start: usize) -> usize {
  let start = start.min(contents.len());
  if start == 0 {
    return 0;
  }

  // Tool call ids are provider-generated and unique within a conversation. Should one
  // repeat anyway, the *earliest* position is the only conservative answer: keeping from
  // there keeps every occurrence, whereas keeping from a later one would leave the
  // earlier call dropped while its result survived — exactly the orphan this function
  // exists to prevent.
  let mut call_index: HashMap<&str, usize> = HashMap::new();
  for (index, item) in contents.iter().enumerate() {
    if let ContentItem::ToolCall { tool_call_id, .. } = item {
      call_index.entry(tool_call_id.as_str()).or_insert(index);
    }
  }

  let mut safe = start;
  for index in (0..contents.len()).rev() {
    // Everything below the current boundary is already outside the kept range, and since
    // the scan only moves downward it can no longer pull the boundary back any further.
    if index < safe {
      break;
    }
    if let ContentItem::ToolResult { tool_call_id, .. } = &contents[index]
      && let Some(&call_at) = call_index.get(tool_call_id.as_str())
      && call_at < safe
    {
      safe = call_at;
    }
  }

  safe
}

#[cfg(test)]
mod tests {
  use serde_json::json;

  use super::*;
  use crate::agent::ToolResultStatus;

  fn msg(text: &str) -> ContentItem {
    ContentItem::Message {
      role: "user".to_owned(),
      content: text.to_owned(),
    }
  }

  fn call(id: &str) -> ContentItem {
    ContentItem::ToolCall {
      tool_call_id: id.to_owned(),
      name: "calculator".to_owned(),
      arguments: json!({}),
    }
  }

  fn result(id: &str) -> ContentItem {
    ContentItem::ToolResult {
      tool_call_id: id.to_owned(),
      name: "calculator".to_owned(),
      status: ToolResultStatus::Success,
      content: "ok".to_owned(),
    }
  }

  #[test]
  fn start_of_zero_is_already_safe() {
    let contents = vec![call("a"), result("a")];
    assert_eq!(find_safe_start(&contents, 0), 0);
  }

  #[test]
  fn a_start_with_no_orphans_is_left_alone() {
    let contents = vec![msg("hi"), call("a"), result("a")];
    assert_eq!(
      find_safe_start(&contents, 1),
      1,
      "call and result both kept"
    );
  }

  #[test]
  fn pulls_the_boundary_back_to_include_the_missing_call() {
    // Cutting at 2 would keep `result(a)` while dropping `call(a)`.
    let contents = vec![msg("hi"), call("a"), result("a")];
    assert_eq!(find_safe_start(&contents, 2), 1);
  }

  #[test]
  fn walks_back_across_several_orphans() {
    // Cutting at 4 orphans both results; the earliest call needed is at index 0.
    let contents = vec![call("a"), call("b"), msg("x"), result("b"), result("a")];
    assert_eq!(find_safe_start(&contents, 4), 0);
  }

  #[test]
  fn a_result_whose_call_is_absent_does_not_force_a_full_rewind() {
    let contents = vec![msg("hi"), msg("there"), result("ghost")];
    assert_eq!(
      find_safe_start(&contents, 2),
      2,
      "no call exists to reunite it with, so the boundary stays put"
    );
  }

  #[test]
  fn start_past_the_end_is_clamped() {
    let contents = vec![msg("hi")];
    assert_eq!(find_safe_start(&contents, 99), 1);
  }

  /// A duplicated tool call id has to resolve to its *earliest* position. Picking the
  /// later one would satisfy the `call_at < safe` check for that occurrence while
  /// leaving the first call dropped and its result orphaned.
  #[test]
  fn a_repeated_call_id_resolves_to_its_earliest_position() {
    let contents = vec![call("a"), msg("x"), call("a"), result("a")];
    assert_eq!(find_safe_start(&contents, 3), 0);
  }
}
