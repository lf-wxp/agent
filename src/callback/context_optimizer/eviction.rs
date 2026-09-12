//! Hard eviction — the backstop stage of [`super::ContextOptimizer`].
//!
//! Compaction and summarization are both *best-effort*: compaction only recognizes the
//! tool names it knows about, and summarization depends on a model call that can fail or
//! be disabled. Neither can promise the request ends up under budget. This stage can,
//! because it simply drops content until it does.
//!
//! The shape it preserves is the one the whole module is built around:
//!
//! ```text
//! [ head ]  the task — pinned, never dropped
//! [ ~~~~ ]  the middle — evicted from here
//! [ tail ]  as much recent context as the budget allows
//! ```
//!
//! Dropping from the middle rather than the front is what makes this safe to apply to a
//! single long run, where the opening message *is* the task and losing it would leave the
//! model working blind.

use crate::{
  agent::ContentItem,
  callback::context_optimizer::{safety::find_safe_start, tokens},
};

/// End of the pinned head: `keep_head` items, but never extending into a tool call.
///
/// A [`ContentItem::ToolCall`] kept without its result is just as invalid to most
/// providers as a result kept without its call, and pinning a call would force its result
/// to be pinned too — silently expanding the head far beyond what was asked for. Stopping
/// short is the honest choice; in practice the head is the opening user message, so this
/// never triggers.
fn safe_head_end(contents: &[ContentItem], keep_head: usize) -> usize {
  let limit = keep_head.min(contents.len());
  contents[..limit]
    .iter()
    .position(|item| matches!(item, ContentItem::ToolCall { .. }))
    .unwrap_or(limit)
}

fn is_user_message(item: &ContentItem) -> bool {
  matches!(item, ContentItem::Message { role, .. } if role == "user")
}

/// First index at or after `start` holding a user-role message.
fn next_user_message(contents: &[ContentItem], start: usize) -> Option<usize> {
  let start = start.min(contents.len());
  contents[start..]
    .iter()
    .position(is_user_message)
    .map(|offset| start + offset)
}

/// Walk backwards from the end, taking items while they fit in what the budget leaves
/// once the pinned head is paid for — and taking the first `keep_recent_min` regardless.
///
/// Returns where the surviving tail starts. `costs[i]` is the cost of item `i`, so
/// `costs.len()` is the conversation length.
fn tail_start_within(
  costs: &[usize],
  head_end: usize,
  available: usize,
  keep_recent_min: usize,
) -> usize {
  let head_cost: usize = costs[..head_end].iter().sum();
  // The head is pinned, so it comes off the top of what the tail has to work with.
  let tail_budget = available.saturating_sub(head_cost);

  let mut tail_start = costs.len();
  let mut tail_cost = 0usize;
  while tail_start > head_end {
    let index = tail_start - 1;
    let taken_if_kept = costs.len() - index;
    let must_keep = taken_if_kept <= keep_recent_min;
    if !must_keep && tail_cost + costs[index] > tail_budget {
      break;
    }
    tail_cost += costs[index];
    tail_start = index;
  }
  tail_start
}

/// Drop items from the middle until `contents` fits in `available` tokens.
///
/// The entry point for a caller writing its own
/// [`BeforeLlmCallback`](crate::agent::BeforeLlmCallback) rather than using
/// [`super::ContextOptimizer`]: it owns its own accounting, so there is no invariant to
/// uphold and nothing it can be called with that panics.
/// [`evict_middle_with_costs`] is the crate-internal variant for a caller that already has
/// per-item costs and does not want to pay for them twice.
pub fn evict_middle(
  contents: &mut Vec<ContentItem>,
  available: usize,
  keep_head: usize,
  keep_recent_min: usize,
) -> usize {
  if contents.is_empty() {
    return 0;
  }

  // Cheap upper bound first; only pay for tokenization when the text could plausibly
  // exceed the budget.
  if tokens::contents_cost_upper_bound(contents) <= available {
    return 0;
  }

  let mut costs = tokens::item_costs(contents);
  evict_middle_with_costs(contents, &mut costs, available, keep_head, keep_recent_min)
}

/// Drop items from the middle until `contents` fits in `available` tokens, reusing
/// `costs` instead of tokenizing again.
///
/// Keeps the first `keep_head` items and as many trailing items as the budget allows,
/// never fewer than `keep_recent_min` — the model needs *some* recent context to act on,
/// even when that overshoots the budget. Returns how many items were dropped.
///
/// Two structural invariants survive the cut: no tool result is separated from its call
/// (see [`find_safe_start`]), and the result opens on a user message. Honoring the latter
/// can cost a little more content than the budget asked for, or — where the only user
/// message sits behind the boundary — can force the opening message to be pinned even
/// under `keep_head = 0`, since the alternative is a request no provider should accept.
///
/// `costs[i]` is the token cost of `contents[i]`; the two are kept in step, so whatever
/// range is dropped from one is dropped from the other. A caller holding a
/// [`tokens::Ledger`] therefore stays accurate across the call without re-measuring.
///
/// Crate-internal, along with the `(contents, costs)` pairing it requires: keeping two
/// parallel vectors in step is an invariant the caller has to uphold by hand, which is a
/// reasonable thing to ask of the one call site in [`super::ContextOptimizer`] and not of
/// an outside caller. Everything outside the crate goes through that optimizer, where the
/// pairing is the ledger's job rather than anyone else's.
///
/// # Panics
///
/// If `costs.len() != contents.len()`. The two are one data structure split in half, and
/// a mismatch means the caller's accounting is already wrong — silently trusting the
/// shorter of the two would trim against a budget computed from the wrong items. An
/// assertion is the right shape for an invariant the crate controls end to end.
pub(crate) fn evict_middle_with_costs(
  contents: &mut Vec<ContentItem>,
  costs: &mut Vec<usize>,
  available: usize,
  keep_head: usize,
  keep_recent_min: usize,
) -> usize {
  assert_eq!(
    costs.len(),
    contents.len(),
    "per-item costs must line up with the conversation"
  );
  if contents.is_empty() {
    return 0;
  }
  if costs.iter().sum::<usize>() <= available {
    return 0;
  }

  let mut head_end = safe_head_end(contents, keep_head);
  let tail_start = tail_start_within(costs, head_end, available, keep_recent_min);

  // Pulling the boundary back to avoid orphaning a tool result can only ever keep more
  // than planned, so it may cross into the head; clamp it there.
  let mut tail_start = find_safe_start(contents, tail_start).max(head_end);

  // With nothing pinned at the front, the surviving conversation has to open on a user
  // message. A sequence that starts with an assistant tool call is rejected outright by
  // Anthropic-style endpoints, and even where it is accepted it asks the model to
  // continue work it can no longer see the request for.
  if head_end == 0
    && contents
      .get(tail_start)
      .is_some_and(|item| !is_user_message(item))
  {
    // Moving the boundary *forward* drops a little more, so the budget still holds —
    // always the better fix, as long as it does not strand a tool result whose call sits
    // in what would now be dropped. A user message is a turn boundary, so in a
    // well-formed transcript it never does.
    let forward =
      next_user_message(contents, tail_start).filter(|&at| find_safe_start(contents, at) == at);

    match forward {
      Some(at) => tail_start = at,
      // Nothing ahead to open on: every user message is behind the boundary, which is
      // the single-long-run shape — one task message followed by pure tool traffic.
      // `keep_head = 0` simply cannot be honored there without emitting an invalid
      // request, so the opening message is pinned after all.
      None if contents.first().is_some_and(is_user_message) => {
        head_end = 1;
        // The tail was sized against a budget that assumed nothing was pinned. Now that
        // something is, re-walk it against what is actually left — otherwise the result
        // comes back over budget by exactly the pinned message's length, which for a
        // long opening task is not a rounding error.
        tail_start = find_safe_start(
          contents,
          tail_start_within(costs, head_end, available, keep_recent_min),
        )
        .max(head_end);
      }
      // No user message anywhere. Nothing to anchor on, so leave the split alone rather
      // than trade one invalid request for another.
      None => {}
    }
  }

  if tail_start <= head_end {
    return 0;
  }

  let dropped = tail_start - head_end;
  contents.drain(head_end..tail_start);
  costs.drain(head_end..tail_start);
  dropped
}

#[cfg(test)]
mod tests {
  use serde_json::json;

  use super::*;
  use crate::agent::ToolResultStatus;

  fn user_msg(text: &str) -> ContentItem {
    ContentItem::Message {
      role: "user".to_owned(),
      content: text.to_owned(),
    }
  }

  fn assistant_msg(text: &str) -> ContentItem {
    ContentItem::Message {
      role: "assistant".to_owned(),
      content: text.to_owned(),
    }
  }

  fn tool_pair(id: &str, payload: &str) -> [ContentItem; 2] {
    [
      ContentItem::ToolCall {
        tool_call_id: id.to_owned(),
        name: "web_search".to_owned(),
        arguments: json!({}),
      },
      ContentItem::ToolResult {
        tool_call_id: id.to_owned(),
        name: "web_search".to_owned(),
        status: ToolResultStatus::Success,
        content: payload.to_owned(),
      },
    ]
  }

  /// One task message followed by bulky tool traffic: the shape of a single long run.
  fn long_run(rounds: usize) -> Vec<ContentItem> {
    let mut contents = vec![user_msg("the original task")];
    for i in 0..rounds {
      contents.extend(tool_pair(&format!("call_{i}"), &"result ".repeat(300)));
    }
    contents
  }

  /// Several questions, each with tool work behind it: the shape of a chat session.
  fn multi_turn(turns: usize) -> Vec<ContentItem> {
    let mut contents = Vec::new();
    for turn in 0..turns {
      contents.push(user_msg(&format!("question {turn}")));
      contents.extend(tool_pair(&format!("t{turn}"), &"result ".repeat(300)));
      contents.push(assistant_msg("answer"));
    }
    contents
  }

  fn assert_structurally_valid(items: &[ContentItem]) {
    let mut seen = Vec::new();
    for item in items {
      match item {
        ContentItem::ToolCall { tool_call_id, .. } => seen.push(tool_call_id.as_str()),
        ContentItem::ToolResult { tool_call_id, .. } => assert!(
          seen.contains(&tool_call_id.as_str()),
          "tool result {tool_call_id} was orphaned from its call"
        ),
        ContentItem::Message { .. } => {}
      }
    }

    if let Some(first) = items.first() {
      assert!(
        is_user_message(first),
        "the conversation must open on a user message, found: {first:?}"
      );
    }
  }

  #[test]
  fn an_empty_conversation_is_left_alone() {
    let mut contents = Vec::new();
    assert_eq!(evict_middle(&mut contents, 10, 1, 2), 0);
  }

  #[test]
  fn a_conversation_within_budget_is_left_alone() {
    let mut contents = vec![user_msg("hi"), assistant_msg("hello")];
    assert_eq!(evict_middle(&mut contents, 10_000, 1, 2), 0);
    assert_eq!(contents.len(), 2);
  }

  #[test]
  fn the_pinned_head_survives_eviction() {
    let mut contents = long_run(30);
    evict_middle(&mut contents, 2_000, 1, 4);

    let ContentItem::Message { content, .. } = &contents[0] else {
      panic!("the head should still be the opening message");
    };
    assert_eq!(content, "the original task");
  }

  #[test]
  fn eviction_brings_the_conversation_under_budget() {
    let mut contents = long_run(30);
    evict_middle(&mut contents, 2_000, 1, 2);

    assert!(
      tokens::count_contents(&contents) <= 2_000,
      "eviction must actually converge on the budget"
    );
  }

  #[test]
  fn the_most_recent_context_is_what_survives() {
    let mut contents = long_run(30);
    evict_middle(&mut contents, 2_000, 1, 2);

    let last_result = contents
      .iter()
      .rev()
      .find_map(|item| match item {
        ContentItem::ToolResult { tool_call_id, .. } => Some(tool_call_id.as_str()),
        _ => None,
      })
      .expect("some tool result should survive");
    assert_eq!(last_result, "call_29", "the newest round must be kept");
  }

  #[test]
  fn a_tool_result_is_never_left_without_its_call() {
    let mut contents = long_run(30);
    // A boundary landing between a call and its result must be pushed back.
    evict_middle(&mut contents, 1_500, 1, 1);
    assert_structurally_valid(&contents);
  }

  #[test]
  fn keep_recent_min_wins_over_the_budget() {
    let mut contents = long_run(30);
    // A budget far too small for six items still has to leave six.
    evict_middle(&mut contents, 1, 1, 6);
    assert!(
      contents.len() >= 6,
      "the floor on recent context must be respected"
    );
  }

  #[test]
  fn a_head_that_would_split_a_tool_pair_is_shortened() {
    // Index 1 is a ToolCall, so a head of 3 must stop at 1 rather than pin a call whose
    // result it cannot guarantee.
    let mut contents = vec![user_msg("task")];
    contents.extend(tool_pair("call_0", &"result ".repeat(400)));
    contents.extend(tool_pair("call_1", &"result ".repeat(400)));
    contents.push(assistant_msg("done"));

    evict_middle(&mut contents, 500, 3, 1);
    assert_structurally_valid(&contents);
  }

  /// `keep_head = 0` pins nothing, so the boundary has to land on a user message by
  /// moving forward — otherwise the request opens on an assistant tool call.
  #[test]
  fn with_nothing_pinned_the_result_still_opens_on_a_user_message() {
    for budget in [500, 1_500, 2_500, 4_000] {
      let mut contents = multi_turn(20);
      evict_middle(&mut contents, budget, 0, 1);
      assert_structurally_valid(&contents);
    }
  }

  /// Moving forward is preferred because it keeps the budget intact.
  #[test]
  fn opening_on_a_user_message_does_not_blow_the_budget() {
    let mut contents = multi_turn(20);
    evict_middle(&mut contents, 2_000, 0, 1);

    assert!(is_user_message(&contents[0]));
    assert!(
      tokens::count_contents(&contents) <= 2_000,
      "advancing to a user message drops more, so the budget must still hold"
    );
  }

  /// The single-long-run shape has exactly one user message, at the front. There is
  /// nothing ahead to open on, so `keep_head = 0` has to give way and pin it — the only
  /// outcome that is both valid and within budget.
  #[test]
  fn a_single_run_pins_its_task_even_with_keep_head_zero() {
    let mut contents = long_run(30);
    evict_middle(&mut contents, 2_000, 0, 1);

    assert_structurally_valid(&contents);
    let ContentItem::Message { content, .. } = &contents[0] else {
      panic!("expected the task to have been pinned");
    };
    assert_eq!(content, "the original task");
    assert!(
      tokens::count_contents(&contents) <= 2_000,
      "pinning one message must not cost the budget"
    );
  }

  /// The same fallback, but with an opening task large enough that charging it to the
  /// budget matters: the tail has to be re-walked against what is actually left, or the
  /// result comes back over budget by the length of the pinned message.
  ///
  /// Deliberately free of tool pairs, so the only thing under test is the head
  /// accounting — [`find_safe_start`] rewinding to keep a call with its result is a
  /// separate, documented reason to overshoot, and would muddy the assertion.
  #[test]
  fn pinning_a_large_task_re_walks_the_tail() {
    let mut contents = vec![user_msg(&"task ".repeat(300))];
    for i in 0..30 {
      contents.push(assistant_msg(&format!("step {i} {}", "word ".repeat(100))));
    }

    evict_middle(&mut contents, 2_000, 0, 1);

    assert_structurally_valid(&contents);
    assert!(
      tokens::count_contents(&contents) <= 2_000,
      "the pinned head must be charged to the budget, not added on top of it"
    );
  }

  /// A transcript with no user message at all cannot be made to open on one. Eviction
  /// should still behave — not panic, and not orphan anything.
  #[test]
  fn a_transcript_without_any_user_message_is_handled() {
    let mut contents = Vec::new();
    for i in 0..20 {
      contents.extend(tool_pair(&format!("call_{i}"), &"result ".repeat(300)));
    }

    evict_middle(&mut contents, 1_000, 0, 1);

    // No user message exists to anchor on, so the opening-message rule cannot apply;
    // the tool pairing rule still must.
    let mut seen = Vec::new();
    for item in &contents {
      match item {
        ContentItem::ToolCall { tool_call_id, .. } => seen.push(tool_call_id.as_str()),
        ContentItem::ToolResult { tool_call_id, .. } => assert!(
          seen.contains(&tool_call_id.as_str()),
          "tool result {tool_call_id} was orphaned"
        ),
        ContentItem::Message { .. } => {}
      }
    }
  }

  #[test]
  fn nothing_is_dropped_when_head_and_tail_already_meet() {
    // Two items, both pinned by head/tail: there is no middle to evict.
    let mut contents = vec![user_msg(&"x".repeat(40_000)), assistant_msg("short")];
    assert_eq!(evict_middle(&mut contents, 1, 1, 1), 0);
    assert_eq!(contents.len(), 2);
  }

  /// The costs a caller passes in must come back describing exactly the items that
  /// survived, or its running total silently drifts from the request it describes.
  #[test]
  fn supplied_costs_stay_in_step_with_the_conversation() {
    let mut contents = long_run(30);
    let mut costs = tokens::item_costs(&contents);

    let dropped = evict_middle_with_costs(&mut contents, &mut costs, 2_000, 1, 2);

    assert!(dropped > 0);
    assert_eq!(costs.len(), contents.len());
    assert_eq!(
      costs.iter().sum::<usize>(),
      tokens::count_contents(&contents),
      "the reused costs must still describe what is left"
    );
  }

  /// Reusing costs must not change *what* gets evicted — only what it costs to find out.
  #[test]
  fn reusing_costs_matches_tokenizing_again() {
    let mut fresh = long_run(30);
    evict_middle(&mut fresh, 2_000, 1, 2);

    let mut reused = long_run(30);
    let mut costs = tokens::item_costs(&reused);
    evict_middle_with_costs(&mut reused, &mut costs, 2_000, 1, 2);

    assert_eq!(reused.len(), fresh.len());
  }

  #[test]
  #[should_panic(expected = "per-item costs must line up")]
  fn mismatched_costs_are_rejected() {
    let mut contents = long_run(4);
    let mut costs = vec![1, 2, 3];
    evict_middle_with_costs(&mut contents, &mut costs, 10, 1, 1);
  }
}
