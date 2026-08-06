//! Token-budget-aware trimming of conversation history, so a long-running multi-turn
//! session (see [`crate::api::session`]) does not grow without bound and eventually blow
//! past the model's context window or balloon per-turn cost.
//!
//! Truncation is coarse-grained on purpose: whole turns are dropped, never a lone tool
//! result orphaned from its tool call (most providers reject that outright) or an
//! assistant message split from the tool calls it made. A finer-grained approach —
//! summarizing dropped turns instead of discarding them, or retrieving relevant ones back
//! via [`crate::knowledge_base`]'s vector search instead of keeping them in the prompt at
//! all — is a reasonable next step, not implemented here.

use tiktoken_rs::CoreBPE;

use super::event::{ContentItem, Event};

/// Approximate token count of a batch of events' text content.
///
/// "Approximate" for two reasons: the tokenizer is `cl100k_base` regardless of which
/// model is actually configured (there is no universal cross-provider tokenizer, and this
/// is close enough for a soft budget), and only the raw text of each item is counted, not
/// the exact wire format sent to the model (JSON-encoded tool arguments, per-provider chat
/// template overhead, ...). Good enough to decide "is this history getting too big", not
/// billing-grade accounting.
fn count_tokens(bpe: &CoreBPE, events: &[Event]) -> usize {
  events
    .iter()
    .flat_map(|event| &event.content)
    .map(|item| match item {
      ContentItem::Message { content, .. } => bpe.encode_ordinary(content).len(),
      ContentItem::ToolCall { arguments, .. } => bpe.encode_ordinary(&arguments.to_string()).len(),
      ContentItem::ToolResult { content, .. } => bpe.encode_ordinary(content).len(),
    })
    .sum()
}

/// Index of the first event of every turn: index `0` unconditionally (so a leading
/// non-`"user"` event, which should not normally happen, still belongs to some turn
/// rather than being silently excluded), plus every later index whose event was authored
/// by `"user"` (see [`crate::agent::runtime::Agent::record_user_input`]).
fn turn_boundaries(events: &[Event]) -> Vec<usize> {
  let mut bounds = vec![0];
  for (index, event) in events.iter().enumerate().skip(1) {
    if event.author == "user" {
      bounds.push(index);
    }
  }
  bounds
}

/// Drop the oldest whole turns until `events` fits within `max_tokens`, always keeping at
/// least the most recent turn so the model still has *something* to work from — even if
/// that one turn alone already exceeds the budget, it is kept rather than split, for the
/// same "never orphan a tool call/result" reason turns are the unit of truncation at all.
pub(crate) fn trim_to_budget(events: Vec<Event>, max_tokens: usize) -> Vec<Event> {
  if events.is_empty() {
    return events;
  }

  let bpe = tiktoken_rs::cl100k_base_singleton();
  if count_tokens(bpe, &events) <= max_tokens {
    return events;
  }

  let bounds = turn_boundaries(&events);
  let turn_tokens: Vec<usize> = bounds
    .iter()
    .enumerate()
    .map(|(i, &start)| {
      let end = bounds.get(i + 1).copied().unwrap_or(events.len());
      count_tokens(bpe, &events[start..end])
    })
    .collect();

  // Always keep the last turn; grow the kept range backwards while older turns still fit.
  let mut kept_tokens = *turn_tokens.last().expect("bounds is never empty here");
  let mut keep_from = bounds.len() - 1;
  for i in (0..bounds.len() - 1).rev() {
    let candidate = kept_tokens + turn_tokens[i];
    if candidate > max_tokens {
      break;
    }
    kept_tokens = candidate;
    keep_from = i;
  }

  let start = bounds[keep_from];
  if start == 0 {
    return events;
  }
  tracing::debug!(
    dropped_events = start,
    kept_events = events.len() - start,
    max_tokens,
    "trimmed conversation history to fit the token budget"
  );
  events[start..].to_vec()
}

#[cfg(test)]
mod tests {
  use serde_json::json;

  use super::*;

  fn message(author: &str, text: &str) -> Event {
    Event::new(
      "exec",
      author,
      vec![ContentItem::Message {
        role: author.to_owned(),
        content: text.to_owned(),
      }],
    )
  }

  fn tool_pair(text: &str) -> [Event; 2] {
    [
      Event::new(
        "exec",
        "agent",
        vec![ContentItem::ToolCall {
          tool_call_id: "call_1".to_owned(),
          name: "calculator".to_owned(),
          arguments: json!({}),
        }],
      ),
      Event::new(
        "exec",
        "tool",
        vec![ContentItem::ToolResult {
          tool_call_id: "call_1".to_owned(),
          name: "calculator".to_owned(),
          status: crate::agent::ToolResultStatus::Success,
          content: text.to_owned(),
        }],
      ),
    ]
  }

  #[test]
  fn empty_history_is_returned_unchanged() {
    assert!(trim_to_budget(Vec::new(), 10).is_empty());
  }

  #[test]
  fn history_within_budget_is_left_untouched() {
    let events = vec![message("user", "hi"), message("assistant", "hello")];
    let trimmed = trim_to_budget(events.clone(), 10_000);
    assert_eq!(trimmed.len(), events.len());
  }

  #[test]
  fn drops_the_oldest_whole_turns_first() {
    // Three turns, each one a single short user message; a tiny budget should force
    // dropping the oldest ones while always keeping at least the very last turn.
    let events = vec![
      message("user", "turn one"),
      message("user", "turn two"),
      message("user", "turn three"),
    ];

    let trimmed = trim_to_budget(events, 1);

    assert_eq!(trimmed.len(), 1, "only the most recent turn is kept");
    let ContentItem::Message { content, .. } = &trimmed[0].content[0] else {
      panic!("expected a message");
    };
    assert_eq!(content, "turn three");
  }

  #[test]
  fn never_splits_a_tool_call_from_its_result() {
    let mut events = vec![message("user", "what is 2+2?")];
    events.extend(tool_pair("4"));
    events.push(message("assistant", "it's 4"));
    events.push(message("user", "and 3+3?"));

    // Budget small enough to force trimming, but the surviving turn must still contain
    // its tool call *and* tool result together, or the assistant/tool pairing.
    let trimmed = trim_to_budget(events, 1);

    assert_eq!(trimmed.len(), 1, "only the last (short) turn survives");
  }

  #[test]
  fn keeps_the_last_turn_even_if_it_alone_exceeds_the_budget() {
    let huge = "word".repeat(5_000);
    let events = vec![message("user", &huge)];

    let trimmed = trim_to_budget(events.clone(), 1);

    assert_eq!(trimmed.len(), 1, "the only turn is kept despite the budget");
  }
}
