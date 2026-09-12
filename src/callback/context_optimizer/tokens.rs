//! Token accounting shared by every strategy in [`super`].
//!
//! One tokenizer, one counting convention. `cl100k_base` via
//! [`tiktoken_rs::cl100k_base_singleton`], so the (non-trivial) vocabulary is built once
//! for the process rather than per call.
//!
//! The tokenizer is fixed regardless of which model is actually configured — there is no
//! universal cross-provider tokenizer, and every strategy here enforces a *soft* budget
//! where being a few percent off is irrelevant. This is emphatically not billing-grade
//! accounting.
//!
//! What it does try to be is *honest about direction*: the count includes the system
//! prompt, any extra instructions, and a per-message framing constant, so it errs
//! slightly high rather than low. A budget that silently undercounts is worse than a
//! conservative one — it fails exactly when the context window is already tight.

use std::io;

use serde_json::Value;
use tiktoken_rs::CoreBPE;
use tokio::runtime::{Handle, RuntimeFlavor};

use crate::agent::{ContentItem, llm_request::LlmRequest};

/// Rough per-message framing overhead (role, delimiters) in the chat wire format.
/// The exact figure is provider-specific; 4 is the conventional approximation.
const PER_MESSAGE_OVERHEAD: usize = 4;

/// The process-wide tokenizer.
pub fn bpe() -> &'static CoreBPE {
  tiktoken_rs::cl100k_base_singleton()
}

/// Run a tokenization pass without starving the async runtime.
///
/// Everything below is plain synchronous CPU work, and on a conversation near a large
/// context window a full [`count_request`] is tens of milliseconds of it. Called straight
/// from [`super::ContextOptimizer`]'s `BeforeLlmCallback` impl — which is `async` and
/// therefore runs on a runtime worker — that stalls every other task sharing the thread,
/// including the in-flight requests of *other* concurrent runs.
///
/// [`tokio::task::block_in_place`] is the right tool: it hands the worker's remaining
/// queue to another thread for the duration instead of copying the request onto a
/// `spawn_blocking` task (which would need `'static` data, i.e. a clone of the very thing
/// being measured).
///
/// It is only valid on a multi-thread runtime, and panics on a current-thread one — which
/// includes the default `#[tokio::test]`. So the flavor is checked first and the work runs
/// inline everywhere else: on a current-thread runtime there is no other worker to hand
/// the queue to, making the call a no-op at best regardless.
pub fn offload<T>(work: impl FnOnce() -> T) -> T {
  match Handle::try_current().map(|handle| handle.runtime_flavor()) {
    // Outside a runtime, or on a single-threaded one: nothing to hand off to.
    Err(_) | Ok(RuntimeFlavor::CurrentThread) => work(),
    _ => tokio::task::block_in_place(work),
  }
}

/// Token count of one item's text, without framing overhead.
fn item_text(bpe: &CoreBPE, item: &ContentItem) -> usize {
  match item {
    ContentItem::Message { content, .. } => bpe.encode_ordinary(content).len(),
    ContentItem::ToolCall {
      name, arguments, ..
    } => bpe.encode_ordinary(name).len() + bpe.encode_ordinary(&arguments.to_string()).len(),
    ContentItem::ToolResult { content, .. } => bpe.encode_ordinary(content).len(),
  }
}

/// What one item costs in the request, framing included.
pub fn item_cost(bpe: &CoreBPE, item: &ContentItem) -> usize {
  PER_MESSAGE_OVERHEAD + item_text(bpe, item)
}

/// A cheap *upper* bound on [`item_cost`].
///
/// Every `cl100k` token encodes at least one byte, so an item's byte length can only
/// ever exceed its token count. That makes "bytes already fit the budget" a sound proof
/// that tokens do too — and the comparatively expensive BPE pass can be skipped outright.
/// This is the common case for most rounds of most conversations, which is exactly where
/// that cost would be pure waste.
///
/// Being an upper bound is what makes it safe to skip on: a *lower* bound could clear the
/// budget while the real count exceeded it.
pub fn item_cost_upper_bound(item: &ContentItem) -> usize {
  let text = match item {
    ContentItem::Message { content, .. } => content.len(),
    // Tool arguments have to be measured as serialized text, exactly as `item_text`
    // does — but this path runs on *every* round, including the overwhelmingly common
    // one that is already within budget, so it counts the serialization instead of
    // materializing it. See `json_len`.
    ContentItem::ToolCall {
      name, arguments, ..
    } => name.len() + json_len(arguments),
    ContentItem::ToolResult { content, .. } => content.len(),
  };
  PER_MESSAGE_OVERHEAD + text
}

/// Serialized byte length of `value`, without allocating the serialization.
///
/// `Value::to_string` would allocate a `String` per tool call per round purely to read
/// its `.len()`. Serializing into a counting sink gives the identical number — the exact
/// length, not an estimate — for no allocation at all.
fn json_len(value: &Value) -> usize {
  /// Discards everything written and keeps only the byte count.
  struct Counter(usize);

  impl io::Write for Counter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
      self.0 += buf.len();
      Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
      Ok(())
    }
  }

  let mut counter = Counter(0);
  // Infallible in practice: a `Value` is always representable as JSON, and the sink
  // above cannot fail. Falling back to the count collected so far keeps this a pure
  // function either way — and since the caller uses it as an upper bound to *skip*
  // tokenizing, an undercount here would only ever cost a BPE pass, never correctness.
  let _ = serde_json::to_writer(&mut counter, value);
  counter.0
}

/// [`item_cost_upper_bound`] summed over a conversation.
pub fn contents_cost_upper_bound(items: &[ContentItem]) -> usize {
  items.iter().map(item_cost_upper_bound).sum()
}

/// [`item_cost_upper_bound`] summed over a whole request, instructions included.
///
/// Lets a caller answer "is this request definitely within budget?" without tokenizing
/// anything; only a request that fails this check needs [`count_request`].
pub fn request_cost_upper_bound(request: &LlmRequest) -> usize {
  let instructions: usize = request
    .instructions
    .iter()
    .map(|instruction| PER_MESSAGE_OVERHEAD + instruction.len())
    .sum();
  instructions + contents_cost_upper_bound(&request.contents)
}

/// Cost of the parts of the request that trimming the conversation cannot reduce: the
/// system-level instructions, the agent's own prompt among them.
///
/// Strategies subtract this from their budget before deciding how much conversation
/// they can afford to keep.
pub fn count_fixed(request: &LlmRequest) -> usize {
  let bpe = bpe();
  request
    .instructions
    .iter()
    .map(|instruction| PER_MESSAGE_OVERHEAD + bpe.encode_ordinary(instruction).len())
    .sum()
}

/// Cost of the conversation itself.
pub fn count_contents(items: &[ContentItem]) -> usize {
  let bpe = bpe();
  items.iter().map(|item| item_cost(bpe, item)).sum()
}

/// Per-item costs for a conversation, in order.
pub fn item_costs(items: &[ContentItem]) -> Vec<usize> {
  let bpe = bpe();
  items.iter().map(|item| item_cost(bpe, item)).collect()
}

/// Approximate token count of the request as it will actually be sent.
pub fn count_request(request: &LlmRequest) -> usize {
  count_fixed(request) + count_contents(&request.contents)
}

/// Per-item token costs for one request, kept in step with it across a multi-stage
/// pipeline.
///
/// [`super::ContextOptimizer`] edits a request in up to three stages and has to know the
/// total after each one. Re-tokenizing the whole request every time would run the BPE
/// pass four or five times per over-budget round — tens of milliseconds of pure CPU on a
/// conversation near a large context window, on a thread that is supposed to be driving
/// async work. A ledger instead pays for tokenization once and charges each stage only
/// for what it actually changed:
///
/// - [`Self::refresh`] for a stage that rewrites items in place ([`super::Compaction`]).
/// - [`Self::rebuild`] for a stage that adds or removes them ([`super::Summarization`]).
/// - [`Self::contents_mut`] for a stage that drops a contiguous range
///   ([`super::eviction`]), which keeps the two in step by construction.
#[derive(Debug, Clone)]
pub struct Ledger {
  fixed: usize,
  items: Vec<usize>,
}

impl Ledger {
  /// Tokenize `request` once and record what each part costs.
  pub fn new(request: &LlmRequest) -> Self {
    Self {
      fixed: count_fixed(request),
      items: item_costs(&request.contents),
    }
  }

  /// What the whole request currently costs.
  pub fn total(&self) -> usize {
    self.fixed + self.items.iter().sum::<usize>()
  }

  /// Cost of the parts trimming the conversation cannot reduce; see [`count_fixed`].
  pub fn fixed(&self) -> usize {
    self.fixed
  }

  /// Per-item costs, parallel to [`LlmRequest::contents`], mutable so a stage that drops
  /// a range of items can drop the same range here.
  ///
  /// Crate-internal: keeping this vector in step with the request it describes is an
  /// invariant held by hand, and a ledger that drifted would report a request as fitting
  /// the budget when it does not. That is a reasonable thing to ask of the pipeline in
  /// [`super`] and not of an outside caller, who has [`Self::total`] and can rebuild.
  pub(crate) fn contents_mut(&mut self) -> &mut Vec<usize> {
    &mut self.items
  }

  /// Re-tokenize only the items at `changed`, for a stage that rewrote them in place.
  ///
  /// Out-of-range indices are ignored rather than panicking: this is an accounting
  /// optimization, and a caller that reports an index it no longer has should get a
  /// slightly stale total, not a crashed run.
  pub fn refresh(&mut self, contents: &[ContentItem], changed: &[usize]) {
    let bpe = bpe();
    for &index in changed {
      if let (Some(item), Some(cost)) = (contents.get(index), self.items.get_mut(index)) {
        *cost = item_cost(bpe, item);
      }
    }
  }

  /// Discard everything and tokenize `request` again, for a stage whose edits moved or
  /// removed items so per-item costs no longer line up.
  pub fn rebuild(&mut self, request: &LlmRequest) {
    self.fixed = count_fixed(request);
    self.items = item_costs(&request.contents);
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn msg(text: &str) -> ContentItem {
    ContentItem::Message {
      role: "user".to_owned(),
      content: text.to_owned(),
    }
  }

  #[test]
  fn an_empty_conversation_costs_nothing() {
    assert_eq!(count_contents(&[]), 0);
  }

  #[test]
  fn longer_text_costs_more() {
    assert!(count_contents(&[msg("hello world")]) > count_contents(&[msg("hi")]));
  }

  #[test]
  fn the_upper_bound_never_underestimates() {
    // Every cl100k token encodes at least one byte, so bytes >= tokens for any input.
    // This is the direction that matters: the bound is used to *skip* tokenizing, so
    // underestimating would clear a budget the real count exceeds.
    for text in [
      "hi",
      "hello world",
      "a much longer sentence here",
      "中文内容",
      "",
    ] {
      let item = msg(text);
      assert!(
        item_cost_upper_bound(&item) >= item_cost(bpe(), &item),
        "the upper bound must never underestimate the real cost for {text:?}"
      );
    }
  }

  /// The whole-request bound has to hold for the instructions too, or the pre-check
  /// could skip a request whose system prompt alone blows the budget.
  #[test]
  fn the_request_upper_bound_covers_instructions() {
    let request = LlmRequest {
      instructions: vec![
        "a fairly long system prompt".to_owned(),
        "中文指令".to_owned(),
      ],
      contents: vec![msg("hi"), msg("中文内容")],
    };

    assert!(request_cost_upper_bound(&request) >= count_request(&request));
  }

  #[test]
  fn framing_overhead_is_counted_per_item() {
    let one = count_contents(&[msg("hi")]);
    let two = count_contents(&[msg("hi"), msg("hi")]);
    assert!(
      two >= one * 2,
      "each item must carry its own framing overhead"
    );
  }

  /// Instructions carry the agent's own system prompt among them, so a budget that
  /// ignored them would undercount by exactly the part a caller is most likely to have
  /// made large.
  #[test]
  fn instructions_are_part_of_the_cost() {
    let mut request = LlmRequest {
      instructions: Vec::new(),
      contents: vec![msg("hi")],
    };
    let without = count_request(&request);

    request.push_instruction("be brief");
    assert!(count_request(&request) > without);
  }

  /// The agent's prompt reaches the count through [`LlmRequest::new`], not a field of
  /// its own.
  #[test]
  fn the_agent_prompt_lands_in_the_instructions() {
    let bare = LlmRequest::new(None, &[]);
    let prompted = LlmRequest::new(Some("you are a careful assistant".to_owned()), &[]);

    assert!(bare.instructions.is_empty());
    assert_eq!(prompted.instructions.len(), 1);
    assert!(count_request(&prompted) > count_request(&bare));
  }

  #[test]
  fn fixed_and_conversation_costs_sum_to_the_whole_request() {
    let request = LlmRequest {
      instructions: vec!["sys".to_owned(), "extra".to_owned()],
      contents: vec![msg("hi"), msg("there")],
    };

    assert_eq!(
      count_request(&request),
      count_fixed(&request) + count_contents(&request.contents)
    );
  }

  /// The allocation-free counter must agree exactly with the serialization it replaces,
  /// or the upper bound it feeds stops being one.
  #[test]
  fn json_len_matches_the_serialized_length() {
    for value in [
      serde_json::json!({}),
      serde_json::json!({ "file_path": "/tmp/x.rs" }),
      serde_json::json!({ "query": "中文 with \"quotes\" and \\ escapes" }),
      serde_json::json!({ "nested": { "a": [1, 2.5, true, null] } }),
      serde_json::json!("bare string"),
      serde_json::json!(12345),
    ] {
      assert_eq!(
        json_len(&value),
        value.to_string().len(),
        "mismatch for {value}"
      );
    }
  }

  #[test]
  fn the_upper_bound_never_underestimates_a_tool_call() {
    let item = ContentItem::ToolCall {
      tool_call_id: "c0".to_owned(),
      name: "read_file".to_owned(),
      arguments: serde_json::json!({ "file_path": "src/中文.rs", "limit": 200 }),
    };

    assert!(item_cost_upper_bound(&item) >= item_cost(bpe(), &item));
  }

  fn ledger_request() -> LlmRequest {
    LlmRequest {
      instructions: vec!["sys".to_owned()],
      contents: vec![msg("hello world"), msg("second item"), msg("third item")],
    }
  }

  #[test]
  fn a_ledger_agrees_with_counting_from_scratch() {
    let request = ledger_request();
    let mut ledger = Ledger::new(&request);

    assert_eq!(ledger.total(), count_request(&request));
    assert_eq!(ledger.fixed(), count_fixed(&request));
    assert_eq!(ledger.contents_mut().len(), request.contents.len());
  }

  /// The whole point of `refresh`: after an in-place rewrite the ledger must match a
  /// from-scratch count, without having re-tokenized the untouched items.
  #[test]
  fn refresh_tracks_an_in_place_rewrite() {
    let mut request = ledger_request();
    let mut ledger = Ledger::new(&request);

    request.contents[1] = msg("a considerably longer replacement for the second item");
    ledger.refresh(&request.contents, &[1]);

    assert_eq!(ledger.total(), count_request(&request));
  }

  #[test]
  fn refresh_ignores_an_out_of_range_index() {
    let request = ledger_request();
    let mut ledger = Ledger::new(&request);
    let before = ledger.total();

    ledger.refresh(&request.contents, &[99]);

    assert_eq!(ledger.total(), before, "a stale index must not panic");
  }

  #[test]
  fn dropping_a_range_keeps_the_ledger_in_step() {
    let mut request = ledger_request();
    let mut ledger = Ledger::new(&request);

    request.contents.drain(1..2);
    ledger.contents_mut().drain(1..2);

    assert_eq!(ledger.total(), count_request(&request));
  }

  #[test]
  fn rebuild_recovers_from_arbitrary_edits() {
    let mut request = ledger_request();
    let mut ledger = Ledger::new(&request);

    request.push_instruction("a summary of earlier progress");
    request.contents.drain(0..2);
    ledger.rebuild(&request);

    assert_eq!(ledger.total(), count_request(&request));
  }
}
