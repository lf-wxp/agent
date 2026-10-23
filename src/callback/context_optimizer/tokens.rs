//! Token accounting shared by every strategy in [`super`].
//!
//! One counting convention, two vocabularies, picked by model name
//! ([`Encoding::for_model`]): `o200k_base` for the OpenAI models that use it (GPT-4o,
//! GPT-4.1, GPT-5, the `o` series), `cl100k_base` for everything else. Each is built once
//! for the process ([`tiktoken_rs`]'s singletons) rather than per call.
//!
//! For a model outside OpenAI's line — DeepSeek, Qwen, Claude, a local one — there is no
//! local tokenizer to use, so `cl100k_base` stands in and the count is an estimate. That
//! is acceptable because every strategy here enforces a *soft* budget, where being a few
//! percent off is irrelevant. This is emphatically not billing-grade accounting.
//!
//! The free functions ([`count_request`], [`count_contents`], ...) count with the default,
//! `cl100k_base`; the methods of the same names on [`Encoding`] count with a chosen one.
//!
//! What it does try to be is *honest about direction*: the count includes the system
//! prompt, any extra instructions, and a per-message framing constant, so it errs
//! slightly high rather than low. A budget that silently undercounts is worse than a
//! conservative one — it fails exactly when the context window is already tight.

use std::io;

use serde_json::Value;
use tiktoken_rs::{CoreBPE, tokenizer::Tokenizer};
use tokio::runtime::{Handle, RuntimeFlavor};

use crate::agent::{ContentItem, llm_request::LlmRequest};

/// Rough per-message framing overhead (role, delimiters) in the chat wire format.
/// The exact figure is provider-specific; 4 is the conventional approximation.
const PER_MESSAGE_OVERHEAD: usize = 4;

/// Which BPE vocabulary counts the tokens; see the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum Encoding {
  /// GPT-4 and GPT-3.5's vocabulary, and the stand-in for every model not known to use
  /// another.
  #[default]
  Cl100kBase,
  /// GPT-4o, GPT-4.1, GPT-5, `gpt-oss` and the `o` series (`o1`, `o3`, `o4-mini`, ...).
  O200kBase,
}

impl Encoding {
  /// The encoding `model` is tokenized with, by `tiktoken`'s own model table, so a new
  /// OpenAI model is picked up by updating `tiktoken-rs` rather than this crate.
  ///
  /// The name is matched case-insensitively, a provider prefix is ignored (`openai/gpt-4o`,
  /// as OpenRouter names it, is `gpt-4o`), and so is a fine-tune's `ft:` wrapper. A model
  /// the table does not know — every other provider's — gets `cl100k_base`.
  ///
  /// `gpt-oss`'s `o200k_harmony` shares `o200k_base`'s vocabulary and differs only in
  /// special tokens, which are not counted here, so it maps to `o200k_base`.
  pub fn for_model(model: &str) -> Self {
    let name = model.trim().to_ascii_lowercase();
    let name = name.rsplit('/').next().unwrap_or(&name);
    match tiktoken_rs::tokenizer::get_tokenizer(name) {
      Some(Tokenizer::O200kBase | Tokenizer::O200kHarmony) => Self::O200kBase,
      _ => Self::Cl100kBase,
    }
  }

  /// The vocabulary itself, built on first use and shared by the whole process.
  pub fn bpe(self) -> &'static CoreBPE {
    match self {
      Self::Cl100kBase => tiktoken_rs::cl100k_base_singleton(),
      Self::O200kBase => tiktoken_rs::o200k_base_singleton(),
    }
  }

  /// `tiktoken`'s name for it, e.g. for a log line.
  pub fn name(self) -> &'static str {
    match self {
      Self::Cl100kBase => "cl100k_base",
      Self::O200kBase => "o200k_base",
    }
  }

  /// [`count_fixed`], counted with this encoding.
  pub fn count_fixed(self, request: &LlmRequest) -> usize {
    let bpe = self.bpe();
    request
      .instructions
      .iter()
      .map(|instruction| PER_MESSAGE_OVERHEAD + bpe.encode_ordinary(instruction).len())
      .sum()
  }

  /// [`count_contents`], counted with this encoding.
  pub fn count_contents(self, items: &[ContentItem]) -> usize {
    let bpe = self.bpe();
    items.iter().map(|item| item_cost(bpe, item)).sum()
  }

  /// [`item_costs`], counted with this encoding.
  pub fn item_costs(self, items: &[ContentItem]) -> Vec<usize> {
    let bpe = self.bpe();
    items.iter().map(|item| item_cost(bpe, item)).collect()
  }

  /// [`count_request`], counted with this encoding.
  pub fn count_request(self, request: &LlmRequest) -> usize {
    self.count_fixed(request) + self.count_contents(&request.contents)
  }
}

/// The default encoding's vocabulary (`cl100k_base`); [`Encoding::bpe`] for another.
pub fn bpe() -> &'static CoreBPE {
  Encoding::default().bpe()
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
    ContentItem::Reasoning { text } => bpe.encode_ordinary(text).len(),
  }
}

/// What one item costs in the request, framing included.
pub fn item_cost(bpe: &CoreBPE, item: &ContentItem) -> usize {
  PER_MESSAGE_OVERHEAD + item_text(bpe, item)
}

/// A cheap *upper* bound on [`item_cost`], whichever [`Encoding`] counts it.
///
/// Every BPE token — `cl100k_base`'s and `o200k_base`'s alike — encodes at least one
/// byte, so an item's byte length can only ever exceed its token count. That makes
/// "bytes already fit the budget" a sound proof that tokens do too — and the
/// comparatively expensive BPE pass can be skipped outright. This is the common case for
/// most rounds of most conversations, which is exactly where that cost would be pure
/// waste.
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
    ContentItem::Reasoning { text } => text.len(),
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
/// they can afford to keep. Counted with the default encoding; see
/// [`Encoding::count_fixed`].
pub fn count_fixed(request: &LlmRequest) -> usize {
  Encoding::default().count_fixed(request)
}

/// Cost of the conversation itself, with the default encoding; see
/// [`Encoding::count_contents`].
pub fn count_contents(items: &[ContentItem]) -> usize {
  Encoding::default().count_contents(items)
}

/// Per-item costs for a conversation, in order, with the default encoding; see
/// [`Encoding::item_costs`].
pub fn item_costs(items: &[ContentItem]) -> Vec<usize> {
  Encoding::default().item_costs(items)
}

/// Approximate token count of the request as it will actually be sent, with the default
/// encoding; see [`Encoding::count_request`].
pub fn count_request(request: &LlmRequest) -> usize {
  Encoding::default().count_request(request)
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
///
/// A ledger counts with one [`Encoding`] throughout, so a stage's re-measurement is in
/// the same units as everything else in it.
#[derive(Debug, Clone)]
pub struct Ledger {
  encoding: Encoding,
  fixed: usize,
  items: Vec<usize>,
}

impl Ledger {
  /// Tokenize `request` once, with the default encoding, and record what each part
  /// costs.
  pub fn new(request: &LlmRequest) -> Self {
    Self::with_encoding(request, Encoding::default())
  }

  /// [`Self::new`], counting with `encoding`.
  pub fn with_encoding(request: &LlmRequest, encoding: Encoding) -> Self {
    Self {
      encoding,
      fixed: encoding.count_fixed(request),
      items: encoding.item_costs(&request.contents),
    }
  }

  /// The encoding every count in this ledger was made with.
  pub fn encoding(&self) -> Encoding {
    self.encoding
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
    let bpe = self.encoding.bpe();
    for &index in changed {
      if let (Some(item), Some(cost)) = (contents.get(index), self.items.get_mut(index)) {
        *cost = item_cost(bpe, item);
      }
    }
  }

  /// Discard everything and tokenize `request` again, for a stage whose edits moved or
  /// removed items so per-item costs no longer line up.
  pub fn rebuild(&mut self, request: &LlmRequest) {
    self.fixed = self.encoding.count_fixed(request);
    self.items = self.encoding.item_costs(&request.contents);
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
    // Every BPE token encodes at least one byte, so bytes >= tokens for any input, in
    // either encoding. This is the direction that matters: the bound is used to *skip*
    // tokenizing, so underestimating would clear a budget the real count exceeds.
    for encoding in [Encoding::Cl100kBase, Encoding::O200kBase] {
      for text in [
        "hi",
        "hello world",
        "a much longer sentence here",
        "中文内容",
        "",
      ] {
        let item = msg(text);
        assert!(
          item_cost_upper_bound(&item) >= item_cost(encoding.bpe(), &item),
          "the upper bound must never underestimate the real cost for {text:?} ({})",
          encoding.name()
        );
      }
    }
  }

  #[test]
  fn openai_models_on_o200k_are_counted_with_it() {
    for model in [
      "gpt-4o",
      "gpt-4o-mini",
      "GPT-4o-2024-08-06",
      "chatgpt-4o-latest",
      "gpt-4.1-mini",
      "gpt-5",
      "gpt-5.1",
      "o1",
      "o3-mini",
      "o4-mini",
      "gpt-oss-120b",
      "openai/gpt-4o",
      "ft:gpt-4o-mini:acme::abc123",
    ] {
      assert_eq!(Encoding::for_model(model), Encoding::O200kBase, "{model}");
    }
  }

  /// Older OpenAI models use `cl100k_base` themselves; every other provider's model gets
  /// it as the stand-in — which is what every model was counted with before.
  #[test]
  fn other_models_fall_back_to_cl100k() {
    for model in [
      "gpt-4",
      "gpt-4-turbo",
      "gpt-3.5-turbo",
      "deepseek-v4-flash",
      "qwen-max",
      "anthropic/claude-sonnet-4",
      "llama3:8b",
      "",
    ] {
      assert_eq!(
        Encoding::for_model(model),
        Encoding::Cl100kBase,
        "{model:?}"
      );
    }
  }

  /// The choice matters most where the vocabularies differ most: `o200k_base` spends
  /// noticeably fewer tokens on Chinese, so counting a GPT-4o conversation with
  /// `cl100k_base` would trim it sooner than it needs.
  #[test]
  fn the_two_encodings_count_differently() {
    let items = [msg(
      "上下文窗口管理：在不丢失关键信息的前提下，把请求控制在模型的预算之内。",
    )];
    let cl100k = Encoding::Cl100kBase.count_contents(&items);
    let o200k = Encoding::O200kBase.count_contents(&items);
    assert!(o200k < cl100k, "o200k {o200k} vs cl100k {cl100k}");
    assert_eq!(count_contents(&items), cl100k, "the default is cl100k_base");
  }

  /// Every re-measurement uses the ledger's own encoding, not the default.
  #[test]
  fn a_ledger_keeps_its_encoding_through_refresh_and_rebuild() {
    let mut request = ledger_request();
    request.contents[0] = msg("中文内容会让两种编码的计数不一样");
    let mut ledger = Ledger::with_encoding(&request, Encoding::O200kBase);
    assert_eq!(ledger.encoding(), Encoding::O200kBase);
    assert_eq!(ledger.total(), Encoding::O200kBase.count_request(&request));
    assert_ne!(ledger.total(), count_request(&request));

    request.contents[1] = msg("再换一段更长的中文内容，看看局部刷新是否仍然一致");
    ledger.refresh(&request.contents, &[1]);
    assert_eq!(ledger.total(), Encoding::O200kBase.count_request(&request));

    request.contents.drain(2..);
    ledger.rebuild(&request);
    assert_eq!(ledger.total(), Encoding::O200kBase.count_request(&request));
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
