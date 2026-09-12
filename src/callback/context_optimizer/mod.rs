//! Context-window management: keep a request inside the model's budget without losing
//! the thread of what it is doing.
//!
//! One pipeline, not a menu of alternatives. Every stage preserves the same shape:
//!
//! ```text
//! [ head ]  system prompt + the task      pinned, never dropped
//! [ ~~~~ ]  the middle                    compacted, then summarized, then evicted
//! [ tail ]  the most recent exchanges     kept verbatim
//! ```
//!
//! The stages escalate, cheapest first, and stop as soon as the request fits:
//!
//! 1. [`compaction`] — rewrite spent tool results in place. Free, and usually enough:
//!    tool output is where the bloat overwhelmingly lives.
//! 2. [`summarization`] — replace the middle with an LLM-written recap. Costs an extra
//!    model round, so it is opt-in. Reads the *original* transcript rather than what
//!    compaction left behind, so the recap describes the findings and not the notes that
//!    replaced them.
//! 3. [`eviction`] — drop the middle outright. Free, and the only stage that *guarantees*
//!    convergence, so it always runs last. Without it a request could still exceed the
//!    budget whenever the stages above fall short — compaction only knows the tools it
//!    was given, and summarization can fail or be switched off.
//!
//! Even eviction cannot always reach the budget: the `keep_recent_min` floor and the
//! opening-user-message rule both outrank it, because a request that is small but invalid
//! is worse than one that is valid but large. When that happens the round is logged at
//! `warn` — the provider's own context-length error would not say why.
//!
//! Throughout, two invariants hold: a
//! [`ContentItem::ToolResult`](crate::agent::ContentItem::ToolResult) is never separated
//! from the call that produced it (see [`safety`]), and the opening message survives,
//! because in a long agentic run it *is* the task.
//!
//! The whole pipeline tokenizes the request once, through a [`tokens::Ledger`] that each
//! stage updates with only what it changed.
//!
//! Everything operates on the per-round [`LlmRequest`] copy.
//! [`ExecutionContext::events`] is the authoritative transcript and stays complete for
//! persistence — trimming the prompt and keeping the full history are not in conflict.

pub mod compaction;
pub mod eviction;
pub mod safety;
pub mod summarization;
pub mod tokens;

pub use compaction::{Compaction, Describe};
pub use summarization::Summarization;

use crate::{
  agent::{BeforeLlmCallback, ExecutionContext, llm_request::LlmRequest},
  llm::provider::Provider,
};

/// Items pinned at the front. One, by default: the opening user message, which carries
/// the task.
const DEFAULT_KEEP_HEAD: usize = 1;

/// Trailing items kept even when the budget says otherwise — the model needs something
/// recent to act on, and a request trimmed to nothing is useless regardless of its size.
const DEFAULT_KEEP_RECENT_MIN: usize = 4;

/// Tail items [`Compaction`] will not rewrite, so the model still sees the full output of
/// what it just did.
///
/// Deliberately a separate constant from [`DEFAULT_KEEP_RECENT_MIN`] despite the equal
/// value: one is a floor on what eviction may *delete*, the other a window compaction may
/// not *rewrite*. They answer different questions and should be free to diverge.
const DEFAULT_COMPACTION_KEEP_RECENT: usize = 4;

/// The default [`BeforeLlmCallback`], installed by [`crate::agent::Agent::new`].
///
/// See the module docs for the pipeline. Configured through the `with_*` builders;
/// summarization is off unless [`Self::with_summarization`] is called, since it spends an
/// extra model round per invocation.
#[derive(Debug)]
pub struct ContextOptimizer {
  /// Budget for the whole request — system prompt and instructions included, not just
  /// the conversation.
  max_tokens: usize,
  /// Leading items never dropped.
  keep_head: usize,
  /// Floor on how many trailing items survive eviction.
  keep_recent_min: usize,
  compaction: Option<Compaction>,
  summarization: Option<Summarization>,
}

impl ContextOptimizer {
  pub fn new(max_tokens: usize) -> Self {
    Self {
      max_tokens,
      keep_head: DEFAULT_KEEP_HEAD,
      keep_recent_min: DEFAULT_KEEP_RECENT_MIN,
      compaction: Some(Compaction::new(DEFAULT_COMPACTION_KEEP_RECENT)),
      summarization: None,
    }
  }

  /// Enable the summarization stage, which costs one auxiliary model call whenever
  /// compaction alone leaves the request over budget.
  ///
  /// `provider` is the agent's own, so the recap honors the same credentials,
  /// concurrency budget and retry policy as the run it belongs to.
  #[must_use]
  pub fn with_summarization(
    mut self,
    provider: Provider,
    model: impl Into<String>,
    keep_recent: usize,
  ) -> Self {
    self.summarization = Some(Summarization::new(provider, model, keep_recent));
    self
  }

  /// Pin more than just the opening message — e.g. a task brief spanning several
  /// messages. `0` drops the head entirely, letting old turns fall away completely,
  /// which suits a pure chat session where the earliest exchange is rarely relevant.
  #[must_use]
  pub fn with_keep_head(mut self, keep_head: usize) -> Self {
    self.keep_head = keep_head;
    self
  }

  #[must_use]
  pub fn with_keep_recent_min(mut self, keep_recent_min: usize) -> Self {
    self.keep_recent_min = keep_recent_min;
    self
  }

  /// Replace the compaction stage — e.g. to register a caller's own reproducible tools,
  /// or to exempt a built-in whose output is not actually cheap to fetch again.
  #[must_use]
  pub fn with_compaction(mut self, compaction: Compaction) -> Self {
    self.compaction = Some(compaction);
    self
  }

  #[must_use]
  pub fn without_compaction(mut self) -> Self {
    self.compaction = None;
    self
  }

  /// Budget for the whole request, instructions included.
  pub fn max_tokens(&self) -> usize {
    self.max_tokens
  }
}

#[async_trait::async_trait]
impl BeforeLlmCallback for ContextOptimizer {
  async fn call(&self, context: &ExecutionContext, request: &mut LlmRequest) {
    // Staying under budget is the overwhelmingly common case, and it is the one case
    // where tokenizing the whole request buys nothing. Byte length can only overstate
    // token count, so clearing the budget on bytes alone is proof enough to skip the
    // BPE pass entirely.
    if tokens::request_cost_upper_bound(request) <= self.max_tokens {
      return;
    }

    // Past this point the request is tokenized exactly once, and each stage pays only
    // for what it changed. Re-counting the whole request after every stage would run the
    // BPE pass four or five times over a conversation that is, by definition, near the
    // context window. Every one of those passes is synchronous CPU work on a runtime
    // worker, so each is additionally wrapped in `tokens::offload` — see its docs.
    let mut ledger = tokens::offload(|| tokens::Ledger::new(request));
    if ledger.total() <= self.max_tokens {
      return;
    }
    tracing::debug!(
      tokens = ledger.total(),
      budget = self.max_tokens,
      "context optimizer engaged"
    );

    if let Some(compaction) = &self.compaction {
      // Rewrites in place, so only the rewritten items need re-measuring.
      tokens::offload(|| {
        let rewritten = compaction.apply(request);
        ledger.refresh(&request.contents, &rewritten);
      });
      if ledger.total() <= self.max_tokens {
        tracing::debug!("compaction alone brought the request under budget");
        return;
      }
    }

    if let Some(summarization) = &self.summarization {
      // Best-effort: a failed recap leaves the request as compaction left it, which is
      // still valid — just larger. Eviction below will handle it either way, so there is
      // nothing here worth failing the whole run over.
      match summarization.apply(context, request).await {
        Ok(()) => {
          // Adds an instruction and removes a range of items, so per-item costs no
          // longer line up and the ledger has to start over.
          tokens::offload(|| ledger.rebuild(request));
          if ledger.total() <= self.max_tokens {
            tracing::debug!("summarization brought the request under budget");
            return;
          }
        }
        Err(err) => tracing::warn!("summarization skipped: {err}"),
      }
    }

    // Backstop. The budget left for the conversation is what remains once the parts
    // trimming cannot shrink are paid for — recomputed from the ledger, since
    // summarization may have pushed an instruction that eats into the same budget.
    let available = self.max_tokens.saturating_sub(ledger.fixed());
    let dropped = eviction::evict_middle_with_costs(
      &mut request.contents,
      ledger.contents_mut(),
      available,
      self.keep_head,
      self.keep_recent_min,
    );
    if dropped > 0 {
      tracing::debug!(
        dropped_items = dropped,
        tokens = ledger.total(),
        "evicted the middle of the conversation"
      );
    }

    // Eviction converges on the budget wherever it is allowed to. Where it is not — the
    // `keep_recent_min` floor, the opening-user-message rule, a tool pair that cannot be
    // split — the request goes out over budget and the provider answers with a
    // context-length error that says nothing about why. Saying so here is the difference
    // between a one-line explanation and an afternoon of guessing.
    let final_tokens = ledger.total();
    if final_tokens > self.max_tokens {
      tracing::warn!(
        tokens = final_tokens,
        budget = self.max_tokens,
        keep_head = self.keep_head,
        keep_recent_min = self.keep_recent_min,
        "context optimizer could not reach the budget; the structural floors take \
         precedence over it"
      );
    }
  }
}

#[cfg(test)]
mod tests {
  use serde_json::json;

  use super::*;
  use crate::{agent::ContentItem, agent::ToolResultStatus, config};

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

  fn read_file_pair(id: &str, payload: &str) -> [ContentItem; 2] {
    [
      ContentItem::ToolCall {
        tool_call_id: id.to_owned(),
        name: "read_file".to_owned(),
        arguments: json!({ "file_path": "/tmp/x.rs" }),
      },
      ContentItem::ToolResult {
        tool_call_id: id.to_owned(),
        name: "read_file".to_owned(),
        status: ToolResultStatus::Success,
        content: payload.to_owned(),
      },
    ]
  }

  fn request_of(contents: Vec<ContentItem>) -> LlmRequest {
    LlmRequest {
      instructions: Vec::new(),
      contents,
    }
  }

  /// A single long run: one task message, then many rounds of bulky tool traffic.
  fn long_run(rounds: usize) -> Vec<ContentItem> {
    let mut contents = vec![user_msg("the original task")];
    for i in 0..rounds {
      contents.extend(read_file_pair(&format!("call_{i}"), &"line ".repeat(300)));
      contents.push(assistant_msg("thinking"));
    }
    contents
  }

  /// A multi-turn chat: several user questions, each with some work behind it.
  fn multi_turn(turns: usize) -> Vec<ContentItem> {
    let mut contents = Vec::new();
    for turn in 0..turns {
      contents.push(user_msg(&format!("question {turn}")));
      contents.extend(read_file_pair(&format!("t{turn}"), &"line ".repeat(300)));
      contents.push(assistant_msg("answer"));
    }
    contents
  }

  #[test]
  fn the_default_budget_follows_the_shared_config() {
    let optimizer = ContextOptimizer::new(config::max_history_tokens());
    assert_eq!(optimizer.max_tokens, config::max_history_tokens());
  }

  #[test]
  fn summarization_is_off_by_default() {
    assert!(
      ContextOptimizer::new(1_000).summarization.is_none(),
      "an extra model call must be opted into, never implicit"
    );
  }

  #[tokio::test]
  async fn a_request_within_budget_is_untouched() {
    let mut request = request_of(vec![user_msg("hi"), assistant_msg("hello")]);
    let before = request.contents.clone();

    ContextOptimizer::new(100_000)
      .call(&ExecutionContext::new(), &mut request)
      .await;

    assert_eq!(request.contents.len(), before.len());
    assert!(request.instructions.is_empty());
  }

  /// The gap the old two-strategy split left open: a single run never crosses a turn
  /// boundary, so anything keyed on turns did nothing at all here.
  #[tokio::test]
  async fn a_single_long_run_is_brought_under_budget() {
    let mut request = request_of(long_run(30));
    assert!(tokens::count_request(&request) > 6_000);

    ContextOptimizer::new(6_000)
      .call(&ExecutionContext::new(), &mut request)
      .await;

    assert!(
      tokens::count_request(&request) <= 6_000,
      "a single run must be protected too"
    );
  }

  #[tokio::test]
  async fn a_multi_turn_conversation_is_brought_under_budget() {
    let mut request = request_of(multi_turn(40));
    assert!(tokens::count_request(&request) > 6_000);

    ContextOptimizer::new(6_000)
      .call(&ExecutionContext::new(), &mut request)
      .await;

    assert!(tokens::count_request(&request) <= 6_000);
  }

  #[tokio::test]
  async fn the_task_survives_in_both_shapes() {
    for contents in [long_run(30), multi_turn(12)] {
      let mut request = request_of(contents);
      ContextOptimizer::new(4_000)
        .call(&ExecutionContext::new(), &mut request)
        .await;

      let ContentItem::Message { role, .. } = &request.contents[0] else {
        panic!("the head should still be a message");
      };
      assert_eq!(role, "user", "the opening task must be pinned");
    }
  }

  #[tokio::test]
  async fn structural_validity_is_preserved() {
    let mut request = request_of(long_run(30));
    ContextOptimizer::new(3_000)
      .call(&ExecutionContext::new(), &mut request)
      .await;

    let mut seen = Vec::new();
    for item in &request.contents {
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

  /// Compaction rewrites spent `read_file` output, so a conversation that is bulky but
  /// not enormous should come back under budget without dropping anything.
  #[tokio::test]
  async fn compaction_alone_can_be_enough() {
    let mut request = request_of(long_run(6));
    let before = request.contents.len();

    ContextOptimizer::new(2_000)
      .call(&ExecutionContext::new(), &mut request)
      .await;

    assert_eq!(
      request.contents.len(),
      before,
      "nothing should be dropped when a rewrite suffices"
    );
    assert!(tokens::count_request(&request) <= 2_000);
  }

  /// With the free stage disabled, the backstop still has to deliver.
  #[tokio::test]
  async fn eviction_alone_still_converges() {
    let mut request = request_of(long_run(30));

    ContextOptimizer::new(5_000)
      .without_compaction()
      .call(&ExecutionContext::new(), &mut request)
      .await;

    assert!(tokens::count_request(&request) <= 5_000);
  }

  /// A large system prompt competes for the same budget, so the conversation has to give
  /// way — otherwise the request sent would exceed `max_tokens` by its length.
  #[tokio::test]
  async fn the_system_prompt_competes_with_the_conversation() {
    let mut request = request_of(long_run(20));
    request.push_instruction("word ".repeat(400));

    ContextOptimizer::new(4_000)
      .call(&ExecutionContext::new(), &mut request)
      .await;

    assert!(tokens::count_request(&request) <= 4_000);
  }

  /// `keep_head = 0` gives back the "let the oldest fall away entirely" behavior that
  /// suits a pure chat session, where the opening question is rarely still relevant.
  #[tokio::test]
  async fn keep_head_controls_whether_the_opening_message_survives() {
    let opening = |request: &LlmRequest| match &request.contents[0] {
      ContentItem::Message { content, .. } => Some(content.clone()),
      _ => None,
    };

    // Compaction off so the eviction stage is guaranteed to be what decides this.
    let mut pinned = request_of(multi_turn(40));
    ContextOptimizer::new(3_000)
      .without_compaction()
      .call(&ExecutionContext::new(), &mut pinned)
      .await;

    let mut unpinned = request_of(multi_turn(40));
    ContextOptimizer::new(3_000)
      .without_compaction()
      .with_keep_head(0)
      .call(&ExecutionContext::new(), &mut unpinned)
      .await;

    assert_eq!(
      opening(&pinned).as_deref(),
      Some("question 0"),
      "keep_head = 1 must pin the opening message"
    );
    assert_ne!(
      opening(&unpinned).as_deref(),
      Some("question 0"),
      "keep_head = 0 must let the oldest turn fall away"
    );
  }

  /// `keep_recent_min` outranks the budget, so a floor large enough to exceed it has to
  /// win — the request goes out oversized rather than stripped of anything to act on.
  /// The M-4 warning exists for exactly this outcome.
  #[tokio::test]
  async fn the_recent_floor_outranks_the_budget() {
    let mut request = request_of(long_run(30));

    ContextOptimizer::new(10)
      .without_compaction()
      .with_keep_recent_min(8)
      .call(&ExecutionContext::new(), &mut request)
      .await;

    assert!(
      request.contents.len() >= 8,
      "the floor on recent context must survive an impossible budget"
    );
    assert!(
      tokens::count_request(&request) > 10,
      "this is the documented case where the budget cannot be met"
    );
  }

  /// A caller's own tool can be taught to compact, which is the whole reason the tool
  /// table is a registry rather than a match arm.
  #[tokio::test]
  async fn a_custom_compaction_registry_is_honored() {
    let mut contents = vec![user_msg("the original task")];
    for i in 0..20 {
      contents.push(ContentItem::ToolCall {
        tool_call_id: format!("q{i}"),
        name: "run_query".to_owned(),
        arguments: json!({ "sql": "select 1" }),
      });
      contents.push(ContentItem::ToolResult {
        tool_call_id: format!("q{i}"),
        name: "run_query".to_owned(),
        status: ToolResultStatus::Success,
        content: "row ".repeat(300),
      });
    }
    let mut request = request_of(contents);
    let before = request.contents.len();

    ContextOptimizer::new(3_000)
      .with_compaction(
        Compaction::empty(4).with_tool("run_query", |_| "query already run".to_owned()),
      )
      .call(&ExecutionContext::new(), &mut request)
      .await;

    assert_eq!(
      request.contents.len(),
      before,
      "a registered tool means the rewrite is enough on its own"
    );
    assert!(tokens::count_request(&request) <= 3_000);
  }

  /// The budget is enforced against the request the ledger describes, so the two must
  /// never disagree — a drifted ledger would decide the request fits when it does not.
  #[tokio::test]
  async fn the_running_total_matches_a_count_from_scratch() {
    for budget in [1_500, 3_000, 6_000] {
      let mut request = request_of(long_run(30));
      ContextOptimizer::new(budget)
        .call(&ExecutionContext::new(), &mut request)
        .await;

      let ledger = tokens::Ledger::new(&request);
      assert_eq!(
        ledger.total(),
        tokens::count_request(&request),
        "the pipeline's accounting must survive every stage"
      );
    }
  }
}
