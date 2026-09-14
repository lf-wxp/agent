use std::{borrow::Cow, collections::VecDeque};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::event::Event;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct TokenUsage {
  pub prompt_tokens: u32,
  pub completion_tokens: u32,
  pub total_tokens: u32,
}

impl TokenUsage {
  /// Accumulate one round's reported usage.
  ///
  /// Saturating rather than wrapping: these are a reporting aid, and a conversation long
  /// enough to exhaust a `u32` should report an implausibly large number rather than
  /// panic in debug or silently restart from zero in release.
  pub fn add(&mut self, prompt_tokens: u32, completion_tokens: u32, total_tokens: u32) {
    self.prompt_tokens = self.prompt_tokens.saturating_add(prompt_tokens);
    self.completion_tokens = self.completion_tokens.saturating_add(completion_tokens);
    self.total_tokens = self.total_tokens.saturating_add(total_tokens);
  }
}

/// The prior turns handed to [`crate::agent::Agent::run_continuing`], plus — optionally —
/// the identity of the conversation they belong to.
///
/// `Agent` stays a stateless function of "prior events + new input"; this type does not
/// change that. It exists because a *run* and a *conversation* are different spans, and
/// only the caller knows the latter: every call gets a fresh
/// [`ExecutionContext::execution_id`], so anything that wants to accumulate across turns
/// (see [`crate::callback::context_optimizer::Summarization`], which would otherwise
/// re-summarize the whole history on every turn) needs an identifier that outlives one
/// call. The natural one is whatever key the caller already stores the session under —
/// see [`crate::agent::session::SessionStore`].
///
/// `From<Vec<Event>>` covers the case where there is no such identity: a one-shot run, or
/// a caller that simply keeps history in a local variable.
#[derive(Debug, Clone, Default)]
pub struct Conversation {
  /// Stable across every turn of one conversation, or `None` when the caller has no such
  /// notion. Purely an identifier — nothing is looked up or persisted by it here.
  pub id: Option<String>,
  /// Namespace [`Self::id`] is unique *within*, mirroring
  /// [`crate::agent::session::SessionStore`]'s own `scope` parameter.
  ///
  /// Without it, two callers that legitimately isolate their storage by scope — the
  /// documented example being a terminal-originated turn and a web-originated turn in
  /// one process — would still collide in anything keyed on
  /// [`ExecutionContext::continuity_key`] the moment they picked the same `id`. A hook's
  /// accumulated state would then leak across a boundary the session store deliberately
  /// enforces. Pass the same scope here that is passed to the store.
  pub scope: Option<String>,
  pub events: Vec<Event>,
}

impl Conversation {
  /// Prior turns belonging to the conversation identified by `id`.
  pub fn new(id: impl Into<String>, events: Vec<Event>) -> Self {
    Self {
      id: Some(id.into()),
      scope: None,
      events,
    }
  }

  /// Prior turns with no conversation identity attached.
  pub fn anonymous(events: Vec<Event>) -> Self {
    Self {
      id: None,
      scope: None,
      events,
    }
  }

  /// Namespace [`Self::id`] is unique within — the same value handed to
  /// [`crate::agent::session::SessionStore`].
  #[must_use]
  pub fn with_scope(mut self, scope: impl Into<String>) -> Self {
    self.scope = Some(scope.into());
    self
  }
}

impl From<Vec<Event>> for Conversation {
  fn from(events: Vec<Event>) -> Self {
    Self::anonymous(events)
  }
}

impl From<Conversation> for Vec<Event> {
  fn from(conversation: Conversation) -> Self {
    conversation.events
  }
}

/// Everything one in-progress run consists of, as data.
///
/// Serializable on purpose, and that is a design statement rather than a convenience:
/// what a run "is" at any moment is its transcript plus a little bookkeeping, never a
/// suspended call stack. That is what makes a run interruptible — see
/// [`crate::agent::runtime::AgentRunState`], which persists one of these so a turn
/// stopped waiting on a human can be resumed in another process.
/// `Clone` because a checkpoint has to snapshot the transcript mid-round without
/// disturbing the run producing it (see [`crate::agent::runtime::RunCheckpoint`]).
/// Cloning is a deep copy of the whole event list, so it belongs on that path and not
/// in the per-round request path, which deliberately borrows instead (see
/// [`crate::agent::LlmRequest`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionContext {
  /// Identifies this one run. A fresh value per
  /// [`crate::agent::Agent::run_continuing`] call, including successive turns of the same
  /// conversation — use [`Self::conversation_id`] to correlate those.
  pub execution_id: String,
  /// Identity of the conversation this run continues, when the caller supplied one (see
  /// [`Conversation`]). `None` for a one-shot [`crate::agent::Agent::run`].
  pub conversation_id: Option<String>,
  /// Namespace [`Self::conversation_id`] is unique within; see [`Conversation::scope`].
  pub conversation_scope: Option<String>,
  pub events: Vec<Event>,
  pub current_step: u32,
  pub final_result: Option<String>,
  pub usage: TokenUsage,
}

impl ExecutionContext {
  pub fn new() -> Self {
    Self {
      execution_id: Uuid::new_v4().to_string(),
      conversation_id: None,
      conversation_scope: None,
      events: Vec::new(),
      current_step: 0,
      final_result: None,
      usage: TokenUsage::default(),
    }
  }

  /// The most stable identity available for this run: the conversation it belongs to when
  /// the caller named one, falling back to this single run.
  ///
  /// This is the key anything accumulating state across turns should use. Falling back to
  /// [`Self::execution_id`] keeps such state correct-but-per-run for an anonymous
  /// conversation, rather than letting unrelated runs collide on one shared bucket.
  ///
  /// [`Self::conversation_scope`] is folded in when present, so a key can never span two
  /// scopes that the caller isolates on purpose. The separator is a NUL byte for the same
  /// reason [`crate::agent::session::FileSessionStore`] uses one: it cannot occur in
  /// either part, so `("a", "bc")` and `("ab", "c")` stay distinct.
  ///
  /// Borrowed in the common (unscoped) case; only a scoped conversation pays for the
  /// concatenation, and only once per round.
  pub fn continuity_key(&self) -> Cow<'_, str> {
    let id = self
      .conversation_id
      .as_deref()
      .unwrap_or(&self.execution_id);
    continuity_key_for(self.conversation_scope.as_deref(), id)
  }

  pub fn add_event(&mut self, event: Event) {
    self.events.push(event);
  }

  pub fn increment_step(&mut self) {
    self.current_step += 1;
  }
}

impl Default for ExecutionContext {
  fn default() -> Self {
    Self::new()
  }
}

/// The [`ExecutionContext::continuity_key`] a given `(scope, id)` pair produces.
///
/// Exposed so a caller holding only those two values — rather than a live
/// [`ExecutionContext`] — can name the same bucket. The case this exists for is clearing
/// per-conversation hook state when a session is reset: the reset happens between turns,
/// where no context exists, yet it has to address exactly the key the turns themselves
/// used. Re-deriving the format at that call site instead would put the same NUL-joining
/// rule in two places, and a silent mismatch there means state that survives a reset that
/// was meant to clear it.
pub fn continuity_key_for<'a>(scope: Option<&str>, id: &'a str) -> Cow<'a, str> {
  match scope {
    Some(scope) => Cow::Owned(format!("{scope}\0{id}")),
    None => Cow::Borrowed(id),
  }
}

/// Fixed-capacity, least-recently-used map keyed by [`ExecutionContext::continuity_key`].
///
/// The shape every cross-turn cache in [`crate::callback`] needs: a
/// [`crate::agent::BeforeLlmCallback`] is shared across concurrent runs and never learns
/// that one has ended, so anything it accumulates has to be bounded and per-conversation.
/// Reads refresh recency, so an active conversation is not evicted in favor of a burst of
/// short-lived ones; evicting a live one only costs whatever work it had cached.
#[derive(Debug)]
pub struct ContinuityCache<T> {
  capacity: usize,
  entries: VecDeque<(String, T)>,
}

impl<T> ContinuityCache<T> {
  pub fn new(capacity: usize) -> Self {
    Self {
      // A capacity of 0 would discard every insert, silently disabling the caller.
      capacity: capacity.max(1),
      entries: VecDeque::new(),
    }
  }

  /// Current entry for `key`, if any, refreshing its recency.
  ///
  /// `Clone` is required here and nowhere else: the entry has to stay in the cache to
  /// keep being useful next round, so the caller gets a copy rather than the original.
  pub fn get(&mut self, key: &str) -> Option<T>
  where
    T: Clone,
  {
    let index = self.entries.iter().position(|(id, _)| id == key)?;
    let entry = self
      .entries
      .remove(index)
      .expect("index came from position");
    let value = entry.1.clone();
    self.entries.push_back(entry);
    Some(value)
  }

  /// Insert or replace `key`'s entry, evicting the least recently used if full.
  ///
  /// `merge` decides what to keep when an entry already exists, receiving
  /// `(existing, incoming)`. Concurrent rounds of one conversation can otherwise let a
  /// slower pass overwrite a faster one's progress.
  pub fn put_with(&mut self, key: &str, value: T, merge: impl FnOnce(&T, T) -> T) {
    if let Some(index) = self.entries.iter().position(|(id, _)| id == key) {
      let mut entry = self
        .entries
        .remove(index)
        .expect("index came from position");
      entry.1 = merge(&entry.1, value);
      self.entries.push_back(entry);
      return;
    }
    if self.entries.len() >= self.capacity {
      self.entries.pop_front();
    }
    self.entries.push_back((key.to_owned(), value));
  }

  pub fn len(&self) -> usize {
    self.entries.len()
  }

  pub fn is_empty(&self) -> bool {
    self.entries.is_empty()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn a_vec_of_events_converts_to_an_anonymous_conversation() {
    let conversation: Conversation = Vec::new().into();
    assert!(conversation.id.is_none());
    assert!(conversation.scope.is_none());
  }

  #[test]
  fn continuity_key_prefers_the_conversation_over_the_run() {
    let mut context = ExecutionContext::new();
    assert_eq!(
      context.continuity_key(),
      context.execution_id,
      "an anonymous conversation falls back to this run"
    );

    context.conversation_id = Some("session-7".to_owned());
    assert_eq!(context.continuity_key(), "session-7");
  }

  /// The scope a caller isolates its session storage by has to isolate hook state too,
  /// or two front-ends that happen to pick the same session id share one bucket.
  #[test]
  fn continuity_key_isolates_scopes_that_reuse_one_id() {
    let key_for = |scope: &str| {
      let mut context = ExecutionContext::new();
      context.conversation_id = Some("session-7".to_owned());
      context.conversation_scope = Some(scope.to_owned());
      context.continuity_key().into_owned()
    };

    assert_ne!(key_for("local"), key_for("web"));
  }

  /// A NUL separator is what makes that isolation total: without it `("a", "bc")` and
  /// `("ab", "c")` would produce the same key.
  #[test]
  fn continuity_key_cannot_be_confused_across_a_scope_boundary() {
    let key_for = |scope: &str, id: &str| {
      let mut context = ExecutionContext::new();
      context.conversation_id = Some(id.to_owned());
      context.conversation_scope = Some(scope.to_owned());
      context.continuity_key().into_owned()
    };

    assert_ne!(key_for("a", "bc"), key_for("ab", "c"));
  }

  #[test]
  fn token_usage_saturates_instead_of_overflowing() {
    let mut usage = TokenUsage::default();
    usage.add(u32::MAX, u32::MAX, u32::MAX);
    usage.add(10, 10, 10);

    assert_eq!(usage.prompt_tokens, u32::MAX);
    assert_eq!(usage.completion_tokens, u32::MAX);
    assert_eq!(usage.total_tokens, u32::MAX);
  }

  /// The premise behind resuming an interrupted run: a run in progress is data, so it
  /// survives a round trip through storage with nothing lost. Every field is checked
  /// rather than a sample, since one silently dropped on the way out (a mid-run
  /// `current_step`, say) would resume as a subtly different run instead of failing.
  #[test]
  fn an_in_progress_context_round_trips_through_json() {
    use crate::agent::ContentItem;

    let mut context = ExecutionContext::new();
    context.conversation_id = Some("session-7".to_owned());
    context.conversation_scope = Some("local".to_owned());
    context.current_step = 3;
    context.usage.add(11, 22, 33);
    context.add_event(Event::new(
      "exec-1",
      "user",
      vec![ContentItem::Message {
        role: "user".to_owned(),
        content: "hi".to_owned(),
      }],
    ));

    let json = serde_json::to_string(&context).unwrap();
    let back: ExecutionContext = serde_json::from_str(&json).unwrap();

    assert_eq!(back.execution_id, context.execution_id);
    assert_eq!(back.conversation_id.as_deref(), Some("session-7"));
    assert_eq!(back.conversation_scope.as_deref(), Some("local"));
    assert_eq!(back.current_step, 3);
    assert_eq!(back.usage.prompt_tokens, 11);
    assert_eq!(back.usage.completion_tokens, 22);
    assert_eq!(back.usage.total_tokens, 33);
    assert_eq!(back.events.len(), 1);
    assert_eq!(
      back.continuity_key(),
      context.continuity_key(),
      "the key hook state is bucketed under has to survive, or a resumed run would \
       read someone else's accumulated state"
    );
  }

  /// A finished run carries its answer. Losing it on the way through storage would turn
  /// a completed run into one that looks like it still has work to do.
  #[test]
  fn a_final_result_survives_serialization() {
    let mut context = ExecutionContext::new();
    context.final_result = Some("42".to_owned());

    let json = serde_json::to_string(&context).unwrap();
    let back: ExecutionContext = serde_json::from_str(&json).unwrap();

    assert_eq!(back.final_result.as_deref(), Some("42"));
  }

  #[test]
  fn cache_returns_none_for_an_unknown_key() {
    let mut cache: ContinuityCache<u32> = ContinuityCache::new(4);
    assert!(cache.get("nope").is_none());
  }

  #[test]
  fn cache_round_trips_a_value() {
    let mut cache = ContinuityCache::new(4);
    cache.put_with("a", 1, |_, incoming| incoming);
    assert_eq!(cache.get("a"), Some(1));
  }

  #[test]
  fn cache_keeps_keys_separate() {
    let mut cache = ContinuityCache::new(4);
    cache.put_with("a", 1, |_, incoming| incoming);
    assert!(cache.get("b").is_none());
  }

  #[test]
  fn cache_merge_decides_what_survives_a_second_insert() {
    let mut cache = ContinuityCache::new(4);
    cache.put_with("a", 5, |_, incoming| incoming);
    // Keep the larger value, the way summarization keeps whichever pass got furthest.
    cache.put_with("a", 3, |existing, incoming| incoming.max(*existing));
    assert_eq!(cache.get("a"), Some(5));

    cache.put_with("a", 9, |existing, incoming| incoming.max(*existing));
    assert_eq!(cache.get("a"), Some(9));
  }

  #[test]
  fn cache_is_bounded() {
    let mut cache = ContinuityCache::new(3);
    for i in 0..10 {
      cache.put_with(&format!("k{i}"), i, |_, incoming| incoming);
    }
    assert_eq!(cache.len(), 3);
    assert!(cache.get("k0").is_none(), "the oldest should be gone");
    assert_eq!(cache.get("k9"), Some(9));
  }

  #[test]
  fn reading_an_entry_protects_it_from_eviction() {
    let mut cache = ContinuityCache::new(2);
    cache.put_with("keep", 1, |_, incoming| incoming);
    cache.put_with("filler", 2, |_, incoming| incoming);

    // Touch `keep` so `filler` becomes the least recently used.
    assert_eq!(cache.get("keep"), Some(1));
    cache.put_with("new", 3, |_, incoming| incoming);

    assert_eq!(cache.get("keep"), Some(1), "recently read must survive");
    assert!(cache.get("filler").is_none());
  }

  #[test]
  fn a_zero_capacity_cache_still_holds_one_entry() {
    let mut cache = ContinuityCache::new(0);
    cache.put_with("a", 1, |_, incoming| incoming);
    assert_eq!(cache.get("a"), Some(1));
  }
}
