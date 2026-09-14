//! [`DualApprovalCallback`]: the same "ask a human before running a dangerous tool" idea
//! as [`crate::callback::approval::ApprovalCallback`], but able to ask that human through
//! whichever front-ends happen to be attached to the session — a terminal, one or more
//! browser tabs, or both at once.
//!
//! # Approval belongs to the session, not to one front-end
//!
//! A terminal and a browser pointed at the same session are two *views* of one
//! conversation, not two deployments. So a prompt raised by a turn typed in the terminal
//! has to be answerable from the browser, and vice versa: whoever is looking gets to
//! decide. [`ApprovalChannel::Session`] is that shape — the prompt is published once, to
//! everyone, and the first decision to come back wins ([`ApprovalRegistry`] enforces the
//! "first" part).
//!
//! Routing a prompt to only the front-end that happened to start the turn would strand it
//! whenever that view walked away, and since a turn holds the session's turn lock for its
//! whole duration, a stranded prompt does not merely stall itself — it wedges every other
//! view of that session too.
//!
//! For the same reason a prompt is not only *published* but also *queryable*:
//! [`ApprovalRegistry::pending_snapshot`] lets a view that arrived after the fact — a
//! browser tab reloaded mid-turn — discover what is outstanding. A published event alone
//! reaches nobody who was not already listening, and an approval is deliberately absent
//! from the transcript (it is a gate on a call, not a part of the conversation), so
//! replaying history would not surface one either.
//!
//! # Nothing waits forever
//!
//! Every wait is bounded (see [`DualApprovalCallback::with_timeout`]) and every way of
//! failing to get an answer — timeout, no front-end listening, a view that received the
//! prompt and vanished — denies. Fail-closed is the only safe default for a callback whose
//! entire job is gating destructive operations, and a bound is what makes "nobody
//! answered" recoverable instead of terminal.
//!
//! [`with_approval_channel`] attaches a channel to one turn: a [`tokio::task_local!`]
//! carries it from wherever the turn starts down to whichever [`BeforeToolCallback::call`]
//! it triggers, however many tool calls that turn ends up making.

use std::{
  collections::HashMap,
  future::Future,
  io::{self, Write},
  sync::{Arc, Mutex},
  time::Duration,
};

use tokio::sync::{Mutex as AsyncMutex, mpsc, oneshot};

use crate::{
  agent::{
    ContinuityCache, ExecutionContext,
    callback::{BeforeToolCallback, ToolCallDecision, ToolCallView},
  },
  config,
};

/// How many conversations' remembered answers to keep before evicting the least recently
/// used. Generous for the CLI (one session at a time) while still bounding a long-lived
/// library consumer that runs many; losing an entry only costs one extra prompt.
const STICKY_CONVERSATIONS: usize = 32;

tokio::task_local! {
  /// Which [`ApprovalChannel`] the turn currently executing should use for any
  /// [`DualApprovalCallback`] prompts it triggers. Not set outside of
  /// [`with_approval_channel`]; [`DualApprovalCallback::call`] falls back to
  /// [`ApprovalChannel::Terminal`] when that is the case, so a caller that never heard of
  /// the web front-end (an existing test, an example, a library consumer with no web
  /// mode) keeps behaving exactly like the plain [`crate::callback::approval::
  /// ApprovalCallback`].
  static APPROVAL_CHANNEL: ApprovalChannel;
}

/// A human's answer to one approval prompt.
///
/// Carries more than a bare `bool` because "yes" and "yes, and stop asking me about this
/// tool" are different answers, and only the person answering knows which they mean. A
/// conversation that deletes twenty files would otherwise ask twenty times — and a
/// prompt answered by reflex is a prompt that has stopped being a safeguard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApprovalOutcome {
  pub approved: bool,
  /// Apply this same answer to every later call of the same tool in this conversation,
  /// without asking again. Scoped per conversation and cleared by `/reset`/`--fresh`, so
  /// "forget this conversation" also forgets what was standing permission within it.
  pub sticky: bool,
  /// Why the call was refused, to be recorded in place of it. Ignored when `approved`.
  ///
  /// The model reads this as the tool's result, so it is the difference between learning
  /// "that is not allowed" and learning *what to do instead*. A bare refusal tends to
  /// produce either a retry of the same call or a dead end, whereas "not that path — the
  /// scratch files are under /tmp" redirects it. See
  /// [`DualApprovalCallback::with_rejection_formatter`] for the run-wide default this
  /// overrides.
  pub reason: Option<String>,
}

impl ApprovalOutcome {
  /// A decision for this call only.
  pub fn once(approved: bool) -> Self {
    Self {
      approved,
      sticky: false,
      reason: None,
    }
  }

  /// A decision to apply to this tool for the rest of the conversation.
  pub fn sticky(approved: bool) -> Self {
    Self {
      approved,
      sticky: true,
      reason: None,
    }
  }

  /// Attach the reason to report to the model for a refusal.
  ///
  /// A blank reason is dropped rather than recorded: it would otherwise override the
  /// run-wide formatter (and the default) with an empty tool result, leaving the model
  /// with a failure it cannot interpret at all — worse than the generic message it
  /// replaced.
  #[must_use]
  pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
    let reason = reason.into();
    self.reason = (!reason.trim().is_empty()).then_some(reason);
    self
  }
}

/// Everything about one pending approval *except* the decision channel: what a front-end
/// needs in order to render the prompt.
///
/// Split from [`PendingApproval`] so the two can be stored apart. A decision channel is
/// single-use and cannot be cloned, whereas the prompt's description has to be renderable
/// repeatedly — including by a view that was not watching when it was raised, such as a
/// browser tab that reloaded mid-turn (see [`ApprovalRegistry::pending_snapshot`]).
/// Keeping both in one struct forced a caller that only needed the description to invent
/// a throwaway channel to stand in for the real one.
///
/// `id` is the tool call's own id — the same one the model assigned it, and the same one
/// a [`crate::agent::AgentStreamEvent::ToolCallsStarted`] the browser already received
/// carries — so no separate id scheme is needed just for approvals.
#[derive(Debug, Clone)]
pub struct ApprovalMeta {
  pub id: String,
  pub tool: String,
  /// Exactly the string the tool would be handed; see
  /// [`crate::agent::callback::ToolCallView::raw_arguments`] for why a prompt shows this
  /// rather than the parsed form.
  pub raw_arguments: String,
  /// Unix seconds at which the prompt was raised, so a snapshot can be ordered the way
  /// the prompts actually arrived rather than in `HashMap` iteration order.
  pub requested_at: i64,
}

/// One tool call waiting on a human decision.
///
/// Published by [`DualApprovalCallback`] to whichever [`ApprovalChannel::Session`] is
/// active for the current turn. The receiving end — the CLI's terminal turn loop, its
/// `POST /api/approve/{id}` route, or both — is responsible for showing it to whoever is
/// watching and resolving `decision` once someone decides.
pub struct PendingApproval {
  pub meta: ApprovalMeta,
  pub decision: oneshot::Sender<ApprovalOutcome>,
}

/// The session's in-flight approvals, keyed by tool call id.
///
/// Shared by every front-end attached to a session, which is what lets any of them answer
/// any prompt. [`Self::resolve`] removes the entry it answers, so the first decision wins
/// and a second one is a no-op rather than a panic on an already-consumed sender.
///
/// Each entry keeps its [`ApprovalMeta`] alongside the decision channel, which is what
/// makes [`Self::pending_snapshot`] possible: a front-end that joins mid-turn can ask
/// what is currently outstanding instead of having to have witnessed the
/// [`crate::agent::AgentStreamEvent`] that announced it.
///
/// A plain [`std::sync::Mutex`] is enough: every critical section is a single
/// non-blocking `HashMap` operation, never held across an `.await`.
#[derive(Default)]
pub struct ApprovalRegistry {
  pending: Mutex<HashMap<String, (ApprovalMeta, oneshot::Sender<ApprovalOutcome>)>>,
}

impl ApprovalRegistry {
  pub fn new() -> Self {
    Self::default()
  }

  /// Take custody of a prompt's decision channel, alongside the description a front-end
  /// needs to render it, so any front-end can resolve it later.
  pub fn register(&self, meta: ApprovalMeta, decision: oneshot::Sender<ApprovalOutcome>) {
    self.lock().insert(meta.id.clone(), (meta, decision));
  }

  /// Answer a pending prompt. `true` if this call is the one that decided it; `false` if
  /// it was already resolved, timed out, or never existed.
  pub fn resolve(&self, id: &str, outcome: ApprovalOutcome) -> bool {
    let Some((_, decision)) = self.lock().remove(id) else {
      return false;
    };
    // `send` fails only when the waiting side already gave up (it timed out, or its turn
    // ended) — the decision arrived too late to matter either way.
    decision.send(outcome).is_ok()
  }

  /// Drop entries without answering them, for a turn that has ended. The waiting side has
  /// already stopped waiting, so this only reclaims the map slots — without it, every
  /// abandoned prompt would linger for the life of the process.
  pub fn discard(&self, ids: &[String]) {
    let mut pending = self.lock();
    for id in ids {
      pending.remove(id);
    }
  }

  /// Whether `id` is still awaiting a decision. Lets a front-end skip prompting for
  /// something another view already answered.
  pub fn is_pending(&self, id: &str) -> bool {
    self.lock().contains_key(id)
  }

  /// Every prompt currently awaiting a decision, oldest first.
  ///
  /// This is what a front-end that joined *after* a prompt was raised asks for. A live
  /// announcement (`ChatEvent::ApprovalRequired` over the CLI's broadcast) only reaches
  /// views that were already listening, and the broadcast does not replay — so a browser
  /// tab reloaded while a turn sits waiting would otherwise never learn that anything is
  /// pending, and could only watch the approval time out. The transcript cannot cover for
  /// that either: an approval is deliberately not part of it (see this module's docs).
  pub fn pending_snapshot(&self) -> Vec<ApprovalMeta> {
    let mut items: Vec<ApprovalMeta> = self.lock().values().map(|(meta, _)| meta.clone()).collect();
    items.sort_by(compare_by_age);
    items
  }

  /// Id of the *oldest* prompt currently awaiting a decision, if any. Lets a front-end
  /// that is not running the turn — a terminal sitting at its prompt while a
  /// browser-submitted turn asks about `delete_file` — offer to answer it.
  ///
  /// Oldest rather than arbitrary: several prompts can be outstanding at once (tool calls
  /// in one round run concurrently), and a terminal answering them in `HashMap` iteration
  /// order would resolve a different one than the prompt it just displayed.
  pub fn any_pending(&self) -> Option<String> {
    self
      .lock()
      .values()
      .min_by(|left, right| compare_by_age(&left.0, &right.0))
      .map(|(meta, _)| meta.id.clone())
  }

  pub fn is_empty(&self) -> bool {
    self.lock().is_empty()
  }

  /// A poisoned mutex means a previous holder panicked while holding it. Nothing in the
  /// single-operation critical sections above can panic, so this is unreachable in
  /// practice; recovering beats turning one historical panic into a panic on every
  /// subsequent approval.
  fn lock(
    &self,
  ) -> std::sync::MutexGuard<'_, HashMap<String, (ApprovalMeta, oneshot::Sender<ApprovalOutcome>)>>
  {
    self
      .pending
      .lock()
      .unwrap_or_else(|poisoned| poisoned.into_inner())
  }
}

/// Order two prompts oldest-first, breaking ties on id.
///
/// The tie-break is what makes the ordering total: several prompts raised in the same
/// round land in the same `requested_at` second, and without it `pending_snapshot` and
/// `any_pending` could disagree about which one is "first" — the terminal would then
/// display one prompt and resolve another.
fn compare_by_age(left: &ApprovalMeta, right: &ApprovalMeta) -> std::cmp::Ordering {
  left
    .requested_at
    .cmp(&right.requested_at)
    .then_with(|| left.id.cmp(&right.id))
}

/// Interpret a typed answer to an approval prompt. `None` for anything that is not one.
///
/// The single parser for both consoles that read one: [`DualApprovalCallback::
/// prompt_terminal`] for the standalone [`ApprovalChannel::Terminal`] shape, and the
/// CLI's own REPL, which reads the answer itself when a prompt was raised by another view
/// of the session. Two parsers would drift, and the vocabulary is part of the prompt's
/// contract — a `d` accepted in one place and ignored in the other reads as a bug either
/// way.
///
/// A trailing `: reason` (or `：理由`) attaches [`ApprovalOutcome::reason`], so
/// `n: 这些是生产数据` refuses *and* tells the model why. Keeping it optional is what
/// lets the common case stay a single keypress: making every refusal prompt for an
/// explanation would tax the safe answer, and a reason typed under protest is unlikely to
/// be worth what it cost to collect.
///
/// `None` rather than a default so the caller decides what silence means: the console
/// prompt denies (fail-closed), while the REPL holds the line back and re-states the
/// question, since there a stray word is far more likely to be a mistyped message than an
/// intent to reject.
pub fn parse_approval_answer(line: &str) -> Option<ApprovalOutcome> {
  // Split on the first colon of either width, so a reason can be typed in either script
  // without the separator itself becoming a thing to get right.
  let (verdict, reason) = match line.split_once([':', '：']) {
    Some((verdict, reason)) => (verdict, Some(reason)),
    None => (line, None),
  };

  let outcome = match verdict.trim().to_ascii_lowercase().as_str() {
    "y" | "yes" | "/approve" => ApprovalOutcome::once(true),
    "n" | "no" | "/deny" => ApprovalOutcome::once(false),
    // Deliberately not `a`/`always` as a prefix of "approve": the sticky answers are the
    // consequential ones, so they get their own unambiguous words.
    "a" | "always" | "/always" => ApprovalOutcome::sticky(true),
    "d" | "never" | "/never" => ApprovalOutcome::sticky(false),
    _ => return None,
  };

  Some(match reason {
    // `with_reason` drops a blank one, so a stray trailing colon does not turn into an
    // empty tool result.
    Some(reason) => outcome.with_reason(reason.trim()),
    None => outcome,
  })
}

/// Where a [`DualApprovalCallback`] should send its prompt for the turn currently
/// executing. See [`with_approval_channel`] for how one gets attached to a turn.
#[derive(Clone)]
pub enum ApprovalChannel {
  /// Prompt on the console and block on stdin. The standalone shape, equivalent to
  /// [`crate::callback::approval::ApprovalCallback`], for a caller with exactly one
  /// front-end and no session-wide broker to publish to; also the fallback when no
  /// channel was attached at all (see [`APPROVAL_CHANNEL`]'s docs).
  Terminal,
  /// Publish a [`PendingApproval`] to the whole session and wait for any front-end to
  /// answer it. This is what a caller with more than one view of a session uses — see
  /// the module docs.
  Session(mpsc::UnboundedSender<PendingApproval>),
}

/// Attach `channel` to every [`DualApprovalCallback`] prompt triggered while `fut` runs.
///
/// Meant to wrap exactly one turn — see `run_turn`/`run_turn_stream` in `bin/cli` — not
/// an individual tool call: the channel is looked up once per call from inside
/// [`DualApprovalCallback::call`], so attaching it any more granularly than "for this
/// whole turn" would not change anything.
pub async fn with_approval_channel<F: Future>(channel: ApprovalChannel, fut: F) -> F::Output {
  APPROVAL_CHANNEL.scope(channel, fut).await
}

/// When calls to one tool need a human decision.
///
/// A tool absent from [`DualApprovalCallback`]'s rule set is never gated at all; this
/// only describes tools that are.
pub enum ApprovalRule {
  /// Every call to this tool is gated. The right rule for a tool that is dangerous
  /// regardless of how it is called, and the only sensible one for a tool that takes no
  /// arguments — a predicate would have nothing to inspect (see
  /// [`Self::When`]'s note on unreadable arguments).
  Always,
  /// Gated only for calls the predicate returns `true` for: "deleting under `/tmp` is
  /// fine, deleting anything else needs a human".
  ///
  /// The predicate sees the whole [`ToolCallView`], but the field it exists for is
  /// [`ToolCallView::arguments`] — the parsed form, since deciding by substring on the
  /// raw JSON is how a check meant to match `/tmp/x` ends up matching `/etc/tmp-x` too.
  ///
  /// **It is not consulted when those arguments cannot be read.** A payload that is
  /// empty, unparsable, or not a JSON object gives a predicate nothing to decide on, and
  /// the only safe reading of "cannot tell" for a gate on destructive operations is
  /// "ask" — so such a call is gated as if the rule were [`Self::Always`]. Without that,
  /// the cheapest way past the gate would be to emit a malformed payload: every field
  /// the predicate looks for would be missing, and a predicate phrased as "gate it when
  /// the path is outside `/tmp`" would wave it straight through.
  When(Arc<dyn Fn(&ToolCallView<'_>) -> bool + Send + Sync>),
}

impl ApprovalRule {
  /// [`Self::When`] from a plain closure, so a caller does not have to name [`Arc`].
  pub fn when(predicate: impl Fn(&ToolCallView<'_>) -> bool + Send + Sync + 'static) -> Self {
    Self::When(Arc::new(predicate))
  }
}

impl std::fmt::Debug for ApprovalRule {
  fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Always => formatter.write_str("Always"),
      // A closure has nothing printable about it, but which *variant* this is remains
      // worth seeing in a log line or a failing assertion.
      Self::When(_) => formatter.write_str("When(<predicate>)"),
    }
  }
}

/// Why a call's arguments cannot be handed to an [`ApprovalRule::When`] predicate.
///
/// Every variant means the same thing operationally — gate the call without asking the
/// predicate — but they are distinguished so the log line says which shape of payload
/// caused it, and so each is pinned by its own test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UnreadableArguments {
  /// Nothing at all was sent. Also what a no-argument tool looks like, which is why such
  /// a tool wants [`ApprovalRule::Always`] rather than a predicate.
  Empty,
  /// Not valid JSON. Covers `NaN`/`Infinity`/`-Infinity` too: `serde_json` rejects those
  /// non-standard constants outright, so they arrive here rather than as parsed numbers.
  Unparsable,
  /// Valid JSON, but not an object — `null`, an array, a bare scalar. A predicate reads
  /// named fields, and none of these has any.
  NotAnObject,
}

/// Whether `tool_call`'s arguments are unfit for a predicate, and why.
///
/// [`ToolCallView::arguments`] alone cannot answer this: it is [`Value::Null`] both when
/// the model genuinely sent `null` and when its payload failed to parse (see
/// `Agent::execute_tool_calls`, which builds the view with
/// `serde_json::from_str(..).unwrap_or(Value::Null)`). Telling those apart needs
/// [`ToolCallView::raw_arguments`] — which is also the only faithful record of what was
/// actually asked for.
fn unreadable_arguments(tool_call: &ToolCallView<'_>) -> Option<UnreadableArguments> {
  let raw = tool_call.raw_arguments.trim();
  if raw.is_empty() {
    return Some(UnreadableArguments::Empty);
  }
  if tool_call.arguments.is_null() && raw != "null" {
    return Some(UnreadableArguments::Unparsable);
  }
  if !tool_call.arguments.is_object() {
    return Some(UnreadableArguments::NotAnObject);
  }
  None
}

/// What to do about a prompt nobody answered — the timeout expired, no front-end was
/// listening, or the decision was dropped.
///
/// Note that neither option ever *allows* the call: the choice is between refusing it now
/// and asking again later.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WhenUnanswered {
  /// Refuse the call. The default, and the only sound choice for a caller that cannot
  /// come back to the question: it ends the round with a decision rather than leaving a
  /// turn holding a lock while nobody is being asked anything.
  #[default]
  Refuse,
  /// Suspend the call, so the run can stop and be resumed once someone answers. See
  /// [`ToolCallDecision::Suspend`].
  ///
  /// Safe to set even for an entry point that cannot resume: `Agent::run` and the
  /// streaming routes record a suspended call as unanswered, which lands in the same
  /// place `Refuse` would. So this expresses intent, and the runtime degrades it when the
  /// intent cannot be honoured.
  Suspend,
}

/// Builds the result recorded for a refused call that carried no reason of its own — the
/// run-wide default, as set by [`DualApprovalCallback::with_rejection_formatter`].
pub type RejectionFormatter = Arc<dyn Fn(&ToolCallView<'_>) -> String + Send + Sync>;

/// A remembered answer: the verdict, plus the reason given when it was a refusal.
///
/// The reason is kept alongside the verdict rather than discarded because a remembered
/// refusal keeps being reported to the model, and it should keep reporting what the human
/// actually said. Remembering "no, the scratch files are under /tmp" and then answering
/// every later call with a generic refusal would throw away the only part the model can
/// act on.
#[derive(Debug, Clone)]
struct StickyDecision {
  approved: bool,
  reason: Option<String>,
}

/// Asks a human before letting a gated tool run, through whichever
/// [`ApprovalChannel`] [`with_approval_channel`] attached to the turn currently running.
/// Denying — including by timeout — records an error result in place of the call, and the
/// model carries on without it.
pub struct DualApprovalCallback {
  /// Which tools are gated, and under what condition. Absent from this map means never
  /// gated. Keyed on the exact tool name: matching is neither case-insensitive nor by
  /// prefix, so `delete` does not gate `delete_file`.
  rules: HashMap<String, ApprovalRule>,
  timeout: Duration,
  /// Answers a human asked to have remembered, per conversation, keyed on tool name.
  ///
  /// [`ContinuityCache`] rather than a plain map because this callback is shared across
  /// concurrent runs and is never told when one ends (see that type's docs): anything it
  /// accumulates has to be bounded and scoped per conversation, or a long-lived process
  /// would grow one entry per conversation forever and — worse — standing permission
  /// granted in one conversation would silently apply in another.
  ///
  /// [`ExecutionContext::continuity_key`] already folds in the conversation scope, so a
  /// terminal-scoped and a web-scoped session that happen to share an id stay separate.
  sticky: Mutex<ContinuityCache<HashMap<String, StickyDecision>>>,
  /// What to record for a refusal that came with no reason of its own. `None` falls back
  /// to a generic message; see [`Self::with_rejection_formatter`].
  rejection_formatter: Option<RejectionFormatter>,
  /// See [`WhenUnanswered`].
  when_unanswered: WhenUnanswered,
  // Serializes the terminal prompt/read pair below: tool calls in the same round run
  // concurrently (see `Agent::execute_tool_calls`), and without this lock two concurrent
  // dangerous calls would interleave their console prompts and could read the wrong
  // `y`/`n` answer for the wrong tool call. A session-channel prompt does not need this:
  // each call gets its own `PendingApproval` with its own `decision` channel, so several
  // can be shown (and resolved) at once without any of them reading another's answer.
  terminal_prompt_lock: AsyncMutex<()>,
}

impl DualApprovalCallback {
  /// Gate every call to each named tool ([`ApprovalRule::Always`]).
  pub fn new(dangerous_tools: impl IntoIterator<Item = impl Into<String>>) -> Self {
    Self {
      rules: dangerous_tools
        .into_iter()
        .map(|name| (name.into(), ApprovalRule::Always))
        .collect(),
      timeout: config::approval_timeout(),
      sticky: Mutex::new(ContinuityCache::new(STICKY_CONVERSATIONS)),
      rejection_formatter: None,
      when_unanswered: WhenUnanswered::default(),
      terminal_prompt_lock: AsyncMutex::new(()),
    }
  }

  /// Gate `tool` under `rule`, replacing any rule already set for it.
  #[must_use]
  pub fn with_rule(mut self, tool: impl Into<String>, rule: ApprovalRule) -> Self {
    self.rules.insert(tool.into(), rule);
    self
  }

  /// How long to wait for a human before denying. Defaults to
  /// [`config::approval_timeout`].
  pub fn with_timeout(mut self, timeout: Duration) -> Self {
    self.timeout = timeout;
    self
  }

  /// What to record for a refusal the human gave no reason for.
  ///
  /// The run-wide default, below [`ApprovalOutcome::reason`] and above the built-in
  /// generic message. Useful for saying something the model can act on without asking a
  /// human to type it every time — "publishing is disabled in this workspace; propose the
  /// change instead" beats a bare refusal, and unlike a per-call reason it costs nothing
  /// at the prompt.
  ///
  /// It receives the call being refused, so it can name the tool or read an argument, and
  /// it also covers the refusals nobody typed an answer for at all: a timeout, a dropped
  /// decision, no front-end listening.
  #[must_use]
  pub fn with_rejection_formatter(
    mut self,
    formatter: impl Fn(&ToolCallView<'_>) -> String + Send + Sync + 'static,
  ) -> Self {
    self.rejection_formatter = Some(Arc::new(formatter));
    self
  }

  /// What to do when nobody answers a prompt. Defaults to [`WhenUnanswered::Refuse`].
  #[must_use]
  pub fn when_unanswered(mut self, policy: WhenUnanswered) -> Self {
    self.when_unanswered = policy;
    self
  }

  /// Whether this call has to be put to a human.
  ///
  /// `None` for a tool with no rule (never gated); otherwise the rule decides, with
  /// unreadable arguments forcing a prompt rather than reaching the predicate — see
  /// [`ApprovalRule::When`].
  fn requires_approval(&self, tool_call: &ToolCallView<'_>) -> bool {
    let Some(rule) = self.rules.get(tool_call.name) else {
      return false;
    };
    let ApprovalRule::When(predicate) = rule else {
      return true;
    };
    if let Some(reason) = unreadable_arguments(tool_call) {
      tracing::warn!(
        tool = %tool_call.name,
        ?reason,
        "cannot evaluate the approval predicate, asking for approval instead"
      );
      return true;
    }
    predicate(tool_call)
  }

  /// A remembered answer for `tool` in this conversation, if one was given.
  fn sticky_decision(&self, context: &ExecutionContext, tool: &str) -> Option<StickyDecision> {
    let key = context.continuity_key();
    let mut cache = self.lock_sticky();
    // `ContinuityCache::get` refreshes recency, which is what keeps an actively used
    // conversation's remembered answers from being evicted by a burst of short-lived
    // ones.
    cache.get(key.as_ref())?.get(tool).cloned()
  }

  /// Remember `outcome`'s verdict (and its reason, if a refusal) for `tool` for the rest
  /// of this conversation.
  fn remember(&self, context: &ExecutionContext, tool: &str, outcome: &ApprovalOutcome) {
    let key = context.continuity_key();
    let mut entry = HashMap::new();
    entry.insert(
      tool.to_owned(),
      StickyDecision {
        approved: outcome.approved,
        reason: outcome.reason.clone(),
      },
    );
    self
      .lock_sticky()
      .put_with(key.as_ref(), entry, |existing, incoming| {
        // Merge rather than replace: two tools can each have a remembered answer in the
        // same conversation, and a second `put_with` carrying only one of them must not
        // drop the other. The incoming answer wins on a genuine conflict — it is the more
        // recent instruction for that tool.
        let mut merged = existing.clone();
        merged.extend(incoming);
        merged
      });
  }

  /// Forget every remembered answer for the conversation identified by `continuity_key`.
  ///
  /// Called when a session is reset (`/reset`, `--fresh`). "Forget this conversation" has
  /// to include the standing permission granted inside it: leaving a remembered "always
  /// allow" in place across a reset would carry the riskiest piece of state over exactly
  /// the boundary a user asked to draw.
  pub fn forget_sticky(&self, continuity_key: &str) {
    // `ContinuityCache` has no removal, so an empty map stands in for "nothing
    // remembered". A dropped entry and an entry with no answers read the same to
    // `sticky_decision`, which looks a tool up inside whatever it finds.
    self
      .lock_sticky()
      .put_with(continuity_key, HashMap::new(), |_, incoming| incoming);
  }

  /// See [`ApprovalRegistry::lock`] for why a poisoned lock is recovered rather than
  /// propagated.
  fn lock_sticky(
    &self,
  ) -> std::sync::MutexGuard<'_, ContinuityCache<HashMap<String, StickyDecision>>> {
    self
      .sticky
      .lock()
      .unwrap_or_else(|poisoned| poisoned.into_inner())
  }

  /// Console prompt for the standalone [`ApprovalChannel::Terminal`] shape.
  ///
  /// `None` when no answer arrived — stdin could not be read (a non-interactive process,
  /// say), what was typed was not an answer, or nobody answered within the timeout. See
  /// [`Self::prompt_session`] for why that is kept distinct from an explicit refusal.
  async fn prompt_terminal(&self, tool_call: &ToolCallView<'_>) -> Option<ApprovalOutcome> {
    let _guard = self.terminal_prompt_lock.lock().await;

    eprintln!("\n⚠️  即将执行高危操作");
    eprintln!("工具: {}", tool_call.name);
    // The raw string rather than the parsed arguments: an unparseable payload would show
    // up as `null`, and approving a call whose arguments you cannot see is worse than no
    // prompt at all.
    eprintln!("参数: {}", tool_call.raw_arguments);

    let read = tokio::task::spawn_blocking(|| {
      eprint!("是否执行？(y=本次允许 / n=本次拒绝 / a=本会话总是允许 / d=本会话总是拒绝): ");
      if let Err(err) = io::stderr().flush() {
        tracing::warn!("failed to flush approval prompt: {err}");
      }
      let mut input = String::new();
      if let Err(err) = io::stdin().read_line(&mut input) {
        tracing::warn!("failed to read approval answer: {err}");
        return None;
      }
      parse_approval_answer(&input)
    });

    // A timed-out read is abandoned, not cancelled: a blocking stdin read cannot be
    // interrupted, so its thread stays parked until a line eventually arrives (and is
    // then discarded).
    //
    // That leaves a reader sitting on stdin. Harmless for a caller whose only use of the
    // console is this prompt — the stray line is simply swallowed — but a caller that
    // *also* drives a line editor must not let the two overlap: a full-screen editor puts
    // the terminal in raw mode and queries it for the cursor position, and this reader
    // would consume the reply. Such callers should use `ApprovalChannel::Session` instead,
    // which keeps the console read under their own control (see `bin/cli`).
    let outcome = match tokio::time::timeout(self.timeout, read).await {
      // A line that is not an answer counts as no answer, same as a closed stdin: the
      // console has no way to ask again, so it is left to the caller's policy.
      Ok(result) => result.ok().flatten(),
      Err(_) => {
        eprintln!("\n⏳ 审批超时");
        None
      }
    };

    match outcome
      .as_ref()
      .map(|outcome| (outcome.approved, outcome.sticky))
    {
      Some((true, false)) => eprintln!("✅ 已批准，继续执行...\n"),
      Some((true, true)) => eprintln!("✅ 已批准，本会话内不再询问该工具...\n"),
      Some((false, false)) => eprintln!("❌ 已拒绝，跳过执行\n"),
      Some((false, true)) => eprintln!("❌ 已拒绝，本会话内将自动拒绝该工具\n"),
      None => {}
    }
    outcome
  }

  /// Publishes a [`PendingApproval`] to the session and waits for any front-end to answer
  /// it.
  ///
  /// `None` when no answer arrived at all — nobody was listening, whoever received it
  /// dropped the decision, or the timeout expired. Distinguished from `Some(refused)`
  /// because the absence of an answer is not a decision, and only the caller knows
  /// whether the question can be asked again later (see [`Self::when_unanswered`]).
  async fn prompt_session(
    sender: &mpsc::UnboundedSender<PendingApproval>,
    tool_call: &ToolCallView<'_>,
    timeout: Duration,
  ) -> Option<ApprovalOutcome> {
    let (decision_tx, decision_rx) = oneshot::channel();
    let request = PendingApproval {
      meta: ApprovalMeta {
        id: tool_call.tool_call_id.to_owned(),
        tool: tool_call.name.to_owned(),
        raw_arguments: tool_call.raw_arguments.to_owned(),
        requested_at: chrono::Utc::now().timestamp(),
      },
      decision: decision_tx,
    };

    if sender.send(request).is_err() {
      tracing::warn!("no front-end is listening for approvals");
      return None;
    }

    match tokio::time::timeout(timeout, decision_rx).await {
      Ok(Ok(outcome)) => Some(outcome),
      // Every front-end holding the decision dropped it without answering.
      Ok(Err(_)) => {
        tracing::warn!("approval was abandoned without a decision");
        None
      }
      Err(_) => {
        tracing::warn!(
          tool = %tool_call.name,
          timeout_secs = timeout.as_secs(),
          "nobody approved in time"
        );
        None
      }
    }
  }

  /// The result recorded in place of a call that was not allowed to run.
  ///
  /// Three sources for the explanation, most specific first: the reason this particular
  /// answer carried, the run-wide [`Self::with_rejection_formatter`], then nothing. All
  /// are recorded as [`ToolResultStatus::Error`] — from the model's side a refusal is
  /// simply a tool call that did not succeed, and it needs no concept of "approval" to
  /// carry on.
  ///
  /// # Why the explanation is always prefixed
  ///
  /// Whatever the explanation is, it is reported as `User denied execution of <tool>:
  /// <reason>` rather than on its own. The prefix is not decoration — it marks the text
  /// as *the decision of the person running this agent*, and a refusal reason without it
  /// is indistinguishable from content the tool itself emitted.
  ///
  /// That distinction decides whether the reason works at all. A well-aligned model
  /// treats instructions surfacing from tool output as untrusted — which is correct, and
  /// exactly what it should do — so a bare "delete tmp-b.log instead" reads to it as an
  /// injection attempt smuggled through a tool result, and it refuses to act on it. This
  /// was observed in practice: the model declined the redirection and said so, naming
  /// prompt injection as the reason. Prefixed, the same text is attributable to the human
  /// and can be acted on.
  fn denial(&self, tool_call: &ToolCallView<'_>, reason: Option<String>) -> ToolCallDecision {
    let explanation = reason.or_else(|| {
      self
        .rejection_formatter
        .as_ref()
        .map(|format| format(tool_call))
    });
    let content = match explanation {
      Some(reason) => format!("User denied execution of {}: {reason}", tool_call.name),
      None => format!("User denied execution of {}", tool_call.name),
    };
    ToolCallDecision::deny(content)
  }
}

#[async_trait::async_trait]
impl BeforeToolCallback for DualApprovalCallback {
  async fn call(
    &self,
    context: &ExecutionContext,
    tool_call: ToolCallView<'_>,
  ) -> ToolCallDecision {
    if !self.requires_approval(&tool_call) {
      return ToolCallDecision::Proceed;
    }

    // A remembered answer short-circuits before any prompt is raised — which is the
    // whole point: no event is published, so no front-end shows a prompt that was
    // already decided, and nothing can time out.
    if let Some(remembered) = self.sticky_decision(context, tool_call.name) {
      tracing::debug!(
        tool = %tool_call.name,
        approved = remembered.approved,
        "applying a remembered approval decision without prompting"
      );
      return if remembered.approved {
        ToolCallDecision::Proceed
      } else {
        // Carries the reason forward, so a remembered refusal keeps telling the model
        // what the human originally said rather than degrading to the generic message.
        self.denial(&tool_call, remembered.reason)
      };
    }

    let channel = APPROVAL_CHANNEL
      .try_with(Clone::clone)
      .unwrap_or(ApprovalChannel::Terminal);

    let outcome = match channel {
      ApprovalChannel::Terminal => self.prompt_terminal(&tool_call).await,
      ApprovalChannel::Session(sender) => {
        Self::prompt_session(&sender, &tool_call, self.timeout).await
      }
    };

    // Nobody answered. Not a decision, so nothing is remembered and nothing is inferred
    // about what the answer would have been — see `WhenUnanswered`.
    let Some(outcome) = outcome else {
      return match self.when_unanswered {
        WhenUnanswered::Refuse => self.denial(&tool_call, None),
        WhenUnanswered::Suspend => ToolCallDecision::Suspend,
      };
    };

    if outcome.sticky {
      self.remember(context, tool_call.name, &outcome);
    }

    if outcome.approved {
      ToolCallDecision::Proceed
    } else {
      self.denial(&tool_call, outcome.reason)
    }
  }
}

#[cfg(test)]
mod tests {
  use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
  };

  use serde_json::{Value, json};

  use super::*;
  use crate::agent::ToolResultStatus;

  /// A view whose raw payload is a well-formed (if empty) object, so nothing about it
  /// trips the unreadable-arguments check — what every test that is not specifically
  /// about a malformed payload wants.
  fn view<'a>(name: &'a str, arguments: &'a Value) -> ToolCallView<'a> {
    raw_view(name, arguments, "{}")
  }

  /// A view with `raw_arguments` stated independently of `arguments`, for the cases where
  /// the two genuinely disagree — a payload the model sent that failed to parse, which is
  /// exactly what [`unreadable_arguments`] exists to detect.
  fn raw_view<'a>(name: &'a str, arguments: &'a Value, raw_arguments: &'a str) -> ToolCallView<'a> {
    ToolCallView {
      tool_call_id: "call-1",
      name,
      arguments,
      raw_arguments,
    }
  }

  /// Whether `tool_call` was put to a human at all.
  ///
  /// Answers on a session channel that denies immediately, so the return value reflects
  /// only the gating decision and not any waiting: `true` means a prompt was raised (and
  /// refused), `false` means the call was never gated in the first place.
  async fn gated(approval: &DualApprovalCallback, tool_call: ToolCallView<'_>) -> bool {
    let context = ExecutionContext::new();
    let (tx, mut rx) = mpsc::unbounded_channel::<PendingApproval>();
    let call = with_approval_channel(
      ApprovalChannel::Session(tx),
      approval.call(&context, tool_call),
    );
    let deny = async {
      if let Some(pending) = rx.recv().await {
        let _ = pending.decision.send(ApprovalOutcome::once(false));
      }
    };
    let (result, ()) = tokio::join!(call, deny);
    !matches!(result, ToolCallDecision::Proceed)
  }

  /// An [`ApprovalMeta`] for a registry test, where only `id` matters. `requested_at` is
  /// fixed rather than "now" so the age-ordering tests can set it deliberately.
  fn meta(id: &str) -> ApprovalMeta {
    ApprovalMeta {
      id: id.to_owned(),
      tool: "delete_file".to_owned(),
      raw_arguments: "{}".to_owned(),
      requested_at: 0,
    }
  }

  /// Long enough that no test below reaches it by accident; the timeout path has its own
  /// test that sets a deliberately tiny one.
  fn approval() -> DualApprovalCallback {
    DualApprovalCallback::new(["delete_file"]).with_timeout(Duration::from_secs(30))
  }

  // None of the early cases below name a tool the callback treats as dangerous, so `call`
  // returns from its guard clause before ever prompting — meaning it never touches
  // stdin/stderr, nor does it need an `ApprovalChannel` attached, and cannot block the
  // test suite waiting for an answer.

  #[tokio::test]
  async fn lets_a_tool_outside_the_dangerous_list_through_untouched() {
    let context = ExecutionContext::new();
    let args = json!({ "path": "notes.txt" });

    assert!(
      approval()
        .call(&context, view("read_file", &args))
        .await
        .is_proceed()
    );
  }

  #[tokio::test]
  async fn matches_dangerous_tool_names_exactly_not_as_a_substring() {
    let approval = DualApprovalCallback::new(["delete"]);
    let context = ExecutionContext::new();
    let args = json!({});

    assert!(
      approval
        .call(&context, view("delete_file", &args))
        .await
        .is_proceed()
    );
  }

  // ---- `ApprovalRule::When`: deciding by argument ----------------------------------

  /// The point of a predicate: the same tool is waved through or stopped depending on
  /// what it was asked to do.
  #[tokio::test]
  async fn a_predicate_gates_only_the_calls_it_selects() {
    let approval = DualApprovalCallback::new(Vec::<String>::new()).with_rule(
      "delete_file",
      ApprovalRule::when(|call| {
        !call.arguments["path"]
          .as_str()
          .is_some_and(|path| path.starts_with("/tmp/"))
      }),
    );

    let inside_tmp = json!({ "path": "/tmp/scratch" });
    assert!(
      !gated(&approval, view("delete_file", &inside_tmp)).await,
      "a path the predicate accepts must not be gated"
    );

    let outside_tmp = json!({ "path": "/etc/passwd" });
    assert!(
      gated(&approval, view("delete_file", &outside_tmp)).await,
      "a path the predicate rejects must be gated"
    );
  }

  /// A rule set through `with_rule` replaces the blanket one `new` installed, rather than
  /// both applying.
  #[tokio::test]
  async fn with_rule_replaces_an_always_rule_from_new() {
    let approval = DualApprovalCallback::new(["delete_file"])
      .with_rule("delete_file", ApprovalRule::when(|_| false));
    let args = json!({ "path": "notes.txt" });

    assert!(
      !gated(&approval, view("delete_file", &args)).await,
      "the later rule is the one in force"
    );
  }

  /// A predicate only governs the tool it was attached to; anything else stays ungated.
  #[tokio::test]
  async fn a_rule_applies_only_to_its_own_tool() {
    let approval = DualApprovalCallback::new(Vec::<String>::new())
      .with_rule("delete_file", ApprovalRule::when(|_| true));
    let args = json!({ "path": "notes.txt" });

    assert!(!gated(&approval, view("read_file", &args)).await);
  }

  // ---- fail-closed: unreadable arguments never reach the predicate -----------------

  /// The case that makes fail-closed necessary rather than merely tidy: a predicate
  /// phrased as "gate it unless the path is under `/tmp`" reads a missing field as "not
  /// under /tmp"... but one phrased the other way round would wave a malformed payload
  /// straight through. So an unreadable payload must not reach the predicate at all.
  #[tokio::test]
  async fn unreadable_arguments_are_gated_without_consulting_the_predicate() {
    // Only the raw payload is stated: `arguments` is derived from it below exactly the
    // way the runtime derives it, so a case cannot accidentally describe a pairing that
    // could never actually occur.
    let cases = [
      ("nothing at all", ""),
      ("whitespace only", "   "),
      ("malformed json", "{\"path\": "),
      ("a bare null", "null"),
      ("an array", "[1, 2]"),
      ("a scalar", "42"),
      ("nan", "{\"n\": NaN}"),
      ("infinity", "{\"n\": Infinity}"),
      ("negative infinity", "{\"n\": -Infinity}"),
    ];

    for (label, raw) in cases {
      let calls = Arc::new(AtomicUsize::new(0));
      let counter = Arc::clone(&calls);
      // Returns `false` — "no approval needed". If the predicate were consulted, the
      // call would sail through, so `gated` being true proves it was bypassed.
      let approval = DualApprovalCallback::new(Vec::<String>::new()).with_rule(
        "delete_file",
        ApprovalRule::when(move |_| {
          counter.fetch_add(1, Ordering::SeqCst);
          false
        }),
      );

      // Mirrors `Agent::execute_tool_calls`: anything that fails to parse becomes
      // `Value::Null`.
      let arguments = serde_json::from_str::<Value>(raw).unwrap_or(Value::Null);

      assert!(
        gated(&approval, raw_view("delete_file", &arguments, raw)).await,
        "{label}: an unreadable payload must be gated"
      );
      assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "{label}: the predicate must not be consulted at all"
      );
    }
  }

  /// Readable arguments, by contrast, do reach the predicate.
  #[tokio::test]
  async fn readable_arguments_reach_the_predicate() {
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&calls);
    let approval = DualApprovalCallback::new(Vec::<String>::new()).with_rule(
      "delete_file",
      ApprovalRule::when(move |_| {
        counter.fetch_add(1, Ordering::SeqCst);
        false
      }),
    );
    let args = json!({ "path": "notes.txt" });

    assert!(
      !gated(
        &approval,
        raw_view("delete_file", &args, r#"{"path":"notes.txt"}"#)
      )
      .await
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
  }

  /// `Always` does not care whether the arguments are readable — it gates either way, so
  /// the fail-closed path must not change its behaviour.
  #[tokio::test]
  async fn an_always_rule_gates_regardless_of_the_payload() {
    let approval = DualApprovalCallback::new(["delete_file"]);
    let args = Value::Null;

    assert!(gated(&approval, raw_view("delete_file", &args, "")).await);
  }

  #[test]
  fn unreadable_arguments_classifies_each_shape() {
    let null = Value::Null;
    let object = json!({ "path": "notes.txt" });
    let array = json!([1, 2]);

    assert_eq!(
      unreadable_arguments(&raw_view("t", &null, "")),
      Some(UnreadableArguments::Empty)
    );
    assert_eq!(
      unreadable_arguments(&raw_view("t", &null, "  \n ")),
      Some(UnreadableArguments::Empty),
      "whitespace is as empty as nothing"
    );
    assert_eq!(
      unreadable_arguments(&raw_view("t", &null, "{\"path\": ")),
      Some(UnreadableArguments::Unparsable)
    );
    assert_eq!(
      unreadable_arguments(&raw_view("t", &null, "null")),
      Some(UnreadableArguments::NotAnObject),
      "a genuine `null` parsed fine; it is just not an object"
    );
    assert_eq!(
      unreadable_arguments(&raw_view("t", &array, "[1, 2]")),
      Some(UnreadableArguments::NotAnObject)
    );
    assert_eq!(
      unreadable_arguments(&raw_view("t", &object, r#"{"path":"notes.txt"}"#)),
      None,
      "a plain object is exactly what a predicate wants"
    );
  }

  /// `Value::Null` means two different things on the way in — the model sent `null`, or
  /// its payload did not parse — and the raw string is the only way to tell. Both are
  /// unreadable, but they must be reported as different reasons, or the log line for a
  /// malformed payload would claim the model sent `null`.
  #[test]
  fn a_genuine_null_is_distinguished_from_a_parse_failure() {
    let null = Value::Null;

    assert_eq!(
      unreadable_arguments(&raw_view("t", &null, "null")),
      Some(UnreadableArguments::NotAnObject)
    );
    assert_eq!(
      unreadable_arguments(&raw_view("t", &null, "{oops")),
      Some(UnreadableArguments::Unparsable)
    );
  }

  #[tokio::test]
  async fn session_channel_approves_when_a_front_end_says_yes() {
    let context = ExecutionContext::new();
    let args = json!({ "path": "notes.txt" });
    let (tx, mut rx) = mpsc::unbounded_channel::<PendingApproval>();

    let approval = approval();
    let call_future = with_approval_channel(
      ApprovalChannel::Session(tx),
      approval.call(&context, view("delete_file", &args)),
    );

    let respond_future = async {
      let request = rx.recv().await.expect("a request should have been sent");
      assert_eq!(request.meta.tool, "delete_file");
      request.decision.send(ApprovalOutcome::once(true)).unwrap();
    };

    let (result, ()) = tokio::join!(call_future, respond_future);
    assert!(
      matches!(result, ToolCallDecision::Proceed),
      "approved calls are not short-circuited"
    );
  }

  #[tokio::test]
  async fn session_channel_denies_when_a_front_end_says_no() {
    let context = ExecutionContext::new();
    let args = json!({});
    let (tx, mut rx) = mpsc::unbounded_channel::<PendingApproval>();

    let approval = approval();
    let call_future = with_approval_channel(
      ApprovalChannel::Session(tx),
      approval.call(&context, view("delete_file", &args)),
    );

    let respond_future = async {
      let request = rx.recv().await.expect("a request should have been sent");
      request.decision.send(ApprovalOutcome::once(false)).unwrap();
    };

    let (result, ()) = tokio::join!(call_future, respond_future);
    assert!(matches!(
      result,
      ToolCallDecision::ShortCircuit(ToolResultStatus::Error, _)
    ));
  }

  #[tokio::test]
  async fn session_channel_denies_when_nothing_is_listening() {
    let context = ExecutionContext::new();
    let args = json!({});

    let (tx, rx) = mpsc::unbounded_channel::<PendingApproval>();
    drop(rx); // No front-end attached: `sender.send` fails immediately.

    let result = with_approval_channel(
      ApprovalChannel::Session(tx),
      approval().call(&context, view("delete_file", &args)),
    )
    .await;

    assert!(matches!(
      result,
      ToolCallDecision::ShortCircuit(ToolResultStatus::Error, _)
    ));
  }

  #[tokio::test]
  async fn session_channel_denies_when_the_decision_is_dropped_unused() {
    let context = ExecutionContext::new();
    let args = json!({});
    let (tx, mut rx) = mpsc::unbounded_channel::<PendingApproval>();

    let approval = approval();
    let call_future = with_approval_channel(
      ApprovalChannel::Session(tx),
      approval.call(&context, view("delete_file", &args)),
    );

    let drop_future = async {
      // Received but never decided: dropping `decision` closes the oneshot, so the
      // waiting side resolves to `Err` rather than hanging.
      let request = rx.recv().await.expect("a request should have been sent");
      drop(request.decision);
    };

    let (result, ()) = tokio::join!(call_future, drop_future);
    assert!(matches!(
      result,
      ToolCallDecision::ShortCircuit(ToolResultStatus::Error, _)
    ));
  }

  /// The case that used to wedge the whole session: a front-end receives the prompt,
  /// holds on to the decision, and nobody ever answers.
  #[tokio::test]
  async fn session_channel_denies_once_the_timeout_expires() {
    let approval =
      DualApprovalCallback::new(["delete_file"]).with_timeout(Duration::from_millis(30));
    let context = ExecutionContext::new();
    let args = json!({});
    let (tx, mut rx) = mpsc::unbounded_channel::<PendingApproval>();

    // Held, never resolved — exactly what an abandoned browser tab leaves behind.
    let hold_future = async {
      let request = rx.recv().await.expect("a request should have been sent");
      tokio::time::sleep(Duration::from_secs(30)).await;
      drop(request);
    };

    let result = tokio::select! {
      result = with_approval_channel(
        ApprovalChannel::Session(tx),
        approval.call(&context, view("delete_file", &args)),
      ) => result,
      () = hold_future => panic!("the approval should have timed out first"),
    };

    assert!(
      matches!(
        result,
        ToolCallDecision::ShortCircuit(ToolResultStatus::Error, _)
      ),
      "an unanswered approval must deny rather than hang"
    );
  }

  /// The regression behind "reject in the browser, and both front-ends go silent": a
  /// decision from another view has to unblock the waiting side *immediately*, without
  /// the view that raised the prompt having to type anything. A driver that awaited its
  /// own console read inline would stop pumping the agent's event stream here, so nothing
  /// would be printed anywhere until someone pressed Enter locally.
  #[tokio::test]
  async fn another_view_can_resolve_while_the_raising_view_is_still_asking() {
    let registry = Arc::new(ApprovalRegistry::new());
    let approval = DualApprovalCallback::new(["delete_file"]).with_timeout(Duration::from_secs(30));
    let context = ExecutionContext::new();
    let args = json!({});
    let (tx, mut rx) = mpsc::unbounded_channel::<PendingApproval>();

    let registry_for_view = Arc::clone(&registry);
    let raising_view = async move {
      let pending = rx.recv().await.expect("a prompt should have been raised");
      let id = pending.meta.id.clone();
      registry_for_view.register(pending.meta, pending.decision);
      // Stands in for a console read nobody ever answers.
      tokio::time::sleep(Duration::from_secs(30)).await;
      id
    };

    let other_view = async {
      // Wait for the prompt to be registered, then answer from "elsewhere".
      while registry.any_pending().is_none() {
        tokio::task::yield_now().await;
      }
      let id = registry.any_pending().expect("checked just above");
      assert!(
        registry.resolve(&id, ApprovalOutcome::once(false)),
        "the other view decides"
      );
    };

    let result = tokio::select! {
      result = with_approval_channel(
        ApprovalChannel::Session(tx),
        approval.call(&context, view("delete_file", &args)),
      ) => result,
      _ = async { tokio::join!(raising_view, other_view) } => {
        panic!("the approval should have resolved without a local answer")
      }
    };

    assert!(
      matches!(
        result,
        ToolCallDecision::ShortCircuit(ToolResultStatus::Error, _)
      ),
      "a rejection from another view must deny the call"
    );
  }

  // ---- sticky decisions -------------------------------------------------------------

  /// A context keyed like a real conversation, so remembered answers land in a bucket
  /// `forget_sticky` can name.
  fn conversation(scope: &str, id: &str) -> ExecutionContext {
    let mut context = ExecutionContext::new();
    context.conversation_id = Some(id.to_owned());
    context.conversation_scope = Some(scope.to_owned());
    context
  }

  /// Answer one prompt with `outcome`, returning whether the call was allowed through.
  async fn answer_once(
    approval: &DualApprovalCallback,
    context: &ExecutionContext,
    tool_call: ToolCallView<'_>,
    outcome: ApprovalOutcome,
  ) -> bool {
    let (tx, mut rx) = mpsc::unbounded_channel::<PendingApproval>();
    let call = with_approval_channel(
      ApprovalChannel::Session(tx),
      approval.call(context, tool_call),
    );
    let respond = async {
      let pending = rx.recv().await.expect("a prompt should have been raised");
      let _ = pending.decision.send(outcome);
    };
    let (result, ()) = tokio::join!(call, respond);
    matches!(result, ToolCallDecision::Proceed)
  }

  /// Attempt a call with no front-end listening.
  ///
  /// [`ToolCallDecision::Proceed`] therefore means the call never needed a prompt — which
  /// is how a remembered *approval* is distinguished from a fresh one; anything else means
  /// a prompt was needed and failed closed for want of anyone to ask.
  async fn call_without_a_front_end(
    approval: &DualApprovalCallback,
    context: &ExecutionContext,
    tool_call: ToolCallView<'_>,
  ) -> ToolCallDecision {
    let (tx, rx) = mpsc::unbounded_channel::<PendingApproval>();
    drop(rx);
    with_approval_channel(
      ApprovalChannel::Session(tx),
      approval.call(context, tool_call),
    )
    .await
  }

  /// The point of a sticky answer: the second call of the same tool is not asked about.
  /// Verified by leaving no front-end attached — a fresh prompt would fail closed, so
  /// getting through proves no prompt was raised.
  #[tokio::test]
  async fn a_sticky_approval_applies_without_asking_again() {
    let approval = approval();
    let context = conversation("local", "s1");
    let args = json!({ "path": "notes.txt" });

    assert!(
      answer_once(
        &approval,
        &context,
        view("delete_file", &args),
        ApprovalOutcome::sticky(true)
      )
      .await
    );

    assert!(
      call_without_a_front_end(&approval, &context, view("delete_file", &args))
        .await
        .is_proceed(),
      "a remembered approval must apply without raising a prompt"
    );
  }

  /// The same mechanism in the other direction: a remembered rejection denies without
  /// asking, and without waiting out the timeout.
  #[tokio::test]
  async fn a_sticky_rejection_applies_without_asking_again() {
    let approval = approval();
    let context = conversation("local", "s1");
    let args = json!({ "path": "notes.txt" });

    assert!(
      !answer_once(
        &approval,
        &context,
        view("delete_file", &args),
        ApprovalOutcome::sticky(false)
      )
      .await
    );

    // No front-end *and* no waiting: were a prompt raised, this would deny for that
    // reason instead — so the assertion below is backed by the previous test showing the
    // approval direction is distinguishable at all.
    assert!(
      matches!(
        call_without_a_front_end(&approval, &context, view("delete_file", &args)).await,
        ToolCallDecision::ShortCircuit(ToolResultStatus::Error, _)
      ),
      "a remembered rejection must deny without asking"
    );
  }

  /// A one-off answer is exactly that: the next call asks again.
  #[tokio::test]
  async fn a_one_off_answer_is_not_remembered() {
    let approval = approval();
    let context = conversation("local", "s1");
    let args = json!({ "path": "notes.txt" });

    assert!(
      answer_once(
        &approval,
        &context,
        view("delete_file", &args),
        ApprovalOutcome::once(true)
      )
      .await
    );

    assert!(
      !call_without_a_front_end(&approval, &context, view("delete_file", &args))
        .await
        .is_proceed(),
      "a one-off approval must not carry over to the next call"
    );
  }

  /// Remembering is per tool, not per conversation: allowing one tool must not allow a
  /// different one that happens to be gated in the same conversation.
  #[tokio::test]
  async fn a_sticky_answer_covers_only_its_own_tool() {
    let approval = DualApprovalCallback::new(["delete_file", "shell_exec"])
      .with_timeout(Duration::from_secs(30));
    let context = conversation("local", "s1");
    let args = json!({ "path": "notes.txt" });

    assert!(
      answer_once(
        &approval,
        &context,
        view("delete_file", &args),
        ApprovalOutcome::sticky(true)
      )
      .await
    );

    assert!(
      !call_without_a_front_end(&approval, &context, view("shell_exec", &args))
        .await
        .is_proceed(),
      "the other tool was never decided and must still be gated"
    );
  }

  /// Remembered answers are scoped per conversation. Two conversations sharing this
  /// callback — a terminal-scoped and a web-scoped session, say — must not inherit each
  /// other's standing permission.
  #[tokio::test]
  async fn a_sticky_answer_does_not_leak_across_conversations() {
    let approval = approval();
    let args = json!({ "path": "notes.txt" });

    assert!(
      answer_once(
        &approval,
        &conversation("local", "s1"),
        view("delete_file", &args),
        ApprovalOutcome::sticky(true)
      )
      .await
    );

    for (scope, id, why) in [
      ("local", "s2", "a different session in the same scope"),
      ("web", "s1", "the same session id in a different scope"),
    ] {
      assert!(
        !call_without_a_front_end(
          &approval,
          &conversation(scope, id),
          view("delete_file", &args)
        )
        .await
        .is_proceed(),
        "{why} must not inherit the remembered answer"
      );
    }
  }

  /// Two tools can each hold a remembered answer at once: the second `remember` must
  /// merge into the conversation's entry rather than replace it.
  #[tokio::test]
  async fn remembering_a_second_tool_keeps_the_first() {
    let approval = DualApprovalCallback::new(["delete_file", "shell_exec"])
      .with_timeout(Duration::from_secs(30));
    let context = conversation("local", "s1");
    let args = json!({ "path": "notes.txt" });

    for tool in ["delete_file", "shell_exec"] {
      assert!(
        answer_once(
          &approval,
          &context,
          view(tool, &args),
          ApprovalOutcome::sticky(true)
        )
        .await
      );
    }

    for tool in ["delete_file", "shell_exec"] {
      assert!(
        call_without_a_front_end(&approval, &context, view(tool, &args))
          .await
          .is_proceed(),
        "{tool}'s remembered answer should have survived the other's"
      );
    }
  }

  /// `/reset` and `--fresh` route here. Standing permission for a destructive tool is the
  /// one piece of session state where surviving a reset would be actively dangerous.
  #[tokio::test]
  async fn forgetting_a_conversation_drops_its_remembered_answers() {
    let approval = approval();
    let context = conversation("local", "s1");
    let args = json!({ "path": "notes.txt" });

    assert!(
      answer_once(
        &approval,
        &context,
        view("delete_file", &args),
        ApprovalOutcome::sticky(true)
      )
      .await
    );

    approval.forget_sticky(&context.continuity_key());

    assert!(
      !call_without_a_front_end(&approval, &context, view("delete_file", &args))
        .await
        .is_proceed(),
      "after forgetting, the tool must be gated again"
    );
  }

  /// A timeout is the absence of an answer. Recording it as standing permission — in
  /// either direction — would invent an instruction nobody gave, and in the deny
  /// direction would silently disable the tool for the rest of the conversation.
  #[tokio::test]
  async fn a_timeout_is_never_remembered() {
    let approval =
      DualApprovalCallback::new(["delete_file"]).with_timeout(Duration::from_millis(30));
    let context = conversation("local", "s1");
    let args = json!({ "path": "notes.txt" });

    // Received and held, never answered — the prompt times out.
    let (tx, mut rx) = mpsc::unbounded_channel::<PendingApproval>();
    let hold = async {
      let pending = rx.recv().await.expect("a prompt should have been raised");
      tokio::time::sleep(Duration::from_secs(30)).await;
      drop(pending);
    };
    let timed_out = tokio::select! {
      result = with_approval_channel(
        ApprovalChannel::Session(tx),
        approval.call(&context, view("delete_file", &args)),
      ) => result,
      () = hold => panic!("the approval should have timed out first"),
    };
    assert!(matches!(
      timed_out,
      ToolCallDecision::ShortCircuit(ToolResultStatus::Error, _)
    ));

    // The next call must still raise a prompt rather than reuse the timeout as a
    // remembered rejection.
    let (tx, mut rx) = mpsc::unbounded_channel::<PendingApproval>();
    let call = with_approval_channel(
      ApprovalChannel::Session(tx),
      approval.call(&context, view("delete_file", &args)),
    );
    let observe = async { rx.recv().await.is_some() };
    let (_, prompted) = tokio::join!(call, observe);
    assert!(prompted, "a timeout must not be remembered as a decision");
  }

  #[test]
  fn approval_answers_are_parsed_from_their_accepted_spellings() {
    for line in ["y", "Y", " yes ", "/approve"] {
      assert_eq!(
        parse_approval_answer(line),
        Some(ApprovalOutcome::once(true))
      );
    }
    for line in ["n", "no", "/deny"] {
      assert_eq!(
        parse_approval_answer(line),
        Some(ApprovalOutcome::once(false))
      );
    }
    for line in ["a", "always", "/always"] {
      assert_eq!(
        parse_approval_answer(line),
        Some(ApprovalOutcome::sticky(true))
      );
    }
    for line in ["d", "never", "/never"] {
      assert_eq!(
        parse_approval_answer(line),
        Some(ApprovalOutcome::sticky(false))
      );
    }
  }

  /// Anything else is *not* an answer. The callers rely on this being distinguishable
  /// from a real "no": the console prompt denies, while the REPL holds the line back and
  /// re-states the question rather than rejecting on a typo.
  #[test]
  fn an_unrecognized_line_is_not_an_answer() {
    for line in ["", "  ", "maybe", "what is 2+2?", "approve", "yep"] {
      assert_eq!(
        parse_approval_answer(line),
        None,
        "{line:?} must not be read as a decision"
      );
    }
  }

  // ---- rejection messages -----------------------------------------------------------

  /// What was recorded in place of a refused call.
  async fn denial_text(
    approval: &DualApprovalCallback,
    context: &ExecutionContext,
    tool_call: ToolCallView<'_>,
    outcome: ApprovalOutcome,
  ) -> String {
    let (tx, mut rx) = mpsc::unbounded_channel::<PendingApproval>();
    let call = with_approval_channel(
      ApprovalChannel::Session(tx),
      approval.call(context, tool_call),
    );
    let respond = async {
      let pending = rx.recv().await.expect("a prompt should have been raised");
      let _ = pending.decision.send(outcome);
    };
    let (result, ()) = tokio::join!(call, respond);
    denial_content(result)
  }

  /// The content a refusal recorded, for asserting on what the model is actually told.
  /// Panics on anything but a refusal, so a test that meant to check a message cannot
  /// quietly pass against a call that was allowed through.
  fn denial_content(decision: ToolCallDecision) -> String {
    match decision {
      ToolCallDecision::ShortCircuit(ToolResultStatus::Error, content) => content,
      other => panic!("expected a refusal, got {other:?}"),
    }
  }

  #[tokio::test]
  async fn a_refusal_without_a_reason_records_the_generic_message() {
    let args = json!({ "path": "notes.txt" });
    let text = denial_text(
      &approval(),
      &conversation("local", "s1"),
      view("delete_file", &args),
      ApprovalOutcome::once(false),
    )
    .await;

    assert_eq!(text, "User denied execution of delete_file");
  }

  /// The reason is what the model actually reads, so it has to reach the transcript
  /// verbatim rather than being summarized into the generic message.
  #[tokio::test]
  async fn a_reason_given_at_the_prompt_is_what_gets_recorded() {
    let args = json!({ "path": "notes.txt" });
    let text = denial_text(
      &approval(),
      &conversation("local", "s1"),
      view("delete_file", &args),
      ApprovalOutcome::once(false).with_reason("scratch files live under /tmp"),
    )
    .await;

    assert_eq!(
      text, "User denied execution of delete_file: scratch files live under /tmp",
      "the reason is attributed to the user, not left to look like tool output"
    );
  }

  #[tokio::test]
  async fn the_run_wide_formatter_covers_refusals_with_no_reason() {
    let approval = DualApprovalCallback::new(["delete_file"])
      .with_timeout(Duration::from_secs(30))
      .with_rejection_formatter(|call| format!("{} is disabled in this workspace", call.name));
    let args = json!({ "path": "notes.txt" });

    let text = denial_text(
      &approval,
      &conversation("local", "s1"),
      view("delete_file", &args),
      ApprovalOutcome::once(false),
    )
    .await;

    assert_eq!(
      text,
      "User denied execution of delete_file: delete_file is disabled in this workspace"
    );
  }

  /// Three sources, most specific first. A per-call reason has to win: it is the one
  /// someone typed about *this* call.
  #[tokio::test]
  async fn a_per_call_reason_outranks_the_formatter() {
    let approval = DualApprovalCallback::new(["delete_file"])
      .with_timeout(Duration::from_secs(30))
      .with_rejection_formatter(|_| "run-wide default".to_owned());
    let args = json!({ "path": "notes.txt" });

    let text = denial_text(
      &approval,
      &conversation("local", "s1"),
      view("delete_file", &args),
      ApprovalOutcome::once(false).with_reason("this one specifically"),
    )
    .await;

    assert_eq!(
      text,
      "User denied execution of delete_file: this one specifically"
    );
  }

  /// A blank reason must not win: it would replace a usable message with an empty tool
  /// result, leaving the model with a failure it cannot interpret at all.
  #[tokio::test]
  async fn a_blank_reason_falls_through_to_the_formatter() {
    let approval = DualApprovalCallback::new(["delete_file"])
      .with_timeout(Duration::from_secs(30))
      .with_rejection_formatter(|_| "run-wide default".to_owned());
    let args = json!({ "path": "notes.txt" });

    for blank in ["", "   ", "\n\t "] {
      let text = denial_text(
        &approval,
        &conversation("local", "s1"),
        view("delete_file", &args),
        ApprovalOutcome::once(false).with_reason(blank),
      )
      .await;
      assert_eq!(
        text, "User denied execution of delete_file: run-wide default",
        "{blank:?} should not be recorded"
      );
    }
  }

  /// The formatter also covers the refusals nobody answered — otherwise a timeout would
  /// report a generic message while an explicit refusal reported the configured one.
  #[tokio::test]
  async fn the_formatter_covers_a_refusal_nobody_answered() {
    let approval = DualApprovalCallback::new(["delete_file"])
      .with_timeout(Duration::from_secs(30))
      .with_rejection_formatter(|_| "nobody was there to ask".to_owned());
    let args = json!({ "path": "notes.txt" });

    // No front-end listening at all: the callback fails closed without a decision.
    let result = call_without_a_front_end(
      &approval,
      &conversation("local", "s1"),
      view("delete_file", &args),
    )
    .await;

    assert_eq!(
      denial_content(result),
      "User denied execution of delete_file: nobody was there to ask"
    );
  }

  /// A remembered refusal keeps reporting what the human said. Degrading to the generic
  /// message on the second call would throw away the only part the model can act on.
  #[tokio::test]
  async fn a_remembered_refusal_keeps_its_reason() {
    let approval = approval();
    let context = conversation("local", "s1");
    let args = json!({ "path": "notes.txt" });

    let first = denial_text(
      &approval,
      &context,
      view("delete_file", &args),
      ApprovalOutcome::sticky(false).with_reason("production data, never delete"),
    )
    .await;
    assert_eq!(
      first,
      "User denied execution of delete_file: production data, never delete"
    );

    // Second call: answered from memory, with no front-end attached at all.
    assert_eq!(
      denial_content(
        call_without_a_front_end(&approval, &context, view("delete_file", &args)).await
      ),
      "User denied execution of delete_file: production data, never delete",
      "the remembered reason must be reported again"
    );
  }

  #[tokio::test]
  async fn an_approval_ignores_any_reason_attached_to_it() {
    let approval = approval();
    let args = json!({ "path": "notes.txt" });

    let allowed = answer_once(
      &approval,
      &conversation("local", "s1"),
      view("delete_file", &args),
      ApprovalOutcome::once(true).with_reason("ignored"),
    )
    .await;

    assert!(
      allowed,
      "an approved call runs; there is no result to replace"
    );
  }

  /// Observed in practice, and the reason the prefix exists: given a bare reason, the
  /// model treated it as an instruction smuggled through tool output and refused to act
  /// on it — naming prompt injection, correctly, as the reason. Every refusal must
  /// therefore attribute its explanation to the user rather than letting it read as
  /// something the tool emitted.
  #[tokio::test]
  async fn every_refusal_attributes_its_reason_to_the_user() {
    let args = json!({ "path": "notes.txt" });
    let with_formatter = DualApprovalCallback::new(["delete_file"])
      .with_timeout(Duration::from_secs(30))
      .with_rejection_formatter(|_| "policy text".to_owned());

    let cases = [
      (&approval(), ApprovalOutcome::once(false), "no reason"),
      (
        &approval(),
        ApprovalOutcome::once(false).with_reason("typed reason"),
        "a typed reason",
      ),
      (
        &with_formatter,
        ApprovalOutcome::once(false),
        "the run-wide formatter",
      ),
    ];

    for (approval, outcome, label) in cases {
      let text = denial_text(
        approval,
        &conversation("local", "s1"),
        view("delete_file", &args),
        outcome,
      )
      .await;
      assert!(
        text.starts_with("User denied execution of delete_file"),
        "{label}: the refusal must be attributable to the user, got {text:?}"
      );
    }
  }

  #[test]
  fn an_answer_can_carry_a_trailing_reason() {
    assert_eq!(
      parse_approval_answer("n: scratch files live under /tmp"),
      Some(ApprovalOutcome::once(false).with_reason("scratch files live under /tmp"))
    );
    // Full-width colon, so a reason can be typed in either script.
    assert_eq!(
      parse_approval_answer("d：生产数据"),
      Some(ApprovalOutcome::sticky(false).with_reason("生产数据"))
    );
    // A trailing colon with nothing after it is not a reason.
    assert_eq!(
      parse_approval_answer("n:   "),
      Some(ApprovalOutcome::once(false))
    );
    // The verdict is still parsed with surrounding whitespace tolerated.
    assert_eq!(
      parse_approval_answer("  no : because  "),
      Some(ApprovalOutcome::once(false).with_reason("because"))
    );
  }

  /// A reason must not rescue an unrecognized verdict: `maybe: ...` is still not an
  /// answer, and reading it as one would resolve a prompt nobody decided.
  #[test]
  fn a_reason_does_not_make_an_unknown_verdict_valid() {
    assert_eq!(parse_approval_answer("maybe: whatever"), None);
  }

  // ---- when nobody answers ----------------------------------------------------------

  /// The default. A caller with nowhere to come back to must end the round with a
  /// decision rather than leave a turn holding the session lock while nobody is being
  /// asked anything.
  #[tokio::test]
  async fn an_unanswered_prompt_is_refused_by_default() {
    let approval = approval();
    let args = json!({ "path": "notes.txt" });

    let decision = call_without_a_front_end(
      &approval,
      &conversation("local", "s1"),
      view("delete_file", &args),
    )
    .await;

    assert!(matches!(
      decision,
      ToolCallDecision::ShortCircuit(ToolResultStatus::Error, _)
    ));
  }

  #[tokio::test]
  async fn an_unanswered_prompt_can_suspend_instead() {
    let approval = DualApprovalCallback::new(["delete_file"])
      .with_timeout(Duration::from_secs(30))
      .when_unanswered(WhenUnanswered::Suspend);
    let args = json!({ "path": "notes.txt" });

    let decision = call_without_a_front_end(
      &approval,
      &conversation("local", "s1"),
      view("delete_file", &args),
    )
    .await;

    assert!(
      matches!(decision, ToolCallDecision::Suspend),
      "with nobody to ask, the question should be kept rather than answered"
    );
  }

  /// A timeout under this policy suspends too: waiting out the clock and finding nobody
  /// there is the same situation as finding nobody there immediately.
  #[tokio::test]
  async fn a_timeout_suspends_under_that_policy() {
    let approval = DualApprovalCallback::new(["delete_file"])
      .with_timeout(Duration::from_millis(30))
      .when_unanswered(WhenUnanswered::Suspend);
    let context = conversation("local", "s1");
    let args = json!({ "path": "notes.txt" });

    // Received and held, never answered.
    let (tx, mut rx) = mpsc::unbounded_channel::<PendingApproval>();
    let hold = async {
      let pending = rx.recv().await.expect("a prompt should have been raised");
      tokio::time::sleep(Duration::from_secs(30)).await;
      drop(pending);
    };
    let decision = tokio::select! {
      result = with_approval_channel(
        ApprovalChannel::Session(tx),
        approval.call(&context, view("delete_file", &args)),
      ) => result,
      () = hold => panic!("the approval should have timed out first"),
    };

    assert!(matches!(decision, ToolCallDecision::Suspend));
  }

  /// An explicit refusal is still a refusal under the suspend policy — the policy only
  /// governs the *absence* of an answer, and confusing the two would make "no" mean "ask
  /// me again later".
  #[tokio::test]
  async fn an_explicit_refusal_is_not_turned_into_a_suspension() {
    let approval = DualApprovalCallback::new(["delete_file"])
      .with_timeout(Duration::from_secs(30))
      .when_unanswered(WhenUnanswered::Suspend);
    let context = conversation("local", "s1");
    let args = json!({ "path": "notes.txt" });

    let (tx, mut rx) = mpsc::unbounded_channel::<PendingApproval>();
    let call = with_approval_channel(
      ApprovalChannel::Session(tx),
      approval.call(&context, view("delete_file", &args)),
    );
    let respond = async {
      let pending = rx.recv().await.expect("a prompt should have been raised");
      let _ = pending.decision.send(ApprovalOutcome::once(false));
    };
    let (decision, ()) = tokio::join!(call, respond);

    assert!(matches!(
      decision,
      ToolCallDecision::ShortCircuit(ToolResultStatus::Error, _)
    ));
  }

  /// Nor is a suspension remembered. It is not a decision, so recording it would invent
  /// standing permission — or a standing refusal — that nobody gave.
  #[tokio::test]
  async fn a_suspension_is_never_remembered() {
    let approval = DualApprovalCallback::new(["delete_file"])
      .with_timeout(Duration::from_secs(30))
      .when_unanswered(WhenUnanswered::Suspend);
    let context = conversation("local", "s1");
    let args = json!({ "path": "notes.txt" });

    for _ in 0..2 {
      assert!(
        matches!(
          call_without_a_front_end(&approval, &context, view("delete_file", &args)).await,
          ToolCallDecision::Suspend
        ),
        "each attempt must ask again rather than reuse the last suspension"
      );
    }
  }

  /// A remembered answer still applies under this policy: it short-circuits before any
  /// prompt is raised, so there is nothing to go unanswered.
  #[tokio::test]
  async fn a_remembered_answer_still_applies_under_the_suspend_policy() {
    let approval = DualApprovalCallback::new(["delete_file"])
      .with_timeout(Duration::from_secs(30))
      .when_unanswered(WhenUnanswered::Suspend);
    let context = conversation("local", "s1");
    let args = json!({ "path": "notes.txt" });

    assert!(
      answer_once(
        &approval,
        &context,
        view("delete_file", &args),
        ApprovalOutcome::sticky(true)
      )
      .await
    );

    assert!(
      call_without_a_front_end(&approval, &context, view("delete_file", &args))
        .await
        .is_proceed(),
      "the remembered approval should apply without a prompt to leave unanswered"
    );
  }

  #[test]
  fn registry_resolves_a_registered_approval_once() {
    let registry = ApprovalRegistry::new();
    let (tx, _rx) = oneshot::channel();
    registry.register(meta("call-1"), tx);

    assert!(
      registry.resolve("call-1", ApprovalOutcome::once(true)),
      "the first answer decides"
    );
    assert!(
      !registry.resolve("call-1", ApprovalOutcome::once(false)),
      "a second answer must be a no-op"
    );
    assert!(registry.is_empty());
  }

  #[test]
  fn registry_reports_an_unknown_approval() {
    let registry = ApprovalRegistry::new();
    assert!(!registry.resolve("nope", ApprovalOutcome::once(true)));
  }

  #[tokio::test]
  async fn registry_delivers_the_decision_to_the_waiting_side() {
    let registry = ApprovalRegistry::new();
    let (tx, rx) = oneshot::channel();
    registry.register(meta("call-1"), tx);

    registry.resolve("call-1", ApprovalOutcome::sticky(true));
    assert_eq!(
      rx.await.unwrap(),
      ApprovalOutcome::sticky(true),
      "the decision must arrive exactly as sent, sticky flag included — the callback \
       reads that flag to decide whether to remember the answer"
    );
  }

  #[test]
  fn registry_exposes_a_pending_id_for_another_front_end_to_answer() {
    let registry = ApprovalRegistry::new();
    assert!(registry.any_pending().is_none());

    let (tx, _rx) = oneshot::channel();
    registry.register(meta("call-1"), tx);
    assert_eq!(registry.any_pending().as_deref(), Some("call-1"));
  }

  /// The case behind `pending_snapshot` existing at all: a front-end that was not
  /// listening when the prompt was raised — a reloaded browser tab — has to be able to
  /// ask what is outstanding, with enough detail to render it.
  #[test]
  fn registry_snapshot_describes_every_pending_approval() {
    let registry = ApprovalRegistry::new();
    assert!(
      registry.pending_snapshot().is_empty(),
      "nothing pending, nothing to describe"
    );

    let (tx, _rx) = oneshot::channel();
    registry.register(meta("call-1"), tx);

    let snapshot = registry.pending_snapshot();
    assert_eq!(snapshot.len(), 1);
    assert_eq!(snapshot[0].id, "call-1");
    assert_eq!(snapshot[0].tool, "delete_file");
    assert_eq!(
      snapshot[0].raw_arguments, "{}",
      "the raw payload has to survive, since that is what a human is shown"
    );
  }

  /// A resolved prompt must leave the snapshot, or a tab loading a moment later would
  /// render a decision that has already been made.
  #[test]
  fn registry_snapshot_drops_a_resolved_approval() {
    let registry = ApprovalRegistry::new();
    let (tx, _rx) = oneshot::channel();
    registry.register(meta("call-1"), tx);

    registry.resolve("call-1", ApprovalOutcome::once(true));

    assert!(registry.pending_snapshot().is_empty());
  }

  /// Both readers order by age, and they must agree: the terminal displays the prompt
  /// `pending_snapshot` puts first and resolves whatever `any_pending` returns, so a
  /// disagreement would answer a different prompt than the one on screen.
  #[test]
  fn registry_orders_pending_approvals_oldest_first() {
    let registry = ApprovalRegistry::new();
    for (id, requested_at) in [("newer", 200), ("oldest", 100), ("newest", 300)] {
      let (tx, _rx) = oneshot::channel();
      registry.register(
        ApprovalMeta {
          requested_at,
          ..meta(id)
        },
        tx,
      );
    }

    let ids: Vec<String> = registry
      .pending_snapshot()
      .into_iter()
      .map(|item| item.id)
      .collect();
    assert_eq!(ids, vec!["oldest", "newer", "newest"]);
    assert_eq!(registry.any_pending().as_deref(), Some("oldest"));
  }

  /// Tool calls in one round run concurrently, so several prompts share one
  /// `requested_at` second. The id tie-break is what keeps the order total — without it
  /// these two readers could disagree.
  #[test]
  fn registry_breaks_ties_on_id_so_the_order_is_total() {
    let registry = ApprovalRegistry::new();
    for id in ["call-b", "call-a"] {
      let (tx, _rx) = oneshot::channel();
      registry.register(meta(id), tx); // Same `requested_at` for both.
    }

    let ids: Vec<String> = registry
      .pending_snapshot()
      .into_iter()
      .map(|item| item.id)
      .collect();
    assert_eq!(ids, vec!["call-a", "call-b"]);
    assert_eq!(
      registry.any_pending().as_deref(),
      Some("call-a"),
      "both readers must pick the same one"
    );
  }

  #[test]
  fn registry_discards_abandoned_entries() {
    let registry = ApprovalRegistry::new();
    for id in ["call-1", "call-2"] {
      let (tx, _rx) = oneshot::channel();
      registry.register(meta(id), tx);
    }

    registry.discard(&["call-1".to_owned(), "call-2".to_owned()]);
    assert!(
      registry.is_empty(),
      "a finished turn must not leave entries behind"
    );
  }
}
