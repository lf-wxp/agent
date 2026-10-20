//! Charging token usage to the run that caused it.
//!
//! An [`crate::agent::Agent`] run spends tokens on its own requests, but also on calls it
//! never sees directly: a sub-agent ([`crate::agent::AgentTool`]), a vision call
//! (`read_image`), a history summary
//! ([`crate::callback::context_optimizer::Summarization`]), any custom tool or hook that
//! calls a model. Those happen *inside* the run — while a tool or hook it is awaiting
//! executes — but behind interfaces that have no handle on the run's
//! [`ExecutionContext`].
//!
//! [`charge`] bridges that gap: whoever makes a model call reports what it spent, and
//! the run it happened inside adds it to [`ExecutionContext::usage`] /
//! [`ExecutionContext::usage_by_model`] when it ends. The agent charges its own requests
//! the same way, and so do the stateless helpers in [`crate::llm`] (`tool_loop::run`, and
//! through it `chat_complete` and the structured helpers), so a tool built on them is
//! covered without doing anything. A tool calling
//! [`crate::llm::provider::Provider::chat`] directly charges the response's usage itself.
//!
//! # What "inside" means
//!
//! The run sets a task-local ledger around everything it awaits. A charge made while
//! being polled from the run's future — on the same task — lands in it. Work moved onto
//! another task (`tokio::spawn`, `spawn_blocking`) is outside it, and [`charge`] reports
//! that by returning `false`.
//!
//! # Nesting
//!
//! A run started inside another (a sub-agent) gets a ledger of its own, linked to the
//! outer one, and every charge is passed up the chain *as it is made*. Each level's
//! `usage` therefore covers everything spent below it, counted once — and what a
//! sub-agent spends reaches its parent even when the sub-agent never finishes (it fails,
//! is cancelled, or is cut off by a timeout), since the tokens are spent either way.

use std::{
  collections::BTreeMap,
  future::Future,
  sync::{Arc, Mutex},
};

use crate::agent::{ExecutionContext, TokenUsage};

tokio::task_local! {
  static LEDGER: Ledger;
}

/// Charges collected for one run until it settles them into its context.
#[derive(Clone, Default)]
pub(crate) struct Ledger {
  charges: Arc<Mutex<BTreeMap<String, TokenUsage>>>,
  /// The ledger of the run this one started inside, which every charge also reaches.
  outer: Option<Box<Ledger>>,
}

impl Ledger {
  /// A ledger for a run starting now: linked to the run this is called from, if any.
  pub(crate) fn for_new_run() -> Self {
    Self {
      charges: Arc::default(),
      outer: LEDGER.try_with(|outer| Box::new(outer.clone())).ok(),
    }
  }

  /// Run `future` with this ledger receiving every [`charge`] made inside it.
  pub(crate) fn scope<F: Future>(&self, future: F) -> impl Future<Output = F::Output> {
    LEDGER.scope(self.clone(), future)
  }

  /// [`Self::scope`] for one synchronous step — one `poll` of a stream.
  pub(crate) fn sync_scope<R>(&self, f: impl FnOnce() -> R) -> R {
    LEDGER.sync_scope(self.clone(), f)
  }

  /// Move everything charged so far into `context`, leaving the ledger empty so a later
  /// settle cannot count it twice. The outer runs keep their copies until they settle.
  pub(crate) fn settle(&self, context: &mut ExecutionContext) {
    let charged = std::mem::take(&mut *self.lock());
    for (model, usage) in charged {
      context.record_usage(&model, usage);
    }
  }

  fn add(&self, model: &str, usage: &TokenUsage) {
    let mut ledger = Some(self);
    while let Some(current) = ledger {
      current
        .lock()
        .entry(model.to_owned())
        .or_default()
        .merge(usage);
      ledger = current.outer.as_deref();
    }
  }

  /// A poisoned lock still holds a valid map: every write is a single `merge`.
  fn lock(&self) -> std::sync::MutexGuard<'_, BTreeMap<String, TokenUsage>> {
    self
      .charges
      .lock()
      .unwrap_or_else(|poisoned| poisoned.into_inner())
  }
}

/// Charge `usage`, spent on `model`, to the run this is called from (and the runs that
/// one is nested in).
///
/// Returns whether there was a run to charge it to; `false` means the call happened
/// outside any [`crate::agent::Agent`] run (or on another task), and the usage was not
/// recorded anywhere. Empty usage is accepted and ignored.
pub fn charge(model: &str, usage: TokenUsage) -> bool {
  LEDGER
    .try_with(|ledger| {
      if !usage.is_empty() {
        ledger.add(model, &usage);
      }
    })
    .is_ok()
}

/// Settle what has been charged to the current run so far into its `context`, so the
/// context is up to date mid-run (a checkpoint taken next carries it). A no-op outside
/// a run.
pub(crate) fn settle_current(context: &mut ExecutionContext) {
  let _ = LEDGER.try_with(|ledger| ledger.settle(context));
}

#[cfg(test)]
mod tests {
  use super::*;

  fn tokens(prompt: u32, completion: u32) -> TokenUsage {
    TokenUsage {
      prompt_tokens: prompt,
      completion_tokens: completion,
      total_tokens: prompt + completion,
      ..TokenUsage::default()
    }
  }

  #[test]
  fn charging_outside_a_run_says_so() {
    assert!(!charge("m", tokens(1, 1)));
  }

  #[tokio::test]
  async fn charges_inside_the_scope_settle_into_the_context() {
    let ledger = Ledger::for_new_run();
    let charged = ledger
      .scope(async {
        charge("big", tokens(10, 1));
        charge("small", tokens(5, 1));
        charge("big", tokens(1, 1))
      })
      .await;
    assert!(charged);

    let mut context = ExecutionContext::new();
    ledger.settle(&mut context);
    assert_eq!(context.usage.prompt_tokens, 16);
    assert_eq!(context.usage_by_model["big"].prompt_tokens, 11);

    ledger.settle(&mut context);
    assert_eq!(
      context.usage.prompt_tokens, 16,
      "settling twice counts nothing twice"
    );
  }

  /// A nested run keeps its own account, and the outer one sees everything — including
  /// what the inner run spent, at the moment it spent it.
  #[tokio::test]
  async fn a_nested_run_charges_both_levels_once() {
    let outer = Ledger::for_new_run();
    let inner = outer
      .scope(async {
        let inner = Ledger::for_new_run();
        inner.scope(async { charge("m", tokens(7, 0)) }).await;
        charge("m", tokens(1, 0));
        inner
      })
      .await;

    let mut outer_context = ExecutionContext::new();
    let mut inner_context = ExecutionContext::new();
    outer.settle(&mut outer_context);
    inner.settle(&mut inner_context);
    assert_eq!(inner_context.usage.prompt_tokens, 7);
    assert_eq!(outer_context.usage.prompt_tokens, 8);
  }

  /// The point of passing charges up as they happen: an inner run that is dropped
  /// before it settles has still spent its tokens, and the outer run still counts them.
  #[tokio::test]
  async fn a_dropped_inner_run_still_charges_the_outer_one() {
    let outer = Ledger::for_new_run();
    outer
      .scope(async {
        let inner = Ledger::for_new_run();
        inner.scope(async { charge("m", tokens(5, 2)) }).await;
        drop(inner);
      })
      .await;

    let mut context = ExecutionContext::new();
    outer.settle(&mut context);
    assert_eq!(context.usage.prompt_tokens, 5);
    assert_eq!(context.usage.completion_tokens, 2);
  }

  /// Two runs polled side by side on one task keep their charges apart.
  #[tokio::test]
  async fn concurrent_runs_on_one_task_stay_separate() {
    let (a, b) = (Ledger::for_new_run(), Ledger::for_new_run());
    futures::join!(
      a.scope(async {
        tokio::task::yield_now().await;
        charge("m", tokens(1, 0));
      }),
      b.scope(async {
        tokio::task::yield_now().await;
        charge("m", tokens(2, 0));
      }),
    );

    let (mut ca, mut cb) = (ExecutionContext::new(), ExecutionContext::new());
    a.settle(&mut ca);
    b.settle(&mut cb);
    assert_eq!(ca.usage.prompt_tokens, 1);
    assert_eq!(cb.usage.prompt_tokens, 2);
  }
}
