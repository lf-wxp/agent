//! Ordering of one round's tool calls: which may overlap, which must take turns.
//!
//! See [`crate::tools::Tool::concurrency_safe`] for what a tool promises by being
//! concurrency-safe. Shared by [`crate::agent::Agent`] and the stateless
//! [`crate::llm::tool_loop`], so the two paths cannot disagree about it.
//!
//! # Why gates rather than running the calls one after another
//!
//! A round's calls all start at once, because the part that can take a long time is not
//! always the tool: it is a before-tool hook waiting on a human. Running the calls
//! sequentially would also line those waits up — four calls with nobody at the keyboard
//! would hold the turn for four approval timeouts instead of one. So everything except
//! the tool's own execution stays concurrent, and only that step waits for its [`turn`].
//!
//! [`turn`]: CallGate::turn

use tokio::sync::watch;

/// One call's place in the round's order; see [`plan`].
#[derive(Debug, Default)]
pub struct CallGate {
  /// Calls that must be finished before this one may execute.
  wait_for: Vec<watch::Receiver<()>>,
  /// Dropped when this call is finished, which is what releases the calls waiting on it.
  /// Dropping rather than sending means every way a call can end — result, denial,
  /// suspension, cancellation, the future being dropped — releases its successors, so
  /// none of them can be left waiting on a call that will never report.
  done: Option<watch::Sender<()>>,
}

impl CallGate {
  /// Resolves once every call this one must wait for has finished.
  ///
  /// Immediately for a call with nothing before it. Cancel-safe: dropping the future
  /// loses nothing, and calling it again simply waits again.
  pub async fn turn(&mut self) {
    for earlier in &mut self.wait_for {
      // Nothing is ever sent on the channel, so `changed` can only return once the
      // sender is dropped — as an `Err`, which is the signal being waited for.
      let _ = earlier.changed().await;
    }
    self.wait_for.clear();
  }

  /// This call is over; let the calls queued behind it go.
  ///
  /// Optional — dropping the gate does the same — but lets a caller release them as soon
  /// as the tool itself is done, rather than after whatever follows it (e.g. after-hooks).
  pub fn finish(&mut self) {
    self.done = None;
  }
}

/// One gate per call, in call order, given whether each call is concurrency-safe.
///
/// Consecutive safe calls overlap; a call that is not safe runs alone, after everything
/// before it has finished and before anything after it starts. So
/// `[safe, safe, unsafe, safe]` executes as `{safe, safe}`, then `{unsafe}`, then
/// `{safe}`.
///
/// Order is by position and nothing else: two unsafe calls never overlap even when they
/// touch unrelated files. That is deliberately coarse — whether two arbitrary tool calls
/// conflict is not something the scheduler can know, and a model that asks for several
/// writes in one message gets them applied in the order it wrote them.
pub fn plan(concurrency_safe: &[bool]) -> Vec<CallGate> {
  // The common case, and the one that must stay free of overhead: nothing to order.
  if concurrency_safe.iter().all(|safe| *safe) {
    return concurrency_safe
      .iter()
      .map(|_| CallGate::default())
      .collect();
  }

  // The last call that was not safe: everything after it waits on it, and it alone is
  // enough, because it in turn waited on everything before it.
  let mut barrier: Option<watch::Receiver<()>> = None;
  // Safe calls since that barrier: the next barrier has to wait for these as well.
  let mut since_barrier: Vec<watch::Receiver<()>> = Vec::new();

  concurrency_safe
    .iter()
    .map(|safe| {
      let (done, finished) = watch::channel(());
      if *safe {
        let gate = CallGate {
          wait_for: barrier.iter().cloned().collect(),
          done: Some(done),
        };
        since_barrier.push(finished);
        gate
      } else {
        let mut wait_for: Vec<_> = barrier.take().into_iter().collect();
        wait_for.append(&mut since_barrier);
        barrier = Some(finished);
        CallGate {
          wait_for,
          done: Some(done),
        }
      }
    })
    .collect()
}

#[cfg(test)]
mod tests {
  use std::{
    sync::{
      Arc, Mutex,
      atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
  };

  use futures::future::join_all;

  use super::*;

  /// Records the highest number of calls running at once, and the order they started in.
  #[derive(Default, Clone)]
  struct Probe {
    running: Arc<AtomicUsize>,
    peak: Arc<AtomicUsize>,
    started: Arc<Mutex<Vec<usize>>>,
  }

  impl Probe {
    /// What a call does once it is its turn: take a little time, visibly.
    async fn run(&self, id: usize, mut gate: CallGate) -> usize {
      gate.turn().await;
      self.started.lock().unwrap().push(id);
      let now = self.running.fetch_add(1, Ordering::SeqCst) + 1;
      self.peak.fetch_max(now, Ordering::SeqCst);
      tokio::time::sleep(Duration::from_millis(20)).await;
      self.running.fetch_sub(1, Ordering::SeqCst);
      gate.finish();
      id
    }

    async fn run_all(&self, flags: &[bool]) -> Vec<usize> {
      join_all(
        plan(flags)
          .into_iter()
          .enumerate()
          .map(|(id, gate)| self.run(id, gate)),
      )
      .await
    }
  }

  #[tokio::test]
  async fn safe_calls_overlap() {
    let probe = Probe::default();
    let outputs = probe.run_all(&[true; 4]).await;

    assert_eq!(outputs, vec![0, 1, 2, 3]);
    assert_eq!(probe.peak.load(Ordering::SeqCst), 4);
  }

  #[tokio::test]
  async fn unsafe_calls_never_overlap_and_start_in_call_order() {
    let probe = Probe::default();
    let outputs = probe.run_all(&[false; 4]).await;

    assert_eq!(outputs, vec![0, 1, 2, 3]);
    assert_eq!(probe.peak.load(Ordering::SeqCst), 1);
    assert_eq!(*probe.started.lock().unwrap(), vec![0, 1, 2, 3]);
  }

  /// `[safe, safe, unsafe, safe]`: the unsafe call waits for the first two, and the last
  /// waits for it — so at most the first pair ever overlaps.
  #[tokio::test]
  async fn an_unsafe_call_is_a_barrier_in_both_directions() {
    let probe = Probe::default();
    probe.run_all(&[true, true, false, true]).await;

    assert_eq!(probe.peak.load(Ordering::SeqCst), 2);
    let started = probe.started.lock().unwrap().clone();
    assert_eq!(started[2], 2, "got: {started:?}");
    assert_eq!(started[3], 3, "got: {started:?}");
  }

  /// A call that ends without ever reaching its turn (denied, suspended, cancelled) must
  /// still release the calls behind it.
  #[tokio::test]
  async fn a_call_that_drops_its_gate_releases_the_next() {
    let mut gates = plan(&[false, false]).into_iter();
    let first = gates.next().unwrap();
    let mut second = gates.next().unwrap();

    drop(first);
    tokio::time::timeout(Duration::from_secs(2), second.turn())
      .await
      .expect("the second call was left waiting on a call that is gone");
  }

  #[tokio::test]
  async fn a_waiting_call_stays_waiting_while_its_predecessor_is_alive() {
    let mut gates = plan(&[false, false]).into_iter();
    let first = gates.next().unwrap();
    let mut second = gates.next().unwrap();

    let waited = tokio::time::timeout(Duration::from_millis(50), second.turn()).await;
    assert!(waited.is_err(), "went ahead of a call still running");
    drop(first);
  }

  #[test]
  fn no_calls_is_no_gates() {
    assert!(plan(&[]).is_empty());
  }
}
