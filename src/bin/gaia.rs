use std::collections::HashMap;

use agent::{
  config,
  gaia::{dataset::load_gaia_level1, evaluator::evaluate_gaia_single, models::GaiaEvalResult},
  llm::provider::Provider,
  telemetry,
  tools::ToolRegistry,
};
use std::sync::Arc;
use tokio::task::JoinSet;

/// Number of problems to evaluate.
const SAMPLE_SIZE: usize = 5;

const GROUP_WITHOUT_TOOLS: &str = "without_tools";
const GROUP_WITH_TOOLS: &str = "with_tools";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
  telemetry::init()?;
  gaia_level1_experiment().await
}

pub async fn gaia_level1_experiment() -> anyhow::Result<()> {
  let problems = load_gaia_level1(SAMPLE_SIZE).await?;

  // Fetch once up front: `config::model()` returns a 'static value, so it can be moved into each task directly.
  let model = config::model();
  // Concurrency is enforced inside every model call now (see `Provider::acquire`), not by
  // holding a permit for a whole task here, so tasks waiting on tool I/O no longer starve
  // others out of their turn on the shared budget.
  let provider = Provider::shared();

  // Both arms run the same problems, so the comparison is paired rather than across samples.
  // `Arc` because each spawned task needs its own handle on the shared registry.
  let groups: [(&'static str, Arc<ToolRegistry>); 2] = [
    (GROUP_WITHOUT_TOOLS, Arc::new(ToolRegistry::empty())),
    (GROUP_WITH_TOOLS, Arc::new(ToolRegistry::builtin()?)),
  ];

  let mut set = JoinSet::new();
  for (group, registry) in groups {
    // Clone rather than consume: the problems are needed by the other arm too.
    for problem in problems.iter().cloned() {
      let registry = Arc::clone(&registry);
      set.spawn(async move {
        let eval = evaluate_gaia_single(provider, problem, model, &registry).await;
        Ok::<_, anyhow::Error>((group, eval))
      });
    }
  }

  let mut results: HashMap<&str, Vec<GaiaEvalResult>> = HashMap::new();
  // Note: do not write `while let Some(Ok(result))` here — if a task panics (JoinError),
  // the pattern match fails and silently ends the loop, dropping results of all remaining tasks.
  while let Some(joined) = set.join_next().await {
    match joined {
      Ok(Ok((group, eval))) => {
        tracing::info!("[{group}] {eval:#?}");
        results.entry(group).or_default().push(eval);
      }
      Ok(Err(err)) => tracing::error!("task failed: {err:#}"),
      Err(err) => tracing::error!("task panicked: {err}"),
    }
  }

  report(&results);

  Ok(())
}

/// Print accuracy grouped by category.
fn report(results: &HashMap<&str, Vec<GaiaEvalResult>>) {
  for group in [GROUP_WITH_TOOLS, GROUP_WITHOUT_TOOLS] {
    let Some(evals) = results.get(group) else {
      continue;
    };
    log_accuracy(group, evals);

    // Failures never produced an answer, so they sit in neither budget bucket below.
    // Surfacing the count explains a gap between the headline total and those buckets.
    let failed = evals.iter().filter(|eval| eval.error.is_some()).count();
    if failed > 0 {
      tracing::warn!(
        "{group}: {failed}/{} failed to produce an answer",
        evals.len()
      );
    }

    // Answers produced after the tool budget ran out rest on partial information, so they
    // are broken out: mixing them into the headline number hides why a run scored badly.
    if evals.iter().any(is_budget_exhausted) {
      log_accuracy(
        &format!("{group} / budget exhausted"),
        evals.iter().filter(|eval| is_budget_exhausted(eval)),
      );
      log_accuracy(
        &format!("{group} / within budget"),
        evals
          .iter()
          .filter(|eval| eval.budget_exhausted == Some(false)),
      );
    }
  }
}

fn is_budget_exhausted(eval: &GaiaEvalResult) -> bool {
  eval.budget_exhausted == Some(true)
}

/// Log `correct/total (pct)` for a set of results; silent when the set is empty,
/// which also avoids printing NaN from 0/0.
fn log_accuracy<'a>(label: &str, evals: impl IntoIterator<Item = &'a GaiaEvalResult>) {
  let mut total = 0usize;
  let mut correct = 0usize;
  for eval in evals {
    total += 1;
    correct += usize::from(eval.correct);
  }

  if total == 0 {
    return;
  }

  tracing::info!(
    "{label}: {correct}/{total} ({:.1}%)",
    correct as f64 / total as f64 * 100.0
  );
}
