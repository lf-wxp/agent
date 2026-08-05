use std::collections::HashMap;
// use std::sync::Arc;

use agent::{
  config,
  gaia::{
    dataset::load_gaia_level1,
    // evaluator::{evaluate_gaia_single, evaluate_gaia_single_with_tools},
    evaluator::evaluate_gaia_single,
    models::GaiaEvalResult,
  },
  llm::semaphore::get_semaphore,
  telemetry,
  // tools::build_toolbox,
};
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
  // let toolbox = Arc::new(build_toolbox().await?);

  // Fetch once up front: `config::model()` returns a 'static value, so it can be moved into each task directly.
  let model = config::model();

  let mut set = JoinSet::new();

  // Use clone rather than into_iter: the same problems are also fed to the "with tools" group for comparison.
  for problem in problems.iter().cloned() {
    set.spawn(async move {
      let _permit = get_semaphore().acquire().await?;
      let eval = evaluate_gaia_single(problem, model).await;
      Ok::<_, anyhow::Error>((GROUP_WITHOUT_TOOLS, eval))
    });
  }

  // for problem in problems.iter().cloned() {
  //   let toolbox = toolbox.clone();
  //   set.spawn(async move {
  //     let _permit = get_semaphore().acquire().await?;
  //     let eval = evaluate_gaia_single_with_tools(problem, model, toolbox).await;
  //     Ok::<_, anyhow::Error>((GROUP_WITH_TOOLS, eval))
  //   });
  // }

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
    // Skip empty groups to avoid printing NaN from 0/0.
    let Some(evals) = results.get(group).filter(|evals| !evals.is_empty()) else {
      continue;
    };
    let total = evals.len();
    let correct = evals.iter().filter(|eval| eval.correct).count();
    tracing::info!(
      "{group}: {correct}/{total} ({:.1}%)",
      correct as f64 / total as f64 * 100.0
    );
  }
}
