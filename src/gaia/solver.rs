use async_openai::types::chat::FinishReason;
use backon::{ExponentialBuilder, Retryable};

// use crate::{agent::Agent, gaia::models::GaiaOutput, tools::ToolBox};
use crate::{
  gaia::models::GaiaOutput,
  llm::structured::{TruncatedOutput, chat_complete_structured_raw},
};

/// Max retry attempts.
const MAX_RETRY_TIMES: usize = 3;

pub async fn solve_problem_with_retry(
  model: &str,
  system: &str,
  prompt: &str,
) -> anyhow::Result<GaiaOutput> {
  let op = || solve_problem(model, system, prompt);
  op.retry(ExponentialBuilder::default().with_max_times(MAX_RETRY_TIMES))
    // A max_tokens truncation is a deterministic failure; retrying only consumes tokens again.
    .when(|err: &anyhow::Error| err.downcast_ref::<TruncatedOutput>().is_none())
    .notify(|err, dur| tracing::warn!("retrying after {dur:?}: {err}"))
    .await
}

async fn solve_problem(model: &str, system: &str, prompt: &str) -> anyhow::Result<GaiaOutput> {
  // The structured approach (native schema / prompt injection) is decided automatically by the model; see `llm::structured`.
  let choice = chat_complete_structured_raw::<GaiaOutput>(model, Some(system), prompt).await?;

  // When blocked by content filtering there is no parseable JSON; degrade directly to an "unsolvable" result,
  // to avoid the upper layer treating it as a parse failure and triggering a meaningless retry.
  if choice.finish_reason() == Some(FinishReason::ContentFilter) {
    return Ok(GaiaOutput {
      is_solvable: false,
      unsolvable_reason: "Model refused to answer (content filter)".to_owned(),
      final_answer: String::new(),
    });
  }

  choice.parse()
}

// pub async fn solve_problem_with_tools(
//   model: &str,
//   system: &str,
//   prompt: &str,
//   toolbox: Arc<ToolBox>,
// ) -> anyhow::Result<GaiaOutput> {
//   let agent = Agent::new(model, Some(system), toolbox).with_max_steps(15);
//   let result = agent.run_structured::<GaiaOutput>(prompt).await?;
//   Ok(result.output)
// }
