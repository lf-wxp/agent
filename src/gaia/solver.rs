use async_openai::types::chat::FinishReason;

use crate::{
  gaia::models::{GaiaOutput, Solution},
  llm::{
    provider::Provider,
    retry::with_retry,
    structured::{TruncatedOutput, chat_complete_structured_raw},
    tool_loop::BudgetExhausted,
  },
  tools::ToolRegistry,
};

pub async fn solve_problem_with_retry(
  provider: &Provider,
  model: &str,
  system: &str,
  prompt: &str,
  registry: &ToolRegistry,
) -> anyhow::Result<Solution> {
  with_retry(
    || solve_problem(provider, model, system, prompt, registry),
    // Skip deterministic failures, where a retry only burns resources for the same outcome:
    // a max_tokens truncation repeats token spend, and a budget-exhausted attempt replays
    // the entire tool round budget.
    |err: &anyhow::Error| {
      err.downcast_ref::<TruncatedOutput>().is_none()
        && err.downcast_ref::<BudgetExhausted>().is_none()
    },
  )
  .await
}

async fn solve_problem(
  provider: &Provider,
  model: &str,
  system: &str,
  prompt: &str,
  registry: &ToolRegistry,
) -> anyhow::Result<Solution> {
  // The structured approach (native schema / prompt injection) is decided automatically by the model; see `llm::structured`.
  let choice =
    chat_complete_structured_raw::<GaiaOutput>(provider, model, Some(system), prompt, registry)
      .await?;

  // Read before `parse` consumes the choice.
  let budget_exhausted = choice.budget_exhausted();

  // When blocked by content filtering there is no parseable JSON; degrade directly to an "unsolvable" result,
  // to avoid the upper layer treating it as a parse failure and triggering a meaningless retry.
  if choice.finish_reason() == Some(FinishReason::ContentFilter) {
    return Ok(Solution {
      output: GaiaOutput {
        is_solvable: false,
        unsolvable_reason: "Model refused to answer (content filter)".to_owned(),
        final_answer: String::new(),
      },
      budget_exhausted,
    });
  }

  Ok(Solution {
    output: choice.parse()?,
    budget_exhausted,
  })
}
