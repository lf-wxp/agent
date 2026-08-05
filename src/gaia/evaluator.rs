// use std::sync::Arc;

use crate::gaia::{
  models::{GaiaEvalResult, GaiaOutput, GaiaRow},
  solver::solve_problem_with_retry,
  // solver::{solve_problem_with_retry, solve_problem_with_tools},
};
// use crate::tools::ToolBox;

pub const GAIA_PROMPT: &str = r#"You are a general AI assistant. I will ask you a question.
First, determine if you can solve this problem with your current capabilities and set "is_solvable" accordingly.
If you can solve it, set "is_solvable" to true and provide your answer in "final_answer".
If you cannot solve it, set "is_solvable" to false and explain why in "unsolvable_reason".
Your final answer should be a number OR as few words as possible OR a comma-separated list of numbers and/or strings.
If you are asked for a number, don't use a comma to write your number neither use units such as $ or percent sign unless specified otherwise.
If you are asked for a string, don't use articles, neither abbreviations (e.g., for cities), and write the digits in plain text unless specified otherwise.
If you are asked for a comma-separated list, apply the above rules depending on whether the element is a number or a string."#;

/// GAIA's official scoring is exact match; here we relax it to "trim + case-insensitive".
fn is_correct(prediction: &str, answer: &str) -> bool {
  let prediction = prediction.trim();
  !prediction.is_empty() && prediction.eq_ignore_ascii_case(answer.trim())
}

fn to_eval_result(
  problem: GaiaRow,
  model: &str,
  result: anyhow::Result<GaiaOutput>,
) -> GaiaEvalResult {
  match result {
    Ok(output) => GaiaEvalResult {
      task_id: problem.task_id,
      model: model.to_owned(),
      correct: is_correct(&output.final_answer, &problem.final_answer),
      is_solvable: Some(output.is_solvable),
      prediction: Some(output.final_answer),
      answer: problem.final_answer,
      unsolvable_reason: Some(output.unsolvable_reason),
      error: None,
    },
    Err(err) => GaiaEvalResult {
      task_id: problem.task_id,
      model: model.to_owned(),
      correct: false,
      is_solvable: None,
      prediction: None,
      answer: problem.final_answer,
      // Use `{err:#}` to print the full error chain, otherwise only the outermost message is kept and root cause is hard to locate.
      error: Some(format!("{err:#}")),
      unsolvable_reason: None,
    },
  }
}

pub async fn evaluate_gaia_single(problem: GaiaRow, model: &str) -> GaiaEvalResult {
  let result = solve_problem_with_retry(model, GAIA_PROMPT, &problem.question).await;
  to_eval_result(problem, model, result)
}

// pub async fn evaluate_gaia_single_with_tools(
//   problem: GaiaRow,
//   model: &str,
//   toolbox: Arc<ToolBox>,
// ) -> GaiaEvalResult {
//   let result = solve_problem_with_tools(model, GAIA_PROMPT, &problem.question, toolbox).await;
//   to_eval_result(problem, model, result)
// }

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn ignores_case_and_surrounding_space() {
    assert!(is_correct("  Paris ", "paris"));
  }

  #[test]
  fn rejects_blank_prediction() {
    // A blank model answer should not "accidentally score" just because the reference answer is also blank.
    assert!(!is_correct("   ", ""));
  }

  #[test]
  fn rejects_different_answer() {
    assert!(!is_correct("Paris", "Lyon"));
  }
}
