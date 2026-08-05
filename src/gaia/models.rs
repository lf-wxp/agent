use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct HfResponse {
  pub rows: Vec<HfRow>,
}

#[derive(Debug, Deserialize)]
pub struct HfRow {
  pub row: GaiaRow,
}

#[derive(Debug, Deserialize, Clone)]
pub struct GaiaRow {
  pub task_id: String,
  #[serde(rename = "Question")]
  pub question: String,
  #[serde(rename = "Level")]
  pub level: String,
  #[serde(rename = "Final answer")]
  pub final_answer: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct GaiaOutput {
  pub is_solvable: bool,
  pub unsolvable_reason: String,
  pub final_answer: String,
}

/// A solved problem plus metadata about how the answer was produced.
///
/// Kept separate from [`GaiaOutput`], which is the model's own schema: adding fields there
/// would make the model responsible for filling them in.
#[derive(Debug)]
pub struct Solution {
  pub output: GaiaOutput,
  /// `true` when the tool round budget ran out before the model finished, so the answer
  /// rests on partial information and should be scored separately.
  pub budget_exhausted: bool,
}

#[derive(Debug, Serialize)]
pub struct GaiaEvalResult {
  pub task_id: String,
  pub model: String,
  pub correct: bool,
  pub is_solvable: Option<bool>,
  pub prediction: Option<String>,
  pub answer: String,
  pub unsolvable_reason: Option<String>,
  /// `None` when solving failed outright, so the question never got an answer.
  pub budget_exhausted: Option<bool>,
  pub error: Option<String>,
}
