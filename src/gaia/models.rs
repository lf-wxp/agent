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

#[derive(Debug, Serialize)]
pub struct GaiaEvalResult {
  pub task_id: String,
  pub model: String,
  pub correct: bool,
  pub is_solvable: Option<bool>,
  pub prediction: Option<String>,
  pub answer: String,
  pub unsolvable_reason: Option<String>,
  pub error: Option<String>,
}
