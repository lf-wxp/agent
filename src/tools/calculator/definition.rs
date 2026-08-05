use serde_json::{Value, json};

use crate::tools::calculator::Operator;

/// What the tool does, shown to the model.
pub const DESCRIPTION: &str = "Perform basic arithmetic operations.";

/// JSON Schema for [`super::CalculatorArgs`].
///
/// Hand-written rather than derived via `schemars`: a derived schema emits `$ref` /
/// `$defs` for the operator enum, which not every OpenAI-compatible endpoint accepts
/// inside `function.parameters`. The `enum` list still comes from [`Operator::ALL`], so
/// it cannot drift from the executor.
pub fn parameters() -> Value {
  json!({
    "type": "object",
    "properties": {
      "operator": {
        "type": "string",
        "description": "Arithmetic operation to perform",
        "enum": Operator::ALL.map(Operator::as_str),
      },
      "first_number": {
        "type": "number",
        "description": "First number for the calculation"
      },
      "second_number": {
        "type": "number",
        "description": "Second number for the calculation"
      }
    },
    "required": ["operator", "first_number", "second_number"],
    "additionalProperties": false
  })
}
