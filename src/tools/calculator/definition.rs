use async_openai::types::chat::{ChatCompletionTool, ChatCompletionTools, FunctionObjectArgs};
use serde_json::json;

use crate::tools::calculator::{NAME, Operator};

/// Function definition advertised to the model.
///
/// The schema is hand-written rather than derived from `CalculatorArgs` via `schemars`:
/// a derived schema emits `$ref` / `$defs` for the operator enum, which not every
/// OpenAI-compatible endpoint accepts inside `function.parameters`. The `enum` list is
/// still generated from [`Operator::ALL`], so it cannot drift from the executor.
pub fn definition() -> ChatCompletionTools {
  ChatCompletionTools::Function(ChatCompletionTool {
    function: FunctionObjectArgs::default()
      .name(NAME)
      .description("Perform basic arithmetic operations.")
      .parameters(json!({
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
      }))
      .build()
      // The definition is a constant: a build failure is a programming error, not a
      // runtime condition, so there is nothing for the caller to recover from.
      .expect("calculator tool definition must be valid"),
  })
}
