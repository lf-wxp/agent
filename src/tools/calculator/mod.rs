//! Calculator tool.
//!
//! The definition and the executor share these argument types so the JSON Schema
//! advertised to the model can never drift from what the executor accepts.

pub mod definition;
pub mod execute;

use serde::Deserialize;

/// Tool name, used both in the definition and in dispatch.
pub const NAME: &str = "calculator";

/// Supported arithmetic operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Operator {
  Add,
  Subtract,
  Multiply,
  Divide,
}

impl Operator {
  /// Single source of truth for the `enum` list in the JSON Schema.
  pub const ALL: [Self; 4] = [Self::Add, Self::Subtract, Self::Multiply, Self::Divide];

  /// Must match the `serde` representation above.
  pub const fn as_str(self) -> &'static str {
    match self {
      Self::Add => "add",
      Self::Subtract => "subtract",
      Self::Multiply => "multiply",
      Self::Divide => "divide",
    }
  }
}

/// Arguments as produced by the model.
///
/// `operator` is an enum rather than a `String`: an unknown operation is rejected at
/// deserialization time instead of being handled by a fallback branch in the executor.
#[derive(Debug, Deserialize)]
pub struct CalculatorArgs {
  pub operator: Operator,
  pub first_number: f64,
  pub second_number: f64,
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn as_str_matches_serde_representation() {
    for operator in Operator::ALL {
      let json = format!("\"{}\"", operator.as_str());
      assert_eq!(
        serde_json::from_str::<Operator>(&json).unwrap(),
        operator,
        "schema value {json} must deserialize back"
      );
    }
  }
}
