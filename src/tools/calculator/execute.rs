use anyhow::Context;

use crate::tools::calculator::{CalculatorArgs, Operator};

/// Run the tool against the raw JSON arguments produced by the model.
///
/// Errors are returned rather than formatted: the registry turns them into a tool message
/// so the model can retry with corrected arguments.
pub fn run(arguments: &str) -> anyhow::Result<String> {
  let args =
    serde_json::from_str::<CalculatorArgs>(arguments).context("invalid calculator arguments")?;

  Ok(calculate(args.operator, args.first_number, args.second_number)?.to_string())
}

/// Pure arithmetic, kept apart from JSON handling so it can be tested directly.
pub fn calculate(operator: Operator, first_number: f64, second_number: f64) -> anyhow::Result<f64> {
  match operator {
    Operator::Add => Ok(first_number + second_number),
    Operator::Subtract => Ok(first_number - second_number),
    Operator::Multiply => Ok(first_number * second_number),
    // Float division by zero yields inf / NaN instead of panicking, which would feed a
    // meaningless number back to the model; reject it explicitly.
    Operator::Divide if second_number == 0.0 => anyhow::bail!("cannot divide by zero"),
    Operator::Divide => Ok(first_number / second_number),
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn computes_each_operation() {
    assert_eq!(calculate(Operator::Add, 2.0, 3.0).unwrap(), 5.0);
    assert_eq!(calculate(Operator::Subtract, 2.0, 3.0).unwrap(), -1.0);
    assert_eq!(calculate(Operator::Multiply, 2.0, 3.0).unwrap(), 6.0);
    assert_eq!(calculate(Operator::Divide, 6.0, 3.0).unwrap(), 2.0);
  }

  #[test]
  fn rejects_division_by_zero() {
    assert!(calculate(Operator::Divide, 1.0, 0.0).is_err());
  }

  #[test]
  fn runs_from_json_arguments() {
    let arguments = r#"{"operator":"multiply","first_number":5875,"second_number":467}"#;
    assert_eq!(run(arguments).unwrap(), "2743625");
  }

  #[test]
  fn reports_unknown_operator() {
    let arguments = r#"{"operator":"pow","first_number":2,"second_number":8}"#;
    assert!(run(arguments).is_err());
  }

  #[test]
  fn reports_malformed_json() {
    assert!(run("not json").is_err());
  }
}
