use crate::tools::calculator::{CalculatorArgs, Operator};

/// Run the tool against the raw JSON arguments produced by the model.
///
/// Returns a string in both the success and failure cases: tool errors are fed back as
/// tool messages so the model can retry with corrected arguments, rather than aborting
/// the whole conversation.
pub fn run(arguments: &str) -> String {
  let args = match serde_json::from_str::<CalculatorArgs>(arguments) {
    Ok(args) => args,
    Err(err) => return format!("Error: invalid arguments: {err}"),
  };

  match calculate(args.operator, args.first_number, args.second_number) {
    Ok(value) => value.to_string(),
    Err(err) => format!("Error: {err}"),
  }
}

/// Pure arithmetic, kept apart from JSON handling so it can be tested directly.
pub fn calculate(operator: Operator, first_number: f64, second_number: f64) -> Result<f64, String> {
  match operator {
    Operator::Add => Ok(first_number + second_number),
    Operator::Subtract => Ok(first_number - second_number),
    Operator::Multiply => Ok(first_number * second_number),
    // Float division by zero yields inf / NaN instead of panicking, which would feed a
    // meaningless number back to the model; reject it explicitly.
    Operator::Divide if second_number == 0.0 => Err("cannot divide by zero".to_owned()),
    Operator::Divide => Ok(first_number / second_number),
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn computes_each_operation() {
    assert_eq!(calculate(Operator::Add, 2.0, 3.0), Ok(5.0));
    assert_eq!(calculate(Operator::Subtract, 2.0, 3.0), Ok(-1.0));
    assert_eq!(calculate(Operator::Multiply, 2.0, 3.0), Ok(6.0));
    assert_eq!(calculate(Operator::Divide, 6.0, 3.0), Ok(2.0));
  }

  #[test]
  fn rejects_division_by_zero() {
    assert!(calculate(Operator::Divide, 1.0, 0.0).is_err());
  }

  #[test]
  fn runs_from_json_arguments() {
    let arguments = r#"{"operator":"multiply","first_number":5875,"second_number":467}"#;
    assert_eq!(run(arguments), "2743625");
  }

  #[test]
  fn reports_unknown_operator_as_text() {
    let arguments = r#"{"operator":"pow","first_number":2,"second_number":8}"#;
    assert!(run(arguments).starts_with("Error: invalid arguments"));
  }

  #[test]
  fn reports_malformed_json_as_text() {
    assert!(run("not json").starts_with("Error: invalid arguments"));
  }
}
