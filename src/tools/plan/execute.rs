use crate::{
  error::ResultExt,
  tools::plan::{
    MAX_EXPLANATION_CHARS, MAX_STEP_CHARS, MAX_STEPS, PlanStep, StepStatus, UpdatePlanArgs,
  },
};

/// Validate the plan and render it as a checklist.
///
/// Rejected — so the model can correct it — when a step is blank, when there are more
/// than [`MAX_STEPS`], when a step or the explanation is longer than [`MAX_STEP_CHARS`] /
/// [`MAX_EXPLANATION_CHARS`], or when more than one step is `in_progress`: "the step I am
/// on" is what makes the plan useful to read, and two of them means it says nothing.
/// Too long is refused rather than cut, because the model decides what a step says and a
/// silently clipped one may no longer say it.
pub fn run(args_json: &str) -> crate::Result<String> {
  let UpdatePlanArgs { plan, explanation } = serde_json::from_str(args_json)
    .wrap_err(crate::Error::tool, "invalid update_plan arguments")?;

  crate::error::ensure!(
    plan.len() <= MAX_STEPS,
    tool,
    "the plan has {} steps; keep it to at most {MAX_STEPS} by merging small ones",
    plan.len()
  );
  if let Some(position) = plan.iter().position(|step| step.step.trim().is_empty()) {
    crate::error::bail!(
      tool,
      "step {} is blank; every step needs a description",
      position + 1
    );
  }
  if let Some((position, step)) = plan
    .iter()
    .enumerate()
    .find(|(_, step)| step.step.trim().chars().count() > MAX_STEP_CHARS)
  {
    crate::error::bail!(
      tool,
      "step {} is {} characters; keep each step to at most {MAX_STEP_CHARS} and move the \
       detail into your reply",
      position + 1,
      step.step.trim().chars().count()
    );
  }
  if let Some(note) = &explanation
    && note.trim().chars().count() > MAX_EXPLANATION_CHARS
  {
    crate::error::bail!(
      tool,
      "the explanation is {} characters; keep it to at most {MAX_EXPLANATION_CHARS}",
      note.trim().chars().count()
    );
  }
  let in_progress: Vec<&str> = plan
    .iter()
    .filter(|step| step.status == StepStatus::InProgress)
    .map(|step| step.step.trim())
    .collect();
  crate::error::ensure!(
    in_progress.len() <= 1,
    tool,
    "{} steps are in_progress ({}); keep exactly one in_progress and mark the others \
     pending or completed",
    in_progress.len(),
    in_progress.join("; ")
  );

  Ok(render(&plan, explanation.as_deref()))
}

fn render(plan: &[PlanStep], explanation: Option<&str>) -> String {
  let note = explanation.map(str::trim).filter(|note| !note.is_empty());

  if plan.is_empty() {
    // Why it was cleared is worth keeping: it is the only record of the decision.
    return match note {
      Some(note) => format!("Plan cleared.\nNote: {note}"),
      None => "Plan cleared.".to_owned(),
    };
  }

  let done = plan
    .iter()
    .filter(|step| step.status == StepStatus::Completed)
    .count();
  let mut out = format!("Plan updated ({done}/{} completed):", plan.len());
  if let Some(note) = note {
    out.push_str(&format!("\nNote: {note}"));
  }
  for step in plan {
    let mark = match step.status {
      StepStatus::Completed => "[x]",
      StepStatus::InProgress => "[>]",
      StepStatus::Pending => "[ ]",
    };
    out.push_str(&format!("\n{mark} {}", step.step.trim()));
  }
  out
}

#[cfg(test)]
mod tests {
  use serde_json::json;

  use super::*;

  fn run_json(args: serde_json::Value) -> crate::Result<String> {
    run(&args.to_string())
  }

  #[test]
  fn renders_the_plan_as_a_checklist_with_progress() {
    let output = run_json(json!({
      "plan": [
        { "step": "Read the config", "status": "completed" },
        { "step": " Write the parser ", "status": "in_progress" },
        { "step": "Add tests", "status": "pending" },
      ],
    }))
    .unwrap();
    assert_eq!(
      output,
      "Plan updated (1/3 completed):\n[x] Read the config\n[>] Write the parser\n[ ] Add tests"
    );
  }

  #[test]
  fn an_explanation_is_shown_under_the_header() {
    let output = run_json(json!({
      "plan": [{ "step": "Fix the lexer", "status": "in_progress" }],
      "explanation": "the bug is in the lexer",
    }))
    .unwrap();
    assert!(
      output.starts_with("Plan updated (0/1 completed):\nNote: the bug is in the lexer\n"),
      "got: {output}"
    );
  }

  #[test]
  fn an_empty_plan_clears_it() {
    assert_eq!(run_json(json!({ "plan": [] })).unwrap(), "Plan cleared.");
  }

  #[test]
  fn clearing_a_plan_keeps_the_reason() {
    let output = run_json(json!({ "plan": [], "explanation": "  the task changed  " })).unwrap();
    assert_eq!(output, "Plan cleared.\nNote: the task changed");
  }

  #[test]
  fn a_step_longer_than_the_limit_is_rejected_not_cut() {
    let long = "x".repeat(MAX_STEP_CHARS + 1);
    let err = run_json(json!({
      "plan": [
        { "step": "short", "status": "pending" },
        { "step": long, "status": "pending" },
      ],
    }))
    .unwrap_err()
    .to_string();
    assert!(err.contains("step 2 is 201 characters"), "got: {err}");

    // Characters, not bytes: the limit is not tighter for non-ASCII text.
    let at_limit = "步".repeat(MAX_STEP_CHARS);
    assert!(run_json(json!({ "plan": [{ "step": at_limit, "status": "pending" }] })).is_ok());
  }

  #[test]
  fn an_explanation_longer_than_the_limit_is_rejected() {
    let err = run_json(json!({
      "plan": [{ "step": "a", "status": "pending" }],
      "explanation": "y".repeat(MAX_EXPLANATION_CHARS + 1),
    }))
    .unwrap_err()
    .to_string();
    assert!(
      err.contains("the explanation is 501 characters"),
      "got: {err}"
    );
  }

  #[test]
  fn two_steps_in_progress_are_rejected() {
    let err = run_json(json!({
      "plan": [
        { "step": "a", "status": "in_progress" },
        { "step": "b", "status": "in_progress" },
      ],
    }))
    .unwrap_err()
    .to_string();
    assert!(err.contains("2 steps are in_progress"), "got: {err}");
  }

  #[test]
  fn a_blank_step_is_rejected() {
    let err = run_json(json!({
      "plan": [
        { "step": "a", "status": "pending" },
        { "step": "  ", "status": "pending" },
      ],
    }))
    .unwrap_err()
    .to_string();
    assert!(err.contains("step 2 is blank"), "got: {err}");
  }

  #[test]
  fn an_oversized_plan_is_rejected() {
    let plan: Vec<_> = (0..=MAX_STEPS)
      .map(|i| json!({ "step": format!("step {i}"), "status": "pending" }))
      .collect();
    let err = run_json(json!({ "plan": plan })).unwrap_err().to_string();
    assert!(err.contains("at most"), "got: {err}");
  }

  #[test]
  fn an_unknown_status_is_rejected() {
    assert!(run_json(json!({ "plan": [{ "step": "a", "status": "done" }] })).is_err());
  }
}
