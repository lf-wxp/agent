//! `update_plan`: the model's own checklist for a multi-step task.
//!
//! # Stateless by design
//!
//! Every call carries the *whole* plan, which replaces the previous one; the tool keeps
//! nothing between calls. The current plan is simply the arguments of the latest
//! `update_plan` call in the transcript. That is what makes it safe to share one
//! instance across agents and conversations (a [`crate::tools::ToolRegistry`] is shared
//! behind an `Arc`), and what makes the plan survive everything the transcript survives —
//! persistence, suspension and resume, a process restart — with no extra storage.
//!
//! The tool's job is therefore only to validate the plan and render it back as a
//! checklist: the model sees its plan confirmed in a compact form, and a front-end
//! showing tool results shows the human the same list.
//!
//! # What does not survive: context trimming
//!
//! "Survives the transcript" is not "is always in front of the model". The transcript is
//! complete, but each *request* is trimmed to a token budget
//! ([`crate::callback::context_optimizer`]): the opening message is pinned and the middle
//! is dropped first, so a plan that was last sent long ago can fall out of what the model
//! is shown, with nothing to say it did. Two things keep that rare. Every update is a fresh
//! copy in the recent tail, and the tool's description tells the model to update at each
//! status change — so the plan is only as old as the current step. And the result is never
//! compacted to a stub (see `compaction`), so while a copy is in view it is whole. A step
//! that itself takes more rounds than the budget holds is the case that can still lose it;
//! the transcript, and any front-end showing it, still has it.

pub mod execute;

use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

use crate::tools::tool::Tool;

/// Tool name, used both in the definition and in dispatch.
pub const NAME: &str = "update_plan";

/// Most steps one plan may hold. A plan is meant to be read at a glance — by the model
/// on every later round, and by a human watching — and it is re-sent in full on every
/// update, so a runaway list costs tokens each time it changes.
pub const MAX_STEPS: usize = 50;

/// Most characters in one step's description. A step is a label to read at a glance, not
/// a place for detail; the cap also bounds what every update re-sends.
pub const MAX_STEP_CHARS: usize = 200;

/// Most characters in an update's `explanation`.
pub const MAX_EXPLANATION_CHARS: usize = 500;

const DESCRIPTION: &str = "Record your plan for a multi-step task and keep it up to date. \
  Use it when a task needs three or more distinct steps; skip it for simple requests. \
  Send the whole plan every time: it replaces the previous one. While working, keep \
  exactly one step `in_progress`, and mark a step `completed` as soon as it is done, \
  before starting the next: send the plan again at every status change, so the latest \
  copy is always recent. Keep each step to a short phrase (at most 200 characters) and \
  the explanation to a sentence or two. Revise the plan when you learn something that \
  changes it.";

/// Where one step stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum StepStatus {
  /// Not started yet.
  Pending,
  /// The step being worked on right now. At most one step in a plan has this status.
  InProgress,
  /// Done. Mark it as soon as it is, before starting the next step.
  Completed,
}

/// One step of the plan.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct PlanStep {
  /// What this step does, in a few words (at most 200 characters), e.g. "Add tests for
  /// the parser".
  pub step: String,
  /// Where this step stands.
  pub status: StepStatus,
}

/// Arguments as produced by the model.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct UpdatePlanArgs {
  /// The complete plan, in order. Replaces the previous plan; an empty list clears it.
  pub plan: Vec<PlanStep>,
  /// Optional short note on what changed and why (at most 500 characters), e.g. "the bug
  /// is in the lexer, not the parser". Shown with an empty plan too, to say why it was
  /// cleared.
  #[serde(default)]
  pub explanation: Option<String>,
}

/// See the module docs.
#[derive(Debug, Clone, Copy, Default)]
pub struct UpdatePlanTool;

#[async_trait::async_trait]
impl Tool for UpdatePlanTool {
  fn name(&self) -> &str {
    NAME
  }

  fn description(&self) -> &str {
    DESCRIPTION
  }

  fn parameters(&self) -> Value {
    crate::tools::schema_of::<UpdatePlanArgs>()
  }

  async fn execute(&self, args_json: &str) -> crate::Result<String> {
    // Pure validation and formatting: nothing to await, nothing worth a blocking thread.
    execute::run(args_json)
  }

  /// Touches nothing outside the call: it keeps no state at all (see the module docs).
  fn concurrency_safe(&self) -> bool {
    true
  }
}

#[cfg(test)]
mod tests {
  use std::sync::Arc;

  use serde_json::json;

  use super::*;
  use crate::{
    agent::Agent,
    llm::{
      settings::ModelSettings,
      testing::{ScriptedBackend, answer, tool_call},
    },
    tools::ToolRegistry,
  };

  #[test]
  fn the_schema_offers_exactly_the_three_statuses() {
    let schema = UpdatePlanTool.parameters().to_string();
    for status in ["pending", "in_progress", "completed"] {
      assert!(
        schema.contains(&format!("\"{status}\"")),
        "missing {status}: {schema}"
      );
    }
  }

  #[tokio::test]
  async fn executes_through_the_trait() {
    let output = UpdatePlanTool
      .execute(r#"{"plan":[{"step":"Read the code","status":"in_progress"}]}"#)
      .await
      .unwrap();
    assert!(output.contains("Read the code"), "got: {output}");
  }

  /// Through a real run rather than the tool alone: the model is offered the tool, its
  /// calls are executed, and each rendered checklist is what it reads back next round.
  #[tokio::test]
  async fn an_agent_keeps_a_plan_across_rounds() {
    let backend = ScriptedBackend::replying(vec![
      tool_call(
        "call_1",
        NAME,
        json!({ "plan": [
          { "step": "Read", "status": "in_progress" },
          { "step": "Write", "status": "pending" },
        ] }),
      ),
      tool_call(
        "call_2",
        NAME,
        json!({ "plan": [
          { "step": "Read", "status": "completed" },
          { "step": "Write", "status": "in_progress" },
        ] }),
      ),
      answer("all done"),
    ]);
    let agent = Agent::new(
      backend.provider(),
      "gpt-test",
      Some("You plan."),
      Arc::new(ToolRegistry::builtin().unwrap()),
    )
    .with_model_settings(ModelSettings::default());

    let result = agent.run("do the thing").await.unwrap();

    assert_eq!(result.output, "all done");
    let requests = backend.seen();
    assert!(
      requests[0]["tools"].to_string().contains(NAME),
      "the model is offered the tool by default"
    );
    let tool_results = |request: &serde_json::Value| -> Vec<String> {
      request["messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|message| message["role"] == "tool")
        .filter_map(|message| message["content"].as_str().map(str::to_owned))
        .collect()
    };
    assert_eq!(
      tool_results(&requests[1]),
      ["Plan updated (0/2 completed):\n[>] Read\n[ ] Write"]
    );
    assert_eq!(
      tool_results(&requests[2]),
      [
        "Plan updated (0/2 completed):\n[>] Read\n[ ] Write",
        "Plan updated (1/2 completed):\n[x] Read\n[>] Write",
      ]
    );
  }
}
