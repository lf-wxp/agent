//! `reflect`: the model's own check of its work, written down.
//!
//! The model calls it on its own initiative — after a milestone, after a surprising or
//! failing tool result, before a final answer on something hard, or when it suspects it is
//! going in circles — to say what it has done, what is wrong with it, and what it will do
//! next. The tool does not judge anything: there is no second model behind it. Its value is
//! that the assessment has to be *written* in a fixed shape, and that it comes back as the
//! tool result, in the most recent part of the transcript, right where the next round
//! starts.
//!
//! # Stateless by design
//!
//! Like `update_plan` (see [`crate::tools::plan`]), it keeps nothing between calls: it
//! validates the arguments and renders them back. That makes one instance safe to share
//! across agents and conversations, and a reflection survives exactly what the transcript
//! survives. Nothing outside the call is touched, so it is also [`Tool::concurrency_safe`].
//!
//! # What it refuses
//!
//! A verdict of `needs_correction` or `blocked` with no issues listed says nothing, so it
//! is rejected and the model is told to name the problem (or to use `on_track`). Blank and
//! over-long entries are rejected rather than cut, because the model decides what an entry
//! says and a silently clipped one may no longer say it.

use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

use crate::{error::ResultExt, tools::tool::Tool};

/// Tool name, used both in the definition and in dispatch.
pub const NAME: &str = "reflect";

/// Most issues one reflection may list. It is meant to be read at a glance, by the model
/// on the next round and by a human watching; a longer list is a sign the model should
/// fix the first few before reflecting again.
pub const MAX_ISSUES: usize = 10;

/// Most characters in one issue.
pub const MAX_ISSUE_CHARS: usize = 300;

/// Most characters in the summary and in the next step.
pub const MAX_TEXT_CHARS: usize = 500;

const DESCRIPTION: &str = "Stop and critically check your own work. Use it after finishing a \
  milestone, after a tool call fails or returns something unexpected, before giving the \
  final answer to a hard or high-stakes request, or when you suspect you are going in \
  circles; skip it for simple requests. State what you have done so far and compare it \
  with what the user actually asked, list concrete problems (wrong or unverified \
  assumptions, errors, missing cases, skipped steps), say whether you are on track, and \
  commit to one next step. Be honest rather than reassuring: an empty issue list means you \
  looked and found none, not that you did not look. It changes nothing in the world; its \
  only effect is the written assessment you read back, so act on it.";

/// Where the work stands after the reflection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
  /// What you have is right so far (any issues are minor) and the approach still holds.
  OnTrack,
  /// Something is wrong or missing, and you will fix it before going on. Name it in
  /// `issues`.
  NeedsCorrection,
  /// You cannot go on without something you do not have: information, access or a
  /// decision. Name it in `issues`.
  Blocked,
}

impl Verdict {
  fn as_str(self) -> &'static str {
    match self {
      Self::OnTrack => "on_track",
      Self::NeedsCorrection => "needs_correction",
      Self::Blocked => "blocked",
    }
  }

  /// What to do about the verdict, appended to the result. `None` for `on_track`: there
  /// is nothing to add to "carry on".
  fn guidance(self) -> Option<&'static str> {
    match self {
      Self::OnTrack => None,
      Self::NeedsCorrection => Some(
        "Fix the issues above before going on, check that the fix worked, and do not repeat \
         what failed unchanged.",
      ),
      Self::Blocked => Some(
        "If you cannot remove the blocker yourself, tell the user plainly what is missing \
         and what you need, instead of guessing.",
      ),
    }
  }
}

/// Arguments as produced by the model.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReflectArgs {
  /// What you have done and found so far, and how it compares with what was asked (at
  /// most 500 characters).
  pub summary: String,
  /// Concrete problems you found, one per entry (at most 10, each at most 300
  /// characters). Leave empty only when you checked and found none.
  #[serde(default)]
  pub issues: Vec<String>,
  /// Where the work stands. `needs_correction` and `blocked` need at least one issue.
  pub verdict: Verdict,
  /// The single next action you will take (at most 500 characters), e.g. "re-run the
  /// tests after fixing the off-by-one in the parser".
  pub next_step: String,
}

/// See the module docs.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReflectTool;

#[async_trait::async_trait]
impl Tool for ReflectTool {
  fn name(&self) -> &str {
    NAME
  }

  fn description(&self) -> &str {
    DESCRIPTION
  }

  fn parameters(&self) -> Value {
    crate::tools::schema_of::<ReflectArgs>()
  }

  async fn execute(&self, args_json: &str) -> crate::Result<String> {
    // Pure validation and formatting: nothing to await, nothing worth a blocking thread.
    run(args_json)
  }

  /// Touches nothing outside the call: it keeps no state at all (see the module docs).
  fn concurrency_safe(&self) -> bool {
    true
  }
}

/// Validate the reflection and render it back.
fn run(args_json: &str) -> crate::Result<String> {
  let ReflectArgs {
    summary,
    issues,
    verdict,
    next_step,
  } = serde_json::from_str(args_json).wrap_err(crate::Error::tool, "invalid reflect arguments")?;

  let summary = checked("the summary", &summary, MAX_TEXT_CHARS)?;
  let next_step = checked("the next step", &next_step, MAX_TEXT_CHARS)?;
  crate::error::ensure!(
    issues.len() <= MAX_ISSUES,
    tool,
    "{} issues are listed; keep it to at most {MAX_ISSUES} by merging related ones or fixing \
     the first few before reflecting again",
    issues.len()
  );
  let issues = issues
    .iter()
    .enumerate()
    .map(|(index, issue)| checked(&format!("issue {}", index + 1), issue, MAX_ISSUE_CHARS))
    .collect::<crate::Result<Vec<_>>>()?;
  crate::error::ensure!(
    verdict == Verdict::OnTrack || !issues.is_empty(),
    tool,
    "the verdict is {} but no issues are listed; name what is wrong or blocking, or use \
     on_track if nothing is",
    verdict.as_str()
  );

  Ok(render(summary, &issues, verdict, next_step))
}

/// `text` trimmed, or an error naming `what` when it is blank or longer than `max`.
fn checked<'a>(what: &str, text: &'a str, max: usize) -> crate::Result<&'a str> {
  let text = text.trim();
  crate::error::ensure!(
    !text.is_empty(),
    tool,
    "{what} is blank; it needs a description"
  );
  let length = text.chars().count();
  crate::error::ensure!(
    length <= max,
    tool,
    "{what} is {length} characters; keep it to at most {max}"
  );
  Ok(text)
}

fn render(summary: &str, issues: &[&str], verdict: Verdict, next_step: &str) -> String {
  let mut out = format!("Reflection ({}):\nSummary: {summary}", verdict.as_str());
  if issues.is_empty() {
    out.push_str("\nIssues: none");
  } else {
    out.push_str("\nIssues:");
    for (index, issue) in issues.iter().enumerate() {
      out.push_str(&format!("\n{}. {issue}", index + 1));
    }
  }
  out.push_str(&format!("\nNext: {next_step}"));
  if let Some(guidance) = verdict.guidance() {
    out.push_str(&format!("\n{guidance}"));
  }
  out
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

  fn run_json(args: serde_json::Value) -> crate::Result<String> {
    run(&args.to_string())
  }

  fn err_of(args: serde_json::Value) -> String {
    run_json(args).unwrap_err().to_string()
  }

  #[test]
  fn renders_the_reflection_with_numbered_issues() {
    let output = run_json(json!({
      "summary": " Parser handles ints, not floats ",
      "issues": ["floats are rejected", "no test for negative numbers"],
      "verdict": "needs_correction",
      "next_step": "add float support, then test it",
    }))
    .unwrap();
    assert_eq!(
      output,
      "Reflection (needs_correction):\n\
       Summary: Parser handles ints, not floats\n\
       Issues:\n\
       1. floats are rejected\n\
       2. no test for negative numbers\n\
       Next: add float support, then test it\n\
       Fix the issues above before going on, check that the fix worked, and do not repeat \
       what failed unchanged."
    );
  }

  #[test]
  fn on_track_with_no_issues_says_so_and_adds_no_guidance() {
    let output = run_json(json!({
      "summary": "All three files read and compared",
      "verdict": "on_track",
      "next_step": "write the answer",
    }))
    .unwrap();
    assert_eq!(
      output,
      "Reflection (on_track):\nSummary: All three files read and compared\nIssues: none\nNext: \
       write the answer"
    );
  }

  #[test]
  fn blocked_tells_the_model_to_ask_rather_than_guess() {
    let output = run_json(json!({
      "summary": "Cannot read the config",
      "issues": ["permission denied on /etc/app.toml"],
      "verdict": "blocked",
      "next_step": "ask the user for access",
    }))
    .unwrap();
    assert!(output.starts_with("Reflection (blocked):"), "got: {output}");
    assert!(output.contains("tell the user plainly"), "got: {output}");
  }

  #[test]
  fn a_verdict_of_trouble_without_issues_is_rejected() {
    for verdict in ["needs_correction", "blocked"] {
      let err = err_of(json!({
        "summary": "s",
        "verdict": verdict,
        "next_step": "n",
      }));
      assert!(
        err.contains(&format!(
          "the verdict is {verdict} but no issues are listed"
        )),
        "got: {err}"
      );
    }
  }

  #[test]
  fn minor_issues_are_allowed_while_on_track() {
    assert!(
      run_json(json!({
        "summary": "s",
        "issues": ["a typo in a comment"],
        "verdict": "on_track",
        "next_step": "n",
      }))
      .is_ok()
    );
  }

  #[test]
  fn blank_entries_are_rejected() {
    let base = json!({ "summary": "s", "verdict": "on_track", "next_step": "n" });

    let mut blank_summary = base.clone();
    blank_summary["summary"] = json!("  ");
    assert!(err_of(blank_summary).contains("the summary is blank"));

    let mut blank_next = base.clone();
    blank_next["next_step"] = json!("");
    assert!(err_of(blank_next).contains("the next step is blank"));

    let mut blank_issue = base;
    blank_issue["issues"] = json!(["real", " "]);
    assert!(err_of(blank_issue).contains("issue 2 is blank"));
  }

  #[test]
  fn over_long_entries_are_rejected_not_cut() {
    let err = err_of(json!({
      "summary": "x".repeat(MAX_TEXT_CHARS + 1),
      "verdict": "on_track",
      "next_step": "n",
    }));
    assert!(err.contains("the summary is 501 characters"), "got: {err}");

    let err = err_of(json!({
      "summary": "s",
      "issues": ["short", "y".repeat(MAX_ISSUE_CHARS + 1)],
      "verdict": "needs_correction",
      "next_step": "n",
    }));
    assert!(err.contains("issue 2 is 301 characters"), "got: {err}");

    // Characters, not bytes: the limit is not tighter for non-ASCII text.
    assert!(
      run_json(json!({
        "summary": "步".repeat(MAX_TEXT_CHARS),
        "verdict": "on_track",
        "next_step": "n",
      }))
      .is_ok()
    );
  }

  #[test]
  fn too_many_issues_are_rejected() {
    let issues: Vec<_> = (0..=MAX_ISSUES).map(|i| format!("issue {i}")).collect();
    let err = err_of(json!({
      "summary": "s",
      "issues": issues,
      "verdict": "needs_correction",
      "next_step": "n",
    }));
    assert!(err.contains("keep it to at most"), "got: {err}");
  }

  #[test]
  fn malformed_arguments_are_rejected() {
    assert!(run("not json").is_err());
    assert!(
      run_json(json!({ "summary": "s", "verdict": "fine", "next_step": "n" })).is_err(),
      "an unknown verdict"
    );
    assert!(
      run_json(json!({ "summary": "s", "verdict": "on_track" })).is_err(),
      "a missing next_step"
    );
  }

  #[test]
  fn the_schema_offers_exactly_the_three_verdicts() {
    let schema = ReflectTool.parameters().to_string();
    for verdict in ["on_track", "needs_correction", "blocked"] {
      assert!(
        schema.contains(&format!("\"{verdict}\"")),
        "missing {verdict}: {schema}"
      );
    }
  }

  /// The limits reach the model only as prose in the field descriptions, written by hand;
  /// this keeps those numbers in step with the constants that enforce them.
  #[test]
  fn the_schema_states_the_limits_that_are_enforced() {
    let schema = ReflectTool.parameters();
    let description = |field: &str| {
      schema["properties"][field]["description"]
        .as_str()
        .unwrap_or_default()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
    };
    let text_limit = format!("at most {MAX_TEXT_CHARS} characters");
    for field in ["summary", "next_step"] {
      assert!(
        description(field).contains(&text_limit),
        "`{field}`: {}",
        description(field)
      );
    }
    let issue_limits = format!("at most {MAX_ISSUES}, each at most {MAX_ISSUE_CHARS} characters");
    assert!(
      description("issues").contains(&issue_limits),
      "`issues`: {}",
      description("issues")
    );
  }

  #[test]
  fn issues_are_optional_in_the_schema() {
    let schema = ReflectTool.parameters();
    let required: Vec<&str> = schema["required"]
      .as_array()
      .unwrap()
      .iter()
      .filter_map(Value::as_str)
      .collect();
    assert!(!required.contains(&"issues"), "got: {required:?}");
    for field in ["summary", "verdict", "next_step"] {
      assert!(
        required.contains(&field),
        "`{field}` not required: {required:?}"
      );
    }
  }

  #[tokio::test]
  async fn executes_through_the_trait() {
    let output = ReflectTool
      .execute(r#"{"summary":"done","verdict":"on_track","next_step":"answer"}"#)
      .await
      .unwrap();
    assert!(output.contains("Summary: done"), "got: {output}");
    assert!(ReflectTool.concurrency_safe());
  }

  /// Through a real run rather than the tool alone: the model is offered the tool by
  /// default, calls it, and the rendered assessment is what it reads back next round.
  #[tokio::test]
  async fn an_agent_reads_its_reflection_back() {
    let backend = ScriptedBackend::replying(vec![
      tool_call(
        "call_1",
        NAME,
        json!({
          "summary": "Drafted the answer",
          "issues": ["did not check the units"],
          "verdict": "needs_correction",
          "next_step": "recompute in metres",
        }),
      ),
      answer("fixed"),
    ]);
    let agent = Agent::new(
      backend.provider(),
      "gpt-test",
      Some("You reflect."),
      Arc::new(ToolRegistry::builtin().unwrap()),
    )
    .with_model_settings(ModelSettings::default());

    let result = agent.run("convert it").await.unwrap();

    assert_eq!(result.output, "fixed");
    let requests = backend.seen();
    assert!(
      requests[0]["tools"].to_string().contains(NAME),
      "the model is offered the tool by default"
    );
    let tool_results: Vec<String> = requests[1]["messages"]
      .as_array()
      .unwrap()
      .iter()
      .filter(|message| message["role"] == "tool")
      .filter_map(|message| message["content"].as_str().map(str::to_owned))
      .collect();
    assert_eq!(tool_results.len(), 1);
    assert!(
      tool_results[0].starts_with(
        "Reflection (needs_correction):\nSummary: Drafted the answer\nIssues:\n1. did not check \
         the units\nNext: recompute in metres"
      ),
      "got: {}",
      tool_results[0]
    );
  }
}
