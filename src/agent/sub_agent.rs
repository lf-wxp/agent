//! [`AgentTool`]: an [`Agent`] offered to another agent as a tool — a sub-agent.
//!
//! The parent model delegates a self-contained task by calling the tool; the sub-agent
//! runs its own tool loop, with its own model, instructions, tools and hooks, and the
//! parent gets back only its final answer. That is the point of delegating: the
//! sub-agent's intermediate steps (searches, file reads, dead ends) are spent in *its*
//! context, not the parent's, which stays small and on topic.
//!
//! # What the sub-agent sees
//!
//! Nothing of the parent's conversation. Every call starts a fresh run whose only input
//! is the `task` argument, so the parent model has to put everything the sub-agent needs
//! into it — which the argument's schema tells it. Calls are also independent of each
//! other: a second call does not remember the first.
//!
//! # Nesting
//!
//! A sub-agent may itself have sub-agents, but never itself: an [`Agent`]'s tools are
//! fixed when it is built, and an [`AgentTool`] can only wrap an agent that already
//! exists, so a cycle cannot be constructed. Depth is whatever the caller builds.
//!
//! # Sandbox and approval are not inherited
//!
//! The sub-agent's hooks apply to its own tool calls, and *only* those. A parent guarded by
//! a [`crate::callback::path_guard::WorkspaceGuardCallback`] and a
//! [`crate::callback::dual_approval::DualApprovalCallback`] does not pass either on: the
//! parent's guard sees one call to the sub-agent's tool, whose only argument is a `task`
//! string, and has nothing to check it against. A sub-agent given `write_file` or
//! `run_command` but no hooks of its own is therefore a way around the parent's sandbox
//! and approval step.
//!
//! So build the sub-agent with the same gates it would need if it were the top-level
//! agent. [`AgentTool::new`] logs a warning when it is handed an agent that has tools which
//! are not read-only (see [`crate::tools::Tool::concurrency_safe`]) and no before-tool hook
//! at all; it does not refuse, since a caller may have good reason (a throwaway scratch
//! directory, say).
//!
//! A call nobody answers is recorded as unanswered and the sub-agent carries on, as
//! [`Agent::run`] always does; it cannot suspend the parent.
//!
//! # Cancellation and time
//!
//! Cancelling the parent run, or the parent's tool timeout expiring, drops the sub-agent's
//! run mid-flight, which abandons its model request and its running tools like any other
//! cancelled tool call. A sub-agent working through many rounds can legitimately need
//! longer than the parent's default tool timeout; see [`AgentTool::with_timeout`].
//!
//! # Limits are the sub-agent's own
//!
//! Nothing the parent has spent or been allowed carries over. The parent's round budget
//! ([`Agent::with_max_steps`]) counts one round for the delegation however long it takes;
//! the sub-agent runs under *its own* budget, so one call can cost up to that many model
//! rounds, and several delegated in the same round multiply it. Bound the sub-agent with
//! its own `with_max_steps`, keep [`AgentTool::with_timeout`] honest, and cap overlap with
//! the parent's [`Agent::with_tool_concurrency`]. When the sub-agent runs out of rounds the
//! answer it gives is flagged as possibly incomplete (a note is appended to it).
//!
//! # Failures
//!
//! A failed run comes back as a tool error that carries the whole cause chain (the parent
//! model sees only the outermost message, so the cause is what it can act on), cut to
//! 1000 characters: a provider error can quote a whole response body, and that does not
//! belong in the parent's context. The text is still model-facing and ends up in the
//! parent's transcript, so a sub-agent whose errors can carry secrets needs hooks of its
//! own to scrub them.
//!
//! # Token usage
//!
//! What the sub-agent spends is charged to the parent's run as it is spent (see
//! [`crate::agent::usage`]), so it shows up in the parent's `usage` / `usage_by_model` —
//! under the sub-agent's model, which is what pricing needs — even for a delegation that
//! fails or is cut off. It is also accumulated on the tool itself, across calls, for a
//! caller that wants the sub-agent's share on its own: [`AgentTool::usage`]. That one
//! only counts calls that finish, since a run that fails or is dropped loses the
//! transcript its usage was recorded in.

use std::{
  sync::{Arc, Mutex},
  time::Duration,
};

use anyhow::Context;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

use crate::{
  agent::{Agent, TokenUsage},
  tools::Tool,
  util::truncate_chars,
};

/// Longest tool name providers accept.
const MAX_NAME_CHARS: usize = 64;

/// Most characters of a failed run's cause chain handed to the parent model.
const MAX_ERROR_CHARS: usize = 1000;

/// Appended to an answer produced after the sub-agent ran out of tool rounds, so the
/// parent model does not take a best effort for a finished job.
const BUDGET_EXHAUSTED_NOTE: &str = "\n\n[The sub-agent ran out of tool rounds before \
  finishing, so this answer may be incomplete.]";

/// Arguments as produced by the parent model.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct AgentToolArgs {
  /// The task to hand over, written so it can be done without any other context: the
  /// sub-agent sees nothing of this conversation, so include every detail it needs
  /// (names, paths, constraints, what a good answer looks like).
  pub task: String,
}

/// Names of the tools `agent` has that are not read-only, when nothing gates its calls.
///
/// "Not read-only" is [`Tool::concurrency_safe`] being `false`, which is also what a tool
/// of unknown behaviour (an MCP tool) reports: the same conservative reading the scheduler
/// uses. Empty when the agent has any before-tool hook, whatever it checks — whether the
/// hook is the right one is not something this can judge.
fn ungated_side_effect_tools(agent: &Agent) -> Vec<&str> {
  if agent.has_before_tool_callbacks() {
    return Vec::new();
  }
  agent
    .toolbox()
    .iter()
    .filter(|tool| !tool.concurrency_safe())
    .map(|tool| tool.name())
    .collect()
}

/// The error text for a failed run: the whole cause chain, cut to [`MAX_ERROR_CHARS`].
fn failure_message(name: &str, err: &anyhow::Error) -> String {
  let cause = format!("{err:#}");
  let shown = truncate_chars(&cause, MAX_ERROR_CHARS);
  let ellipsis = if shown.len() < cause.len() { "…" } else { "" };
  format!("sub-agent `{name}` failed: {shown}{ellipsis}")
}

/// See the module docs.
pub struct AgentTool {
  name: String,
  description: String,
  agent: Arc<Agent>,
  timeout: Option<Duration>,
  usage: Mutex<TokenUsage>,
}

impl AgentTool {
  /// Offer `agent` to another agent as the tool `name`.
  ///
  /// `description` is what the parent model reads to decide when to delegate, so say
  /// what the sub-agent is good at and what it returns, e.g. "Researches a question on
  /// the web and answers with sources".
  ///
  /// # Errors
  ///
  /// If `name` is not 1–64 ASCII letters, digits, `_` or `-` — what providers accept as
  /// a tool name — or `description` is blank.
  ///
  /// # Sandbox and approval
  ///
  /// `agent` keeps only its own hooks; the parent's sandbox and approval step do not reach
  /// it (see the module docs). A warning is logged when it has tools that are not
  /// read-only and no before-tool hook at all.
  pub fn new(
    name: impl Into<String>,
    description: impl Into<String>,
    agent: impl Into<Arc<Agent>>,
  ) -> anyhow::Result<Self> {
    let name = name.into();
    let description = description.into();
    anyhow::ensure!(
      !name.is_empty()
        && name.len() <= MAX_NAME_CHARS
        && name
          .chars()
          .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'),
      "invalid sub-agent tool name `{name}`: use 1-{MAX_NAME_CHARS} ASCII letters, digits, \
       `_` or `-`"
    );
    anyhow::ensure!(
      !description.trim().is_empty(),
      "sub-agent tool `{name}` needs a description, or the model cannot tell when to use it"
    );
    let agent = agent.into();
    let ungated = ungated_side_effect_tools(&agent);
    if !ungated.is_empty() {
      tracing::warn!(
        sub_agent = %name,
        tools = ?ungated,
        "sub-agent has tools that are not read-only and no before-tool hook: the parent's \
         sandbox and approval step do not apply to it"
      );
    }
    Ok(Self {
      name,
      description,
      agent,
      timeout: None,
      usage: Mutex::new(TokenUsage::default()),
    })
  }

  /// How long one delegated task may take before the parent gives up on it, overriding
  /// the parent's default tool timeout (see [`crate::tools::Tool::timeout`]).
  #[must_use]
  pub fn with_timeout(mut self, timeout: Duration) -> Self {
    self.timeout = Some(timeout);
    self
  }

  /// Tokens the sub-agent has spent across every call that finished. The same tokens
  /// are also in the parent run's usage; see the module docs.
  pub fn usage(&self) -> TokenUsage {
    *self
      .usage
      .lock()
      .unwrap_or_else(|poisoned| poisoned.into_inner())
  }
}

#[async_trait::async_trait]
impl Tool for AgentTool {
  fn name(&self) -> &str {
    &self.name
  }

  fn description(&self) -> &str {
    &self.description
  }

  fn parameters(&self) -> Value {
    serde_json::to_value(schemars::schema_for!(AgentToolArgs))
      .expect("schema is always serializable")
  }

  async fn execute(&self, args_json: &str) -> anyhow::Result<String> {
    let AgentToolArgs { task } =
      serde_json::from_str(args_json).context("invalid sub-agent arguments")?;
    anyhow::ensure!(
      !task.trim().is_empty(),
      "the task is empty; describe what the sub-agent should do"
    );

    // `{err:#}`, not `?`: the parent turns this error into the text its model reads, and
    // that only shows the outermost message — the cause is what the model can act on.
    // Cut, because a provider error can quote a whole response body.
    let result = self
      .agent
      .run(&task)
      .await
      .map_err(|err| anyhow::anyhow!(failure_message(&self.name, &err)))?;

    let spent = result.context.usage;
    self
      .usage
      .lock()
      .unwrap_or_else(|poisoned| poisoned.into_inner())
      .merge(&spent);
    tracing::info!(
      sub_agent = %self.name,
      steps = result.context.current_step,
      usage = ?spent,
      budget_exhausted = result.budget_exhausted,
      "sub-agent finished"
    );

    let mut output = result.output;
    anyhow::ensure!(
      !output.trim().is_empty(),
      "sub-agent `{}` finished without an answer",
      self.name
    );
    if result.budget_exhausted {
      output.push_str(BUDGET_EXHAUSTED_NOTE);
    }
    Ok(output)
  }

  fn timeout(&self) -> Option<Duration> {
    self.timeout
  }

  /// Safe to overlap with its siblings exactly when everything the sub-agent can do is:
  /// its model calls are independent, so its tools decide. One tool with side effects —
  /// or one of unknown behaviour, such as an MCP tool — makes the whole delegation take
  /// its turn.
  fn concurrency_safe(&self) -> bool {
    self
      .agent
      .toolbox()
      .iter()
      .all(|tool| tool.concurrency_safe())
  }
}

#[cfg(test)]
mod tests {
  use std::sync::atomic::{AtomicBool, Ordering};

  use serde_json::json;

  use super::*;
  use crate::{
    callback::path_guard::WorkspaceGuardCallback,
    llm::{
      backend::{ChatBackend, ChatChunkStream, ChatRequest, ChatResponse},
      provider::Provider,
      settings::ModelSettings,
      testing::{ScriptedBackend, answer, tool_call, with_usage},
    },
    tools::ToolRegistry,
  };

  fn agent_on(backend: &Arc<ScriptedBackend>, toolbox: ToolRegistry, instructions: &str) -> Agent {
    Agent::new(
      backend.provider(),
      "gpt-test",
      Some(instructions),
      Arc::new(toolbox),
    )
    .with_model_settings(ModelSettings::default())
  }

  fn researcher(backend: &Arc<ScriptedBackend>) -> AgentTool {
    AgentTool::new(
      "researcher",
      "Finds things out",
      agent_on(backend, ToolRegistry::empty(), "You research."),
    )
    .unwrap()
  }

  /// Every message the request sent, as `(role, content)`.
  fn messages(request: &Value) -> Vec<(String, String)> {
    request["messages"]
      .as_array()
      .unwrap()
      .iter()
      .map(|m| {
        (
          m["role"].as_str().unwrap_or_default().to_owned(),
          m["content"].as_str().unwrap_or_default().to_owned(),
        )
      })
      .collect()
  }

  #[tokio::test]
  async fn delegates_the_task_and_returns_the_answer() {
    let sub = ScriptedBackend::replying(vec![answer("Paris")]);
    let tool = researcher(&sub);

    let output = tool
      .execute(&json!({ "task": "What is the capital of France?" }).to_string())
      .await
      .unwrap();

    assert_eq!(output, "Paris");
    let sent = messages(&sub.seen()[0]);
    assert_eq!(
      sent,
      [
        ("system".to_owned(), "You research.".to_owned()),
        (
          "user".to_owned(),
          "What is the capital of France?".to_owned()
        ),
      ],
      "the sub-agent starts from its own instructions and the task, nothing else"
    );
  }

  /// The whole point: the parent delegates, the sub-agent works in its own context, and
  /// only the answer comes back — the parent's conversation never reaches the sub-agent.
  #[tokio::test]
  async fn a_parent_agent_delegates_through_the_tool() {
    let sub = ScriptedBackend::replying(vec![answer("It was founded in 1998.")]);
    let parent = ScriptedBackend::replying(vec![
      tool_call(
        "call_1",
        "researcher",
        json!({ "task": "When was Example Corp founded?" }),
      ),
      answer("Example Corp dates from 1998."),
    ]);
    let mut toolbox = ToolRegistry::empty();
    toolbox.add(Arc::new(researcher(&sub))).unwrap();
    let agent = agent_on(&parent, toolbox, "You are the lead.");

    let result = agent
      .run("PRIVATE-NOTE: tell me about Example Corp")
      .await
      .unwrap();

    assert_eq!(result.output, "Example Corp dates from 1998.");
    let follow_up = &parent.seen()[1];
    assert!(
      messages(follow_up)
        .iter()
        .any(|(role, content)| role == "tool" && content == "It was founded in 1998."),
      "the sub-agent's answer is the tool result: {follow_up}"
    );
    assert!(
      !sub.seen()[0].to_string().contains("PRIVATE-NOTE"),
      "the parent's conversation leaked into the sub-agent"
    );
  }

  /// A best effort after running out of rounds must not read as a finished job.
  #[tokio::test]
  async fn an_answer_given_after_the_round_budget_ran_out_says_so() {
    let sub = ScriptedBackend::replying(vec![answer("Partial findings")]);
    let calculator = ToolRegistry::select(&["calculator".to_owned()]).unwrap();
    let tool = AgentTool::new(
      "worker",
      "Does work",
      agent_on(&sub, calculator, "You work.").with_max_steps(0),
    )
    .unwrap();

    let output = tool.execute(r#"{"task":"do it"}"#).await.unwrap();

    assert!(output.starts_with("Partial findings"), "got: {output}");
    assert!(output.contains("may be incomplete"), "got: {output}");
  }

  #[tokio::test]
  async fn usage_accumulates_across_calls() {
    let sub = ScriptedBackend::replying(vec![
      with_usage(answer("one"), 10, 5),
      with_usage(answer("two"), 20, 7),
    ]);
    let tool = researcher(&sub);

    tool.execute(r#"{"task":"first"}"#).await.unwrap();
    tool.execute(r#"{"task":"second"}"#).await.unwrap();

    let usage = tool.usage();
    assert_eq!(usage.prompt_tokens, 30);
    assert_eq!(usage.completion_tokens, 12);
    assert_eq!(usage.total_tokens, 42);
  }

  #[tokio::test]
  async fn an_empty_task_is_rejected_without_calling_the_model() {
    let sub = ScriptedBackend::replying(Vec::new());
    let tool = researcher(&sub);

    let err = tool.execute(r#"{"task":"  "}"#).await.unwrap_err();

    assert!(err.to_string().contains("task is empty"), "got: {err}");
    assert!(sub.seen().is_empty());
  }

  /// The parent's model only sees the outermost message, so the cause has to be in it.
  #[tokio::test]
  async fn a_failing_sub_agent_reports_why() {
    // Well-formed but empty: fails after the retry layer, so it fails exactly once.
    let mut no_choices = answer("unused");
    no_choices["choices"] = json!([]);
    let sub = ScriptedBackend::replying(vec![no_choices]);
    let tool = researcher(&sub);

    let err = tool
      .execute(r#"{"task":"go"}"#)
      .await
      .unwrap_err()
      .to_string();

    assert_eq!(err, "sub-agent `researcher` failed: No choices in response");
  }

  /// Walking the chain is what makes the cause visible; a long one is cut, on a character
  /// boundary, so a quoted response body cannot flood the parent's context.
  #[test]
  fn the_cause_chain_is_shown_but_bounded() {
    let chained = anyhow::anyhow!("boom").context("outer");
    assert_eq!(
      failure_message("worker", &chained),
      "sub-agent `worker` failed: outer: boom"
    );

    let long = anyhow::anyhow!("{}", "é".repeat(MAX_ERROR_CHARS * 5));
    let message = failure_message("worker", &long);
    let prefix = "sub-agent `worker` failed: ";
    assert!(message.ends_with('…'), "marked as cut");
    assert_eq!(message.chars().count(), prefix.len() + MAX_ERROR_CHARS + 1);
  }

  /// A model request that never returns, which notices when it is abandoned.
  struct HangingBackend {
    abandoned: Arc<AtomicBool>,
  }

  struct SetOnDrop(Arc<AtomicBool>);

  impl Drop for SetOnDrop {
    fn drop(&mut self) {
      self.0.store(true, Ordering::SeqCst);
    }
  }

  #[async_trait::async_trait]
  impl ChatBackend for HangingBackend {
    async fn create(&self, _request: ChatRequest) -> anyhow::Result<ChatResponse> {
      let _marker = SetOnDrop(Arc::clone(&self.abandoned));
      std::future::pending::<()>().await;
      unreachable!("pending never resolves")
    }

    async fn create_stream(&self, _request: ChatRequest) -> anyhow::Result<ChatChunkStream> {
      anyhow::bail!("only the non-streaming route is used here")
    }
  }

  /// A sub-agent tool whose model never answers, plus the flag that says its request was
  /// dropped.
  fn hanging_tool() -> (AgentTool, Arc<AtomicBool>) {
    let abandoned = Arc::new(AtomicBool::new(false));
    let backend = HangingBackend {
      abandoned: Arc::clone(&abandoned),
    };
    let agent = Agent::new(
      Provider::with_backend(Arc::new(backend), 1),
      "gpt-test",
      Some("x"),
      Arc::new(ToolRegistry::empty()),
    )
    .with_model_settings(ModelSettings::default());
    (
      AgentTool::new("slow", "Never finishes", agent).unwrap(),
      abandoned,
    )
  }

  /// The module docs promise that giving up on a delegation gives up on its model request.
  #[tokio::test]
  async fn dropping_a_delegation_abandons_the_sub_agents_request() {
    let (tool, abandoned) = hanging_tool();

    let outcome =
      tokio::time::timeout(Duration::from_millis(50), tool.execute(r#"{"task":"go"}"#)).await;

    assert!(outcome.is_err(), "the call should still have been running");
    assert!(
      abandoned.load(Ordering::SeqCst),
      "the sub-agent's request was left running"
    );
  }

  /// The same promise, end to end: the parent's tool timeout is what drops it, and the
  /// parent carries on with a timed-out tool result instead of hanging.
  #[tokio::test]
  async fn a_parent_tool_timeout_abandons_the_sub_agent_and_the_parent_carries_on() {
    let (slow, abandoned) = hanging_tool();
    let parent = ScriptedBackend::replying(vec![
      tool_call("call_1", "slow", json!({ "task": "go" })),
      answer("gave up waiting"),
    ]);
    let mut toolbox = ToolRegistry::empty();
    toolbox.add(Arc::new(slow)).unwrap();
    let agent =
      agent_on(&parent, toolbox, "You lead.").with_tool_timeout(Duration::from_millis(50));

    let result = agent.run("start").await.unwrap();

    assert_eq!(result.output, "gave up waiting");
    assert!(
      abandoned.load(Ordering::SeqCst),
      "the sub-agent kept running"
    );
    assert!(
      messages(&parent.seen()[1])
        .iter()
        .any(|(role, content)| role == "tool" && content.contains("did not finish")),
      "the parent's model was told the call timed out"
    );
  }

  /// Handing a sub-agent tools that change things, with nothing in front of them, is the
  /// way around the parent's sandbox; the constructor names them so it can say so.
  #[test]
  fn tools_with_side_effects_and_no_hook_are_flagged() {
    let backend = ScriptedBackend::replying(Vec::new());
    let agent = |names: &[&str]| {
      let names: Vec<String> = names.iter().map(|name| (*name).to_owned()).collect();
      agent_on(&backend, ToolRegistry::select(&names).unwrap(), "x")
    };

    assert_eq!(
      ungated_side_effect_tools(&agent(&["read_file", "write_file", "run_command"])),
      ["write_file", "run_command"]
    );
    assert!(ungated_side_effect_tools(&agent(&["read_file", "calculator"])).is_empty());
    assert!(ungated_side_effect_tools(&agent(&[])).is_empty());

    let guarded = agent(&["write_file"]).with_before_tool_callback(Arc::new(
      WorkspaceGuardCallback::new(std::env::temp_dir()).unwrap(),
    ));
    assert!(
      ungated_side_effect_tools(&guarded).is_empty(),
      "any before-tool hook counts as a gate"
    );
  }

  #[test]
  fn concurrency_safety_follows_the_sub_agents_tools() {
    let backend = ScriptedBackend::replying(Vec::new());
    let wrap = |names: &[&str]| {
      let names: Vec<String> = names.iter().map(|name| (*name).to_owned()).collect();
      let toolbox = ToolRegistry::select(&names).unwrap();
      AgentTool::new("sub", "a sub-agent", agent_on(&backend, toolbox, "x")).unwrap()
    };

    assert!(wrap(&[]).concurrency_safe(), "no tools, nothing to collide");
    assert!(wrap(&["calculator", "read_file"]).concurrency_safe());
    assert!(
      !wrap(&["read_file", "write_file"]).concurrency_safe(),
      "one tool with side effects makes the delegation take its turn"
    );
  }

  #[test]
  fn names_providers_would_reject_are_refused_up_front() {
    let backend = ScriptedBackend::replying(Vec::new());
    let agent = || agent_on(&backend, ToolRegistry::empty(), "x");

    for bad in ["", "has space", "ünïcode", &"x".repeat(65)] {
      assert!(
        AgentTool::new(bad, "does things", agent()).is_err(),
        "`{bad}` should be refused"
      );
    }
    assert!(AgentTool::new("code-reviewer_2", "does things", agent()).is_ok());
    assert!(
      AgentTool::new("reviewer", "  ", agent()).is_err(),
      "blank description"
    );
  }

  #[test]
  fn the_timeout_is_the_callers_unless_set() {
    let backend = ScriptedBackend::replying(Vec::new());
    let tool = researcher(&backend);
    assert_eq!(tool.timeout(), None);
    let tool = tool.with_timeout(Duration::from_secs(900));
    assert_eq!(tool.timeout(), Some(Duration::from_secs(900)));
  }

  /// Registered like any other tool: the parent advertises it with the task schema.
  #[test]
  fn it_registers_with_the_task_schema() {
    let backend = ScriptedBackend::replying(Vec::new());
    let mut registry = ToolRegistry::empty();
    registry.add(Arc::new(researcher(&backend))).unwrap();

    let definition = serde_json::to_value(&registry.definitions()[0]).unwrap();
    assert_eq!(definition["function"]["name"], "researcher");
    assert_eq!(
      definition["function"]["parameters"]["required"],
      json!(["task"])
    );
  }
}
