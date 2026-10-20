//! Demonstrates [`AgentTool`] (a sub-agent) and the `update_plan` tool together.
//!
//! The lead agent plans its work with `update_plan` and hands codebase exploration to a
//! read-only "explorer" sub-agent. The explorer's file reads and searches happen in its
//! own context; the lead only ever sees the explorer's summaries, so its own context
//! stays small however much the explorer had to read.
//!
//! Run with (from the repository root, so the explorer has something to look at):
//!
//! ```sh
//! cargo run --example sub_agent
//! ```

use std::{sync::Arc, time::Duration};

use agent::{
  Agent, AgentTool,
  callback::path_guard::WorkspaceGuardCallback,
  config,
  llm::provider::Provider,
  telemetry,
  tools::{ToolRegistry, file_list, file_read, grep, plan},
};

const LEAD_PROMPT: &str = "You are a lead engineer. Plan multi-step work with update_plan \
  and keep the plan current. Delegate reading code to the explorer: give it a precise, \
  self-contained question each time, then build your answer from what it reports.";

const EXPLORER_PROMPT: &str = "You explore a codebase to answer one question. Use \
  list_files, grep and read_file, then answer concisely, citing file paths.";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
  telemetry::init()?;

  // Read-only on purpose: with no tool that changes anything, the explorer needs no
  // approval step, and the lead may run several explorations at once. Read-only is not
  // the same as harmless, though: the explorer would otherwise read any path the model
  // names (`~/.ssh`, `/etc`), so it is confined to the directory it was started in. The
  // lead's own hooks do not reach it — each agent is gated by what it is given.
  let explorer_tools = ToolRegistry::select(&[
    file_list::NAME.to_owned(),
    grep::NAME.to_owned(),
    file_read::NAME.to_owned(),
  ])?;
  let explorer = Agent::new(
    Provider::shared().clone(),
    config::model(),
    Some(EXPLORER_PROMPT),
    Arc::new(explorer_tools),
  )
  .with_before_tool_callback(Arc::new(WorkspaceGuardCallback::new(
    std::env::current_dir()?,
  )?));
  // Kept as an `Arc` so its token usage can be read after the run.
  let explorer = Arc::new(
    AgentTool::new(
      "explorer",
      "Answers one question about the code in the current directory by reading it; \
       returns a short answer citing file paths.",
      explorer,
    )?
    // A delegation can take several rounds; give it more than the default tool timeout.
    .with_timeout(Duration::from_secs(600)),
  );

  let mut lead_tools = ToolRegistry::select(&[plan::NAME.to_owned()])?;
  lead_tools.add(Arc::clone(&explorer) as Arc<dyn agent::tools::Tool>)?;
  let lead = Agent::new(
    Provider::shared().clone(),
    config::model(),
    Some(LEAD_PROMPT),
    Arc::new(lead_tools),
  );

  let result = lead
    .run(
      "Which built-in tools does this project ship, and which of them need human \
       approval in the CLI by default?",
    )
    .await?;

  println!("{}", result.output);
  tracing::info!(
    lead_usage = ?result.context.usage,
    explorer_usage = ?explorer.usage(),
    steps = result.context.current_step,
    budget_exhausted = result.budget_exhausted,
    "run finished"
  );
  Ok(())
}
