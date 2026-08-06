//! Demonstrates [`agent::Agent`]: unlike [`agent::llm::complete::chat_complete`], every
//! step (user input, tool calls, tool results, final answer) is kept as an `Event` in the
//! returned `ExecutionContext`, so it can be inspected after the run finishes.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example agent_run
//! ```

use std::sync::Arc;

use agent::{Agent, config, models::action_plan::ActionPlan, telemetry, tools::ToolRegistry};

const SYSTEM_PROMPT: &str = "You are a general-purpose assistant.";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
  telemetry::init()?;

  let toolbox = Arc::new(ToolRegistry::builtin()?);
  let agent = Agent::new(config::model(), Some(SYSTEM_PROMPT), Arc::clone(&toolbox));

  // Arithmetic: the model is expected to call the calculator tool along the way.
  let result = agent
    .run("What is 5875 multiplied by 467? Then add 100 to the result.")
    .await?;

  tracing::info!("Answer: {}", result.output);
  tracing::info!(
    steps = result.context.current_step,
    events = result.context.events.len(),
    usage = ?result.context.usage,
    budget_exhausted = result.budget_exhausted,
    "run finished"
  );
  for event in &result.context.events {
    tracing::debug!(author = %event.author, content = ?event.content, "event");
  }

  // Structured: the model ends the loop by calling a synthetic `final_answer` tool
  // carrying `ActionPlan`, but can still call the real tools beforehand.
  let structured = agent
    .run_structured::<ActionPlan>(
      "Plan a 3-day trip to Hangzhou, and work out how many hours that is.",
    )
    .await?;

  tracing::info!(plan = ?structured.output, "structured answer");
  tracing::info!(
    steps = structured.context.current_step,
    events = structured.context.events.len(),
    budget_exhausted = structured.budget_exhausted,
    "structured run finished"
  );

  Ok(())
}
