//! Demonstrates [`Agent::run_stream`]: same tool-calling loop and event recording as
//! [`Agent::run`] (see the `agent_run` example), but assistant text is printed as soon as
//! the model emits it, and tool calls run mid-stream.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example agent_stream
//! ```

use std::sync::Arc;

use agent::{
  Agent, AgentStreamEvent, config, llm::provider::Provider, telemetry, tools::ToolRegistry,
};
use futures::StreamExt;

const SYSTEM_PROMPT: &str = "You are a general-purpose assistant.";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
  telemetry::init()?;

  let toolbox = Arc::new(ToolRegistry::builtin()?);
  let agent = Agent::new(
    Provider::shared().clone(),
    config::model(),
    Some(SYSTEM_PROMPT),
    toolbox,
  );

  let stream = agent.run_stream("What is 5875 multiplied by 467? Then add 100 to the result.");
  futures::pin_mut!(stream);

  while let Some(event) = stream.next().await {
    match event? {
      // Printed without a newline: chunks are meant to be concatenated as they arrive.
      AgentStreamEvent::Token(text) => print!("{text}"),
      AgentStreamEvent::ToolCallsStarted(calls) => {
        for call in &calls {
          if let agent::agent::ContentItem::ToolCall {
            name, arguments, ..
          } = call
          {
            println!("\n[calling {name} with {arguments}]");
          }
        }
      }
      AgentStreamEvent::ToolCallsFinished(results) => {
        for result in &results {
          if let agent::agent::ContentItem::ToolResult { name, status, .. } = result {
            println!("[{name} finished: {status:?}]");
          }
        }
      }
      AgentStreamEvent::Done {
        context,
        budget_exhausted,
        ..
      } => {
        println!();
        tracing::info!(
          steps = context.current_step,
          events = context.events.len(),
          usage = ?context.usage,
          budget_exhausted,
          "run finished"
        );
      }
    }
  }

  Ok(())
}
