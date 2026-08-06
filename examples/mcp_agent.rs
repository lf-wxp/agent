//! Connects to the `mcp_server` example over stdio and lets the model use its tools
//! alongside the built-in ones.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example mcp_agent
//! ```
//!
//! The server is spawned through `cargo run --example mcp_server`, so no manual build
//! step is needed. Cargo's own progress output goes to stderr, leaving stdout free for
//! the MCP stream.

use agent::{
  config,
  llm::{complete::chat_complete, provider::Provider},
  telemetry,
  tools::{ToolRegistry, mcp::McpConnection},
};
use tokio::process::Command;

const SYSTEM_PROMPT: &str = "You are a general-purpose assistant. Prefer tools over \
                             guessing for dates and arithmetic.";

/// Namespaces the server's tools, e.g. `demo__days_between`.
const SERVER_LABEL: &str = "demo";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
  telemetry::init()?;

  let mut command = Command::new("cargo");
  command.args(["run", "--quiet", "--example", "mcp_server"]);

  let connection = McpConnection::spawn(SERVER_LABEL, command).await?;

  // Built-in and MCP tools land in one registry, so the model sees a single flat tool
  // list and the agent loop treats them identically.
  let mut registry = ToolRegistry::builtin()?;
  registry.extend(connection.tools().await?)?;
  tracing::info!(tools = ?registry, "registry ready");

  // Needs both a remote tool (date arithmetic) and a local one (multiplication).
  let answer = chat_complete(
    Provider::shared(),
    config::model(),
    Some(SYSTEM_PROMPT),
    "How many days are there from 2026-08-06 to 2026-12-25? \
     Then multiply that number of days by 24 to get the hours.",
    &registry,
  )
  .await?;

  tracing::info!("Answer: {answer}");

  // Drop the registry first: the connection cannot shut down while its tools are alive.
  drop(registry);
  connection.shutdown().await?;

  Ok(())
}
