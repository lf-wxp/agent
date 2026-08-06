//! Loads MCP servers from an `mcp.json` and lets the model use them next to the built-in
//! tools.
//!
//! ```sh
//! cp mcp.example.json mcp.json
//! cargo run --example mcp_config
//!
//! # or point at a specific file
//! cargo run --example mcp_config -- path/to/mcp.json
//! ```
//!
//! Without a config file this falls back to the built-in tools, which is the same thing
//! that happens in a fresh checkout.

use agent::{
  config,
  llm::{complete::chat_complete, provider::Provider},
  telemetry,
  tools::ToolRegistry,
};

const SYSTEM_PROMPT: &str = "You are a general-purpose assistant. Prefer tools over \
                             guessing for dates and arithmetic.";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
  telemetry::init()?;

  let path = std::env::args()
    .nth(1)
    .map_or_else(config::mcp_config_path, Into::into);

  let (registry, connections) = ToolRegistry::with_mcp(&path).await?;
  tracing::info!(?registry, servers = connections.len(), "registry ready");

  let answer = chat_complete(
    Provider::shared(),
    config::model(),
    Some(SYSTEM_PROMPT),
    "How many days are there from 2026-08-06 to 2026-12-25?",
    &registry,
  )
  .await?;

  tracing::info!("Answer: {answer}");

  // Drop the registry first: a connection cannot shut down while its tools are alive.
  drop(registry);
  for connection in connections {
    connection.shutdown().await?;
  }

  Ok(())
}
