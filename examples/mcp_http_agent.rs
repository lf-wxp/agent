//! Same as `examples/mcp_agent.rs`, but the MCP server is reached over **Streamable HTTP**
//! instead of stdio — the transport you need for a server you do not launch yourself.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example mcp_http_agent
//! ```
//!
//! Self-contained: the demo server is served in-process on an ephemeral loopback port, so
//! there is no second terminal, no fixed port to collide on, and no external dependency.
//! Point `McpConnection::connect` at a real URL to talk to someone else's server instead;
//! `examples/mcp_http_server.rs` runs the same handler as a standalone process.
//!
//! The connection also carries an `Authorization` header, which the in-process server
//! verifies. Note the value is written out in full: headers are forwarded verbatim, so the
//! `Bearer ` prefix belongs in the value and must not be added twice.

use std::{collections::BTreeMap, sync::Arc};

use agent::{
  Agent, config,
  llm::provider::Provider,
  telemetry,
  tools::{ToolRegistry, mcp::McpConnection},
};

#[path = "shared/demo_http.rs"]
mod demo_http;
#[path = "shared/demo_server.rs"]
mod demo_server;

const SYSTEM_PROMPT: &str = "You are a general-purpose assistant. Prefer tools over \
                             guessing for dates and arithmetic.";

/// Namespaces the server's tools, e.g. `demo__days_between`.
const SERVER_LABEL: &str = "demo";

/// Only ever leaves the loopback interface, so a literal is fine here; a real deployment
/// reads its credentials from the environment.
const DEMO_TOKEN: &str = "demo-token";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
  telemetry::init()?;

  // Port 0: the OS picks a free port, so concurrent runs never fight over one.
  let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
  let (bound, server) = demo_http::serve("127.0.0.1:0", Some(DEMO_TOKEN.to_owned()), async {
    // A dropped sender means the example is unwinding; shut down rather than hang.
    let _ = shutdown_rx.await;
  })
  .await?;

  let url = format!("http://{bound}{}", demo_http::MCP_PATH);
  tracing::info!(%url, "in-process MCP server listening");

  let mut headers = BTreeMap::new();
  headers.insert("Authorization".to_owned(), format!("Bearer {DEMO_TOKEN}"));

  let connection = McpConnection::connect(SERVER_LABEL, &url, &headers).await?;

  // Registration needs `&mut`, so it happens while the registry is still uniquely owned;
  // only the finished registry is shared through `Arc`.
  let mut registry = ToolRegistry::builtin()?;
  registry.extend(connection.tools().await?)?;
  let toolbox = Arc::new(registry);
  tracing::info!(tools = ?toolbox, "registry ready");

  let agent = Agent::new(
    Provider::shared().clone(),
    config::model(),
    Some(SYSTEM_PROMPT),
    Arc::clone(&toolbox),
  );

  // Needs both a remote tool (date arithmetic) and a local one (multiplication).
  let answer = agent
    .run(
      "How many days are there from 2026-08-06 to 2026-12-25? \
       Then multiply that number of days by 24 to get the hours.",
    )
    .await?;

  tracing::info!(
    budget_exhausted = answer.budget_exhausted,
    "Answer: {}",
    answer.output
  );

  // Drop every holder of the MCP tools first: `shutdown` needs the last reference to the
  // server, and the agent keeps a handle on the registry of its own.
  drop(agent);
  drop(toolbox);
  connection.shutdown().await?;

  // Only now is it safe to stop the listener: the DELETE that ends the MCP session goes
  // over the same HTTP server.
  let _ = shutdown_tx.send(());
  server.await?;

  Ok(())
}
