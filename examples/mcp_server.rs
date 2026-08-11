//! A minimal MCP server over **stdio**, used as the counterpart for `examples/mcp_agent.rs`.
//!
//! Meant to be spawned as a child process rather than run by hand. Everything it logs goes
//! to stderr: stdout is the protocol channel, and a stray `println!` there would corrupt
//! the stream.
//!
//! The tools themselves live in `examples/shared/demo_server.rs`, shared with
//! `mcp_http_server.rs` — same handler, different transport.

use rmcp::ServiceExt;
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

#[path = "shared/demo_server.rs"]
mod demo_server;

use demo_server::DemoServer;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
  // stderr only: stdout carries the MCP protocol.
  let subscriber = FmtSubscriber::builder()
    .with_max_level(Level::INFO)
    .with_writer(std::io::stderr)
    .finish();
  tracing::subscriber::set_global_default(subscriber)?;

  tracing::info!("MCP demo server starting on stdio");

  let service = DemoServer::new().serve(rmcp::transport::stdio()).await?;
  // Blocks until the client disconnects or cancels.
  service.waiting().await?;

  Ok(())
}
