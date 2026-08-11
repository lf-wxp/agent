//! The demo MCP server over **Streamable HTTP**, the counterpart for
//! `examples/mcp_http_agent.rs` when you want client and server in separate processes.
//!
//! ```sh
//! # default: http://127.0.0.1:3001/mcp, no authentication
//! cargo run --example mcp_http_server
//!
//! # custom address, and require a token on every request
//! MCP_BEARER_TOKEN=s3cret cargo run --example mcp_http_server -- 127.0.0.1:4000
//! ```
//!
//! With it running, the HTTP paths of the other examples become exercisable:
//!
//! ```sh
//! cargo run --example mcp_inspect -- --url http://127.0.0.1:3001/mcp
//! # or flip `remote.disabled` to false in mcp.json, then:
//! cargo run --example mcp_config
//! ```
//!
//! Unlike the stdio sibling, stdout carries no protocol traffic here, so logging is free to
//! use it.

use agent::telemetry;

#[path = "shared/demo_http.rs"]
mod demo_http;
#[path = "shared/demo_server.rs"]
mod demo_server;

/// Loopback by default: the transport rejects non-loopback `Host` headers unless
/// `allowed_hosts` is widened, and an unauthenticated tool server has no business
/// listening on a public interface.
const DEFAULT_ADDR: &str = "127.0.0.1:3001";

/// Optional token clients must present; shared with `mcp_inspect`'s spelling.
const ENV_BEARER_TOKEN: &str = "MCP_BEARER_TOKEN";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
  telemetry::init()?;

  let addr = std::env::args()
    .nth(1)
    .unwrap_or_else(|| DEFAULT_ADDR.to_owned());
  let token = std::env::var(ENV_BEARER_TOKEN).ok();

  // `std::future::pending()`: nothing here should trigger a shutdown, so the server runs
  // until interrupted.
  let (bound, handle) = demo_http::serve(&addr, token.clone(), std::future::pending()).await?;

  tracing::info!(
    url = %format!("http://{bound}{}", demo_http::MCP_PATH),
    authenticated = token.is_some(),
    "MCP demo server listening"
  );

  handle.await?;

  Ok(())
}
