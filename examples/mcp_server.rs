//! A minimal MCP server, used as the counterpart for `examples/mcp_agent.rs`.
//!
//! Speaks MCP over stdio, so it is meant to be spawned as a child process rather than run
//! by hand. Everything it logs goes to stderr: stdout is the protocol channel, and a
//! stray `println!` there would corrupt the stream.
//!
//! The two tools are deliberately date arithmetic — something language models are
//! unreliable at, and whose answers are easy to verify.

use chrono::{Datelike, NaiveDate, Utc};
use rmcp::{
  ErrorData, ServerHandler, ServiceExt,
  handler::server::{router::tool::ToolRouter, wrapper::Parameters},
  model::{ServerCapabilities, ServerInfo},
  tool, tool_handler, tool_router,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

/// Arguments for `days_between`.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct DaysBetweenRequest {
  /// Start date, `YYYY-MM-DD`.
  from: String,
  /// End date, `YYYY-MM-DD`.
  to: String,
}

/// Demo server exposing a couple of date tools.
#[derive(Debug, Clone)]
struct DemoServer {
  tool_router: ToolRouter<Self>,
}

impl DemoServer {
  fn new() -> Self {
    Self {
      tool_router: Self::tool_router(),
    }
  }
}

#[tool_router(router = tool_router)]
impl DemoServer {
  /// Current UTC date and time.
  #[tool(
    name = "current_time",
    description = "Get the current date and time in UTC, as an RFC 3339 timestamp."
  )]
  async fn current_time(&self) -> String {
    Utc::now().to_rfc3339()
  }

  /// Whole days between two dates.
  #[tool(
    name = "days_between",
    description = "Count the whole days between two dates given as YYYY-MM-DD. \
                   Negative when the end date precedes the start date."
  )]
  async fn days_between(
    &self,
    Parameters(request): Parameters<DaysBetweenRequest>,
  ) -> Result<String, ErrorData> {
    let from = parse_date(&request.from)?;
    let to = parse_date(&request.to)?;

    let days = (to - from).num_days();
    Ok(format!(
      "{days} days from {} to {} (weekday {} to {})",
      from,
      to,
      from.weekday(),
      to.weekday()
    ))
  }
}

/// Parse a date, reporting a message the model can act on.
fn parse_date(value: &str) -> Result<NaiveDate, ErrorData> {
  NaiveDate::parse_from_str(value.trim(), "%Y-%m-%d").map_err(|err| {
    ErrorData::invalid_params(format!("`{value}` is not a YYYY-MM-DD date: {err}"), None)
  })
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for DemoServer {
  fn get_info(&self) -> ServerInfo {
    // Advertising the tools capability is what makes the client call `tools/list`.
    ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
  }
}

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
