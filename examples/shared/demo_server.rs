//! The demo MCP server shared by the transport examples.
//!
//! Not an example target of its own: Cargo only picks up `examples/*.rs` and
//! `examples/*/main.rs`, so this file is compiled solely through the `#[path]` module
//! declarations in `mcp_server.rs` (stdio), `mcp_http_server.rs` and `mcp_http_agent.rs`
//! (Streamable HTTP). Keeping one definition means the transports differ only in how they
//! are served, which is the whole point of the comparison.
//!
//! The two tools are deliberately date arithmetic — something language models are
//! unreliable at, and whose answers are easy to verify.

use chrono::{Datelike, NaiveDate, Utc};
use rmcp::{
  ErrorData, ServerHandler,
  handler::server::{router::tool::ToolRouter, wrapper::Parameters},
  model::{ServerCapabilities, ServerInfo},
  tool, tool_handler, tool_router,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Arguments for `days_between`.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct DaysBetweenRequest {
  /// Start date, `YYYY-MM-DD`.
  pub from: String,
  /// End date, `YYYY-MM-DD`.
  pub to: String,
}

/// Demo server exposing a couple of date tools.
#[derive(Debug, Clone)]
pub struct DemoServer {
  tool_router: ToolRouter<Self>,
}

impl DemoServer {
  pub fn new() -> Self {
    Self {
      tool_router: Self::tool_router(),
    }
  }
}

impl Default for DemoServer {
  fn default() -> Self {
    Self::new()
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
