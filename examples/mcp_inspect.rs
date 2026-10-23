//! Inspect any MCP server: connect, list its tools, and report anything that would not
//! survive the trip to the chat completions API.
//!
//! ```sh
//! # stdio: spawn the server as a child process
//! cargo run --example mcp_inspect -- npx -y @modelcontextprotocol/server-everything
//!
//! # Streamable HTTP: connect to a remote server
//! cargo run --example mcp_inspect -- --url http://localhost:3001/mcp
//! ```
//!
//! Useful before wiring a third-party server into the agent: the MCP tool schema is more
//! permissive than what `function.parameters` accepts, so incompatibilities are worth
//! finding here rather than mid-conversation.
//!
//! Set `MCP_BEARER_TOKEN` for HTTP servers that require authorization, or
//! `MCP_HEADERS` as `Name: Value` pairs separated by newlines or semicolons.

use std::collections::BTreeMap;

use agent::{
  telemetry,
  tools::{ToolRegistry, mcp::McpConnection},
};
use anyhow::Context;
use serde_json::Value;
use tokio::process::Command;

/// Environment variable holding the bearer token for HTTP servers.
const ENV_BEARER_TOKEN: &str = "MCP_BEARER_TOKEN";

/// Environment variable holding extra headers, as `Name: Value` pairs.
const ENV_HEADERS: &str = "MCP_HEADERS";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
  let _telemetry = telemetry::init()?;

  let connection = connect().await?;

  let mut registry = ToolRegistry::empty();
  registry.extend(connection.tools().await?)?;

  println!("\ndiscovered {} tool(s)\n", registry.len());
  for definition in registry.definitions() {
    println!("  - {}", definition.name);
    if !definition.description.is_empty() {
      println!("    {}", first_line(&definition.description));
    }
    for warning in warnings(&definition.parameters) {
      println!("    ! {warning}");
    }
  }
  println!();

  drop(registry);
  connection.shutdown().await?;

  Ok(())
}

/// Build a connection from the command line: `--url <uri>` for HTTP, otherwise a command
/// to spawn over stdio.
async fn connect() -> anyhow::Result<McpConnection> {
  let mut args = std::env::args().skip(1);
  let first = args
    .next()
    .context("usage: mcp_inspect (--url <uri> | <command> [args...])")?;

  if first == "--url" {
    let url = args.next().context("--url needs a URI")?;
    return Ok(McpConnection::connect("probe", &url, &headers()).await?);
  }

  let mut command = Command::new(first);
  command.args(args);
  Ok(McpConnection::spawn("probe", command).await?)
}

/// Report traits that make a tool's schema risky to advertise as an OpenAI function.
///
/// Names need no check here: the client sanitizes them to what the API accepts, and the
/// registry rejects anything else when the tool is added.
fn warnings(parameters: &Value) -> Vec<String> {
  let mut warnings = Vec::new();

  if !parameters.is_object() {
    warnings.push("no parameter schema".to_owned());
    return warnings;
  }

  // `$ref` / `$defs` are legal JSON Schema and common in MCP, but several
  // OpenAI-compatible endpoints reject them inside `function.parameters`.
  if contains_key(parameters, "$ref") {
    warnings.push("schema uses `$ref`, which some endpoints reject".to_owned());
  }
  if contains_key(parameters, "$defs") || contains_key(parameters, "definitions") {
    warnings.push("schema uses `$defs`, which some endpoints reject".to_owned());
  }
  for keyword in ["oneOf", "anyOf", "allOf"] {
    if contains_key(parameters, keyword) {
      warnings.push(format!(
        "schema uses `{keyword}`, support varies by endpoint"
      ));
    }
  }

  warnings
}

/// Whether `key` appears anywhere in the schema.
fn contains_key(value: &Value, key: &str) -> bool {
  match value {
    Value::Object(map) => {
      map.contains_key(key) || map.values().any(|value| contains_key(value, key))
    }
    Value::Array(items) => items.iter().any(|item| contains_key(item, key)),
    _ => false,
  }
}

/// Collect headers from the environment.
fn headers() -> BTreeMap<String, String> {
  let mut headers = BTreeMap::new();

  if let Ok(token) = std::env::var(ENV_BEARER_TOKEN) {
    headers.insert("Authorization".to_owned(), format!("Bearer {token}"));
  }

  for pair in std::env::var(ENV_HEADERS)
    .unwrap_or_default()
    .split(['\n', ';'])
  {
    if let Some((name, value)) = pair.split_once(':') {
      headers.insert(name.trim().to_owned(), value.trim().to_owned());
    }
  }

  headers
}

fn first_line(text: &str) -> &str {
  text.lines().next().unwrap_or(text).trim()
}
