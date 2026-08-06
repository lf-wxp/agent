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
use async_openai::types::chat::ChatCompletionTools;
use serde_json::Value;
use tokio::process::Command;

/// The chat completions API rejects function names longer than this.
const MAX_FUNCTION_NAME_CHARS: usize = 64;

/// Environment variable holding the bearer token for HTTP servers.
const ENV_BEARER_TOKEN: &str = "MCP_BEARER_TOKEN";

/// Environment variable holding extra headers, as `Name: Value` pairs.
const ENV_HEADERS: &str = "MCP_HEADERS";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
  telemetry::init()?;

  let connection = connect().await?;

  let mut registry = ToolRegistry::empty();
  registry.extend(connection.tools().await?)?;

  println!("\ndiscovered {} tool(s)\n", registry.len());
  for definition in registry.definitions() {
    let ChatCompletionTools::Function(tool) = definition else {
      println!("  (non-function tool, skipped)");
      continue;
    };

    println!("  - {}", tool.function.name);
    if let Some(description) = &tool.function.description {
      println!("    {}", first_line(description));
    }
    for warning in warnings(&tool.function.name, tool.function.parameters.as_ref()) {
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
    return McpConnection::connect("probe", &url, &headers()).await;
  }

  let mut command = Command::new(first);
  command.args(args);
  McpConnection::spawn("probe", command).await
}

/// Report traits that make a tool risky to advertise as an OpenAI function.
fn warnings(name: &str, parameters: Option<&Value>) -> Vec<String> {
  let mut warnings = Vec::new();

  let length = name.chars().count();
  if length > MAX_FUNCTION_NAME_CHARS {
    warnings.push(format!(
      "name is {length} chars, over the {MAX_FUNCTION_NAME_CHARS} char limit"
    ));
  }

  let Some(parameters) = parameters else {
    warnings.push("no parameter schema".to_owned());
    return warnings;
  };

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
