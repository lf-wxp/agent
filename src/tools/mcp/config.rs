//! `mcp.json` parsing.
//!
//! There is no official schema for this file: the MCP specification covers the protocol,
//! not client configuration. The shape below is the de facto convention established by
//! Claude Desktop's `claude_desktop_config.json` and followed, with variations, by Cursor,
//! VS Code and others. This parser is deliberately permissive:
//!
//! - accepts both `mcpServers` (Claude Desktop, Cursor) and `servers` (VS Code);
//! - infers the transport from `command` / `url` when `type` is absent;
//! - ignores unknown keys (`description`, `timeout`, `autoApprove`, ...), since every
//!   client adds its own.
//!
//! ```json
//! {
//!   "mcpServers": {
//!     "filesystem": {
//!       "command": "npx",
//!       "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
//!       "env": { "LOG_LEVEL": "info" }
//!     },
//!     "remote": {
//!       "url": "https://example.com/mcp",
//!       "headers": { "Authorization": "Bearer ${env:MCP_TOKEN}" }
//!     },
//!     "retired": { "command": "old-server", "disabled": true }
//!   }
//! }
//! ```
//!
//! ## Trust boundary
//!
//! A stdio entry's `command` / `args` / `env` / `cwd` are spawned as a child process
//! verbatim (see [`spawn_stdio`]), and [`crate::config::mcp_config_path`] (env
//! `MCP_CONFIG_PATH`) lets the file location itself be overridden. Both are meant to be
//! set by whoever operates the agent, not by untrusted end users: treat `mcp.json` and
//! `MCP_CONFIG_PATH` the same way you would a shell command — anything that can control
//! their contents can run arbitrary processes on this host.

use std::{collections::BTreeMap, path::Path, sync::Arc};

use anyhow::Context;
use serde::Deserialize;
use tokio::process::Command;

use crate::tools::{mcp::client::McpConnection, tool::Tool};

/// A parsed `mcp.json`.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpConfig {
  /// `mcpServers` is the common spelling; `servers` is what VS Code uses.
  #[serde(alias = "servers", default)]
  pub mcp_servers: BTreeMap<String, ServerConfig>,
}

/// How to reach one server.
///
/// Fields from both transports live together because that is how the format works: a
/// `command` entry means stdio, a `url` entry means HTTP.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ServerConfig {
  /// Explicit transport. Inferred from the other fields when absent.
  #[serde(default, rename = "type", alias = "transport")]
  pub kind: Option<TransportKind>,

  /// Executable to spawn (stdio).
  #[serde(default)]
  pub command: Option<String>,
  #[serde(default)]
  pub args: Vec<String>,
  #[serde(default)]
  pub env: BTreeMap<String, String>,
  /// Working directory for the spawned process.
  #[serde(default)]
  pub cwd: Option<String>,

  /// Endpoint to connect to (HTTP).
  #[serde(default)]
  pub url: Option<String>,
  #[serde(default)]
  pub headers: BTreeMap<String, String>,

  /// Skip this server. Both spellings are in the wild.
  #[serde(default)]
  pub disabled: bool,
  #[serde(default)]
  pub enabled: Option<bool>,
}

/// Transports named in the wild. Values are matched case-insensitively.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TransportKind {
  Stdio,
  Http,
  #[serde(
    alias = "streamable-http",
    alias = "streamablehttp",
    alias = "streamableHttp",
    alias = "streamable_http"
  )]
  StreamableHttp,
  Sse,
}

impl ServerConfig {
  /// Whether this entry should be started.
  ///
  /// `enabled: false` and `disabled: true` mean the same thing; `enabled` wins when both
  /// are present, since it is the more explicit of the two.
  pub fn is_enabled(&self) -> bool {
    self.enabled.unwrap_or(!self.disabled)
  }

  /// Resolve the transport, inferring it when `type` is absent.
  fn transport(&self) -> anyhow::Result<TransportKind> {
    if let Some(kind) = self.kind {
      return Ok(kind);
    }

    match (self.command.is_some(), self.url.is_some()) {
      (true, false) => Ok(TransportKind::Stdio),
      (false, true) => Ok(TransportKind::StreamableHttp),
      (true, true) => {
        anyhow::bail!("both `command` and `url` are set; add `type` to disambiguate")
      }
      (false, false) => anyhow::bail!("needs either `command` (stdio) or `url` (http)"),
    }
  }
}

impl McpConfig {
  /// Parse from JSON text.
  pub fn from_json(text: &str) -> anyhow::Result<Self> {
    serde_json::from_str(text).context("failed to parse mcp.json")
  }

  /// Read and parse a config file.
  pub async fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
    let path = path.as_ref();
    let text = tokio::fs::read_to_string(path)
      .await
      .with_context(|| format!("failed to read MCP config `{}`", path.display()))?;

    Self::from_json(&text)
  }

  /// Servers that are not disabled.
  pub fn enabled(&self) -> impl Iterator<Item = (&String, &ServerConfig)> {
    self
      .mcp_servers
      .iter()
      .filter(|(_, server)| server.is_enabled())
  }

  /// Connect every enabled server and collect their tools.
  ///
  /// A server that fails to start is logged and skipped rather than failing the whole
  /// batch: one broken entry in `mcp.json` should not take down every other tool.
  pub async fn connect_all(&self) -> (Vec<McpConnection>, Vec<Arc<dyn Tool>>) {
    let mut connections = Vec::new();
    let mut tools = Vec::new();

    for (label, server) in self.enabled() {
      match connect_one(label, server).await {
        Ok((connection, discovered)) => {
          connections.push(connection);
          tools.extend(discovered);
        }
        Err(err) => tracing::warn!("skipping MCP server `{label}`: {err:#}"),
      }
    }

    (connections, tools)
  }
}

/// Start one server and list its tools.
async fn connect_one(
  label: &str,
  server: &ServerConfig,
) -> anyhow::Result<(McpConnection, Vec<Arc<dyn Tool>>)> {
  let connection = match server.transport()? {
    TransportKind::Stdio => spawn_stdio(label, server).await?,
    TransportKind::Http | TransportKind::StreamableHttp => connect_http(label, server).await?,
    // rmcp supports it, but the feature is not enabled and the transport is deprecated in
    // favour of Streamable HTTP.
    TransportKind::Sse => anyhow::bail!("the `sse` transport is not supported; use `http`"),
  };

  let tools = connection.tools().await?;
  Ok((connection, tools))
}

async fn spawn_stdio(label: &str, server: &ServerConfig) -> anyhow::Result<McpConnection> {
  let program = server
    .command
    .as_deref()
    .context("stdio server needs `command`")?;

  let mut command = Command::new(expand(program)?);
  for arg in &server.args {
    command.arg(expand(arg)?);
  }
  for (key, value) in &server.env {
    command.env(key, expand(value)?);
  }
  if let Some(cwd) = &server.cwd {
    command.current_dir(expand(cwd)?);
  }

  McpConnection::spawn(label, command).await
}

async fn connect_http(label: &str, server: &ServerConfig) -> anyhow::Result<McpConnection> {
  let url = server.url.as_deref().context("http server needs `url`")?;

  // Names are expanded too: some deployments key the header itself off the environment.
  let mut headers = BTreeMap::new();
  for (name, value) in &server.headers {
    headers.insert(expand(name)?, expand(value)?);
  }

  McpConnection::connect(label, &expand(url)?, &headers).await
}

/// Substitute `${VAR}` and `${env:VAR}` from the environment.
///
/// Keeps credentials out of the config file, which is usually committed. An undefined
/// variable is an error rather than being left as-is: passing a literal `${TOKEN}` to a
/// server produces a far more confusing failure downstream.
///
/// Shared with [`crate::api::tenant`], which needs the same substitution for its own
/// (also usually-committed) tenant config file.
pub(crate) fn expand(text: &str) -> anyhow::Result<String> {
  let mut out = String::with_capacity(text.len());
  let mut rest = text;

  while let Some(start) = rest.find("${") {
    out.push_str(&rest[..start]);
    let after = &rest[start + 2..];
    let end = after
      .find('}')
      .with_context(|| format!("unterminated `${{` in `{text}`"))?;

    let name = after[..end].trim();
    let name = name.strip_prefix("env:").unwrap_or(name);
    let value = std::env::var(name)
      .with_context(|| format!("`{text}` references undefined environment variable `{name}`"))?;

    out.push_str(&value);
    rest = &after[end + 1..];
  }

  out.push_str(rest);
  Ok(out)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn parses_claude_desktop_shape() {
    let config = McpConfig::from_json(
      r#"{
        "mcpServers": {
          "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
          }
        }
      }"#,
    )
    .unwrap();

    let server = &config.mcp_servers["filesystem"];
    assert_eq!(server.command.as_deref(), Some("npx"));
    assert_eq!(server.args.len(), 3);
    assert_eq!(server.transport().unwrap(), TransportKind::Stdio);
  }

  #[test]
  fn parses_vscode_servers_key_and_explicit_type() {
    let config =
      McpConfig::from_json(r#"{"servers": {"demo": {"type": "stdio", "command": "server"}}}"#)
        .unwrap();

    assert_eq!(
      config.mcp_servers["demo"].transport().unwrap(),
      TransportKind::Stdio
    );
  }

  #[test]
  fn infers_http_from_url() {
    let config =
      McpConfig::from_json(r#"{"mcpServers": {"remote": {"url": "https://x/mcp"}}}"#).unwrap();

    assert_eq!(
      config.mcp_servers["remote"].transport().unwrap(),
      TransportKind::StreamableHttp
    );
  }

  #[test]
  fn accepts_streamable_http_spellings() {
    for spelling in [
      "http",
      "streamable-http",
      "streamableHttp",
      "streamable_http",
    ] {
      let json = format!(r#"{{"mcpServers":{{"r":{{"type":"{spelling}","url":"https://x"}}}}}}"#);
      let config = McpConfig::from_json(&json).unwrap_or_else(|e| panic!("{spelling}: {e}"));
      assert!(config.mcp_servers["r"].transport().is_ok(), "{spelling}");
    }
  }

  #[test]
  fn ignores_client_specific_keys() {
    // Other clients add their own fields; they must not break parsing.
    let config = McpConfig::from_json(
      r#"{"mcpServers": {"x": {
        "command": "s", "description": "d", "timeout": 60, "autoApprove": ["a"]
      }}}"#,
    )
    .unwrap();

    assert!(config.mcp_servers.contains_key("x"));
  }

  #[test]
  fn rejects_entry_without_command_or_url() {
    let config = McpConfig::from_json(r#"{"mcpServers": {"x": {}}}"#).unwrap();
    assert!(config.mcp_servers["x"].transport().is_err());
  }

  #[test]
  fn rejects_ambiguous_entry() {
    let config =
      McpConfig::from_json(r#"{"mcpServers":{"x":{"command":"c","url":"https://u"}}}"#).unwrap();
    assert!(config.mcp_servers["x"].transport().is_err());
  }

  #[test]
  fn honours_both_disable_spellings() {
    let config = McpConfig::from_json(
      r#"{"mcpServers": {
        "a": {"command": "s"},
        "b": {"command": "s", "disabled": true},
        "c": {"command": "s", "enabled": false},
        "d": {"command": "s", "enabled": true, "disabled": true}
      }}"#,
    )
    .unwrap();

    let enabled: Vec<&str> = config.enabled().map(|(name, _)| name.as_str()).collect();
    assert_eq!(enabled, ["a", "d"]);
  }

  #[test]
  fn defaults_to_no_servers() {
    assert!(McpConfig::from_json("{}").unwrap().mcp_servers.is_empty());
  }

  #[test]
  fn expands_environment_variables() {
    // SAFETY: single-threaded test, and the name is unique to this test.
    unsafe { std::env::set_var("AGENT_TEST_MCP_TOKEN", "s3cret") };

    assert_eq!(
      expand("Bearer ${AGENT_TEST_MCP_TOKEN}").unwrap(),
      "Bearer s3cret"
    );
    assert_eq!(expand("${env:AGENT_TEST_MCP_TOKEN}").unwrap(), "s3cret");
    assert_eq!(
      expand("a${AGENT_TEST_MCP_TOKEN}b${env:AGENT_TEST_MCP_TOKEN}").unwrap(),
      "as3cretbs3cret"
    );
  }

  #[test]
  fn leaves_plain_text_untouched() {
    assert_eq!(expand("npx").unwrap(), "npx");
    assert_eq!(expand("").unwrap(), "");
  }

  #[test]
  fn rejects_undefined_and_unterminated_variables() {
    assert!(expand("${AGENT_TEST_DEFINITELY_UNSET}").is_err());
    assert!(expand("${oops").is_err());
  }
}
