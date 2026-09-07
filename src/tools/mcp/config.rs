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

use std::{
  collections::{BTreeMap, HashSet},
  path::Path,
  sync::Arc,
};

use anyhow::Context;
use serde::Deserialize;
use tokio::process::Command;

use crate::tools::{
  mcp::client::{self, McpConnection},
  tool::Tool,
};

/// Environment variables a spawned stdio server inherits from this process by default,
/// on top of whatever its own `env` in `mcp.json` declares — see [`stdio_env`].
///
/// Deliberately not "the whole environment": a stdio entry's `command` is, per the
/// module docs' "Trust boundary" section, already treated like a shell command the
/// operator chose to run — but the *environment that command sees* is a separate trust
/// boundary a well-behaved config should not have to think about. Without this
/// allowlist, [`tokio::process::Command`] inherits every variable this agent process
/// itself has (API keys, cloud credentials, ...), regardless of whether the server ever
/// needed them, which turns "one npx-installed MCP server" into "one npx-installed MCP
/// server that can read every secret this agent has". Only what the OS needs to even
/// locate and start the program survives; anything the server itself needs must be
/// listed explicitly in `env`.
#[cfg(unix)]
const INHERITED_ENV_VARS: &[&str] = &["PATH", "HOME", "TMPDIR", "LANG", "LC_ALL"];
#[cfg(windows)]
const INHERITED_ENV_VARS: &[&str] = &[
  "PATH",
  "SystemRoot",
  "SystemDrive",
  "TEMP",
  "TMP",
  "USERPROFILE",
  "APPDATA",
  "LOCALAPPDATA",
  "ProgramData",
  "ComSpec",
  "windir",
];

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

  /// Restrict this server to exactly these tools, named as the server itself advertises
  /// them (i.e. before [`client::local_name`] adds the `label__` prefix) — see
  /// [`apply_allow_list`]. `None` (the default, and the only option before this field
  /// existed) exposes every tool the server advertises, unchanged.
  ///
  /// Exists because a server is discovered, not authored, by this agent: an operator who
  /// otherwise trusts a server (enough to run its `command`, per the module docs) may
  /// still want to hold back one specific tool it happens to expose — e.g. a
  /// `filesystem` server's `write_file` alongside a `read_file` they do want — without
  /// forking or patching the server itself.
  #[serde(default)]
  pub allowed_tools: Option<Vec<String>>,
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
  let tools = apply_allow_list(label, server, tools);
  Ok((connection, tools))
}

/// Keep only the tools named in `server.allowed_tools`, matched by computing the same
/// [`client::local_name`] each one would already have been adapted with; `None` (the
/// field's default) is a no-op, returning `tools` unchanged.
///
/// A configured name that matches nothing `tools` actually contains is logged, not an
/// error: a typo, or a server that changed its tool set since the config was written,
/// should not take the rest of the allowlist — or the server's other, still-valid
/// tools — down with it.
fn apply_allow_list(
  label: &str,
  server: &ServerConfig,
  tools: Vec<Arc<dyn Tool>>,
) -> Vec<Arc<dyn Tool>> {
  let Some(allowed) = &server.allowed_tools else {
    return tools;
  };

  let allowed_names: HashSet<String> = allowed
    .iter()
    .map(|name| client::local_name(label, name))
    .collect();

  for configured in allowed {
    let local = client::local_name(label, configured);
    if !tools.iter().any(|tool| tool.name() == local) {
      tracing::warn!(
        label,
        tool = configured,
        "MCP server `{label}`'s `allowedTools` names a tool it did not advertise"
      );
    }
  }

  tools
    .into_iter()
    .filter(|tool| allowed_names.contains(tool.name()))
    .collect()
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

  // See `INHERITED_ENV_VARS`'s docs: the child does not get this process's full
  // environment by default, only the fixed, minimal allowlist plus whatever the config
  // itself declares.
  command.env_clear();
  for (key, value) in stdio_env(server)? {
    command.env(key, value);
  }

  if let Some(cwd) = &server.cwd {
    command.current_dir(expand(cwd)?);
  }

  McpConnection::spawn(label, command).await
}

/// The environment a spawned stdio server ends up with: [`INHERITED_ENV_VARS`] as found
/// in this process's own environment, overridden/extended by `server.env`. Split out of
/// [`spawn_stdio`] as a pure function — returning the composed map instead of mutating a
/// [`Command`] directly — so a test can check exactly what ends up in it without
/// actually spawning a process.
fn stdio_env(server: &ServerConfig) -> anyhow::Result<BTreeMap<String, String>> {
  let mut env = BTreeMap::new();
  for var in INHERITED_ENV_VARS {
    if let Ok(value) = std::env::var(var) {
      env.insert((*var).to_owned(), value);
    }
  }
  for (key, value) in &server.env {
    env.insert(key.clone(), expand(value)?);
  }
  Ok(env)
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
/// Reusable anywhere else a config file needs the same substitution (e.g. a future
/// deployment-specific config file that should not commit real credentials).
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

  /// Bare-minimum [`Tool`] for [`apply_allow_list`]'s tests: only `name` is ever
  /// inspected there, so nothing else needs to do anything real.
  struct StubTool(String);

  #[async_trait::async_trait]
  impl Tool for StubTool {
    fn name(&self) -> &str {
      &self.0
    }

    fn description(&self) -> &str {
      "stub"
    }

    fn parameters(&self) -> serde_json::Value {
      serde_json::json!({})
    }

    async fn execute(&self, _args_json: &str) -> anyhow::Result<String> {
      unreachable!("apply_allow_list never calls a tool")
    }
  }

  fn stub_tools(names: &[&str]) -> Vec<Arc<dyn Tool>> {
    names
      .iter()
      .map(|name| Arc::new(StubTool((*name).to_owned())) as Arc<dyn Tool>)
      .collect()
  }

  #[test]
  fn allow_list_absent_keeps_every_tool() {
    let server = ServerConfig::default();
    let tools = stub_tools(&["demo__read_file", "demo__write_file"]);
    assert_eq!(apply_allow_list("demo", &server, tools).len(), 2);
  }

  #[test]
  fn allow_list_keeps_only_the_named_remote_tools() {
    let server = ServerConfig {
      allowed_tools: Some(vec!["read_file".to_owned()]),
      ..Default::default()
    };
    let tools = stub_tools(&["demo__read_file", "demo__write_file"]);

    let filtered = apply_allow_list("demo", &server, tools);
    let names: Vec<&str> = filtered.iter().map(|tool| tool.name()).collect();
    assert_eq!(names, ["demo__read_file"]);
  }

  #[test]
  fn allow_list_of_empty_vec_keeps_nothing() {
    let server = ServerConfig {
      allowed_tools: Some(Vec::new()),
      ..Default::default()
    };
    let tools = stub_tools(&["demo__read_file"]);
    assert!(apply_allow_list("demo", &server, tools).is_empty());
  }

  #[test]
  fn allow_list_naming_an_unadvertised_tool_does_not_panic_or_drop_the_rest() {
    let server = ServerConfig {
      allowed_tools: Some(vec!["read_file".to_owned(), "typo_tool".to_owned()]),
      ..Default::default()
    };
    let tools = stub_tools(&["demo__read_file"]);

    let filtered = apply_allow_list("demo", &server, tools);
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].name(), "demo__read_file");
  }

  #[test]
  fn stdio_env_does_not_leak_unrelated_variables() {
    // SAFETY: single-threaded test, and the name is unique to this test.
    unsafe { std::env::set_var("AGENT_TEST_MCP_SECRET", "s3cret") };

    let server = ServerConfig::default();
    let env = stdio_env(&server).unwrap();

    assert!(
      !env.contains_key("AGENT_TEST_MCP_SECRET"),
      "an env var outside INHERITED_ENV_VARS must not be passed to the child process"
    );
  }

  #[test]
  fn stdio_env_carries_only_the_inherited_allowlist_by_default() {
    let server = ServerConfig::default();
    let env = stdio_env(&server).unwrap();

    for key in env.keys() {
      assert!(
        INHERITED_ENV_VARS.contains(&key.as_str()),
        "`{key}` is not in INHERITED_ENV_VARS and `server.env` is empty here"
      );
    }
  }

  #[test]
  fn stdio_env_lets_configured_vars_override_the_inherited_ones() {
    // No need to touch the real `PATH` here: whatever it already is, `server.env`
    // setting the same key must still win, since `stdio_env` applies it after the
    // inherited-allowlist loop.
    let server = ServerConfig {
      env: BTreeMap::from([("PATH".to_owned(), "/configured/path".to_owned())]),
      ..Default::default()
    };
    let env = stdio_env(&server).unwrap();

    assert_eq!(
      env.get("PATH").map(String::as_str),
      Some("/configured/path")
    );
  }

  #[test]
  fn stdio_env_expands_configured_values() {
    // SAFETY: single-threaded test, and the name is unique to this test.
    unsafe { std::env::set_var("AGENT_TEST_MCP_EXPAND_SOURCE", "expanded") };

    let server = ServerConfig {
      env: BTreeMap::from([(
        "TARGET".to_owned(),
        "${AGENT_TEST_MCP_EXPAND_SOURCE}".to_owned(),
      )]),
      ..Default::default()
    };
    let env = stdio_env(&server).unwrap();

    assert_eq!(env.get("TARGET").map(String::as_str), Some("expanded"));
  }
}
