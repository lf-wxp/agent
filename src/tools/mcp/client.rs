//! MCP client: connect to a server and adapt its tools to the local [`Tool`] trait.
//!
//! Two transports are supported: stdio for servers launched as a child process, and
//! Streamable HTTP for remote ones.

use std::{
  collections::{BTreeMap, HashMap},
  sync::Arc,
  sync::LazyLock,
  time::Duration,
};

use anyhow::Context;
use reqwest::header::{HeaderName, HeaderValue};
use rmcp::{
  RoleClient, ServiceExt,
  model::{CallToolRequestParams, ContentBlock, JsonObject},
  service::RunningService,
  transport::{
    StreamableHttpClientTransport, TokioChildProcess,
    streamable_http_client::StreamableHttpClientTransportConfig,
  },
};
use serde_json::Value;
use tokio::process::Command;

use crate::tools::tool::Tool;

/// Separates the server label from the remote tool name in the local name.
const LABEL_SEPARATOR: &str = "__";

/// The chat completions API rejects function names longer than this.
const MAX_NAME_CHARS: usize = 64;

/// How long to wait for the TCP/TLS handshake with a remote MCP server.
const HTTP_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Dedicated client for MCP over HTTP.
///
/// Deliberately not [`crate::http::client`]: that one caps every request at 30s, which
/// would sever the long-lived SSE stream Streamable HTTP keeps open for server messages.
/// Only the connect phase is bounded here.
static HTTP_CLIENT: LazyLock<reqwest::Client> = LazyLock::new(|| {
  reqwest::Client::builder()
    .connect_timeout(HTTP_CONNECT_TIMEOUT)
    .build()
    // Only fails when the TLS backend cannot be initialized, which is a startup-time
    // environment problem rather than something a caller could recover from.
    .expect("MCP HTTP client must be constructible")
});

/// A live connection to an MCP server.
///
/// Owns the running service; dropping it tears down the transport and, for stdio
/// servers, the child process. Every [`McpTool`] keeps a shared handle so the connection
/// cannot die while tools that need it are still registered.
pub struct McpConnection {
  service: Arc<RunningService<RoleClient, ()>>,
  label: String,
}

impl McpConnection {
  /// Connect over stdio, spawning `command` as a child process.
  ///
  /// `label` namespaces the server's tools locally; see [`local_name`].
  pub async fn spawn(label: &str, command: Command) -> anyhow::Result<Self> {
    Self::check_label(label)?;

    let transport = TokioChildProcess::new(command)
      .with_context(|| format!("failed to spawn MCP server `{label}`"))?;

    Self::serve(label, transport).await
  }

  /// Connect to a remote server over Streamable HTTP.
  ///
  /// Every entry in `headers` is forwarded verbatim with each request, `Authorization`
  /// included; see [`build_headers`].
  pub async fn connect(
    label: &str,
    url: &str,
    headers: &BTreeMap<String, String>,
  ) -> anyhow::Result<Self> {
    Self::check_label(label)?;
    anyhow::ensure!(!url.trim().is_empty(), "server url must not be empty");

    let mut config = StreamableHttpClientTransportConfig::with_uri(url.trim().to_owned());
    let headers = build_headers(headers)?;
    if !headers.is_empty() {
      config = config.custom_headers(headers);
    }

    let transport = StreamableHttpClientTransport::with_client(HTTP_CLIENT.clone(), config);

    Self::serve(label, transport).await
  }

  fn check_label(label: &str) -> anyhow::Result<()> {
    anyhow::ensure!(!label.trim().is_empty(), "server label must not be empty");
    Ok(())
  }

  /// Run the MCP handshake over an already-built transport.
  async fn serve<T, E, A>(label: &str, transport: T) -> anyhow::Result<Self>
  where
    T: rmcp::transport::IntoTransport<RoleClient, E, A>,
    E: std::error::Error + Send + Sync + 'static,
  {
    // `()` is the client handler: this agent exposes no sampling, roots or elicitation
    // back to the server, it only calls tools.
    let service = ()
      .serve(transport)
      .await
      .with_context(|| format!("failed to initialize MCP server `{label}`"))?;

    let info = service.peer_info();
    tracing::info!(label, ?info, "connected to MCP server");

    Ok(Self {
      service: Arc::new(service),
      label: label.to_owned(),
    })
  }

  /// Discover the server's tools, adapted to the local trait.
  pub async fn tools(&self) -> anyhow::Result<Vec<Arc<dyn Tool>>> {
    let tools = self
      .service
      .list_all_tools()
      .await
      .with_context(|| format!("failed to list tools of MCP server `{}`", self.label))?;

    let adapted = tools
      .into_iter()
      .map(|tool| {
        let remote_name = tool.name.to_string();
        let local = McpTool {
          service: Arc::clone(&self.service),
          name: local_name(&self.label, &remote_name),
          remote_name,
          description: tool
            .description
            .map(|description| description.to_string())
            // A description is optional in MCP but the model needs *something* to decide
            // when to call the tool.
            .unwrap_or_else(|| format!("Tool provided by the `{}` MCP server.", self.label)),
          // `input_schema` is already a JSON Schema object, which is what
          // `function.parameters` expects.
          parameters: Value::Object((*tool.input_schema).clone()),
        };
        Arc::new(local) as Arc<dyn Tool>
      })
      .collect::<Vec<_>>();

    tracing::info!(
      label = %self.label,
      count = adapted.len(),
      tools = ?adapted.iter().map(|tool| tool.name()).collect::<Vec<_>>(),
      "discovered MCP tools"
    );

    Ok(adapted)
  }

  /// Close the connection and wait for the server to finish.
  ///
  /// Fails when tools from this server are still registered somewhere, since shutting
  /// down would leave them unusable. Drop the registry first.
  pub async fn shutdown(self) -> anyhow::Result<()> {
    let label = self.label;
    let service = Arc::try_unwrap(self.service).map_err(|_| {
      anyhow::anyhow!("cannot shut down MCP server `{label}` while its tools are still in use")
    })?;

    service
      .cancel()
      .await
      .with_context(|| format!("failed to shut down MCP server `{label}`"))?;

    tracing::info!(%label, "disconnected from MCP server");
    Ok(())
  }
}

/// One tool exposed by an MCP server.
pub struct McpTool {
  /// Keeps the connection alive for as long as this tool exists.
  service: Arc<RunningService<RoleClient, ()>>,
  /// Name advertised to the model, namespaced by server label.
  name: String,
  /// Name the server itself knows the tool by.
  remote_name: String,
  description: String,
  parameters: Value,
}

#[async_trait::async_trait]
impl Tool for McpTool {
  fn name(&self) -> &str {
    &self.name
  }

  fn description(&self) -> &str {
    &self.description
  }

  fn parameters(&self) -> Value {
    self.parameters.clone()
  }

  async fn execute(&self, args_json: &str) -> anyhow::Result<String> {
    let request = CallToolRequestParams::new(self.remote_name.clone())
      .with_arguments(parse_arguments(args_json)?);

    let result = self
      .service
      .call_tool(request)
      .await
      .with_context(|| format!("MCP tool `{}` call failed", self.remote_name))?;

    let rendered = render_content(&result.content, result.structured_content.as_ref());

    // MCP reports tool-level failures in-band rather than as protocol errors; turn them
    // into an `Err` so the registry feeds the text back to the model like any other
    // tool failure.
    anyhow::ensure!(
      !result.is_error.unwrap_or(false),
      "MCP tool `{}` reported an error: {rendered}",
      self.remote_name
    );

    Ok(rendered)
  }
}

/// Convert configured headers for the transport.
///
/// Every header is passed through verbatim, including `Authorization`. The transport's
/// dedicated `auth_header` slot is deliberately unused: it runs the value through
/// `bearer_auth`, which prepends `Bearer ` — so a config written the normal way
/// (`"Authorization": "Bearer <token>"`) would arrive as `Bearer Bearer <token>`, and no
/// other scheme (`Basic`, custom) could be expressed at all.
///
/// An invalid name or value is an error rather than a silent drop: a header that never
/// arrives usually surfaces as an opaque 401 or 403 from the server.
///
/// Messages name the offending header but never quote its value, which is often a secret.
fn build_headers(
  headers: &BTreeMap<String, String>,
) -> anyhow::Result<HashMap<HeaderName, HeaderValue>> {
  let mut built = HashMap::with_capacity(headers.len());

  for (name, value) in headers {
    let name = name.trim();
    let header_name = HeaderName::try_from(name)
      .with_context(|| format!("`{name}` is not a valid HTTP header name"))?;
    let header_value = HeaderValue::from_str(value.trim())
      .with_context(|| format!("header `{name}` has a value the HTTP layer rejects"))?;

    built.insert(header_name, header_value);
  }

  Ok(built)
}

/// Namespace a remote tool name.
///
/// Two reasons for the prefix: different servers (and the built-in tools) may use the
/// same name, and the registry rejects duplicates; and it keeps the tool's origin
/// visible in logs and in the model's transcript.
///
/// Characters outside `[A-Za-z0-9_-]` are replaced, and the result is truncated, because
/// that is all the chat completions API accepts in a function name. A truncation that
/// happens to collide with another tool is caught by the registry's duplicate check.
///
/// `pub(crate)` (not private) so [`crate::tools::mcp::config`]'s `allowedTools` filter
/// can compute the same local name a configured remote tool name will end up with,
/// without duplicating the sanitization rules here.
pub(crate) fn local_name(label: &str, remote_name: &str) -> String {
  let sanitized = format!("{label}{LABEL_SEPARATOR}{remote_name}")
    .chars()
    .map(|c| {
      if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
        c
      } else {
        '_'
      }
    })
    .take(MAX_NAME_CHARS)
    .collect::<String>();

  if sanitized.chars().count() == MAX_NAME_CHARS {
    tracing::warn!(
      label,
      remote_name,
      "tool name truncated to {MAX_NAME_CHARS} chars to fit the API limit"
    );
  }

  sanitized
}

/// Whether `name` looks like a local MCP tool name, i.e. `label__remote_name` (see
/// [`local_name`]).
///
/// For callbacks (e.g. [`crate::callback::mcp_guard::McpGuardCallback`]) that want to
/// single out MCP tools generically — as opposed to the built-in tools, whose fixed
/// names and argument schemas a callback like [`crate::callback::path_guard::
/// WorkspaceGuardCallback`] already knows by name — without hand-maintaining a second
/// list of every server's tools. A false positive (a non-MCP tool whose own name happens
/// to contain [`LABEL_SEPARATOR`]) is possible in principle but not in practice: no
/// built-in tool name does.
pub(crate) fn is_mcp_tool_name(name: &str) -> bool {
  name.contains(LABEL_SEPARATOR)
}

/// Parse the model's arguments into the object MCP expects.
///
/// Models send an empty string or `null` for tools that take no arguments, neither of
/// which deserializes into an object.
fn parse_arguments(args_json: &str) -> anyhow::Result<JsonObject> {
  let trimmed = args_json.trim();
  if trimmed.is_empty() || trimmed == "null" {
    return Ok(JsonObject::default());
  }

  serde_json::from_str(trimmed).context("MCP tool arguments must be a JSON object")
}

/// Flatten the result into the text a chat tool message can carry.
///
/// Non-text blocks are noted instead of inlined: a tool message is plain text, so binary
/// payloads cannot be passed through, and silently dropping them would leave the model
/// believing the tool returned nothing.
fn render_content(content: &[ContentBlock], structured: Option<&Value>) -> String {
  let mut parts: Vec<String> = content
    .iter()
    .map(|block| match block {
      ContentBlock::Text(text) => text.text.clone(),
      ContentBlock::Image(_) => "[image content omitted]".to_owned(),
      ContentBlock::Audio(_) => "[audio content omitted]".to_owned(),
      ContentBlock::Resource(_) => "[embedded resource omitted]".to_owned(),
      ContentBlock::ResourceLink(resource) => format!("[resource link: {}]", resource.uri),
      // `ContentBlock` is `#[non_exhaustive]`: a newer protocol revision may add block
      // kinds this build does not know about.
      _ => "[unsupported content omitted]".to_owned(),
    })
    .collect();

  // Newer servers may answer with `structured_content` only; fall back to it so the model
  // does not receive an empty tool message.
  if parts.iter().all(|part| part.trim().is_empty())
    && let Some(structured) = structured
  {
    parts = vec![structured.to_string()];
  }

  let joined = parts.join("\n").trim().to_owned();
  if joined.is_empty() {
    "(the tool returned no content)".to_owned()
  } else {
    joined
  }
}

#[cfg(test)]
mod tests {
  use rmcp::model::TextContent;
  use serde_json::json;

  use super::*;

  #[test]
  fn namespaces_remote_names() {
    assert_eq!(local_name("demo", "current_time"), "demo__current_time");
  }

  #[test]
  fn recognizes_mcp_tool_names_by_the_label_separator() {
    assert!(is_mcp_tool_name("demo__current_time"));
    assert!(!is_mcp_tool_name("delete_file"));
    assert!(!is_mcp_tool_name("calculator"));
  }

  /// Build a header map from pairs.
  fn headers(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
      .iter()
      .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
      .collect()
  }

  #[test]
  fn forwards_every_header_including_authorization() {
    let built = build_headers(&headers(&[
      ("Authorization", "Bearer t0ken"),
      ("X-Api-Key", "k123"),
      ("X-Tenant", "acme"),
    ]))
    .unwrap();

    assert_eq!(built.len(), 3);
    // Verbatim: the transport must not re-add a `Bearer` prefix.
    assert_eq!(
      built[&HeaderName::from_static("authorization")],
      "Bearer t0ken"
    );
    assert_eq!(built[&HeaderName::from_static("x-api-key")], "k123");
    assert_eq!(built[&HeaderName::from_static("x-tenant")], "acme");
  }

  #[test]
  fn preserves_non_bearer_auth_schemes() {
    let built = build_headers(&headers(&[("Authorization", "Basic dXNlcjpwYXNz")])).unwrap();
    assert_eq!(
      built[&HeaderName::from_static("authorization")],
      "Basic dXNlcjpwYXNz"
    );
  }

  #[test]
  fn trims_surrounding_whitespace() {
    let built = build_headers(&headers(&[("  X-Api-Key  ", "  k123  ")])).unwrap();
    assert_eq!(built[&HeaderName::from_static("x-api-key")], "k123");
  }

  #[test]
  fn accepts_no_headers() {
    assert!(build_headers(&BTreeMap::new()).unwrap().is_empty());
  }

  #[test]
  fn rejects_invalid_header_names() {
    assert!(build_headers(&headers(&[("bad header", "v")])).is_err());
  }

  #[test]
  fn rejects_invalid_header_values_without_leaking_them() {
    // A newline would let a value inject another header.
    let err = build_headers(&headers(&[("X-Token", "secret\nInjected: 1")])).unwrap_err();
    let message = format!("{err:#}");

    assert!(message.contains("X-Token"), "got: {message}");
    assert!(!message.contains("secret"), "value leaked: {message}");
  }

  #[test]
  fn sanitizes_characters_the_api_rejects() {
    // Dots and slashes are legal in MCP but not in a function name.
    assert_eq!(local_name("demo", "time.now/utc"), "demo__time_now_utc");
  }

  #[test]
  fn keeps_hyphens_which_the_api_allows() {
    // Official servers use them, e.g. `get-sum` on server-everything.
    assert_eq!(local_name("everything", "get-sum"), "everything__get-sum");
  }

  #[test]
  fn truncates_names_over_the_api_limit() {
    let name = local_name("demo", &"x".repeat(100));
    assert_eq!(name.chars().count(), MAX_NAME_CHARS);
    assert!(name.starts_with("demo__"));
  }

  #[test]
  fn treats_blank_arguments_as_an_empty_object() {
    assert!(parse_arguments("").unwrap().is_empty());
    assert!(parse_arguments("  ").unwrap().is_empty());
    assert!(parse_arguments("null").unwrap().is_empty());
    assert!(parse_arguments("{}").unwrap().is_empty());
  }

  #[test]
  fn parses_object_arguments() {
    let args = parse_arguments(r#"{"city":"Paris"}"#).unwrap();
    assert_eq!(args.get("city").unwrap(), "Paris");
  }

  #[test]
  fn rejects_non_object_arguments() {
    assert!(parse_arguments("[1,2]").is_err());
  }

  #[test]
  fn joins_text_blocks() {
    let content = vec![
      ContentBlock::Text(TextContent::new("first")),
      ContentBlock::Text(TextContent::new("second")),
    ];
    assert_eq!(render_content(&content, None), "first\nsecond");
  }

  #[test]
  fn notes_non_text_blocks() {
    let content = vec![ContentBlock::Image(rmcp::model::ImageContent::new(
      "data",
      "image/png",
    ))];
    assert_eq!(render_content(&content, None), "[image content omitted]");
  }

  #[test]
  fn falls_back_to_structured_content() {
    let structured = json!({"temperature": 21});
    assert_eq!(
      render_content(&[], Some(&structured)),
      structured.to_string()
    );
  }

  #[test]
  fn reports_an_empty_result_explicitly() {
    assert_eq!(render_content(&[], None), "(the tool returned no content)");
  }
}
