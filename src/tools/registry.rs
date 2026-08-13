//! The set of tools available to a model: the definitions to advertise plus dispatch.

use std::{path::Path, sync::Arc};

use async_openai::types::chat::ChatCompletionTools;

use crate::tools::{
  calculator::{self, Calculator},
  file_delete::{self, DeleteFileTool},
  file_list::{self, ListFileTool},
  file_read::{self, ReadFileTool},
  file_upzip::{self, UnzipFileTool},
  mcp::{McpConfig, McpConnection},
  read_image::{self, ReadImageTool},
  tool::Tool,
  web_search::{self, WebSearch},
};

/// Tools offered for a conversation.
///
/// Definitions and dispatch are derived from the same list, so the model can never be
/// offered a tool that cannot be executed, or execute one it was never told about.
///
/// A value rather than a global: MCP tools are discovered at runtime, and the GAIA
/// experiment needs two registries side by side (with tools and without).
#[derive(Default, Clone)]
pub struct ToolRegistry {
  tools: Vec<Arc<dyn Tool>>,
  definitions: Vec<ChatCompletionTools>,
}

impl ToolRegistry {
  /// No tools: the model must answer from its own knowledge.
  pub fn empty() -> Self {
    Self::default()
  }

  /// The built-in tools, which need no external process to construct.
  ///
  /// Includes the filesystem tools (`file_delete`, `file_list`, `file_read`,
  /// `file_upzip`, `read_image`) by default, alongside `calculator` / `web_search`:
  /// this agent is meant to run with local filesystem access, so there is no
  /// safer-by-default subset to fall back to. Callers that need to withhold them
  /// (e.g. a sandboxed or read-only deployment) should build a registry with
  /// [`Self::select`] instead.
  pub fn builtin() -> anyhow::Result<Self> {
    let mut registry = Self::empty();
    registry.add(Arc::new(Calculator))?;
    registry.add(Arc::new(WebSearch))?;
    registry.add(Arc::new(DeleteFileTool))?;
    registry.add(Arc::new(ListFileTool))?;
    registry.add(Arc::new(ReadFileTool))?;
    registry.add(Arc::new(UnzipFileTool))?;
    registry.add(Arc::new(ReadImageTool))?;
    Ok(registry)
  }

  /// A registry containing exactly the named built-in tools, e.g. `["calculator"]`.
  ///
  /// For callers (the HTTP agent API, see [`crate::api`]) that let a request pick a
  /// subset of tools by name rather than always getting the full [`Self::builtin`] set.
  /// MCP tools are not selectable this way: they only exist once discovered from
  /// `mcp.json` (see [`Self::with_mcp`]), so there is no name to select before that.
  pub fn select(names: &[String]) -> anyhow::Result<Self> {
    let mut registry = Self::empty();
    for name in names {
      let tool: Arc<dyn Tool> = match name.as_str() {
        calculator::NAME => Arc::new(Calculator),
        web_search::NAME => Arc::new(WebSearch),
        file_delete::NAME => Arc::new(DeleteFileTool),
        file_list::NAME => Arc::new(ListFileTool),
        file_read::NAME => Arc::new(ReadFileTool),
        file_upzip::NAME => Arc::new(UnzipFileTool),
        read_image::NAME => Arc::new(ReadImageTool),
        other => anyhow::bail!("unknown tool `{other}`"),
      };
      registry.add(tool)?;
    }
    Ok(registry)
  }

  /// Built-in tools plus every enabled server declared in an `mcp.json`.
  ///
  /// A missing config file is not an error: MCP is optional, so the result is simply the
  /// built-in tools.
  ///
  /// The returned connections must be kept alive for as long as the registry is used, and
  /// shut down only after it is dropped. Individual servers and tools that fail are logged
  /// and skipped, so one bad entry cannot take the rest down with it.
  pub async fn with_mcp(path: impl AsRef<Path>) -> anyhow::Result<(Self, Vec<McpConnection>)> {
    let mut registry = Self::builtin()?;
    let path = path.as_ref();

    if !tokio::fs::try_exists(path).await.unwrap_or(false) {
      tracing::info!(
        "no MCP config at `{}`; continuing with built-in tools only",
        path.display()
      );
      return Ok((registry, Vec::new()));
    }

    let (connections, tools) = McpConfig::load(path).await?.connect_all().await;
    for tool in tools {
      let name = tool.name().to_owned();
      if let Err(err) = registry.add(tool) {
        tracing::warn!("skipping MCP tool `{name}`: {err:#}");
      }
    }

    Ok((registry, connections))
  }

  /// Definitions advertised to the model.
  pub fn definitions(&self) -> &[ChatCompletionTools] {
    &self.definitions
  }

  pub fn is_empty(&self) -> bool {
    self.tools.is_empty()
  }

  pub fn len(&self) -> usize {
    self.tools.len()
  }

  pub fn contains(&self, name: &str) -> bool {
    self.tools.iter().any(|tool| tool.name() == name)
  }

  /// Every registered tool's name, in registration order. Mainly for callers (e.g.
  /// [`crate::callback::path_guard`]'s tests) that need to walk the actual tool list
  /// rather than hand-maintaining a second copy of it.
  pub fn names(&self) -> impl Iterator<Item = &str> {
    self.tools.iter().map(|tool| tool.name())
  }

  /// Look up one tool by name.
  ///
  /// Unlike [`Self::execute`], this returns the tool itself rather than a formatted
  /// string, so a caller that already tracks its own [`crate::agent::ExecutionContext`]
  /// (e.g. [`crate::agent::runtime::Agent`]) can distinguish success from failure instead
  /// of guessing from the text.
  pub fn get(&self, name: &str) -> Option<&Arc<dyn Tool>> {
    self.tools.iter().find(|tool| tool.name() == name)
  }

  /// Register one tool.
  ///
  /// Rejects an already-taken name: dispatch resolves by name, so a duplicate would make
  /// one of the two permanently unreachable with no error at call time. This matters most
  /// for MCP servers, whose tool names are only known at runtime.
  pub fn add(&mut self, tool: Arc<dyn Tool>) -> anyhow::Result<()> {
    let name = tool.name();
    anyhow::ensure!(
      !self.contains(name),
      "duplicate tool name `{name}`; rename it or drop one of the sources"
    );

    self.definitions.push(tool.definition()?);
    self.tools.push(tool);
    Ok(())
  }

  /// Register several tools, e.g. everything discovered on one MCP server.
  pub fn extend<I>(&mut self, tools: I) -> anyhow::Result<()>
  where
    I: IntoIterator<Item = Arc<dyn Tool>>,
  {
    for tool in tools {
      self.add(tool)?;
    }
    Ok(())
  }

  /// Execute a tool call by name.
  ///
  /// Never fails: an unknown or failing tool must still yield a tool message, otherwise
  /// the follow-up request is rejected for a `tool_call_id` without a matching result.
  /// The error text goes back to the model so it can correct itself, which is why
  /// [`Tool::execute`] returns a `Result` and the formatting happens here instead of in
  /// every tool.
  pub async fn execute(&self, name: &str, arguments: &str) -> String {
    // Linear scan: registries hold a handful of tools, so a map would not pay for itself.
    let Some(tool) = self.tools.iter().find(|tool| tool.name() == name) else {
      return format!("Error: unknown tool `{name}`");
    };

    match tool.execute(arguments).await {
      Ok(output) => output,
      Err(err) => {
        // Keep the full chain in the logs; the model only needs the summary.
        tracing::warn!("tool `{name}` failed: {err:#}");
        format!("Error: {err}")
      }
    }
  }
}

impl std::fmt::Debug for ToolRegistry {
  /// Lists names rather than the trait objects, which are not `Debug`.
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("ToolRegistry")
      .field(
        "tools",
        &self
          .tools
          .iter()
          .map(|tool| tool.name())
          .collect::<Vec<_>>(),
      )
      .finish()
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::tools::{
    calculator, file_delete, file_list, file_read, file_upzip, read_image, web_search,
  };

  #[test]
  fn empty_registry_advertises_nothing() {
    let registry = ToolRegistry::empty();
    assert!(registry.is_empty());
    assert!(registry.definitions().is_empty());
  }

  #[test]
  fn builtin_registers_every_tool_once() {
    let registry = ToolRegistry::builtin().unwrap();
    assert_eq!(registry.len(), registry.definitions().len());
    for name in [
      calculator::NAME,
      web_search::NAME,
      file_delete::NAME,
      file_list::NAME,
      file_read::NAME,
      file_upzip::NAME,
      read_image::NAME,
    ] {
      assert!(registry.contains(name), "missing built-in tool `{name}`");
    }
  }

  #[test]
  fn select_builds_only_the_named_tools() {
    let registry = ToolRegistry::select(&[calculator::NAME.to_owned()]).unwrap();
    assert!(registry.contains(calculator::NAME));
    assert!(!registry.contains(web_search::NAME));
  }

  #[test]
  fn select_accepts_filesystem_tool_names() {
    let registry = ToolRegistry::select(&[file_delete::NAME.to_owned()]).unwrap();
    assert!(registry.contains(file_delete::NAME));
    assert!(!registry.contains(calculator::NAME));
  }

  #[test]
  fn select_rejects_unknown_tool_names() {
    let err = ToolRegistry::select(&["nope".to_owned()]).unwrap_err();
    assert!(err.to_string().contains("unknown tool"));
  }

  #[test]
  fn select_of_empty_list_is_the_empty_registry() {
    assert!(ToolRegistry::select(&[]).unwrap().is_empty());
  }

  #[test]
  fn rejects_duplicate_names() {
    let mut registry = ToolRegistry::builtin().unwrap();
    let err = registry.add(Arc::new(Calculator)).unwrap_err();
    assert!(err.to_string().contains("duplicate tool name"));
  }

  #[tokio::test]
  async fn reports_unknown_tool_instead_of_failing() {
    let registry = ToolRegistry::builtin().unwrap();
    assert!(registry.execute("nope", "{}").await.starts_with("Error:"));
  }

  #[tokio::test]
  async fn reports_tool_failure_as_text() {
    let registry = ToolRegistry::builtin().unwrap();
    let output = registry.execute(calculator::NAME, "not json").await;
    assert!(output.starts_with("Error:"), "got: {output}");
  }

  #[tokio::test]
  async fn dispatches_to_the_named_tool() {
    let registry = ToolRegistry::builtin().unwrap();
    let output = registry
      .execute(
        calculator::NAME,
        r#"{"operator":"add","first_number":2,"second_number":3}"#,
      )
      .await;

    assert_eq!(output, "5");
  }
}
