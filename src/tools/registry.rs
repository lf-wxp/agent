//! The set of tools available to a model: the definitions to advertise plus dispatch.

use std::{path::Path, sync::Arc};

use async_openai::types::chat::ChatCompletionTools;

use crate::config;
use crate::tools::{
  calculator::{self, Calculator},
  file_delete::{self, DeleteFileTool},
  file_edit::{self, EditFileTool},
  file_list::{self, ListFileTool},
  file_read::{self, ReadFileTool},
  file_upzip::{self, UnzipFileTool},
  file_write::{self, WriteFileTool},
  grep::{self, GrepTool},
  mcp::{McpConfig, McpConnection},
  plan::{self, UpdatePlanTool},
  read_image::{self, ReadImageTool},
  shell::{self, RunCommandTool},
  tool::{Tool, execute_with_timeout},
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

  /// The built-in tools that cannot write an arbitrary file or run a program, and need
  /// no external process to construct.
  ///
  /// `calculator`, `web_search`, `update_plan` (the model's own checklist, see
  /// [`crate::tools::plan`]), and the filesystem tools that read (`list_files`,
  /// `read_file`, `read_image`, `grep`) plus the two that already existed before the
  /// write/execute tools did (`delete_file`, `unzip_file`). This is what a caller gets
  /// when it reaches for "the defaults" without having set up a sandbox and an approval
  /// step, so it deliberately leaves out `write_file`, `edit_file` and `run_command`:
  /// next to `web_search`, whose results are attacker-influenced text, a shell tool with
  /// no gate in front of it is a prompt-injection-to-code-execution path.
  ///
  /// `delete_file` and `unzip_file` are still destructive — they stay only because they
  /// predate the sandbox/approval split, and dropping them here would be a breaking
  /// change. A caller that cannot gate them (an unattended job) should pick its tools
  /// with [`Self::select`] rather than read this list as "harmless".
  ///
  /// Use [`Self::builtin_all`] when the caller does supply that gate (as the CLI does),
  /// or [`Self::select`] to pick tools by name.
  ///
  /// # Model calls made by a tool
  ///
  /// `read_image` asks a vision model, and the registry cannot know which tenant it is
  /// for: here (and in [`Self::builtin_all`] and [`Self::select`]) it goes through
  /// [`crate::llm::provider::Provider::shared`], i.e. the credentials in the environment
  /// — not through the
  /// provider of whichever [`crate::agent::Agent`] the registry ends up in. A
  /// multi-tenant caller leaves it out of the registry it builds (`select` with every
  /// other name) and adds [`ReadImageTool::new`] with that tenant's provider, so image
  /// requests are billed to, and rate-limited with, the right tenant:
  ///
  /// ```ignore
  /// let mut registry = ToolRegistry::select(&["read_file".to_owned()])?;
  /// registry.add(Arc::new(ReadImageTool::new(tenant_provider.clone())))?;
  /// ```
  pub fn builtin() -> anyhow::Result<Self> {
    let mut registry = Self::empty();
    registry.add(Arc::new(Calculator))?;
    registry.add(Arc::new(WebSearch))?;
    registry.add(Arc::new(DeleteFileTool))?;
    registry.add(Arc::new(ListFileTool))?;
    registry.add(Arc::new(ReadFileTool))?;
    registry.add(Arc::new(UnzipFileTool))?;
    registry.add(Arc::new(ReadImageTool::default()))?;
    registry.add(Arc::new(GrepTool))?;
    registry.add(Arc::new(UpdatePlanTool))?;
    Ok(registry)
  }

  /// Every built-in tool: [`Self::builtin`] plus `write_file`, `edit_file` and
  /// `run_command`.
  ///
  /// This agent is meant to run with local filesystem and shell access, but the tools
  /// that write and execute do nothing to confine themselves — pair them with
  /// [`crate::callback::path_guard::WorkspaceGuardCallback`] and
  /// [`crate::callback::dual_approval::DualApprovalCallback`] as the CLI does. A caller
  /// that cannot (an unattended job, a library embedding with no human to ask) should
  /// use [`Self::builtin`] or [`Self::select`] instead.
  pub fn builtin_all() -> anyhow::Result<Self> {
    let mut registry = Self::builtin()?;
    registry.add(Arc::new(WriteFileTool))?;
    registry.add(Arc::new(EditFileTool))?;
    registry.add(Arc::new(RunCommandTool))?;
    Ok(registry)
  }

  /// A registry containing exactly the named built-in tools, e.g. `["calculator"]`.
  ///
  /// For callers that let a request pick a subset of tools by name rather than always
  /// getting the full [`Self::builtin`] set.
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
        read_image::NAME => Arc::new(ReadImageTool::default()),
        file_write::NAME => Arc::new(WriteFileTool),
        file_edit::NAME => Arc::new(EditFileTool),
        grep::NAME => Arc::new(GrepTool),
        shell::NAME => Arc::new(RunCommandTool),
        plan::NAME => Arc::new(UpdatePlanTool),
        other => anyhow::bail!("unknown tool `{other}`"),
      };
      registry.add(tool)?;
    }
    Ok(registry)
  }

  /// [`Self::builtin`] plus every enabled server declared in an `mcp.json`.
  ///
  /// A missing config file is not an error: MCP is optional, so the result is simply the
  /// built-in tools.
  ///
  /// The returned connections must be kept alive for as long as the registry is used, and
  /// shut down only after it is dropped. Individual servers and tools that fail are logged
  /// and skipped, so one bad entry cannot take the rest down with it.
  pub async fn with_mcp(path: impl AsRef<Path>) -> anyhow::Result<(Self, Vec<McpConnection>)> {
    Self::with_mcp_over(Self::builtin()?, path).await
  }

  /// Like [`Self::with_mcp`], but on top of a registry the caller already built — e.g.
  /// [`Self::builtin_all`], which `with_mcp` deliberately does not use by itself.
  pub async fn with_mcp_over(
    mut registry: Self,
    path: impl AsRef<Path>,
  ) -> anyhow::Result<(Self, Vec<McpConnection>)> {
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

  /// Every registered tool, in registration order.
  pub fn iter(&self) -> impl Iterator<Item = &Arc<dyn Tool>> {
    self.tools.iter()
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

    match execute_with_timeout(tool.as_ref(), arguments, config::tool_timeout()).await {
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
    calculator, file_delete, file_edit, file_list, file_read, file_upzip, file_write, grep, plan,
    read_image, shell, web_search,
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
      grep::NAME,
      plan::NAME,
    ] {
      assert!(registry.contains(name), "missing built-in tool `{name}`");
    }
  }

  /// The point of the split: whoever asks for "the defaults" without having built a
  /// sandbox and an approval step must not receive a shell or arbitrary file writes.
  #[test]
  fn builtin_leaves_out_the_write_and_execute_tools() {
    let registry = ToolRegistry::builtin().unwrap();
    for name in [file_write::NAME, file_edit::NAME, shell::NAME] {
      assert!(!registry.contains(name), "`{name}` is in the default set");
    }
  }

  #[test]
  fn builtin_all_adds_exactly_the_write_and_execute_tools() {
    let all = ToolRegistry::builtin_all().unwrap();
    let default = ToolRegistry::builtin().unwrap();
    assert_eq!(all.len(), default.len() + 3);
    for name in [file_write::NAME, file_edit::NAME, shell::NAME] {
      assert!(all.contains(name), "missing `{name}`");
    }
    for name in default.names() {
      assert!(all.contains(name), "`{name}` was dropped");
    }
  }

  /// `select` and `builtin_all` are two hand-written lists of the same tools; every
  /// built-in must be selectable by name, or `--tools` silently cannot reach it.
  #[test]
  fn every_builtin_is_selectable_by_name() {
    let builtin = ToolRegistry::builtin_all().unwrap();
    let names: Vec<String> = builtin.names().map(str::to_owned).collect();
    let selected = ToolRegistry::select(&names).unwrap();
    assert_eq!(selected.len(), builtin.len());
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
