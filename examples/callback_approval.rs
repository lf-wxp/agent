//! Demonstrates [`agent::callback::approval::ApprovalCallback`]: a `BeforeToolCallback`
//! that stops the run and asks for a `y`/`n` on the console before any tool on its
//! dangerous list is executed.
//!
//! Two things worth watching for:
//! - Only the listed tools are gated. `list_files` runs untouched; `delete_file` prompts.
//! - Denying does not abort the run — the call is short-circuited, the model is handed an
//!   error result in place of the tool's output, and it carries on without it. The tool
//!   result recorded in the transcript makes that visible after the fact.
//!
//! The prompt reads from stdin, so run this in a terminal:
//!
//! ```sh
//! cargo run --example callback_approval
//! ```
//!
//! The model is asked to delete two files; answering `y` to one and `n` to the other
//! shows both paths in a single run.

use std::{
  fs,
  path::{Path, PathBuf},
  sync::Arc,
};

use agent::{
  Agent,
  agent::ContentItem,
  callback::approval::ApprovalCallback,
  config,
  llm::provider::Provider,
  telemetry,
  tools::{ToolRegistry, file_delete, file_list},
};

const SYSTEM_PROMPT: &str = "You are a filesystem assistant. Use the tools rather than \
                             guessing, and never ask the user for confirmation yourself — \
                             just call the tool.";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
  telemetry::init()?;

  let workspace = workspace_dir();
  setup_workspace(&workspace)?;
  tracing::info!(workspace = %workspace.display(), "demo workspace ready");

  // Only the two tools this demo needs, so the model cannot wander off into web search.
  let toolbox = Arc::new(ToolRegistry::select(&[
    file_list::NAME.to_owned(),
    file_delete::NAME.to_owned(),
  ])?);

  let agent = Agent::new(
    Provider::shared().clone(),
    config::model(),
    Some(SYSTEM_PROMPT),
    toolbox,
  )
  .with_before_tool_callback(Arc::new(ApprovalCallback::new([file_delete::NAME])));

  // stderr, like the prompt itself, so the two cannot end up interleaved out of order.
  eprintln!("\nTip: approve one deletion (y) and deny the other (n) to see both outcomes.");

  let result = agent
    .run(&format!(
      "In the directory `{}`: list what is in there, then delete every file whose name \
       starts with `tmp-`. Report what you deleted.",
      workspace.display()
    ))
    .await?;

  tracing::info!("Answer: {}", result.output);

  // The transcript is where an approval decision leaves its trace: a denied call is
  // recorded as an `Error` result carrying the callback's message, never having reached
  // the real tool.
  for event in &result.context.events {
    for item in &event.content {
      if let ContentItem::ToolResult {
        name,
        status,
        content,
        ..
      } = item
      {
        tracing::info!(tool = %name, ?status, result = %content, "recorded tool result");
      }
    }
  }

  // The model's prose is not proof; check the filesystem itself.
  let mut remaining = fs::read_dir(&workspace)?
    .filter_map(|entry| Some(entry.ok()?.file_name().to_string_lossy().into_owned()))
    .collect::<Vec<_>>();
  remaining.sort();
  tracing::info!(?remaining, "files left on disk");

  fs::remove_dir_all(&workspace).ok();
  Ok(())
}

/// Fixed path so re-running the example does not accumulate stray directories under the
/// OS temp dir; [`setup_workspace`] wipes and recreates it on every run.
fn workspace_dir() -> PathBuf {
  std::env::temp_dir().join("agent-callback-approval-demo")
}

/// Two `tmp-` files for the model to try to delete, plus one it should leave alone.
fn setup_workspace(dir: &Path) -> anyhow::Result<()> {
  fs::remove_dir_all(dir).ok();
  fs::create_dir_all(dir)?;
  fs::write(dir.join("keep.txt"), "important, do not delete\n")?;
  fs::write(dir.join("tmp-a.log"), "throwaway a\n")?;
  fs::write(dir.join("tmp-b.log"), "throwaway b\n")?;
  Ok(())
}
