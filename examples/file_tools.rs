//! Demonstrates the filesystem tools (`list_files`, `read_file`, `unzip_file`,
//! `delete_file`) via [`agent::llm::complete::chat_complete`] — one focused prompt per
//! tool, same style as `tool_call_complete.rs`.
//!
//! `read_image` is deliberately left out: it needs a vision-capable model, and none of
//! the models configured for this workspace support image input yet.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example file_tools
//! ```

use std::{
  fs,
  io::Write,
  path::{Path, PathBuf},
};

use agent::{
  config,
  llm::{complete::chat_complete, provider::Provider},
  telemetry,
  tools::ToolRegistry,
};

const SYSTEM_PROMPT: &str = "You are a general-purpose assistant with access to filesystem \
                             tools. Always use the tools instead of guessing at file contents.";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
  telemetry::init()?;

  let workspace = workspace_dir();
  setup_workspace(&workspace)?;
  tracing::info!(workspace = %workspace.display(), "demo workspace ready");

  let provider = Provider::shared();
  let model = config::model();
  // `ToolRegistry::builtin()` already includes the filesystem tools alongside
  // `calculator` / `web_search` — no `select()` needed.
  let registry = ToolRegistry::builtin()?;

  // list_files: the model has to call the tool to know what is actually in there.
  let answer = chat_complete(
    provider,
    model,
    Some(SYSTEM_PROMPT),
    &format!(
      "What files and directories are directly inside `{}`?",
      workspace.display()
    ),
    &registry,
  )
  .await?;
  tracing::info!("list_files answer: {answer}");

  // read_file (plain text): the answer must come from the file's actual contents.
  let answer = chat_complete(
    provider,
    model,
    Some(SYSTEM_PROMPT),
    &format!(
      "Read `{}` and tell me exactly what its second line says.",
      workspace.join("notes.txt").display()
    ),
    &registry,
  )
  .await?;
  tracing::info!("read_file (text) answer: {answer}");

  // read_file (csv): exercises the markdown-table rendering path.
  let answer = chat_complete(
    provider,
    model,
    Some(SYSTEM_PROMPT),
    &format!(
      "Read `{}` and tell me the total of the `amount` column.",
      workspace.join("data.csv").display()
    ),
    &registry,
  )
  .await?;
  tracing::info!("read_file (csv) answer: {answer}");

  // unzip_file: extraction target left to the tool's default (zip name, minus extension).
  let answer = chat_complete(
    provider,
    model,
    Some(SYSTEM_PROMPT),
    &format!(
      "There is a zip archive at `{}`. Extract it and tell me the names of the files it contained.",
      workspace.join("bundle.zip").display()
    ),
    &registry,
  )
  .await?;
  tracing::info!("unzip_file answer: {answer}");

  // delete_file: verified independently below, since the model's prose isn't proof.
  let scratch = workspace.join("scratch.txt");
  let answer = chat_complete(
    provider,
    model,
    Some(SYSTEM_PROMPT),
    &format!(
      "Delete the file at `{}` — it is temporary scratch data no longer needed.",
      scratch.display()
    ),
    &registry,
  )
  .await?;
  tracing::info!("delete_file answer: {answer}");
  tracing::info!(
    scratch_still_exists = scratch.exists(),
    "verifying delete_file actually ran"
  );

  fs::remove_dir_all(&workspace).ok();
  Ok(())
}

/// Fixed path so re-running the example does not accumulate stray directories under the
/// OS temp dir; [`setup_workspace`] wipes and recreates it on every run.
fn workspace_dir() -> PathBuf {
  std::env::temp_dir().join("agent-file-tools-demo")
}

/// Lays out the files each prompt above needs:
/// - `notes.txt` — plain text, for the `read_file` text path.
/// - `data.csv` — for the `read_file` markdown-table path.
/// - `bundle.zip` — a small archive, for `unzip_file`.
/// - `scratch.txt` — throwaway file for `delete_file` to remove.
fn setup_workspace(dir: &Path) -> anyhow::Result<()> {
  fs::remove_dir_all(dir).ok();
  fs::create_dir_all(dir)?;

  fs::write(
    dir.join("notes.txt"),
    "This is the first line.\nThis is the second, important line.\nAnd a third one.\n",
  )?;

  fs::write(
    dir.join("data.csv"),
    "item,amount\napple,3\nbanana,5\ncherry,2\n",
  )?;

  write_zip(
    &dir.join("bundle.zip"),
    "hello.txt",
    b"hello from inside the zip",
  )?;

  fs::write(dir.join("scratch.txt"), "temporary, safe to delete\n")?;

  Ok(())
}

/// Writes a single-entry zip archive at `path`.
fn write_zip(path: &Path, entry_name: &str, contents: &[u8]) -> anyhow::Result<()> {
  let file = fs::File::create(path)?;
  let mut zip = zip::ZipWriter::new(file);
  zip.start_file(entry_name, zip::write::SimpleFileOptions::default())?;
  zip.write_all(contents)?;
  zip.finish()?;
  Ok(())
}
