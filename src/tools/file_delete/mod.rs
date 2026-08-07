//! Delete-a-path tool.
//!
//! Despite the name, it removes whatever is at `file_path` — a single file or a whole
//! directory tree — since the model has no reliable way to know which one it is before
//! asking. See [`execute::run`] for the actual removal logic and its caveats.

pub mod execute;

use schemars::JsonSchema;
use serde::Deserialize;

use crate::tools::macros::simple_tool;

/// Tool name, used both in the definition and in dispatch.
pub const NAME: &str = "delete_file";

/// Arguments as produced by the model.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DeleteFileArgs {
  /// Absolute or relative path to remove. Not sandboxed to any project root: the model
  /// can delete anything the process has permission to, so this tool should only be
  /// offered to trusted callers.
  pub file_path: String,
}

simple_tool!(
  DeleteFileTool,
  DeleteFileArgs,
  name = NAME,
  description = "Deletes a file. This action cannot be undone.",
  run = execute::run,
);
