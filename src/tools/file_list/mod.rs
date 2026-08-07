//! List-a-directory tool.
//!
//! Gives the model a way to explore the filesystem one level at a time before deciding
//! what to [`crate::tools::file_read`] or [`crate::tools::file_delete`]. See
//! [`execute::run`] for ordering and filtering rules.

pub mod execute;

use schemars::JsonSchema;
use serde::Deserialize;

use crate::tools::macros::simple_tool;

/// Tool name, used both in the definition and in dispatch.
pub const NAME: &str = "list_files";

/// Arguments as produced by the model.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListFilesArgs {
  /// Directory to list. Defaults to the current working directory: models routinely
  /// omit it when they mean "look around from here".
  #[serde(default = "default_path")]
  pub path: String,
}

fn default_path() -> String {
  ".".to_string()
}

simple_tool!(
  ListFileTool,
  ListFilesArgs,
  name = NAME,
  description = "List files and directories at a given path, directories listed first.",
  run = execute::run,
);
