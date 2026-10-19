//! Write-a-whole-file tool.
//!
//! Creates or overwrites `file_path` with `content`, creating missing parent directories.
//! For changing part of an existing file prefer [`crate::tools::file_edit`], which does
//! not require the model to reproduce (and possibly corrupt) everything it is not
//! changing. See [`execute::run`] for details.

pub mod execute;

use schemars::JsonSchema;
use serde::Deserialize;

use crate::tools::macros::simple_tool;

/// Tool name, used both in the definition and in dispatch.
pub const NAME: &str = "write_file";

/// Arguments as produced by the model.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct WriteFileArgs {
  /// Absolute or relative path of the file to write. Missing parent directories are
  /// created. Not sandboxed by the tool itself; see
  /// [`crate::callback::path_guard::WorkspaceGuardCallback`].
  pub file_path: String,
  /// The complete new content of the file. Any existing content is replaced.
  pub content: String,
}

simple_tool!(
  WriteFileTool,
  WriteFileArgs,
  name = NAME,
  description = "Create a file or overwrite it entirely with the given content, creating parent \
                 directories as needed. To change part of an existing file, prefer edit_file.",
  async run = execute::run,
);
