//! Unzip-an-archive tool.
//!
//! Lets the model pull a zip's contents onto disk so they can be explored with
//! [`crate::tools::file_list`] and [`crate::tools::file_read`] instead of the model
//! having to reason about the archive's binary contents directly. See
//! [`execute::run`] for a zip-slip caveat in the extraction path.

pub mod execute;

use schemars::JsonSchema;
use serde::Deserialize;

use crate::tools::macros::simple_tool;

/// Tool name, used both in the definition and in dispatch.
pub const NAME: &str = "unzip_file";

/// Arguments as produced by the model.
#[derive(Deserialize, Debug, JsonSchema)]
pub struct UnzipFileArgs {
  pub zip_path: String,

  /// Destination directory. Defaults to `zip_path` with its extension stripped (e.g.
  /// `archive.zip` -> `archive/`), so a model that only cares about "unzip this" does
  /// not have to invent a destination.
  #[serde(default)]
  pub extract_to: Option<String>,
}

simple_tool!(
  UnzipFileTool,
  UnzipFileArgs,
  name = NAME,
  description =
    "Extract a zip archive so its contents can be explored with list_files and read_file.",
  run = execute::run,
);
