//! Exact-string search-and-replace edit tool.
//!
//! The model names the text to replace verbatim instead of rewriting the whole file (see
//! [`crate::tools::file_write`]), so an edit cannot silently drop or mangle the parts of
//! the file it was not about. An ambiguous `old_string` is rejected rather than guessed
//! at. See [`execute::run`].

pub mod execute;

use schemars::JsonSchema;
use serde::Deserialize;

use crate::tools::macros::simple_tool;

/// Tool name, used both in the definition and in dispatch.
pub const NAME: &str = "edit_file";

/// Arguments as produced by the model.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct EditFileArgs {
  /// Absolute or relative path of the existing file to edit.
  pub file_path: String,
  /// The exact text to replace, including whitespace and indentation. Must occur exactly
  /// once in the file unless `replace_all` is set; include surrounding lines to make it
  /// unique.
  pub old_string: String,
  /// The text to replace it with. Must differ from `old_string`.
  pub new_string: String,
  /// Replace every occurrence of `old_string` instead of requiring it to be unique.
  #[serde(default)]
  pub replace_all: bool,
}

simple_tool!(
  EditFileTool,
  EditFileArgs,
  name = NAME,
  description = "Edit an existing file by replacing an exact string with a new one. old_string \
                 must match the file verbatim (whitespace included) and occur exactly once, \
                 unless replace_all is true. Read the file first to copy the text exactly.",
  async run = execute::run,
);
