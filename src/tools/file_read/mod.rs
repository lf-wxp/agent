//! Read-a-file tool.
//!
//! Text files come back annotated with line numbers (so the model can refer to
//! specific lines, e.g. when asking [`crate::tools::file_delete`]-adjacent follow-ups),
//! and `.csv` files come back as a markdown table instead, since raw CSV is harder for
//! the model to line up into columns than prose. See [`execute::run`] for both paths.

pub mod execute;

use schemars::JsonSchema;
use serde::Deserialize;

use crate::tools::macros::simple_tool;

/// Tool name, used both in the definition and in dispatch.
pub const NAME: &str = "read_file";

/// Arguments as produced by the model.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReadFileArgs {
  pub file_path: String,

  /// 1-based, inclusive. Defaults to the first line; ignored entirely for `.csv` files.
  #[serde(default = "default_start")]
  pub start_line: usize,

  /// 1-based, inclusive. Negative (the default, `-1`) means "read to the end of the
  /// file" — a plain `usize` couldn't express that without a magic sentinel value that
  /// collides with a real line number.
  #[serde(default = "default_end")]
  pub end_line: i64,
}

fn default_start() -> usize {
  1
}

fn default_end() -> i64 {
  -1
}

simple_tool!(
  ReadFileTool,
  ReadFileArgs,
  name = NAME,
  description = "Read a text file (returned with line numbers, optionally a range) or a CSV file (returned as a markdown table).",
  run = execute::run,
);
