//! Regex content search across a directory tree.
//!
//! Lets the model find where something is defined or used without reading files one by
//! one via [`crate::tools::file_read`]. Read-only and reproducible, which is why
//! [`crate::callback::context_optimizer::compaction`] may rewrite its spent results.
//! See [`execute::run`] for what is skipped and how output is capped.

pub mod execute;

use schemars::JsonSchema;
use serde::Deserialize;

use crate::tools::macros::simple_tool;

/// Tool name, used both in the definition and in dispatch.
pub const NAME: &str = "grep";

/// The path searched when the model omits `path`.
pub const DEFAULT_PATH: &str = ".";

/// Matches reported when the model omits `max_results`.
pub const DEFAULT_MAX_RESULTS: usize = 100;

/// Upper bound on `max_results`, however large the model asks for.
pub const MAX_RESULTS_CAP: usize = 1000;

/// Arguments as produced by the model.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GrepArgs {
  /// Regular expression (Rust `regex` syntax) to search file contents for.
  pub pattern: String,
  /// File or directory to search. Defaults to the current working directory.
  #[serde(default = "default_path")]
  pub path: String,
  /// Only search files whose name matches this glob, e.g. `*.rs`. Supports `*` and `?`.
  #[serde(default)]
  pub include: Option<String>,
  /// Match case-insensitively.
  #[serde(default)]
  pub case_insensitive: bool,
  /// Maximum number of matching lines to return (default 100, at most 1000).
  #[serde(default)]
  pub max_results: Option<usize>,
}

fn default_path() -> String {
  DEFAULT_PATH.to_string()
}

simple_tool!(
  GrepTool,
  GrepArgs,
  name = NAME,
  description = "Search file contents with a regular expression. Returns matching lines as \
                 `path:line: text`. Skips hidden files and directories, target/, \
                 node_modules/, binary and very large files (large ones are counted in the \
                 output).",
  async run = execute::run,
  concurrency_safe = true,
);
