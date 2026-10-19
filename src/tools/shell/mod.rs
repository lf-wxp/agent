//! Run-a-shell-command tool.
//!
//! The most powerful built-in by far: anything the process may do, a command may do. The
//! tool itself only bounds *how long* and *how much output* (see [`execute::run`]); the
//! command string cannot be path-checked the way the file tools' arguments can, so
//! whether it runs at all is up to a human via
//! [`crate::callback::dual_approval::DualApprovalCallback`] — the CLI gates it by default.

pub mod execute;

use schemars::JsonSchema;
use serde::Deserialize;

use crate::tools::macros::simple_tool;

/// Tool name, used both in the definition and in dispatch.
pub const NAME: &str = "run_command";

/// Seconds a command may run when the model omits `timeout_secs`.
pub const DEFAULT_TIMEOUT_SECS: u64 = 120;

/// Upper bound on `timeout_secs`, however long the model asks for.
pub const MAX_TIMEOUT_SECS: u64 = 600;

/// Arguments as produced by the model.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RunCommandArgs {
  /// The command line, run through `sh -c` (`cmd /C` on Windows). stdin is closed, so
  /// interactive commands fail instead of waiting for input.
  pub command: String,
  /// Directory to run in. Defaults to the current working directory.
  #[serde(default)]
  pub workdir: Option<String>,
  /// Kill the command after this many seconds (default 120, at most 600).
  #[serde(default)]
  pub timeout_secs: Option<u64>,
}

simple_tool!(
  RunCommandTool,
  RunCommandArgs,
  name = NAME,
  description = "Run a shell command and return its exit code, stdout and stderr. \
                 Non-interactive (stdin is closed); long-running commands are killed after \
                 timeout_secs. Long output is truncated in the middle.",
  async run = execute::run,
  // Its own, per-call limit (`timeout_secs`) is enforced inside and kills the process
  // group; the outer limit only has to stay out of its way, so it sits just above the
  // largest value the model may ask for.
  timeout = std::time::Duration::from_secs(MAX_TIMEOUT_SECS + 10),
);
