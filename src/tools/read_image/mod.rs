//! Read-an-image tool.
//!
//! Bridges a text-only tool-calling loop to a vision-capable model: the image never
//! enters the conversation history verbatim (which would blow past most context
//! windows once base64-encoded), only the vision model's textual answer does. See
//! [`execute::run`] for the request it builds.

pub mod execute;

use schemars::JsonSchema;
use serde::Deserialize;

use crate::tools::macros::simple_tool;

/// Tool name, used both in the definition and in dispatch.
pub const NAME: &str = "read_image";

/// Arguments as produced by the model.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReadImageArgs {
  /// Path to a local image file (`.png`, `.gif`, `.webp` are recognized by extension;
  /// anything else is sent as `image/jpeg`).
  pub file_path: String,

  /// What to ask the vision model about the image, e.g. "what does this chart show?".
  pub query: String,

  /// Which vision-capable model to call. Not defaulted: the calling model is expected
  /// to name one it knows supports image input, since the agent has no registry of
  /// which configured models do.
  pub model: String,
}

simple_tool!(
  ReadImageTool,
  ReadImageArgs,
  name = NAME,
  description = "Analyze an image file with a vision model to answer a question about what it shows.",
  async run = execute::run,
);
