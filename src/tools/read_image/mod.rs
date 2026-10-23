//! Read-an-image tool.
//!
//! Bridges a text-only tool-calling loop to a vision-capable model: the image never
//! enters the conversation history verbatim (which would blow past most context
//! windows once base64-encoded), only the vision model's textual answer does. See
//! [`execute::run`] for the request it builds.
//!
//! The vision request goes through a [`Provider`] like every other model call — its
//! backend, concurrency budget and retry policy — rather than a client of its own. By
//! default that is [`Provider::shared`]; a multi-tenant caller that wants image requests
//! billed to (and rate-limited with) a specific tenant registers
//! [`ReadImageTool::new`] with that tenant's provider instead, e.g. via
//! [`crate::tools::ToolRegistry::select`] without `read_image` plus
//! [`crate::tools::ToolRegistry::add`].

pub mod execute;

use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

use crate::{llm::provider::Provider, tools::tool::Tool};

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

/// See the module docs.
#[derive(Clone, Default)]
pub struct ReadImageTool {
  /// `None` resolves to [`Provider::shared`] when the tool runs, not when it is built:
  /// the shared provider reads its credentials from the environment on first use, and a
  /// registry is routinely built before `.env` has been loaded.
  provider: Option<Provider>,
}

impl ReadImageTool {
  /// Send vision requests through `provider` instead of the shared default.
  pub fn new(provider: Provider) -> Self {
    Self {
      provider: Some(provider),
    }
  }
}

#[async_trait::async_trait]
impl Tool for ReadImageTool {
  fn name(&self) -> &str {
    NAME
  }

  fn description(&self) -> &str {
    "Analyze an image file with a vision model to answer a question about what it shows."
  }

  fn parameters(&self) -> Value {
    crate::tools::schema_of::<ReadImageArgs>()
  }

  async fn execute(&self, args_json: &str) -> crate::Result<String> {
    let provider = self.provider.as_ref().unwrap_or_else(|| Provider::shared());
    execute::run(provider, args_json).await
  }

  /// Read-only: it reads a file and asks a model about it.
  fn concurrency_safe(&self) -> bool {
    true
  }
}
