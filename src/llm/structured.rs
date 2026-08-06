//! Structured output: automatically choose the implementation based on model capability.
//!
//! The two approaches differ only in "how the output is constrained", so dispatch is unified in this module:
//! - [`StructuredMode::NativeSchema`]: `response_format = json_schema`, server enforces the structure;
//! - [`StructuredMode::JsonObject`]: `response_format = json_object` + prompt-injected Schema.

use std::{fmt, str::FromStr};

use async_openai::types::chat::{ChatChoice, FinishReason, ResponseFormat};
use schemars::JsonSchema;
use serde::de::DeserializeOwned;

use crate::{
  config,
  llm::{
    client::{build_messages, ensure_valid_params},
    schema::{merge_system, native_schema_format, schema_instruction, strip_code_fence},
    tool_loop::{self, BudgetExhausted},
  },
  tools::ToolRegistry,
  util::truncate_chars,
};

/// Output budget for native schema mode: the structure is server-enforced and the body is usually compact,
/// and such models (e.g. gpt-4o) have a small `max_tokens` ceiling, so it cannot be set too high.
const NATIVE_SCHEMA_MAX_TOKENS: u32 = 4096;

/// Output budget for `json_object` mode.
///
/// Note: for reasoning models like DeepSeek, `max_tokens` covers **chain-of-thought + final answer**.
/// When too small, the chain-of-thought consumes the whole budget and `content` returns empty (`finish_reason = length`),
/// surfacing as `serde_json` "EOF while parsing a value at line 1 column 0".
const JSON_OBJECT_MAX_TOKENS: u32 = 32768;

/// Length (in chars) of raw content kept in error messages, to avoid log bloat.
const RAW_CONTENT_PREVIEW_CHARS: usize = 512;

/// Lowercased model-name keywords known not to support `response_format = json_schema`.
///
/// Use `contains` rather than `starts_with`: hosted model names often carry prefixes or suffixes,
/// e.g. `deepseek-ai/DeepSeek-V3`, `deepseek-v4-flash`.
const JSON_OBJECT_ONLY_KEYWORDS: [&str; 1] = ["deepseek"];

/// Structured-output implementation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructuredMode {
  /// Native `json_schema`: the server guarantees the structure; most reliable.
  ///
  /// Note: `strict = true` requires the schema to satisfy OpenAI's strict subset
  /// (no unknown fields, all properties in `required`). The target type should add
  /// `#[schemars(deny_unknown_fields)]` and avoid `Option` fields.
  NativeSchema,
  /// Only `json_object` is supported: only valid JSON is guaranteed; structure is guided by the injected Schema prompt.
  JsonObject,
}

impl StructuredMode {
  /// Choose the mode by model name; can be forced via `LLM_STRUCTURED_MODE`.
  pub fn for_model(model: &str) -> Self {
    match config::structured_mode_override() {
      Some(raw) => match raw.parse() {
        Ok(mode) => mode,
        // A misconfigured value should not fail the whole flow; fall back to inference and leave a clue.
        Err(err) => {
          tracing::warn!("ignoring LLM_STRUCTURED_MODE: {err}");
          Self::infer(model)
        }
      },
      None => Self::infer(model),
    }
  }

  /// Infer purely from the model name, without reading environment variables.
  fn infer(model: &str) -> Self {
    let model = model.to_ascii_lowercase();
    if JSON_OBJECT_ONLY_KEYWORDS
      .iter()
      .any(|keyword| model.contains(keyword))
    {
      Self::JsonObject
    } else {
      Self::NativeSchema
    }
  }

  /// Output token budget for this mode.
  pub fn max_tokens(self) -> u32 {
    match self {
      Self::NativeSchema => NATIVE_SCHEMA_MAX_TOKENS,
      Self::JsonObject => JSON_OBJECT_MAX_TOKENS,
    }
  }
}

impl FromStr for StructuredMode {
  type Err = anyhow::Error;

  fn from_str(value: &str) -> Result<Self, Self::Err> {
    match value.trim().to_ascii_lowercase().as_str() {
      "native" | "native_schema" | "json_schema" => Ok(Self::NativeSchema),
      "json_object" | "prompt" => Ok(Self::JsonObject),
      other => {
        anyhow::bail!("unknown structured mode `{other}`; expected `json_schema` or `json_object`")
      }
    }
  }
}

/// Output was truncated by `max_tokens`. This is a deterministic failure; retrying only burns tokens again,
/// so callers can skip retries accordingly (see `gaia::solver`).
#[derive(Debug)]
pub struct TruncatedOutput {
  pub limit: u32,
}

impl fmt::Display for TruncatedOutput {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(
      f,
      "output truncated by max_tokens ({}): the reasoning chain consumed the whole budget, so the \
       JSON body is empty or incomplete",
      self.limit
    )
  }
}

impl std::error::Error for TruncatedOutput {}

/// The first choice returned by a structured request, along with the mode used to make it.
///
/// Carrying `mode` lets the parse stage give accurate diagnostics (e.g. the real budget when truncated).
#[derive(Debug)]
pub struct StructuredChoice {
  choice: ChatChoice,
  mode: StructuredMode,
  budget_exhausted: bool,
}

impl StructuredChoice {
  pub fn finish_reason(&self) -> Option<FinishReason> {
    self.choice.finish_reason
  }

  pub fn mode(&self) -> StructuredMode {
    self.mode
  }

  /// `true` when the tool round budget ran out before the model finished, so the answer
  /// rests on partial information. See [`crate::llm::tool_loop::Completion`].
  pub fn budget_exhausted(&self) -> bool {
    self.budget_exhausted
  }

  /// Parse the JSON text into `T`, converting every failure into a diagnosable error.
  ///
  /// Failures are tagged with [`BudgetExhausted`] when the tool budget had run out, so
  /// callers can skip a retry that would spend the whole budget again.
  pub fn parse<T: DeserializeOwned>(self) -> anyhow::Result<T> {
    let finish_reason = self.finish_reason();
    let budget_exhausted = self.budget_exhausted;

    // When truncated by max_tokens the JSON is necessarily incomplete (under reasoning models content is often empty);
    // throw a dedicated error up front to avoid degrading into the cryptic "EOF while parsing a value".
    if finish_reason == Some(FinishReason::Length) {
      return Err(
        TruncatedOutput {
          limit: self.mode.max_tokens(),
        }
        .into(),
      );
    }

    let content = self
      .choice
      .message
      .content
      .filter(|content| !content.trim().is_empty())
      .ok_or_else(|| {
        tag_budget(
          anyhow::anyhow!("Empty content in response (finish_reason: {finish_reason:?})"),
          budget_exhausted,
        )
      })?;

    serde_json::from_str::<T>(strip_code_fence(&content)).map_err(|err| {
      tag_budget(
        anyhow::anyhow!(
          "Failed to parse structured output: {err}; raw content: {}",
          truncate_chars(&content, RAW_CONTENT_PREVIEW_CHARS)
        ),
        budget_exhausted,
      )
    })
  }
}

/// Mark a failure as caused by a budget-exhausted attempt, so retries can be skipped.
fn tag_budget(err: anyhow::Error, budget_exhausted: bool) -> anyhow::Error {
  if budget_exhausted {
    err.context(BudgetExhausted)
  } else {
    err
  }
}

/// Issue one structured completion and return the first choice without parsing.
///
/// Use this when you need special handling based on `finish_reason` (e.g. content filtering, see `gaia::solver`);
/// otherwise use [`chat_complete_structured`] directly.
pub async fn chat_complete_structured_raw<T: JsonSchema>(
  model: &str,
  system: Option<&str>,
  prompt: &str,
  registry: &ToolRegistry,
) -> anyhow::Result<StructuredChoice> {
  ensure_valid_params(model, prompt)?;

  let mode = StructuredMode::for_model(model);
  let (system_content, response_format) = match mode {
    StructuredMode::NativeSchema => (system.map(str::to_owned), native_schema_format::<T>()),
    // The Schema must be generated from the **target type**, otherwise the model answers with a different structure.
    StructuredMode::JsonObject => (
      Some(merge_system(system, &schema_instruction::<T>())),
      ResponseFormat::JsonObject,
    ),
  };
  tracing::debug!(model, ?mode, "structured completion started");

  // Goes through the tool loop so a structured request can also use tools: without it the
  // model's tool call would leave `content` empty and surface as a bogus parse error.
  let completion = tool_loop::run(
    model,
    build_messages(system_content.as_deref(), prompt)?,
    registry,
    mode.max_tokens(),
    Some(response_format),
  )
  .await?;

  Ok(StructuredChoice {
    choice: completion.choice,
    mode,
    budget_exhausted: completion.budget_exhausted,
  })
}

/// Convenience wrapper: one request + parse.
pub async fn chat_complete_structured<T>(
  model: &str,
  system: Option<&str>,
  prompt: &str,
  registry: &ToolRegistry,
) -> anyhow::Result<T>
where
  T: JsonSchema + DeserializeOwned,
{
  chat_complete_structured_raw::<T>(model, system, prompt, registry)
    .await?
    .parse()
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn infers_json_object_for_deepseek_variants() {
    for model in [
      "deepseek-v4-flash",
      "DeepSeek-V3",
      "deepseek-ai/DeepSeek-R1",
    ] {
      assert_eq!(StructuredMode::infer(model), StructuredMode::JsonObject);
    }
  }

  #[test]
  fn infers_native_schema_by_default() {
    assert_eq!(
      StructuredMode::infer("gpt-4o-mini"),
      StructuredMode::NativeSchema
    );
  }

  #[test]
  fn parses_mode_aliases() {
    assert_eq!(
      "json_schema".parse::<StructuredMode>().unwrap(),
      StructuredMode::NativeSchema
    );
    assert_eq!(
      " JSON_OBJECT ".parse::<StructuredMode>().unwrap(),
      StructuredMode::JsonObject
    );
    assert!("nope".parse::<StructuredMode>().is_err());
  }
}
