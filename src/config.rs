//! Unified entry point for reading environment-variable configuration.
//!
//! All values are read once and cached, so this must be accessed only after
//! [`crate::telemetry::init`] (which loads `.env` internally); otherwise values
//! set in `.env` will not take effect.

use std::{path::PathBuf, sync::LazyLock};

use tracing::Level;

/// Environment variable: model name.
const ENV_MODEL: &str = "LLM_MODEL";

/// Environment variable: max concurrency.
const ENV_MAX_CONCURRENCY: &str = "LLM_MAX_CONCURRENCY";

/// Environment variable: log level, e.g. `trace` / `debug` / `info` / `warn` / `error`.
const ENV_LOG_LEVEL: &str = "RUST_LOG";

/// Environment variable: force a specific structured-output mode, `json_schema` / `json_object`.
const ENV_STRUCTURED_MODE: &str = "LLM_STRUCTURED_MODE";

/// Environment variable: force whether the model is treated as supporting `tool_choice`.
///
/// `1`/`true`/`yes`/`on` force "supported"; `0`/`false`/`no`/`off` force "unsupported".
/// Unset falls back to inference from the model name.
const ENV_SUPPORTS_TOOL_CHOICE: &str = "LLM_SUPPORTS_TOOL_CHOICE";

/// Environment variable: rounds of tool execution allowed before a final answer is forced.
const ENV_MAX_TOOL_ROUNDS: &str = "LLM_MAX_TOOL_ROUNDS";

/// Environment variable: path to the MCP server config file.
const ENV_MCP_CONFIG_PATH: &str = "MCP_CONFIG_PATH";

/// Environment variable: Tavily API key, used by the `web_search` tool.
const ENV_TAVILY_API_KEY: &str = "TAVILY_API_KEY";

/// Environment variable: Tavily search depth, `basic` / `advanced` / `fast` / `ultra-fast`.
const ENV_TAVILY_SEARCH_DEPTH: &str = "TAVILY_SEARCH_DEPTH";

/// Model used when `LLM_MODEL` is not configured.
const DEFAULT_MODEL: &str = "deepseek-v4-flash";

/// Default max concurrency: most LLM services rate-limit requests per minute, so default conservatively to 3.
const DEFAULT_MAX_CONCURRENCY: usize = 3;

/// Default tool-round budget: enough for multi-step tasks while keeping cost bounded.
const DEFAULT_MAX_TOOL_ROUNDS: usize = 10;

/// Default MCP config location, relative to the working directory.
const DEFAULT_MCP_CONFIG_PATH: &str = "mcp.json";

/// Default Tavily search depth: `basic` costs 1 credit and balances latency against relevance.
const DEFAULT_TAVILY_SEARCH_DEPTH: &str = "basic";

/// Lowercased model-name keywords for models that reject any `tool_choice` constraint.
///
/// Reasoning / "thinking" models (e.g. DeepSeek reasoners) return
/// `400 ... Thinking mode does not support this tool_choice`, so a caller that relies on
/// forcing `tool_choice` (see [`crate::agent::runtime::Agent::run_structured`]) must fall
/// back to a `response_format` route for them. `contains` (not `starts_with`) because
/// hosted names carry prefixes/suffixes, e.g. `deepseek-ai/DeepSeek-V3`, `deepseek-v4-flash`.
const NO_TOOL_CHOICE_KEYWORDS: [&str; 1] = ["deepseek"];

static MODEL: LazyLock<String> =
  LazyLock::new(|| non_empty_var(ENV_MODEL).unwrap_or_else(|| DEFAULT_MODEL.to_owned()));

/// Model name. Override with `LLM_MODEL`.
pub fn model() -> &'static str {
  &MODEL
}

/// Max concurrency. Override with `LLM_MAX_CONCURRENCY`; invalid values (non-numeric or 0) fall back to the default.
pub fn max_concurrency() -> usize {
  non_empty_var(ENV_MAX_CONCURRENCY)
    .and_then(|value| value.parse::<usize>().ok())
    .filter(|value| *value > 0)
    .unwrap_or(DEFAULT_MAX_CONCURRENCY)
}

/// Log level. Override with `RUST_LOG`; falls back to `INFO` when unparseable.
pub fn log_level() -> Level {
  non_empty_var(ENV_LOG_LEVEL)
    .and_then(|value| value.parse::<Level>().ok())
    .unwrap_or(Level::INFO)
}

/// Forced override for the structured-output mode; auto-inferred from the model name when unset.
///
/// Returns a raw string instead of an enum: let the `llm` layer parse it so the config layer does not depend on concrete implementation types.
pub fn structured_mode_override() -> Option<String> {
  non_empty_var(ENV_STRUCTURED_MODE)
}

/// Whether the model accepts a `tool_choice` constraint (e.g. forcing a tool call).
///
/// Reasoning / "thinking" models reject it outright, so [`crate::agent::runtime::Agent::run_structured`]
/// uses this to decide between the forced-`tool_choice` route and a `response_format` route.
/// Override with `LLM_SUPPORTS_TOOL_CHOICE`; unset infers from the model name.
pub fn model_supports_tool_choice(model: &str) -> bool {
  if let Some(forced) = parse_bool_var(ENV_SUPPORTS_TOOL_CHOICE) {
    return forced;
  }
  infer_tool_choice_support(model)
}

/// Infer tool_choice support purely from the model name, without reading env vars.
fn infer_tool_choice_support(model: &str) -> bool {
  let model = model.to_ascii_lowercase();
  !NO_TOOL_CHOICE_KEYWORDS
    .iter()
    .any(|keyword| model.contains(keyword))
}

/// Rounds of tool execution allowed before tools are disabled and the model must answer
/// from what it already gathered. Override with `LLM_MAX_TOOL_ROUNDS`.
///
/// Unlike the other limits, `0` is meaningful here: it disables tool calling entirely.
pub fn max_tool_rounds() -> usize {
  non_empty_var(ENV_MAX_TOOL_ROUNDS)
    .and_then(|value| value.parse::<usize>().ok())
    .unwrap_or(DEFAULT_MAX_TOOL_ROUNDS)
}

/// Path to the `mcp.json` declaring MCP servers. Override with `MCP_CONFIG_PATH`.
pub fn mcp_config_path() -> PathBuf {
  non_empty_var(ENV_MCP_CONFIG_PATH)
    .map(PathBuf::from)
    .unwrap_or_else(|| PathBuf::from(DEFAULT_MCP_CONFIG_PATH))
}

/// Tavily API key used by the `web_search` tool. `None` when unset.
pub fn tavily_api_key() -> Option<String> {
  non_empty_var(ENV_TAVILY_API_KEY)
}

/// Tavily search depth. Override with `TAVILY_SEARCH_DEPTH`.
///
/// This is an operational trade-off (credits and latency versus relevance), so it is
/// configured here rather than exposed in the tool schema for the model to pick.
pub fn tavily_search_depth() -> String {
  non_empty_var(ENV_TAVILY_SEARCH_DEPTH).unwrap_or_else(|| DEFAULT_TAVILY_SEARCH_DEPTH.to_owned())
}

/// Read an environment variable and trim leading/trailing whitespace; unset or blank is treated as not configured.
fn non_empty_var(key: &str) -> Option<String> {
  std::env::var(key)
    .ok()
    .map(|value| value.trim().to_owned())
    .filter(|value| !value.is_empty())
}

/// Parse a boolean-ish env var; unrecognized values are treated as unset (returns `None`).
fn parse_bool_var(key: &str) -> Option<bool> {
  match non_empty_var(key)?.to_ascii_lowercase().as_str() {
    "1" | "true" | "yes" | "on" => Some(true),
    "0" | "false" | "no" | "off" => Some(false),
    other => {
      tracing::warn!("ignoring {key}: unrecognized boolean value `{other}`");
      None
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn reasoning_models_do_not_support_tool_choice() {
    for model in [
      "deepseek-v4-flash",
      "DeepSeek-V3",
      "deepseek-ai/DeepSeek-R1",
    ] {
      assert!(
        !infer_tool_choice_support(model),
        "{model} should be treated as unsupported"
      );
    }
  }

  #[test]
  fn other_models_support_tool_choice_by_default() {
    for model in ["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet"] {
      assert!(infer_tool_choice_support(model), "{model} should support");
    }
  }
}
