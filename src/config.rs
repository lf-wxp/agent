//! Unified entry point for reading environment-variable configuration.
//!
//! All values are read once and cached, so this must be accessed only after
//! [`crate::telemetry::init`] (which loads `.env` internally); otherwise values
//! set in `.env` will not take effect.

use std::{path::PathBuf, str::FromStr, sync::LazyLock, time::Duration};

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

/// Environment variable: max attempts for a retryable LLM request ([`crate::llm::retry::with_retry`]).
const ENV_MAX_RETRIES: &str = "LLM_MAX_RETRIES";

/// Environment variable: path to the MCP server config file.
const ENV_MCP_CONFIG_PATH: &str = "MCP_CONFIG_PATH";

/// Environment variable: directory the `cli` binary persists its
/// [`crate::agent::session::FileSessionStore`] sessions under.
const ENV_CLI_SESSION_DIR: &str = "AGENT_CLI_SESSION_DIR";

/// Environment variable: port the `cli` binary's `--mode web`/`--mode both` local web
/// server binds to (always on `127.0.0.1`, see `src/bin/cli/web.rs`).
const ENV_CLI_WEB_PORT: &str = "AGENT_CLI_WEB_PORT";

/// Environment variable: directory the `cli` binary's web server serves its static
/// front-end assets from — the `crates/web-ui` crate's `trunk build` output.
const ENV_CLI_WEB_DIST_DIR: &str = "AGENT_CLI_WEB_DIST_DIR";

/// Environment variable: soft token budget for conversation history passed to
/// [`crate::agent::Agent::run_continuing`]; see [`crate::callback::context_optimizer::ContextOptimizer`].
const ENV_MAX_HISTORY_TOKENS: &str = "LLM_MAX_HISTORY_TOKENS";

/// Environment variable: seconds a dangerous-tool approval waits for a human.
const ENV_APPROVAL_TIMEOUT_SECS: &str = "AGENT_APPROVAL_TIMEOUT_SECS";

/// Environment variable: Tavily API key, used by the `web_search` tool.
const ENV_TAVILY_API_KEY: &str = "TAVILY_API_KEY";

/// Environment variable: Hugging Face read token, used by the `gaia` dataset loader.
const ENV_HF_TOKEN: &str = "HF_TOKEN";

/// Environment variable: Tavily search depth, `basic` / `advanced` / `fast` / `ultra-fast`.
const ENV_TAVILY_SEARCH_DEPTH: &str = "TAVILY_SEARCH_DEPTH";

/// Environment variable: Tavily search depth, `basic` / `advanced` / `fast` / `ultra-fast`.
const ENV_EMBED_MODEL: &str = "EMBED_MODEL";

/// Environment variable: base URL for the embeddings API.
///
/// Chat completions and embeddings often need different providers (e.g. a reasoning-only
/// model has no embeddings endpoint), so this is deliberately separate from
/// `OPENAI_BASE_URL`. Unset falls back to it, so a single OpenAI-compatible provider that
/// serves both still works with no extra configuration.
const ENV_EMBED_BASE_URL: &str = "EMBED_BASE_URL";

/// Environment variable: API key for the embeddings API. Unset falls back to `OPENAI_API_KEY`,
/// see [`ENV_EMBED_BASE_URL`].
const ENV_EMBED_API_KEY: &str = "EMBED_API_KEY";

/// Model used when `LLM_MODEL` is not configured.
const DEFAULT_MODEL: &str = "deepseek-v4-flash";

/// Default max concurrency: most LLM services rate-limit requests per minute, so default conservatively to 3.
const DEFAULT_MAX_CONCURRENCY: usize = 3;

/// Default tool-round budget: enough for multi-step tasks while keeping cost bounded.
const DEFAULT_MAX_TOOL_ROUNDS: usize = 10;

/// Default retry attempts for a transient LLM request failure.
const DEFAULT_MAX_RETRIES: usize = 3;

/// Default MCP config location, relative to the working directory.
const DEFAULT_MCP_CONFIG_PATH: &str = "mcp.json";

/// Default directory for the `cli` binary's session files, relative to the working
/// directory.
const DEFAULT_CLI_SESSION_DIR: &str = ".agent/sessions";

/// Default port for the `cli` binary's local web server.
const DEFAULT_CLI_WEB_PORT: u16 = 4173;

/// Default soft token budget for conversation history. Deliberately well under typical
/// 32k-128k model context windows: it leaves headroom for the system prompt, tool
/// definitions, and the model's own output, and keeps per-turn cost bounded for models
/// billed by input tokens.
const DEFAULT_MAX_HISTORY_TOKENS: usize = 6_000;

/// Default wait for a human approval decision: long enough to step away from the keyboard
/// and come back, short enough that an abandoned prompt does not wedge the session for the
/// rest of the day.
const DEFAULT_APPROVAL_TIMEOUT_SECS: u64 = 300;

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
  parsed_var_nonzero(ENV_MAX_CONCURRENCY, DEFAULT_MAX_CONCURRENCY)
}

/// Log level. Override with `RUST_LOG`; falls back to `INFO` when unparseable.
pub fn log_level() -> Level {
  parsed_var(ENV_LOG_LEVEL, Level::INFO)
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
  parsed_var(ENV_MAX_TOOL_ROUNDS, DEFAULT_MAX_TOOL_ROUNDS)
}

/// Max attempts for a retryable LLM request ([`crate::llm::retry::with_retry`]), which
/// every retry-capable call site in this crate shares (a single model request, a whole
/// stream, a whole GAIA structured solve). Override with `LLM_MAX_RETRIES`; `0` disables
/// retrying. Invalid (non-numeric) values fall back to the default.
pub fn max_retries() -> usize {
  parsed_var(ENV_MAX_RETRIES, DEFAULT_MAX_RETRIES)
}

/// Path to the `mcp.json` declaring MCP servers. Override with `MCP_CONFIG_PATH`.
pub fn mcp_config_path() -> PathBuf {
  non_empty_var(ENV_MCP_CONFIG_PATH)
    .map(PathBuf::from)
    .unwrap_or_else(|| PathBuf::from(DEFAULT_MCP_CONFIG_PATH))
}

/// Directory the `cli` binary persists its
/// [`crate::agent::session::FileSessionStore`] sessions under. Override with
/// `AGENT_CLI_SESSION_DIR`.
pub fn cli_session_dir() -> PathBuf {
  non_empty_var(ENV_CLI_SESSION_DIR)
    .map(PathBuf::from)
    .unwrap_or_else(|| PathBuf::from(DEFAULT_CLI_SESSION_DIR))
}

/// Port the `cli` binary's local web server binds to on `127.0.0.1`. Override with
/// `AGENT_CLI_WEB_PORT`; invalid values (non-numeric or 0) fall back to the default.
pub fn cli_web_port() -> u16 {
  parsed_var_nonzero(ENV_CLI_WEB_PORT, DEFAULT_CLI_WEB_PORT)
}

/// How long a dangerous-tool approval waits for a human before giving up and denying
/// (see [`crate::callback::dual_approval::DualApprovalCallback`]). Override with
/// `AGENT_APPROVAL_TIMEOUT_SECS`; invalid values (non-numeric or 0) fall back to the
/// default.
///
/// A bound is required rather than merely nice: a turn holds the session's turn lock for
/// its whole duration, so an approval nobody ever answers would otherwise wedge every
/// front-end sharing that session — permanently, with no way out but killing the process.
pub fn approval_timeout() -> Duration {
  Duration::from_secs(parsed_var_nonzero(
    ENV_APPROVAL_TIMEOUT_SECS,
    DEFAULT_APPROVAL_TIMEOUT_SECS,
  ))
}

/// Directory the `cli` binary's web server serves static front-end assets from.
/// Override with `AGENT_CLI_WEB_DIST_DIR`; defaults to `crates/web-ui/dist` resolved
/// against *this crate's* source directory (via `CARGO_MANIFEST_DIR`, fixed at compile
/// time) rather than the process's current working directory — `--workspace` retargets
/// the latter (see `src/bin/cli/main.rs`'s docs) to whatever directory the model's tools
/// should operate in, which has nothing to do with where the compiled-in front-end
/// assets live on disk. This default only resolves correctly on the machine the binary
/// was built on; a binary copied elsewhere needs `AGENT_CLI_WEB_DIST_DIR` set explicitly.
pub fn cli_web_dist_dir() -> PathBuf {
  non_empty_var(ENV_CLI_WEB_DIST_DIR)
    .map(PathBuf::from)
    .unwrap_or_else(|| PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/crates/web-ui/dist")))
}

/// Soft token budget for conversation history handed to [`crate::agent::Agent::run_continuing`]
/// (see [`crate::callback::context_optimizer::ContextOptimizer`]). Override with `LLM_MAX_HISTORY_TOKENS`;
/// invalid values (non-numeric or 0) fall back to the default.
///
/// This is the budget for the *request*, and the model's context window has to hold the
/// request **plus** the answer it generates. Nothing here checks that: raising this past
/// `window - `[`crate::llm::client::DEFAULT_MAX_TOKENS`] (or past the tighter ceiling the
/// structured routes of [`crate::agent::Agent`] apply) leaves a request that fits on its
/// own and still overflows once the model starts writing. Tool definitions come out of the
/// same window and are not counted either, so a large toolbox wants more headroom still.
/// The default is chosen to leave room for all of it on a 32k window.
pub fn max_history_tokens() -> usize {
  parsed_var_nonzero(ENV_MAX_HISTORY_TOKENS, DEFAULT_MAX_HISTORY_TOKENS)
}

/// Tavily API key used by the `web_search` tool. `None` when unset.
pub fn tavily_api_key() -> Option<String> {
  non_empty_var(ENV_TAVILY_API_KEY)
}

/// Hugging Face read token used by [`crate::gaia::dataset`] to fetch the GAIA dataset.
/// `None` when unset. Create one at <https://huggingface.co/settings/tokens>.
pub fn hf_token() -> Option<String> {
  non_empty_var(ENV_HF_TOKEN)
}

/// Tavily search depth. Override with `TAVILY_SEARCH_DEPTH`.
///
/// This is an operational trade-off (credits and latency versus relevance), so it is
/// configured here rather than exposed in the tool schema for the model to pick.
pub fn tavily_search_depth() -> String {
  non_empty_var(ENV_TAVILY_SEARCH_DEPTH).unwrap_or_else(|| DEFAULT_TAVILY_SEARCH_DEPTH.to_owned())
}

/// Tavily API key used by the `web_search` tool. `None` when unset.
pub fn embed_model() -> Option<String> {
  non_empty_var(ENV_EMBED_MODEL)
}

/// Base URL for the embeddings API. Override with `EMBED_BASE_URL`; `None` when unset,
/// letting the caller fall back to the default OpenAI-compatible base URL.
pub fn embed_base_url() -> Option<String> {
  non_empty_var(ENV_EMBED_BASE_URL)
}

/// API key for the embeddings API. Override with `EMBED_API_KEY`; `None` when unset,
/// letting the caller fall back to `OPENAI_API_KEY`.
pub fn embed_api_key() -> Option<String> {
  non_empty_var(ENV_EMBED_API_KEY)
}

/// Read an environment variable and trim leading/trailing whitespace; unset or blank is treated as not configured.
fn non_empty_var(key: &str) -> Option<String> {
  std::env::var(key)
    .ok()
    .map(|value| value.trim().to_owned())
    .filter(|value| !value.is_empty())
}

/// Parse a numeric/enum env var, falling back to `default` when unset, blank, or
/// unparseable. Every parsed value is accepted as-is, including a type's zero value; see
/// [`parsed_var_nonzero`] for the common case where zero should also fall back to
/// `default`.
fn parsed_var<T: FromStr>(key: &str, default: T) -> T {
  non_empty_var(key)
    .and_then(|value| value.parse::<T>().ok())
    .unwrap_or(default)
}

/// Like [`parsed_var`], but also falls back to `default` when the parsed value equals the
/// type's zero value — for settings where `0` would be nonsensical (a concurrency budget,
/// a TTL in seconds, a token budget, ...) rather than a deliberate choice. Contrast with
/// [`max_tool_rounds`] / [`max_retries`], which use [`parsed_var`] because `0` is
/// meaningful for them (disables tool calling / retrying).
fn parsed_var_nonzero<T: FromStr + PartialEq + Default>(key: &str, default: T) -> T {
  non_empty_var(key)
    .and_then(|value| value.parse::<T>().ok())
    .filter(|value| *value != T::default())
    .unwrap_or(default)
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

  // Each test below uses its own env var name (never read by any `pub fn` in this
  // module) so it cannot collide with another test running in parallel.

  #[test]
  fn parsed_var_falls_back_on_unset_blank_or_unparseable() {
    for key in [
      "AGENT_TEST_PARSED_VAR_UNSET",
      "AGENT_TEST_PARSED_VAR_BLANK",
      "AGENT_TEST_PARSED_VAR_JUNK",
    ] {
      assert_eq!(parsed_var::<usize>(key, 7), 7, "{key} should fall back");
    }
  }

  #[test]
  fn parsed_var_accepts_zero_and_any_other_valid_value() {
    unsafe { std::env::set_var("AGENT_TEST_PARSED_VAR_ZERO", "0") };
    assert_eq!(parsed_var::<usize>("AGENT_TEST_PARSED_VAR_ZERO", 7), 0);

    unsafe { std::env::set_var("AGENT_TEST_PARSED_VAR_VALUE", "5") };
    assert_eq!(parsed_var::<usize>("AGENT_TEST_PARSED_VAR_VALUE", 7), 5);
  }

  #[test]
  fn parsed_var_nonzero_treats_zero_as_invalid() {
    unsafe { std::env::set_var("AGENT_TEST_PARSED_VAR_NONZERO_ZERO", "0") };
    assert_eq!(
      parsed_var_nonzero::<usize>("AGENT_TEST_PARSED_VAR_NONZERO_ZERO", 7),
      7,
      "0 should fall back to the default"
    );

    unsafe { std::env::set_var("AGENT_TEST_PARSED_VAR_NONZERO_VALUE", "5") };
    assert_eq!(
      parsed_var_nonzero::<usize>("AGENT_TEST_PARSED_VAR_NONZERO_VALUE", 7),
      5
    );

    assert_eq!(
      parsed_var_nonzero::<usize>("AGENT_TEST_PARSED_VAR_NONZERO_UNSET", 7),
      7
    );
  }
}
