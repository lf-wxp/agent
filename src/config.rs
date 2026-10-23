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

/// Environment variable: directory the `cli` binary persists suspended runs under.
const ENV_CLI_APPROVAL_DIR: &str = "AGENT_CLI_APPROVAL_DIR";

/// Environment variable: SQLite file the `cli` binary keeps long-term memory in.
const ENV_CLI_MEMORY_PATH: &str = "AGENT_CLI_MEMORY_PATH";

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

/// Environment variable: seconds one tool call may run before it is abandoned.
const ENV_TOOL_TIMEOUT_SECS: &str = "AGENT_TOOL_TIMEOUT_SECS";

/// Environment variable: tool calls of one round that may execute at the same time.
const ENV_TOOL_MAX_CONCURRENCY: &str = "AGENT_TOOL_MAX_CONCURRENCY";

/// Environment variable: output token cap of one ordinary model request.
const ENV_MAX_OUTPUT_TOKENS: &str = "LLM_MAX_TOKENS";

/// Environment variable: sampling temperature, `0`..=`2`.
const ENV_TEMPERATURE: &str = "LLM_TEMPERATURE";

/// Environment variable: nucleus sampling `top_p`, `0`..=`1`.
const ENV_TOP_P: &str = "LLM_TOP_P";

/// Environment variable: `reasoning_effort` sent to reasoning models, e.g. `low` / `high`.
const ENV_REASONING_EFFORT: &str = "LLM_REASONING_EFFORT";

/// Environment variable: which request field carries the output cap, `max_tokens` or
/// `max_completion_tokens`.
const ENV_TOKEN_LIMIT_FIELD: &str = "LLM_TOKEN_LIMIT_FIELD";

/// Environment variable: send earlier turns' chain of thought back to the model.
const ENV_REPLAY_REASONING: &str = "LLM_REPLAY_REASONING";

/// Environment variable: path to a JSON price list, for cost estimates.
const ENV_PRICING_FILE: &str = "LLM_PRICING_FILE";

/// Environment variable: Tavily API key, used by the `web_search` tool.
const ENV_TAVILY_API_KEY: &str = "TAVILY_API_KEY";

/// Environment variable: Hugging Face read token, used by the `gaia` dataset loader.
const ENV_HF_TOKEN: &str = "HF_TOKEN";

/// Environment variable: Tavily search depth, `basic` / `advanced` / `fast` / `ultra-fast`.
const ENV_TAVILY_SEARCH_DEPTH: &str = "TAVILY_SEARCH_DEPTH";

/// Environment variable: the embedding model, used by retrieval, search compression and
/// `--memory`. Served from the chat provider's own endpoint.
const ENV_EMBED_MODEL: &str = "EMBED_MODEL";

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

/// Default directory for the `cli` binary's suspended runs. A sibling of the session
/// directory rather than a subdirectory of it: the two have opposite lifetimes (see
/// [`crate::agent::approval_store`]), and nesting would make a recursive clear of one
/// silently take the other with it.
const DEFAULT_CLI_APPROVAL_DIR: &str = ".agent/approvals";

/// Default file for the `cli` binary's long-term memory (`--memory`). Under `.agent/`
/// with the rest of its state, and therefore **per workspace**: `main.rs` switches into
/// `--workspace` before reading this, so two projects keep two sets of memories. Point
/// [`ENV_CLI_MEMORY_PATH`] at an absolute path to share one store across workspaces.
///
/// Note that `.gitignore` excluding `.agent/` is this repository's own doing, not a
/// guarantee for a user's projects: a workspace with memory on will have a
/// `.agent/memory.db` holding what the agent was told about them.
const DEFAULT_CLI_MEMORY_PATH: &str = ".agent/memory.db";

/// Default port for the `cli` binary's local web server.
const DEFAULT_CLI_WEB_PORT: u16 = 4173;

/// Default soft token budget for conversation history. Deliberately well under typical
/// 32k-128k model context windows: it leaves headroom for the system prompt, tool
/// definitions, and the model's own output, and keeps per-turn cost bounded for models
/// billed by input tokens.
const DEFAULT_MAX_HISTORY_TOKENS: usize = 6_000;

/// Default wait for a human approval decision: long enough to step away from the keyboard
/// and come back, short enough that an abandoned prompt does not hold the session's turn
/// lock for the rest of the day.
const DEFAULT_APPROVAL_TIMEOUT_SECS: u64 = 300;

/// Default per-call tool timeout. Generous, because it is a backstop against a call that
/// never returns (a wedged MCP server, a stalled download), not a latency target: a slow
/// but working tool killed at this point loses real work.
const DEFAULT_TOOL_TIMEOUT_SECS: u64 = 300;

/// Default number of a round's tool calls executing at once. Models rarely ask for more
/// than a handful per round; the cap is for the one that asks for fifty file reads or
/// searches at once, which would otherwise all hit the disk / network together.
const DEFAULT_TOOL_MAX_CONCURRENCY: usize = 8;

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

/// Directory the `cli` binary persists suspended runs under — a turn that stopped
/// waiting on an approval nobody answered. See [`crate::agent::approval_store`].
/// Override with `AGENT_CLI_APPROVAL_DIR`.
pub fn cli_approval_dir() -> PathBuf {
  non_empty_var(ENV_CLI_APPROVAL_DIR)
    .map(PathBuf::from)
    .unwrap_or_else(|| PathBuf::from(DEFAULT_CLI_APPROVAL_DIR))
}

/// SQLite file the `cli` binary keeps long-term memory in when started with `--memory`
/// (see [`crate::memory`]). Override with `AGENT_CLI_MEMORY_PATH`; relative paths resolve
/// against `--workspace`.
///
/// So the default is per workspace, not per machine: two projects get two sets of
/// memories. An absolute `AGENT_CLI_MEMORY_PATH` is how one store is shared across them.
pub fn cli_memory_path() -> PathBuf {
  non_empty_var(ENV_CLI_MEMORY_PATH)
    .map(PathBuf::from)
    .unwrap_or_else(|| PathBuf::from(DEFAULT_CLI_MEMORY_PATH))
}

/// Port the `cli` binary's local web server binds to on `127.0.0.1`. Override with
/// `AGENT_CLI_WEB_PORT`; invalid values (non-numeric or 0) fall back to the default.
pub fn cli_web_port() -> u16 {
  parsed_var_nonzero(ENV_CLI_WEB_PORT, DEFAULT_CLI_WEB_PORT)
}

/// How long a dangerous-tool approval waits for a human before the run stops waiting
/// (see [`crate::callback::dual_approval::DualApprovalCallback`]). Override with
/// `AGENT_APPROVAL_TIMEOUT_SECS`; invalid values (non-numeric or 0) fall back to the
/// default.
///
/// A bound is required rather than merely nice: a turn holds the session's turn lock for
/// its whole duration, so an approval nobody ever answers would otherwise block every
/// front-end sharing that session — permanently, with no way out but killing the process.
///
/// What happens when it expires is a separate question, and not this setting's to answer:
/// see [`crate::callback::dual_approval::WhenUnanswered`]. By default the run is
/// suspended and stored, so the timeout releases the lock without throwing the turn away
/// — the question stays open, it just stops occupying the session while nobody is there
/// to answer it.
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
/// `window - `[`crate::llm::settings::ModelSettings::max_tokens`] (or past the tighter ceiling the
/// structured routes of [`crate::agent::Agent`] apply) leaves a request that fits on its
/// own and still overflows once the model starts writing. Tool definitions come out of the
/// same window and are not counted either, so a large toolbox wants more headroom still.
/// The default is chosen to leave room for all of it on a 32k window.
pub fn max_history_tokens() -> usize {
  parsed_var_nonzero(ENV_MAX_HISTORY_TOKENS, DEFAULT_MAX_HISTORY_TOKENS)
}

/// How long one tool call may run before it is abandoned and reported to the model as
/// timed out. Applies to every tool that does not declare its own limit via
/// [`crate::tools::Tool::timeout`]. Override with `AGENT_TOOL_TIMEOUT_SECS`; invalid
/// values (non-numeric or 0) fall back to the default.
pub fn tool_timeout() -> Duration {
  Duration::from_secs(parsed_var_nonzero(
    ENV_TOOL_TIMEOUT_SECS,
    DEFAULT_TOOL_TIMEOUT_SECS,
  ))
}

/// How many of one round's tool calls may execute at the same time. Override with
/// `AGENT_TOOL_MAX_CONCURRENCY`; invalid values (non-numeric or 0) fall back to the
/// default.
pub fn tool_max_concurrency() -> usize {
  parsed_var_nonzero(ENV_TOOL_MAX_CONCURRENCY, DEFAULT_TOOL_MAX_CONCURRENCY)
}

/// Output token cap of one ordinary model request, when configured. Override with
/// `LLM_MAX_TOKENS`; `None` when unset or invalid (non-numeric or 0, which is logged),
/// leaving the caller to its own default (see
/// [`crate::llm::settings::ModelSettings::from_config`]).
pub fn max_output_tokens() -> Option<u32> {
  match parsed_var_opt::<u32>(ENV_MAX_OUTPUT_TOKENS)? {
    0 => {
      tracing::warn!("ignoring {ENV_MAX_OUTPUT_TOKENS}: must be greater than 0");
      None
    }
    value => Some(value),
  }
}

/// Sampling temperature, when configured (`LLM_TEMPERATURE`). Raw: the range check
/// lives with the type that sends it, [`crate::llm::settings::ModelSettings`]. A value
/// that is not a number is logged and treated as unset.
pub fn temperature() -> Option<f32> {
  parsed_var_opt(ENV_TEMPERATURE)
}

/// Nucleus sampling `top_p`, when configured (`LLM_TOP_P`). Raw, like [`temperature`].
pub fn top_p() -> Option<f32> {
  parsed_var_opt(ENV_TOP_P)
}

/// Whether the chain of thought of earlier assistant turns is sent back to the model
/// (`LLM_REPLAY_REASONING`). Off unless set: only providers that require it (DeepSeek's
/// thinking mode with tools) accept the extra field, and others reject it.
pub fn replay_reasoning() -> bool {
  parse_bool_var(ENV_REPLAY_REASONING).unwrap_or(false)
}

/// `reasoning_effort` to send, when configured (`LLM_REASONING_EFFORT`).
///
/// A raw string for the same reason as [`structured_mode_override`]: the `llm` layer
/// parses it, so this module does not depend on the request types.
pub fn reasoning_effort_override() -> Option<String> {
  non_empty_var(ENV_REASONING_EFFORT)
}

/// Which request field carries the output cap, when configured
/// (`LLM_TOKEN_LIMIT_FIELD`). Raw, like [`reasoning_effort_override`].
pub fn token_limit_field_override() -> Option<String> {
  non_empty_var(ENV_TOKEN_LIMIT_FIELD)
}

/// The price list cost estimates are made from (`LLM_PRICING_FILE`); see
/// [`crate::llm::pricing`] for its format. `None` when unset, which means no estimates.
pub fn pricing_file() -> Option<PathBuf> {
  non_empty_var(ENV_PRICING_FILE).map(PathBuf::from)
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

/// The embedding model. `None` when `EMBED_MODEL` is unset.
pub fn embed_model() -> Option<String> {
  non_empty_var(ENV_EMBED_MODEL)
}

/// Where text sent for embedding goes, in a form fit to print: the host (and port) of the
/// embeddings endpoint, which is the chat provider's own.
///
/// Mirrors how the client picks its endpoint — `OPENAI_BASE_URL`, else the OpenAI default —
/// but prints only the authority, never the path or anything before an `@`: a base URL can
/// carry credentials (`https://user:key@host/v1`), and a line meant to tell people where
/// their data goes must not become a place a key shows up.
pub fn embed_endpoint_label() -> String {
  let (url, source) = match non_empty_var("OPENAI_BASE_URL") {
    Some(url) => (url, "the chat provider"),
    None => ("https://api.openai.com/v1".to_owned(), "the default"),
  };
  format!("{} via {source}", url_authority(&url))
}

/// The environment variables that used to point embeddings at a provider of their own.
///
/// They were removed when embeddings were unified onto the chat provider's endpoint. Kept
/// as names so [`removed_embed_overrides`] can tell the user their setting stopped doing
/// anything: the alternative is a silent redirect of their documents and memory queries to
/// a host they did not choose.
const REMOVED_EMBED_OVERRIDES: &[&str] = &["EMBED_BASE_URL", "EMBED_API_KEY"];

/// Which of the removed embedding overrides are still set, by name.
///
/// Returns names only, never values: one of them is a credential, and this feeds a line
/// printed to a terminal and, under `--mode web`, to a log. An empty result means nothing
/// to say.
pub fn removed_embed_overrides() -> Vec<&'static str> {
  set_removed_overrides(|name| non_empty_var(name).is_some())
}

fn set_removed_overrides(is_set: impl Fn(&str) -> bool) -> Vec<&'static str> {
  REMOVED_EMBED_OVERRIDES
    .iter()
    .copied()
    .filter(|name| is_set(name))
    .collect()
}

/// The warning for a removed embedding override that is still set, or `None` when there is
/// nothing to warn about.
///
/// One place builds the sentence so the CLI and the examples cannot drift apart. `None`
/// when `EMBED_MODEL` is unset too: nothing is embedded then, so no text goes anywhere and
/// the warning would be noise. Names only, see [`removed_embed_overrides`].
pub fn removed_embed_overrides_warning() -> Option<String> {
  embed_model()?;
  embed_override_warning(&removed_embed_overrides(), &embed_endpoint_label())
}

fn embed_override_warning(ignored: &[&str], endpoint: &str) -> Option<String> {
  if ignored.is_empty() {
    return None;
  }
  let verb = if ignored.len() == 1 { "is" } else { "are" };
  Some(format!(
    "warning: {} {verb} set but no longer read; embeddings go to {endpoint}.",
    ignored.join(" and ")
  ))
}

/// `host[:port]` of `url`, without scheme, path, query or userinfo; `url` itself, trimmed
/// to its first 60 characters, when it has no recognizable scheme.
fn url_authority(url: &str) -> String {
  let Some((_, rest)) = url.split_once("://") else {
    return url.chars().take(60).collect();
  };
  let authority = rest.split(['/', '?', '#']).next().unwrap_or(rest);
  authority
    .rsplit_once('@')
    .map_or(authority, |(_, host)| host)
    .to_owned()
}

/// Environment variable read by the OpenAI client for the chat API key.
const ENV_OPENAI_API_KEY: &str = "OPENAI_API_KEY";

/// The credentials this process itself reads from its environment — the ones nothing it
/// launches on a model's behalf has any business seeing.
///
/// Derived from the same constants the getters above use rather than retyped where it is
/// needed, so a credential added to this module cannot be forgotten by a consumer that
/// has to strip them (see `run_command`).
pub const SECRET_ENV_VARS: &[&str] = &[ENV_OPENAI_API_KEY, ENV_TAVILY_API_KEY, ENV_HF_TOKEN];

/// Whether an environment variable *name* looks like it holds a credential.
///
/// A heuristic on the suffix, as a net under [`SECRET_ENV_VARS`]: the process may have
/// been given other keys through `.env` (a different provider's key, a cloud token) that
/// this module never reads but a shell child could still print. Deliberately by name and
/// not by value — values are secrets and must not be inspected or logged — and
/// deliberately blunt: a false positive costs a command one variable, a false negative
/// hands a prompt-injected page a key.
pub fn is_secret_env_name(name: &str) -> bool {
  const SUFFIXES: &[&str] = &["_API_KEY", "_SECRET_KEY", "_SECRET", "_PASSWORD", "_TOKEN"];
  let name = name.to_ascii_uppercase();
  SECRET_ENV_VARS.contains(&name.as_str()) || SUFFIXES.iter().any(|suffix| name.ends_with(suffix))
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

/// Parse an env var that has no sensible default of its own: `None` when unset or blank,
/// and also — after a warning, so a typo does not silently do nothing — when unparseable.
fn parsed_var_opt<T: FromStr>(key: &str) -> Option<T> {
  let raw = non_empty_var(key)?;
  match raw.parse::<T>() {
    Ok(value) => Some(value),
    Err(_) => {
      tracing::warn!("ignoring {key}: `{raw}` is not a valid value");
      None
    }
  }
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

  /// A credential is recognised by name, whether or not this crate reads that variable:
  /// the ones it does read are matched exactly, anything else by its suffix. Names that
  /// merely mention a token (a tokenizer's thread count) are not credentials.
  #[test]
  fn secret_env_names_are_recognised_by_the_names_that_carry_credentials() {
    for name in [
      "OPENAI_API_KEY",
      "TAVILY_API_KEY",
      "EMBED_API_KEY",
      "HF_TOKEN",
      "GITHUB_TOKEN",
      "AWS_SECRET_KEY",
      "db_password",
      "SOME_SECRET",
    ] {
      assert!(
        is_secret_env_name(name),
        "`{name}` should be treated as a secret"
      );
    }
    for name in [
      "PATH",
      "HOME",
      "CARGO_HOME",
      "VIRTUAL_ENV",
      "TOKENIZERS_PARALLELISM",
    ] {
      assert!(!is_secret_env_name(name), "`{name}` should be left alone");
    }
    // Matching is on the *end* of the name only, and deliberately blunt about it: a
    // variable that merely mentions a credential word in the middle is left alone, while
    // anything that ends in one is stripped even if it is not truly secret — the accepted
    // cost of never guessing wrong in the other direction.
    assert!(!is_secret_env_name("PASSWORD_STORE_DIR"));
    assert!(is_secret_env_name("PASSWORD_STORE_DIR_IS_NOT_A_PASSWORD"));
  }

  /// A removed override that is still set is reported; the ones that are not, are not.
  /// Takes the lookup as a closure so no test has to touch the process environment.
  #[test]
  fn removed_embed_overrides_lists_only_the_ones_still_set() {
    assert!(set_removed_overrides(|_| false).is_empty());
    assert_eq!(
      set_removed_overrides(|name| name == "EMBED_BASE_URL"),
      ["EMBED_BASE_URL"]
    );
    assert_eq!(
      set_removed_overrides(|_| true),
      ["EMBED_BASE_URL", "EMBED_API_KEY"]
    );
  }

  /// The warning names the variables and the endpoint, agrees in number, and says nothing
  /// at all when there is nothing to say. It must not be able to carry a value: it is built
  /// from names only.
  #[test]
  fn embed_override_warning_names_the_variables_and_the_endpoint() {
    assert_eq!(
      embed_override_warning(&[], "host via the chat provider"),
      None
    );

    let one = embed_override_warning(&["EMBED_BASE_URL"], "api.example.com via the chat provider")
      .expect("one override set");
    assert_eq!(
      one,
      "warning: EMBED_BASE_URL is set but no longer read; embeddings go to \
       api.example.com via the chat provider."
    );

    let both = embed_override_warning(
      &["EMBED_BASE_URL", "EMBED_API_KEY"],
      "api.example.com via the chat provider",
    )
    .expect("both overrides set");
    assert!(both.contains("EMBED_BASE_URL and EMBED_API_KEY are set"));
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

  /// A setting with no default of its own is simply absent when unset, blank or
  /// unparseable (the last also logs, which is the point of the helper) — never a panic,
  /// and never a silently different value.
  #[test]
  fn parsed_var_opt_is_none_unless_the_value_parses() {
    assert_eq!(
      parsed_var_opt::<f32>("AGENT_TEST_PARSED_VAR_OPT_UNSET"),
      None
    );

    unsafe { std::env::set_var("AGENT_TEST_PARSED_VAR_OPT_BLANK", "   ") };
    assert_eq!(
      parsed_var_opt::<f32>("AGENT_TEST_PARSED_VAR_OPT_BLANK"),
      None
    );

    unsafe { std::env::set_var("AGENT_TEST_PARSED_VAR_OPT_JUNK", "warm") };
    assert_eq!(
      parsed_var_opt::<f32>("AGENT_TEST_PARSED_VAR_OPT_JUNK"),
      None
    );

    unsafe { std::env::set_var("AGENT_TEST_PARSED_VAR_OPT_VALUE", " 0.7 ") };
    assert_eq!(
      parsed_var_opt::<f32>("AGENT_TEST_PARSED_VAR_OPT_VALUE"),
      Some(0.7)
    );
  }

  #[test]
  fn the_endpoint_label_shows_where_text_goes_and_never_a_credential() {
    assert_eq!(url_authority("https://api.openai.com/v1"), "api.openai.com");
    assert_eq!(
      url_authority("http://localhost:11434/v1"),
      "localhost:11434"
    );
    assert_eq!(url_authority("https://host.example"), "host.example");
    assert_eq!(url_authority("https://host.example?x=1"), "host.example");
    assert_eq!(
      url_authority("https://user:sk-secret@host.example:8443/v1"),
      "host.example:8443",
      "userinfo is dropped"
    );
    assert_eq!(url_authority("https://a@b@host.example/x"), "host.example");
    let junk = url_authority(&"x".repeat(200));
    assert_eq!(
      junk.chars().count(),
      60,
      "no scheme: bounded, not echoed whole"
    );
  }
}
