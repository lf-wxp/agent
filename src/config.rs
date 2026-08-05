//! Unified entry point for reading environment-variable configuration.
//!
//! All values are read once and cached, so this must be accessed only after
//! [`crate::telemetry::init`] (which loads `.env` internally); otherwise values
//! set in `.env` will not take effect.

use std::sync::LazyLock;

use tracing::Level;

/// Environment variable: model name.
const ENV_MODEL: &str = "LLM_MODEL";

/// Environment variable: max concurrency.
const ENV_MAX_CONCURRENCY: &str = "LLM_MAX_CONCURRENCY";

/// Environment variable: log level, e.g. `trace` / `debug` / `info` / `warn` / `error`.
const ENV_LOG_LEVEL: &str = "RUST_LOG";

/// Environment variable: force a specific structured-output mode, `json_schema` / `json_object`.
const ENV_STRUCTURED_MODE: &str = "LLM_STRUCTURED_MODE";

/// Model used when `LLM_MODEL` is not configured.
const DEFAULT_MODEL: &str = "deepseek-v4-flash";

/// Default max concurrency: most LLM services rate-limit requests per minute, so default conservatively to 3.
const DEFAULT_MAX_CONCURRENCY: usize = 3;

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

/// Read an environment variable and trim leading/trailing whitespace; unset or blank is treated as not configured.
fn non_empty_var(key: &str) -> Option<String> {
  std::env::var(key)
    .ok()
    .map(|value| value.trim().to_owned())
    .filter(|value| !value.is_empty())
}
