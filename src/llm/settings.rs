//! [`ModelSettings`]: the per-request knobs that are about *how* the model answers
//! (length, sampling, reasoning depth), as opposed to *what* it is asked.
//!
//! One value, applied by [`crate::llm::client::request_builder`] to every request built
//! from it, so the plain, streaming and structured routes cannot drift apart on, say,
//! whether `temperature` is sent.

use std::{str::FromStr, sync::OnceLock};

use async_openai::types::chat::{CreateChatCompletionRequestArgs, ReasoningEffort};

use crate::config;

/// Output token budget for normal conversations, when `LLM_MAX_TOKENS` is not set.
///
/// Structured / reasoning scenarios need a larger budget; see
/// `structured::NATIVE_SCHEMA_MAX_TOKENS`. Re-exported as
/// [`crate::llm::client::DEFAULT_MAX_TOKENS`], where it used to live.
pub const DEFAULT_MAX_TOKENS: u32 = 2048;

/// Which request field carries [`ModelSettings::max_tokens`].
///
/// Two exist because the API changed under its users: `max_tokens` is what nearly every
/// OpenAI-compatible server (DeepSeek, vLLM, Ollama, ...) honors, while OpenAI's own
/// reasoning models reject it outright and accept only `max_completion_tokens`. Neither
/// works everywhere, so it is a setting rather than a guess.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TokenLimitField {
  #[default]
  MaxTokens,
  MaxCompletionTokens,
}

impl FromStr for TokenLimitField {
  type Err = crate::Error;

  fn from_str(value: &str) -> Result<Self, Self::Err> {
    match value.trim().to_ascii_lowercase().as_str() {
      "max_tokens" => Ok(Self::MaxTokens),
      "max_completion_tokens" => Ok(Self::MaxCompletionTokens),
      other => crate::error::bail!(
        config,
        "unknown token limit field `{other}`; expected `max_tokens` or `max_completion_tokens`"
      ),
    }
  }
}

/// Parse a `reasoning_effort` name, as accepted by the API.
pub fn parse_reasoning_effort(value: &str) -> crate::Result<ReasoningEffort> {
  Ok(match value.trim().to_ascii_lowercase().as_str() {
    "none" => ReasoningEffort::None,
    "minimal" => ReasoningEffort::Minimal,
    "low" => ReasoningEffort::Low,
    "medium" => ReasoningEffort::Medium,
    "high" => ReasoningEffort::High,
    "xhigh" => ReasoningEffort::Xhigh,
    "max" => ReasoningEffort::Max,
    other => crate::error::bail!(
      config,
      "unknown reasoning effort `{other}`; expected one of none, minimal, low, medium, high, \
       xhigh, max"
    ),
  })
}

/// How the model is asked to answer.
///
/// `None` for a sampling knob means "do not send it", which is not the same as sending
/// the API default: several reasoning models reject a request that names `temperature`
/// or `top_p` at all, so the only setting that works everywhere is to leave them out.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelSettings {
  /// Output token cap of one ordinary request. Routes that need more room raise it for
  /// their own requests (see [`Self::at_least`]), never lower it.
  pub max_tokens: u32,
  /// Sent as [`TokenLimitField::MaxTokens`] unless configured otherwise.
  pub token_limit_field: TokenLimitField,
  /// `0..=2`.
  pub temperature: Option<f32>,
  /// `0..=1`.
  pub top_p: Option<f32>,
  /// How hard a reasoning model thinks before answering. Ignored (or rejected) by models
  /// that do not reason, so unset by default.
  pub reasoning_effort: Option<ReasoningEffort>,
  /// Send the chain of thought of earlier assistant turns back to the model, as
  /// `reasoning_content` on those messages.
  ///
  /// Off by default, because it is a provider-specific field: DeepSeek's thinking mode
  /// *requires* it whenever the request carries tools (and answers 400 without it),
  /// while OpenAI and others reject or ignore it. When on, an [`crate::agent::Agent`]
  /// also records the reasoning into the transcript
  /// ([`crate::agent::ContentItem::Reasoning`]) so it survives across turns and a
  /// resumed run, and the stateless tool loops keep it for the rounds of one call.
  pub replay_reasoning: bool,
}

impl Default for ModelSettings {
  /// The built-in defaults, without reading the environment: [`DEFAULT_MAX_TOKENS`] and
  /// nothing else sent. Use [`Self::from_config`] for the configured values.
  fn default() -> Self {
    Self {
      max_tokens: DEFAULT_MAX_TOKENS,
      token_limit_field: TokenLimitField::default(),
      temperature: None,
      top_p: None,
      reasoning_effort: None,
      replay_reasoning: false,
    }
  }
}

impl ModelSettings {
  /// The settings configured through the environment (`LLM_MAX_TOKENS`,
  /// `LLM_TEMPERATURE`, `LLM_TOP_P`, `LLM_REASONING_EFFORT`, `LLM_TOKEN_LIMIT_FIELD`,
  /// `LLM_REPLAY_REASONING`), falling back to [`Self::default`] for anything unset.
  ///
  /// A value that is set but invalid — unparseable, out of range, or not a name the API
  /// knows — is logged and ignored rather than failing: a typo in `.env` should not take
  /// the whole process down, and a request that leaves the knob out still works.
  ///
  /// Read once per process and cached, like [`crate::llm::provider::Provider::shared`]:
  /// every helper in [`crate::llm`] asks for it on each call, and re-reading would repeat
  /// the warning above on every one. Call it after `.env` has been loaded. For a value
  /// that must follow the environment later, build the settings yourself.
  pub fn from_config() -> Self {
    static CONFIGURED: OnceLock<ModelSettings> = OnceLock::new();
    CONFIGURED.get_or_init(Self::read_environment).clone()
  }

  fn read_environment() -> Self {
    let defaults = Self::default();
    Self {
      max_tokens: config::max_output_tokens().unwrap_or(defaults.max_tokens),
      token_limit_field: config::token_limit_field_override()
        .and_then(|raw| warn_on_error("LLM_TOKEN_LIMIT_FIELD", raw.parse()))
        .unwrap_or(defaults.token_limit_field),
      temperature: config::temperature()
        .and_then(|value| in_range("LLM_TEMPERATURE", value, 0.0, 2.0)),
      top_p: config::top_p().and_then(|value| in_range("LLM_TOP_P", value, 0.0, 1.0)),
      reasoning_effort: config::reasoning_effort_override()
        .and_then(|raw| warn_on_error("LLM_REASONING_EFFORT", parse_reasoning_effort(&raw))),
      replay_reasoning: config::replay_reasoning(),
    }
  }

  /// Only the output cap (and the field that carries it): no sampling, no reasoning
  /// effort, no replay.
  ///
  /// For a request whose model is not the one these settings were written for — a vision
  /// model a tool names, an auxiliary summarizer — where `temperature` or
  /// `reasoning_effort` may be rejected outright. A cap is the one knob every model has.
  #[must_use]
  pub fn cap_only(&self) -> Self {
    Self {
      max_tokens: self.max_tokens,
      token_limit_field: self.token_limit_field,
      ..Self::default()
    }
  }

  /// A copy whose output cap is at least `floor`.
  ///
  /// For a route that knows it needs more room than an ordinary answer — structured
  /// output on a reasoning model, where the chain of thought shares the cap with the JSON
  /// and truncation is a hard failure. Raising rather than replacing keeps a caller who
  /// configured an even larger cap from being cut down to the route's own guess.
  #[must_use]
  pub fn at_least(&self, floor: u32) -> Self {
    let mut settings = self.clone();
    settings.max_tokens = settings.max_tokens.max(floor);
    settings
  }

  /// Write these settings into `builder`. Knobs that are `None` are left untouched.
  pub fn apply(&self, builder: &mut CreateChatCompletionRequestArgs) {
    match self.token_limit_field {
      TokenLimitField::MaxTokens => builder.max_tokens(self.max_tokens),
      TokenLimitField::MaxCompletionTokens => builder.max_completion_tokens(self.max_tokens),
    };
    if let Some(temperature) = self.temperature {
      builder.temperature(temperature);
    }
    if let Some(top_p) = self.top_p {
      builder.top_p(top_p);
    }
    if let Some(effort) = &self.reasoning_effort {
      builder.reasoning_effort(effort.clone());
    }
  }
}

fn warn_on_error<T>(key: &str, parsed: crate::Result<T>) -> Option<T> {
  parsed
    .inspect_err(|err| tracing::warn!("ignoring {key}: {err}"))
    .ok()
}

fn in_range(key: &str, value: f32, min: f32, max: f32) -> Option<f32> {
  if (min..=max).contains(&value) {
    Some(value)
  } else {
    tracing::warn!("ignoring {key}: {value} is outside {min}..={max}");
    None
  }
}

#[cfg(test)]
mod tests {
  use serde_json::Value;

  use super::*;

  fn applied(settings: &ModelSettings) -> Value {
    let mut builder = CreateChatCompletionRequestArgs::default();
    builder.model("gpt-test").messages(Vec::new());
    settings.apply(&mut builder);
    serde_json::to_value(builder.build().unwrap()).unwrap()
  }

  /// Leaving a knob out is the only setting every model accepts; the default must not
  /// send any of them.
  #[test]
  fn the_default_sends_only_the_output_cap() {
    let body = applied(&ModelSettings::default());
    assert_eq!(body["max_tokens"], DEFAULT_MAX_TOKENS);
    for absent in [
      "max_completion_tokens",
      "temperature",
      "top_p",
      "reasoning_effort",
    ] {
      assert!(body.get(absent).is_none(), "`{absent}` was sent: {body}");
    }
  }

  #[test]
  fn every_configured_knob_reaches_the_request() {
    let settings = ModelSettings {
      max_tokens: 777,
      token_limit_field: TokenLimitField::MaxTokens,
      temperature: Some(0.25),
      top_p: Some(0.5),
      reasoning_effort: Some(ReasoningEffort::High),
      replay_reasoning: false,
    };
    let body = applied(&settings);
    assert_eq!(body["max_tokens"], 777);
    assert_eq!(body["temperature"], 0.25);
    assert_eq!(body["top_p"], 0.5);
    assert_eq!(body["reasoning_effort"], "high");
  }

  /// OpenAI's reasoning models reject `max_tokens`; the cap has to move, not be doubled.
  #[test]
  fn the_cap_can_be_sent_as_max_completion_tokens_instead() {
    let settings = ModelSettings {
      token_limit_field: TokenLimitField::MaxCompletionTokens,
      ..ModelSettings::default()
    };
    let body = applied(&settings);
    assert_eq!(body["max_completion_tokens"], DEFAULT_MAX_TOKENS);
    assert!(body.get("max_tokens").is_none(), "got: {body}");
  }

  #[test]
  fn at_least_raises_but_never_lowers_the_cap() {
    let settings = ModelSettings {
      max_tokens: 1000,
      ..ModelSettings::default()
    };
    assert_eq!(settings.at_least(4096).max_tokens, 4096);
    assert_eq!(settings.at_least(10).max_tokens, 1000);
  }

  /// A request for some other model keeps the cap and nothing the model might refuse.
  #[test]
  fn cap_only_drops_everything_but_the_output_cap() {
    let settings = ModelSettings {
      max_tokens: 777,
      token_limit_field: TokenLimitField::MaxCompletionTokens,
      temperature: Some(0.25),
      top_p: Some(0.5),
      reasoning_effort: Some(ReasoningEffort::High),
      replay_reasoning: true,
    };
    let capped = settings.cap_only();
    assert_eq!(capped.max_tokens, 777);
    assert_eq!(
      capped.token_limit_field,
      TokenLimitField::MaxCompletionTokens
    );
    assert_eq!(
      ModelSettings {
        max_tokens: 777,
        token_limit_field: TokenLimitField::MaxCompletionTokens,
        ..ModelSettings::default()
      },
      capped
    );
  }

  #[test]
  fn parses_token_limit_field_names() {
    assert_eq!(
      "MAX_COMPLETION_TOKENS".parse::<TokenLimitField>().unwrap(),
      TokenLimitField::MaxCompletionTokens
    );
    assert_eq!(
      " max_tokens ".parse::<TokenLimitField>().unwrap(),
      TokenLimitField::MaxTokens
    );
    assert!("tokens".parse::<TokenLimitField>().is_err());
  }

  #[test]
  fn parses_reasoning_effort_names() {
    assert_eq!(
      parse_reasoning_effort("High").unwrap(),
      ReasoningEffort::High
    );
    assert_eq!(
      parse_reasoning_effort("minimal").unwrap(),
      ReasoningEffort::Minimal
    );
    assert!(parse_reasoning_effort("extreme").is_err());
  }

  #[test]
  fn out_of_range_sampling_values_are_dropped() {
    assert_eq!(in_range("T", 2.5, 0.0, 2.0), None);
    assert_eq!(in_range("T", -0.1, 0.0, 2.0), None);
    assert_eq!(in_range("T", 1.0, 0.0, 2.0), Some(1.0));
  }
}
