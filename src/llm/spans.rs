//! The `chat` span around one model request; see [`crate::telemetry`] for the span
//! layout as a whole.
//!
//! Attribute names follow the OpenTelemetry GenAI semantic conventions
//! (`gen_ai.request.model`, `gen_ai.usage.input_tokens`, ...), and `otel.*` fields are
//! the ones `tracing-opentelemetry` maps onto the exported span's name, kind and status.

use async_openai::{error::OpenAIError, types::chat::FinishReason};
use tracing::{Span, field::Empty};

use crate::{
  llm::backend::{ChatChunk, ChatResponse},
  util::truncate_chars,
};

/// Chars of an error message kept as a span's status description.
const STATUS_DESCRIPTION_CHARS: usize = 300;

/// A span for one request to `model`; usage and outcome are recorded on it as they
/// become known (see [`record_response`], [`record_chunk`], [`record_error`]).
pub(crate) fn chat_span(model: &str, streaming: bool) -> Span {
  tracing::info_span!(
    "chat",
    otel.name = %format!("chat {model}"),
    otel.kind = "client",
    otel.status_code = Empty,
    otel.status_description = Empty,
    gen_ai.operation.name = "chat",
    gen_ai.request.model = %model,
    gen_ai.request.stream = streaming,
    gen_ai.response.id = Empty,
    gen_ai.response.model = Empty,
    gen_ai.response.finish_reasons = Empty,
    gen_ai.usage.input_tokens = Empty,
    gen_ai.usage.output_tokens = Empty,
    gen_ai.usage.cached_input_tokens = Empty,
    gen_ai.usage.reasoning_tokens = Empty,
    error.type = Empty,
  )
}

pub(crate) fn record_response(span: &Span, response: &ChatResponse) {
  let response = &response.response;
  span.record("gen_ai.response.id", response.id.as_str());
  span.record("gen_ai.response.model", response.model.as_str());
  if let Some(reason) = response
    .choices
    .first()
    .and_then(|choice| choice.finish_reason)
  {
    span.record("gen_ai.response.finish_reasons", finish_reason_name(reason));
  }
  if let Some(usage) = &response.usage {
    record_usage(span, &usage.into());
  }
}

/// Called for every chunk; each one carries a different part of what the span records.
pub(crate) fn record_chunk(span: &Span, chunk: &ChatChunk) {
  let chunk = &chunk.chunk;
  if let Some(reason) = chunk
    .choices
    .first()
    .and_then(|choice| choice.finish_reason)
  {
    span.record("gen_ai.response.id", chunk.id.as_str());
    span.record("gen_ai.response.model", chunk.model.as_str());
    span.record("gen_ai.response.finish_reasons", finish_reason_name(reason));
  }
  if let Some(usage) = &chunk.usage {
    record_usage(span, &usage.into());
  }
}

/// The name the API itself uses (`tool_calls`, `content_filter`), not the Rust variant's.
fn finish_reason_name(reason: FinishReason) -> String {
  serde_json::to_value(reason)
    .ok()
    .and_then(|value| value.as_str().map(str::to_owned))
    .unwrap_or_else(|| format!("{reason:?}"))
}

/// Token counts go in as `i64`: `tracing-opentelemetry` exports a `u64` field as its
/// debug *string*, an `i64` as an integer attribute — which is what a backend can sum.
fn record_usage(span: &Span, usage: &crate::agent::TokenUsage) {
  span.record("gen_ai.usage.input_tokens", i64::from(usage.prompt_tokens));
  span.record(
    "gen_ai.usage.output_tokens",
    i64::from(usage.completion_tokens),
  );
  if usage.cached_prompt_tokens > 0 {
    span.record(
      "gen_ai.usage.cached_input_tokens",
      i64::from(usage.cached_prompt_tokens),
    );
  }
  if usage.reasoning_tokens > 0 {
    span.record(
      "gen_ai.usage.reasoning_tokens",
      i64::from(usage.reasoning_tokens),
    );
  }
}

/// Mark `span` failed. Shared with the agent and tool spans, which declare the same
/// `otel.status_*` / `error.type` fields.
pub(crate) fn record_error(span: &Span, err: &anyhow::Error) {
  span.record("otel.status_code", "error");
  span.record(
    "otel.status_description",
    truncate_chars(&format!("{err:#}"), STATUS_DESCRIPTION_CHARS),
  );
  span.record("error.type", error_type(err));
}

/// A low-cardinality class for `err`, as `error.type` asks for: the HTTP status when the
/// provider answered with one, otherwise a coarse kind.
fn error_type(err: &anyhow::Error) -> String {
  match err.downcast_ref::<OpenAIError>() {
    Some(OpenAIError::ApiError(response)) => response.status_code.as_u16().to_string(),
    Some(OpenAIError::Reqwest(_)) => "transport".to_owned(),
    Some(OpenAIError::StreamError(_)) => "stream".to_owned(),
    Some(OpenAIError::JSONDeserialize(..)) => "invalid_response".to_owned(),
    Some(_) => "openai_error".to_owned(),
    None => "_OTHER".to_owned(),
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn an_api_error_is_classed_by_its_status() {
    let err = anyhow::Error::new(OpenAIError::ApiError(
      async_openai::error::ApiErrorResponse {
        status_code: reqwest::StatusCode::TOO_MANY_REQUESTS,
        api_error: async_openai::error::ApiError {
          message: "slow down".to_owned(),
          r#type: None,
          param: None,
          code: None,
          misalignment: None,
        },
      },
    ));
    assert_eq!(error_type(&err), "429");
    assert_eq!(error_type(&anyhow::anyhow!("other")), "_OTHER");
  }

  #[test]
  fn finish_reasons_use_the_apis_own_names() {
    assert_eq!(finish_reason_name(FinishReason::ToolCalls), "tool_calls");
    assert_eq!(
      finish_reason_name(FinishReason::ContentFilter),
      "content_filter"
    );
    assert_eq!(finish_reason_name(FinishReason::Stop), "stop");
  }
}
