//! HTTP handlers for the agent API.

use std::sync::Arc;

use axum::{
  Json,
  extract::{FromRequestParts, State},
  http::{HeaderMap, request::Parts},
  response::{IntoResponse, Response},
};
use serde_json::Value;

use crate::{
  Agent,
  api::{
    AppState,
    dto::{RunRequest, RunResponse},
    error::ApiError,
    idempotency::Reservation,
  },
  config,
  llm::schema::validate_schema_name,
  tools::ToolRegistry,
};

/// Request header carrying a client-chosen idempotency key. See
/// [`crate::api::idempotency`].
const IDEMPOTENCY_KEY_HEADER: &str = "Idempotency-Key";

/// The tenant a request authenticated as, extracted once per request from its
/// `Authorization: Bearer <token>` header.
///
/// Implementing [`FromRequestParts`] rather than checking the header by hand inside every
/// handler means a handler that forgets to declare this parameter simply does not compile
/// with tenant-scoped access — there is no code path that reaches a handler body without
/// having resolved (and thus authorized) a tenant first.
pub struct AuthenticatedTenant {
  /// The raw bearer token itself, kept around (not just the human-readable `label`) so
  /// [`crate::agent::session::SessionStore`] and [`crate::api::idempotency::IdempotencyStore`]
  /// can scope a `sessionId`/`Idempotency-Key` to the exact tenant that owns it — two
  /// tenants sharing a `label` (a config mistake, but not one this layer should silently
  /// paper over into a data leak) must still never see each other's data.
  pub token: String,
  pub label: String,
  pub provider: crate::llm::provider::Provider,
  pub default_model: Option<String>,
}

impl FromRequestParts<AppState> for AuthenticatedTenant {
  type Rejection = ApiError;

  async fn from_request_parts(
    parts: &mut Parts,
    state: &AppState,
  ) -> Result<Self, Self::Rejection> {
    let header = parts
      .headers
      .get(axum::http::header::AUTHORIZATION)
      .ok_or_else(|| ApiError::unauthorized("missing `Authorization` header"))?;
    let value = header
      .to_str()
      .map_err(|_| ApiError::unauthorized("`Authorization` header is not valid UTF-8"))?;
    let token = value
      .strip_prefix("Bearer ")
      .ok_or_else(|| ApiError::unauthorized("`Authorization` header must be `Bearer <token>`"))?;

    let tenant = state
      .tenants
      .lookup(token)
      .ok_or_else(|| ApiError::unauthorized("unknown or revoked token"))?;

    Ok(Self {
      token: token.to_owned(),
      label: tenant.label.clone(),
      provider: tenant.provider.clone(),
      default_model: tenant.default_model.clone(),
    })
  }
}

/// `GET /healthz`: no auth required, so a load balancer/orchestrator can probe liveness
/// without needing a tenant token.
pub async fn health() -> axum::Json<serde_json::Value> {
  axum::Json(serde_json::json!({ "status": "ok" }))
}

/// `POST /v1/agent/run`: run [`Agent::run_continuing`] once on behalf of the
/// authenticated tenant, honoring an optional `Idempotency-Key` header.
///
/// Without `sessionId` this is exactly as before: a fresh
/// [`crate::agent::ExecutionContext`] every call, nothing persisted (`run_continuing`
/// with an empty history is equivalent to [`Agent::run`]). With `sessionId`, prior turns
/// are loaded from — and the updated transcript saved back to —
/// [`crate::agent::session::SessionStore`], scoped to this tenant's token.
///
/// With an `Idempotency-Key` header (see [`crate::api::idempotency`]), a retry of the
/// exact same request is safe: a repeat with the same key returns the cached response
/// instead of running the agent (and spending tokens, and re-triggering any
/// side-effecting tools) a second time. This function only orchestrates that
/// reserve/run/complete-or-release sequence; the actual agent run is in
/// [`run_authenticated`] so a failure partway through cannot skip releasing the key.
pub async fn run(
  tenant: AuthenticatedTenant,
  State(state): State<AppState>,
  headers: HeaderMap,
  Json(request): Json<RunRequest>,
) -> Result<Response, ApiError> {
  if request.input.trim().is_empty() {
    return Err(ApiError::bad_request("`input` must not be empty"));
  }

  let idempotency_key = idempotency_key_from(&headers)?;
  // Captured up front: `tenant` is consumed by `run_authenticated` below, but the key
  // (if any) must still be released/completed against this same token afterwards.
  let tenant_token = tenant.token.clone();

  if let Some(key) = &idempotency_key {
    match state.idempotency.reserve(&tenant_token, key) {
      Reservation::Duplicate(cached) => return Ok(Json(cached).into_response()),
      Reservation::InProgress => {
        return Err(ApiError::conflict(
          "a request with this `Idempotency-Key` is already being processed",
        ));
      }
      Reservation::Fresh => {}
    }
  }

  let response = match run_authenticated(tenant, &state, request).await {
    Ok(response) => response,
    Err(err) => {
      if let Some(key) = &idempotency_key {
        state.idempotency.release(&tenant_token, key);
      }
      return Err(err);
    }
  };

  if let Some(key) = &idempotency_key {
    let cached = serde_json::to_value(&response).map_err(|err| ApiError::internal(err.into()))?;
    state.idempotency.complete(&tenant_token, key, cached);
  }

  Ok(Json(response).into_response())
}

/// The actual agent run, once authentication and idempotency have both been resolved.
async fn run_authenticated(
  tenant: AuthenticatedTenant,
  state: &AppState,
  request: RunRequest,
) -> Result<RunResponse, ApiError> {
  if let Some(schema_request) = &request.response_schema {
    if request.session_id.is_some() {
      return Err(ApiError::bad_request(
        "`responseSchema` cannot be combined with `sessionId` yet: a structured run has no \
         history-seeding counterpart of `run_continuing` to feed the prior turns into",
      ));
    }
    // Validated here, ahead of the run, so a bad `name` is a `400` naming the actual
    // problem rather than a `500` surfaced from deep inside the agent loop.
    validate_schema_name(&schema_request.name)
      .map_err(|err| ApiError::bad_request(err.to_string()))?;
  }

  let registry = match &request.tools {
    Some(names) => {
      ToolRegistry::select(names).map_err(|err| ApiError::bad_request(err.to_string()))?
    }
    None => ToolRegistry::builtin().map_err(ApiError::internal)?,
  };

  let model = request
    .model
    .or(tenant.default_model)
    .unwrap_or_else(|| config::model().to_owned());

  let mut agent = Agent::new(
    tenant.provider,
    model,
    request.instructions,
    Arc::new(registry),
  );
  if let Some(max_steps) = request.max_steps {
    agent = agent.with_max_steps(max_steps);
  }

  let (output, budget_exhausted, context) = match request.response_schema {
    Some(schema_request) => {
      let result = agent
        .run_structured_raw(&request.input, &schema_request.name, schema_request.schema)
        .await
        .map_err(ApiError::internal)?;
      (result.output, result.budget_exhausted, result.context)
    }
    None => {
      let history = match &request.session_id {
        Some(session_id) => state.sessions.history(&tenant.token, session_id).await,
        None => Vec::new(),
      };
      let result = agent
        .run_continuing(history, &request.input)
        .await
        .map_err(ApiError::internal)?;
      (
        Value::String(result.output),
        result.budget_exhausted,
        result.context,
      )
    }
  };

  // `response_schema` and `session_id` are mutually exclusive (rejected above), so this
  // only ever runs for the free-text path — nothing to persist for a structured run yet.
  if let Some(session_id) = &request.session_id {
    state
      .sessions
      .save(&tenant.token, session_id, context.events.clone())
      .await;
  }

  tracing::info!(
    tenant = %tenant.label,
    session_id = request.session_id.as_deref().unwrap_or("-"),
    budget_exhausted,
    steps = context.current_step,
    "agent run finished"
  );

  Ok(RunResponse {
    output,
    budget_exhausted,
    usage: context.usage.into(),
    steps: context.current_step,
    session_id: request.session_id,
  })
}

/// Extract and validate the optional `Idempotency-Key` header.
fn idempotency_key_from(headers: &HeaderMap) -> Result<Option<String>, ApiError> {
  headers
    .get(IDEMPOTENCY_KEY_HEADER)
    .map(|value| {
      value
        .to_str()
        .map(str::to_owned)
        .map_err(|_| ApiError::bad_request("`Idempotency-Key` header is not valid UTF-8"))
    })
    .transpose()
}

#[cfg(test)]
mod tests {
  use axum::http::HeaderValue;

  use super::*;

  #[test]
  fn idempotency_key_from_absent_header_is_none() {
    let headers = HeaderMap::new();
    assert_eq!(idempotency_key_from(&headers).unwrap(), None);
  }

  #[test]
  fn idempotency_key_from_present_header_is_some() {
    let mut headers = HeaderMap::new();
    headers.insert(IDEMPOTENCY_KEY_HEADER, HeaderValue::from_static("abc-123"));
    assert_eq!(
      idempotency_key_from(&headers).unwrap(),
      Some("abc-123".to_owned())
    );
  }

  #[test]
  fn idempotency_key_from_non_utf8_header_is_a_bad_request() {
    let mut headers = HeaderMap::new();
    // `0xff` is not valid UTF-8 in any position; `to_str()` must fail rather than panic.
    headers.insert(
      IDEMPOTENCY_KEY_HEADER,
      HeaderValue::from_bytes(&[0xff]).unwrap(),
    );
    let err = idempotency_key_from(&headers).unwrap_err();
    let debug = format!("{err:?}");
    assert!(debug.contains("400"), "got: {debug}");
    assert!(debug.contains("not valid UTF-8"), "got: {debug}");
  }
}
