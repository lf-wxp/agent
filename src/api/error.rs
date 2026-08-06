//! Uniform error type for HTTP handlers: every failure mode collapses into an HTTP status
//! code plus a JSON `{"error": "..."}` body, so handlers never need to hand-roll a response.

use axum::{
  Json,
  http::StatusCode,
  response::{IntoResponse, Response},
};
use serde_json::json;

/// An error that has already decided which HTTP status it maps to.
///
/// Handlers construct one directly for client mistakes ([`Self::bad_request`],
/// [`Self::unauthorized`]) and get one for free from any `anyhow::Error` via `?`
/// (mapped to `500`, since by the time an `anyhow::Error` reaches a handler it is, by
/// construction, something the caller could not have fixed by sending a different request).
#[derive(Debug)]
pub struct ApiError {
  status: StatusCode,
  message: String,
}

impl ApiError {
  pub fn bad_request(message: impl Into<String>) -> Self {
    Self {
      status: StatusCode::BAD_REQUEST,
      message: message.into(),
    }
  }

  pub fn unauthorized(message: impl Into<String>) -> Self {
    Self {
      status: StatusCode::UNAUTHORIZED,
      message: message.into(),
    }
  }

  pub fn conflict(message: impl Into<String>) -> Self {
    Self {
      status: StatusCode::CONFLICT,
      message: message.into(),
    }
  }

  pub fn internal(err: anyhow::Error) -> Self {
    // Log the full chain server-side; the client only gets a generic message so internal
    // details (stack of causes, potentially including upstream provider error bodies)
    // never leak into a response.
    tracing::error!("request failed: {err:#}");
    Self {
      status: StatusCode::INTERNAL_SERVER_ERROR,
      message: "internal error".to_owned(),
    }
  }
}

impl From<anyhow::Error> for ApiError {
  fn from(err: anyhow::Error) -> Self {
    Self::internal(err)
  }
}

impl IntoResponse for ApiError {
  fn into_response(self) -> Response {
    (self.status, Json(json!({ "error": self.message }))).into_response()
  }
}
