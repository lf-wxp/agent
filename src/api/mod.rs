//! HTTP agent server: exposes [`crate::Agent`] over a REST API so this crate can run as a
//! standalone, independently deployable service instead of only as a Rust library
//! dependency.
//!
//! Multi-tenancy is handled at this layer: every request authenticates with a bearer
//! token (see [`handlers::AuthenticatedTenant`]), which [`tenant::TenantRegistry`] maps to
//! a [`crate::llm::provider::Provider`] — its own upstream credentials and its own
//! concurrency budget (see [`crate::llm::provider::Provider::acquire`]). One tenant's
//! traffic, rate limit, or revoked key can never affect another's.
//!
//! ```text
//! POST /v1/agent/run
//! Authorization: Bearer <tenant token>
//! Idempotency-Key: retry-1234           # optional, see idempotency::IdempotencyStore
//! { "input": "What is the capital of Nepal?", "sessionId": "conversation-1" }
//! ```
//!
//! Multi-turn conversations are opt-in per request via [`dto::RunRequest::session_id`]: a
//! client picks its own id and sends it on every turn, and a [`session::SessionStore`]
//! (see that module for the pluggable-backend design) keeps that conversation's history
//! between calls — trimmed to a token budget by [`crate::agent::history::trim_to_budget`]
//! before it is ever sent back to the model, so a long-lived session cannot grow past the
//! model's context window. Omitting `sessionId` keeps the previous stateless, single-turn
//! behavior.
//!
//! An optional `Idempotency-Key` header (see [`idempotency::IdempotencyStore`]) protects
//! retries — e.g. after a client-side timeout — from running the agent (and spending
//! tokens, and re-triggering any side-effecting tools) more than once for the same
//! logical request.
//!
//! Structured output ([`crate::Agent::run_structured`]) is generic over a Rust type known at
//! compile time, which has no equivalent for an arbitrary HTTP client; [`dto::RunRequest::
//! response_schema`] is the HTTP-facing counterpart instead, backed by
//! [`crate::Agent::run_structured_raw`] — a JSON Schema supplied in the request body rather
//! than a Rust type. It cannot be combined with `sessionId` yet: `run_structured_raw` has no
//! history-seeding counterpart of [`crate::Agent::run_continuing`] to feed prior turns into.
//!
//! See `src/bin/server.rs` for the binary that wires this router up, starts listening,
//! and periodically sweeps expired sessions/idempotency records.

pub mod dto;
pub mod error;
pub mod handlers;
pub mod idempotency;
pub mod session;
pub mod tenant;

use std::sync::Arc;

use axum::{
  Router,
  routing::{get, post},
};
use tower_http::trace::TraceLayer;

use idempotency::IdempotencyStore;
use session::SessionStore;
use tenant::TenantRegistry;

/// Shared, `Clone`-cheap state every handler/extractor gets access to.
#[derive(Clone)]
pub struct AppState {
  pub tenants: Arc<TenantRegistry>,
  pub sessions: Arc<dyn SessionStore>,
  pub idempotency: Arc<IdempotencyStore>,
}

/// Build the router. Separate from the binary so an integration test (or an embedder that
/// wants this service mounted under its own router) can construct one without going
/// through `main`.
pub fn router(state: AppState) -> Router {
  Router::new()
    .route("/healthz", get(handlers::health))
    .route("/v1/agent/run", post(handlers::run))
    .layer(TraceLayer::new_for_http())
    .with_state(state)
}
