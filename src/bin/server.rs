//! Standalone HTTP agent server: `cargo run --bin server`.
//!
//! Loads a tenant registry (see [`agent::api::tenant`]) and serves [`agent::api::router`]
//! on [`agent::config::http_addr`] (`AGENT_HTTP_ADDR`, default `0.0.0.0:8080`).
//!
//! Tenant config path: [`agent::config::tenants_config_path`] (`AGENT_TENANTS_PATH`,
//! default `tenants.json`). See `tenants.example.json` for the shape.
//!
//! Also spawns background tasks that periodically evict idle multi-turn sessions
//! ([`agent::config::session_ttl`], `AGENT_SESSION_TTL_SECS`) and expired idempotency
//! records ([`agent::config::idempotency_ttl`], `AGENT_IDEMPOTENCY_TTL_SECS`), so
//! long-lived-but-abandoned entries do not grow this process's memory forever.

use std::{sync::Arc, time::Duration};

use agent::{
  api::{
    self,
    idempotency::IdempotencyStore,
    session::{MemorySessionStore, SessionStore},
  },
  config, telemetry,
};

/// How often the idle-session/idempotency sweeps run. Unlike their TTLs (which change
/// user-visible behavior and are therefore configurable, see [`agent::config::session_ttl`]
/// / [`agent::config::idempotency_ttl`]), this is purely an implementation detail, so it
/// stays a constant rather than another environment variable.
const SWEEP_INTERVAL: Duration = Duration::from_secs(60);

#[tokio::main]
async fn main() -> anyhow::Result<()> {
  telemetry::init()?;

  let tenants_path = config::tenants_config_path();
  let tenants = api::tenant::TenantRegistry::load(&tenants_path)
    .await
    .map_err(|err| {
      anyhow::anyhow!(
        "failed to load tenant config `{}` (see tenants.example.json, or set \
         AGENT_TENANTS_PATH): {err:#}",
        tenants_path.display()
      )
    })?;

  if tenants.is_empty() {
    tracing::warn!(
      "tenant registry is empty: every request to /v1/agent/run will be rejected as \
       unauthorized until at least one tenant is configured"
    );
  } else {
    tracing::info!(tenants = tenants.len(), "tenant registry loaded");
  }

  let sessions = Arc::new(MemorySessionStore::new(config::session_ttl()));
  let idempotency = Arc::new(IdempotencyStore::new(config::idempotency_ttl()));
  spawn_session_sweeper(Arc::clone(&sessions));
  spawn_idempotency_sweeper(Arc::clone(&idempotency));

  let app = api::router(api::AppState {
    tenants: Arc::new(tenants),
    sessions,
    idempotency,
  });

  let addr = config::http_addr();
  let listener = tokio::net::TcpListener::bind(&addr).await?;
  tracing::info!(%addr, "agent HTTP server listening");
  axum::serve(listener, app).await?;

  Ok(())
}

/// Periodically evict sessions idle past their TTL. Runs for the lifetime of the
/// process; there is nothing to join on shutdown since it only ever does harmless,
/// idempotent cleanup work.
fn spawn_session_sweeper(sessions: Arc<MemorySessionStore>) {
  tokio::spawn(async move {
    let mut ticker = tokio::time::interval(SWEEP_INTERVAL);
    loop {
      ticker.tick().await;
      sessions.sweep_expired().await;
    }
  });
}

/// Periodically evict idempotency records past their TTL. Same lifetime/shutdown
/// reasoning as [`spawn_session_sweeper`].
fn spawn_idempotency_sweeper(idempotency: Arc<IdempotencyStore>) {
  tokio::spawn(async move {
    let mut ticker = tokio::time::interval(SWEEP_INTERVAL);
    loop {
      ticker.tick().await;
      idempotency.sweep_expired();
    }
  });
}
