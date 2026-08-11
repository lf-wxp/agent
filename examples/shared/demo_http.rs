//! Serves [`DemoServer`] over **Streamable HTTP**, shared by `mcp_http_server.rs` and
//! `mcp_http_agent.rs`.
//!
//! Same rules as `demo_server.rs`: not an example target, only reachable through a
//! `#[path]` module declaration.
//!
//! Streamable HTTP puts POST (client → server messages), GET (the server's SSE stream) and
//! DELETE (session teardown) on one path, which is why the whole route is a single
//! `route_service` rather than per-method handlers.

use anyhow::Context;
use axum::{
  Router,
  extract::Request,
  http::{StatusCode, header::AUTHORIZATION},
  middleware::{self, Next},
};
use rmcp::transport::{
  StreamableHttpServerConfig, StreamableHttpService,
  streamable_http_server::session::local::LocalSessionManager,
};

use crate::demo_server::DemoServer;

/// Path the MCP endpoint is mounted at, matching `mcp.example.json`'s `remote` entry.
pub const MCP_PATH: &str = "/mcp";

/// Build the MCP router, optionally gated behind `Authorization: Bearer <token>`.
///
/// The token check exists to exercise the client's header handling end to end: MCP
/// deployments are routinely behind a gateway that expects one, and a header that silently
/// fails to arrive shows up as an opaque 401 rather than a useful error.
pub fn router(bearer_token: Option<String>) -> Router {
  let service = StreamableHttpService::new(
    // Called once per session, so each client conversation gets its own handler instance.
    || Ok(DemoServer::new()),
    LocalSessionManager::default().into(),
    // Defaults only accept loopback `Host` headers, which is what these examples bind to.
    // A public deployment must widen `allowed_hosts` / `allowed_origins` explicitly.
    StreamableHttpServerConfig::default(),
  );

  let router = Router::new().route_service(MCP_PATH, service);

  let Some(token) = bearer_token else {
    return router;
  };

  let expected = format!("Bearer {token}");
  router.layer(middleware::from_fn(move |request: Request, next: Next| {
    let expected = expected.clone();
    async move {
      let presented = request
        .headers()
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok());

      // Rejecting before the handler keeps unauthenticated callers from opening a session.
      if presented != Some(expected.as_str()) {
        return Err(StatusCode::UNAUTHORIZED);
      }

      Ok(next.run(request).await)
    }
  }))
}

/// Bind `addr`, serving until `shutdown` resolves.
///
/// Returns the address actually bound, which matters when the caller asks for port 0 and
/// needs to know where to connect.
pub async fn serve(
  addr: &str,
  bearer_token: Option<String>,
  shutdown: impl Future<Output = ()> + Send + 'static,
) -> anyhow::Result<(std::net::SocketAddr, tokio::task::JoinHandle<()>)> {
  let listener = tokio::net::TcpListener::bind(addr)
    .await
    .with_context(|| format!("failed to bind `{addr}`"))?;
  let bound = listener.local_addr()?;

  let router = router(bearer_token);
  let handle = tokio::spawn(async move {
    if let Err(err) = axum::serve(listener, router)
      .with_graceful_shutdown(shutdown)
      .await
    {
      tracing::error!("MCP HTTP server stopped: {err:#}");
    }
  });

  Ok((bound, handle))
}
