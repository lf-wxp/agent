//! Shared HTTP client for outbound requests.

use std::{sync::LazyLock, time::Duration};

/// Request timeout.
///
/// Matters most for tools: tool execution happens inline in the tool-calling loop, so a
/// hung endpoint would stall the whole agent instead of just one request.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

static CLIENT: LazyLock<reqwest::Client> = LazyLock::new(|| {
  reqwest::Client::builder()
    .timeout(REQUEST_TIMEOUT)
    .build()
    // Only fails when the TLS backend cannot be initialized: a startup-time environment
    // problem, not something a caller could recover from.
    .expect("HTTP client must be constructible")
});

/// Reuse a single client within the process: avoid rebuilding the connection pool per call.
pub fn client() -> &'static reqwest::Client {
  &CLIENT
}
