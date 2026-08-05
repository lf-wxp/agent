//! Observability initialization. The three entry points (`main` / `bin/gaia` / examples) share one configuration.

use tracing_subscriber::FmtSubscriber;

use crate::config;

/// Load `.env` and install the global tracing subscriber.
///
/// Must be called before reading any [`crate::config`] configuration, otherwise values in `.env` will not take effect.
/// A missing `.env` is normal (e.g. when configuration is injected via real environment variables), so it is not treated as an error.
pub fn init() -> anyhow::Result<()> {
  dotenvy::dotenv().ok();

  let subscriber = FmtSubscriber::builder()
    .with_max_level(config::log_level())
    .finish();
  tracing::subscriber::set_global_default(subscriber)?;

  Ok(())
}
