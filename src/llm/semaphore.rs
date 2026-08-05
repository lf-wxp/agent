use std::sync::LazyLock;

use tokio::sync::Semaphore;

use crate::config;

/// Global concurrency gate. `LazyLock` avoids the `get_or_init` branch on every access that `OnceLock` would incur.
static SEMAPHORE: LazyLock<Semaphore> = LazyLock::new(|| Semaphore::new(config::max_concurrency()));

pub fn get_semaphore() -> &'static Semaphore {
  &SEMAPHORE
}
