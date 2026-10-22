//! Tool registry and the tools themselves.

pub mod calculator;
pub mod file_delete;
pub mod file_edit;
pub mod file_list;
pub mod file_read;
pub mod file_upzip;
pub mod file_write;
pub mod grep;
pub(crate) mod macros;
pub mod mcp;
pub mod plan;
pub mod read_image;
pub mod registry;
#[cfg(feature = "sqlite")]
pub mod retrieve;
pub mod schedule;
pub mod shell;
pub mod tool;
pub mod web_search;

pub use registry::ToolRegistry;
pub use tool::{Tool, schema_of};

/// Test-only helpers shared by the tools' unit tests.
#[cfg(test)]
pub(crate) mod testing {
  use std::path::{Path, PathBuf};

  /// A fresh directory under the OS temp dir that no other test can collide with,
  /// removed again when dropped — so a test run does not leave a trail of
  /// `agent-test-*` directories behind.
  pub struct TempDir(PathBuf);

  impl TempDir {
    pub fn new(label: &str) -> Self {
      let path = std::env::temp_dir().join(format!("agent-test-{label}-{}", uuid::Uuid::new_v4()));
      std::fs::create_dir_all(&path).expect("the temp dir should be creatable");
      Self(path)
    }

    pub fn path(&self) -> &Path {
      &self.0
    }

    pub fn join(&self, relative: impl AsRef<Path>) -> PathBuf {
      self.0.join(relative)
    }
  }

  impl Drop for TempDir {
    fn drop(&mut self) {
      let _ = std::fs::remove_dir_all(&self.0);
    }
  }
}
