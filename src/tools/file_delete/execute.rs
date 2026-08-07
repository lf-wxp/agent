use std::{fs, path::Path};

use anyhow::{Context, Ok};

use crate::tools::file_delete::DeleteFileArgs;

/// Removes `file_path`, recursively if it is a directory.
///
/// There is no confirmation step and no trash/undo: this is a permanent, synchronous
/// removal, matching the "cannot be undone" warning in the tool description.
pub fn run(args_json: &str) -> anyhow::Result<String> {
  let DeleteFileArgs { file_path } =
    serde_json::from_str::<DeleteFileArgs>(args_json).context("invalid file delete arguments")?;
  let path = Path::new(&file_path);
  if !path.exists() {
    anyhow::bail!("Path not found: {}", path.display());
  }

  if path.is_dir() {
    fs::remove_dir_all(path)?;
  } else {
    fs::remove_file(path)?;
  }

  let result = format!("Path: {}\n", path.display());
  Ok(result)
}

#[cfg(test)]
mod tests {
  use std::path::PathBuf;

  use super::*;

  /// A path under the OS temp dir that no other test can collide with.
  fn unique_temp_path(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!("agent-test-{label}-{}", uuid::Uuid::new_v4()))
  }

  #[test]
  fn rejects_malformed_json() {
    let err = run("not json").unwrap_err();
    assert!(err.to_string().contains("invalid file delete arguments"));
  }

  #[test]
  fn rejects_missing_path() {
    let path = unique_temp_path("missing");
    let args = format!(r#"{{"file_path":"{}"}}"#, path.display());
    let err = run(&args).unwrap_err();
    assert!(err.to_string().contains("Path not found"));
  }

  #[test]
  fn deletes_a_plain_file() {
    let path = unique_temp_path("file.txt");
    fs::write(&path, b"hello").unwrap();

    let args = format!(r#"{{"file_path":"{}"}}"#, path.display());
    let output = run(&args).unwrap();

    assert!(!path.exists(), "file should have been removed");
    assert!(output.contains(&path.display().to_string()));
  }

  #[test]
  fn deletes_a_directory_and_its_contents() {
    let dir = unique_temp_path("dir");
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("nested.txt"), b"hello").unwrap();

    let args = format!(r#"{{"file_path":"{}"}}"#, dir.display());
    run(&args).unwrap();

    assert!(!dir.exists(), "directory should have been removed");
  }
}
