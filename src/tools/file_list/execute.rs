use std::{fs, path::Path};

use anyhow::{Context, Ok};

use crate::tools::file_list::ListFilesArgs;

/// Lists the immediate children of `path`: directories first, then files, each group
/// sorted alphabetically. Entries whose name starts with `.` are skipped — dotfiles are
/// rarely what the model is looking for, and surfacing them (`.git`, `.env`, ...) would
/// just add noise and risk.
///
/// Not recursive: the model is expected to call this again on a subdirectory it wants
/// to go into.
pub fn run(args_json: &str) -> anyhow::Result<String> {
  let ListFilesArgs { path } =
    serde_json::from_str::<ListFilesArgs>(args_json).context("invalid file read arguments")?;
  let path = Path::new(&path);
  if !path.exists() {
    anyhow::bail!("Path not found: {}", path.display());
  }
  if !path.is_dir() {
    anyhow::bail!("Not a directory: {}", path.display());
  }

  let mut dirs = Vec::new();
  let mut files = Vec::new();

  for entry in fs::read_dir(path)? {
    let entry = entry?;
    let name = entry.file_name().to_string_lossy().into_owned();
    if name.starts_with('.') {
      continue;
    }
    if entry.file_type()?.is_dir() {
      dirs.push(format!("{name}/"));
    } else {
      files.push(name);
    }
  }
  dirs.sort();
  files.sort();

  let mut result = format!("Directory: {}\n", path.display());
  for item in dirs.into_iter().chain(files) {
    result.push_str(&format!(". {item}\n"));
  }

  Ok(result)
}

#[cfg(test)]
mod tests {
  use std::path::PathBuf;

  use super::*;

  /// A freshly created directory under the OS temp dir that no other test can collide with.
  fn unique_temp_dir(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("agent-test-{label}-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir).unwrap();
    dir
  }

  #[test]
  fn rejects_malformed_json() {
    let err = run("not json").unwrap_err();
    assert!(err.to_string().contains("invalid file read arguments"));
  }

  #[test]
  fn rejects_missing_path() {
    let path = std::env::temp_dir().join(format!("agent-test-missing-{}", uuid::Uuid::new_v4()));
    let args = format!(r#"{{"path":"{}"}}"#, path.display());
    let err = run(&args).unwrap_err();
    assert!(err.to_string().contains("Path not found"));
  }

  #[test]
  fn rejects_a_path_that_is_a_file() {
    let dir = unique_temp_dir("list-not-dir");
    let file = dir.join("a.txt");
    fs::write(&file, b"hello").unwrap();

    let args = format!(r#"{{"path":"{}"}}"#, file.display());
    let err = run(&args).unwrap_err();
    assert!(err.to_string().contains("Not a directory"));
  }

  #[test]
  fn lists_directories_before_files_and_skips_hidden_entries() {
    let dir = unique_temp_dir("list-contents");
    fs::create_dir_all(dir.join("b_dir")).unwrap();
    fs::write(dir.join("a_file.txt"), b"hello").unwrap();
    fs::write(dir.join(".hidden"), b"secret").unwrap();

    let args = format!(r#"{{"path":"{}"}}"#, dir.display());
    let output = run(&args).unwrap();

    let dir_pos = output.find("b_dir/").expect("directory should be listed");
    let file_pos = output.find("a_file.txt").expect("file should be listed");
    assert!(dir_pos < file_pos, "directories must be listed first");
    assert!(
      !output.contains(".hidden"),
      "hidden entries must be skipped"
    );
  }

  #[test]
  fn defaults_path_to_current_directory_when_omitted() {
    let output = run("{}").unwrap();
    assert!(output.starts_with("Directory: ."));
  }
}
