use std::path::Path;

use anyhow::Context;

use crate::tools::{file_edit::EditFileArgs, file_write::execute::write_atomic};

/// Replaces `old_string` with `new_string` in `file_path`.
///
/// Every failure mode is an error the model can act on, and none of them touches the
/// file: an empty or no-op edit, a missing file, no match, or (without `replace_all`)
/// more than one match. Guessing which of several matches was meant is exactly how an
/// edit tool corrupts a file.
///
/// The write itself is atomic (see [`write_atomic`]), so a crash mid-edit cannot leave a
/// truncated file. It is *not* safe against a second edit to the same file running at the
/// same moment — both would read the original and the later write would win. That is
/// prevented one level up, by this tool not being
/// [`concurrency_safe`](crate::tools::Tool::concurrency_safe): edits within a round take
/// turns.
pub async fn run(args_json: &str) -> anyhow::Result<String> {
  let EditFileArgs {
    file_path,
    old_string,
    new_string,
    replace_all,
  } = serde_json::from_str::<EditFileArgs>(args_json).context("invalid file edit arguments")?;

  anyhow::ensure!(
    !old_string.is_empty(),
    "old_string must not be empty; use write_file to create a file"
  );
  anyhow::ensure!(
    old_string != new_string,
    "old_string and new_string are identical; nothing to change"
  );

  let content = tokio::fs::read_to_string(&file_path)
    .await
    .with_context(|| format!("failed to read `{file_path}`"))?;

  let occurrences = content.matches(old_string.as_str()).count();
  match occurrences {
    0 => anyhow::bail!(
      "old_string was not found in `{file_path}`; read the file again and copy the text \
       exactly, including whitespace"
    ),
    n if n > 1 && !replace_all => anyhow::bail!(
      "old_string occurs {n} times in `{file_path}`; include more surrounding context to \
       make it unique, or set replace_all to true"
    ),
    _ => {}
  }

  let updated = if replace_all {
    content.replace(old_string.as_str(), &new_string)
  } else {
    content.replacen(old_string.as_str(), &new_string, 1)
  };
  write_atomic(Path::new(&file_path), updated.as_bytes())
    .await
    .with_context(|| format!("failed to write `{file_path}`"))?;

  let replaced = if replace_all { occurrences } else { 1 };
  Ok(format!(
    "Replaced {replaced} occurrence{} in {file_path}",
    if replaced == 1 { "" } else { "s" }
  ))
}

#[cfg(test)]
mod tests {
  use std::path::PathBuf;

  use serde_json::json;

  use super::*;
  use crate::tools::testing::TempDir;

  /// The file lives inside the returned directory, which removes it again on drop.
  fn temp_file(label: &str, content: &str) -> (TempDir, PathBuf) {
    let dir = TempDir::new(&format!("edit-{label}"));
    let path = dir.join("f.txt");
    std::fs::write(&path, content).unwrap();
    (dir, path)
  }

  fn args(path: &Path, old: &str, new: &str, replace_all: bool) -> String {
    json!({
      "file_path": path.to_string_lossy(),
      "old_string": old,
      "new_string": new,
      "replace_all": replace_all,
    })
    .to_string()
  }

  #[tokio::test]
  async fn replaces_a_unique_match() {
    let (_dir, path) = temp_file("unique", "fn a() {}\nfn b() {}\n");
    let output = run(&args(&path, "fn b() {}", "fn c() {}", false))
      .await
      .unwrap();

    assert_eq!(
      std::fs::read_to_string(&path).unwrap(),
      "fn a() {}\nfn c() {}\n"
    );
    assert_eq!(
      output,
      format!("Replaced 1 occurrence in {}", path.display())
    );
  }

  #[tokio::test]
  async fn rejects_an_ambiguous_match_without_touching_the_file() {
    let (_dir, path) = temp_file("ambiguous", "x\nx\n");
    let err = run(&args(&path, "x", "y", false)).await.unwrap_err();

    assert!(err.to_string().contains("occurs 2 times"), "got: {err}");
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "x\nx\n");
  }

  #[tokio::test]
  async fn replace_all_replaces_every_match() {
    let (_dir, path) = temp_file("all", "x\nx\n");
    let output = run(&args(&path, "x", "y", true)).await.unwrap();

    assert_eq!(std::fs::read_to_string(&path).unwrap(), "y\ny\n");
    assert!(output.contains("2 occurrences"), "got: {output}");
  }

  #[tokio::test]
  async fn reports_a_missing_match() {
    let (_dir, path) = temp_file("missing", "hello");
    let err = run(&args(&path, "absent", "y", false)).await.unwrap_err();
    assert!(err.to_string().contains("not found"), "got: {err}");
  }

  #[tokio::test]
  async fn rejects_an_empty_old_string() {
    let (_dir, path) = temp_file("empty", "hello");
    let err = run(&args(&path, "", "y", false)).await.unwrap_err();
    assert!(err.to_string().contains("must not be empty"), "got: {err}");
  }

  #[tokio::test]
  async fn rejects_a_no_op_edit() {
    let (_dir, path) = temp_file("noop", "hello");
    let err = run(&args(&path, "hello", "hello", false))
      .await
      .unwrap_err();
    assert!(err.to_string().contains("identical"), "got: {err}");
  }

  #[tokio::test]
  async fn reports_a_missing_file() {
    let dir = TempDir::new("edit-nofile");
    let err = run(&args(&dir.join("absent.txt"), "a", "b", false))
      .await
      .unwrap_err();
    assert!(err.to_string().contains("failed to read"), "got: {err}");
  }

  /// An edit goes through the atomic writer, so it keeps the file's mode and leaves no
  /// temporary file behind.
  #[cfg(unix)]
  #[tokio::test]
  async fn an_edit_keeps_the_mode_and_leaves_no_temporary_file() {
    use std::os::unix::fs::PermissionsExt;

    let (dir, path) = temp_file("mode", "echo a\n");
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o750)).unwrap();

    run(&args(&path, "echo a", "echo b", false)).await.unwrap();

    assert_eq!(std::fs::read_to_string(&path).unwrap(), "echo b\n");
    assert_eq!(
      std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
      0o750
    );
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
  }
}
