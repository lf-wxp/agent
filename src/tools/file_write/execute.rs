use std::{io, path::Path};

use tokio::io::AsyncWriteExt;

use crate::{error::ResultExt, tools::file_write::WriteFileArgs};

/// Writes `content` to `file_path`, creating parent directories first.
///
/// Refuses to write over a directory: `tokio::fs::write` would fail with an OS error
/// anyway, but a message that names the actual problem lets the model correct itself.
pub async fn run(args_json: &str) -> crate::Result<String> {
  let WriteFileArgs { file_path, content } = serde_json::from_str::<WriteFileArgs>(args_json)
    .wrap_err(crate::Error::tool, "invalid file write arguments")?;
  let path = Path::new(&file_path);

  if tokio::fs::metadata(path)
    .await
    .is_ok_and(|meta| meta.is_dir())
  {
    crate::error::bail!(tool, "`{file_path}` is a directory, not a file");
  }
  let existed = tokio::fs::try_exists(path).await.unwrap_or(false);

  if let Some(parent) = path
    .parent()
    .filter(|parent| !parent.as_os_str().is_empty())
  {
    tokio::fs::create_dir_all(parent)
      .await
      .wrap_err_with(crate::Error::tool, || {
        format!("failed to create parent directory `{}`", parent.display())
      })?;
  }
  write_atomic(path, content.as_bytes())
    .await
    .wrap_err_with(crate::Error::tool, || {
      format!("failed to write `{file_path}`")
    })?;

  let action = if existed { "Overwrote" } else { "Created" };
  Ok(format!(
    "{action} {file_path} ({} bytes, {} lines)",
    content.len(),
    content.lines().count()
  ))
}

/// Replaces the contents of `path` so that a reader — or a crash — sees either all of the
/// old content or all of the new, never a truncated file.
///
/// `tokio::fs::write` truncates first and writes second; a process killed between the two
/// leaves a half-written (or empty) file where a working one was. This writes a sibling
/// temporary file and renames it over the target, which the filesystem does in one step.
///
/// What it preserves, because replacing a file by rename would otherwise change them:
///
/// - **Permissions** of an existing file (an executable script stays executable).
/// - **Symlinks**: the file the link points at is replaced, not the link. A rename onto
///   the link itself would silently turn it into a regular file.
///
/// What it does not: ownership, extended attributes and hard links of the original — the
/// result is a new file. That is the ordinary trade of an atomic save, and is why editors
/// that do this offer a way to opt out; none is offered here because the target is a
/// workspace file a model is rewriting, not a system file.
pub(crate) async fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
  let target = match tokio::fs::canonicalize(path).await {
    Ok(real) => real,
    // A link pointing nowhere cannot be resolved, but must still be written *through*
    // (creating what it points at), which only a plain write does.
    Err(_)
      if tokio::fs::symlink_metadata(path)
        .await
        .is_ok_and(|meta| meta.is_symlink()) =>
    {
      return tokio::fs::write(path, bytes).await;
    }
    Err(_) => path.to_path_buf(),
  };
  let permissions = tokio::fs::metadata(&target)
    .await
    .ok()
    .map(|meta| meta.permissions());

  let dir = target
    .parent()
    .filter(|dir| !dir.as_os_str().is_empty())
    .unwrap_or_else(|| Path::new("."));
  // Hidden, and named after the target so a leftover (a crash between write and rename)
  // says what it was for. Truncated: a long file name plus the suffix could exceed the
  // filesystem's limit and fail a write that would otherwise have worked.
  let stem: String = target
    .file_name()
    .map(|name| name.to_string_lossy().chars().take(100).collect())
    .unwrap_or_default();
  let temp = dir.join(format!(".{stem}.{}.tmp", uuid::Uuid::new_v4()));

  let result = async {
    write_temp(&temp, bytes, permissions.as_ref()).await?;
    // The mode-at-create already covers the permission bits; this only restores what a
    // creation mode cannot express (setuid/setgid/sticky), exactly.
    if let Some(permissions) = permissions {
      tokio::fs::set_permissions(&temp, permissions).await?;
    }
    tokio::fs::rename(&temp, &target).await
  }
  .await;

  if result.is_err() {
    let _ = tokio::fs::remove_file(&temp).await;
  }
  result
}

/// Writes `bytes` to `temp`, applying the target's permissions at creation time on Unix
/// so a sensitive file (e.g. mode `0o600`) is never briefly world-readable before a later
/// [`std::fs::set_permissions`] fixes it.
///
/// `create_new` rather than `create`+`truncate`: the uuid-named path should not exist,
/// and refusing to open an existing one means a stray file (or a link placed there) is
/// not silently written through.
async fn write_temp(
  temp: &Path,
  bytes: &[u8],
  permissions: Option<&std::fs::Permissions>,
) -> io::Result<()> {
  let mut options = tokio::fs::OpenOptions::new();
  options.write(true).create_new(true);
  #[cfg(unix)]
  if let Some(permissions) = permissions {
    use std::os::unix::fs::PermissionsExt;
    options.mode(permissions.mode() & 0o7777);
  }
  #[cfg(not(unix))]
  let _ = permissions;
  let mut file = options.open(temp).await?;
  AsyncWriteExt::write_all(&mut file, bytes).await?;
  Ok(())
}

#[cfg(test)]
mod tests {
  use serde_json::json;

  use super::*;
  use crate::tools::testing::TempDir;

  fn args(path: &Path, content: &str) -> String {
    json!({ "file_path": path.to_string_lossy(), "content": content }).to_string()
  }

  #[tokio::test]
  async fn rejects_malformed_json() {
    let err = run("not json").await.unwrap_err();
    assert!(err.to_string().contains("invalid file write arguments"));
  }

  #[tokio::test]
  async fn creates_a_new_file_and_its_parent_directories() {
    let dir = TempDir::new("write-nested");
    let path = dir.join("a/b/new.txt");
    let output = run(&args(&path, "hello\nworld\n")).await.unwrap();

    assert_eq!(std::fs::read_to_string(&path).unwrap(), "hello\nworld\n");
    assert!(output.starts_with("Created"), "got: {output}");
    assert!(output.contains("2 lines"), "got: {output}");
  }

  #[tokio::test]
  async fn overwrites_an_existing_file() {
    let dir = TempDir::new("write-overwrite");
    let path = dir.join("f.txt");
    std::fs::write(&path, "old").unwrap();

    let output = run(&args(&path, "new")).await.unwrap();

    assert_eq!(std::fs::read_to_string(&path).unwrap(), "new");
    assert!(output.starts_with("Overwrote"), "got: {output}");
  }

  #[tokio::test]
  async fn refuses_to_write_over_a_directory() {
    let dir = TempDir::new("write-dir");

    let err = run(&args(dir.path(), "x")).await.unwrap_err();
    assert!(err.to_string().contains("is a directory"), "got: {err}");
  }

  /// A finished write leaves nothing but the file.
  #[tokio::test]
  async fn leaves_no_temporary_file_behind() {
    let dir = TempDir::new("write-clean");
    let path = dir.join("f.txt");
    run(&args(&path, "one")).await.unwrap();
    run(&args(&path, "two")).await.unwrap();

    let names: Vec<_> = std::fs::read_dir(dir.path())
      .unwrap()
      .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
      .collect();
    assert_eq!(names, vec!["f.txt".to_owned()]);
  }

  #[cfg(unix)]
  #[tokio::test]
  async fn keeps_the_permissions_of_an_overwritten_file() {
    use std::os::unix::fs::PermissionsExt;

    let dir = TempDir::new("write-mode");
    let path = dir.join("run.sh");
    std::fs::write(&path, "#!/bin/sh\n").unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o750)).unwrap();

    run(&args(&path, "#!/bin/sh\necho hi\n")).await.unwrap();

    let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o750);
  }

  /// Writing through a link replaces what it points at; the link stays a link.
  #[cfg(unix)]
  #[tokio::test]
  async fn writes_through_a_symlink_instead_of_replacing_it() {
    let dir = TempDir::new("write-link");
    let real = dir.join("real.txt");
    let link = dir.join("link.txt");
    std::fs::write(&real, "old").unwrap();
    std::os::unix::fs::symlink(&real, &link).unwrap();

    run(&args(&link, "new")).await.unwrap();

    assert!(std::fs::symlink_metadata(&link).unwrap().is_symlink());
    assert_eq!(std::fs::read_to_string(&real).unwrap(), "new");
  }

  /// A link to a file that does not exist yet is still written through, creating it.
  #[cfg(unix)]
  #[tokio::test]
  async fn writes_through_a_dangling_symlink() {
    let dir = TempDir::new("write-dangling");
    let real = dir.join("not-yet.txt");
    let link = dir.join("link.txt");
    std::os::unix::fs::symlink(&real, &link).unwrap();

    run(&args(&link, "created")).await.unwrap();

    assert!(std::fs::symlink_metadata(&link).unwrap().is_symlink());
    assert_eq!(std::fs::read_to_string(&real).unwrap(), "created");
  }

  #[tokio::test]
  async fn a_failed_write_leaves_the_original_and_no_temporary_file() {
    let dir = TempDir::new("write-fail");
    let path = dir.join("f.txt");
    std::fs::write(&path, "original").unwrap();
    // A directory where the rename must land, so the final step fails.
    let target = dir.join("blocked");
    std::fs::create_dir(&target).unwrap();
    std::fs::write(target.join("inner"), "").unwrap();

    let err = write_atomic(&target, b"data").await;
    assert!(err.is_err());

    assert_eq!(std::fs::read_to_string(&path).unwrap(), "original");
    let leftovers: Vec<_> = std::fs::read_dir(dir.path())
      .unwrap()
      .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
      .filter(|name| name.ends_with(".tmp"))
      .collect();
    assert!(leftovers.is_empty(), "left behind: {leftovers:?}");
  }
}
