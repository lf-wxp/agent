use std::{
  fs, io,
  path::{Path, PathBuf},
};

use anyhow::Context;

use crate::tools::file_upzip::UnzipFileArgs;

/// Extracts every entry of `zip_path` into `extract_to`, creating it if needed, and
/// returns a summary listing (capped at 20 names) of what was written.
///
/// Caveat: `entry.name()` from the archive is joined onto `extract_to` as-is, so a
/// malicious archive with a `../`-prefixed entry name ("zip slip") could in principle
/// write outside `extract_to`. Not sanitized here because this tool is meant for
/// archives the caller already trusts (its own downloads/attachments), not arbitrary
/// third-party zips — treat it the same way you would treat running an installer.
pub fn run(args_json: &str) -> anyhow::Result<String> {
  let UnzipFileArgs {
    zip_path,
    extract_to,
  } = serde_json::from_str::<UnzipFileArgs>(args_json).context("invalid file unzip arguments")?;
  let zip_path = Path::new(&zip_path);
  if !zip_path.exists() {
    anyhow::bail!("File not found: {}", zip_path.display());
  }

  let extract_to: PathBuf = match extract_to {
    Some(dir) => PathBuf::from(dir),
    None => zip_path.with_extension(""),
  };
  fs::create_dir_all(&extract_to)?;

  let file = fs::File::open(zip_path)?;
  let mut archive = zip::ZipArchive::new(file)?;

  let mut names = Vec::with_capacity(archive.len());
  for i in 0..archive.len() {
    let mut entry = archive.by_index(i)?;
    let out_path = extract_to.join(entry.name());
    names.push(entry.name().to_string());

    if entry.is_dir() {
      fs::create_dir_all(&out_path)?;
    } else {
      if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)?;
      }
      let mut out_file = fs::File::create(&out_path)?;
      io::copy(&mut entry, &mut out_file)?;
    }
  }

  let mut summary = format!(
    "Extracted {} files to {}/\n\nContents:\n",
    names.len(),
    extract_to.display()
  );
  for name in names.iter().take(20) {
    summary.push_str(&format!(". - {name}\n"));
  }
  if names.len() > 20 {
    summary.push_str(&format!("  ... and {} more files\n", names.len() - 20));
  }

  Ok(summary)
}

#[cfg(test)]
mod tests {
  use std::io::Write;

  use super::*;

  fn unique_temp_path(label: &str, extension: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
      "agent-test-{label}-{}.{extension}",
      uuid::Uuid::new_v4()
    ))
  }

  /// Builds a zip archive at `path` containing a single `entry.txt` with `contents`.
  fn write_zip_with_one_entry(path: &Path, contents: &[u8]) {
    let file = fs::File::create(path).unwrap();
    let mut zip = zip::ZipWriter::new(file);
    zip
      .start_file("entry.txt", zip::write::SimpleFileOptions::default())
      .unwrap();
    zip.write_all(contents).unwrap();
    zip.finish().unwrap();
  }

  #[test]
  fn rejects_malformed_json() {
    let err = run("not json").unwrap_err();
    assert!(err.to_string().contains("invalid file unzip arguments"));
  }

  #[test]
  fn rejects_missing_zip() {
    let path = unique_temp_path("missing", "zip");
    let args = format!(r#"{{"zip_path":"{}"}}"#, path.display());
    let err = run(&args).unwrap_err();
    assert!(err.to_string().contains("File not found"));
  }

  #[test]
  fn extracts_into_the_requested_directory() {
    let zip_path = unique_temp_path("archive", "zip");
    write_zip_with_one_entry(&zip_path, b"hello world");
    let extract_to = unique_temp_path("extracted", "dir");

    let args = format!(
      r#"{{"zip_path":"{}","extract_to":"{}"}}"#,
      zip_path.display(),
      extract_to.display()
    );
    let summary = run(&args).unwrap();

    assert!(summary.contains("Extracted 1 files"));
    assert!(summary.contains("entry.txt"));
    let extracted_content = fs::read_to_string(extract_to.join("entry.txt")).unwrap();
    assert_eq!(extracted_content, "hello world");
  }

  #[test]
  fn defaults_extract_dir_to_the_zip_name_without_extension() {
    let zip_path = unique_temp_path("archive-default", "zip");
    write_zip_with_one_entry(&zip_path, b"hi");

    let args = format!(r#"{{"zip_path":"{}"}}"#, zip_path.display());
    run(&args).unwrap();

    let expected_dir = zip_path.with_extension("");
    assert!(expected_dir.join("entry.txt").exists());
  }
}
