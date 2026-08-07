use std::{fs, path::Path};

use anyhow::{Context, Ok};

use crate::tools::file_read::ReadFileArgs;

/// Dispatches on the file extension: `.csv` renders as a markdown table, everything
/// else as line-numbered text.
pub fn run(args_json: &str) -> anyhow::Result<String> {
  let ReadFileArgs {
    file_path,
    end_line,
    start_line,
  } = serde_json::from_str::<ReadFileArgs>(args_json).context("invalid file read arguments")?;
  let path = Path::new(&file_path);
  if !path.exists() {
    anyhow::bail!("File not found: {}", file_path);
  }

  match path.extension().and_then(|ext| ext.to_str()) {
    Some("csv") => read_csv_as_markdown(path),
    _ => read_text_with_line_numbers(path, start_line, end_line),
  }
}

/// Renders `path` as `{line number} | {content}`, one line per row.
///
/// `start_line`/`end_line` are the 1-based, inclusive bounds from [`ReadFileArgs`];
/// they are clamped rather than validated, so an out-of-range or reversed range from
/// the model yields an empty (not an error) result — `.get(start..end)` on a slice
/// returns `None` for that case, handled below via `unwrap_or(&[])`.
fn read_text_with_line_numbers(
  path: &Path,
  start_line: usize,
  end_line: i64,
) -> anyhow::Result<String> {
  let content = fs::read_to_string(path)?;
  let lines: Vec<&str> = content.lines().collect();

  // `usize` line numbers are 1-based; the slice index below is 0-based.
  let start_index = start_line.saturating_sub(1);
  let end_index = if end_line < 0 {
    lines.len()
  } else {
    (end_line as usize).min(lines.len())
  };

  let mut result = String::new();
  for (offset, line) in lines
    .get(start_index..end_index)
    .unwrap_or(&[])
    .iter()
    .enumerate()
  {
    result.push_str(&format!("{:>4} | {}\n", start_index + offset + 1, line));
  }

  Ok(result)
}

/// Renders `path` as a GitHub-flavoured markdown table: header row, `---` separator,
/// then one row per record, in file order.
///
/// Uses the `csv` crate's default dialect (comma-delimited, `"`-quoted) rather than
/// sniffing the delimiter: `.csv` implies that dialect strongly enough that
/// autodetection would add complexity for a case that essentially never occurs.
fn read_csv_as_markdown(path: &Path) -> anyhow::Result<String> {
  let mut reader = csv::Reader::from_path(path)?;
  let headers = reader.headers()?.clone();

  let mut table = String::new();
  table.push_str(&format!(
    "| {} |\n",
    headers.iter().collect::<Vec<_>>().join(" | ")
  ));
  table.push_str(&format!("|{}\n", "---|".repeat(headers.len())));

  for record in reader.records() {
    let record = record?;
    table.push_str(&format!(
      "| {} |\n",
      record.iter().collect::<Vec<_>>().join(" | ")
    ));
  }

  Ok(table)
}

#[cfg(test)]
mod tests {
  use std::path::PathBuf;

  use super::*;

  /// A path under the OS temp dir that no other test can collide with.
  fn unique_temp_path(label: &str, extension: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
      "agent-test-{label}-{}.{extension}",
      uuid::Uuid::new_v4()
    ))
  }

  #[test]
  fn rejects_malformed_json() {
    let err = run("not json").unwrap_err();
    assert!(err.to_string().contains("invalid file read arguments"));
  }

  #[test]
  fn rejects_missing_file() {
    let path = unique_temp_path("missing", "txt");
    let args = format!(r#"{{"file_path":"{}"}}"#, path.display());
    let err = run(&args).unwrap_err();
    assert!(err.to_string().contains("File not found"));
  }

  #[test]
  fn reads_a_text_file_with_line_numbers_by_default() {
    let path = unique_temp_path("plain", "txt");
    fs::write(&path, "a\nb\nc\n").unwrap();

    let args = format!(r#"{{"file_path":"{}"}}"#, path.display());
    let output = run(&args).unwrap();

    assert_eq!(output, "   1 | a\n   2 | b\n   3 | c\n");
  }

  #[test]
  fn reads_only_the_requested_line_range() {
    let path = unique_temp_path("range", "txt");
    fs::write(&path, "a\nb\nc\nd\ne\n").unwrap();

    let args = format!(
      r#"{{"file_path":"{}","start_line":2,"end_line":4}}"#,
      path.display()
    );
    let output = run(&args).unwrap();

    assert_eq!(output, "   2 | b\n   3 | c\n   4 | d\n");
  }

  #[test]
  fn reads_a_csv_file_as_a_markdown_table() {
    let path = unique_temp_path("table", "csv");
    fs::write(&path, "a,b\n1,2\n3,4\n").unwrap();

    let args = format!(r#"{{"file_path":"{}"}}"#, path.display());
    let output = run(&args).unwrap();

    assert_eq!(output, "| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n");
  }
}
