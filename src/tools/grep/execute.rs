use std::{
  fmt::Write as _,
  fs::File,
  io::{BufRead, BufReader},
  path::Path,
  sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
  },
};

use regex::{Regex, RegexBuilder};
use walkdir::{DirEntry, WalkDir};

use crate::{
  error::ResultExt,
  tools::grep::{DEFAULT_MAX_RESULTS, GrepArgs, MAX_RESULTS_CAP},
  util::truncate_chars,
};

/// Directories never descended into: build output and dependency trees are huge and
/// almost never what the model is looking for.
const SKIPPED_DIRS: &[&str] = &["target", "node_modules"];

/// Files larger than this are skipped rather than scanned.
const MAX_FILE_BYTES: u64 = 2 * 1024 * 1024;

/// Chars of one matching line echoed back; minified files have enormous lines.
const MAX_LINE_CHARS: usize = 300;

/// Searches `path` for `pattern`, one output line per matching line.
///
/// The walk is synchronous (`walkdir` + `std::fs`), so it runs on the blocking pool
/// rather than stalling the async runtime for the duration of a large tree.
///
/// A blocking task cannot be dropped the way an async one can, so the search checks a
/// flag as it goes ([`CancelOnDrop`]): when the tool call is abandoned — it timed out, or
/// the run was cancelled — the future is dropped, the flag is set, and the thread stops
/// at its next file instead of walking the rest of the tree for nobody.
pub async fn run(args_json: &str) -> crate::Result<String> {
  let args = serde_json::from_str::<GrepArgs>(args_json)
    .wrap_err(crate::Error::tool, "invalid grep arguments")?;

  let cancelled = Arc::new(AtomicBool::new(false));
  let _stop_on_drop = CancelOnDrop(Arc::clone(&cancelled));
  tokio::task::spawn_blocking(move || search(args, &cancelled))
    .await
    .wrap_err(crate::Error::tool, "grep task panicked")?
}

/// Sets its flag when dropped; see [`run`].
struct CancelOnDrop(Arc<AtomicBool>);

impl Drop for CancelOnDrop {
  fn drop(&mut self) {
    self.0.store(true, Ordering::Relaxed);
  }
}

fn search(args: GrepArgs, cancelled: &AtomicBool) -> crate::Result<String> {
  let GrepArgs {
    pattern,
    path,
    include,
    case_insensitive,
    max_results,
  } = args;

  let regex = RegexBuilder::new(&pattern)
    .case_insensitive(case_insensitive)
    .build()
    .wrap_err_with(crate::Error::tool, || format!("invalid regex `{pattern}`"))?;
  let include = include
    .filter(|glob| !glob.is_empty())
    .map(|glob| glob_to_regex(&glob))
    .transpose()?;
  let limit = max_results
    .unwrap_or(DEFAULT_MAX_RESULTS)
    .clamp(1, MAX_RESULTS_CAP);

  let root = Path::new(&path);
  crate::error::ensure!(root.exists(), tool, "Path not found: {path}");

  let mut output = String::new();
  let mut matches = 0usize;
  let mut truncated = false;
  let mut too_large = 0usize;
  // Reused for every line of every file, instead of one `String` per line.
  let mut line = Vec::new();

  let walker = WalkDir::new(root)
    .sort_by_file_name()
    .into_iter()
    .filter_entry(|entry| entry.depth() == 0 || !is_skipped(entry));

  'files: for entry in walker.filter_map(Result::ok) {
    if cancelled.load(Ordering::Relaxed) {
      crate::error::bail!(tool, "search cancelled");
    }
    if !entry.file_type().is_file() {
      continue;
    }
    if let Some(include) = &include {
      let name = entry.file_name().to_string_lossy();
      if !include.is_match(&name) {
        continue;
      }
    }

    // One open serves the size check, the binary sniff and the read itself.
    let Ok(file) = File::open(entry.path()) else {
      continue;
    };
    if file
      .metadata()
      .is_ok_and(|meta| meta.len() > MAX_FILE_BYTES)
    {
      too_large += 1;
      continue;
    }
    let mut reader = BufReader::new(file);
    // A NUL in the first buffer's worth marks the file as binary. Peeking leaves the
    // bytes in place, so nothing is read twice.
    match reader.fill_buf() {
      Ok(head) if head.contains(&0) => continue,
      Ok(_) => {}
      Err(_) => continue,
    }

    let display = entry.path().display().to_string();
    let mut number = 0usize;
    loop {
      line.clear();
      match reader.read_until(b'\n', &mut line) {
        Ok(0) | Err(_) => break,
        Ok(_) => {}
      }
      number += 1;
      // A non-UTF-8 line ends the file: it is almost certainly not text after all.
      let Ok(text) = std::str::from_utf8(&line) else {
        break;
      };
      if !regex.is_match(text) {
        continue;
      }
      if matches == limit {
        truncated = true;
        break 'files;
      }
      matches += 1;
      let _ = writeln!(
        output,
        "{display}:{number}: {}",
        truncate_chars(text.trim_end(), MAX_LINE_CHARS)
      );
    }
  }

  if matches == 0 {
    output = format!("No matches for `{pattern}` in {path}\n");
  }
  if truncated {
    let _ = writeln!(
      output,
      "[stopped after {limit} matches; narrow the pattern, path or include to see more]"
    );
  }
  // Said out loud: a file that was skipped looks exactly like a file with no match, and
  // "no matches" from a search that quietly passed over the one big file is misleading.
  if too_large > 0 {
    let _ = writeln!(
      output,
      "[skipped {too_large} file(s) over {} MiB; read them directly or narrow the path]",
      MAX_FILE_BYTES / (1024 * 1024)
    );
  }
  Ok(output)
}

/// Hidden entries (`.git`, `.venv`, `.env`, ...) and [`SKIPPED_DIRS`]. Only applied
/// below the root, so explicitly searching `.github` — or one named dotfile — still
/// works.
///
/// Hidden *files* are skipped as well as hidden directories, matching `list_files`
/// (which never shows them): the usual contents of a dotfile are credentials and local
/// configuration, and `grep API_KEY` would otherwise print the value straight out of
/// `.env` into the model's context.
fn is_skipped(entry: &DirEntry) -> bool {
  let name = entry.file_name().to_string_lossy();
  if name.starts_with('.') {
    return true;
  }
  entry.file_type().is_dir() && SKIPPED_DIRS.contains(&name.as_ref())
}

/// Translate a filename glob (`*`, `?`, everything else literal) into an anchored regex.
fn glob_to_regex(glob: &str) -> crate::Result<Regex> {
  let mut pattern = String::from("^");
  for ch in glob.chars() {
    match ch {
      '*' => pattern.push_str(".*"),
      '?' => pattern.push('.'),
      // Escape regex metacharacters one at a time, rather than `regex::escape`-ing each
      // char through a temporary `String`. `*` and `?` are handled above; the rest is
      // regex's own metacharacter set (outside a character class).
      other
        if matches!(
          other,
          '.' | '^' | '$' | '+' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '\\'
        ) =>
      {
        pattern.push('\\');
        pattern.push(other);
      }
      other => pattern.push(other),
    }
  }
  pattern.push('$');
  Regex::new(&pattern).wrap_err_with(crate::Error::tool, || {
    format!("invalid include glob `{glob}`")
  })
}

#[cfg(test)]
mod tests {
  use serde_json::json;

  use super::*;
  use crate::tools::testing::TempDir;

  /// A small tree: two Rust files, one text file, hidden dir / hidden file / `target`
  /// dir that must be skipped, and one binary file.
  fn fixture() -> TempDir {
    let root = TempDir::new("grep");
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::create_dir_all(root.join(".git")).unwrap();
    std::fs::create_dir_all(root.join("target")).unwrap();
    std::fs::write(root.join("src/a.rs"), "fn alpha() {}\nfn beta() {}\n").unwrap();
    std::fs::write(root.join("src/b.rs"), "// Alpha again\n").unwrap();
    std::fs::write(root.join("notes.txt"), "alpha in text\n").unwrap();
    std::fs::write(root.join(".git/config"), "alpha hidden\n").unwrap();
    std::fs::write(root.join(".env"), "alpha=secret\n").unwrap();
    std::fs::write(root.join("target/out.rs"), "fn alpha() {}\n").unwrap();
    std::fs::write(root.join("blob.bin"), b"alpha\0binary").unwrap();
    root
  }

  async fn grep(value: serde_json::Value) -> crate::Result<String> {
    run(&value.to_string()).await
  }

  #[tokio::test]
  async fn finds_matches_and_skips_hidden_build_and_binary_files() {
    let root = fixture();
    let output = grep(json!({ "pattern": "alpha", "path": root.path().to_string_lossy() }))
      .await
      .unwrap();

    assert!(output.contains("a.rs:1: fn alpha() {}"), "got: {output}");
    assert!(
      output.contains("notes.txt:1: alpha in text"),
      "got: {output}"
    );
    assert!(!output.contains(".git"), "hidden dir searched: {output}");
    assert!(!output.contains(".env"), "hidden file searched: {output}");
    assert!(!output.contains("secret"), "leaked a dotfile: {output}");
    assert!(!output.contains("target"), "target dir searched: {output}");
    assert!(
      !output.contains("blob.bin"),
      "binary file searched: {output}"
    );
    assert!(
      !output.contains("b.rs"),
      "case-sensitive by default: {output}"
    );
  }

  /// Naming a hidden path outright is a deliberate act and is honoured.
  #[tokio::test]
  async fn a_hidden_path_named_explicitly_is_still_searched() {
    let root = fixture();

    let dir = grep(json!({ "pattern": "alpha", "path": root.join(".git").to_string_lossy() }))
      .await
      .unwrap();
    assert!(dir.contains("config:1: alpha hidden"), "got: {dir}");

    let file = grep(json!({ "pattern": "alpha", "path": root.join(".env").to_string_lossy() }))
      .await
      .unwrap();
    assert!(file.contains("alpha=secret"), "got: {file}");
  }

  #[tokio::test]
  async fn case_insensitive_and_include_filter() {
    let root = fixture();
    let output = grep(json!({
      "pattern": "alpha",
      "path": root.path().to_string_lossy(),
      "include": "*.rs",
      "case_insensitive": true,
    }))
    .await
    .unwrap();

    assert!(output.contains("a.rs:1:"), "got: {output}");
    assert!(output.contains("b.rs:1:"), "got: {output}");
    assert!(!output.contains("notes.txt"), "include ignored: {output}");
  }

  #[tokio::test]
  async fn caps_the_number_of_results() {
    let root = fixture();
    let output = grep(json!({
      "pattern": "fn",
      "path": root.join("src").to_string_lossy(),
      "max_results": 1,
    }))
    .await
    .unwrap();

    assert_eq!(
      output.lines().filter(|line| line.contains("a.rs:")).count(),
      1
    );
    assert!(output.contains("stopped after 1 matches"), "got: {output}");
  }

  #[tokio::test]
  async fn searches_a_single_file() {
    let root = fixture();
    let output =
      grep(json!({ "pattern": "beta", "path": root.join("src/a.rs").to_string_lossy() }))
        .await
        .unwrap();
    assert!(output.contains(":2: fn beta() {}"), "got: {output}");
  }

  #[tokio::test]
  async fn reports_no_matches() {
    let root = fixture();
    let output = grep(json!({ "pattern": "zzz_nothing", "path": root.path().to_string_lossy() }))
      .await
      .unwrap();
    assert!(output.starts_with("No matches"), "got: {output}");
  }

  /// A file passed over for size must be mentioned, or "no matches" reads as "checked
  /// everything and found nothing".
  #[tokio::test]
  async fn says_when_a_large_file_was_skipped() {
    let root = fixture();
    let big = vec![b'x'; (MAX_FILE_BYTES + 1) as usize];
    std::fs::write(root.join("huge.log"), big).unwrap();

    let output = grep(json!({ "pattern": "zzz_nothing", "path": root.path().to_string_lossy() }))
      .await
      .unwrap();
    assert!(output.starts_with("No matches"), "got: {output}");
    assert!(
      output.contains("skipped 1 file(s) over 2 MiB"),
      "got: {output}"
    );
  }

  #[tokio::test]
  async fn a_final_line_without_a_newline_is_still_searched() {
    let root = TempDir::new("grep-eof");
    std::fs::write(root.join("f.txt"), "one\ntwo needle").unwrap();
    let output = grep(json!({ "pattern": "needle", "path": root.path().to_string_lossy() }))
      .await
      .unwrap();
    assert!(output.contains("f.txt:2: two needle"), "got: {output}");
  }

  #[tokio::test]
  async fn rejects_an_invalid_regex() {
    let err = grep(json!({ "pattern": "(" })).await.unwrap_err();
    assert!(err.to_string().contains("invalid regex"), "got: {err}");
  }

  #[tokio::test]
  async fn rejects_a_missing_path() {
    let err = grep(json!({ "pattern": "x", "path": "/definitely/not/here" }))
      .await
      .unwrap_err();
    assert!(err.to_string().contains("Path not found"), "got: {err}");
  }

  /// The blocking thread must notice that nobody is waiting any more.
  #[test]
  fn a_cancelled_search_stops_walking() {
    let root = fixture();
    let args: GrepArgs = serde_json::from_value(json!({
      "pattern": "alpha",
      "path": root.path().to_string_lossy(),
    }))
    .unwrap();

    let err = search(args, &AtomicBool::new(true)).unwrap_err();
    assert!(err.to_string().contains("cancelled"), "got: {err}");
  }

  #[test]
  fn dropping_the_guard_raises_the_flag() {
    let flag = Arc::new(AtomicBool::new(false));
    drop(CancelOnDrop(Arc::clone(&flag)));
    assert!(flag.load(Ordering::Relaxed));
  }

  #[test]
  fn glob_matches_whole_file_names_only() {
    let glob = glob_to_regex("*.rs").unwrap();
    assert!(glob.is_match("main.rs"));
    assert!(!glob.is_match("main.rs.bak"));
    assert!(glob_to_regex("a?.txt").unwrap().is_match("ab.txt"));
    assert!(!glob_to_regex("a.txt").unwrap().is_match("abtxt"));
  }
}
