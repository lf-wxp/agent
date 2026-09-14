use std::path::{Component, Path, PathBuf};

use anyhow::Context;
use serde_json::Value;

use crate::{
  agent::{
    ExecutionContext,
    callback::{BeforeToolCallback, ToolCallDecision, ToolCallView},
  },
  tools::{file_delete, file_list, file_read, file_upzip, read_image},
};

/// Denies a built-in filesystem tool call whose path argument resolves outside a fixed
/// `root`, before the call ever reaches the real tool.
///
/// None of the filesystem tools (`delete_file`, `read_file`, `list_files`,
/// `unzip_file`) sandbox themselves — see e.g. [`file_delete`]'s module docs, "not
/// sandboxed to any project root" — because a library caller may legitimately want the
/// whole filesystem. An interactive CLI is a different story: a human approves a call by
/// reading a path string, not by mentally resolving where it points, and a model (or a
/// typo) producing `../../etc` or a stray absolute path is exactly the kind of mistake
/// this exists to catch before it does anything — including before an
/// [`ApprovalCallback`](crate::callback::approval::ApprovalCallback) prompt for it ever
/// asks a human to weigh in, if this is registered first in the before-tool chain (see
/// [`crate::agent::Agent::with_before_tool_callback`]'s ordering).
///
/// # Known limitations
///
/// - **Built-in tools only**: an MCP server's tools are discovered at runtime with no
///   fixed argument schema this callback could rely on, so a filesystem-touching MCP
///   tool is not covered by *this* boundary — only enable MCP servers you already trust
///   with full filesystem access (see [`crate::tools::ToolRegistry::with_mcp`]).
///   [`crate::callback::mcp_guard::McpGuardCallback`] covers MCP tools specifically, but
///   with a narrower, schema-agnostic check (known credential paths, not an arbitrary
///   workspace root) — the two are meant to be registered together, not as
///   alternatives.
/// - **A symlink inside a not-yet-existing suffix is not resolved**: [`resolve_best_effort`]
///   canonicalizes as much of a path as actually exists on disk — including resolving
///   any symlink along *that* prefix — but a symlink that is itself part of the
///   remaining, not-yet-created suffix (e.g. `unzip_file`'s `extract_to` before
///   extraction creates it) cannot be inspected, since there is nothing on disk yet to
///   read.
/// - **TOCTOU on an existing symlink**: [`Self::violation`] and the tool's own
///   filesystem access are two separate syscalls with no lock between them — a symlink
///   that resolves inside the root when this callback checks it could, in principle, be
///   repointed outside the root by another process before the guarded tool actually
///   reads/writes through it. Acceptable for the single-user, single-machine CLI this is
///   built for; a multi-tenant or otherwise adversarial deployment would need an
///   open-and-verify (e.g. `openat2` with `RESOLVE_BENEATH`) approach instead.
pub struct WorkspaceGuardCallback {
  root: PathBuf,
}

impl WorkspaceGuardCallback {
  /// `root` must already exist: it is canonicalized once here (resolving symlinks and
  /// `.`/`..`) so every check below compares against one stable, absolute path instead
  /// of re-resolving it — and potentially getting a different answer if the filesystem
  /// changes mid-run — on every call.
  pub fn new(root: impl AsRef<Path>) -> anyhow::Result<Self> {
    let root = root.as_ref();
    let canonical = std::fs::canonicalize(root)
      .with_context(|| format!("workspace root `{}` does not exist", root.display()))?;
    Ok(Self { root: canonical })
  }

  /// The path-bearing argument name(s) to check for a guarded tool, or an empty slice
  /// for anything else this callback leaves untouched (including every MCP tool — see
  /// the type docs).
  fn guarded_fields(tool_name: &str) -> &'static [&'static str] {
    if tool_name == file_delete::NAME
      || tool_name == file_read::NAME
      || tool_name == read_image::NAME
    {
      &["file_path"]
    } else if tool_name == file_list::NAME {
      &["path"]
    } else if tool_name == file_upzip::NAME {
      &["zip_path", "extract_to"]
    } else {
      &[]
    }
  }

  /// `None` if `candidate` — resolved against `self.root` when relative, taken as-is
  /// when absolute — stays inside the workspace; `Some(reason)` naming the resolved
  /// path otherwise.
  fn violation(&self, candidate: &str) -> Option<String> {
    let requested = Path::new(candidate);
    let joined = if requested.is_absolute() {
      requested.to_path_buf()
    } else {
      self.root.join(requested)
    };

    let resolved = resolve_best_effort(&joined);

    if resolved.starts_with(&self.root) {
      None
    } else {
      Some(format!(
        "`{candidate}` resolves to `{}`, outside the workspace root `{}`",
        resolved.display(),
        self.root.display(),
      ))
    }
  }
}

#[async_trait::async_trait]
impl BeforeToolCallback for WorkspaceGuardCallback {
  async fn call(
    &self,
    _context: &ExecutionContext,
    tool_call: ToolCallView<'_>,
  ) -> ToolCallDecision {
    for field in Self::guarded_fields(tool_call.name) {
      // A missing or non-string field is left to the tool's own argument parsing to
      // reject; this callback only rules on a path it can actually read.
      let Some(candidate) = tool_call.arguments.get(*field).and_then(Value::as_str) else {
        continue;
      };
      if let Some(reason) = self.violation(candidate) {
        tracing::warn!(
          tool = tool_call.name,
          field = *field,
          %reason,
          "blocked a filesystem call outside the workspace root"
        );
        return ToolCallDecision::deny(format!(
          "Denied: {reason}. This tool is restricted to the workspace root; ask for a \
           path inside it."
        ));
      }
    }
    ToolCallDecision::Proceed
  }
}

/// Resolve `.`/`..` components without touching the filesystem: for a path that does
/// not exist yet, [`std::fs::canonicalize`] cannot be used at all, but a component-wise
/// resolution is still enough to catch a plain `..`-escape.
///
/// `pub(crate)` so [`crate::callback::mcp_guard`] can reuse the exact same escape logic
/// for its own, differently-scoped check instead of re-implementing it.
pub(crate) fn lexical_normalize(path: &Path) -> PathBuf {
  let mut out = PathBuf::new();
  for component in path.components() {
    match component {
      Component::ParentDir => {
        out.pop();
      }
      Component::CurDir => {}
      other => out.push(other),
    }
  }
  out
}

/// Best-effort canonical form of `path`, tolerating a suffix that does not exist yet
/// (unlike [`std::fs::canonicalize`], which requires every component to exist).
///
/// First lexically resolves every `.`/`..` in `path` regardless of existence (via
/// [`lexical_normalize`]), so an escape is caught even inside a path nothing on disk
/// backs yet. Then canonicalizes the *longest prefix of that which actually exists* —
/// resolving any symlink along it — and reattaches the remaining, not-yet-existing
/// suffix components unresolved (see the type docs' "Known limitations" for what that
/// last step cannot see).
///
/// `pub(crate)` for the same reason as [`lexical_normalize`]: [`crate::callback::
/// mcp_guard`] resolves candidate paths the same best-effort way, just against a
/// different set of denied roots.
pub(crate) fn resolve_best_effort(path: &Path) -> PathBuf {
  let normalized = lexical_normalize(path);
  let components: Vec<Component> = normalized.components().collect();

  for split in (1..=components.len()).rev() {
    let prefix: PathBuf = components[..split].iter().copied().collect();
    if let Ok(canonical) = std::fs::canonicalize(&prefix) {
      let suffix: PathBuf = components[split..].iter().copied().collect();
      return canonical.join(suffix);
    }
  }
  normalized
}

#[cfg(test)]
mod tests {
  use serde_json::json;

  use super::*;
  use crate::agent::ToolResultStatus;
  use crate::tools::ToolRegistry;

  /// A freshly created directory under the OS temp dir that no other test can collide
  /// with; not cleaned up afterwards, same tradeoff as `agent::session`'s tests.
  fn unique_temp_dir(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
      "agent-path-guard-test-{label}-{}",
      uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
  }

  fn view<'a>(name: &'a str, arguments: &'a Value) -> ToolCallView<'a> {
    ToolCallView {
      tool_call_id: "call-1",
      name,
      arguments,
      raw_arguments: "",
    }
  }

  #[tokio::test]
  async fn allows_a_relative_path_inside_the_root() {
    let root = unique_temp_dir("allow-relative");
    let guard = WorkspaceGuardCallback::new(&root).unwrap();
    let context = ExecutionContext::new();

    let args = json!({ "file_path": "notes.txt" });
    assert!(matches!(
      guard.call(&context, view(file_delete::NAME, &args)).await,
      ToolCallDecision::Proceed
    ));
  }

  #[tokio::test]
  async fn allows_an_absolute_path_inside_the_root() {
    let root = unique_temp_dir("allow-absolute");
    let guard = WorkspaceGuardCallback::new(&root).unwrap();
    let context = ExecutionContext::new();

    let inside = root.join("notes.txt");
    let args = json!({ "file_path": inside.to_string_lossy() });
    assert!(matches!(
      guard.call(&context, view(file_delete::NAME, &args)).await,
      ToolCallDecision::Proceed
    ));
  }

  #[tokio::test]
  async fn denies_a_parent_directory_escape() {
    let root = unique_temp_dir("deny-dotdot");
    let guard = WorkspaceGuardCallback::new(&root).unwrap();
    let context = ExecutionContext::new();

    let args = json!({ "file_path": "../../etc/passwd" });
    let result = guard.call(&context, view(file_delete::NAME, &args)).await;
    assert!(matches!(
      result,
      ToolCallDecision::ShortCircuit(ToolResultStatus::Error, _)
    ));
  }

  #[tokio::test]
  async fn denies_an_absolute_path_outside_the_root() {
    let root = unique_temp_dir("deny-absolute");
    let guard = WorkspaceGuardCallback::new(&root).unwrap();
    let context = ExecutionContext::new();

    let args = json!({ "file_path": "/etc/passwd" });
    let result = guard.call(&context, view(file_delete::NAME, &args)).await;
    assert!(matches!(
      result,
      ToolCallDecision::ShortCircuit(ToolResultStatus::Error, _)
    ));
  }

  #[tokio::test]
  async fn checks_the_field_list_files_actually_uses() {
    let root = unique_temp_dir("field-per-tool");
    let guard = WorkspaceGuardCallback::new(&root).unwrap();
    let context = ExecutionContext::new();

    let args = json!({ "path": "../outside" });
    let result = guard.call(&context, view(file_list::NAME, &args)).await;
    assert!(!result.is_proceed());
  }

  #[tokio::test]
  async fn checks_both_fields_of_unzip_file() {
    let root = unique_temp_dir("unzip-fields");
    let guard = WorkspaceGuardCallback::new(&root).unwrap();
    let context = ExecutionContext::new();

    let args = json!({ "zip_path": "archive.zip", "extract_to": "../escape" });
    let result = guard.call(&context, view(file_upzip::NAME, &args)).await;
    assert!(!result.is_proceed());
  }

  #[tokio::test]
  async fn allows_a_not_yet_existing_path_via_lexical_resolution() {
    let root = unique_temp_dir("lexical-nonexistent");
    let guard = WorkspaceGuardCallback::new(&root).unwrap();
    let context = ExecutionContext::new();

    // A fresh extraction directory does not exist yet, so this must fall back to
    // lexical `.`/`..` resolution rather than failing outright.
    let args = json!({ "zip_path": "archive.zip", "extract_to": "brand-new-subdir" });
    assert!(matches!(
      guard.call(&context, view(file_upzip::NAME, &args)).await,
      ToolCallDecision::Proceed
    ));
  }

  #[tokio::test]
  async fn denies_read_image_escaping_the_root() {
    let root = unique_temp_dir("read-image-escape");
    let guard = WorkspaceGuardCallback::new(&root).unwrap();
    let context = ExecutionContext::new();

    let args =
      json!({ "file_path": "../outside.png", "query": "what is this?", "model": "vision" });
    let result = guard.call(&context, view(read_image::NAME, &args)).await;
    assert!(matches!(
      result,
      ToolCallDecision::ShortCircuit(ToolResultStatus::Error, _)
    ));
  }

  /// Regression guard: every built-in tool that reads/writes a filesystem path (as
  /// opposed to e.g. `calculator`, which never touches a path at all) must be named in
  /// [`WorkspaceGuardCallback::guarded_fields`] — a new filesystem tool added to
  /// [`ToolRegistry::builtin`] without a matching entry here would silently bypass the
  /// sandbox. This walks the actual built-in tool list rather than hand-maintaining a
  /// second copy of it, so the two cannot drift apart unnoticed.
  #[test]
  fn every_path_taking_builtin_tool_is_guarded() {
    let path_taking_tools = [
      file_delete::NAME,
      file_list::NAME,
      file_read::NAME,
      file_upzip::NAME,
      read_image::NAME,
    ];
    let registry = ToolRegistry::builtin().unwrap();
    for name in registry.names() {
      if path_taking_tools.contains(&name) {
        assert!(
          !WorkspaceGuardCallback::guarded_fields(name).is_empty(),
          "`{name}` reads/writes a filesystem path but is not guarded by \
           WorkspaceGuardCallback"
        );
      }
    }
  }

  #[tokio::test]
  async fn ignores_tools_it_does_not_guard() {
    let root = unique_temp_dir("ignore-others");
    let guard = WorkspaceGuardCallback::new(&root).unwrap();
    let context = ExecutionContext::new();

    let args = json!({ "operator": "add", "first_number": 1, "second_number": 2 });
    assert!(matches!(
      guard.call(&context, view("calculator", &args)).await,
      ToolCallDecision::Proceed
    ));
  }

  #[test]
  fn new_rejects_a_root_that_does_not_exist() {
    let missing =
      std::env::temp_dir().join(format!("agent-path-guard-missing-{}", uuid::Uuid::new_v4()));
    assert!(WorkspaceGuardCallback::new(&missing).is_err());
  }

  /// The whole reason [`WorkspaceGuardCallback::new`] canonicalizes `root` up front,
  /// and [`resolve_best_effort`] canonicalizes as much of a candidate as exists: a
  /// symlink *inside* the root that actually points outside it must be caught, not
  /// just a literal `..` in the argument string.
  #[cfg(unix)]
  #[tokio::test]
  async fn denies_an_existing_symlink_that_points_outside_the_root() {
    let outside = unique_temp_dir("symlink-target-outside");
    let root = unique_temp_dir("symlink-escape-root");
    let link = root.join("escape");
    std::os::unix::fs::symlink(&outside, &link).unwrap();

    let guard = WorkspaceGuardCallback::new(&root).unwrap();
    let context = ExecutionContext::new();

    let args = json!({ "file_path": "escape/secret.txt" });
    let result = guard.call(&context, view(file_delete::NAME, &args)).await;
    assert!(matches!(
      result,
      ToolCallDecision::ShortCircuit(ToolResultStatus::Error, _)
    ));
  }
}
