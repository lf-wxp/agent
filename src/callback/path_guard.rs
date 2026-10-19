use std::path::{Component, Path, PathBuf};

use anyhow::Context;
use serde_json::Value;

use crate::{
  agent::{
    ExecutionContext,
    callback::{BeforeToolCallback, ToolCallDecision, ToolCallView},
  },
  tools::{
    file_delete, file_edit, file_list, file_read, file_upzip, file_write, grep, read_image, shell,
  },
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
/// - **`run_command` is only guarded on its `workdir`**: the command string itself is
///   arbitrary shell, and no path check on it could be sound (`cd .. && rm x`, `$HOME`,
///   `eval`, ...). Whether it runs at all is left to a human via
///   [`crate::callback::dual_approval::DualApprovalCallback`].
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
      || tool_name == file_write::NAME
      || tool_name == file_edit::NAME
    {
      &["file_path"]
    } else if tool_name == file_list::NAME || tool_name == grep::NAME {
      &["path"]
    } else if tool_name == file_upzip::NAME {
      &["zip_path", "extract_to"]
    } else if tool_name == shell::NAME {
      &["workdir"]
    } else {
      &[]
    }
  }

  /// Whether `tool` *changes* what `field` names, as opposed to only reading it. Decides
  /// which of the [`protected`] paths apply: reading `.git/config` is ordinary, writing
  /// it is not.
  ///
  /// `unzip_file`'s `extract_to` is a write and its `zip_path` is not; everything else
  /// the guard knows about that changes anything does so through `file_path`.
  fn is_write(tool_name: &str, field: &str) -> bool {
    match tool_name {
      name if name == file_delete::NAME || name == file_write::NAME || name == file_edit::NAME => {
        true
      }
      name if name == file_upzip::NAME => field == "extract_to",
      _ => false,
    }
  }

  /// `None` if `candidate` — resolved against `self.root` when relative, taken as-is
  /// when absolute — stays inside the workspace and is not a [`protected`] path for this
  /// kind of access; `Some(reason)` naming the resolved path otherwise.
  fn violation(&self, candidate: &str, write: bool) -> Option<String> {
    let requested = Path::new(candidate);
    let joined = if requested.is_absolute() {
      requested.to_path_buf()
    } else {
      self.root.join(requested)
    };

    let resolved = resolve_best_effort(&joined);

    let Ok(inside) = resolved.strip_prefix(&self.root) else {
      return Some(format!(
        "`{candidate}` resolves to `{}`, outside the workspace root `{}`",
        resolved.display(),
        self.root.display(),
      ));
    };
    protected(inside, write).map(|why| {
      format!(
        "`{candidate}` resolves to `{}`, which is {why}",
        resolved.display()
      )
    })
  }
}

/// Why a path *inside* the workspace is still off limits, or `None` if it is not.
///
/// Being inside the root only says a path is the project's, not that it is ordinary
/// project content. A few things in there are worth more than their size suggests, and a
/// model that can overwrite them has been handed more than file access:
///
/// - `.env*` — credentials. Refused for reading as well as writing, since reading one
///   puts the secrets straight into the model's context (and the provider's logs).
/// - `.git/` — hooks run as the user the next time git does anything.
/// - `.agent/` — the session and approval stores; rewriting them forges history and
///   pending approvals.
/// - `mcp.json` — each enabled entry is a command launched at startup, so writing it is
///   code execution on the next run, with no approval prompt in the way.
///
/// `relative` is relative to the workspace root, so a root that itself sits under a
/// directory with one of these names is not affected. The match is on whole path
/// components, case-insensitively, since the filesystems this runs on are not all
/// case-sensitive.
fn protected(relative: &Path, write: bool) -> Option<&'static str> {
  for component in relative.components() {
    let Component::Normal(name) = component else {
      continue;
    };
    let name = name.to_string_lossy().to_ascii_lowercase();
    if name == ".env" || name.starts_with(".env.") {
      return Some("a credentials file, which these tools may not read or write");
    }
    if !write {
      continue;
    }
    match name.as_str() {
      ".git" => return Some("inside `.git`, which these tools may not modify"),
      ".agent" => {
        return Some("inside `.agent` (sessions and approvals), which these tools may not modify");
      }
      _ => {}
    }
  }
  if write {
    let is_mcp_config = relative
      .file_name()
      .is_some_and(|name| name.to_string_lossy().eq_ignore_ascii_case("mcp.json"));
    if is_mcp_config {
      return Some("the MCP server config, whose entries run as commands at startup");
    }
  }
  None
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
      let write = Self::is_write(tool_call.name, field);
      if let Some(reason) = self.violation(candidate, write) {
        tracing::warn!(
          tool = tool_call.name,
          field = *field,
          %reason,
          "blocked a filesystem call outside the workspace root or on a protected path"
        );
        return ToolCallDecision::deny(format!(
          "Denied: {reason}. This tool is restricted to the workspace root and may not \
           touch protected files; ask for a different path."
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
  use crate::tools::{ToolRegistry, calculator, web_search};

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

  /// Regression guard: every built-in tool must be *classified* — either it takes a
  /// filesystem path and is named in [`WorkspaceGuardCallback::guarded_fields`], or it is
  /// on the explicit list of tools that never touch a path. Walking the actual tool list
  /// (rather than only the path-taking subset) is what makes this catch a new tool: one
  /// added to [`ToolRegistry::builtin_all`] and named in neither place fails here, rather
  /// than silently bypassing the sandbox.
  #[test]
  fn every_builtin_tool_is_classified_for_the_guard() {
    let path_free_tools = [calculator::NAME, web_search::NAME];
    let registry = ToolRegistry::builtin_all().unwrap();
    for name in registry.names() {
      let guarded = !WorkspaceGuardCallback::guarded_fields(name).is_empty();
      let path_free = path_free_tools.contains(&name);
      assert!(
        guarded ^ path_free,
        "`{name}` is {}; classify it in WorkspaceGuardCallback::guarded_fields or in \
         this test's list of path-free tools",
        if guarded {
          "both guarded and listed as path-free"
        } else {
          "neither guarded nor listed as path-free"
        }
      );
    }
  }

  #[tokio::test]
  async fn denies_writes_edits_and_searches_escaping_the_root() {
    let root = unique_temp_dir("write-tools-escape");
    let guard = WorkspaceGuardCallback::new(&root).unwrap();
    let context = ExecutionContext::new();

    let cases = [
      (
        file_write::NAME,
        json!({ "file_path": "../x.txt", "content": "x" }),
      ),
      (
        file_edit::NAME,
        json!({ "file_path": "/etc/hosts", "old_string": "a", "new_string": "b" }),
      ),
      (grep::NAME, json!({ "pattern": "x", "path": "../" })),
      (shell::NAME, json!({ "command": "ls", "workdir": "/" })),
    ];
    for (name, args) in cases {
      let result = guard.call(&context, view(name, &args)).await;
      assert!(!result.is_proceed(), "`{name}` escaped the workspace");
    }
  }

  async fn decision(root: &Path, tool: &str, args: Value) -> ToolCallDecision {
    let guard = WorkspaceGuardCallback::new(root).unwrap();
    let context = ExecutionContext::new();
    guard.call(&context, view(tool, &args)).await
  }

  /// Inside the root is not enough: these writes are a way to run code or forge state.
  #[tokio::test]
  async fn denies_writing_to_protected_paths_inside_the_root() {
    let root = unique_temp_dir("protected-write");
    for (tool, args) in [
      (
        file_write::NAME,
        json!({ "file_path": "mcp.json", "content": "{}" }),
      ),
      (
        file_write::NAME,
        json!({ "file_path": "sub/MCP.JSON", "content": "{}" }),
      ),
      (
        file_write::NAME,
        json!({ "file_path": ".git/hooks/pre-commit", "content": "x" }),
      ),
      (
        file_edit::NAME,
        json!({ "file_path": ".agent/sessions/a.json", "old_string": "a", "new_string": "b" }),
      ),
      (file_delete::NAME, json!({ "file_path": ".git/config" })),
      (
        file_upzip::NAME,
        json!({ "zip_path": "a.zip", "extract_to": ".git" }),
      ),
    ] {
      let result = decision(&root, tool, args.clone()).await;
      assert!(!result.is_proceed(), "`{tool}` {args} was allowed");
    }
  }

  /// A protected path is protected against *changes*; looking is ordinary.
  #[tokio::test]
  async fn allows_reading_the_non_secret_protected_paths() {
    let root = unique_temp_dir("protected-read");
    for (tool, args) in [
      (file_read::NAME, json!({ "file_path": ".git/config" })),
      (file_read::NAME, json!({ "file_path": "mcp.json" })),
      (file_list::NAME, json!({ "path": ".agent" })),
      (
        file_upzip::NAME,
        json!({ "zip_path": ".git/x.zip", "extract_to": "out" }),
      ),
    ] {
      let result = decision(&root, tool, args.clone()).await;
      assert!(result.is_proceed(), "`{tool}` {args} was denied");
    }
  }

  /// Credentials files are refused for reading too: a read puts the secrets in the model's
  /// context.
  #[tokio::test]
  async fn denies_any_access_to_env_files() {
    let root = unique_temp_dir("protected-env");
    for (tool, args) in [
      (file_read::NAME, json!({ "file_path": ".env" })),
      (file_read::NAME, json!({ "file_path": "app/.env.local" })),
      (grep::NAME, json!({ "pattern": "KEY", "path": ".env" })),
      (
        read_image::NAME,
        json!({ "file_path": ".env", "query": "q", "model": "m" }),
      ),
      (
        file_write::NAME,
        json!({ "file_path": ".env", "content": "x" }),
      ),
    ] {
      let result = decision(&root, tool, args.clone()).await;
      assert!(!result.is_proceed(), "`{tool}` {args} was allowed");
    }
  }

  /// Whole components only: a name that merely contains one is ordinary.
  #[tokio::test]
  async fn does_not_overreach_on_similar_names() {
    let root = unique_temp_dir("protected-similar");
    for (tool, args) in [
      (
        file_write::NAME,
        json!({ "file_path": "environment.md", "content": "x" }),
      ),
      (
        file_write::NAME,
        json!({ "file_path": "docs/.github/ci.yml", "content": "x" }),
      ),
      (
        file_write::NAME,
        json!({ "file_path": "my-mcp.json.bak", "content": "x" }),
      ),
      (
        file_write::NAME,
        json!({ "file_path": "gitignore.txt", "content": "x" }),
      ),
    ] {
      let result = decision(&root, tool, args.clone()).await;
      assert!(result.is_proceed(), "`{tool}` {args} was denied");
    }
  }

  /// A root that itself lives under a directory with a protected name is unaffected: only
  /// the part of the path below the root counts.
  #[tokio::test]
  async fn a_protected_name_above_the_root_does_not_count() {
    let parent = unique_temp_dir("protected-above");
    let root = parent.join(".agent").join("work");
    std::fs::create_dir_all(&root).unwrap();

    let result = decision(
      &root,
      file_write::NAME,
      json!({ "file_path": "notes.txt", "content": "x" }),
    )
    .await;
    assert!(result.is_proceed());
  }

  #[tokio::test]
  async fn allows_run_command_without_a_workdir() {
    let root = unique_temp_dir("shell-no-workdir");
    let guard = WorkspaceGuardCallback::new(&root).unwrap();
    let context = ExecutionContext::new();

    let args = json!({ "command": "ls" });
    assert!(
      guard
        .call(&context, view(shell::NAME, &args))
        .await
        .is_proceed()
    );
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
