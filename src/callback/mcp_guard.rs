//! [`McpGuardCallback`]: a schema-agnostic guard for MCP tool calls specifically.
//!
//! [`crate::callback::path_guard::WorkspaceGuardCallback`] cannot cover MCP tools: it
//! decides by field name (`file_path`, `path`, ...), and those names are only known for
//! the fixed set of built-in tools, not for a tool an MCP server defines and describes at
//! runtime. This callback takes a different, more conservative approach that needs no
//! schema at all: it does not try to tell *which* argument of an unknown tool is a path,
//! it just walks every string value the call carries — however deeply nested — and denies
//! the call if any of them plainly names a well-known credential or cloud-config location
//! under the current user's home directory (`~/.ssh`, `~/.aws`, ...; see
//! [`SENSITIVE_HOME_SUBPATHS`]).
//!
//! This is deliberately narrower than a full workspace boundary: a generic "does this
//! string look like a path outside some root" check has no way to tell a legitimate path
//! argument from an ordinary piece of text that happens to contain a slash (a
//! `web_search`-style query, a commit message, ...) once the field name itself is
//! unknown — which is exactly the situation every MCP tool is in from this callback's
//! point of view. Restricting the check to a short, well-known list of credential
//! locations keeps false positives (denying a call that was never actually trying to
//! touch a secret) rare, at the cost of not catching every possible exfiltration path —
//! see the type docs' "Known limitations" for what is deliberately left uncovered.
//!
//! Meant to be registered alongside [`crate::callback::path_guard::WorkspaceGuardCallback`]
//! (which keeps covering the built-in tools it already knows the exact fields of), not
//! instead of it.

use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::{
  agent::{
    ExecutionContext,
    callback::{BeforeToolCallback, ToolCallDecision, ToolCallView},
  },
  callback::path_guard::resolve_best_effort,
  tools::mcp::client,
};

/// Home-relative locations that almost always hold credentials or cloud/tool
/// configuration, never something an ordinary tool call should need to name directly.
/// Modeled on the exclusion list Windows Sandbox itself uses to protect a user profile
/// from a sandboxed process (`.ssh`, `.aws`, `.kube`, `.docker`, ...) — the same
/// rationale applies here: an MCP server is code this process did not write, running
/// with (at least some of) this user's privileges.
///
/// Deliberately not configurable (yet): every entry here is something no legitimate MCP
/// tool call should ever need, in any deployment, so there is no known case where an
/// operator would want to shrink this list — only extend it, which is exactly why it is
/// a plain constant rather than a config field: only trust boundary maintainers should
/// need to touch it, not every `mcp.json` author.
const SENSITIVE_HOME_SUBPATHS: &[&str] = &[
  ".ssh",
  ".aws",
  ".azure",
  ".kube",
  ".docker",
  ".gnupg",
  ".npmrc",
  ".npm",
  ".pki",
  ".terraform.d",
  ".netrc",
  ".git-credentials",
  ".config/gh",
  ".config/gcloud",
];

/// Denies an MCP tool call ([`client::is_mcp_tool_name`]) whose arguments name a path
/// under [`SENSITIVE_HOME_SUBPATHS`], before the call ever reaches the remote server —
/// see the module docs for why this is the check this callback runs, and what it
/// deliberately does not attempt.
///
/// # Known limitations
///
/// - **Only the well-known list above, only under the home directory**: a credential
///   living somewhere else entirely (a project-local `.env`, a secret bind-mounted at
///   `/run/secrets`, ...) is not covered. This callback is one layer, not the whole
///   story — an MCP server's own sandboxing (or simply not enabling servers you do not
///   trust) still matters.
/// - **Text, not semantics**: a call that first base64-encodes, splits, or otherwise
///   obfuscates the path string defeats this the same way it would defeat any other
///   static string check — this is a speed bump for an unintentionally overreaching
///   tool call, not a defense against a server actively trying to evade detection.
/// - **HTTP/remote MCP servers are not spawned processes**: this callback still
///   examines their arguments the same way, but has no bearing on what such a server
///   does with them once the call leaves this process over the network.
pub struct McpGuardCallback;

impl McpGuardCallback {
  pub fn new() -> Self {
    Self
  }
}

impl Default for McpGuardCallback {
  fn default() -> Self {
    Self::new()
  }
}

#[async_trait::async_trait]
impl BeforeToolCallback for McpGuardCallback {
  async fn call(
    &self,
    _context: &ExecutionContext,
    tool_call: ToolCallView<'_>,
  ) -> ToolCallDecision {
    if !client::is_mcp_tool_name(tool_call.name) {
      return ToolCallDecision::Proceed;
    }

    let mut candidates = Vec::new();
    collect_strings(tool_call.arguments, &mut candidates);

    for candidate in candidates {
      if let Some(reason) = violation(candidate) {
        tracing::warn!(
          tool = tool_call.name,
          %reason,
          "blocked an MCP tool call touching a sensitive credential path"
        );
        return ToolCallDecision::deny(format!(
          "Denied: {reason}. MCP tools are not allowed to reference credential/config \
           paths under the home directory."
        ));
      }
    }
    ToolCallDecision::Proceed
  }
}

/// Recursively collect every string leaf in a JSON value — an MCP tool's arguments can
/// nest a path inside an object or array (e.g. `{"paths": ["~/.ssh/id_rsa"]}`), and the
/// schema is unknown, so every string anywhere in the tree is a candidate.
fn collect_strings<'a>(value: &'a Value, out: &mut Vec<&'a str>) {
  match value {
    Value::String(text) => out.push(text),
    Value::Array(items) => {
      for item in items {
        collect_strings(item, out);
      }
    }
    Value::Object(fields) => {
      for field in fields.values() {
        collect_strings(field, out);
      }
    }
    Value::Null | Value::Bool(_) | Value::Number(_) => {}
  }
}

/// `None` if `candidate` does not plainly name a sensitive location; `Some(reason)`
/// naming the resolved path otherwise. Only two shapes of `candidate` are treated as
/// naming a path at all — a `~`-relative one, or an absolute one — precisely because
/// those are the only shapes unambiguous enough to check without a known base directory
/// to resolve a plain relative string against (see the type docs' rationale for why a
/// broader check would be mostly false positives).
fn violation(candidate: &str) -> Option<String> {
  let path = absolute_or_home_relative(candidate)?;
  let home = home_dir()?;
  let resolved = resolve_best_effort(&path);

  SENSITIVE_HOME_SUBPATHS.iter().find_map(|subpath| {
    let sensitive_root = resolve_best_effort(&home.join(subpath));
    resolved.starts_with(&sensitive_root).then(|| {
      format!(
        "`{candidate}` resolves to `{}`, inside the sensitive path `{}`",
        resolved.display(),
        sensitive_root.display(),
      )
    })
  })
}

/// `candidate` as a [`PathBuf`] if it is absolute or starts with `~`/`~/`; `None` for
/// anything else (relative paths, or plain text with no path shape at all), since there
/// is no base directory to resolve either of those against here.
fn absolute_or_home_relative(candidate: &str) -> Option<PathBuf> {
  let trimmed = candidate.trim();

  if trimmed == "~" {
    return home_dir();
  }
  if let Some(rest) = trimmed.strip_prefix("~/") {
    return home_dir().map(|home| home.join(rest));
  }

  let path = Path::new(trimmed);
  path.is_absolute().then(|| path.to_path_buf())
}

/// The current user's home directory, read directly from the platform's own environment
/// variable rather than a crate dependency: `HOME` on Unix, `USERPROFILE` on Windows.
/// `None` propagates to [`violation`] returning `None` — with nothing to resolve `~`/an
/// absolute path's sensitivity against, this callback has no basis to deny anything, so
/// it stays out of the way rather than guessing.
fn home_dir() -> Option<PathBuf> {
  #[cfg(windows)]
  let var = "USERPROFILE";
  #[cfg(not(windows))]
  let var = "HOME";

  std::env::var_os(var).map(PathBuf::from)
}

#[cfg(test)]
mod tests {
  use serde_json::json;

  use super::*;
  use crate::agent::ToolResultStatus;

  /// A freshly created directory under the OS temp dir, used as a fake `$HOME` so tests
  /// never touch the real one. Not cleaned up afterwards, same tradeoff as
  /// `agent::session`'s and `path_guard`'s tests.
  fn fake_home(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
      "agent-mcp-guard-test-{label}-{}",
      uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
  }

  fn set_home(dir: &Path) {
    // SAFETY: tests in this module do not run other tests' assertions concurrently with
    // this mutation in a way that would observe a torn value — each test sets its own
    // freshly generated, uniquely-named directory before making any assertion.
    #[cfg(windows)]
    unsafe {
      std::env::set_var("USERPROFILE", dir)
    };
    #[cfg(not(windows))]
    unsafe {
      std::env::set_var("HOME", dir)
    };
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
  async fn ignores_non_mcp_tool_names() {
    let guard = McpGuardCallback::new();
    let context = ExecutionContext::new();
    let args = json!({ "file_path": "~/.ssh/id_rsa" });

    // No `__` in the name: this callback leaves it entirely to
    // `WorkspaceGuardCallback`/the tool's own validation.
    assert!(matches!(
      guard.call(&context, view("delete_file", &args)).await,
      ToolCallDecision::Proceed
    ));
  }

  #[tokio::test]
  async fn denies_a_tilde_relative_sensitive_path() {
    let home = fake_home("tilde");
    set_home(&home);
    let guard = McpGuardCallback::new();
    let context = ExecutionContext::new();

    let args = json!({ "path": "~/.ssh/id_rsa" });
    let result = guard.call(&context, view("demo__read_file", &args)).await;
    assert!(matches!(
      result,
      ToolCallDecision::ShortCircuit(ToolResultStatus::Error, _)
    ));
  }

  #[tokio::test]
  async fn denies_an_absolute_sensitive_path() {
    let home = fake_home("absolute");
    set_home(&home);
    let guard = McpGuardCallback::new();
    let context = ExecutionContext::new();

    let args = json!({ "path": home.join(".aws/credentials").to_string_lossy() });
    let result = guard.call(&context, view("demo__read_file", &args)).await;
    assert!(matches!(
      result,
      ToolCallDecision::ShortCircuit(ToolResultStatus::Error, _)
    ));
  }

  #[tokio::test]
  async fn denies_a_sensitive_path_nested_inside_an_array() {
    let home = fake_home("nested");
    set_home(&home);
    let guard = McpGuardCallback::new();
    let context = ExecutionContext::new();

    let args = json!({ "paths": ["notes.txt", "~/.ssh/id_rsa"] });
    let result = guard.call(&context, view("demo__read_many", &args)).await;
    assert!(matches!(
      result,
      ToolCallDecision::ShortCircuit(ToolResultStatus::Error, _)
    ));
  }

  #[tokio::test]
  async fn allows_an_ordinary_relative_path() {
    let home = fake_home("relative");
    set_home(&home);
    let guard = McpGuardCallback::new();
    let context = ExecutionContext::new();

    let args = json!({ "path": "notes.txt" });
    assert!(matches!(
      guard.call(&context, view("demo__read_file", &args)).await,
      ToolCallDecision::Proceed
    ));
  }

  #[tokio::test]
  async fn allows_free_text_that_merely_contains_a_slash() {
    let home = fake_home("free-text");
    set_home(&home);
    let guard = McpGuardCallback::new();
    let context = ExecutionContext::new();

    // A web-search-style query: has a `/` in it, is not remotely a path, and — being
    // relative-looking — is never even resolved against `home` in the first place.
    let args = json!({ "query": "rust async/await tutorial" });
    assert!(matches!(
      guard.call(&context, view("demo__web_search", &args)).await,
      ToolCallDecision::Proceed
    ));
  }

  #[tokio::test]
  async fn allows_an_absolute_path_outside_the_sensitive_list() {
    let home = fake_home("outside-list");
    set_home(&home);
    let guard = McpGuardCallback::new();
    let context = ExecutionContext::new();

    let args = json!({ "path": home.join("Documents/report.pdf").to_string_lossy() });
    assert!(matches!(
      guard.call(&context, view("demo__read_file", &args)).await,
      ToolCallDecision::Proceed
    ));
  }
}
