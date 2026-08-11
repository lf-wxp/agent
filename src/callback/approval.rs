use std::{
  collections::HashSet,
  io::{self, Write},
};

use tokio::sync::Mutex;

use crate::agent::{
  ExecutionContext, ToolResultStatus,
  callback::{BeforeToolCallback, ToolCallView},
};

/// Asks a human on the console before letting any listed tool run; denying records an
/// error result in place of the call, and the model carries on without it.
///
/// For interactive CLI use only. The prompt blocks on stdin with no timeout, so a run
/// waits indefinitely for an answer — inside a request handler (see [`crate::api`]) that
/// would hang the request and pin a thread from the blocking pool. A non-interactive
/// process denies everything instead, since reading a closed stdin yields no `y`.
///
/// Prompts go to stderr, not stdout: stdout is a protocol channel for anything speaking
/// MCP over stdio (see `examples/mcp_server.rs`), and a prompt written there would corrupt
/// the stream.
pub struct ApprovalCallback {
  dangerous_tools: HashSet<String>,
  // Serializes the prompt/read pair below: tool calls in the same round run
  // concurrently (see `Agent::execute_tool_calls`), and without this lock two
  // concurrent dangerous calls would interleave their console prompts and could
  // read the wrong `y`/`n` answer for the wrong tool call.
  prompt_lock: Mutex<()>,
}

impl ApprovalCallback {
  pub fn new(dangerous_tools: impl IntoIterator<Item = impl Into<String>>) -> Self {
    Self {
      dangerous_tools: dangerous_tools.into_iter().map(Into::into).collect(),
      prompt_lock: Mutex::new(()),
    }
  }
}

#[async_trait::async_trait]
impl BeforeToolCallback for ApprovalCallback {
  async fn call(
    &self,
    _context: &ExecutionContext,
    tool_call: ToolCallView<'_>,
  ) -> Option<(ToolResultStatus, String)> {
    if !self.dangerous_tools.contains(tool_call.name) {
      return None;
    }

    // Hold the lock across the whole prompt+read so concurrent dangerous calls in the
    // same round queue up one at a time instead of racing on stdout/stdin.
    let _guard = self.prompt_lock.lock().await;

    eprintln!("\n⚠️  即将执行高危操作");
    eprintln!("工具: {}", tool_call.name);
    // The raw string rather than the parsed arguments: an unparseable payload would show
    // up as `null`, and approving a call whose arguments you cannot see is worse than no
    // prompt at all.
    eprintln!("参数: {}", tool_call.raw_arguments);

    let approved = tokio::task::spawn_blocking(|| {
      eprint!("是否执行？(y/n): ");
      if let Err(err) = io::stderr().flush() {
        tracing::warn!("failed to flush approval prompt: {err}");
      }
      let mut input = String::new();
      if let Err(err) = io::stdin().read_line(&mut input) {
        tracing::warn!("failed to read approval answer, denying by default: {err}");
        return false;
      }
      input.trim().eq_ignore_ascii_case("y")
    })
    .await
    .unwrap_or(false);

    if approved {
      eprintln!("✅ 已批准，继续执行...\n");
      None
    } else {
      eprintln!("❌ 已拒绝，跳过执行\n");
      Some((
        ToolResultStatus::Error,
        format!("User denied execution of {}", tool_call.name),
      ))
    }
  }
}
