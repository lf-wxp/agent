//! The `Tool` abstraction implemented by every tool.

use std::time::Duration;

use async_openai::types::chat::{ChatCompletionTool, ChatCompletionTools, FunctionObjectArgs};
use serde_json::Value;

/// A tool the model can call.
///
/// Uses `async_trait` rather than a native `async fn`: the registry keeps tools as
/// `Box<dyn Tool>`, and native async fns in traits are not dyn-compatible.
#[async_trait::async_trait]
pub trait Tool: Send + Sync {
  /// Name used both in the advertised definition and for dispatch; must be unique
  /// across the registry, otherwise only the first match is ever reached.
  fn name(&self) -> &str;

  /// What the tool does, and when the model should reach for it.
  fn description(&self) -> &str;

  /// JSON Schema for the arguments, used as `function.parameters`.
  fn parameters(&self) -> Value;

  /// Run the tool against the raw JSON arguments produced by the model.
  ///
  /// Return `Err` for anything that did not work — malformed arguments, a failing
  /// request. The registry turns it into a tool message so the model can correct
  /// itself, which is why implementations never format their own error text.
  ///
  /// Takes only the raw arguments, not the caller's [`crate::agent::ExecutionContext`]:
  /// no built-in tool has ever needed the surrounding conversation to do its job, and a
  /// parameter every implementation had to name `_context` to ignore was pure
  /// speculative generality. Add it back (as a `Vec<Event>` snapshot or similar) if a
  /// concrete tool needs it — e.g. to avoid repeating a lookup already made earlier in
  /// the same turn.
  async fn execute(&self, args_json: &str) -> anyhow::Result<String>;

  /// This tool's own time limit, overriding the caller's default (see
  /// [`execute_with_timeout`]). `None` — the default — means "use the caller's".
  ///
  /// Override for a tool whose legitimate running time is known to differ from the
  /// general-purpose default, e.g. one that enforces its own, longer limit internally
  /// and must not be cut off by the outer one first.
  fn timeout(&self) -> Option<Duration> {
    None
  }

  /// Whether this call may run at the same time as its siblings in one round without
  /// them being able to affect each other — in practice: it changes nothing outside
  /// itself.
  ///
  /// `false` — the default — makes the tool take its turn: within a round, a call that
  /// is not concurrency-safe starts only after every call before it has finished, and
  /// the calls after it wait for it in turn (see [`crate::tools::schedule`]). That is
  /// the conservative reading for anything of unknown behaviour, e.g. a tool discovered
  /// over MCP: two `edit_file` calls on one file would otherwise both read the original
  /// text and the later write would silently discard the earlier edit.
  ///
  /// Override with `true` only for a tool that is read-only, so that independent lookups
  /// (two `web_search` calls) still overlap their latency instead of queueing.
  fn concurrency_safe(&self) -> bool {
    false
  }

  /// Render the definition advertised to the model.
  ///
  /// Fails only when name / description / parameters are inconsistent with what the API
  /// accepts, which is a programming error rather than a runtime condition.
  fn definition(&self) -> anyhow::Result<ChatCompletionTools> {
    let function = FunctionObjectArgs::default()
      .name(self.name())
      .description(self.description())
      .parameters(self.parameters())
      .build()
      .map_err(|e| anyhow::anyhow!("Failed to build tool definition for {}: {e}", self.name()))?;

    Ok(ChatCompletionTools::Function(ChatCompletionTool {
      function,
    }))
  }
}

/// Run `tool`, giving up once its limit — [`Tool::timeout`], else `default` — elapses.
///
/// The single place every call site ([`crate::tools::ToolRegistry::execute`],
/// [`crate::agent::Agent`]) goes through, so no path can execute a tool unbounded.
///
/// Giving up drops the tool's future. For an async tool that cancels it at its next
/// `.await` (and e.g. `run_command`'s child process group is killed on drop). A
/// synchronous tool already running on the blocking pool cannot be interrupted: the
/// caller stops waiting, but that thread runs to completion in the background.
///
/// Because of that last case, the error text for a tool that is not
/// [`Tool::concurrency_safe`] does not claim the work stopped: a model told "abandoned"
/// would reasonably retry, and a retried delete or extraction on top of one still
/// running is worse than the timeout. A read-only tool has no such hazard and is told to
/// narrow the request instead.
///
/// A tool's own [`Tool::timeout`] wins over `default` in both directions, so it can be
/// longer *or* shorter than what the caller configured.
pub async fn execute_with_timeout(
  tool: &dyn Tool,
  args_json: &str,
  default: Duration,
) -> anyhow::Result<String> {
  let limit = tool.timeout().unwrap_or(default);
  match tokio::time::timeout(limit, tool.execute(args_json)).await {
    Ok(result) => result,
    Err(_) if tool.concurrency_safe() => anyhow::bail!(
      "`{}` did not finish within {}s and was abandoned; try a smaller or more specific \
       request",
      tool.name(),
      limit.as_secs_f64()
    ),
    Err(_) => anyhow::bail!(
      "`{}` did not finish within {}s and the caller stopped waiting, but it may still be \
       running or may have partly taken effect; check the current state before trying \
       again instead of repeating it",
      tool.name(),
      limit.as_secs_f64()
    ),
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  struct Sleepy {
    own_limit: Option<Duration>,
    read_only: bool,
  }

  #[async_trait::async_trait]
  impl Tool for Sleepy {
    fn name(&self) -> &str {
      "sleepy"
    }
    fn description(&self) -> &str {
      "sleeps"
    }
    fn parameters(&self) -> Value {
      serde_json::json!({ "type": "object" })
    }
    async fn execute(&self, _args_json: &str) -> anyhow::Result<String> {
      tokio::time::sleep(Duration::from_millis(200)).await;
      Ok("woke".to_owned())
    }
    fn timeout(&self) -> Option<Duration> {
      self.own_limit
    }
    fn concurrency_safe(&self) -> bool {
      self.read_only
    }
  }

  #[tokio::test]
  async fn a_tool_outliving_the_default_is_abandoned() {
    let tool = Sleepy {
      own_limit: None,
      read_only: true,
    };
    let err = execute_with_timeout(&tool, "{}", Duration::from_millis(20))
      .await
      .unwrap_err();
    assert!(
      err.to_string().contains("did not finish within"),
      "got: {err}"
    );
    assert!(err.to_string().contains("abandoned"), "got: {err}");
  }

  /// A tool with side effects may still be running, or have partly done its work, so
  /// the model must not be told it was simply abandoned.
  #[tokio::test]
  async fn a_timed_out_tool_with_side_effects_is_not_reported_as_abandoned() {
    let tool = Sleepy {
      own_limit: None,
      read_only: false,
    };
    let err = execute_with_timeout(&tool, "{}", Duration::from_millis(20))
      .await
      .unwrap_err()
      .to_string();
    assert!(err.contains("did not finish within"), "got: {err}");
    assert!(err.contains("check the current state"), "got: {err}");
    assert!(!err.contains("abandoned"), "got: {err}");
  }

  #[tokio::test]
  async fn a_tool_within_its_limit_returns_normally() {
    let tool = Sleepy {
      own_limit: None,
      read_only: true,
    };
    let output = execute_with_timeout(&tool, "{}", Duration::from_secs(5))
      .await
      .unwrap();
    assert_eq!(output, "woke");
  }

  #[tokio::test]
  async fn the_tools_own_limit_overrides_the_default() {
    let tool = Sleepy {
      own_limit: Some(Duration::from_secs(5)),
      read_only: true,
    };
    let output = execute_with_timeout(&tool, "{}", Duration::from_millis(20))
      .await
      .unwrap();
    assert_eq!(output, "woke");
  }

  #[test]
  fn a_tool_is_not_concurrency_safe_unless_it_says_so() {
    struct Plain;
    #[async_trait::async_trait]
    impl Tool for Plain {
      fn name(&self) -> &str {
        "plain"
      }
      fn description(&self) -> &str {
        ""
      }
      fn parameters(&self) -> Value {
        serde_json::json!({ "type": "object" })
      }
      async fn execute(&self, _args_json: &str) -> anyhow::Result<String> {
        Ok(String::new())
      }
    }
    assert!(!Plain.concurrency_safe());
  }
}
