//! The `Tool` abstraction implemented by every tool, and [`ToolDefinition`], what the
//! model is told about one.
//!
//! Nothing here depends on a provider's wire format: a definition is a name, a
//! description and a JSON Schema, and turning that into what a given API expects is the
//! request builder's job (see [`crate::llm::client::request_builder`]). A tool written
//! against this trait works unchanged whichever [`crate::llm::backend::ChatBackend`] sends
//! it.
//!
//! For a tool that is one typed function, [`crate::tools::FnTool`] implements this trait
//! from an argument struct and an async closure.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Longest tool name chat APIs accept.
pub const MAX_TOOL_NAME_CHARS: usize = 64;

/// What the model is told about a tool, independent of any provider's wire format.
///
/// Built by [`Tool::definition`] and kept by [`crate::tools::ToolRegistry`]; converted to
/// the API's own shape only when a request is assembled.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
  /// See [`Tool::name`].
  pub name: String,
  /// See [`Tool::description`].
  pub description: String,
  /// See [`Tool::parameters`].
  pub parameters: Value,
}

/// Check `name` against what chat APIs accept for a tool: 1–[`MAX_TOOL_NAME_CHARS`] ASCII
/// letters, digits, `_` or `-`.
///
/// [`crate::tools::ToolRegistry::add`] applies it to every tool, so a bad name fails when
/// the tool is registered — naming the tool — rather than as an opaque rejection of every
/// request the agent then sends.
///
/// This is the **chat-completions** rule, and the one place this crate's provider-neutral
/// layer assumes a particular provider: a name the API rejects makes every request
/// carrying it fail, so it is checked at registration, where the error can still say which
/// tool is to blame. A backend with a narrower rule is satisfied by any name passing this
/// one; one with a *wider* rule (dotted or namespaced names, say) is held back, and would
/// need this check relaxed for its build.
pub fn validate_tool_name(name: &str) -> crate::Result<()> {
  crate::error::ensure!(
    !name.is_empty()
      && name.len() <= MAX_TOOL_NAME_CHARS
      && name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'),
    invalid_input,
    "invalid tool name `{name}`: use 1-{MAX_TOOL_NAME_CHARS} ASCII letters, digits, `_` or `-`"
  );
  Ok(())
}

/// A tool the model can call.
///
/// Uses `async_trait` rather than a native `async fn`: the registry keeps tools as
/// `Box<dyn Tool>`, and native async fns in traits are not dyn-compatible.
#[async_trait::async_trait]
pub trait Tool: Send + Sync {
  /// Name used both in the advertised definition and for dispatch; must be unique
  /// across the registry, otherwise only the first match is ever reached, and pass
  /// [`validate_tool_name`].
  fn name(&self) -> &str;

  /// What the tool does, and when the model should reach for it.
  fn description(&self) -> &str;

  /// JSON Schema for the arguments: an object schema, whose properties are the
  /// arguments. [`schema_of`] derives one from an argument struct.
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
  async fn execute(&self, args_json: &str) -> crate::Result<String>;

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

  /// The definition advertised to the model: [`Self::name`], [`Self::description`] and
  /// [`Self::parameters`], captured once when the tool is registered.
  fn definition(&self) -> ToolDefinition {
    ToolDefinition {
      name: self.name().to_owned(),
      description: self.description().to_owned(),
      parameters: self.parameters(),
    }
  }
}

/// The JSON Schema of a tool's argument type `T`, as the value for [`Tool::parameters`].
///
/// One place for what every tool otherwise writes out, including the `expect`: a schema
/// derived from a plain `Deserialize` struct always serializes, so a failure here is a
/// bug in the type, not something a caller could handle.
pub fn schema_of<T: schemars::JsonSchema>() -> Value {
  serde_json::to_value(schemars::schema_for!(T)).expect("schema is always serializable")
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
) -> crate::Result<String> {
  let limit = tool.timeout().unwrap_or(default);
  match tokio::time::timeout(limit, tool.execute(args_json)).await {
    Ok(result) => result,
    Err(_) if tool.concurrency_safe() => crate::error::bail!(
      tool,
      "`{}` did not finish within {}s and was abandoned; try a smaller or more specific \
       request",
      tool.name(),
      limit.as_secs_f64()
    ),
    Err(_) => crate::error::bail!(
      tool,
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
    async fn execute(&self, _args_json: &str) -> crate::Result<String> {
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
      async fn execute(&self, _args_json: &str) -> crate::Result<String> {
        Ok(String::new())
      }
    }
    assert!(!Plain.concurrency_safe());
  }

  #[test]
  fn the_definition_carries_name_description_and_schema() {
    let tool = Sleepy {
      own_limit: None,
      read_only: true,
    };
    assert_eq!(
      tool.definition(),
      ToolDefinition {
        name: "sleepy".to_owned(),
        description: "sleeps".to_owned(),
        parameters: serde_json::json!({ "type": "object" }),
      }
    );
  }

  #[test]
  fn tool_names_are_checked_against_what_apis_accept() {
    for good in [
      "calculator",
      "demo__get-sum",
      "a",
      &"x".repeat(MAX_TOOL_NAME_CHARS),
    ] {
      assert!(validate_tool_name(good).is_ok(), "`{good}` was rejected");
    }
    for bad in [
      "",
      "has space",
      "dotted.name",
      "日本",
      &"x".repeat(MAX_TOOL_NAME_CHARS + 1),
    ] {
      let err = validate_tool_name(bad).unwrap_err().to_string();
      assert!(err.contains("invalid tool name"), "`{bad}`: {err}");
    }
  }
}
