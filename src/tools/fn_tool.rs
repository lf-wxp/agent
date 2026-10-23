//! [`FnTool`]: a [`Tool`] made from an argument type and an async function, for the
//! common case where writing out a struct and a trait impl is all ceremony.
//!
//! ```no_run
//! use std::sync::Arc;
//!
//! use agent::tools::{FnTool, ToolRegistry};
//! use schemars::JsonSchema;
//! use serde::Deserialize;
//!
//! /// The schema the model sees is derived from this type, doc comments included.
//! #[derive(Deserialize, JsonSchema)]
//! struct WeatherArgs {
//!   /// City name, e.g. "Lisbon".
//!   city: String,
//! }
//!
//! # fn main() -> agent::Result<()> {
//! let weather = FnTool::new(
//!   "weather",
//!   "Current weather for a city.",
//!   |args: WeatherArgs| async move { Ok(format!("Sunny in {}", args.city)) },
//! )?
//! .read_only();
//!
//! let mut registry = ToolRegistry::builtin()?;
//! registry.add(Arc::new(weather))?;
//! # Ok(())
//! # }
//! ```
//!
//! State the tool needs (a client, a database handle) is captured by the closure; clone
//! it into the `async move` block, since the closure runs once per call. A foreign error
//! becomes the tool's error with [`ResultExt::wrap_err`](crate::error::ResultExt::wrap_err),
//! whose message is what the model reads:
//!
//! ```ignore
//! use agent::error::{Error, ResultExt};
//!
//! let client = reqwest::Client::new();
//! FnTool::new("fetch", "Fetch a URL.", move |args: FetchArgs| {
//!   let client = client.clone();
//!   async move {
//!     let response = client.get(&args.url).send().await
//!       .wrap_err(Error::tool, format!("could not reach `{}`", args.url))?;
//!     response.text().await.wrap_err(Error::tool, "the page could not be read")
//!   }
//! })?
//! ```

use std::{fmt, future::Future, marker::PhantomData, time::Duration};

use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::tools::tool::{Tool, schema_of, validate_tool_name};

/// A [`Tool`] whose arguments are `A`, parsed from the model's JSON, and whose work is
/// `handler`. See the module docs for an example.
///
/// - The parameter schema is [`schema_of::<A>`](schema_of): derive `JsonSchema` on a
///   struct, and its fields (with their doc comments) are what the model fills in. A
///   tool with no arguments takes an empty struct, `struct NoArgs {}` — not a unit struct,
///   whose schema is not an object.
/// - The handler returns [`crate::Result<R>`](crate::Result) for any `R: Into<String>`
///   (`String`, `&'static str`, ...). An `Err` goes back to the model as the tool's error,
///   like any other tool's (see [`crate::tools::ToolRegistry::execute`]), so make its
///   message say what to do differently — [`crate::Error::tool`] for the tool's own
///   failures, [`crate::error::ResultExt::wrap_err`] around a foreign one.
/// - Arguments that do not parse are reported to the model with serde's reason (a
///   missing field, a wrong type), so it can correct the call. Empty arguments are read
///   as `{}`, which some models send for a tool without parameters.
/// - Like any tool, it is not [`Tool::concurrency_safe`] unless told so
///   ([`Self::read_only`]), and uses the caller's time limit unless given its own
///   ([`Self::with_timeout`]).
pub struct FnTool<A, F> {
  name: String,
  description: String,
  parameters: Value,
  handler: F,
  timeout: Option<Duration>,
  read_only: bool,
  /// `fn(A)` rather than `A`: the tool only ever *receives* an `A`, so it is `Send` and
  /// `Sync` whatever `A` is.
  _args: PhantomData<fn(A)>,
}

impl<A, F, Fut, R> FnTool<A, F>
where
  A: DeserializeOwned + JsonSchema + Send + 'static,
  F: Fn(A) -> Fut + Send + Sync + 'static,
  Fut: Future<Output = crate::Result<R>> + Send + 'static,
  R: Into<String>,
{
  /// A tool called `name`, described to the model as `description`, running `handler`
  /// with the parsed arguments.
  ///
  /// # Errors
  ///
  /// When `name` is not one chat APIs accept (see
  /// [`crate::tools::tool::validate_tool_name`]), when `description` is blank (the model
  /// has nothing to decide by), or when `A`'s schema is not an object schema — all
  /// mistakes in the code, caught here rather than as a rejected request.
  pub fn new(
    name: impl Into<String>,
    description: impl Into<String>,
    handler: F,
  ) -> crate::Result<Self> {
    let name = name.into();
    let description = description.into();
    validate_tool_name(&name)?;
    crate::error::ensure!(
      !description.trim().is_empty(),
      invalid_input,
      "tool `{name}` needs a description, or the model cannot tell when to use it"
    );
    let parameters = schema_of::<A>();
    crate::error::ensure!(
      parameters.get("type").and_then(Value::as_str) == Some("object"),
      invalid_input,
      "the arguments of tool `{name}` must be a struct with named fields (an object \
       schema); use `struct NoArgs {{}}` for a tool without arguments"
    );
    Ok(Self {
      name,
      description,
      parameters,
      handler,
      timeout: None,
      read_only: false,
      _args: PhantomData,
    })
  }
}

impl<A, F> FnTool<A, F> {
  /// Give up on a call after `timeout` instead of the caller's default (see
  /// [`Tool::timeout`]).
  #[must_use]
  pub fn with_timeout(mut self, timeout: Duration) -> Self {
    self.timeout = Some(timeout);
    self
  }

  /// Declare that the tool changes nothing outside itself, so calls to it may overlap
  /// their siblings in one round (see [`Tool::concurrency_safe`]). Only for tools that
  /// really only read: two overlapping calls of a tool that writes can lose an update.
  #[must_use]
  pub fn read_only(mut self) -> Self {
    self.read_only = true;
    self
  }
}

impl<A, F> fmt::Debug for FnTool<A, F> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("FnTool")
      .field("name", &self.name)
      .field("timeout", &self.timeout)
      .field("read_only", &self.read_only)
      .finish_non_exhaustive()
  }
}

#[async_trait::async_trait]
impl<A, F, Fut, R> Tool for FnTool<A, F>
where
  A: DeserializeOwned + JsonSchema + Send + 'static,
  F: Fn(A) -> Fut + Send + Sync + 'static,
  Fut: Future<Output = crate::Result<R>> + Send + 'static,
  R: Into<String>,
{
  fn name(&self) -> &str {
    &self.name
  }

  fn description(&self) -> &str {
    &self.description
  }

  fn parameters(&self) -> Value {
    self.parameters.clone()
  }

  async fn execute(&self, args_json: &str) -> crate::Result<String> {
    let args_json = if args_json.trim().is_empty() {
      "{}"
    } else {
      args_json
    };
    // serde's reason goes in the message itself and is *not* kept as a cause: the model
    // is only shown the outermost message, so "invalid arguments" alone would give it
    // nothing to fix — and keeping both would print the same sentence twice in `{err:#}`.
    let args: A = serde_json::from_str(args_json)
      .map_err(|err| crate::Error::tool(format!("invalid arguments for `{}`: {err}", self.name)))?;
    Ok((self.handler)(args).await?.into())
  }

  fn timeout(&self) -> Option<Duration> {
    self.timeout
  }

  fn concurrency_safe(&self) -> bool {
    self.read_only
  }
}

#[cfg(test)]
mod tests {
  use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
  };

  use serde::Deserialize;
  use serde_json::json;

  use super::*;
  use crate::{
    agent::Agent,
    llm::{
      settings::ModelSettings,
      testing::{ScriptedBackend, answer, tool_call},
    },
    tools::ToolRegistry,
  };

  #[derive(Deserialize, JsonSchema)]
  struct AddArgs {
    /// The first number.
    a: i64,
    /// The second number.
    b: i64,
  }

  fn adder() -> FnTool<AddArgs, impl Fn(AddArgs) -> futures::future::Ready<crate::Result<String>>> {
    FnTool::new("add", "Add two integers.", |args: AddArgs| {
      futures::future::ready(Ok((args.a + args.b).to_string()))
    })
    .unwrap()
  }

  #[test]
  fn the_schema_comes_from_the_argument_type() {
    let definition = adder().definition();
    assert_eq!(definition.name, "add");
    assert_eq!(definition.description, "Add two integers.");
    assert_eq!(definition.parameters["type"], "object");
    assert_eq!(definition.parameters["required"], json!(["a", "b"]));
    assert_eq!(
      definition.parameters["properties"]["a"]["description"],
      "The first number."
    );
  }

  #[tokio::test]
  async fn a_call_parses_the_arguments_and_runs_the_handler() {
    assert_eq!(adder().execute(r#"{"a": 2, "b": 3}"#).await.unwrap(), "5");
  }

  /// The model is told what was wrong, so it can send a corrected call.
  #[tokio::test]
  async fn bad_arguments_are_reported_with_serdes_reason() {
    let err = adder()
      .execute(r#"{"a": 2}"#)
      .await
      .unwrap_err()
      .to_string();
    assert!(
      err.starts_with("invalid arguments for `add`: "),
      "got: {err}"
    );
    assert!(err.contains("missing field `b`"), "got: {err}");

    let err = adder()
      .execute(r#"{"a": "two", "b": 3}"#)
      .await
      .unwrap_err()
      .to_string();
    assert!(err.contains("invalid type"), "got: {err}");
  }

  #[derive(Deserialize, JsonSchema)]
  struct NoArgs {}

  /// Some models send `""` for a tool without parameters.
  #[tokio::test]
  async fn empty_arguments_are_read_as_an_empty_object() {
    let tool = FnTool::new("ping", "Ping.", |_: NoArgs| async { Ok("pong") }).unwrap();
    assert_eq!(tool.execute("").await.unwrap(), "pong");
    assert_eq!(tool.execute("  ").await.unwrap(), "pong");
    assert_eq!(tool.execute("{}").await.unwrap(), "pong");
  }

  #[tokio::test]
  async fn the_handlers_error_is_the_tools_error() {
    let tool = FnTool::new("fail", "Always fails.", |_: NoArgs| async {
      Err::<&str, _>(crate::Error::tool("the service is down; try again later"))
    })
    .unwrap();
    let err = tool.execute("{}").await.unwrap_err();
    assert!(matches!(err, crate::Error::Tool(_)));
    assert_eq!(err.to_string(), "the service is down; try again later");
  }

  /// What a handler does with an error from another crate: name what failed, keep the
  /// original as the cause. The model reads the message; a log reads the whole chain.
  #[tokio::test]
  async fn a_foreign_error_is_wrapped_with_a_message_the_model_can_use() {
    use crate::error::ResultExt;

    #[derive(Deserialize, JsonSchema)]
    struct PortArgs {
      port: String,
    }
    let tool = FnTool::new("port", "Parse a port.", |args: PortArgs| async move {
      let port: u16 = args.port.parse().wrap_err(
        crate::Error::tool,
        format!("`{}` is not a port number", args.port),
      )?;
      Ok(port.to_string())
    })
    .unwrap();

    let err = tool.execute(r#"{"port": "http"}"#).await.unwrap_err();
    assert_eq!(err.to_string(), "`http` is not a port number");
    assert_eq!(
      format!("{err:#}"),
      "`http` is not a port number: invalid digit found in string"
    );
  }

  #[test]
  fn timeout_and_read_only_are_opt_in() {
    let tool = adder();
    assert_eq!(tool.timeout(), None);
    assert!(!tool.concurrency_safe());

    let tool = tool.with_timeout(Duration::from_secs(5)).read_only();
    assert_eq!(tool.timeout(), Some(Duration::from_secs(5)));
    assert!(tool.concurrency_safe());
  }

  /// A unit struct deserializes from `null`, never from the `{}` the model sends, so it
  /// is caught when the tool is built rather than at every call.
  #[test]
  fn a_bad_name_or_a_non_object_schema_is_caught_when_built() {
    #[derive(Deserialize, JsonSchema)]
    struct Unit;

    let err = FnTool::new("has space", "Does something.", |_: NoArgs| async { Ok("") })
      .unwrap_err()
      .to_string();
    assert!(err.contains("invalid tool name"), "got: {err}");

    let err = FnTool::new("unit", "Does something.", |_: Unit| async { Ok("") })
      .unwrap_err()
      .to_string();
    assert!(err.contains("object schema"), "got: {err}");

    // Checked here rather than only on the first call: a blank description tells the
    // model nothing, and the same mistake is caught for `AgentTool::new`.
    let err = FnTool::new("blank", "  ", |_: NoArgs| async { Ok("") })
      .unwrap_err()
      .to_string();
    assert!(err.contains("needs a description"), "got: {err}");
  }

  /// State lives in the closure; every call sees the same.
  #[tokio::test]
  async fn the_handler_keeps_captured_state_across_calls() {
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&calls);
    let tool = FnTool::new("count", "Count calls.", move |_: NoArgs| {
      let counter = Arc::clone(&counter);
      async move { Ok((counter.fetch_add(1, Ordering::SeqCst) + 1).to_string()) }
    })
    .unwrap();
    assert_eq!(tool.execute("{}").await.unwrap(), "1");
    assert_eq!(tool.execute("{}").await.unwrap(), "2");
    assert_eq!(calls.load(Ordering::SeqCst), 2);
  }

  /// End to end: the agent advertises it, the model calls it, the result goes back.
  #[tokio::test]
  async fn an_agent_offers_and_calls_it_like_any_tool() {
    let backend = ScriptedBackend::replying(vec![
      tool_call("call_1", "add", json!({ "a": 20, "b": 22 })),
      answer("It is 42."),
    ]);
    let mut toolbox = ToolRegistry::empty();
    toolbox.add(Arc::new(adder())).unwrap();
    let agent = Agent::new(
      backend.provider(),
      "gpt-test",
      Option::<String>::None,
      Arc::new(toolbox),
    )
    .with_model_settings(ModelSettings::default());

    let result = agent.run("what is 20 + 22").await.unwrap();

    assert_eq!(result.output, "It is 42.");
    let sent = backend.seen();
    assert_eq!(sent[0]["tools"][0]["function"]["name"], "add");
    let tool_message = sent[1]["messages"]
      .as_array()
      .unwrap()
      .iter()
      .find(|message| message["role"] == "tool")
      .expect("the tool result was sent back");
    assert_eq!(tool_message["content"], "42");
  }
}
