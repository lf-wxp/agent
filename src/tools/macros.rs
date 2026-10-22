//! Shared boilerplate for tools whose `Tool` impl is entirely mechanical.

/// Implements [`Tool`](crate::tools::tool::Tool) for a tool whose parameters are
/// derived via `schemars` and whose `execute` is a single call into its `execute`
/// module.
///
/// Not used by `calculator` / `web_search`: their schemas are hand-written for real
/// reasons (see their `definition.rs`), and folding them into this macro would hide
/// that reasoning behind indirection.
///
/// `run` takes a function *path* (e.g. `execute::run`), not a call: the macro applies
/// it to `args_json` itself, using `async run = ...` when that function needs an
/// `.await`. A call expression like `execute::run(args_json)` can't be accepted here —
/// `args_json` would be an identifier from the caller's side of the macro boundary,
/// which macro hygiene keeps distinct from the `args_json` parameter generated below.
///
/// Only two arms — `run = ...` and `async run = ...` — because that `.await` can't be
/// made conditional within a single arm: `macro_rules!` has no way to inspect whether
/// `$run` is itself `async`, so the caller has to say which one it is, and that
/// necessarily produces two different token streams for the `execute` body.
///
/// Each arm builds its own complete `execute` method (signature *and* body) and hands
/// it to [`impl_simple_tool`] as one opaque block, rather than handing over just the
/// call expression: `impl_simple_tool` re-emits those tokens verbatim without ever
/// writing its own `args_json`, so there is no second, differently-hygiene-coloured
/// `args_json` for the passed-in one to clash with. Splitting the signature (which
/// declares `args_json`) from the body (which uses it) across the two macros would
/// reintroduce exactly the hygiene error described above — so the `name`/`description`/
/// `parameters` boilerplate is all that gets factored out here.
///
/// A synchronous `run` is executed on tokio's blocking pool (`spawn_blocking`), not
/// inline: the file tools use `std::fs`, and a large read or a recursive delete would
/// otherwise stall the async worker thread — and every other task scheduled on it — for
/// as long as the syscall takes.
///
/// Two optional trailing settings, in this order:
///
/// - `timeout = <Duration expr>` overrides
///   [`Tool::timeout`](crate::tools::tool::Tool::timeout).
/// - `concurrency_safe = <bool expr>` overrides
///   [`Tool::concurrency_safe`](crate::tools::tool::Tool::concurrency_safe). Left out, the
///   tool is *not* concurrency-safe, so a new tool has to opt in to overlapping with its
///   siblings rather than opt out.
macro_rules! simple_tool {
  (
    $tool:ident,
    $args:ty,
    name = $name:expr,
    description = $description:expr,
    run = $run:path
    $(, timeout = $timeout:expr)?
    $(, concurrency_safe = $safe:expr)? $(,)?
  ) => {
    $crate::tools::macros::impl_simple_tool!(
      $tool, $args, $name, $description, [$($timeout)?], [$($safe)?], {
        async fn execute(&self, args_json: &str) -> anyhow::Result<String> {
          let args_json = args_json.to_owned();
          tokio::task::spawn_blocking(move || $run(&args_json))
            .await
            .map_err(|err| anyhow::anyhow!("tool task panicked: {err}"))?
        }
      }
    );
  };

  (
    $tool:ident,
    $args:ty,
    name = $name:expr,
    description = $description:expr,
    async run = $run:path
    $(, timeout = $timeout:expr)?
    $(, concurrency_safe = $safe:expr)? $(,)?
  ) => {
    $crate::tools::macros::impl_simple_tool!(
      $tool, $args, $name, $description, [$($timeout)?], [$($safe)?], {
        async fn execute(&self, args_json: &str) -> anyhow::Result<String> {
          $run(args_json).await
        }
      }
    );
  };
}

/// The part of [`simple_tool`] that stays identical regardless of `run` vs
/// `async run`: the struct, and every `Tool` method except `execute`.
///
/// Not meant to be used directly — [`simple_tool`] is the public entry point, this only
/// exists so both of its arms have somewhere to funnel into instead of repeating the
/// struct/`impl` wrapper. `$execute_fn` arrives pre-built (see [`simple_tool`] for why)
/// and is spliced back in unmodified.
///
/// The two optional settings arrive bracketed (`[]` when absent) so that either can be
/// left out without the other's position becoming ambiguous.
macro_rules! impl_simple_tool {
  (
    $tool:ident, $args:ty, $name:expr, $description:expr,
    [$($timeout:expr)?], [$($safe:expr)?], { $($execute_fn:tt)* }
  ) => {
    pub struct $tool;

    #[async_trait::async_trait]
    impl $crate::tools::tool::Tool for $tool {
      fn name(&self) -> &str {
        $name
      }

      fn description(&self) -> &str {
        $description
      }

      fn parameters(&self) -> serde_json::Value {
        $crate::tools::schema_of::<$args>()
      }

      $(
        fn timeout(&self) -> Option<std::time::Duration> {
          Some($timeout)
        }
      )?

      $(
        fn concurrency_safe(&self) -> bool {
          $safe
        }
      )?

      $($execute_fn)*
    }
  };
}

pub(crate) use {impl_simple_tool, simple_tool};
