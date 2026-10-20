//! Observability initialization. The three entry points (`main` / `bin/gaia` / examples)
//! share one configuration.
//!
//! # Spans
//!
//! The library emits `tracing` spans named after the OpenTelemetry GenAI semantic
//! conventions, so a trace backend shows a run the way it shows any other agent:
//!
//! - `invoke_agent` — one [`crate::agent::Agent`] run (`otel.name` is
//!   `invoke_agent {name}` when the agent has one, see
//!   [`crate::agent::Agent::with_name`]). Records the run's outcome, rounds and token
//!   usage when it ends.
//! - `chat` — one model request, from [`crate::llm::provider::Provider::chat`] /
//!   [`crate::llm::provider::Provider::chat_stream`], i.e. *every* request this crate
//!   makes, retries included (each attempt is its own span). Records the response id,
//!   finish reasons and usage; a streamed request's span lasts until its stream ends.
//! - `execute_tool` — one tool call, approval hooks included. A call counts as an
//!   `error` unless it produced a successful result, so a refusal by an approval hook
//!   is one too; `agent.tool.outcome` says which of `success` / `error` / `suspended`.
//!
//! They nest: a sub-agent's `invoke_agent` sits under the parent's `execute_tool`.
//!
//! **Sum tokens over `chat` spans only.** `invoke_agent` carries the run's total *as a
//! convenience* — its own requests plus its sub-agents' and tools' — so a backend that
//! adds up `gen_ai.usage.*` across every span would count the same tokens twice.
//!
//! # Console
//!
//! With only the console subscriber, spans show up as a short `span{...}:` prefix on the
//! log lines emitted inside them. The prefix carries the identifying fields (agent, tool
//! and model names) and leaves out the ones that exist for the exporter — status codes,
//! token counts, ids — which would otherwise make every line several hundred characters
//! longer.
//!
//! # Exporting (`--features otel`)
//!
//! Built with the `otel` feature, [`init`] also installs an OpenTelemetry layer whenever
//! `OTEL_EXPORTER_OTLP_ENDPOINT` (or `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`) is set, and
//! exports spans to it over OTLP/HTTP (protobuf) — e.g. `http://localhost:4318` for a
//! local collector or Jaeger. The standard variables apply (`OTEL_SERVICE_NAME`,
//! defaulting to `agent`; `OTEL_RESOURCE_ATTRIBUTES`; `OTEL_EXPORTER_OTLP_HEADERS`;
//! `OTEL_SDK_DISABLED=true` turns it off).
//!
//! Spans are exported in batches from a background thread, every few seconds, so a
//! process that exits without flushing loses the last batch — for a short-lived one,
//! most of what it recorded. Keep the [`Telemetry`] guard that [`init`] returns alive for
//! the whole program (`let _telemetry = telemetry::init()?;` at the top of `main`):
//! dropping it flushes, on every way out of `main` including an early `?`. A process
//! that ends with `std::process::exit`, which runs no destructors, calls [`shutdown`]
//! itself first.
//!
//! The flush waits at most [`SHUTDOWN_TIMEOUT`] for the collector, so a collector that
//! accepts the connection and never answers delays the exit by that long, not forever.
//!
//! The console log level (`RUST_LOG`) does not limit what is exported: the exporter sees
//! `INFO` and above regardless, which is where every span above lives.

use std::{fmt, time::Duration};

use tracing::field::{Field, Visit};
use tracing_subscriber::{
  Layer,
  field::{RecordFields, VisitOutput},
  filter::LevelFilter,
  fmt::{
    FormatFields, FormattedFields,
    format::{DefaultVisitor, Writer},
  },
  layer::SubscriberExt,
};

use crate::config;

/// The longest [`shutdown`] waits for the exporter to hand its last batch over.
pub const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(2);

/// Load `.env` and install the global tracing subscriber.
///
/// Must be called before reading any [`crate::config`] configuration, otherwise values in
/// `.env` will not take effect. A missing `.env` is normal (e.g. when configuration is
/// injected via real environment variables), so it is not treated as an error.
///
/// Hold on to the returned guard until the program is done; see the module docs.
///
/// With the `otel` feature this can also fail on a malformed OTLP endpoint; an endpoint
/// that is well-formed but unreachable is not an error here — exports to it just fail in
/// the background.
pub fn init() -> anyhow::Result<Telemetry> {
  dotenvy::dotenv().ok();

  let console = tracing_subscriber::fmt::layer()
    .fmt_fields(ConsoleFields)
    .with_filter(LevelFilter::from_level(config::log_level()));
  let subscriber = tracing_subscriber::registry().with(console);
  #[cfg(feature = "otel")]
  let subscriber = subscriber.with(otel::layer()?);
  tracing::subscriber::set_global_default(subscriber)?;

  Ok(Telemetry { _private: () })
}

/// Flushes pending spans when dropped; returned by [`init`].
#[derive(Debug)]
#[must_use = "dropping the guard flushes the exporter, so bind it for the whole program: \
              `let _telemetry = telemetry::init()?;`"]
pub struct Telemetry {
  _private: (),
}

impl Drop for Telemetry {
  fn drop(&mut self) {
    shutdown();
  }
}

/// Flush spans still waiting to be exported and stop the exporter. A no-op without the
/// `otel` feature, or when no endpoint was configured, and harmless to call twice.
///
/// Blocks until the exporter has finished, or [`SHUTDOWN_TIMEOUT`] has passed. Normally
/// the [`Telemetry`] guard does this; call it directly only before `std::process::exit`.
pub fn shutdown() {
  #[cfg(feature = "otel")]
  otel::shutdown();
}

/// Span fields that exist for the exporter. They are noise in a log line — status codes,
/// token counts, ids — so the console leaves them out; see [`shown_on_console`].
const EXPORT_ONLY_PREFIXES: [&str; 4] = ["otel.", "gen_ai.", "agent.", "error."];

/// The exporter-only fields that still say *what* a span is about, which is the part of
/// the prefix worth reading in a log line.
const IDENTIFYING_FIELDS: [&str; 3] = [
  "gen_ai.agent.name",
  "gen_ai.tool.name",
  "gen_ai.request.model",
];

fn shown_on_console(field: &str) -> bool {
  IDENTIFYING_FIELDS.contains(&field)
    || !EXPORT_ONLY_PREFIXES
      .iter()
      .any(|prefix| field.starts_with(prefix))
}

/// The default field formatter, minus the fields [`shown_on_console`] rejects.
struct ConsoleFields;

impl<'writer> FormatFields<'writer> for ConsoleFields {
  fn format_fields<R: RecordFields>(&self, writer: Writer<'writer>, fields: R) -> fmt::Result {
    let mut visitor = ConsoleVisitor(DefaultVisitor::new(writer, true));
    fields.record(&mut visitor);
    visitor.0.finish()
  }

  /// Fields recorded after a span opened are nearly all export-only, so most of these
  /// calls add nothing. The default would still append a separating space each time.
  fn add_fields(
    &self,
    current: &'writer mut FormattedFields<Self>,
    fields: &tracing::span::Record<'_>,
  ) -> fmt::Result {
    let mut any = AnyShown(false);
    fields.record(&mut any);
    if !any.0 {
      return Ok(());
    }
    if !current.fields.is_empty() {
      current.fields.push(' ');
    }
    self.format_fields(current.as_writer(), fields)
  }
}

struct ConsoleVisitor<'a>(DefaultVisitor<'a>);

impl Visit for ConsoleVisitor<'_> {
  fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
    if shown_on_console(field.name()) {
      self.0.record_debug(field, value);
    }
  }

  fn record_str(&mut self, field: &Field, value: &str) {
    if shown_on_console(field.name()) {
      self.0.record_str(field, value);
    }
  }

  fn record_error(&mut self, field: &Field, value: &(dyn std::error::Error + 'static)) {
    if shown_on_console(field.name()) {
      self.0.record_error(field, value);
    }
  }
}

/// Whether any recorded field would be shown. Every `record_*` of [`Visit`] defaults to
/// `record_debug`, so one method sees them all.
struct AnyShown(bool);

impl Visit for AnyShown {
  fn record_debug(&mut self, field: &Field, _value: &dyn fmt::Debug) {
    self.0 |= shown_on_console(field.name());
  }
}

#[cfg(feature = "otel")]
mod otel {
  use std::sync::{
    OnceLock,
    atomic::{AtomicBool, Ordering},
  };

  use opentelemetry::trace::TracerProvider as _;
  use opentelemetry_sdk::{Resource, trace::SdkTracerProvider};
  use tracing::Subscriber;
  use tracing_subscriber::{Layer, filter::LevelFilter, registry::LookupSpan};

  use super::SHUTDOWN_TIMEOUT;

  /// Kept so [`shutdown`] can flush it; `tracing-opentelemetry` only holds a tracer.
  static PROVIDER: OnceLock<SdkTracerProvider> = OnceLock::new();

  /// Set by the first [`shutdown`], so the guard's drop after an explicit call (or the
  /// other way round) neither waits again nor reports the provider as already shut down.
  static SHUT_DOWN: AtomicBool = AtomicBool::new(false);

  const DEFAULT_SERVICE_NAME: &str = "agent";

  /// The export layer, or `None` when nothing asked for export.
  pub(super) fn layer<S>() -> anyhow::Result<Option<impl Layer<S>>>
  where
    S: Subscriber + for<'span> LookupSpan<'span>,
  {
    if !requested() {
      return Ok(None);
    }

    let exporter = opentelemetry_otlp::SpanExporter::builder()
      .with_http()
      .build()?;
    let provider = SdkTracerProvider::builder()
      .with_batch_exporter(exporter)
      .with_resource(resource())
      .build();
    let tracer = provider.tracer(env!("CARGO_PKG_NAME"));
    let _ = PROVIDER.set(provider);

    Ok(Some(
      tracing_opentelemetry::layer()
        .with_tracer(tracer)
        .with_filter(LevelFilter::INFO),
    ))
  }

  pub(super) fn shutdown() {
    if SHUT_DOWN.swap(true, Ordering::SeqCst) {
      return;
    }
    if let Some(provider) = PROVIDER.get()
      && let Err(err) = provider.shutdown_with_timeout(SHUTDOWN_TIMEOUT)
    {
      // Logged, not returned: the process is on its way out, and a collector that went
      // away first is no reason to exit unsuccessfully.
      eprintln!("failed to flush OpenTelemetry spans: {err}");
    }
  }

  /// Export only when an endpoint was configured: shipping spans to a default
  /// `localhost:4318` that nothing listens on would just be a stream of failed exports.
  fn requested() -> bool {
    let set = |key: &str| std::env::var(key).is_ok_and(|value| !value.trim().is_empty());
    let disabled = std::env::var("OTEL_SDK_DISABLED")
      .is_ok_and(|value| value.trim().eq_ignore_ascii_case("true"));
    !disabled && (set("OTEL_EXPORTER_OTLP_ENDPOINT") || set("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"))
  }

  /// The SDK's own detectors (`OTEL_SERVICE_NAME`, `OTEL_RESOURCE_ATTRIBUTES`), with
  /// this crate's name standing in for the SDK's `unknown_service` when neither names
  /// one.
  fn resource() -> Resource {
    let named = std::env::var_os("OTEL_SERVICE_NAME").is_some()
      || std::env::var("OTEL_RESOURCE_ATTRIBUTES")
        .is_ok_and(|attributes| attributes.contains("service.name"));
    let builder = Resource::builder();
    if named {
      builder.build()
    } else {
      builder.with_service_name(DEFAULT_SERVICE_NAME).build()
    }
  }
}

#[cfg(test)]
mod tests {
  use std::sync::{Arc, Mutex};

  use super::*;

  #[test]
  fn only_the_fields_that_identify_a_span_reach_the_console() {
    for shown in [
      "gen_ai.agent.name",
      "gen_ai.tool.name",
      "gen_ai.request.model",
      "tool",
      "message",
    ] {
      assert!(shown_on_console(shown), "{shown}");
    }
    for hidden in [
      "otel.name",
      "otel.status_code",
      "gen_ai.usage.input_tokens",
      "gen_ai.conversation.id",
      "gen_ai.tool.call.id",
      "agent.execution_id",
      "agent.outcome",
      "error.type",
    ] {
      assert!(!shown_on_console(hidden), "{hidden}");
    }
  }

  #[derive(Clone, Default)]
  struct Captured(Arc<Mutex<Vec<u8>>>);

  impl std::io::Write for Captured {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
      self.0.lock().unwrap().extend_from_slice(bytes);
      Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
      Ok(())
    }
  }

  /// A log line inside a span: the span's name and identifying fields, not the rest, and
  /// no stray padding left behind by fields recorded afterwards.
  #[test]
  fn a_log_line_carries_a_short_span_prefix() {
    let captured = Captured::default();
    let writer = captured.clone();
    let subscriber = tracing_subscriber::fmt()
      .fmt_fields(ConsoleFields)
      .with_ansi(false)
      .without_time()
      .with_writer(move || writer.clone())
      .finish();

    tracing::subscriber::with_default(subscriber, || {
      let span = tracing::info_span!(
        "invoke_agent",
        otel.name = "invoke_agent lead",
        gen_ai.operation.name = "invoke_agent",
        gen_ai.agent.name = "lead",
        gen_ai.request.model = "big",
        agent.execution_id = "0123456789abcdef",
        agent.outcome = tracing::field::Empty,
        gen_ai.usage.input_tokens = tracing::field::Empty,
      );
      let _entered = span.enter();
      span.record("agent.outcome", "done");
      span.record("gen_ai.usage.input_tokens", 330_i64);
      tracing::info!(tool = "calculator", "executing tool");
    });

    let line = String::from_utf8(captured.0.lock().unwrap().clone()).unwrap();
    assert_eq!(
      line.trim(),
      "INFO invoke_agent{gen_ai.agent.name=\"lead\" gen_ai.request.model=\"big\"}: \
       agent::telemetry::tests: executing tool tool=\"calculator\"",
      "{line}"
    );
  }
}
