//! What every run is wrapped in: its `invoke_agent` span, and the usage ledger that
//! picks up what its tools and hooks spend (see [`crate::agent::usage`]).
//!
//! One place for both because they open and close together. The run's future (or
//! stream) is polled inside the span and the ledger, and whatever the run hands back —
//! a result, a suspension, a cancellation — gets the ledger settled into its context
//! and its outcome recorded on the span, on the way out.

use std::{
  pin::Pin,
  task::{Context as TaskContext, Poll},
};

use futures::Stream;
use tracing::{Instrument, Span, field::Empty};

use super::{Agent, AgentOutcome, AgentStreamEvent, StructuredAgentResult};
use crate::{
  agent::{
    ContentItem, ExecutionContext, SuspendedToolCall, TokenUsage, ToolResultStatus, usage::Ledger,
  },
  llm::spans::record_error,
};

/// Span and ledger for one run; see the module docs.
pub(super) struct RunScope {
  span: Span,
  ledger: Ledger,
  /// What the context had already spent when this run (or this leg of a resumed one)
  /// started, so the span reports only what was spent under it.
  start: TokenUsage,
}

impl RunScope {
  pub(super) fn new(agent: &Agent, context: &ExecutionContext) -> Self {
    let span = tracing::info_span!(
      "invoke_agent",
      otel.name = %agent.name.as_deref().map_or_else(
        || "invoke_agent".to_owned(),
        |name| format!("invoke_agent {name}"),
      ),
      otel.status_code = Empty,
      otel.status_description = Empty,
      gen_ai.operation.name = "invoke_agent",
      gen_ai.agent.name = Empty,
      gen_ai.request.model = %agent.model,
      gen_ai.conversation.id = Empty,
      agent.execution_id = %context.execution_id,
      agent.outcome = Empty,
      agent.rounds = Empty,
      gen_ai.usage.input_tokens = Empty,
      gen_ai.usage.output_tokens = Empty,
      error.type = Empty,
    );
    if let Some(name) = &agent.name {
      span.record("gen_ai.agent.name", name.as_str());
    }
    if let Some(id) = &context.conversation_id {
      span.record("gen_ai.conversation.id", id.as_str());
    }
    Self {
      span,
      ledger: Ledger::for_new_run(),
      start: context.usage,
    }
  }

  /// Run `future` inside the span and the ledger.
  pub(super) async fn run<F: Future>(&self, future: F) -> F::Output {
    self
      .ledger
      .scope(future)
      .instrument(self.span.clone())
      .await
  }

  /// Close out a plain run's result.
  pub(super) fn finish(&self, result: crate::Result<AgentOutcome>) -> crate::Result<AgentOutcome> {
    match result {
      Ok(AgentOutcome::Done(mut done)) => {
        self.settle(&mut done.context, "done");
        Ok(AgentOutcome::Done(done))
      }
      Ok(AgentOutcome::Suspended(mut state)) => {
        self.settle(&mut state.context, "suspended");
        Ok(AgentOutcome::Suspended(state))
      }
      Err(err) => Err(self.fail(err)),
    }
  }

  /// Close out a structured run's result.
  pub(super) fn finish_structured<T>(
    &self,
    result: crate::Result<StructuredAgentResult<T>>,
  ) -> crate::Result<StructuredAgentResult<T>> {
    match result {
      Ok(mut done) => {
        self.settle(&mut done.context, "done");
        Ok(done)
      }
      Err(err) => Err(self.fail(err)),
    }
  }

  /// Wrap a streaming run: every poll happens inside the span and the ledger, and the
  /// item that ends the run is settled before the caller sees it.
  pub(super) fn wrap_stream<'a, S>(
    self,
    inner: S,
  ) -> impl Stream<Item = crate::Result<AgentStreamEvent>> + 'a
  where
    S: Stream<Item = crate::Result<AgentStreamEvent>> + 'a,
  {
    ScopedStream {
      inner: Box::pin(inner),
      scope: self,
    }
  }

  fn settle_event(
    &self,
    event: crate::Result<AgentStreamEvent>,
  ) -> crate::Result<AgentStreamEvent> {
    match event {
      Ok(AgentStreamEvent::Done {
        output,
        mut context,
        budget_exhausted,
      }) => {
        self.settle(&mut context, "done");
        Ok(AgentStreamEvent::Done {
          output,
          context,
          budget_exhausted,
        })
      }
      Ok(AgentStreamEvent::Suspended(mut state)) => {
        self.settle(&mut state.context, "suspended");
        Ok(AgentStreamEvent::Suspended(state))
      }
      Err(err) => Err(self.fail(err)),
      other => other,
    }
  }

  /// A failed run still settles when it carries its transcript ([`crate::Cancelled`]), so a
  /// caller persisting a cancelled turn persists what it cost too.
  fn fail(&self, mut err: crate::Error) -> crate::Error {
    if let crate::Error::Cancelled(cancelled) = &mut err {
      self.settle(&mut cancelled.context, "cancelled");
    } else {
      self.span.record("agent.outcome", "error");
      record_error(&self.span, &err);
    }
    err
  }

  fn settle(&self, context: &mut ExecutionContext, outcome: &'static str) {
    self.ledger.settle(context);
    // Counts as `i64`; see `crate::llm::spans` for why not `u32`.
    let spent = |now: u32, before: u32| i64::from(now.saturating_sub(before));
    let span = &self.span;
    span.record("agent.outcome", outcome);
    span.record("agent.rounds", i64::from(context.current_step));
    span.record(
      "gen_ai.usage.input_tokens",
      spent(context.usage.prompt_tokens, self.start.prompt_tokens),
    );
    span.record(
      "gen_ai.usage.output_tokens",
      spent(
        context.usage.completion_tokens,
        self.start.completion_tokens,
      ),
    );
  }
}

/// See [`RunScope::wrap_stream`]. Generic rather than boxed as `dyn Stream`, so the
/// wrapped stream stays `Send` exactly when the run's own stream is.
struct ScopedStream<S> {
  inner: Pin<Box<S>>,
  scope: RunScope,
}

impl<S> Stream for ScopedStream<S>
where
  S: Stream<Item = crate::Result<AgentStreamEvent>>,
{
  type Item = crate::Result<AgentStreamEvent>;

  fn poll_next(self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
    // Both fields are `Unpin` (the inner stream is boxed), so the wrapper is too.
    let this = self.get_mut();
    let _entered = this.scope.span.enter();
    let inner = &mut this.inner;
    let polled = this
      .scope
      .ledger
      .sync_scope(|| inner.as_mut().poll_next(cx));
    match polled {
      Poll::Ready(Some(event)) => Poll::Ready(Some(this.scope.settle_event(event))),
      other => other,
    }
  }
}

/// A span for one tool call, approval hooks and all.
pub(super) fn tool_span(name: &str, call_id: &str) -> Span {
  tracing::info_span!(
    "execute_tool",
    otel.name = %format!("execute_tool {name}"),
    otel.status_code = Empty,
    otel.status_description = Empty,
    gen_ai.operation.name = "execute_tool",
    gen_ai.tool.name = %name,
    gen_ai.tool.call.id = %call_id,
    gen_ai.tool.type = "function",
    agent.tool.outcome = Empty,
  )
}

/// Record how one call of [`Agent::execute_tool_calls`] settled.
///
/// Any call that did not produce a successful result is marked as an error, a refusal by
/// an approval hook included: a hook picks the status it returns, and nothing at this
/// level can tell "refused" from "failed". `agent.tool.outcome` carries the same split
/// (`success` / `error` / `suspended`) for a backend that wants to query it.
pub(super) fn record_tool_outcome(span: &Span, settled: &Result<SuspendedToolCall, ContentItem>) {
  match settled {
    Ok(_) => {
      span.record("agent.tool.outcome", "suspended");
    }
    Err(ContentItem::ToolResult {
      status: ToolResultStatus::Success,
      ..
    }) => {
      span.record("agent.tool.outcome", "success");
    }
    Err(ContentItem::ToolResult { content, .. }) => {
      span.record("agent.tool.outcome", "error");
      span.record("otel.status_code", "error");
      span.record(
        "otel.status_description",
        crate::util::truncate_chars(content, 300),
      );
    }
    Err(_) => {}
  }
}

#[cfg(test)]
mod tests {
  use std::{
    collections::{BTreeMap, HashMap},
    sync::{Arc, Mutex, OnceLock},
  };

  use futures::StreamExt;
  use serde_json::{Value, json};
  use tracing::{
    Subscriber,
    field::{Field, Visit},
    span::{Attributes, Id, Record},
  };
  use tracing_subscriber::{Layer, layer::Context, prelude::*, registry::LookupSpan};

  use crate::{
    agent::{
      Agent, AgentOutcome, AgentStreamEvent, AgentTool, BeforeToolCallback, Conversation,
      ExecutionContext, ResumedDecision, ToolCallDecision, ToolCallView,
    },
    llm::{
      settings::ModelSettings,
      testing::{ScriptedBackend, answer, chunk, tool_call, with_usage},
    },
    tools::ToolRegistry,
  };

  fn agent_on(backend: &Arc<ScriptedBackend>, model: &str, toolbox: ToolRegistry) -> Agent {
    Agent::new(
      backend.provider(),
      model,
      Option::<String>::None,
      Arc::new(toolbox),
    )
    .with_model_settings(ModelSettings::default())
  }

  fn with_helper(helper: Agent, parent_toolbox: &mut ToolRegistry) {
    let helper = AgentTool::new("helper", "helps", helper).unwrap();
    parent_toolbox.add(Arc::new(helper)).unwrap();
  }

  /// A parent on `big` delegating to a sub-agent on `small`: two parent requests around
  /// one delegation, the sub-agent answering in one. Both are named after `label`, which
  /// is how a test finds its own spans among everything the test binary records.
  fn delegation(label: &str, sub_replies: Vec<Value>) -> Agent {
    let sub = ScriptedBackend::replying(sub_replies);
    let parent = ScriptedBackend::replying(vec![
      with_usage(
        tool_call("call_1", "helper", json!({ "task": "help" })),
        100,
        10,
      ),
      with_usage(answer("done"), 200, 20),
    ]);
    let mut toolbox = ToolRegistry::empty();
    with_helper(
      agent_on(&sub, "small", ToolRegistry::empty()).with_name(format!("{label}-helper")),
      &mut toolbox,
    );
    agent_on(&parent, "big", toolbox).with_name(format!("{label}-lead"))
  }

  /// The parent's usage covers the sub-agent's spend, each under its own model —
  /// what pricing needs.
  #[tokio::test]
  async fn a_parent_run_counts_what_its_sub_agent_spent_under_the_sub_agents_model() {
    let lead = delegation("usage", vec![with_usage(answer("helped"), 30, 3)]);

    let result = lead.run("go").await.unwrap();

    let usage = &result.context.usage_by_model;
    assert_eq!(usage["big"].prompt_tokens, 300);
    assert_eq!(usage["small"].prompt_tokens, 30);
    assert_eq!(result.context.usage.prompt_tokens, 330);
    assert_eq!(result.context.usage.completion_tokens, 33);
  }

  /// A sub-agent that fails after spending tokens has still spent them. It needs a tool
  /// to get as far as a second request, which is the one that fails.
  #[tokio::test]
  async fn a_failed_delegation_still_counts_what_it_spent() {
    let mut no_choices = answer("unused");
    no_choices["choices"] = json!([]);
    let sub = ScriptedBackend::replying(vec![
      with_usage(
        tool_call(
          "sub_1",
          "calculator",
          json!({ "operator": "add", "first_number": 1, "second_number": 2 }),
        ),
        40,
        4,
      ),
      no_choices,
    ]);
    let parent = ScriptedBackend::replying(vec![
      tool_call("call_1", "helper", json!({ "task": "help" })),
      answer("the helper failed"),
    ]);
    let calculator = ToolRegistry::select(&["calculator".to_owned()]).unwrap();
    let mut toolbox = ToolRegistry::empty();
    with_helper(agent_on(&sub, "small", calculator), &mut toolbox);
    let lead = agent_on(&parent, "big", toolbox);

    let result = lead.run("go").await.unwrap();

    assert_eq!(
      result.context.usage_by_model["small"].prompt_tokens, 40,
      "the failed sub-agent's first round was paid for"
    );
  }

  /// A streamed reply that asks for a tool: the call, then (after the tool) the answer.
  fn streamed_calculator_round() -> Vec<Vec<Value>> {
    vec![
      vec![chunk(json!({
        "tool_calls": [{
          "index": 0, "id": "call_1", "type": "function",
          "function": {
            "name": "calculator",
            "arguments": "{\"operator\":\"add\",\"first_number\":1,\"second_number\":2}",
          },
        }],
      }))],
      vec![chunk(json!({ "content": "done" }))],
    ]
  }

  async fn drain(agent: &Agent) -> Option<ExecutionContext> {
    let stream = agent.run_stream("go");
    futures::pin_mut!(stream);
    let mut done = None;
    while let Some(event) = stream.next().await {
      if let AgentStreamEvent::Done { context, .. } = event.unwrap() {
        done = Some(context);
      }
    }
    done
  }

  /// Same accounting on the streaming path: `Done` carries the sub-agent's share too.
  #[tokio::test]
  async fn a_streaming_parent_counts_its_sub_agent_too() {
    let sub = ScriptedBackend::replying(vec![with_usage(answer("helped"), 30, 3)]);
    let parent = ScriptedBackend::streaming(vec![
      vec![chunk(json!({
        "tool_calls": [{
          "index": 0, "id": "call_1", "type": "function",
          "function": { "name": "helper", "arguments": "{\"task\":\"help\"}" },
        }],
      }))],
      vec![chunk(json!({ "content": "done" }))],
    ]);
    let mut toolbox = ToolRegistry::empty();
    with_helper(agent_on(&sub, "small", ToolRegistry::empty()), &mut toolbox);
    let lead = agent_on(&parent, "big", toolbox);

    let context = drain(&lead).await.expect("the stream ends with Done");

    assert_eq!(context.usage_by_model["small"].prompt_tokens, 30);
  }

  // ---- spans --------------------------------------------------------------------------
  //
  // One recorder for the whole test binary, installed as the *global* default. A
  // per-test `set_default` is not enough: `tracing` caches, per callsite, whether any
  // subscriber wants it, and a callsite first reached on a thread that has no subscriber
  // — while another test's is the only one alive — is cached as "nobody does". The spans
  // that follow, in whichever test happens to run next, silently go missing. A global
  // default is visible from every thread, so the cache can never be poisoned that way.
  // The price is that every test's spans land here, which is why each test names its
  // agents and looks only at the spans below its own.

  #[derive(Debug, Clone)]
  struct SeenSpan {
    /// Position in the recording: unique for the process, unlike a `tracing` span id,
    /// which the registry reuses once a span closes.
    index: usize,
    name: &'static str,
    parent: Option<usize>,
    fields: BTreeMap<String, String>,
    /// Ticks of a shared clock, so "closed before that one opened" can be asserted.
    opened: usize,
    closed: Option<usize>,
  }

  #[derive(Default)]
  struct Log {
    spans: Vec<SeenSpan>,
    clock: usize,
  }

  #[derive(Clone, Default)]
  struct Recorder(Arc<Mutex<Log>>);

  struct Fields<'a>(&'a mut BTreeMap<String, String>);

  impl Visit for Fields<'_> {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
      self.0.insert(field.name().to_owned(), format!("{value:?}"));
    }
    fn record_str(&mut self, field: &Field, value: &str) {
      self.0.insert(field.name().to_owned(), value.to_owned());
    }
  }

  /// Index into the recording, kept in the span's extensions.
  struct Slot(usize);

  fn slot_of<S: Subscriber + for<'a> LookupSpan<'a>>(
    ctx: &Context<'_, S>,
    id: &Id,
  ) -> Option<usize> {
    ctx
      .span(id)
      .and_then(|span| span.extensions().get::<Slot>().map(|slot| slot.0))
  }

  impl<S: Subscriber + for<'a> LookupSpan<'a>> Layer<S> for Recorder {
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
      let span = ctx.span(id).unwrap();
      let mut fields = BTreeMap::new();
      attrs.record(&mut Fields(&mut fields));
      let parent = span
        .parent()
        .and_then(|parent| parent.extensions().get::<Slot>().map(|slot| slot.0));
      let mut log = self.0.lock().unwrap();
      log.clock += 1;
      let index = log.spans.len();
      let opened = log.clock;
      log.spans.push(SeenSpan {
        index,
        name: span.name(),
        parent,
        fields,
        opened,
        closed: None,
      });
      span.extensions_mut().insert(Slot(index));
    }

    fn on_record(&self, id: &Id, values: &Record<'_>, ctx: Context<'_, S>) {
      let Some(slot) = slot_of(&ctx, id) else {
        return;
      };
      values.record(&mut Fields(&mut self.0.lock().unwrap().spans[slot].fields));
    }

    fn on_close(&self, id: Id, ctx: Context<'_, S>) {
      let Some(slot) = slot_of(&ctx, &id) else {
        return;
      };
      let mut log = self.0.lock().unwrap();
      log.clock += 1;
      let closed = log.clock;
      log.spans[slot].closed = Some(closed);
    }
  }

  /// Every span recorded so far, by every test.
  fn recorded() -> Vec<SeenSpan> {
    static RECORDER: OnceLock<Recorder> = OnceLock::new();
    let recorder = RECORDER.get_or_init(|| {
      let recorder = Recorder::default();
      tracing::subscriber::set_global_default(
        tracing_subscriber::registry().with(recorder.clone()),
      )
      .expect("no test installs another global subscriber");
      recorder
    });
    recorder.0.lock().unwrap().spans.clone()
  }

  /// Makes sure the recorder is installed *before* the code under test runs, since a span
  /// created ahead of it is not recorded.
  fn start_recording() {
    let _ = recorded();
  }

  /// The `invoke_agent` span of the agent named `agent`; panics when there is none or
  /// when more than one run has that name.
  fn runs_of(spans: &[SeenSpan], agent: &str) -> Vec<SeenSpan> {
    spans
      .iter()
      .filter(|span| {
        span.name == "invoke_agent"
          && span.fields.get("gen_ai.agent.name").map(String::as_str) == Some(agent)
      })
      .cloned()
      .collect()
  }

  fn run_of(spans: &[SeenSpan], agent: &str) -> SeenSpan {
    let mut runs = runs_of(spans, agent);
    assert_eq!(runs.len(), 1, "runs of `{agent}`: {runs:#?}");
    runs.remove(0)
  }

  /// Every span under `root`, however deep, in the order they opened.
  fn below(spans: &[SeenSpan], root: usize) -> Vec<SeenSpan> {
    // A span is always recorded after its parent, so one pass finds every descendant.
    let mut inside = vec![root];
    let mut found = Vec::new();
    for span in spans {
      if span.parent.is_some_and(|parent| inside.contains(&parent)) {
        inside.push(span.index);
        found.push(span.clone());
      }
    }
    found
  }

  fn named<'a>(spans: &'a [SeenSpan], name: &str) -> Vec<&'a SeenSpan> {
    spans.iter().filter(|span| span.name == name).collect()
  }

  /// The trace a delegating run produces: the lead's run, its two model requests and
  /// its tool call, and under that call the sub-agent's own run and request — each
  /// carrying the attributes a trace view needs.
  #[tokio::test]
  async fn a_delegating_run_produces_a_nested_trace() {
    start_recording();
    let lead = delegation("trace", vec![with_usage(answer("helped"), 30, 3)]);

    lead.run("go").await.unwrap();

    let spans = recorded();
    let outer = run_of(&spans, "trace-lead");
    assert!(outer.parent.is_none());
    assert_eq!(outer.fields["otel.name"], "invoke_agent trace-lead");
    assert_eq!(outer.fields["gen_ai.request.model"], "big");
    assert_eq!(outer.fields["agent.outcome"], "done");
    assert_eq!(
      outer.fields["gen_ai.usage.input_tokens"], "330",
      "the run's total includes the sub-agent"
    );

    let tree = below(&spans, outer.index);
    let tools = named(&tree, "execute_tool");
    assert_eq!(tools.len(), 1, "{tree:#?}");
    let tool = tools[0];
    assert_eq!(tool.parent, Some(outer.index));
    assert_eq!(tool.fields["gen_ai.tool.name"], "helper");
    assert_eq!(tool.fields["agent.tool.outcome"], "success");

    let runs = named(&tree, "invoke_agent");
    assert_eq!(runs.len(), 1, "{tree:#?}");
    let inner = runs[0];
    assert_eq!(
      inner.parent,
      Some(tool.index),
      "the sub-agent runs under the call"
    );
    assert_eq!(inner.fields["gen_ai.agent.name"], "trace-helper");

    let chats = named(&tree, "chat");
    assert_eq!(chats.len(), 3, "two lead requests, one sub-agent request");
    assert!(
      chats
        .iter()
        .all(|chat| chat.parent == Some(outer.index) || chat.parent == Some(inner.index)),
      "{chats:#?}"
    );
    let small = chats
      .iter()
      .find(|chat| chat.fields["gen_ai.request.model"] == "small")
      .unwrap();
    assert_eq!(small.parent, Some(inner.index));
    assert_eq!(small.fields["gen_ai.usage.input_tokens"], "30");
    assert_eq!(small.fields["gen_ai.response.finish_reasons"], "stop");
  }

  /// A failed request marks its span failed, with a class a dashboard can group by.
  #[tokio::test]
  async fn a_failed_request_marks_its_span() {
    start_recording();
    let mut no_choices = answer("unused");
    no_choices["choices"] = json!([]);
    let backend = ScriptedBackend::replying(vec![no_choices]);

    let err = agent_on(&backend, "m", ToolRegistry::empty())
      .with_name("failing-run")
      .run("go")
      .await
      .unwrap_err();

    let run = run_of(&recorded(), "failing-run");
    assert_eq!(run.fields["agent.outcome"], "error", "{err:#}");
    assert_eq!(run.fields["otel.status_code"], "error");
    assert!(run.fields.contains_key("otel.status_description"));
  }

  /// A streamed request's span is held open by its stream. The stream must be dropped
  /// once it has been read, not carried through the tools the reply asked for — or the
  /// request would appear to last as long as they take.
  #[tokio::test]
  async fn a_streamed_request_closes_before_its_tools_run() {
    start_recording();
    let backend = ScriptedBackend::streaming(streamed_calculator_round());
    let calculator = ToolRegistry::select(&["calculator".to_owned()]).unwrap();
    let agent = agent_on(&backend, "m", calculator).with_name("stream-order");

    drain(&agent).await.expect("the stream ends with Done");

    let spans = recorded();
    let run = run_of(&spans, "stream-order");
    let tree = below(&spans, run.index);
    let chats = named(&tree, "chat");
    assert_eq!(chats.len(), 2, "{tree:#?}");
    let tools = named(&tree, "execute_tool");
    assert_eq!(tools.len(), 1, "{tree:#?}");
    let first_request = chats[0];
    let closed = first_request
      .closed
      .expect("the request's span closes once its stream is dropped");
    assert!(
      closed < tools[0].opened,
      "the request was still open while the tool ran: {first_request:#?} vs {:#?}",
      tools[0]
    );
  }

  /// Suspends every call it sees; a supplied decision then stands in for it.
  struct SuspendAll;

  #[async_trait::async_trait]
  impl BeforeToolCallback for SuspendAll {
    async fn call(
      &self,
      _context: &ExecutionContext,
      _tool_call: ToolCallView<'_>,
    ) -> ToolCallDecision {
      ToolCallDecision::Suspend
    }
  }

  /// A run that suspends and is later resumed is two legs, each with a span of its own
  /// reporting only what it spent. The tools the resume re-attempts — here the sub-agent
  /// the approval was for — belong to the second leg: under its span, billed to its run.
  #[tokio::test]
  async fn a_resumed_run_accounts_for_the_calls_it_re_attempts() {
    start_recording();
    let sub = ScriptedBackend::replying(vec![with_usage(answer("helped"), 30, 3)]);
    let parent = ScriptedBackend::replying(vec![
      with_usage(
        tool_call("call_1", "helper", json!({ "task": "help" })),
        100,
        10,
      ),
      with_usage(answer("done"), 200, 20),
    ]);
    let mut toolbox = ToolRegistry::empty();
    with_helper(
      agent_on(&sub, "small", ToolRegistry::empty()).with_name("resume-helper"),
      &mut toolbox,
    );
    let lead = agent_on(&parent, "big", toolbox)
      .with_name("resume-lead")
      .with_before_tool_callback(Arc::new(SuspendAll));

    let AgentOutcome::Suspended(state) = lead
      .run_continuing_resumable(Conversation::default(), "go")
      .await
      .unwrap()
    else {
      panic!("the approval hook suspends the only call");
    };
    assert_eq!(state.context.usage.prompt_tokens, 100);
    let decisions = HashMap::from([("call_1".to_owned(), ResumedDecision::Approved)]);
    let AgentOutcome::Done(done) = lead.resume(state, &decisions).await.unwrap() else {
      panic!("the approved call lets the run finish");
    };

    assert_eq!(done.context.usage_by_model["small"].prompt_tokens, 30);
    assert_eq!(done.context.usage_by_model["big"].prompt_tokens, 300);
    assert_eq!(done.context.usage.prompt_tokens, 330, "counted once");

    let spans = recorded();
    let mut legs = runs_of(&spans, "resume-lead");
    legs.sort_by_key(|leg| leg.opened);
    assert_eq!(legs.len(), 2, "{legs:#?}");
    let (first, second) = (&legs[0], &legs[1]);
    assert_eq!(first.fields["agent.outcome"], "suspended");
    assert_eq!(first.fields["gen_ai.usage.input_tokens"], "100");
    assert_eq!(second.fields["agent.outcome"], "done");
    assert_eq!(
      second.fields["gen_ai.usage.input_tokens"], "230",
      "the second leg's own request plus the sub-agent it ran"
    );
    let tree = below(&spans, second.index);
    let tools = named(&tree, "execute_tool");
    assert_eq!(tools.len(), 1, "the re-attempted call is traced: {tree:#?}");
    assert!(
      named(&tree, "invoke_agent")
        .iter()
        .any(|run| run.fields["gen_ai.agent.name"] == "resume-helper"
          && run.parent == Some(tools[0].index)),
      "the sub-agent nests under the re-attempted call: {tree:#?}"
    );
  }
}
