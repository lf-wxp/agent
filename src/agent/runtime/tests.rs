use std::sync::{
  Mutex,
  atomic::{AtomicBool, AtomicUsize, Ordering},
};

use serde_json::json;

use super::*;
use crate::{
  agent::llm_request::LlmRequest,
  tools::{
    Tool,
    calculator::{self, Calculator},
  },
};

fn agent_with(toolbox: ToolRegistry) -> Agent {
  Agent::new(
    Provider::shared().clone(),
    "gpt-test",
    Option::<String>::None,
    Arc::new(toolbox),
  )
}

#[test]
fn tool_rounds_remaining_until_budget_spent() {
  assert!(tool_rounds_remaining(0, 3));
  assert!(tool_rounds_remaining(2, 3));
  assert!(!tool_rounds_remaining(3, 3));
}

#[test]
fn new_defaults_max_steps_to_the_shared_tool_round_budget() {
  let agent = agent_with(ToolRegistry::empty());
  assert_eq!(agent.max_steps, config::max_tool_rounds() as u32);
}

#[test]
fn with_max_steps_overrides_the_default() {
  let agent = agent_with(ToolRegistry::empty()).with_max_steps(1);
  assert_eq!(agent.max_steps, 1);
}

#[test]
fn new_registers_the_default_budget_trim() {
  let agent = agent_with(ToolRegistry::empty());
  assert_eq!(
    agent.before_llm_callbacks.len(),
    1,
    "a ContextOptimizer should be registered by default"
  );
}

/// A long conversation under a tiny budget: the request must shrink, the head must
/// survive, and `context.events` must be untouched either way.
#[tokio::test]
async fn a_tight_budget_trims_the_request_but_not_the_transcript() {
  let agent = agent_with(ToolRegistry::empty())
    .clear_before_llm_callbacks()
    .with_before_llm_callback(Arc::new(ContextOptimizer::new(200)));
  let mut context = ExecutionContext::new();
  let id = context.execution_id.clone();

  context.add_event(Event::new(
    id.clone(),
    "user",
    vec![ContentItem::Message {
      role: "user".to_owned(),
      content: "the original task".to_owned(),
    }],
  ));
  for i in 0..20 {
    context.add_event(Event::new(
      id.clone(),
      "agent",
      vec![ContentItem::Message {
        role: "assistant".to_owned(),
        content: format!("step {i} {}", "word ".repeat(200)),
      }],
    ));
  }

  let request = agent.prepare_llm_request(&context).await;

  assert!(
    request.contents.len() < context.events.len(),
    "the request copy should have been trimmed"
  );
  let ContentItem::Message { content, .. } = &request.contents[0] else {
    panic!("expected the pinned head to still be a message");
  };
  assert_eq!(content, "the original task", "the head is pinned");
  assert_eq!(
    context.events.len(),
    21,
    "context.events must stay intact (non-destructive)"
  );
}

#[tokio::test]
async fn clearing_the_chain_drops_the_default_trim() {
  let agent = agent_with(ToolRegistry::empty()).clear_before_llm_callbacks();
  let mut context = ExecutionContext::new();

  context.add_event(Event::new(
    context.execution_id.clone(),
    "user",
    vec![ContentItem::Message {
      role: "user".to_owned(),
      content: "word".repeat(50_000),
    }],
  ));

  let request = agent.prepare_llm_request(&context).await;

  assert_eq!(
    request.contents.len(),
    1,
    "with no hooks left the transcript is sent as-is"
  );
}

#[tokio::test]
async fn hooks_run_in_registration_order_after_the_default_trim() {
  struct AddInstruction;

  #[async_trait::async_trait]
  impl BeforeLlmCallback for AddInstruction {
    async fn call(&self, _context: &ExecutionContext, request: &mut LlmRequest) {
      request.push_instruction(format!("saw {} item(s)", request.contents.len()));
    }
  }

  let agent = agent_with(ToolRegistry::empty())
    .clear_before_llm_callbacks()
    .with_before_llm_callback(Arc::new(ContextOptimizer::new(200)))
    .with_before_llm_callback(Arc::new(AddInstruction));
  let mut context = ExecutionContext::new();
  let id = context.execution_id.clone();

  context.add_event(Event::new(
    id.clone(),
    "user",
    vec![ContentItem::Message {
      role: "user".to_owned(),
      content: "the original task".to_owned(),
    }],
  ));
  for i in 0..20 {
    context.add_event(Event::new(
      id.clone(),
      "agent",
      vec![ContentItem::Message {
        role: "assistant".to_owned(),
        content: format!("step {i} {}", "word ".repeat(200)),
      }],
    ));
  }

  let request = agent.prepare_llm_request(&context).await;

  let seen = request
    .instructions
    .first()
    .expect("the second hook should have pushed an instruction");
  assert_ne!(
    seen, "saw 21 item(s)",
    "the hook must observe the already-trimmed contents, not the full transcript"
  );
  assert_eq!(seen, &format!("saw {} item(s)", request.contents.len()));
}

/// A trailing system message (the `json_object` schema hint) is part of what goes on the
/// wire, so the hook chain has to be able to measure it — otherwise every token budget
/// undercounts by exactly its length, on the route where that text is most likely to be a
/// large generated schema.
#[tokio::test]
async fn a_trailer_is_visible_to_the_hook_chain() {
  struct RecordInstructions(Arc<Mutex<Vec<String>>>);

  #[async_trait::async_trait]
  impl BeforeLlmCallback for RecordInstructions {
    async fn call(&self, _context: &ExecutionContext, request: &mut LlmRequest) {
      *self.0.lock().expect("not poisoned") = request.instructions.clone();
    }
  }

  let seen = Arc::new(Mutex::new(Vec::new()));
  let agent = agent_with(ToolRegistry::empty())
    .clear_before_llm_callbacks()
    .with_before_llm_callback(Arc::new(RecordInstructions(Arc::clone(&seen))));

  let request = agent
    .prepare_llm_request_with_trailer(&ExecutionContext::new(), "reply with JSON")
    .await;

  assert!(
    seen
      .lock()
      .expect("not poisoned")
      .iter()
      .any(|instruction| instruction == "reply with JSON"),
    "the hook must see the trailer, or it cannot charge it to the budget"
  );
  assert!(
    !request.instructions.iter().any(|i| i == "reply with JSON"),
    "the caller places the trailer itself, so it must not also be left up front"
  );
}

/// Only the trailer is taken back out — an agent's own system prompt (and anything a hook
/// added) has to survive it.
#[tokio::test]
async fn removing_the_trailer_leaves_the_other_instructions_alone() {
  let agent = Agent::new(
    Provider::shared().clone(),
    "gpt-test",
    Some("be nice"),
    Arc::new(ToolRegistry::empty()),
  );

  let request = agent
    .prepare_llm_request_with_trailer(&ExecutionContext::new(), "reply with JSON")
    .await;

  assert_eq!(request.instructions, vec!["be nice".to_owned()]);
}

#[test]
fn build_messages_replays_system_user_tool_call_and_result() {
  let agent = Agent::new(
    Provider::shared().clone(),
    "gpt-test",
    Some("be nice"),
    Arc::new(ToolRegistry::empty()),
  );
  let mut context = ExecutionContext::new();
  let id = context.execution_id.clone();

  context.add_event(Event::new(
    id.clone(),
    "user",
    vec![ContentItem::Message {
      role: "user".to_owned(),
      content: "hi".to_owned(),
    }],
  ));
  context.add_event(Event::new(
    id.clone(),
    "agent",
    vec![ContentItem::ToolCall {
      tool_call_id: "call_1".to_owned(),
      name: calculator::NAME.to_owned(),
      arguments: json!({"operator": "add"}),
    }],
  ));
  context.add_event(Event::new(
    id,
    "tool",
    vec![ContentItem::ToolResult {
      tool_call_id: "call_1".to_owned(),
      name: calculator::NAME.to_owned(),
      status: ToolResultStatus::Success,
      content: "3".to_owned(),
    }],
  ));

  let request = LlmRequest::new(Some("be nice".to_owned()), &context.events);
  let messages = agent.build_messages(request).unwrap();

  assert_eq!(messages.len(), 4, "system + user + assistant + tool");
  assert!(matches!(
    messages[0],
    ChatCompletionRequestMessage::System(_)
  ));
  assert!(matches!(messages[1], ChatCompletionRequestMessage::User(_)));
  assert!(matches!(
    messages[2],
    ChatCompletionRequestMessage::Assistant(_)
  ));
  assert!(matches!(messages[3], ChatCompletionRequestMessage::Tool(_)));
}

#[test]
fn build_messages_merges_consecutive_tool_calls_into_one_assistant_message() {
  let agent = agent_with(ToolRegistry::empty());
  let mut context = ExecutionContext::new();
  let id = context.execution_id.clone();

  context.add_event(Event::new(
    id.clone(),
    "agent",
    vec![
      ContentItem::ToolCall {
        tool_call_id: "call_1".to_owned(),
        name: "a".to_owned(),
        arguments: json!({}),
      },
      ContentItem::ToolCall {
        tool_call_id: "call_2".to_owned(),
        name: "b".to_owned(),
        arguments: json!({}),
      },
    ],
  ));

  let request = LlmRequest::new(None, &context.events);
  let messages = agent.build_messages(request).unwrap();
  assert_eq!(messages.len(), 1);
  let ChatCompletionRequestMessage::Assistant(assistant) = &messages[0] else {
    panic!("expected an assistant message");
  };
  assert_eq!(assistant.tool_calls.as_ref().unwrap().len(), 2);
}

#[tokio::test]
async fn execute_tool_calls_records_success_and_unknown_tool() {
  let mut registry = ToolRegistry::empty();
  registry.add(Arc::new(Calculator)).unwrap();
  let agent = agent_with(registry);
  let mut context = ExecutionContext::new();

  let calls = vec![
    ChatCompletionMessageToolCalls::Function(ChatCompletionMessageToolCall {
      id: "call_1".to_owned(),
      function: FunctionCall {
        name: calculator::NAME.to_owned(),
        arguments: r#"{"operator":"add","first_number":1,"second_number":2}"#.to_owned(),
      },
    }),
    ChatCompletionMessageToolCalls::Function(ChatCompletionMessageToolCall {
      id: "call_2".to_owned(),
      function: FunctionCall {
        name: "nope".to_owned(),
        arguments: "{}".to_owned(),
      },
    }),
  ];

  agent
    .execute_tool_calls(&mut context, &calls, &HashMap::new())
    .await;

  let event = context.events.last().unwrap();
  assert_eq!(event.author, "tool");
  assert_eq!(event.content.len(), 2);

  let ContentItem::ToolResult {
    status, content, ..
  } = &event.content[0]
  else {
    panic!("expected a tool result");
  };
  assert_eq!(*status, ToolResultStatus::Success);
  assert_eq!(content, "3");

  let ContentItem::ToolResult {
    status, content, ..
  } = &event.content[1]
  else {
    panic!("expected a tool result");
  };
  assert_eq!(*status, ToolResultStatus::Error);
  assert!(content.contains("unknown tool"), "got: {content}");
}

/// Reports whether it actually ran, so a test can tell "the tool ran and its result was
/// rewritten" apart from "the tool never ran at all".
struct SpyTool {
  executed: Arc<AtomicBool>,
}

#[async_trait::async_trait]
impl Tool for SpyTool {
  fn name(&self) -> &str {
    "spy"
  }

  fn description(&self) -> &str {
    "records that it was executed"
  }

  fn parameters(&self) -> Value {
    json!({"type": "object", "properties": {}})
  }

  async fn execute(&self, _args_json: &str) -> anyhow::Result<String> {
    self.executed.store(true, Ordering::SeqCst);
    Ok("real result".to_owned())
  }
}

struct DenyEverything;

#[async_trait::async_trait]
impl BeforeToolCallback for DenyEverything {
  async fn call(
    &self,
    _context: &ExecutionContext,
    tool_call: ToolCallView<'_>,
  ) -> ToolCallDecision {
    ToolCallDecision::deny(format!("denied {}", tool_call.name))
  }
}

struct RewriteResult;

#[async_trait::async_trait]
impl AfterToolCallback for RewriteResult {
  async fn call(
    &self,
    _context: &ExecutionContext,
    _tool_call_id: &str,
    _tool_name: &str,
    _status: ToolResultStatus,
    content: &str,
  ) -> Option<(ToolResultStatus, String)> {
    Some((ToolResultStatus::Success, format!("rewritten: {content}")))
  }
}

struct CountingAfter(Arc<AtomicUsize>);

#[async_trait::async_trait]
impl AfterToolCallback for CountingAfter {
  async fn call(
    &self,
    _context: &ExecutionContext,
    _tool_call_id: &str,
    _tool_name: &str,
    _status: ToolResultStatus,
    _content: &str,
  ) -> Option<(ToolResultStatus, String)> {
    self.0.fetch_add(1, Ordering::SeqCst);
    None
  }
}

/// Counts its own invocations and always lets the call through, to prove a before-hook
/// chain runs every registered hook in order up to the one that short-circuits (or all
/// of them, if none does) rather than only ever running one.
struct CountingBefore(Arc<AtomicUsize>);

#[async_trait::async_trait]
impl BeforeToolCallback for CountingBefore {
  async fn call(
    &self,
    _context: &ExecutionContext,
    _tool_call: ToolCallView<'_>,
  ) -> ToolCallDecision {
    self.0.fetch_add(1, Ordering::SeqCst);
    ToolCallDecision::Proceed
  }
}

/// Appends a fixed suffix to whatever content it is handed, to prove an after-hook
/// chain threads each hook's output into the next rather than only ever running one or
/// having each see the tool's original, unmodified result.
struct AppendSuffix(&'static str);

#[async_trait::async_trait]
impl AfterToolCallback for AppendSuffix {
  async fn call(
    &self,
    _context: &ExecutionContext,
    _tool_call_id: &str,
    _tool_name: &str,
    status: ToolResultStatus,
    content: &str,
  ) -> Option<(ToolResultStatus, String)> {
    Some((status, format!("{content}{}", self.0)))
  }
}

fn spy_registry(executed: &Arc<AtomicBool>) -> ToolRegistry {
  let mut registry = ToolRegistry::empty();
  registry
    .add(Arc::new(SpyTool {
      executed: Arc::clone(executed),
    }))
    .unwrap();
  registry
}

fn spy_call() -> ChatCompletionMessageToolCalls {
  ChatCompletionMessageToolCalls::Function(ChatCompletionMessageToolCall {
    id: "call_1".to_owned(),
    function: FunctionCall {
      name: "spy".to_owned(),
      arguments: "{}".to_owned(),
    },
  })
}

/// The recorded result alone would not prove much — a callback returning the same text
/// the tool would have produced is indistinguishable — so this asserts on the tool's own
/// account of whether it ran.
#[tokio::test]
async fn before_tool_callback_short_circuits_without_running_the_tool() {
  let executed = Arc::new(AtomicBool::new(false));
  let agent =
    agent_with(spy_registry(&executed)).with_before_tool_callback(Arc::new(DenyEverything));
  let mut context = ExecutionContext::new();

  agent
    .execute_tool_calls(&mut context, &[spy_call()], &HashMap::new())
    .await;

  assert!(!executed.load(Ordering::SeqCst), "the tool must not run");
  let ContentItem::ToolResult {
    status, content, ..
  } = &context.events.last().unwrap().content[0]
  else {
    panic!("expected a tool result");
  };
  assert_eq!(*status, ToolResultStatus::Error);
  assert_eq!(content, "denied spy");
}

#[tokio::test]
async fn after_tool_callback_replaces_the_recorded_result() {
  let executed = Arc::new(AtomicBool::new(false));
  let agent = agent_with(spy_registry(&executed)).with_after_tool_callback(Arc::new(RewriteResult));
  let mut context = ExecutionContext::new();

  agent
    .execute_tool_calls(&mut context, &[spy_call()], &HashMap::new())
    .await;

  assert!(executed.load(Ordering::SeqCst), "the tool should have run");
  let ContentItem::ToolResult { content, .. } = &context.events.last().unwrap().content[0] else {
    panic!("expected a tool result");
  };
  assert_eq!(content, "rewritten: real result");
}

#[tokio::test]
async fn after_tool_callback_is_skipped_for_a_short_circuited_call() {
  let executed = Arc::new(AtomicBool::new(false));
  let after_calls = Arc::new(AtomicUsize::new(0));
  let agent = agent_with(spy_registry(&executed))
    .with_before_tool_callback(Arc::new(DenyEverything))
    .with_after_tool_callback(Arc::new(CountingAfter(Arc::clone(&after_calls))));
  let mut context = ExecutionContext::new();

  agent
    .execute_tool_calls(&mut context, &[spy_call()], &HashMap::new())
    .await;

  assert_eq!(
    after_calls.load(Ordering::SeqCst),
    0,
    "a short-circuited call produced no tool result to post-process"
  );
}

/// Registering `with_before_tool_callback` more than once used to replace the previous
/// hook; it must now add to the chain instead, running every hook up to (and including)
/// the one that short-circuits.
#[tokio::test]
async fn multiple_before_tool_callbacks_run_in_order_until_one_short_circuits() {
  let executed = Arc::new(AtomicBool::new(false));
  let first_calls = Arc::new(AtomicUsize::new(0));
  let agent = agent_with(spy_registry(&executed))
    .with_before_tool_callback(Arc::new(CountingBefore(Arc::clone(&first_calls))))
    .with_before_tool_callback(Arc::new(DenyEverything));
  let mut context = ExecutionContext::new();

  agent
    .execute_tool_calls(&mut context, &[spy_call()], &HashMap::new())
    .await;

  assert_eq!(
    first_calls.load(Ordering::SeqCst),
    1,
    "the first hook must still run before the second one denies the call"
  );
  assert!(!executed.load(Ordering::SeqCst), "the tool must not run");
  let ContentItem::ToolResult {
    status, content, ..
  } = &context.events.last().unwrap().content[0]
  else {
    panic!("expected a tool result");
  };
  assert_eq!(*status, ToolResultStatus::Error);
  assert_eq!(content, "denied spy");
}

/// When no before-hook short-circuits the call, every one of them must still have run,
/// and the real tool must run too — a chain that all pass through is not the same as no
/// chain at all.
#[tokio::test]
async fn multiple_before_tool_callbacks_all_run_when_none_short_circuits() {
  let executed = Arc::new(AtomicBool::new(false));
  let first_calls = Arc::new(AtomicUsize::new(0));
  let second_calls = Arc::new(AtomicUsize::new(0));
  let agent = agent_with(spy_registry(&executed))
    .with_before_tool_callback(Arc::new(CountingBefore(Arc::clone(&first_calls))))
    .with_before_tool_callback(Arc::new(CountingBefore(Arc::clone(&second_calls))));
  let mut context = ExecutionContext::new();

  agent
    .execute_tool_calls(&mut context, &[spy_call()], &HashMap::new())
    .await;

  assert_eq!(first_calls.load(Ordering::SeqCst), 1);
  assert_eq!(second_calls.load(Ordering::SeqCst), 1);
  assert!(executed.load(Ordering::SeqCst), "the tool should have run");
}

/// Registering `with_after_tool_callback` more than once used to replace the previous
/// hook; it must now chain them, each one seeing the result the previous one produced.
#[tokio::test]
async fn multiple_after_tool_callbacks_thread_the_result_through_in_order() {
  let executed = Arc::new(AtomicBool::new(false));
  let agent = agent_with(spy_registry(&executed))
    .with_after_tool_callback(Arc::new(AppendSuffix("-a")))
    .with_after_tool_callback(Arc::new(AppendSuffix("-b")));
  let mut context = ExecutionContext::new();

  agent
    .execute_tool_calls(&mut context, &[spy_call()], &HashMap::new())
    .await;

  let ContentItem::ToolResult { content, .. } = &context.events.last().unwrap().content[0] else {
    panic!("expected a tool result");
  };
  assert_eq!(
    content, "real result-a-b",
    "each hook must see the previous hook's output, in registration order"
  );
}

/// Suspends every call it sees, standing in for an approval that needs a human who is
/// not currently reachable.
struct SuspendEverything;

#[async_trait::async_trait]
impl BeforeToolCallback for SuspendEverything {
  async fn call(
    &self,
    _context: &ExecutionContext,
    _tool_call: ToolCallView<'_>,
  ) -> ToolCallDecision {
    ToolCallDecision::Suspend
  }
}

/// A second tool, so a round can contain one call that completes and one that suspends.
struct OtherSpyTool {
  executed: Arc<AtomicBool>,
}

#[async_trait::async_trait]
impl Tool for OtherSpyTool {
  fn name(&self) -> &str {
    "other"
  }

  fn description(&self) -> &str {
    "records whether it ran"
  }

  fn parameters(&self) -> Value {
    json!({"type": "object", "properties": {}})
  }

  async fn execute(&self, _args_json: &str) -> anyhow::Result<String> {
    self.executed.store(true, Ordering::SeqCst);
    Ok("other result".to_owned())
  }
}

/// Suspends only the tool it is named after, letting everything else through — the shape
/// a real approval rule has, and what makes a partially suspended round possible.
struct SuspendOne(&'static str);

#[async_trait::async_trait]
impl BeforeToolCallback for SuspendOne {
  async fn call(
    &self,
    _context: &ExecutionContext,
    tool_call: ToolCallView<'_>,
  ) -> ToolCallDecision {
    if tool_call.name == self.0 {
      ToolCallDecision::Suspend
    } else {
      ToolCallDecision::Proceed
    }
  }
}

fn call_named(id: &str, name: &str) -> ChatCompletionMessageToolCalls {
  ChatCompletionMessageToolCalls::Function(ChatCompletionMessageToolCall {
    id: id.to_owned(),
    function: FunctionCall {
      name: name.to_owned(),
      arguments: "{}".to_owned(),
    },
  })
}

/// A suspended call must not run, and must not be reported as a result either — it has
/// not produced one.
#[tokio::test]
async fn a_suspended_call_neither_runs_nor_produces_a_result() {
  let executed = Arc::new(AtomicBool::new(false));
  let agent =
    agent_with(spy_registry(&executed)).with_before_tool_callback(Arc::new(SuspendEverything));
  let mut context = ExecutionContext::new();

  let round = agent
    .execute_tool_calls(&mut context, &[spy_call()], &HashMap::new())
    .await;

  assert!(!executed.load(Ordering::SeqCst), "the tool must not run");
  assert!(round.completed.is_empty());
  assert_eq!(round.suspended.len(), 1);
  assert_eq!(round.suspended[0].tool_call_id, "call_1");
  assert_eq!(round.suspended[0].name, "spy");
  assert!(
    context.events.is_empty(),
    "a round that produced no results must not record a tool event, or the transcript \
     would claim a decision that has not been taken"
  );
}

/// The case partial completion exists for: calls in a round run concurrently, so one
/// suspending does not undo the work its siblings already did. Discarding their results
/// would mean re-running them later and duplicating every side effect.
#[tokio::test]
async fn a_partially_suspended_round_keeps_the_results_it_already_has() {
  let spy_executed = Arc::new(AtomicBool::new(false));
  let other_executed = Arc::new(AtomicBool::new(false));

  let mut registry = ToolRegistry::empty();
  registry
    .add(Arc::new(SpyTool {
      executed: Arc::clone(&spy_executed),
    }))
    .unwrap();
  registry
    .add(Arc::new(OtherSpyTool {
      executed: Arc::clone(&other_executed),
    }))
    .unwrap();

  let agent = agent_with(registry).with_before_tool_callback(Arc::new(SuspendOne("spy")));
  let mut context = ExecutionContext::new();

  let round = agent
    .execute_tool_calls(
      &mut context,
      &[call_named("call_1", "spy"), call_named("call_2", "other")],
      &HashMap::new(),
    )
    .await;

  assert!(
    !spy_executed.load(Ordering::SeqCst),
    "the suspended tool must not run"
  );
  assert!(
    other_executed.load(Ordering::SeqCst),
    "its sibling must still have run"
  );

  assert_eq!(round.suspended.len(), 1);
  assert_eq!(round.suspended[0].name, "spy");
  assert_eq!(round.completed.len(), 1);
  let ContentItem::ToolResult {
    tool_call_id, name, ..
  } = &round.completed[0]
  else {
    panic!("expected a tool result");
  };
  assert_eq!(tool_call_id, "call_2");
  assert_eq!(name, "other");

  // Recorded: exactly the one result, so the transcript reflects what happened and
  // nothing more.
  let event = context
    .events
    .last()
    .expect("the result should be recorded");
  assert_eq!(event.content.len(), 1);
}

/// A suspension stops the before-hook chain where it happens: a later hook must not get
/// to rule on a call that is already undecided, and the tool must not run either.
#[tokio::test]
async fn a_suspension_ends_the_before_hook_chain() {
  let executed = Arc::new(AtomicBool::new(false));
  let later_calls = Arc::new(AtomicUsize::new(0));
  let agent = agent_with(spy_registry(&executed))
    .with_before_tool_callback(Arc::new(SuspendEverything))
    .with_before_tool_callback(Arc::new(CountingBefore(Arc::clone(&later_calls))));
  let mut context = ExecutionContext::new();

  agent
    .execute_tool_calls(&mut context, &[spy_call()], &HashMap::new())
    .await;

  assert_eq!(
    later_calls.load(Ordering::SeqCst),
    0,
    "a hook after the one that suspended must not run"
  );
  assert!(!executed.load(Ordering::SeqCst), "the tool must not run");
}

/// The raw argument string has to survive a suspension verbatim: it is what the tool
/// will be handed when the call is retried, and what a human was shown when asked about
/// it. Re-deriving it from a parsed value would hand the tool a different payload than
/// the one that was approved.
#[tokio::test]
async fn a_suspended_call_keeps_its_raw_arguments() {
  let executed = Arc::new(AtomicBool::new(false));
  let agent =
    agent_with(spy_registry(&executed)).with_before_tool_callback(Arc::new(SuspendEverything));
  let mut context = ExecutionContext::new();

  let raw = r#"{"path": "notes.txt", "n": 1}"#;
  let call = ChatCompletionMessageToolCalls::Function(ChatCompletionMessageToolCall {
    id: "call_1".to_owned(),
    function: FunctionCall {
      name: "spy".to_owned(),
      arguments: raw.to_owned(),
    },
  });

  let round = agent
    .execute_tool_calls(&mut context, &[call], &HashMap::new())
    .await;

  assert_eq!(round.suspended[0].raw_arguments, raw);
}

/// An entry point with nowhere to resume to has to close the call out rather than leave
/// it dangling: `record_tool_calls` has already written the call, and a call with no
/// matching result is a conversation most providers reject.
#[tokio::test]
async fn an_entry_point_that_cannot_resume_records_the_call_as_unanswered() {
  let executed = Arc::new(AtomicBool::new(false));
  let agent =
    agent_with(spy_registry(&executed)).with_before_tool_callback(Arc::new(SuspendEverything));
  let mut context = ExecutionContext::new();

  let round = agent
    .execute_tool_calls(&mut context, &[spy_call()], &HashMap::new())
    .await;
  let placeholders = agent.record_unanswered(&mut context, &round.suspended);

  assert_eq!(placeholders.len(), 1);
  let ContentItem::ToolResult {
    tool_call_id,
    status,
    content,
    ..
  } = &placeholders[0]
  else {
    panic!("expected a tool result");
  };
  assert_eq!(tool_call_id, "call_1");
  assert_eq!(*status, ToolResultStatus::Error);
  assert!(
    content.contains("spy"),
    "the message should name the tool that was not run: {content}"
  );

  // Every recorded call now has a result, which is the invariant this exists to restore.
  let results: Vec<&ContentItem> = context
    .events
    .iter()
    .flat_map(|event| &event.content)
    .filter(|item| matches!(item, ContentItem::ToolResult { .. }))
    .collect();
  assert_eq!(results.len(), 1);
}

// ---- resuming a suspended round ----------------------------------------------------

/// A supplied approval lets a previously suspended call run, without the hook that
/// suspended it having to change its mind.
#[tokio::test]
async fn a_supplied_approval_lets_a_suspended_call_run() {
  let executed = Arc::new(AtomicBool::new(false));
  let agent =
    agent_with(spy_registry(&executed)).with_before_tool_callback(Arc::new(SuspendEverything));
  let mut context = ExecutionContext::new();

  let decisions = HashMap::from([("call_1".to_owned(), ResumedDecision::Approved)]);
  let round = agent
    .execute_tool_calls(&mut context, &[spy_call()], &decisions)
    .await;

  assert!(
    executed.load(Ordering::SeqCst),
    "the approved call should have reached the tool"
  );
  assert!(round.suspended.is_empty());
  assert_eq!(round.completed.len(), 1);
}

/// A supplied refusal records the front-end's own wording, since that is what the model
/// reads and what the human actually said.
#[tokio::test]
async fn a_supplied_refusal_records_its_reason_verbatim() {
  let executed = Arc::new(AtomicBool::new(false));
  let agent =
    agent_with(spy_registry(&executed)).with_before_tool_callback(Arc::new(SuspendEverything));
  let mut context = ExecutionContext::new();

  let decisions = HashMap::from([(
    "call_1".to_owned(),
    ResumedDecision::Refused("User denied execution of spy: not this one".to_owned()),
  )]);
  let round = agent
    .execute_tool_calls(&mut context, &[spy_call()], &decisions)
    .await;

  assert!(!executed.load(Ordering::SeqCst), "the tool must not run");
  let ContentItem::ToolResult {
    status, content, ..
  } = &round.completed[0]
  else {
    panic!("expected a tool result");
  };
  assert_eq!(*status, ToolResultStatus::Error);
  assert_eq!(content, "User denied execution of spy: not this one");
}

/// The decision replaces the *suspension*, not the whole chain. A guard registered after
/// the approval hook still gets to rule on an approved call — otherwise a human's "yes"
/// would silently switch off the workspace sandbox.
#[tokio::test]
async fn an_approval_does_not_bypass_the_rest_of_the_chain() {
  let executed = Arc::new(AtomicBool::new(false));
  let agent = agent_with(spy_registry(&executed))
    .with_before_tool_callback(Arc::new(SuspendEverything))
    .with_before_tool_callback(Arc::new(DenyEverything));
  let mut context = ExecutionContext::new();

  let decisions = HashMap::from([("call_1".to_owned(), ResumedDecision::Approved)]);
  let round = agent
    .execute_tool_calls(&mut context, &[spy_call()], &decisions)
    .await;

  assert!(
    !executed.load(Ordering::SeqCst),
    "a later guard must still be able to stop an approved call"
  );
  let ContentItem::ToolResult { content, .. } = &round.completed[0] else {
    panic!("expected a tool result");
  };
  assert_eq!(content, "denied spy");
}

/// A decision for one call must not answer another. Answering part of a round is a
/// supported half-step: the rest comes back suspended rather than erroring or, worse,
/// running unapproved.
#[tokio::test]
async fn a_decision_applies_only_to_the_call_it_names() {
  let spy_executed = Arc::new(AtomicBool::new(false));
  let other_executed = Arc::new(AtomicBool::new(false));

  let mut registry = ToolRegistry::empty();
  registry
    .add(Arc::new(SpyTool {
      executed: Arc::clone(&spy_executed),
    }))
    .unwrap();
  registry
    .add(Arc::new(OtherSpyTool {
      executed: Arc::clone(&other_executed),
    }))
    .unwrap();

  let agent = agent_with(registry).with_before_tool_callback(Arc::new(SuspendEverything));
  let mut context = ExecutionContext::new();

  let decisions = HashMap::from([("call_1".to_owned(), ResumedDecision::Approved)]);
  let round = agent
    .execute_tool_calls(
      &mut context,
      &[call_named("call_1", "spy"), call_named("call_2", "other")],
      &decisions,
    )
    .await;

  assert!(spy_executed.load(Ordering::SeqCst), "call_1 was approved");
  assert!(
    !other_executed.load(Ordering::SeqCst),
    "call_2 had no decision and must not run"
  );
  assert_eq!(round.suspended.len(), 1);
  assert_eq!(round.suspended[0].tool_call_id, "call_2");
}

/// `rebuild_tool_calls` is what a resumed call is re-executed from, so it has to
/// reproduce the call exactly — including an argument string that does not parse, which
/// the approval path relies on being able to tell apart from a literal `null`.
#[test]
fn rebuilding_a_suspended_call_preserves_it_exactly() {
  let suspended = vec![SuspendedToolCall {
    tool_call_id: "call_9".to_owned(),
    name: "delete_file".to_owned(),
    raw_arguments: r#"{"path": "a.txt""#.to_owned(),
  }];

  let rebuilt = rebuild_tool_calls(&suspended);
  let ChatCompletionMessageToolCalls::Function(call) = &rebuilt[0] else {
    panic!("a rebuilt call must stay a function call");
  };

  assert_eq!(call.id, "call_9");
  assert_eq!(call.function.name, "delete_file");
  assert_eq!(call.function.arguments, r#"{"path": "a.txt""#);
}

/// Giving up has to leave a sendable transcript: `record_tool_calls` already wrote the
/// call, and a call with no result is a conversation most providers reject.
#[tokio::test]
async fn abandoning_a_suspended_run_closes_out_every_call() {
  let executed = Arc::new(AtomicBool::new(false));
  let agent =
    agent_with(spy_registry(&executed)).with_before_tool_callback(Arc::new(SuspendEverything));

  let mut context = ExecutionContext::new();
  agent.record_tool_calls(&mut context, &[spy_call()]);
  let round = agent
    .execute_tool_calls(&mut context, &[spy_call()], &HashMap::new())
    .await;

  let state = AgentRunState {
    fingerprint: agent.fingerprint(),
    suspended: round.suspended,
    budget_exhausted: false,
    reason: StopReason::AwaitingDecision,
    context,
  };

  // Before: the call is recorded with nothing answering it.
  assert_eq!(count_items(state.context(), false), 1);
  assert_eq!(count_items(state.context(), true), 0);

  let context = state.abandon(GiveUp::Unanswered);

  assert_eq!(
    count_items(&context, true),
    1,
    "every recorded call must end up with a result"
  );
}

/// Counts `ToolCall`s (`results = false`) or `ToolResult`s (`results = true`).
fn count_items(context: &ExecutionContext, results: bool) -> usize {
  context
    .events
    .iter()
    .flat_map(|event| &event.content)
    .filter(|item| {
      if results {
        matches!(item, ContentItem::ToolResult { .. })
      } else {
        matches!(item, ContentItem::ToolCall { .. })
      }
    })
    .count()
}

/// Tool call ids with no matching result — the shape a provider rejects.
fn unpaired_call_ids(context: &ExecutionContext) -> Vec<String> {
  let items: Vec<&ContentItem> = context
    .events
    .iter()
    .flat_map(|event| &event.content)
    .collect();
  let answered: std::collections::HashSet<&str> = items
    .iter()
    .filter_map(|item| match item {
      ContentItem::ToolResult { tool_call_id, .. } => Some(tool_call_id.as_str()),
      _ => None,
    })
    .collect();
  items
    .iter()
    .filter_map(|item| match item {
      ContentItem::ToolCall { tool_call_id, .. } if !answered.contains(tool_call_id.as_str()) => {
        Some(tool_call_id.clone())
      }
      _ => None,
    })
    .collect()
}

/// The invariant that makes a placeholder-filtering hook unnecessary: a suspended round
/// leaves the transcript unpaired, and answering it pairs it up again — so no request is
/// ever built from the gap. `drive` enforces the other half by returning rather than
/// looping while a round is incomplete.
#[tokio::test]
async fn answering_a_suspended_round_restores_the_pairing_invariant() {
  let executed = Arc::new(AtomicBool::new(false));
  let agent =
    agent_with(spy_registry(&executed)).with_before_tool_callback(Arc::new(SuspendEverything));
  let mut context = ExecutionContext::new();

  agent.record_tool_calls(&mut context, &[spy_call()]);
  let round = agent
    .execute_tool_calls(&mut context, &[spy_call()], &HashMap::new())
    .await;
  assert_eq!(
    unpaired_call_ids(&context),
    vec!["call_1".to_owned()],
    "a suspended call is deliberately left unanswered"
  );

  // The resume step, with the answer in hand.
  let decisions = HashMap::from([("call_1".to_owned(), ResumedDecision::Approved)]);
  let calls = rebuild_tool_calls(&round.suspended);
  agent
    .execute_tool_calls(&mut context, &calls, &decisions)
    .await;

  assert!(
    unpaired_call_ids(&context).is_empty(),
    "answering the call must pair it up, since the next request is built from here"
  );
}

/// Resuming with a different agent is refused rather than attempted — the transcript
/// references tools by name and was produced under instructions that no longer apply.
#[tokio::test]
async fn resuming_with_a_changed_agent_is_refused() {
  let executed = Arc::new(AtomicBool::new(false));
  let original = agent_with(spy_registry(&executed));

  let state = AgentRunState {
    fingerprint: original.fingerprint(),
    suspended: vec![SuspendedToolCall {
      tool_call_id: "call_1".to_owned(),
      name: "spy".to_owned(),
      raw_arguments: "{}".to_owned(),
    }],
    budget_exhausted: false,
    reason: StopReason::AwaitingDecision,
    context: ExecutionContext::new(),
  };

  // Same tools, different model.
  let changed = Agent::new(
    Provider::shared().clone(),
    "gpt-different",
    Option::<String>::None,
    Arc::new(spy_registry(&executed)),
  );

  let err = changed
    .resume(state, &HashMap::new())
    .await
    .expect_err("a changed agent must not resume the run");

  assert!(
    err.to_string().contains("model changed"),
    "the error should say what changed: {err}"
  );
  assert!(
    !executed.load(Ordering::SeqCst),
    "nothing may run once the resume is refused"
  );
}

/// The state is the thing that crosses a process boundary, so it has to survive a round
/// trip with the pending calls and their arguments intact.
#[test]
fn a_suspended_state_round_trips_through_json() {
  let executed = Arc::new(AtomicBool::new(false));
  let agent = agent_with(spy_registry(&executed));
  let mut context = ExecutionContext::new();
  context.conversation_id = Some("s1".to_owned());

  let state = AgentRunState {
    fingerprint: agent.fingerprint(),
    suspended: vec![SuspendedToolCall {
      tool_call_id: "call_1".to_owned(),
      name: "spy".to_owned(),
      raw_arguments: r#"{"n":1}"#.to_owned(),
    }],
    budget_exhausted: true,
    reason: StopReason::AwaitingDecision,
    context,
  };

  let json = serde_json::to_string(&state).unwrap();
  let back: AgentRunState = serde_json::from_str(&json).unwrap();

  assert_eq!(back.suspended, state.suspended);
  assert!(back.budget_exhausted);
  assert!(back.fingerprint.matches(&agent.fingerprint()));
  assert_eq!(back.context().conversation_id.as_deref(), Some("s1"));
}

/// A fingerprint mismatch arrives as the stream's first item rather than as a `Result`
/// wrapping the stream, matching how every other failure on this path is reported — and
/// crucially *before* anything runs.
#[tokio::test]
async fn a_resumed_stream_reports_a_mismatch_before_running_anything() {
  let executed = Arc::new(AtomicBool::new(false));
  let original = agent_with(spy_registry(&executed));
  let state = suspended_state(&original, "call_1");

  let changed = Agent::new(
    Provider::shared().clone(),
    "gpt-different",
    Option::<String>::None,
    Arc::new(spy_registry(&executed)),
  );

  let stream = changed.resume_stream(state, HashMap::new());
  futures::pin_mut!(stream);
  let first = stream.next().await.expect("one item");

  assert!(
    first
      .expect_err("a changed agent must not resume")
      .to_string()
      .contains("model changed")
  );
  assert!(!executed.load(Ordering::SeqCst));
  assert!(stream.next().await.is_none(), "nothing follows the error");
}

/// Resuming without an answer re-attempts the call, and whatever suspended it gets to
/// ask again — which is what lets a front-end treat "resume" and "ask again" as one
/// operation. The stream ends suspended again rather than erroring, and no request is
/// ever issued, so this needs no provider.
#[tokio::test]
async fn a_resumed_stream_that_is_still_unanswered_suspends_again() {
  let executed = Arc::new(AtomicBool::new(false));
  let agent =
    agent_with(spy_registry(&executed)).with_before_tool_callback(Arc::new(SuspendEverything));
  let state = suspended_state(&agent, "call_1");

  let stream = agent.resume_stream(state, HashMap::new());
  futures::pin_mut!(stream);
  let first = stream
    .next()
    .await
    .expect("one item")
    .expect("not an error");

  let AgentStreamEvent::Suspended(again) = first else {
    panic!("expected a suspension, got {first:?}");
  };
  assert_eq!(again.suspended.len(), 1);
  assert_eq!(again.suspended[0].tool_call_id, "call_1");
  assert!(!executed.load(Ordering::SeqCst));
  assert!(stream.next().await.is_none());
}

/// The budget flag survives a suspension. A run cut short before it stopped is still cut
/// short after it resumes, and losing that would have the final answer claim to be
/// based on complete work.
#[tokio::test]
async fn a_resumed_stream_carries_the_budget_flag_across() {
  let executed = Arc::new(AtomicBool::new(false));
  let agent =
    agent_with(spy_registry(&executed)).with_before_tool_callback(Arc::new(SuspendEverything));
  let mut state = suspended_state(&agent, "call_1");
  state.budget_exhausted = true;

  let stream = agent.resume_stream(state, HashMap::new());
  futures::pin_mut!(stream);
  let first = stream
    .next()
    .await
    .expect("one item")
    .expect("not an error");

  let AgentStreamEvent::Suspended(again) = first else {
    panic!("expected a suspension, got {first:?}");
  };
  assert!(again.budget_exhausted);
}

/// A round that partly completes on resume reports the finished half before stopping, so
/// a front-end's timeline shows what actually happened rather than losing it.
#[tokio::test]
async fn a_partly_answered_resume_reports_the_completed_half_first() {
  let spy_executed = Arc::new(AtomicBool::new(false));
  let other_executed = Arc::new(AtomicBool::new(false));

  let mut registry = ToolRegistry::empty();
  registry
    .add(Arc::new(SpyTool {
      executed: Arc::clone(&spy_executed),
    }))
    .unwrap();
  registry
    .add(Arc::new(OtherSpyTool {
      executed: Arc::clone(&other_executed),
    }))
    .unwrap();

  let agent = agent_with(registry).with_before_tool_callback(Arc::new(SuspendEverything));
  let state = AgentRunState {
    fingerprint: agent.fingerprint(),
    suspended: vec![
      SuspendedToolCall {
        tool_call_id: "call_1".to_owned(),
        name: "spy".to_owned(),
        raw_arguments: "{}".to_owned(),
      },
      SuspendedToolCall {
        tool_call_id: "call_2".to_owned(),
        name: "other".to_owned(),
        raw_arguments: "{}".to_owned(),
      },
    ],
    budget_exhausted: false,
    reason: StopReason::AwaitingDecision,
    context: ExecutionContext::new(),
  };

  // Only the first is answered.
  let decisions = HashMap::from([("call_1".to_owned(), ResumedDecision::Approved)]);
  let stream = agent.resume_stream(state, decisions);
  futures::pin_mut!(stream);

  let first = stream
    .next()
    .await
    .expect("one item")
    .expect("not an error");
  let AgentStreamEvent::ToolCallsFinished(items) = first else {
    panic!("expected the completed half first, got {first:?}");
  };
  assert_eq!(items.len(), 1);

  let second = stream
    .next()
    .await
    .expect("one item")
    .expect("not an error");
  let AgentStreamEvent::Suspended(again) = second else {
    panic!("expected a suspension, got {second:?}");
  };
  assert_eq!(again.suspended.len(), 1);
  assert_eq!(again.suspended[0].tool_call_id, "call_2");

  assert!(spy_executed.load(Ordering::SeqCst));
  assert!(!other_executed.load(Ordering::SeqCst));
}

/// A state whose pending call is answered outright runs the tool. The stream then goes
/// on to request the model, which this test does not reach — asserting the tool ran is
/// the point, and is observable before that happens.
#[tokio::test]
async fn a_resumed_stream_runs_an_approved_call() {
  let executed = Arc::new(AtomicBool::new(false));
  let agent =
    agent_with(spy_registry(&executed)).with_before_tool_callback(Arc::new(SuspendEverything));
  let state = suspended_state(&agent, "call_1");

  let decisions = HashMap::from([("call_1".to_owned(), ResumedDecision::Approved)]);
  let stream = agent.resume_stream(state, decisions);
  futures::pin_mut!(stream);
  let first = stream
    .next()
    .await
    .expect("one item")
    .expect("not an error");

  assert!(
    matches!(first, AgentStreamEvent::ToolCallsFinished(ref items) if items.len() == 1),
    "expected the answered call's result, got {first:?}"
  );
  assert!(
    executed.load(Ordering::SeqCst),
    "the approved call should have reached the tool"
  );
}

/// A state with one pending `spy` call and an empty transcript.
fn suspended_state(agent: &Agent, tool_call_id: &str) -> AgentRunState {
  AgentRunState {
    fingerprint: agent.fingerprint(),
    suspended: vec![SuspendedToolCall {
      tool_call_id: tool_call_id.to_owned(),
      name: "spy".to_owned(),
      raw_arguments: "{}".to_owned(),
    }],
    budget_exhausted: false,
    reason: StopReason::AwaitingDecision,
    context: ExecutionContext::new(),
  }
}

// ---- text the model says alongside its tool calls --------------------------------

/// A model routinely explains a call on the same message it makes the call on. Dropping
/// that text loses its stated reasoning from the transcript — on the next round it sees
/// a bare tool call where it had explained itself.
#[test]
fn text_accompanying_a_round_is_recorded_before_the_calls() {
  let agent = agent_with(ToolRegistry::empty());
  let mut context = ExecutionContext::new();

  agent.record_assistant_text(&mut context, "Let me check the file exists first.");
  agent.record_tool_calls(&mut context, &[spy_call()]);

  let items: Vec<&ContentItem> = context
    .events
    .iter()
    .flat_map(|event| &event.content)
    .collect();

  assert_eq!(items.len(), 2);
  assert!(
    matches!(
      items[0],
      ContentItem::Message { role, content }
        if role == "assistant" && content == "Let me check the file exists first."
    ),
    "the text has to come first, so it lands on the same assistant message as the calls"
  );
  assert!(matches!(items[1], ContentItem::ToolCall { .. }));
}

/// Most rounds are a bare tool call. An empty assistant message would be one more thing
/// for every consumer of the transcript to skip, and some providers reject it outright.
#[test]
fn a_round_with_nothing_said_records_no_message() {
  let agent = agent_with(ToolRegistry::empty());
  let mut context = ExecutionContext::new();

  agent.record_assistant_text(&mut context, "");
  agent.record_assistant_text(&mut context, "   \n  ");

  assert!(context.events.is_empty());
}

/// The reason the text is recorded *before* the calls: `build_messages` starts an
/// assistant message for it and then appends the round's tool calls to that same
/// message, reproducing the one message the provider actually sent.
#[test]
fn accompanying_text_and_its_calls_build_into_one_assistant_message() {
  let agent = agent_with(ToolRegistry::empty());
  let mut context = ExecutionContext::new();

  agent.record_assistant_text(&mut context, "checking first");
  agent.record_tool_calls(&mut context, &[spy_call()]);

  let request = LlmRequest::new(None, &context.events);
  let messages = agent.build_messages(request).unwrap();

  assert_eq!(
    messages.len(),
    1,
    "the text and the calls belong to one message, not two"
  );
  let ChatCompletionRequestMessage::Assistant(assistant) = &messages[0] else {
    panic!("expected an assistant message, got {:?}", messages[0]);
  };
  assert!(assistant.content.is_some(), "the text must survive");
  assert_eq!(
    assistant.tool_calls.as_ref().map(Vec::len),
    Some(1),
    "and carry the call alongside it"
  );
}

// ---- surviving a process that does not get to clean up ---------------------------

/// What a crash recovery records has to match what actually happened. A round that was
/// in flight may have taken effect; saying it "was waiting for approval" — the wording
/// for a call that provably did not run — would have the model confidently retry
/// something that already happened.
#[test]
fn an_interrupted_round_is_closed_out_as_unknown_not_as_unapproved() {
  let executed = Arc::new(AtomicBool::new(false));
  let agent = agent_with(spy_registry(&executed));

  let state = AgentRunState {
    fingerprint: agent.fingerprint(),
    suspended: vec![SuspendedToolCall {
      tool_call_id: "call_1".to_owned(),
      name: "spy".to_owned(),
      raw_arguments: "{}".to_owned(),
    }],
    budget_exhausted: false,
    reason: StopReason::RoundInFlight,
    context: ExecutionContext::new(),
  };

  let context = state.abandon(GiveUp::Unanswered);
  let ContentItem::ToolResult { content, .. } = &context.events[0].content[0] else {
    panic!("expected a tool result");
  };

  assert!(
    content.contains("interrupted") && content.contains("unknown"),
    "an interrupted call's outcome is unknown, and the result has to say so: {content}"
  );
  assert!(
    !content.contains("approval"),
    "it must not claim the call was refused, which would be a different fact: {content}"
  );
}

/// The other wording is still used where it is true.
#[test]
fn an_unanswered_call_still_says_it_was_never_approved() {
  let executed = Arc::new(AtomicBool::new(false));
  let agent = agent_with(spy_registry(&executed));
  let state = suspended_state(&agent, "call_1");

  let context = state.abandon(GiveUp::Unanswered);
  let ContentItem::ToolResult { content, .. } = &context.events[0].content[0] else {
    panic!("expected a tool result");
  };

  assert!(content.contains("approval"), "got: {content}");
}

/// A state written before `reason` existed was, by construction, an ordinary
/// suspension — nothing wrote the other kind. Loading it as anything else would make a
/// resumable run look unrecoverable.
#[test]
fn a_state_without_a_reason_loads_as_awaiting_a_decision() {
  let json = serde_json::json!({
    "fingerprint": {
      "state_version": 1,
      "model": "gpt-test",
      "instructions_digest": "0",
      "tools": ["spy"],
    },
    "suspended": [{
      "tool_call_id": "call_1",
      "name": "spy",
      "raw_arguments": "{}",
    }],
    "budget_exhausted": false,
    "context": ExecutionContext::new(),
  });

  let state: AgentRunState = serde_json::from_value(json).expect("an older state should load");

  assert_eq!(state.reason, StopReason::AwaitingDecision);
}

/// The discriminator has to survive the round trip, since it is what decides whether a
/// recovered run may be resumed at all.
#[test]
fn the_stop_reason_round_trips_through_json() {
  let executed = Arc::new(AtomicBool::new(false));
  let agent = agent_with(spy_registry(&executed));
  let mut state = suspended_state(&agent, "call_1");
  state.reason = StopReason::RoundInFlight;

  let back: AgentRunState = serde_json::from_str(&serde_json::to_string(&state).unwrap()).unwrap();

  assert_eq!(back.reason, StopReason::RoundInFlight);
}

/// Records what a run asked its checkpoint to do, so a test can assert the write-ahead
/// happened rather than inferring it from a file on disk.
#[derive(Default)]
struct SpyCheckpoint {
  saved: Mutex<Vec<StopReason>>,
  cleared: AtomicUsize,
}

#[async_trait::async_trait]
impl RunCheckpoint for SpyCheckpoint {
  async fn save(&self, state: &AgentRunState) {
    self
      .saved
      .lock()
      .unwrap_or_else(|poisoned| poisoned.into_inner())
      .push(state.reason);
  }

  async fn clear(&self, _context: &ExecutionContext) {
    self.cleared.fetch_add(1, Ordering::SeqCst);
  }
}

impl SpyCheckpoint {
  fn saved(&self) -> Vec<StopReason> {
    self
      .saved
      .lock()
      .unwrap_or_else(|poisoned| poisoned.into_inner())
      .clone()
  }
}

/// A resumed round has to be checkpointed like any other, and this one matters most: the
/// caller took the stored state out of the store to get here, so between that `take` and
/// this write there is nothing on disk at all. A process dying in that window would lose
/// the pending question outright — strictly worse than never having stored it.
///
/// The non-streaming `resume` used to skip this while `resume_stream` did it, so the loss
/// depended only on which entry point a front-end happened to use.
#[tokio::test]
async fn resuming_checkpoints_the_round_it_re_attempts() {
  let executed = Arc::new(AtomicBool::new(false));
  let checkpoint = Arc::new(SpyCheckpoint::default());
  // Suspends again, so the resume returns before issuing any model request — the
  // checkpoint write is the only thing under test here.
  let agent = agent_with(spy_registry(&executed))
    .with_before_tool_callback(Arc::new(SuspendEverything))
    .with_checkpoint(Arc::clone(&checkpoint) as Arc<dyn RunCheckpoint>);
  let state = suspended_state(&agent, "call_1");

  let outcome = agent
    .resume(state, &HashMap::new())
    .await
    .expect("an unanswered resume suspends again rather than failing");

  assert!(matches!(outcome, AgentOutcome::Suspended(_)));
  assert_eq!(
    checkpoint.saved(),
    vec![StopReason::RoundInFlight],
    "the re-attempted round must be parked on disk before it runs, marked as in flight"
  );
  assert!(!executed.load(Ordering::SeqCst), "the tool must not run");
}

/// A refused resume must not touch the checkpoint: the state it was handed is still the
/// authoritative record, and writing over it — in either direction — would turn a
/// refusal into a mutation of the very run the refusal exists to protect.
#[tokio::test]
async fn a_refused_resume_leaves_the_checkpoint_alone() {
  let executed = Arc::new(AtomicBool::new(false));
  let checkpoint = Arc::new(SpyCheckpoint::default());
  let agent = agent_with(spy_registry(&executed))
    .with_checkpoint(Arc::clone(&checkpoint) as Arc<dyn RunCheckpoint>);
  let mut state = suspended_state(&agent, "call_1");
  state.reason = StopReason::RoundInFlight;

  assert!(
    agent.resume(state, &HashMap::new()).await.is_err(),
    "an interrupted round cannot be resumed"
  );

  assert!(checkpoint.saved().is_empty());
  assert_eq!(checkpoint.cleared.load(Ordering::SeqCst), 0);
}

/// The invariant the whole checkpoint mechanism protects: an interrupted round's calls
/// may already have taken effect, so re-running the batch could repeat a side effect.
/// Enforced on the agent, not just in whichever front-end happens to store the run.
#[tokio::test]
async fn an_interrupted_round_is_refused_by_resume() {
  let executed = Arc::new(AtomicBool::new(false));
  let agent = agent_with(spy_registry(&executed));
  let mut state = suspended_state(&agent, "call_1");
  state.reason = StopReason::RoundInFlight;

  let error = agent
    .resume(state, &HashMap::new())
    .await
    .expect_err("an interrupted round must not be resumed");

  assert!(
    !executed.load(Ordering::SeqCst),
    "the refusal has to come before the call runs, or it repeated the side effect"
  );
  assert!(error.to_string().contains("unknown"), "got: {error}");
}

/// Same invariant on the streaming path, where a refusal arrives as the first item
/// rather than as an `Err` around the stream.
#[tokio::test]
async fn an_interrupted_round_is_refused_by_resume_stream() {
  let executed = Arc::new(AtomicBool::new(false));
  let agent = agent_with(spy_registry(&executed));
  let mut state = suspended_state(&agent, "call_1");
  state.reason = StopReason::RoundInFlight;

  let stream = agent.resume_stream(state, HashMap::new());
  futures::pin_mut!(stream);
  let first = stream.next().await.expect("a refusal is still an item");

  assert!(first.is_err(), "expected a refusal, got {first:?}");
  assert!(!executed.load(Ordering::SeqCst));
}

/// An ordinary suspension stays resumable — the guard above must not have swallowed the
/// case the whole feature exists for.
#[test]
fn an_ordinary_suspension_is_resumable() {
  let executed = Arc::new(AtomicBool::new(false));
  let agent = agent_with(spy_registry(&executed));
  let state = suspended_state(&agent, "call_1");

  assert_eq!(state.reason, StopReason::AwaitingDecision);
  assert!(state.unresumable().is_none());
}

/// Giving up explicitly is a decision, and a decision has to be attributable — the same
/// rule a refusal reason follows. Recording it as "nobody answered" loses the only
/// instruction actually given, and invites the model to retry.
#[test]
fn a_user_giving_up_is_recorded_as_the_users_own_decision() {
  let executed = Arc::new(AtomicBool::new(false));
  let agent = agent_with(spy_registry(&executed));
  let state = suspended_state(&agent, "call_1");

  let context = state.abandon(GiveUp::ByUser);
  let ContentItem::ToolResult { content, .. } = &context.events[0].content[0] else {
    panic!("expected a tool result");
  };

  assert!(
    content.starts_with("User "),
    "it has to read as the runner's decision, not as a fact about the environment: \
     {content}"
  );
  assert!(
    !content.contains("never arrived"),
    "that is the wording for nobody answering, which is a different fact: {content}"
  );
}

/// An interrupted round stays "unknown" whichever way it is given up on: who pressed
/// what does not change whether the call took effect, and that is the more important
/// thing to tell the model.
#[test]
fn an_interrupted_round_reports_unknown_even_when_the_user_gives_up() {
  let executed = Arc::new(AtomicBool::new(false));
  let agent = agent_with(spy_registry(&executed));
  let mut state = suspended_state(&agent, "call_1");
  state.reason = StopReason::RoundInFlight;

  let context = state.abandon(GiveUp::ByUser);
  let ContentItem::ToolResult { content, .. } = &context.events[0].content[0] else {
    panic!("expected a tool result");
  };

  assert!(content.contains("unknown"), "got: {content}");
}

/// The announced results and the filed ones must be the same items: a live view is
/// already showing these calls and completes them from the broadcast, while a view that
/// arrives later reads history — and the two disagreeing is how one tab ends up
/// claiming something the other does not.
#[test]
fn the_results_announced_on_giving_up_are_the_ones_filed() {
  let executed = Arc::new(AtomicBool::new(false));
  let agent = agent_with(spy_registry(&executed));
  let state = suspended_state(&agent, "call_1");

  let announced = state.unanswered_results(GiveUp::ByUser);
  let filed = state.abandon(GiveUp::ByUser);

  // Compared as JSON because that is also the form both actually travel in — one over
  // the event stream, the other into the session file.
  assert_eq!(
    serde_json::to_value(&announced).unwrap(),
    serde_json::to_value(&filed.events[0].content).unwrap()
  );
}

#[test]
fn seed_context_appends_the_new_turn_after_prior_history() {
  let agent = agent_with(ToolRegistry::empty());
  let prior = vec![Event::new(
    "prev-execution",
    "user",
    vec![ContentItem::Message {
      role: "user".to_owned(),
      content: "hi".to_owned(),
    }],
  )];

  let context = agent.seed_context(prior.into(), "follow up");

  assert_eq!(context.events.len(), 2, "prior turn plus the new user turn");
  assert_eq!(context.events[0].author, "user");
  let new_turn = &context.events[1];
  assert_eq!(new_turn.execution_id, context.execution_id);
  let ContentItem::Message { content, .. } = &new_turn.content[0] else {
    panic!("expected a message");
  };
  assert_eq!(content, "follow up");
}

#[test]
fn seed_context_with_empty_history_only_has_the_new_turn() {
  let agent = agent_with(ToolRegistry::empty());
  let context = agent.seed_context(Vec::new().into(), "hi");
  assert_eq!(context.events.len(), 1);
}

#[test]
fn record_tool_calls_falls_back_to_null_on_malformed_arguments() {
  let agent = agent_with(ToolRegistry::empty());
  let mut context = ExecutionContext::new();

  let calls = vec![ChatCompletionMessageToolCalls::Function(
    ChatCompletionMessageToolCall {
      id: "call_1".to_owned(),
      function: FunctionCall {
        name: "calculator".to_owned(),
        arguments: "not json".to_owned(),
      },
    },
  )];

  agent.record_tool_calls(&mut context, &calls);

  let event = context.events.last().unwrap();
  let ContentItem::ToolCall { arguments, .. } = &event.content[0] else {
    panic!("expected a tool call");
  };
  assert!(arguments.is_null());
}
