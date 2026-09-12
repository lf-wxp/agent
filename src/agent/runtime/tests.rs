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

  agent.execute_tool_calls(&mut context, &calls).await;

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
  ) -> Option<(ToolResultStatus, String)> {
    Some((
      ToolResultStatus::Error,
      format!("denied {}", tool_call.name),
    ))
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
  ) -> Option<(ToolResultStatus, String)> {
    self.0.fetch_add(1, Ordering::SeqCst);
    None
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

  agent.execute_tool_calls(&mut context, &[spy_call()]).await;

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

  agent.execute_tool_calls(&mut context, &[spy_call()]).await;

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

  agent.execute_tool_calls(&mut context, &[spy_call()]).await;

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

  agent.execute_tool_calls(&mut context, &[spy_call()]).await;

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

  agent.execute_tool_calls(&mut context, &[spy_call()]).await;

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

  agent.execute_tool_calls(&mut context, &[spy_call()]).await;

  let ContentItem::ToolResult { content, .. } = &context.events.last().unwrap().content[0] else {
    panic!("expected a tool result");
  };
  assert_eq!(
    content, "real result-a-b",
    "each hook must see the previous hook's output, in registration order"
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
