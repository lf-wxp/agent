use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use serde_json::json;

use super::*;
use crate::tools::{
  Tool,
  calculator::{self, Calculator},
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
fn new_defaults_max_history_tokens_to_the_shared_config() {
  let agent = agent_with(ToolRegistry::empty());
  assert_eq!(agent.max_history_tokens, config::max_history_tokens());
}

#[test]
fn with_max_history_tokens_overrides_the_default() {
  let agent = agent_with(ToolRegistry::empty()).with_max_history_tokens(42);
  assert_eq!(agent.max_history_tokens, 42);
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

  let messages = agent.build_messages(&context).unwrap();

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

  let messages = agent.build_messages(&context).unwrap();
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

  let context = agent.seed_context(prior, "follow up");

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
  let context = agent.seed_context(Vec::new(), "hi");
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
