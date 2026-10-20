use futures::StreamExt;
use serde_json::{Value, json};

use super::*;
use crate::llm::{
  settings::{ModelSettings, TokenLimitField},
  testing::{ScriptedBackend, chunk, reply},
};

fn agent_on(backend: Arc<ScriptedBackend>, toolbox: ToolRegistry) -> Agent {
  Agent::new(
    backend.provider(),
    "gpt-test",
    Option::<String>::None,
    Arc::new(toolbox),
  )
  .with_model_settings(ModelSettings::default())
}

fn calculator_call(id: &str) -> Value {
  json!({
    "id": id,
    "type": "function",
    "function": {
      "name": "calculator",
      "arguments": r#"{"operator":"add","first_number":2,"second_number":3}"#,
    },
  })
}

fn calculator_registry() -> ToolRegistry {
  ToolRegistry::select(&["calculator".to_owned()]).unwrap()
}

/// Every text the transcript holds, in order — to assert reasoning never lands in it.
fn transcript_text(context: &ExecutionContext) -> String {
  serde_json::to_string(&context.events).unwrap()
}

#[tokio::test]
async fn a_run_reports_the_reasoning_behind_its_answer() {
  let backend = ScriptedBackend::replying(vec![reply(json!({
    "role": "assistant",
    "content": "42",
    "reasoning_content": "multiply the two numbers",
  }))]);
  let agent = agent_on(Arc::clone(&backend), ToolRegistry::empty());

  let result = agent.run("what is six times seven?").await.unwrap();

  assert_eq!(result.output, "42");
  assert_eq!(
    result.reasoning.as_deref(),
    Some("multiply the two numbers")
  );
  assert!(
    !transcript_text(&result.context).contains("multiply the two numbers"),
    "reasoning must stay out of the transcript, or it is sent back next round"
  );
}

/// The reasoning reported is the final answer's, not a tool round's — and none of it is
/// replayed in the follow-up request.
#[tokio::test]
async fn only_the_final_rounds_reasoning_is_reported_and_none_is_replayed() {
  let backend = ScriptedBackend::replying(vec![
    reply(json!({
      "role": "assistant",
      "content": null,
      "reasoning_content": "I should add first",
      "tool_calls": [calculator_call("call_1")],
    })),
    reply(json!({
      "role": "assistant",
      "content": "5",
      "reasoning_content": "the tool said 5",
    })),
  ]);
  let agent = agent_on(Arc::clone(&backend), calculator_registry());

  let result = agent.run("add 2 and 3").await.unwrap();

  assert_eq!(result.output, "5");
  assert_eq!(result.reasoning.as_deref(), Some("the tool said 5"));
  let seen = backend.seen();
  assert_eq!(seen.len(), 2);
  assert!(
    !seen[1].to_string().contains("I should add first"),
    "the first round's reasoning was replayed: {}",
    seen[1]
  );
}

#[tokio::test]
async fn a_model_that_does_not_reason_reports_none() {
  let backend = ScriptedBackend::replying(vec![reply(json!({
    "role": "assistant",
    "content": "hi",
  }))]);
  let result = agent_on(backend, ToolRegistry::empty())
    .run("hello")
    .await
    .unwrap();
  assert!(result.reasoning.is_none());
}

/// Settings reach every request the agent sends, including a tool round's follow-up.
#[tokio::test]
async fn model_settings_reach_every_request() {
  let backend = ScriptedBackend::replying(vec![
    reply(json!({
      "role": "assistant",
      "content": null,
      "tool_calls": [calculator_call("call_1")],
    })),
    reply(json!({ "role": "assistant", "content": "5" })),
  ]);
  let agent =
    agent_on(Arc::clone(&backend), calculator_registry()).with_model_settings(ModelSettings {
      max_tokens: 321,
      token_limit_field: TokenLimitField::MaxCompletionTokens,
      temperature: Some(0.3),
      top_p: None,
      reasoning_effort: Some(async_openai::types::chat::ReasoningEffort::Low),
      replay_reasoning: false,
    });

  agent.run("add 2 and 3").await.unwrap();

  let seen = backend.seen();
  assert_eq!(seen.len(), 2);
  for request in seen {
    assert_eq!(request["max_completion_tokens"], 321, "got: {request}");
    assert!(request.get("max_tokens").is_none(), "got: {request}");
    assert!(
      (request["temperature"].as_f64().unwrap() - 0.3).abs() < 1e-6,
      "got: {request}"
    );
    assert!(request.get("top_p").is_none(), "got: {request}");
    assert_eq!(request["reasoning_effort"], "low");
  }
}

/// Reasoning streams ahead of the answer, as its own events, and never reaches `Done`'s
/// transcript.
#[tokio::test]
async fn a_stream_forwards_reasoning_ahead_of_the_answer() {
  let backend = ScriptedBackend::streaming(vec![vec![
    chunk(json!({ "role": "assistant", "reasoning_content": "thinking " })),
    chunk(json!({ "reasoning_content": "hard" })),
    chunk(json!({ "content": "4", "reasoning_content": "" })),
    chunk(json!({ "content": "2" })),
  ]]);
  let agent = agent_on(backend, ToolRegistry::empty());

  let stream = agent.run_stream("what is six times seven?");
  futures::pin_mut!(stream);
  let mut order = Vec::new();
  let mut done_context = None;
  while let Some(event) = stream.next().await {
    match event.unwrap() {
      AgentStreamEvent::Reasoning(text) => order.push(format!("R:{text}")),
      AgentStreamEvent::Token(text) => order.push(format!("T:{text}")),
      AgentStreamEvent::Done {
        output, context, ..
      } => {
        assert_eq!(output, "42");
        done_context = Some(context);
      }
      other => panic!("unexpected event {other:?}"),
    }
  }

  assert_eq!(order, ["R:thinking ", "R:hard", "T:4", "T:2"]);
  let context = done_context.expect("the stream ends with Done");
  assert!(!transcript_text(&context).contains("thinking"));
}

/// The streaming route sends the same settings as the plain one.
#[tokio::test]
async fn a_stream_sends_the_model_settings() {
  let backend = ScriptedBackend::streaming(vec![vec![chunk(json!({ "content": "ok" }))]]);
  let agent =
    agent_on(Arc::clone(&backend), ToolRegistry::empty()).with_model_settings(ModelSettings {
      max_tokens: 99,
      top_p: Some(0.9),
      ..ModelSettings::default()
    });

  let stream = agent.run_stream("hi");
  futures::pin_mut!(stream);
  while stream.next().await.is_some() {}

  let request = &backend.seen()[0];
  assert_eq!(request["max_tokens"], 99);
  assert!((request["top_p"].as_f64().unwrap() - 0.9).abs() < 1e-6);
  assert_eq!(request["stream_options"]["include_usage"], true);
}

/// A structured route raises the cap to its own floor, but a larger configured cap
/// stands.
#[tokio::test]
async fn structured_routes_raise_the_cap_but_never_lower_it() {
  #[derive(serde::Deserialize, schemars::JsonSchema)]
  #[allow(dead_code)]
  struct Answer {
    value: u32,
  }

  for (configured, expect_at_least) in [(100, 4096), (50_000, 50_000)] {
    let backend = ScriptedBackend::replying(vec![reply(json!({
      "role": "assistant",
      "content": r#"{"value": 1}"#,
    }))]);
    // A model name that forces the response-format route, which is the one that applies
    // the structured budget.
    let agent = Agent::new(
      backend.provider(),
      "deepseek-test",
      Option::<String>::None,
      Arc::new(ToolRegistry::empty()),
    )
    .with_model_settings(ModelSettings {
      max_tokens: configured,
      ..ModelSettings::default()
    });

    agent.run_structured::<Answer>("give me one").await.unwrap();

    let sent = backend.seen()[0]["max_tokens"].as_u64().unwrap();
    assert!(
      sent >= expect_at_least,
      "configured {configured}, sent {sent}, expected at least {expect_at_least}"
    );
  }
}

fn replay_on() -> ModelSettings {
  ModelSettings {
    replay_reasoning: true,
    ..ModelSettings::default()
  }
}

/// The `reasoning_content` of each message in a recorded request body, in order.
fn replayed(request: &Value) -> Vec<Option<&str>> {
  request["messages"]
    .as_array()
    .unwrap()
    .iter()
    .map(|message| message.get("reasoning_content").and_then(Value::as_str))
    .collect()
}

/// A provider such as DeepSeek's thinking mode rejects a request whose earlier assistant
/// messages arrive without the chain of thought they were produced with — within one
/// run's tool rounds, and again on the next turn.
#[tokio::test]
async fn replayed_reasoning_goes_back_within_a_run_and_on_the_next_turn() {
  let backend = ScriptedBackend::replying(vec![
    reply(json!({
      "role": "assistant",
      "content": null,
      "reasoning_content": "I should add first",
      "tool_calls": [calculator_call("call_1")],
    })),
    reply(json!({
      "role": "assistant",
      "content": "5",
      "reasoning_content": "the tool said 5",
    })),
    reply(json!({ "role": "assistant", "content": "still 5" })),
  ]);
  let agent =
    agent_on(Arc::clone(&backend), calculator_registry()).with_model_settings(replay_on());

  let first = agent.run("add 2 and 3").await.unwrap();
  assert_eq!(first.reasoning.as_deref(), Some("the tool said 5"));
  agent
    .run_continuing(first.context.events.clone(), "and again?")
    .await
    .unwrap();

  let seen = backend.seen();
  assert_eq!(seen.len(), 3);
  assert_eq!(
    replayed(&seen[0]),
    [None],
    "nothing to replay on the first request"
  );
  // user, assistant (the tool call), tool
  assert_eq!(replayed(&seen[1]), [None, Some("I should add first"), None]);
  // user, assistant (the tool call), tool, assistant (the answer), user
  assert_eq!(
    replayed(&seen[2]),
    [
      None,
      Some("I should add first"),
      None,
      Some("the tool said 5"),
      None
    ]
  );
}

/// The reasoning sits in the transcript only because replay asked for it, immediately
/// before the message it belongs to.
#[tokio::test]
async fn replay_records_reasoning_before_its_message_and_default_does_not() {
  let script = || {
    vec![reply(json!({
      "role": "assistant",
      "content": "42",
      "reasoning_content": "multiply",
    }))]
  };

  let replaying = agent_on(ScriptedBackend::replying(script()), ToolRegistry::empty())
    .with_model_settings(replay_on());
  let result = replaying.run("six times seven?").await.unwrap();
  let items: Vec<&ContentItem> = result
    .context
    .events
    .iter()
    .flat_map(|event| &event.content)
    .collect();
  assert!(
    matches!(
      items.as_slice(),
      [
        ContentItem::Message { .. },
        ContentItem::Reasoning { text },
        ContentItem::Message { .. },
      ] if text == "multiply"
    ),
    "got: {items:?}"
  );

  let plain = agent_on(ScriptedBackend::replying(script()), ToolRegistry::empty());
  let result = plain.run("six times seven?").await.unwrap();
  assert!(
    !transcript_text(&result.context).contains("multiply"),
    "reasoning must stay out of the transcript unless replay is on"
  );
}

/// The streaming route assembles the deltas into one chain of thought per round and
/// replays it, just as the plain route does.
#[tokio::test]
async fn a_stream_replays_the_reasoning_it_assembled() {
  let backend = ScriptedBackend::streaming(vec![
    vec![
      chunk(json!({ "role": "assistant", "reasoning_content": "I should " })),
      chunk(json!({ "reasoning_content": "add first" })),
      chunk(json!({ "tool_calls": [{
        "index": 0, "id": "call_1", "type": "function",
        "function": {
          "name": "calculator",
          "arguments": r#"{"operator":"add","first_number":2,"second_number":3}"#,
        },
      }] })),
    ],
    vec![
      chunk(json!({ "reasoning_content": "done" })),
      chunk(json!({ "content": "5" })),
    ],
  ]);
  let agent =
    agent_on(Arc::clone(&backend), calculator_registry()).with_model_settings(replay_on());

  let stream = agent.run_stream("add 2 and 3");
  futures::pin_mut!(stream);
  while let Some(event) = stream.next().await {
    event.unwrap();
  }

  let seen = backend.seen();
  assert_eq!(seen.len(), 2);
  assert_eq!(replayed(&seen[1]), [None, Some("I should add first"), None]);
}

/// The structured `final_answer` route builds the same transcript, so it replays too.
#[tokio::test]
async fn the_structured_route_replays_reasoning_between_its_tool_rounds() {
  #[derive(serde::Deserialize, schemars::JsonSchema)]
  #[allow(dead_code)]
  struct Answer {
    value: u32,
  }

  let backend = ScriptedBackend::replying(vec![
    reply(json!({
      "role": "assistant",
      "content": null,
      "reasoning_content": "add first",
      "tool_calls": [calculator_call("call_1")],
    })),
    reply(json!({
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_2", "type": "function",
        "function": {
          "name": crate::llm::schema::FINAL_ANSWER_TOOL_NAME,
          "arguments": r#"{"value":5}"#,
        },
      }],
    })),
  ]);
  let agent =
    agent_on(Arc::clone(&backend), calculator_registry()).with_model_settings(replay_on());

  agent.run_structured::<Answer>("add 2 and 3").await.unwrap();

  let seen = backend.seen();
  assert_eq!(seen.len(), 2);
  assert_eq!(replayed(&seen[1]), [None, Some("add first"), None]);
}

/// A reasoning whose message is gone must not end up on a different one.
#[test]
fn an_orphaned_reasoning_is_dropped_not_handed_to_the_next_message() {
  let agent = agent_on(ScriptedBackend::replying(Vec::new()), ToolRegistry::empty())
    .with_model_settings(replay_on());
  let events = vec![Event::new(
    "run",
    "agent",
    vec![
      ContentItem::Message {
        role: "user".to_owned(),
        content: "first".to_owned(),
      },
      ContentItem::Reasoning {
        text: "belongs to a message that was trimmed".to_owned(),
      },
      ContentItem::Message {
        role: "user".to_owned(),
        content: "second".to_owned(),
      },
      ContentItem::Message {
        role: "assistant".to_owned(),
        content: "answer".to_owned(),
      },
    ],
  )];

  let wire = agent
    .build_messages(LlmRequest::new(None, &events))
    .unwrap();

  assert_eq!(wire.messages.len(), 3);
  assert!(
    wire.reasoning.is_empty(),
    "the orphan landed on a message: {:?}",
    wire.reasoning
  );
}

/// With replay off, reasoning already sitting in a stored transcript is not sent.
#[test]
fn stored_reasoning_is_not_sent_unless_replay_is_on() {
  let agent = agent_on(ScriptedBackend::replying(Vec::new()), ToolRegistry::empty());
  let events = vec![Event::new(
    "run",
    "agent",
    vec![
      ContentItem::Message {
        role: "user".to_owned(),
        content: "question".to_owned(),
      },
      ContentItem::Reasoning {
        text: "thinking".to_owned(),
      },
      ContentItem::Message {
        role: "assistant".to_owned(),
        content: "answer".to_owned(),
      },
    ],
  )];

  let wire = agent
    .build_messages(LlmRequest::new(None, &events))
    .unwrap();

  assert_eq!(wire.messages.len(), 2);
  assert!(wire.reasoning.is_empty());
}
