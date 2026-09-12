//! The tool-calling loop shared by plain and structured completions.
//!
//! Streaming has its own loop in [`crate::llm::stream`] (fragments must be reassembled
//! first), but reuses [`disable_tools`] and [`append_tool_results`] from here.

use std::fmt;

use async_openai::types::chat::{
  ChatChoice, ChatCompletionMessageCustomToolCall, ChatCompletionMessageToolCall,
  ChatCompletionMessageToolCalls, ChatCompletionRequestAssistantMessageArgs,
  ChatCompletionRequestMessage, ChatCompletionRequestToolMessageArgs,
  ChatCompletionToolChoiceOption, ChatCompletionTools, CreateChatCompletionRequestArgs,
  FunctionCall, ResponseFormat, ToolChoiceOptions,
};
use futures::future::join_all;

use crate::{
  config,
  llm::{
    client::{first_choice, request_builder},
    provider::Provider,
    retry::{is_transient, with_retry},
  },
  tools::ToolRegistry,
  util::truncate_chars,
};

/// Chars of tool arguments/results kept in debug logs, to avoid dumping large or
/// sensitive payloads in full.
const TOOL_LOG_PREVIEW_CHARS: usize = 500;

/// The outcome of a tool-enabled completion.
#[derive(Debug)]
pub struct Completion {
  pub choice: ChatChoice,
  /// `true` when the round budget ran out and the answer was produced with tools
  /// disabled, i.e. the model had to work from partial information. Callers may want
  /// to treat such answers as lower confidence, or report them separately.
  pub budget_exhausted: bool,
}

/// Marker attached to failures that happened after the tool budget was spent.
///
/// Such a failure is effectively deterministic: a retry replays the whole conversation
/// and spends the entire budget again before failing the same way. Callers use it to
/// skip retries (see `gaia::solver`).
#[derive(Debug)]
pub struct BudgetExhausted;

impl fmt::Display for BudgetExhausted {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(
      f,
      "failed after the tool round budget was spent, so retrying would repeat the full budget"
    )
  }
}

/// Drive a completion to a final answer, executing tool calls as the model requests them.
///
/// Returns the first choice of the last response, i.e. one that carries no tool calls.
/// When the registry is empty this collapses to a single request.
///
/// The round budget ([`config::max_tool_rounds`]) is not a hard failure: once it is spent,
/// one last request goes out with tools disabled, so a long task still returns an answer
/// built from the results already gathered instead of losing all of that work. That case
/// is flagged through [`Completion::budget_exhausted`].
///
/// `provider` supplies both the credentials for the model call and the concurrency slot
/// it is charged against (see [`Provider::acquire`]) — held only for the duration of that
/// one call, not the whole loop, so time spent running tools does not also block other
/// callers sharing the same provider from reaching the model.
pub async fn run(
  provider: &Provider,
  model: &str,
  mut messages: Vec<ChatCompletionRequestMessage>,
  registry: &ToolRegistry,
  max_tokens: u32,
  response_format: Option<ResponseFormat>,
) -> anyhow::Result<Completion> {
  let max_rounds = config::max_tool_rounds();
  let mut round: usize = 0;

  loop {
    let tools_allowed = round < max_rounds;
    // Only meaningful when tools were actually on offer: with an empty tool list there is
    // no loop to exhaust, so the flag must stay false.
    let budget_exhausted = !tools_allowed && !registry.is_empty();

    let mut builder = request_builder(model, messages.clone(), max_tokens, registry.definitions());
    if let Some(response_format) = response_format.clone() {
      builder.response_format(response_format);
    }
    if !tools_allowed {
      disable_tools(&mut builder, registry.definitions());
    }
    // Only worth warning about when tools were actually taken away.
    if budget_exhausted {
      tracing::warn!(
        max_rounds,
        "tool round budget spent; forcing a final answer"
      );
    }

    // Transient failures (rate limits, network blips) are retried with backoff.
    // `is_transient` filters out the deterministic ones — a rejected key or a malformed
    // request fails identically on every attempt, so retrying only spends the budget and
    // delays the real error.
    let response = with_retry(
      || async {
        let _permit = provider.acquire().await?;
        let response = provider.client().chat().create(builder.build()?).await?;
        anyhow::Ok(response)
      },
      is_transient,
    )
    .await?;
    // The full response can be long; only log metadata to inspect usage and trace id.
    tracing::debug!(id = %response.id, round, usage = ?response.usage, "completion finished");

    let choice = first_choice(response)?;

    // No tool calls means the model produced its final answer.
    let Some(tool_calls) = choice
      .message
      .tool_calls
      .as_ref()
      .filter(|calls| !calls.is_empty())
      .cloned()
    else {
      return Ok(Completion {
        choice,
        budget_exhausted,
      });
    };

    // The model ignored `tool_choice = none`. Returning its message beats looping
    // forever; the caller still sees whatever content came back.
    if !tools_allowed {
      tracing::warn!("model requested tools after they were disabled");
      return Ok(Completion {
        choice,
        budget_exhausted,
      });
    }

    round += 1;
    append_tool_results(registry, &mut messages, tool_calls, choice.message.content).await?;
  }
}

/// Forbid further tool calls on this request.
///
/// The definitions stay in place so the transcript remains self-consistent; only the
/// choice is pinned to `none`. Skipped when there are no tools, since `tool_choice`
/// without `tools` is meaningless and rejected by some endpoints.
pub(crate) fn disable_tools(
  builder: &mut CreateChatCompletionRequestArgs,
  tools: &[ChatCompletionTools],
) {
  if tools.is_empty() {
    return;
  }
  builder.tool_choice(ChatCompletionToolChoiceOption::Mode(
    ToolChoiceOptions::None,
  ));
}

/// Append the assistant message that requested the tools, then one tool message per call.
///
/// The assistant message must come first, and **every** tool call needs a matching tool
/// message — even a failing or unsupported one: the API rejects a follow-up request that
/// leaves a `tool_call_id` unanswered.
pub(crate) async fn append_tool_results(
  registry: &ToolRegistry,
  messages: &mut Vec<ChatCompletionRequestMessage>,
  tool_calls: Vec<ChatCompletionMessageToolCalls>,
  assistant_text: Option<String>,
) -> anyhow::Result<()> {
  let mut assistant = ChatCompletionRequestAssistantMessageArgs::default();
  assistant.tool_calls(tool_calls.clone());
  // Some models emit prose alongside the tool call; keep it so the context stays coherent.
  if let Some(text) = assistant_text.filter(|text| !text.trim().is_empty()) {
    assistant.content(text);
  }
  messages.push(assistant.build()?.into());

  // Independent tool calls in the same round are executed concurrently: sequential
  // awaits would otherwise add up their network latency instead of overlapping it.
  let results = join_all(
    tool_calls
      .into_iter()
      .map(|tool_call| execute(registry, tool_call)),
  )
  .await;

  for (id, result) in results {
    messages.push(
      ChatCompletionRequestToolMessageArgs::default()
        .tool_call_id(id)
        .content(result)
        .build()?
        .into(),
    );
  }

  Ok(())
}

/// Execute one tool call, returning its id and the text to feed back to the model.
async fn execute(
  registry: &ToolRegistry,
  tool_call: ChatCompletionMessageToolCalls,
) -> (String, String) {
  match tool_call {
    ChatCompletionMessageToolCalls::Function(ChatCompletionMessageToolCall {
      id,
      function: FunctionCall { name, arguments },
    }) => {
      tracing::debug!(
        tool = %name,
        arguments = %truncate_chars(&arguments, TOOL_LOG_PREVIEW_CHARS),
        "executing tool"
      );
      let result = registry.execute(&name, &arguments).await;
      tracing::debug!(
        tool = %name,
        result = %truncate_chars(&result, TOOL_LOG_PREVIEW_CHARS),
        "tool finished"
      );
      (id, result)
    }
    // Custom tools are not supported yet, but still need a reply to keep the
    // tool_call / tool_message pairing valid.
    ChatCompletionMessageToolCalls::Custom(ChatCompletionMessageCustomToolCall {
      id,
      custom_tool,
    }) => {
      tracing::warn!(tool = %custom_tool.name, "unsupported custom tool call");
      (
        id,
        format!("Error: custom tool `{}` is not supported", custom_tool.name),
      )
    }
  }
}

#[cfg(test)]
mod tests {
  use async_openai::types::chat::{
    ChatCompletionRequestAssistantMessageContent, ChatCompletionRequestMessage, CustomTool,
  };

  use super::*;
  use crate::tools::calculator::{self, Calculator};

  fn function_call(id: &str, name: &str, arguments: &str) -> ChatCompletionMessageToolCalls {
    ChatCompletionMessageToolCalls::Function(ChatCompletionMessageToolCall {
      id: id.to_owned(),
      function: FunctionCall {
        name: name.to_owned(),
        arguments: arguments.to_owned(),
      },
    })
  }

  #[test]
  fn disable_tools_is_noop_without_any_tools() {
    let mut builder = request_builder("gpt-test", Vec::new(), 16, &[]);
    disable_tools(&mut builder, &[]);
    let request = builder.build().unwrap();
    assert_eq!(request.tool_choice, None);
  }

  #[test]
  fn disable_tools_pins_choice_to_none_when_tools_are_present() {
    let registry = ToolRegistry::builtin().unwrap();
    let mut builder = request_builder("gpt-test", Vec::new(), 16, registry.definitions());
    disable_tools(&mut builder, registry.definitions());
    let request = builder.build().unwrap();
    assert_eq!(
      request.tool_choice,
      Some(ChatCompletionToolChoiceOption::Mode(
        ToolChoiceOptions::None
      ))
    );
    // The definitions stay on the request; only the choice is pinned.
    assert!(request.tools.is_some());
  }

  #[tokio::test]
  async fn append_tool_results_pairs_every_call_with_a_tool_message() {
    let mut registry = ToolRegistry::empty();
    registry.add(std::sync::Arc::new(Calculator)).unwrap();
    let mut messages = Vec::new();

    let calls = vec![function_call(
      "call_1",
      calculator::NAME,
      r#"{"operator":"add","first_number":1,"second_number":2}"#,
    )];

    append_tool_results(&registry, &mut messages, calls, None)
      .await
      .unwrap();

    assert_eq!(messages.len(), 2, "assistant message + one tool message");
    assert!(matches!(
      messages[0],
      ChatCompletionRequestMessage::Assistant(_)
    ));
    let ChatCompletionRequestMessage::Tool(tool_message) = &messages[1] else {
      panic!("expected a tool message");
    };
    assert_eq!(tool_message.tool_call_id, "call_1");
  }

  #[tokio::test]
  async fn append_tool_results_keeps_narration_alongside_the_tool_call() {
    let registry = ToolRegistry::empty();
    let mut messages = Vec::new();

    let calls = vec![function_call("call_1", "nope", "{}")];
    append_tool_results(
      &registry,
      &mut messages,
      calls,
      Some("let me check".to_owned()),
    )
    .await
    .unwrap();

    let ChatCompletionRequestMessage::Assistant(assistant) = &messages[0] else {
      panic!("expected an assistant message");
    };
    assert_eq!(
      assistant.content,
      Some(ChatCompletionRequestAssistantMessageContent::Text(
        "let me check".to_owned()
      ))
    );
  }

  #[tokio::test]
  async fn append_tool_results_drops_blank_narration() {
    let registry = ToolRegistry::empty();
    let mut messages = Vec::new();

    let calls = vec![function_call("call_1", "nope", "{}")];
    append_tool_results(&registry, &mut messages, calls, Some("   ".to_owned()))
      .await
      .unwrap();

    let ChatCompletionRequestMessage::Assistant(assistant) = &messages[0] else {
      panic!("expected an assistant message");
    };
    assert!(assistant.content.is_none());
  }

  #[tokio::test]
  async fn execute_dispatches_function_calls_through_the_registry() {
    let mut registry = ToolRegistry::empty();
    registry.add(std::sync::Arc::new(Calculator)).unwrap();

    let (id, result) = execute(
      &registry,
      function_call(
        "call_1",
        calculator::NAME,
        r#"{"operator":"add","first_number":2,"second_number":3}"#,
      ),
    )
    .await;

    assert_eq!(id, "call_1");
    assert_eq!(result, "5");
  }

  #[tokio::test]
  async fn execute_reports_custom_tool_calls_as_unsupported() {
    let registry = ToolRegistry::empty();

    let (id, result) = execute(
      &registry,
      ChatCompletionMessageToolCalls::Custom(ChatCompletionMessageCustomToolCall {
        id: "call_1".to_owned(),
        custom_tool: CustomTool {
          name: "shell".to_owned(),
          input: "ls".to_owned(),
        },
      }),
    )
    .await;

    assert_eq!(id, "call_1");
    assert!(result.contains("shell"), "got: {result}");
    assert!(result.contains("not supported"), "got: {result}");
  }
}
