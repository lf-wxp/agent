//! Reassembly of `tool_calls` that arrive as fragments in streaming mode.
//!
//! A streamed tool call is split across chunks: the first fragment carries `id` and
//! `function.name`, later fragments only append to `function.arguments`. The `index`
//! field is the only one present on every fragment, so it is what groups them.

use std::collections::BTreeMap;

use async_openai::types::chat::{
  ChatCompletionMessageToolCall, ChatCompletionMessageToolCallChunk,
  ChatCompletionMessageToolCalls, FunctionCall,
};

/// Arguments sent for a tool that streamed no arguments at all.
const EMPTY_ARGUMENTS: &str = "{}";

/// Accumulates tool-call fragments across streaming chunks.
#[derive(Debug, Default)]
pub struct ToolCallAccumulator {
  /// `BTreeMap` keeps calls ordered by `index`, i.e. the order the model emitted them.
  calls: BTreeMap<u32, Partial>,
}

/// A tool call that is still being assembled.
#[derive(Debug, Default)]
struct Partial {
  id: Option<String>,
  name: Option<String>,
  arguments: String,
}

impl ToolCallAccumulator {
  /// Merge the fragments of a single chunk.
  pub fn push(&mut self, fragments: &[ChatCompletionMessageToolCallChunk]) {
    for fragment in fragments {
      let partial = self.calls.entry(fragment.index).or_default();

      // `id` / `name` appear once, in the opening fragment. Some providers repeat them
      // as empty strings rather than `None`, so blanks must not overwrite what we have.
      if let Some(id) = &fragment.id
        && !id.is_empty()
      {
        partial.id = Some(id.clone());
      }

      let Some(function) = &fragment.function else {
        continue;
      };
      if let Some(name) = &function.name
        && !name.is_empty()
      {
        partial.name = Some(name.clone());
      }
      if let Some(arguments) = &function.arguments {
        partial.arguments.push_str(arguments);
      }
    }
  }

  pub fn is_empty(&self) -> bool {
    self.calls.is_empty()
  }

  /// Turn the fragments into complete tool calls.
  ///
  /// Fails when a call never received an `id` or a name: it cannot be executed, and
  /// dropping it silently would leave the next request with an unanswered tool call.
  pub fn finish(self) -> anyhow::Result<Vec<ChatCompletionMessageToolCalls>> {
    self
      .calls
      .into_iter()
      .map(|(index, partial)| {
        let id = partial
          .id
          .ok_or_else(|| anyhow::anyhow!("streamed tool call at index {index} has no id"))?;
        let name = partial
          .name
          .ok_or_else(|| anyhow::anyhow!("streamed tool call `{id}` has no function name"))?;
        let arguments = if partial.arguments.trim().is_empty() {
          EMPTY_ARGUMENTS.to_owned()
        } else {
          partial.arguments
        };

        Ok(ChatCompletionMessageToolCalls::Function(
          ChatCompletionMessageToolCall {
            id,
            function: FunctionCall { name, arguments },
          },
        ))
      })
      .collect()
  }
}

#[cfg(test)]
mod tests {
  use async_openai::types::chat::{FunctionCallStream, FunctionType};

  use super::*;

  /// Build one fragment the way providers emit them.
  fn fragment(
    index: u32,
    id: Option<&str>,
    name: Option<&str>,
    arguments: Option<&str>,
  ) -> ChatCompletionMessageToolCallChunk {
    ChatCompletionMessageToolCallChunk {
      index,
      id: id.map(str::to_owned),
      r#type: Some(FunctionType::Function),
      function: Some(FunctionCallStream {
        name: name.map(str::to_owned),
        arguments: arguments.map(str::to_owned),
      }),
    }
  }

  /// Extract `(id, name, arguments)` for assertions.
  fn parts(call: &ChatCompletionMessageToolCalls) -> (&str, &str, &str) {
    match call {
      ChatCompletionMessageToolCalls::Function(call) => {
        (&call.id, &call.function.name, &call.function.arguments)
      }
      ChatCompletionMessageToolCalls::Custom(_) => panic!("expected a function call"),
    }
  }

  #[test]
  fn joins_arguments_split_across_chunks() {
    let mut accumulator = ToolCallAccumulator::default();
    accumulator.push(&[fragment(0, Some("call_1"), Some("calculator"), Some(""))]);
    accumulator.push(&[fragment(0, None, None, Some("{\"operator\":"))]);
    accumulator.push(&[fragment(0, None, None, Some("\"add\"}"))]);

    let calls = accumulator.finish().unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(
      parts(&calls[0]),
      ("call_1", "calculator", "{\"operator\":\"add\"}")
    );
  }

  #[test]
  fn groups_parallel_calls_by_index_in_order() {
    let mut accumulator = ToolCallAccumulator::default();
    // Providers may interleave fragments of concurrent calls.
    accumulator.push(&[fragment(1, Some("call_2"), Some("second"), Some("{\"b\":"))]);
    accumulator.push(&[fragment(0, Some("call_1"), Some("first"), Some("{\"a\":"))]);
    accumulator.push(&[fragment(0, None, None, Some("1}"))]);
    accumulator.push(&[fragment(1, None, None, Some("2}"))]);

    let calls = accumulator.finish().unwrap();
    assert_eq!(calls.len(), 2);
    assert_eq!(parts(&calls[0]), ("call_1", "first", "{\"a\":1}"));
    assert_eq!(parts(&calls[1]), ("call_2", "second", "{\"b\":2}"));
  }

  #[test]
  fn blank_repeats_do_not_clobber_id_and_name() {
    let mut accumulator = ToolCallAccumulator::default();
    accumulator.push(&[fragment(0, Some("call_1"), Some("calculator"), None)]);
    accumulator.push(&[fragment(0, Some(""), Some(""), Some("{}"))]);

    let calls = accumulator.finish().unwrap();
    assert_eq!(parts(&calls[0]), ("call_1", "calculator", "{}"));
  }

  #[test]
  fn defaults_missing_arguments_to_empty_object() {
    let mut accumulator = ToolCallAccumulator::default();
    accumulator.push(&[fragment(0, Some("call_1"), Some("now"), None)]);

    assert_eq!(parts(&accumulator.finish().unwrap()[0]).2, "{}");
  }

  #[test]
  fn rejects_call_without_id() {
    let mut accumulator = ToolCallAccumulator::default();
    accumulator.push(&[fragment(0, None, Some("calculator"), Some("{}"))]);

    assert!(accumulator.finish().is_err());
  }

  #[test]
  fn rejects_call_without_name() {
    let mut accumulator = ToolCallAccumulator::default();
    accumulator.push(&[fragment(0, Some("call_1"), None, Some("{}"))]);

    assert!(accumulator.finish().is_err());
  }

  #[test]
  fn ignores_fragments_without_function_payload() {
    let mut accumulator = ToolCallAccumulator::default();
    assert!(accumulator.is_empty());

    accumulator.push(&[ChatCompletionMessageToolCallChunk {
      index: 0,
      id: Some("call_1".to_owned()),
      r#type: Some(FunctionType::Function),
      function: None,
    }]);

    // The call is tracked, but it is unusable without a name.
    assert!(!accumulator.is_empty());
    assert!(accumulator.finish().is_err());
  }
}
