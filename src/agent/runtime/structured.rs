//! Structured output for [`super::Agent`]: [`Agent::run_structured`] / [`Agent::run_structured_raw`].
//!
//! Two routes, chosen by [`config::model_supports_tool_choice`]:
//! - **forced tool_choice** (default): a synthetic `final_answer` tool carrying the
//!   target schema, with `tool_choice = required`, lets the model end the loop while
//!   still calling the real tools along the way. Most reliable.
//! - **response_format** (reasoning / "thinking" models that reject any `tool_choice`):
//!   the real tools still run each round, but the final structure is constrained by
//!   `response_format` (see [`crate::llm::structured::StructuredMode`]) instead of a
//!   forced tool call — the model emits the JSON as message content, which is parsed
//!   into the target type.
//!
//! Each route exists in two flavors that differ only in *where the schema comes from*:
//! a compile-time Rust type (`run_structured*`, for library callers) or a
//! [`serde_json::Value`] supplied at runtime (`run_structured_raw*`, for a caller with no
//! compile-time type, e.g. a web front-end forwarding a JSON Schema from a request).
//! Since [`Value`] itself implements
//! [`DeserializeOwned`], both flavors of a route share one generic core
//! ([`Agent::run_final_answer_loop`] / [`Agent::run_response_format_loop`]) that is
//! generic over the parsed output type `T`; the `_raw` variant is just a thin wrapper
//! that builds the schema-as-data inputs and calls the same core a compile-time-typed
//! caller would reach through `schemars::JsonSchema`.

use async_openai::types::chat::{
  ChatCompletionMessageToolCalls, ChatCompletionToolChoiceOption, ChatCompletionTools,
  FinishReason, ResponseFormat, ToolChoiceOptions,
};
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::{
  config,
  llm::{
    client::{DEFAULT_MAX_TOKENS, first_choice, request_builder},
    schema::{
      FINAL_ANSWER_TOOL_NAME, final_answer_tool, final_answer_tool_from_schema,
      native_schema_format, native_schema_format_from_value, schema_instruction,
      schema_instruction_from_value, strip_code_fence, validate_schema_name,
    },
    structured::{StructuredMode, TruncatedOutput},
    tool_loop::{BudgetExhausted, disable_tools},
  },
  util::truncate_chars,
};

use super::{
  super::{
    context::ExecutionContext,
    event::{ContentItem, Event, ToolResultStatus},
  },
  Agent,
};

/// Outcome of [`Agent::run_structured`] / [`Agent::run_structured_raw`].
#[derive(Debug)]
pub struct StructuredAgentResult<T> {
  pub output: T,
  pub context: ExecutionContext,
  pub budget_exhausted: bool,
}

impl Agent {
  /// Run to a structured final answer of type `T`. See the module docs for the two
  /// routes this dispatches between.
  ///
  /// Both record the full [`Event`] transcript and honor the round budget
  /// ([`StructuredAgentResult::budget_exhausted`]).
  pub async fn run_structured<T>(
    &self,
    user_input: &str,
  ) -> anyhow::Result<StructuredAgentResult<T>>
  where
    T: schemars::JsonSchema + DeserializeOwned,
  {
    if config::model_supports_tool_choice(&self.model) {
      self.run_structured_via_tool_choice(user_input).await
    } else {
      tracing::debug!(
        model = %self.model,
        "model rejects tool_choice; using response_format for structured output"
      );
      self.run_structured_via_response_format(user_input).await
    }
  }

  /// Run to a structured final answer given a JSON Schema `Value` at runtime, instead of a
  /// compile-time Rust type — the counterpart to [`Self::run_structured`] for callers that do
  /// not have (or want) a Rust type for the answer, e.g. the HTTP API (see
  /// [`crate::api::dto::StructuredSchemaRequest`]), where the schema arrives as part of the
  /// request body.
  ///
  /// `name` is OpenAI's naming constraint on `function.name` / `response_format.json_schema.name`
  /// (1-64 characters, `[a-zA-Z0-9_-]`), checked with [`validate_schema_name`] up front — a
  /// schema-as-data caller has no compile-time type to derive a valid one from the way
  /// [`Self::run_structured`] does via `schema_name::<T>()`, so unlike that path, an invalid
  /// `name` here is a caller mistake to be rejected immediately rather than a condition this
  /// crate could ever hit on its own.
  ///
  /// Otherwise behaves exactly like [`Self::run_structured`]: same two routes chosen by
  /// [`config::model_supports_tool_choice`], same round budget and [`Event`] recording; the
  /// only difference is the final step returns the parsed [`Value`] as-is instead of
  /// deserializing into `T`.
  pub async fn run_structured_raw(
    &self,
    user_input: &str,
    name: &str,
    schema: Value,
  ) -> anyhow::Result<StructuredAgentResult<Value>> {
    validate_schema_name(name)?;

    if config::model_supports_tool_choice(&self.model) {
      self
        .run_structured_raw_via_tool_choice(user_input, name, schema)
        .await
    } else {
      tracing::debug!(
        model = %self.model,
        "model rejects tool_choice; using response_format for structured output"
      );
      self
        .run_structured_raw_via_response_format(user_input, schema)
        .await
    }
  }

  /// Structured output by forcing `tool_choice = required` on a synthetic `final_answer`
  /// tool built from `T`'s schema. Thin wrapper around [`Self::run_final_answer_loop`].
  async fn run_structured_via_tool_choice<T>(
    &self,
    user_input: &str,
  ) -> anyhow::Result<StructuredAgentResult<T>>
  where
    T: schemars::JsonSchema + DeserializeOwned,
  {
    self
      .run_final_answer_loop(user_input, final_answer_tool::<T>()?)
      .await
  }

  /// Schema-as-data counterpart of [`Self::run_structured_via_tool_choice`], used by
  /// [`Self::run_structured_raw`].
  async fn run_structured_raw_via_tool_choice(
    &self,
    user_input: &str,
    name: &str,
    schema: Value,
  ) -> anyhow::Result<StructuredAgentResult<Value>> {
    self
      .run_final_answer_loop(user_input, final_answer_tool_from_schema(name, schema)?)
      .await
  }

  /// Shared core of the `tool_choice` route: force the model to end the loop by calling a
  /// synthetic `final_answer` tool carrying the schema `T` must satisfy. `T` only needs
  /// [`DeserializeOwned`] here (not `JsonSchema`) — the schema was already turned into a
  /// tool definition by the caller (either from a Rust type or from a runtime `Value`),
  /// so this loop just needs to parse whatever comes back.
  async fn run_final_answer_loop<T: DeserializeOwned>(
    &self,
    user_input: &str,
    final_answer: ChatCompletionTools,
  ) -> anyhow::Result<StructuredAgentResult<T>> {
    let mut context = ExecutionContext::new();
    self.record_user_input(&mut context, user_input);

    let mut tool_definitions = self.toolbox.definitions().to_vec();
    tool_definitions.push(final_answer.clone());
    let final_answer_only = std::slice::from_ref(&final_answer);

    loop {
      let (tools_allowed, budget_exhausted) = self.round_budget(&context);

      let definitions: &[ChatCompletionTools] = if tools_allowed {
        &tool_definitions
      } else {
        final_answer_only
      };

      let messages = self.build_messages(&context)?;
      let mut builder = request_builder(&self.model, messages, DEFAULT_MAX_TOKENS, definitions);
      builder.tool_choice(ChatCompletionToolChoiceOption::Mode(
        ToolChoiceOptions::Required,
      ));

      let response = self.complete(&builder).await?;
      self.record_usage(&mut context, &response);
      let choice = first_choice(response)?;

      // Truncation is deterministic — a retry burns the same budget and truncates again —
      // so surface it as [`TruncatedOutput`] up front (same treatment as `llm::structured`),
      // rather than letting it degrade into a cryptic empty-tool-call error below.
      if choice.finish_reason == Some(FinishReason::Length) {
        return Err(
          TruncatedOutput {
            limit: DEFAULT_MAX_TOKENS,
          }
          .into(),
        );
      }

      let message = choice.message;

      let tool_calls = message.tool_calls.ok_or_else(|| {
        tag_budget(
          anyhow::anyhow!("Model returned no tool call despite tool_choice = required"),
          budget_exhausted,
        )
      })?;

      self.record_tool_calls(&mut context, &tool_calls);

      let final_call = tool_calls.iter().find_map(|tool_call| match tool_call {
        ChatCompletionMessageToolCalls::Function(f)
          if f.function.name == FINAL_ANSWER_TOOL_NAME =>
        {
          Some(f)
        }
        _ => None,
      });

      if let Some(final_call) = final_call {
        let raw_arguments = final_call.function.arguments.clone();
        let parsed: T = serde_json::from_str(&raw_arguments).map_err(|err| {
          tag_budget(
            anyhow::anyhow!("failed to parse `{FINAL_ANSWER_TOOL_NAME}` arguments: {err}"),
            budget_exhausted,
          )
        })?;

        self.record_final_structured(&mut context, final_call.id.clone(), raw_arguments);

        return Ok(StructuredAgentResult {
          output: parsed,
          context,
          budget_exhausted,
        });
      }

      // Once the budget is spent, `final_answer` is the only tool on offer; anything
      // else here cannot be recovered from without spending another round.
      if !tools_allowed {
        anyhow::bail!(tag_budget(
          anyhow::anyhow!(
            "model did not call `{FINAL_ANSWER_TOOL_NAME}` after the round budget was spent"
          ),
          true
        ));
      }

      self.execute_tool_calls(&mut context, &tool_calls).await;
      context.increment_step();
    }
  }

  /// Structured output for models that reject `tool_choice`: builds `T`'s schema hint /
  /// response format, then delegates to [`Self::run_response_format_loop`].
  async fn run_structured_via_response_format<T>(
    &self,
    user_input: &str,
  ) -> anyhow::Result<StructuredAgentResult<T>>
  where
    T: schemars::JsonSchema + DeserializeOwned,
  {
    let mode = StructuredMode::for_model(&self.model);
    // Under `json_object` the schema is only *guided* by a prompt, so inject it as an
    // extra system instruction; native `json_schema` is server-enforced and needs none.
    let schema_hint = match mode {
      StructuredMode::JsonObject => Some(schema_instruction::<T>()),
      StructuredMode::NativeSchema => None,
    };
    let response_format = match mode {
      StructuredMode::NativeSchema => native_schema_format::<T>(),
      StructuredMode::JsonObject => ResponseFormat::JsonObject,
    };

    self
      .run_response_format_loop(user_input, mode, schema_hint, response_format)
      .await
  }

  /// Schema-as-data counterpart of [`Self::run_structured_via_response_format`], used by
  /// [`Self::run_structured_raw`].
  async fn run_structured_raw_via_response_format(
    &self,
    user_input: &str,
    schema: Value,
  ) -> anyhow::Result<StructuredAgentResult<Value>> {
    let mode = StructuredMode::for_model(&self.model);
    let schema_hint = match mode {
      StructuredMode::JsonObject => Some(schema_instruction_from_value(&schema)),
      StructuredMode::NativeSchema => None,
    };
    let response_format = match mode {
      // `name` only appears in the tool_choice route's `final_answer` description; here it is
      // never surfaced to the model, so a fixed placeholder is fine.
      StructuredMode::NativeSchema => native_schema_format_from_value("StructuredAnswer", schema),
      StructuredMode::JsonObject => ResponseFormat::JsonObject,
    };

    self
      .run_response_format_loop(user_input, mode, schema_hint, response_format)
      .await
  }

  /// Shared core of the `response_format` route: the real tools still run each round, but
  /// the final structure is constrained by `response_format` and the model emits the JSON
  /// as message content, parsed into `T`. `T` only needs [`DeserializeOwned`] — the schema
  /// itself is already baked into `response_format` / `schema_hint` by the caller.
  async fn run_response_format_loop<T: DeserializeOwned>(
    &self,
    user_input: &str,
    mode: StructuredMode,
    schema_hint: Option<String>,
    response_format: ResponseFormat,
  ) -> anyhow::Result<StructuredAgentResult<T>> {
    let mut context = ExecutionContext::new();
    self.record_user_input(&mut context, user_input);

    loop {
      let (tools_allowed, budget_exhausted) = self.round_budget(&context);

      // Constrain the structure on **every** round, not just once tools look done: this
      // model picks when to stop calling tools and answer, and if that round were
      // unconstrained it would reply in prose and fail to parse. On tool-calling rounds
      // `content` is empty anyway, so the constraint is harmless there.
      let messages = match &schema_hint {
        Some(hint) => self.messages_with_schema_hint(&context, hint)?,
        None => self.build_messages(&context)?,
      };
      // Reasoning models keep `max_tokens` covering chain-of-thought + answer, so the
      // structured budget applies here too (see `structured::JSON_OBJECT_MAX_TOKENS`).
      let mut builder = request_builder(
        &self.model,
        messages,
        mode.max_tokens(),
        self.toolbox.definitions(),
      );
      builder.response_format(response_format.clone());
      if !tools_allowed {
        disable_tools(&mut builder, self.toolbox.definitions());
      }

      let response = self.complete(&builder).await?;
      self.record_usage(&mut context, &response);
      let choice = first_choice(response)?;

      if choice.finish_reason == Some(FinishReason::Length) {
        return Err(
          TruncatedOutput {
            limit: mode.max_tokens(),
          }
          .into(),
        );
      }

      let message = choice.message;

      // The model may still call the real tools before answering.
      if let Some(tool_calls) = message.tool_calls.filter(|calls| !calls.is_empty()) {
        if !tools_allowed {
          // Budget spent but the model wants more tools: take whatever content it gave,
          // rather than looping forever.
          tracing::warn!("model requested tools after they were disabled");
        } else {
          self.record_tool_calls(&mut context, &tool_calls);
          self.execute_tool_calls(&mut context, &tool_calls).await;
          context.increment_step();
          continue;
        }
      }

      let content = message
        .content
        .filter(|c| !c.trim().is_empty())
        .ok_or_else(|| {
          tag_budget(
            anyhow::anyhow!("empty content while expecting a structured answer"),
            budget_exhausted,
          )
        })?;

      let parsed: T = serde_json::from_str(strip_code_fence(&content)).map_err(|err| {
        tag_budget(
          anyhow::anyhow!(
            "failed to parse structured content: {err}; raw: {}",
            truncate_chars(&content, 512)
          ),
          budget_exhausted,
        )
      })?;

      self.record_final_answer(&mut context, &content);

      return Ok(StructuredAgentResult {
        output: parsed,
        context,
        budget_exhausted,
      });
    }
  }

  /// Like [`Self::build_messages`] but appends a one-off system message carrying the schema
  /// hint, used by the `json_object` response-format route.
  fn messages_with_schema_hint(
    &self,
    context: &ExecutionContext,
    hint: &str,
  ) -> anyhow::Result<Vec<async_openai::types::chat::ChatCompletionRequestMessage>> {
    use async_openai::types::chat::ChatCompletionRequestSystemMessageArgs;

    let mut messages = self.build_messages(context)?;
    messages.push(
      ChatCompletionRequestSystemMessageArgs::default()
        .content(hint)
        .build()?
        .into(),
    );
    Ok(messages)
  }

  /// Record a structured final answer produced through the `final_answer` tool.
  fn record_final_structured(
    &self,
    context: &mut ExecutionContext,
    tool_call_id: String,
    raw_arguments: String,
  ) {
    context.add_event(Event::new(
      context.execution_id.clone(),
      "tool",
      vec![ContentItem::ToolResult {
        tool_call_id,
        name: FINAL_ANSWER_TOOL_NAME.to_owned(),
        status: ToolResultStatus::Success,
        content: raw_arguments.clone(),
      }],
    ));
    context.final_result = Some(raw_arguments);
  }
}

/// Mark a failure as caused by a budget-exhausted attempt, so retries can be skipped —
/// same convention `llm::structured` uses around [`BudgetExhausted`].
fn tag_budget(err: anyhow::Error, budget_exhausted: bool) -> anyhow::Error {
  if budget_exhausted {
    err.context(BudgetExhausted)
  } else {
    err
  }
}
