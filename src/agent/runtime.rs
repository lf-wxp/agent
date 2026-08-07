//! [`Agent`]: drives a model through a tool-calling loop while recording every step into
//! an [`ExecutionContext`].
//!
//! This differs from [`crate::llm::tool_loop::run`], which only returns the last choice:
//! here the whole transcript (user input, tool calls, tool results, final answer) is kept
//! as [`Event`]s, so a caller can inspect or persist what happened at each step. The model
//! call itself is not reimplemented, and neither is the round-budget behavior: both go
//! through the same shared client, request builder and [`disable_tools`] as `tool_loop`,
//! so a fix made there (retries, concurrency limits, degradation) does not have to be
//! ported by hand.
//!
//! [`Agent::run_stream`] / [`Agent::run_continuing_stream`] are the streaming counterparts
//! of [`Agent::run`] / [`Agent::run_continuing`]: same loop, round budget and event
//! recording, but assistant text is forwarded as it is produced (see
//! [`crate::llm::stream`] for the same idea without event recording).
//!
//! Multi-turn conversations are supported via [`Agent::run_continuing`], which seeds a
//! call with a prior turn's events instead of starting from an empty transcript. `Agent`
//! itself stays a stateless function of "prior events + new input": it does not own a
//! session/storage concept, that lives one layer up (see [`crate::api::session`] for how
//! the HTTP API persists history across requests). It does, however, cap how much of
//! that history it will actually send per call (see [`Self::with_max_history_tokens`] and
//! [`crate::agent::history::trim_to_budget`]), since an unbounded session could otherwise
//! grow past the model's context window.

use std::sync::Arc;

use async_openai::types::chat::{
  ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls,
  ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
  ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestToolMessageArgs,
  ChatCompletionRequestUserMessageArgs, ChatCompletionToolChoiceOption, ChatCompletionTools,
  CreateChatCompletionResponse, FinishReason, FunctionCall, ResponseFormat, ToolChoiceOptions,
};
use async_stream::stream;
use futures::{Stream, StreamExt, future::join_all};

use crate::{
  config,
  llm::{
    client::{DEFAULT_MAX_TOKENS, first_choice, request_builder},
    provider::Provider,
    retry::with_retry,
    schema::{
      FINAL_ANSWER_TOOL_NAME, final_answer_tool, native_schema_format, schema_instruction,
      strip_code_fence,
    },
    structured::{StructuredMode, TruncatedOutput},
    tool_calls::ToolCallAccumulator,
    tool_loop::{BudgetExhausted, disable_tools},
  },
  tools::ToolRegistry,
  util::truncate_chars,
};

use super::{
  context::ExecutionContext,
  event::{ContentItem, Event, ToolResultStatus},
};

/// Chars of tool arguments/results kept in debug logs, to avoid dumping large or
/// sensitive payloads (e.g. fetched web content, credentials echoed by a misbehaving
/// MCP server) into the log stream in full.
const TOOL_LOG_PREVIEW_CHARS: usize = 500;

/// Outcome of [`Agent::run`].
#[derive(Debug)]
pub struct AgentResult {
  pub output: String,
  pub context: ExecutionContext,
  /// `true` when the round budget ran out and this answer was produced with tools
  /// disabled, i.e. built from partial information. Mirrors
  /// [`crate::llm::tool_loop::Completion::budget_exhausted`].
  pub budget_exhausted: bool,
}

/// Outcome of [`Agent::run_structured`].
#[derive(Debug)]
pub struct StructuredAgentResult<T> {
  pub output: T,
  pub context: ExecutionContext,
  pub budget_exhausted: bool,
}

/// One increment produced by [`Agent::run_stream`] / [`Agent::run_continuing_stream`].
///
/// A caller interested only in the text (e.g. forwarding straight to an SSE response) can
/// match on [`Self::Token`] and ignore everything else until the stream ends; a caller that
/// needs to persist the turn (e.g. [`crate::api::session::SessionStore`]) reads
/// [`Self::Done`]'s `context` — the same fields as [`AgentResult`], just delivered as the
/// stream's last item instead of a return value.
#[derive(Debug)]
pub enum AgentStreamEvent {
  /// A chunk of assistant text, forwarded as soon as the model emits it.
  Token(String),
  /// The run finished. Always the last item; nothing follows it.
  Done {
    output: String,
    context: ExecutionContext,
    /// See [`AgentResult::budget_exhausted`].
    budget_exhausted: bool,
  },
}

/// Drives a model to a final answer, executing tool calls as it requests them.
///
/// A value rather than a free function: `provider`, `model`, `instructions` and the tool
/// set are fixed for the lifetime of the agent, while each [`Self::run`] call gets its
/// own fresh [`ExecutionContext`] so concurrent runs never share state.
pub struct Agent {
  provider: Provider,
  model: String,
  instructions: Option<String>,
  toolbox: Arc<ToolRegistry>,
  max_steps: u32,
  max_history_tokens: usize,
}

impl Agent {
  /// Rounds default to [`config::max_tool_rounds`], the same budget `tool_loop` uses, so
  /// the two do not silently drift apart; override with [`Self::with_max_steps`] when an
  /// agent needs a different budget than the rest of the process.
  ///
  /// `provider` is the tenant this agent talks to — its credentials and concurrency
  /// budget (see [`Provider::acquire`]) are used for every request the agent makes. Pass
  /// [`Provider::shared`] for the single-tenant default, or a tenant-specific
  /// [`Provider::new`] to isolate this agent's traffic (rate limit, API key, base URL)
  /// from other tenants running in the same process.
  pub fn new(
    provider: Provider,
    model: impl Into<String>,
    instructions: Option<impl Into<String>>,
    toolbox: Arc<ToolRegistry>,
  ) -> Self {
    Self {
      provider,
      model: model.into(),
      instructions: instructions.map(Into::into),
      toolbox,
      max_steps: config::max_tool_rounds() as u32,
      max_history_tokens: config::max_history_tokens(),
    }
  }

  /// Rounds of tool execution allowed before further rounds fall back to a
  /// tools-disabled request, instead of looping forever on a model that never stops
  /// calling tools. See [`Self::run`] / [`Self::run_structured`].
  pub fn with_max_steps(mut self, max_steps: u32) -> Self {
    self.max_steps = max_steps;
    self
  }

  /// Soft token budget for the `history` passed to [`Self::run_continuing`] — see
  /// [`crate::agent::history::trim_to_budget`] for exactly how it is enforced (whole
  /// turns dropped oldest-first, the most recent turn always kept). Defaults to
  /// [`config::max_history_tokens`]; override when a particular agent talks to a model
  /// with an unusually small or large context window.
  pub fn with_max_history_tokens(mut self, max_history_tokens: usize) -> Self {
    self.max_history_tokens = max_history_tokens;
    self
  }

  /// Whether the current round may still call the real tools, and whether reaching this
  /// point means the round budget just ran out (only meaningful when there are tools to
  /// spend a budget on).
  ///
  /// Extracted because [`Self::run`], [`Self::run_structured_via_tool_choice`] and
  /// [`Self::run_structured_via_response_format`] all make and log this same decision
  /// once per round; keeping one copy means the "budget spent -> warn" policy cannot
  /// drift between them.
  fn round_budget(&self, context: &ExecutionContext) -> (bool, bool) {
    let tools_allowed = tool_rounds_remaining(context.current_step, self.max_steps);
    let budget_exhausted = !tools_allowed && !self.toolbox.is_empty();
    if budget_exhausted {
      tracing::warn!(
        max_steps = self.max_steps,
        "tool round budget spent; forcing a final answer"
      );
    }
    (tools_allowed, budget_exhausted)
  }

  /// Run to a plain-text final answer, starting a brand-new conversation.
  ///
  /// Once the round budget ([`Self::with_max_steps`]) is spent, one last request goes out
  /// with tools disabled (see [`disable_tools`]) so a long task still returns an answer
  /// built from the results already gathered, instead of losing all of that work. That
  /// case is flagged through [`AgentResult::budget_exhausted`].
  ///
  /// Equivalent to [`Self::run_continuing`] with an empty history; use that instead to
  /// send a follow-up turn in an existing conversation.
  pub async fn run(&self, user_input: &str) -> anyhow::Result<AgentResult> {
    self.run_continuing(Vec::new(), user_input).await
  }

  /// Run to a plain-text final answer, continuing a conversation whose prior turns are
  /// `history`.
  ///
  /// `history` is normally a previous call's `AgentResult::context.events` — keep it on
  /// the caller's side (in memory, a database, an HTTP session store, ...) between calls
  /// and hand it back here for the next turn, so the model sees the full exchange so
  /// far. This is what makes multi-turn conversations possible without `Agent` itself
  /// owning any session/storage concept: it stays a pure function of "prior events + new
  /// input" (see [`crate::api::session`] for one way to manage that storage across HTTP
  /// requests).
  ///
  /// The round budget ([`Self::with_max_steps`]) resets every call — `current_step`
  /// starts back at zero — so a long conversation is never penalized for rounds already
  /// spent on earlier turns; only this turn's own tool calls count against it.
  pub async fn run_continuing(
    &self,
    history: Vec<Event>,
    user_input: &str,
  ) -> anyhow::Result<AgentResult> {
    let mut context = self.seed_context(history, user_input);

    loop {
      let (tools_allowed, budget_exhausted) = self.round_budget(&context);

      let messages = self.build_messages(&context)?;
      let mut builder = request_builder(
        &self.model,
        messages,
        DEFAULT_MAX_TOKENS,
        self.toolbox.definitions(),
      );
      if !tools_allowed {
        disable_tools(&mut builder, self.toolbox.definitions());
      }

      let response = with_retry(
        || async {
          let _permit = self.provider.acquire().await?;
          let response = self
            .provider
            .client()
            .chat()
            .create(builder.build()?)
            .await?;
          anyhow::Ok(response)
        },
        |_| true,
      )
      .await?;
      self.record_usage(&mut context, &response);
      let message = first_choice(response)?.message;

      // Some providers send `Some(vec![])` rather than `None` for "no tool calls".
      let Some(tool_calls) = message.tool_calls.filter(|calls| !calls.is_empty()) else {
        let content = message
          .content
          .ok_or_else(|| anyhow::anyhow!("No content in final response"))?;
        self.record_final_answer(&mut context, &content);
        return Ok(AgentResult {
          output: content,
          context,
          budget_exhausted,
        });
      };

      // The model ignored the disabled tools. Returning its message beats looping
      // forever; the caller still sees whatever content came back.
      if !tools_allowed {
        tracing::warn!("model requested tools after they were disabled");
        let content = message.content.unwrap_or_default();
        self.record_final_answer(&mut context, &content);
        return Ok(AgentResult {
          output: content,
          context,
          budget_exhausted: true,
        });
      }

      self.record_tool_calls(&mut context, &tool_calls);
      self.execute_tool_calls(&mut context, &tool_calls).await;
      context.increment_step();
    }
  }

  /// Streaming counterpart of [`Self::run`]: same tool-calling loop and round budget, but
  /// assistant text is forwarded to the caller as soon as the model emits it, instead of
  /// only once the whole run finishes. Equivalent to [`Self::run_continuing_stream`] with
  /// an empty history.
  pub fn run_stream<'a>(
    &'a self,
    user_input: &'a str,
  ) -> impl Stream<Item = anyhow::Result<AgentStreamEvent>> + 'a {
    self.run_continuing_stream(Vec::new(), user_input)
  }

  /// Streaming counterpart of [`Self::run_continuing`].
  ///
  /// Tool-call fragments are reassembled by [`ToolCallAccumulator`], the same way
  /// [`crate::llm::stream`] does it, but tools are executed through
  /// [`Self::execute_tool_calls`] rather than [`crate::tools::ToolRegistry::execute`], so
  /// every call sees the real [`ExecutionContext`] being built for this run — same as
  /// [`Self::run_continuing`] — instead of a throwaway one. Every round is recorded into
  /// that `ExecutionContext` exactly as the non-streaming path does (assistant text that
  /// accompanies a tool call is forwarded to the caller but, like [`Self::run_continuing`],
  /// not persisted into the transcript — only the call itself is), so a caller switching
  /// between the two gets the same transcript shape either way.
  ///
  /// Text is forwarded as [`AgentStreamEvent::Token`]s as soon as it arrives; the final
  /// [`AgentStreamEvent::Done`] carries the same fields as [`AgentResult`] and is always the
  /// last item, so a caller can stream tokens to a client while still waiting for `Done` to
  /// get the context to persist (e.g. into [`crate::api::session::SessionStore`]).
  pub fn run_continuing_stream<'a>(
    &'a self,
    history: Vec<Event>,
    user_input: &'a str,
  ) -> impl Stream<Item = anyhow::Result<AgentStreamEvent>> + 'a {
    stream! {
      let mut context = self.seed_context(history, user_input);

      loop {
        let (tools_allowed, budget_exhausted) = self.round_budget(&context);

        let messages = self.build_messages(&context)?;
        let mut builder = request_builder(
          &self.model,
          messages,
          DEFAULT_MAX_TOKENS,
          self.toolbox.definitions(),
        );
        if !tools_allowed {
          disable_tools(&mut builder, self.toolbox.definitions());
        }

        let mut chunks = with_retry(
          || async {
            let _permit = self.provider.acquire().await?;
            let stream = self
              .provider
              .client()
              .chat()
              .create_stream(builder.build()?)
              .await?;
            anyhow::Ok(stream)
          },
          |_| true,
        )
        .await?;

        let mut accumulator = ToolCallAccumulator::default();
        // Forwarded token by token as it arrives; also kept so the final answer (or the
        // fallback content once tools are disabled) can be recorded as one piece.
        let mut assistant_text = String::new();

        while let Some(chunk) = chunks.next().await {
          let chunk = match chunk {
            Ok(chunk) => chunk,
            // Abort instead of continuing: after a transport error the remaining
            // fragments would assemble into a truncated tool call.
            Err(err) => {
              yield Err(err.into());
              return;
            }
          };

          let Some(choice) = chunk.choices.first() else {
            continue;
          };

          if let Some(fragments) = &choice.delta.tool_calls {
            accumulator.push(fragments);
          }

          // Heartbeat / role-declaration chunks have empty delta.content; skip to avoid
          // flooding downstream with empty tokens.
          if let Some(text) = &choice.delta.content
            && !text.is_empty()
          {
            assistant_text.push_str(text);
            yield Ok(AgentStreamEvent::Token(text.clone()));
          }
        }

        // No tool calls means the model produced its final answer.
        if accumulator.is_empty() {
          self.record_final_answer(&mut context, &assistant_text);
          yield Ok(AgentStreamEvent::Done {
            output: assistant_text,
            context,
            budget_exhausted,
          });
          return;
        }

        // The model ignored the disabled tools. Ending here beats looping forever; the
        // caller has already seen whatever text came back as `Token`s.
        if !tools_allowed {
          tracing::warn!("model requested tools after they were disabled");
          self.record_final_answer(&mut context, &assistant_text);
          yield Ok(AgentStreamEvent::Done {
            output: assistant_text,
            context,
            budget_exhausted: true,
          });
          return;
        }

        let tool_calls = match accumulator.finish() {
          Ok(tool_calls) => tool_calls,
          Err(err) => {
            yield Err(err);
            return;
          }
        };

        self.record_tool_calls(&mut context, &tool_calls);
        self.execute_tool_calls(&mut context, &tool_calls).await;
        context.increment_step();
      }
    }
  }

  /// Run to a structured final answer of type `T`.
  ///
  /// Two routes, chosen by [`config::model_supports_tool_choice`]:
  /// - **forced tool_choice** (default): a synthetic `final_answer` tool carrying `T`'s
  ///   schema, with `tool_choice = required`, lets the model end the loop while still
  ///   calling the real tools along the way. Most reliable.
  /// - **response_format** (reasoning / "thinking" models that reject any `tool_choice`):
  ///   the real tools still run each round, but the final structure is constrained by
  ///   `response_format` (see [`crate::llm::structured::StructuredMode`]) instead of a
  ///   forced tool call — the model emits the JSON as message content, which is parsed
  ///   into `T`.
  ///
  /// Both record the full [`Event`] transcript and honor the round budget
  /// ([`StructuredAgentResult::budget_exhausted`]).
  pub async fn run_structured<T>(
    &self,
    user_input: &str,
  ) -> anyhow::Result<StructuredAgentResult<T>>
  where
    T: schemars::JsonSchema + serde::de::DeserializeOwned,
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

  /// Structured output by forcing `tool_choice = required` on a synthetic `final_answer`
  /// tool. See [`Self::run_structured`].
  async fn run_structured_via_tool_choice<T>(
    &self,
    user_input: &str,
  ) -> anyhow::Result<StructuredAgentResult<T>>
  where
    T: schemars::JsonSchema + serde::de::DeserializeOwned,
  {
    let mut context = ExecutionContext::new();
    self.record_user_input(&mut context, user_input);

    let final_answer = final_answer_tool::<T>()?;
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

      let response = with_retry(
        || async {
          let _permit = self.provider.acquire().await?;
          let response = self
            .provider
            .client()
            .chat()
            .create(builder.build()?)
            .await?;
          anyhow::Ok(response)
        },
        |_| true,
      )
      .await?;
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

  /// Structured output for models that reject `tool_choice`: real tools still run each
  /// round, but the final structure is constrained by `response_format` and the model
  /// emits the JSON as message content. See [`Self::run_structured`].
  async fn run_structured_via_response_format<T>(
    &self,
    user_input: &str,
  ) -> anyhow::Result<StructuredAgentResult<T>>
  where
    T: schemars::JsonSchema + serde::de::DeserializeOwned,
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

      let response = with_retry(
        || async {
          let _permit = self.provider.acquire().await?;
          let response = self
            .provider
            .client()
            .chat()
            .create(builder.build()?)
            .await?;
          anyhow::Ok(response)
        },
        |_| true,
      )
      .await?;
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
            crate::util::truncate_chars(&content, 512)
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
  ) -> anyhow::Result<Vec<ChatCompletionRequestMessage>> {
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

  /// Build the starting [`ExecutionContext`] for a call: a fresh execution id and step
  /// counter (see [`Self::run_continuing`] on why the round budget resets per call), with
  /// `history` trimmed to [`Self::with_max_history_tokens`] (see
  /// [`crate::agent::history::trim_to_budget`]) and spliced in as prior turns before the
  /// new user input is recorded.
  fn seed_context(&self, history: Vec<Event>, user_input: &str) -> ExecutionContext {
    let mut context = ExecutionContext::new();
    context.events = super::history::trim_to_budget(history, self.max_history_tokens);
    self.record_user_input(&mut context, user_input);
    context
  }

  fn record_user_input(&self, context: &mut ExecutionContext, user_input: &str) {
    context.add_event(Event::new(
      context.execution_id.clone(),
      "user",
      vec![ContentItem::Message {
        role: "user".to_owned(),
        content: user_input.to_owned(),
      }],
    ));
  }

  /// Record the assistant's final text and set [`ExecutionContext::final_result`].
  fn record_final_answer(&self, context: &mut ExecutionContext, content: &str) {
    context.add_event(Event::new(
      context.execution_id.clone(),
      "agent",
      vec![ContentItem::Message {
        role: "assistant".to_owned(),
        content: content.to_owned(),
      }],
    ));
    context.final_result = Some(content.to_owned());
  }

  fn record_usage(&self, context: &mut ExecutionContext, response: &CreateChatCompletionResponse) {
    if let Some(usage) = &response.usage {
      context.usage.add(
        usage.prompt_tokens,
        usage.completion_tokens,
        usage.total_tokens,
      );
    }
  }

  fn record_tool_calls(
    &self,
    context: &mut ExecutionContext,
    tool_calls: &[ChatCompletionMessageToolCalls],
  ) {
    let mut call_items = Vec::new();
    for tool_call in tool_calls {
      if let ChatCompletionMessageToolCalls::Function(function_call) = tool_call {
        let arguments: serde_json::Value = serde_json::from_str(&function_call.function.arguments)
          .unwrap_or(serde_json::Value::Null);
        call_items.push(ContentItem::ToolCall {
          tool_call_id: function_call.id.clone(),
          name: function_call.function.name.clone(),
          arguments,
        });
      }
    }
    context.add_event(Event::new(
      context.execution_id.clone(),
      "agent",
      call_items,
    ));
  }

  /// Execute every call in one model turn, feeding each tool the same context the caller
  /// will get back — unlike [`ToolRegistry::execute`] (used by [`crate::llm::tool_loop`]),
  /// which only ever sees a throwaway [`ExecutionContext::default`].
  ///
  /// Calls run concurrently rather than one after another: every [`Tool::execute`] only
  /// reads `context` (`&ExecutionContext`), so independent tool calls requested in the
  /// same turn (e.g. two `web_search` calls) do not have to pay for each other's network
  /// latency in sequence.
  async fn execute_tool_calls(
    &self,
    context: &mut ExecutionContext,
    tool_calls: &[ChatCompletionMessageToolCalls],
  ) {
    // Reborrowed as immutable for the duration of the concurrent calls below;
    // `context.add_event` takes a fresh `&mut` once every result is back.
    let context_ref: &ExecutionContext = context;

    let result_items = join_all(tool_calls.iter().filter_map(|tool_call| {
      let ChatCompletionMessageToolCalls::Function(function_call) = tool_call else {
        return None;
      };

      Some(async move {
        let function_name = &function_call.function.name;
        let arguments = &function_call.function.arguments;

        tracing::debug!(
          tool = %function_name,
          arguments = %truncate_chars(arguments, TOOL_LOG_PREVIEW_CHARS),
          "executing tool"
        );

        let (status, content) = match self.toolbox.get(function_name) {
          Some(tool) => match tool.execute(arguments, context_ref).await {
            Ok(result) => {
              tracing::debug!(
                tool = %function_name,
                result = %truncate_chars(&result, TOOL_LOG_PREVIEW_CHARS),
                "tool finished"
              );
              (ToolResultStatus::Success, result)
            }
            Err(err) => {
              let msg = format!("Tool execution error: {err}");
              tracing::warn!("{msg}");
              (ToolResultStatus::Error, msg)
            }
          },
          None => {
            let msg = format!("Tool execution error: unknown tool {function_name}");
            tracing::warn!("{msg}");
            (ToolResultStatus::Error, msg)
          }
        };

        ContentItem::ToolResult {
          tool_call_id: function_call.id.clone(),
          name: function_name.clone(),
          status,
          content,
        }
      })
    }))
    .await;

    context.add_event(Event::new(
      context.execution_id.clone(),
      "tool",
      result_items,
    ));
  }

  /// Replay the transcript recorded in `context` into the message shape the API expects.
  fn build_messages(
    &self,
    context: &ExecutionContext,
  ) -> anyhow::Result<Vec<ChatCompletionRequestMessage>> {
    let mut messages = Vec::new();

    if let Some(system) = &self.instructions {
      messages.push(
        ChatCompletionRequestSystemMessageArgs::default()
          .content(system.as_str())
          .build()?
          .into(),
      );
    }

    for event in &context.events {
      for item in &event.content {
        match item {
          ContentItem::Message { role, content } => {
            let message: ChatCompletionRequestMessage = if role == "user" {
              ChatCompletionRequestUserMessageArgs::default()
                .content(content.clone())
                .build()?
                .into()
            } else {
              ChatCompletionRequestAssistantMessageArgs::default()
                .content(content.clone())
                .build()?
                .into()
            };
            messages.push(message);
          }
          ContentItem::ToolCall {
            tool_call_id,
            name,
            arguments,
          } => {
            let tool_call =
              ChatCompletionMessageToolCalls::Function(ChatCompletionMessageToolCall {
                id: tool_call_id.clone(),
                function: FunctionCall {
                  name: name.clone(),
                  arguments: arguments.to_string(),
                },
              });

            if let Some(ChatCompletionRequestMessage::Assistant(last)) = messages.last_mut() {
              last.tool_calls.get_or_insert_with(Vec::new).push(tool_call);
            } else {
              messages.push(
                ChatCompletionRequestAssistantMessageArgs::default()
                  .tool_calls(vec![tool_call])
                  .build()?
                  .into(),
              );
            }
          }
          ContentItem::ToolResult {
            tool_call_id,
            content,
            ..
          } => {
            messages.push(
              ChatCompletionRequestToolMessageArgs::default()
                .tool_call_id(tool_call_id.clone())
                .content(content.clone())
                .build()?
                .into(),
            );
          }
        }
      }
    }

    Ok(messages)
  }
}

/// Whether another round may still call the real tools.
///
/// Mirrors `round < max_rounds` in [`crate::llm::tool_loop::run`]: once the budget is
/// spent, the caller must fall back to a tools-disabled (or `final_answer`-only) request
/// instead of looping forever on a model that never stops calling tools.
fn tool_rounds_remaining(current_step: u32, max_steps: u32) -> bool {
  current_step < max_steps
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

#[cfg(test)]
mod tests {
  use serde_json::json;

  use super::*;
  use crate::tools::calculator::{self, Calculator};

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
}
