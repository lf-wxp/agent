//! Demonstrates [`agent::callback::context_optimizer::ContextOptimizer`]: the
//! `BeforeLlmCallback` that keeps a request inside the model's context window.
//!
//! Every round, the agent flattens its transcript into a throwaway [`LlmRequest`] and
//! lets the hook chain edit it. `ContextOptimizer` runs a three-stage pipeline over that
//! copy, cheapest first, stopping as soon as the request fits:
//!
//! ```text
//! [ head ]  the task            pinned, never dropped
//! [ ~~~~ ]  the middle          1. compaction  2. summarization  3. eviction
//! [ tail ]  recent exchanges    kept verbatim
//! ```
//!
//! Crucially, all of it happens on the *copy*. `ExecutionContext::events` stays complete,
//! so the session on disk keeps the full history even as the prompt shrinks.
//!
//! This example is deliberately offline: it builds a transcript by hand and runs the
//! optimizer over it directly, so the stages are observable and the numbers are
//! reproducible. The last section then shows how to install it on a real agent.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example context_optimizer
//! ```

use std::sync::Arc;

use agent::{
  agent::{
    BeforeLlmCallback, ContentItem, Event, ExecutionContext, ToolResultStatus,
    llm_request::LlmRequest,
  },
  callback::context_optimizer::{Compaction, ContextOptimizer, compaction, tokens},
  config,
  llm::provider::Provider,
  telemetry,
  tools::ToolRegistry,
};

const SYSTEM_PROMPT: &str = "You are a research assistant. Read files as needed and cite \
                             what you found.";

/// Tight enough that a long run has to be trimmed, which is the point of the example.
const BUDGET: usize = 3_000;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
  telemetry::init()?;

  let context = long_run_transcript(24);
  println!(
    "A single long run: 1 task message + 24 rounds of file reads, \
     {} events in the transcript.\n",
    context.events.len()
  );

  report("no optimizer (what would be sent raw)", &context, None).await;

  // The default: compaction on, summarization off. Compaction rewrites spent tool
  // results *in place*, so the item count does not move even as the token count
  // collapses — nothing is lost, the old file dumps are just replaced by a note saying
  // the file was already read. Eviction is the backstop if that is not enough.
  report(
    "ContextOptimizer (default: compaction rewrites in place)",
    &context,
    Some(ContextOptimizer::new(BUDGET)),
  )
  .await;

  // With the free stage disabled, eviction alone has to deliver — it drops the middle
  // outright rather than rewriting anything.
  report(
    "eviction only (compaction disabled)",
    &context,
    Some(ContextOptimizer::new(BUDGET).without_compaction()),
  )
  .await;

  // Compaction only rewrites results for tools it was given. A caller's own tool — or one
  // reached over MCP — opts in by registering how to describe its spent output, which is
  // also the escape hatch for exempting a built-in whose output is not cheap to re-fetch.
  report(
    "custom compaction registry (read_file exempted)",
    &context,
    Some(
      ContextOptimizer::new(BUDGET).with_compaction(Compaction::empty(4).with_tool(
        "run_query",
        |arguments| {
          format!(
            "Query '{}' was already run.",
            compaction::argument(arguments, "sql")
          )
        },
      )),
    ),
  )
  .await;

  // `keep_head = 0` stops pinning the opening message, which suits a pure chat session
  // where the earliest question is rarely still relevant.
  //
  // It makes no difference on the transcript above, and cannot: a single long run has
  // exactly one user message, at the front, and the conversation sent to the model has
  // to open on a user message — so that one gets pinned whether asked for or not. The
  // parameter only bites once there are later user messages to fall back to, which is
  // what the chat-shaped transcript below has.
  let chat = chat_transcript(40);
  println!(
    "\nA chat session instead: {} events across 40 questions.\n",
    chat.events.len()
  );

  report(
    "chat, keep_head = 1 (opening question pinned)",
    &chat,
    Some(ContextOptimizer::new(BUDGET).without_compaction()),
  )
  .await;

  report(
    "chat, keep_head = 0 (oldest question may fall away)",
    &chat,
    Some(
      ContextOptimizer::new(BUDGET)
        .without_compaction()
        .with_keep_head(0),
    ),
  )
  .await;

  println!("{}", "=".repeat(66));
  println!(
    "In every run above the transcript came out the same size it went in — the optimizer\n\
     only ever edits the per-round copy, never the record that gets persisted."
  );

  // The floors outrank the budget: a request has to keep something recent to act on and
  // has to stay structurally valid, even when that means going out oversized. The
  // optimizer logs a warning in this case rather than letting the provider's own
  // context-length error be the first sign of trouble.
  report(
    "an impossible budget (structural floors win)",
    &context,
    Some(
      ContextOptimizer::new(50)
        .without_compaction()
        .with_keep_recent_min(6),
    ),
  )
  .await;

  installing_it_on_an_agent()?;

  Ok(())
}

/// Run one configuration over `context` and print what the model would receive.
///
/// The budget shown is the optimizer's own, so a configuration deliberately given an
/// impossible one is judged against it rather than against [`BUDGET`].
async fn report(label: &str, context: &ExecutionContext, optimizer: Option<ContextOptimizer>) {
  let mut request = LlmRequest::new(Some(SYSTEM_PROMPT.to_owned()), &context.events);
  let budget = optimizer
    .as_ref()
    .map_or(BUDGET, ContextOptimizer::max_tokens);

  if let Some(optimizer) = optimizer {
    optimizer.call(context, &mut request).await;
  }

  let total = tokens::count_request(&request);
  println!("{}", "=".repeat(66));
  println!("{label}");
  println!(
    "  sent to model : {} items, ~{total} tokens{}",
    request.contents.len(),
    if total > budget {
      format!("  (over the {budget} budget)")
    } else {
      format!("  (within the {budget} budget)")
    }
  );
  // A configuration aggressive enough to empty the conversation would be a bug, but
  // printing is not the place to find out about it by panicking.
  match request.contents.first() {
    Some(first) => println!("  first item    : {}", describe(first)),
    None => println!("  first item    : (the conversation was emptied)"),
  }
  // The oldest tool result, which is what compaction targets first: if it has been
  // rewritten, this is where it shows.
  if let Some(oldest_result) = request
    .contents
    .iter()
    .find(|item| matches!(item, ContentItem::ToolResult { .. }))
  {
    println!("  oldest result : {}", describe(oldest_result));
  }
  println!(
    "  transcript    : {} events, untouched",
    context.events.len()
  );
}

/// A one-line, truncated view of an item — enough to tell whether the task survived and
/// whether a tool result was rewritten.
fn describe(item: &ContentItem) -> String {
  let preview = |text: &str| text.chars().take(52).collect::<String>();
  match item {
    ContentItem::Message { role, content } => format!("[{role}] {}", preview(content)),
    ContentItem::ToolCall { name, .. } => format!("[tool call] {name}"),
    ContentItem::ToolResult { name, content, .. } => {
      format!("[tool result] {name} -> {}", preview(content))
    }
  }
}

/// One task message followed by many rounds of bulky `read_file` traffic: the shape of a
/// single long agentic run, where growth comes from tool output rather than from the user
/// asking more questions.
fn long_run_transcript(rounds: usize) -> ExecutionContext {
  let mut context = ExecutionContext::new();
  let id = context.execution_id.clone();

  context.add_event(Event::new(
    id.clone(),
    "user",
    vec![ContentItem::Message {
      role: "user".to_owned(),
      content: "Audit every module under src/ and summarize what each one does.".to_owned(),
    }],
  ));

  for i in 0..rounds {
    let path = format!("src/module_{i}.rs");
    context.add_event(Event::new(
      id.clone(),
      "agent",
      vec![ContentItem::ToolCall {
        tool_call_id: format!("call_{i}"),
        name: "read_file".to_owned(),
        arguments: serde_json::json!({ "file_path": path }),
      }],
    ));
    context.add_event(Event::new(
      id.clone(),
      "tool",
      vec![ContentItem::ToolResult {
        tool_call_id: format!("call_{i}"),
        name: "read_file".to_owned(),
        status: ToolResultStatus::Success,
        content: format!("// {path}\n{}", "pub fn work() { /* ... */ }\n".repeat(40)),
      }],
    ));
  }

  context
}

/// Several questions, each with a little work behind it: the shape of a chat session,
/// where growth comes from the user asking more rather than from one task snowballing.
fn chat_transcript(turns: usize) -> ExecutionContext {
  let mut context = ExecutionContext::new();
  let id = context.execution_id.clone();

  for turn in 0..turns {
    context.add_event(Event::new(
      id.clone(),
      "user",
      vec![ContentItem::Message {
        role: "user".to_owned(),
        content: format!("Question {turn}: what does module_{turn} do?"),
      }],
    ));
    context.add_event(Event::new(
      id.clone(),
      "agent",
      vec![ContentItem::Message {
        role: "assistant".to_owned(),
        content: format!("Answer {turn}. {}", "Details follow. ".repeat(30)),
      }],
    ));
  }

  context
}

/// How to actually wire it up. The budget belongs to the callback rather than to the
/// agent, so changing it means installing a differently-configured instance — which is
/// what [`agent::Agent::clear_before_llm_callbacks`] is for.
fn installing_it_on_an_agent() -> anyhow::Result<()> {
  let toolbox = Arc::new(ToolRegistry::empty());

  // An agent built with `new` already has a `ContextOptimizer` carrying
  // `config::max_history_tokens()`, so the common case needs no setup at all.
  let _default = agent::Agent::new(
    Provider::shared().clone(),
    config::model(),
    Some(SYSTEM_PROMPT),
    Arc::clone(&toolbox),
  );

  // A tighter budget, plus summarization: when compaction alone is not enough, the middle
  // is replaced by an LLM-written recap instead of being dropped. That costs one extra
  // model call per invocation, which is why it is opt-in — and it goes through the
  // agent's own `Provider`, so it honors the same credentials, concurrency budget and
  // retry policy as the run it belongs to.
  let _tuned = agent::Agent::new(
    Provider::shared().clone(),
    config::model(),
    Some(SYSTEM_PROMPT),
    toolbox,
  )
  .clear_before_llm_callbacks()
  .with_before_llm_callback(Arc::new(
    ContextOptimizer::new(8_000)
      .with_keep_head(1)
      .with_keep_recent_min(6)
      .with_summarization(Provider::shared().clone(), config::model(), 5),
  ));

  println!(
    "\nBuilt two agents: one with the default optimizer, one with a tuned budget plus\n\
     summarization. Neither is run here — this example makes no API calls."
  );

  Ok(())
}
