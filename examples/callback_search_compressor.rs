//! Demonstrates [`agent::callback::search_compressor::SearchCompressorCallback`]: an
//! `AfterToolCallback` that rewrites a bulky `web_search` result before it is recorded,
//! keeping only the passages that actually answer the query (fixed-length chunking, then
//! vector search against the query the model itself passed to the tool).
//!
//! Why it matters: a raw search result is mostly padding, and every character of it is
//! re-sent to the model on every subsequent round of the loop. Compressing it once, at
//! the point it enters the transcript, shrinks all of those rounds at once.
//!
//! The same question is asked twice — once with a plain agent, once with the callback
//! registered — and the two transcripts are compared, so the saving is measured rather
//! than asserted. Two runs means two sets of API calls; that is the price of the
//! comparison.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example callback_search_compressor
//! ```

use std::sync::Arc;

use agent::{
  Agent, AgentResult,
  agent::ContentItem,
  callback::search_compressor::SearchCompressorCallback,
  config,
  llm::provider::Provider,
  telemetry,
  tools::{ToolRegistry, web_search},
};
use tiktoken_rs::{CoreBPE, cl100k_base};

const SYSTEM_PROMPT: &str = "You are a general-purpose assistant. Search the web for \
                             anything you are not certain about, and cite the source URL.";

/// Broad enough that the search returns far more text than the answer needs — which is
/// exactly the case the callback exists for.
const QUESTION: &str = "What did the most recent stable Rust release change, and when was \
                        it released?";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
  telemetry::init()?;

  let toolbox = Arc::new(ToolRegistry::select(&[web_search::NAME.to_owned()])?);

  let plain = Agent::new(
    Provider::shared().clone(),
    config::model(),
    Some(SYSTEM_PROMPT),
    Arc::clone(&toolbox),
  );
  // Same agent in every other respect; the callback is the only difference between the
  // two runs, so any difference in the numbers below comes from it alone.
  let compressing = Agent::new(
    Provider::shared().clone(),
    config::model(),
    Some(SYSTEM_PROMPT),
    toolbox,
  )
  .with_after_tool_callback(Arc::new(SearchCompressorCallback));

  let baseline = plain.run(QUESTION).await?;
  tracing::info!("Answer (no callback): {}", baseline.output);

  let compressed = compressing.run(QUESTION).await?;
  tracing::info!("Answer (with callback): {}", compressed.output);

  // The model we run is not necessarily tokenized by cl100k_base, so treat these token
  // counts as an estimate for comparison, not as a billing figure.
  let encoder = cl100k_base()?;
  let before = search_result_size(&baseline, &encoder);
  let after = search_result_size(&compressed, &encoder);

  println!("\n{}", "=".repeat(60));
  println!("web_search output recorded in the transcript");
  println!(
    "  without callback: {} chars, ~{} tokens",
    before.0, before.1
  );
  println!("  with callback:    {} chars, ~{} tokens", after.0, after.1);
  if before.1 > 0 {
    println!(
      "  saved:            {:.1}% of search tokens",
      (1.0 - after.1 as f64 / before.1 as f64) * 100.0
    );
  }
  println!(
    "\nwhole-run usage (all rounds, prompt + completion)\n  without callback: {:?}\n  with callback:    {:?}",
    baseline.context.usage, compressed.context.usage
  );

  Ok(())
}

/// Characters and estimated tokens of every `web_search` result in the transcript — i.e.
/// what the model was actually given to read, after the callback (if any) had its say.
fn search_result_size(result: &AgentResult, encoder: &CoreBPE) -> (usize, usize) {
  result
    .context
    .events
    .iter()
    .flat_map(|event| &event.content)
    .filter_map(|item| match item {
      ContentItem::ToolResult { name, content, .. } if name == web_search::NAME => Some(content),
      _ => None,
    })
    .fold((0, 0), |(chars, tokens), content| {
      (
        chars + content.len(),
        tokens + encoder.encode_with_special_tokens(content).len(),
      )
    })
}
