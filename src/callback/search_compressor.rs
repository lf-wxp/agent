use std::collections::HashSet;

use serde_json::Value;

use crate::{
  agent::{ContentItem, ExecutionContext, ToolResultStatus, callback::AfterToolCallback},
  knowledge_base::{chunk::fixed_length_chunking, search::vector_search},
  tools::web_search,
};

/// Results smaller than this are left alone: the embedding round trip would cost more
/// than the context it saves. Counted in `char`s like [`fixed_length_chunking`], so the
/// trigger point does not shift with the script the page happens to be written in — as a
/// byte count it would fire ~3x sooner on Chinese text than on English.
const COMPRESS_THRESHOLD: usize = 2000;
const CHUNK_SIZE: usize = 500;
const CHUNK_OVERLAP: usize = 50;
const TOP_K: usize = 3;

pub struct SearchCompressorCallback;

#[async_trait::async_trait]
impl AfterToolCallback for SearchCompressorCallback {
  async fn call(
    &self,
    context: &ExecutionContext,
    tool_call_id: &str,
    tool_name: &str,
    status: ToolResultStatus,
    content: &str,
  ) -> Option<(ToolResultStatus, String)> {
    if tool_name != web_search::NAME || status != ToolResultStatus::Success {
      return None;
    }

    let content_chars = content.chars().count();
    if content_chars < COMPRESS_THRESHOLD {
      return None;
    }

    let query = extract_query(context, tool_call_id)?;

    let chunks = fixed_length_chunking(content, CHUNK_SIZE, CHUNK_OVERLAP);

    if chunks.is_empty() {
      return None;
    }

    tracing::info!(
      "🔍 Compressing web_search result: {} chars → chunking into {} pieces...",
      content_chars,
      chunks.len(),
    );

    match vector_search(&query, &chunks, TOP_K).await {
      Ok(hits) => {
        // Kept in the order they appear in the page, not in similarity order:
        // `vector_search` ranks by relevance, but the model reads the survivors as
        // continuous prose, and passages that jump backwards through a document read as
        // a non-sequitur.
        let selected: HashSet<&str> = hits.iter().map(|hit| hit.text.as_str()).collect();
        let compressed = chunks
          .iter()
          .filter(|chunk| selected.contains(chunk.as_str()))
          .cloned()
          .collect::<Vec<_>>()
          .join("\n\n");
        tracing::info!(
          "✅ Compression complete: {} chars → {} chars (top {} of {} chunks)",
          content_chars,
          compressed.chars().count(),
          hits.len(),
          chunks.len(),
        );
        Some((status, compressed))
      }
      Err(err) => {
        tracing::warn!("Search compression skipped: {err}");
        None
      }
    }
  }
}

fn extract_query(context: &ExecutionContext, tool_call_id: &str) -> Option<String> {
  context
    .events
    .iter()
    .flat_map(|event| &event.content)
    .find_map(|item| match item {
      ContentItem::ToolCall {
        tool_call_id: id,
        name,
        arguments,
      } if id == tool_call_id && name == web_search::NAME => arguments
        .get("query")
        .and_then(Value::as_str)
        .map(str::to_owned),
      _ => None,
    })
}

#[cfg(test)]
mod tests {
  use serde_json::json;

  use super::*;
  use crate::agent::Event;

  const CALL_ID: &str = "call-1";

  /// A search-result string long enough to clear [`COMPRESS_THRESHOLD`], repeated
  /// so it also chunks into more than one non-empty piece.
  fn long_content() -> String {
    "word ".repeat(COMPRESS_THRESHOLD)
  }

  /// A context whose one event records a `web_search` tool call for `CALL_ID` with the
  /// given `query`, matching what [`extract_query`] looks for.
  fn context_with_query(query: &str) -> ExecutionContext {
    let mut context = ExecutionContext::new();
    context.add_event(Event::new(
      context.execution_id.clone(),
      "assistant",
      vec![ContentItem::ToolCall {
        tool_call_id: CALL_ID.to_owned(),
        name: web_search::NAME.to_owned(),
        arguments: json!({ "query": query }),
      }],
    ));
    context
  }

  #[test]
  fn extract_query_finds_the_query_of_the_matching_tool_call() {
    let context = context_with_query("rust async runtimes");
    assert_eq!(
      extract_query(&context, CALL_ID),
      Some("rust async runtimes".to_owned())
    );
  }

  #[test]
  fn extract_query_ignores_a_tool_call_with_a_different_id() {
    let context = context_with_query("rust async runtimes");
    assert_eq!(extract_query(&context, "some-other-call"), None);
  }

  #[test]
  fn extract_query_ignores_a_matching_id_for_a_different_tool() {
    let mut context = ExecutionContext::new();
    context.add_event(Event::new(
      context.execution_id.clone(),
      "assistant",
      vec![ContentItem::ToolCall {
        tool_call_id: CALL_ID.to_owned(),
        name: "calculator".to_owned(),
        arguments: json!({ "query": "irrelevant" }),
      }],
    ));
    assert_eq!(extract_query(&context, CALL_ID), None);
  }

  #[test]
  fn extract_query_returns_none_when_the_argument_is_missing_or_not_a_string() {
    let mut context = ExecutionContext::new();
    context.add_event(Event::new(
      context.execution_id.clone(),
      "assistant",
      vec![ContentItem::ToolCall {
        tool_call_id: CALL_ID.to_owned(),
        name: web_search::NAME.to_owned(),
        arguments: json!({ "query": 42 }),
      }],
    ));
    assert_eq!(extract_query(&context, CALL_ID), None);
  }

  #[test]
  fn extract_query_returns_none_for_an_empty_context() {
    let context = ExecutionContext::new();
    assert_eq!(extract_query(&context, CALL_ID), None);
  }

  #[tokio::test]
  async fn ignores_a_result_from_a_tool_other_than_web_search() {
    let context = context_with_query("anything");
    let result = SearchCompressorCallback
      .call(
        &context,
        CALL_ID,
        "calculator",
        ToolResultStatus::Success,
        &long_content(),
      )
      .await;
    assert!(result.is_none());
  }

  #[tokio::test]
  async fn ignores_a_failed_search() {
    let context = context_with_query("anything");
    let result = SearchCompressorCallback
      .call(
        &context,
        CALL_ID,
        web_search::NAME,
        ToolResultStatus::Error,
        &long_content(),
      )
      .await;
    assert!(result.is_none());
  }

  #[tokio::test]
  async fn ignores_content_below_the_compression_threshold() {
    let context = context_with_query("anything");
    let short_content = "a short result, nowhere near the threshold";
    let result = SearchCompressorCallback
      .call(
        &context,
        CALL_ID,
        web_search::NAME,
        ToolResultStatus::Success,
        short_content,
      )
      .await;
    assert!(result.is_none());
  }

  #[tokio::test]
  async fn ignores_content_of_only_whitespace_even_above_the_char_threshold() {
    // Every chunk trims to empty, so `fixed_length_chunking` yields no pieces at all —
    // must bail out before ever calling `vector_search`.
    let context = context_with_query("anything");
    let whitespace_content = " ".repeat(COMPRESS_THRESHOLD + 100);
    let result = SearchCompressorCallback
      .call(
        &context,
        CALL_ID,
        web_search::NAME,
        ToolResultStatus::Success,
        &whitespace_content,
      )
      .await;
    assert!(result.is_none());
  }

  #[tokio::test]
  async fn returns_none_when_no_matching_tool_call_is_recorded_in_context() {
    // Content clears the threshold, but the context has no `ToolCall` event for
    // `CALL_ID`, so `extract_query` fails and the call bails out via `?` before ever
    // reaching `vector_search`.
    let context = ExecutionContext::new();
    let result = SearchCompressorCallback
      .call(
        &context,
        CALL_ID,
        web_search::NAME,
        ToolResultStatus::Success,
        &long_content(),
      )
      .await;
    assert!(result.is_none());
  }

  #[tokio::test]
  async fn skips_compression_when_the_embedder_is_not_configured() {
    // Without `EMBED_MODEL` set, `vector_search` fails fast (see
    // `knowledge_base::search::vector_search`) without making any network call; the
    // callback must treat that as "leave the result alone" rather than propagating the
    // error.
    //
    // `EMBED_MODEL` is process-global state, so this saves and restores whatever was
    // there before rather than removing it outright: `cargo test` runs tests from the
    // same binary concurrently on multiple threads, and permanently clearing a real
    // value here (e.g. one loaded from `.env` by `telemetry::init`) would make any other
    // test that happens to run afterwards in this process silently lose it too.
    let previous = std::env::var("EMBED_MODEL").ok();
    unsafe { std::env::remove_var("EMBED_MODEL") };

    let context = context_with_query("anything");
    let result = SearchCompressorCallback
      .call(
        &context,
        CALL_ID,
        web_search::NAME,
        ToolResultStatus::Success,
        &long_content(),
      )
      .await;
    assert!(result.is_none());

    if let Some(value) = previous {
      unsafe { std::env::set_var("EMBED_MODEL", value) };
    }
  }
}
