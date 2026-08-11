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
