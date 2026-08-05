use serde::Deserialize;
use serde_json::json;

use crate::{
  config, http,
  tools::web_search::{NAME, WebSearchArgs},
  util::truncate_chars,
};

/// Tavily search endpoint.
const TAVILY_ENDPOINT: &str = "https://api.tavily.com/search";

/// Chars kept per snippet: enough to judge relevance without flooding the context window
/// when the model asks for 20 results.
const SNIPPET_PREVIEW_CHARS: usize = 500;

/// Chars of an error body kept in the message handed back to the model.
const ERROR_PREVIEW_CHARS: usize = 200;

/// The subset of the Tavily response we use; unknown fields are ignored.
#[derive(Debug, Deserialize)]
struct TavilyResponse {
  /// Only present because the request sets `include_answer`.
  answer: Option<String>,
  /// Defaulted rather than required: an empty result set may omit the field entirely.
  #[serde(default)]
  results: Vec<TavilyResult>,
}

#[derive(Debug, Deserialize)]
struct TavilyResult {
  title: String,
  url: String,
  content: String,
}

/// Run the tool against the raw JSON arguments produced by the model.
///
/// Returns a string in both the success and failure cases: tool errors are fed back as
/// tool messages so the model can rephrase the query or work around the failure, rather
/// than aborting the whole conversation.
pub async fn run(arguments: &str) -> String {
  let args = match serde_json::from_str::<WebSearchArgs>(arguments) {
    Ok(args) => args,
    Err(err) => return format!("Error: invalid arguments: {err}"),
  };

  match search(&args).await {
    Ok(response) => render(&response),
    Err(err) => {
      // Keep the full chain in the logs; the model only needs the summary.
      tracing::warn!("{NAME} failed: {err:#}");
      format!("Error: {err}")
    }
  }
}

/// Issue the search request.
async fn search(args: &WebSearchArgs) -> anyhow::Result<TavilyResponse> {
  let query = args.query.trim();
  anyhow::ensure!(!query.is_empty(), "query must not be empty");

  let api_key = config::tavily_api_key().ok_or_else(|| {
    anyhow::anyhow!("TAVILY_API_KEY is not set; create a key at https://app.tavily.com")
  })?;

  let response = http::client()
    .post(TAVILY_ENDPOINT)
    .bearer_auth(api_key)
    .json(&json!({
      "query": query,
      "search_depth": config::tavily_search_depth(),
      "max_results": args.effective_max_results(),
      "include_answer": true,
    }))
    .send()
    .await?;

  let status = response.status();
  // Read the body as text first: on failure Tavily's `{"detail": ...}` is the only useful
  // diagnostic, and going straight to `.json()` would bury it behind a decode error.
  let body = response.text().await?;

  anyhow::ensure!(
    status.is_success(),
    "Tavily returned {status}: {}",
    truncate_chars(&body, ERROR_PREVIEW_CHARS)
  );

  serde_json::from_str(&body).map_err(|err| {
    anyhow::anyhow!(
      "failed to parse Tavily response: {err}; body: {}",
      truncate_chars(&body, ERROR_PREVIEW_CHARS)
    )
  })
}

/// Format the response as compact text for the model.
///
/// Numbered entries with explicit URLs let the model cite its sources; snippets are
/// truncated so a large result set cannot blow up the context window.
fn render(response: &TavilyResponse) -> String {
  let answer = response
    .answer
    .as_deref()
    .map(str::trim)
    .filter(|answer| !answer.is_empty());

  if response.results.is_empty() {
    return match answer {
      Some(answer) => format!("Answer: {answer}"),
      None => "No results found.".to_owned(),
    };
  }

  let mut out = String::new();
  if let Some(answer) = answer {
    out.push_str("Answer: ");
    out.push_str(answer);
    out.push_str("\n\n");
  }

  for (position, result) in response.results.iter().enumerate() {
    // 1-based numbering matches how models refer back to sources.
    out.push_str(&format!(
      "[{}] {}\n{}\n{}\n\n",
      position + 1,
      result.title.trim(),
      result.url.trim(),
      truncate_chars(result.content.trim(), SNIPPET_PREVIEW_CHARS)
    ));
  }

  // Drop the trailing blank line left by the loop.
  out.truncate(out.trim_end().len());
  out
}

#[cfg(test)]
mod tests {
  use super::*;

  fn result(title: &str, url: &str, content: &str) -> TavilyResult {
    TavilyResult {
      title: title.to_owned(),
      url: url.to_owned(),
      content: content.to_owned(),
    }
  }

  #[test]
  fn renders_answer_and_numbered_results() {
    let response = TavilyResponse {
      answer: Some("Argentina won.".to_owned()),
      results: vec![
        result(
          "Final report",
          "https://example.com/a",
          "Argentina beat France.",
        ),
        result("Match stats", "https://example.com/b", "Score was 3-3."),
      ],
    };

    assert_eq!(
      render(&response),
      "Answer: Argentina won.\n\n\
       [1] Final report\nhttps://example.com/a\nArgentina beat France.\n\n\
       [2] Match stats\nhttps://example.com/b\nScore was 3-3."
    );
  }

  #[test]
  fn omits_answer_section_when_blank() {
    let response = TavilyResponse {
      answer: Some("   ".to_owned()),
      results: vec![result("Title", "https://example.com", "Body")],
    };

    assert_eq!(render(&response), "[1] Title\nhttps://example.com\nBody");
  }

  #[test]
  fn falls_back_to_answer_when_no_results() {
    let response = TavilyResponse {
      answer: Some("Direct answer.".to_owned()),
      results: Vec::new(),
    };

    assert_eq!(render(&response), "Answer: Direct answer.");
  }

  #[test]
  fn reports_empty_result_set() {
    let response = TavilyResponse {
      answer: None,
      results: Vec::new(),
    };

    assert_eq!(render(&response), "No results found.");
  }

  #[test]
  fn truncates_long_snippets() {
    let response = TavilyResponse {
      answer: None,
      results: vec![result("T", "https://example.com", &"x".repeat(1000))],
    };

    // Header lines plus exactly the snippet budget.
    assert!(render(&response).ends_with(&"x".repeat(SNIPPET_PREVIEW_CHARS)));
  }

  #[test]
  fn parses_real_response_shape() {
    // Field layout taken from the Tavily API reference, including fields we ignore.
    let body = r#"{
      "query": "who is Leo Messi?",
      "answer": "An Argentine footballer.",
      "images": [],
      "results": [
        {
          "title": "Lionel Messi Facts",
          "url": "https://www.britannica.com/facts/Lionel-Messi",
          "content": "Widely regarded as one of the greatest.",
          "score": 0.81025416,
          "raw_content": null
        }
      ],
      "response_time": "1.67"
    }"#;

    let response = serde_json::from_str::<TavilyResponse>(body).unwrap();
    assert_eq!(response.results.len(), 1);
    assert!(render(&response).contains("britannica.com"));
  }

  #[tokio::test]
  async fn reports_malformed_arguments_as_text() {
    assert!(
      run("not json")
        .await
        .starts_with("Error: invalid arguments")
    );
  }
}
