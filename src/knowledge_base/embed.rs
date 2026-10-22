//! [`Embedder`]: turning text into vectors, the one model call retrieval needs.
//!
//! A trait for the same reason [`crate::llm::backend::ChatBackend`] is one: the vector
//! store and the memory built on it should not care who computes the vectors — an
//! OpenAI-compatible endpoint ([`OpenAiEmbedder`], the one that ships), a local model, or
//! a deterministic stand-in in tests.
//!
//! # What [`Embedder::model`] promises
//!
//! Vectors from different models live in different spaces, and comparing them yields
//! numbers that look like similarities and mean nothing. So every stored vector is tagged
//! with the model that produced it (see [`crate::knowledge_base::store`]), and that tag
//! is [`Embedder::model`]: two embedders reporting the same name must produce comparable
//! vectors, and one that changes what it computes must change its name.

use std::sync::LazyLock;

use async_openai::{
  Client,
  config::OpenAIConfig,
  types::embeddings::{CreateEmbeddingRequestArgs, EmbeddingInput},
};
use tracing::Instrument;

use crate::{
  agent::{TokenUsage, usage},
  config,
  llm::retry::{is_transient, with_retry},
};

/// Max inputs accepted per embeddings request by OpenAI-compatible APIs. Batches larger
/// than this are split, so one oversized call cannot make the whole batch fail.
const MAX_BATCH_SIZE: usize = 2048;

/// Most text, in bytes, sent in one request. Providers also cap the *total* size of a
/// request (OpenAI: 300 000 tokens across all inputs), which a few thousand ordinary
/// chunks can exceed without ever reaching [`MAX_BATCH_SIZE`] — a 1 MiB file cut into
/// 800-character chunks is ~1 300 inputs and ~260 000 tokens. 200 000 bytes is roughly
/// 50 000 tokens of English and at most about 130 000 of Chinese (3 bytes and 1-2 tokens a
/// character), inside the 300 000 that is the tightest limit in use, and costs a handful
/// of requests where one would have done. An estimate by bytes, not a tokenizer: a text
/// of unusual density can still exceed a provider's limit, and the provider's own error
/// then reaches the caller.
const MAX_BATCH_BYTES: usize = 200_000;

/// Something that turns texts into vectors. See the module docs for what
/// [`Self::model`] has to mean.
#[async_trait::async_trait]
pub trait Embedder: Send + Sync {
  /// Names the vector space: stored with every vector, and only vectors with the same
  /// name are ever compared.
  fn model(&self) -> &str;

  /// One vector per text, in the same order. Implementations must return exactly
  /// `texts.len()` vectors or fail — callers match them back to their texts by position.
  async fn embed(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>>;

  /// [`Self::embed`], checking that promise instead of trusting it: a count that does not
  /// match would shift every vector after it onto the wrong text, and nothing downstream
  /// could tell. What callers that match vectors to texts by position should use.
  async fn embed_checked(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
    let vectors = self.embed(texts).await?;
    anyhow::ensure!(
      vectors.len() == texts.len(),
      "the embedder returned {} vectors for {} texts",
      vectors.len(),
      texts.len()
    );
    Ok(vectors)
  }

  /// The vector of a single text.
  async fn embed_one(&self, text: &str) -> anyhow::Result<Vec<f32>> {
    self
      .embed_checked(&[text.to_owned()])
      .await?
      .pop()
      .ok_or_else(|| anyhow::anyhow!("the embedder returned no vector"))
  }
}

/// Where to cut `texts` into requests: consecutive runs of at most [`MAX_BATCH_SIZE`]
/// texts and [`MAX_BATCH_BYTES`] bytes. A text over the byte budget on its own still gets
/// a request — it is the provider's per-input limit that decides about that one, and
/// failing it here would hide the provider's own message.
fn plan_batches(texts: &[String]) -> Vec<std::ops::Range<usize>> {
  let mut batches = Vec::new();
  let (mut start, mut bytes) = (0, 0usize);
  for (index, text) in texts.iter().enumerate() {
    let in_batch = index - start;
    if in_batch > 0 && (in_batch >= MAX_BATCH_SIZE || bytes + text.len() > MAX_BATCH_BYTES) {
      batches.push(start..index);
      (start, bytes) = (index, 0);
    }
    bytes += text.len();
  }
  if start < texts.len() {
    batches.push(start..texts.len());
  }
  batches
}

/// Reuse a single client within the process: avoid rebuilding the connection pool and
/// re-reading env vars on every call.
///
/// Built from its own config rather than `OpenAIConfig::default()`: chat completions and
/// embeddings frequently need different providers (e.g. a reasoning-only model has no
/// embeddings endpoint), so `EMBED_BASE_URL` / `EMBED_API_KEY` let them be pointed at a
/// separate provider. Either falls back to the default `OPENAI_BASE_URL` / `OPENAI_API_KEY`
/// when unset, so a single provider that serves both still works unchanged.
static CLIENT: LazyLock<Client<OpenAIConfig>> = LazyLock::new(|| {
  let mut config = OpenAIConfig::default();
  if let Some(base_url) = config::embed_base_url() {
    config = config.with_api_base(base_url);
  }
  if let Some(api_key) = config::embed_api_key() {
    config = config.with_api_key(api_key);
  }
  Client::with_config(config)
});

/// Any server speaking the OpenAI embeddings API.
///
/// Every request is retried on transient failures, traced as an `embeddings` span, and
/// its token usage charged to the agent run it happens inside (see
/// [`crate::agent::usage`]) — so what retrieval and memory spend shows up in that run's
/// `usage_by_model`, and in a cost estimate, like any other model call.
#[derive(Clone)]
pub struct OpenAiEmbedder {
  client: Client<OpenAIConfig>,
  model: String,
}

impl OpenAiEmbedder {
  /// An embedder for `model` on the endpoint and credentials in `config`.
  pub fn new(config: OpenAIConfig, model: impl Into<String>) -> Self {
    Self {
      client: Client::with_config(config),
      model: model.into(),
    }
  }

  /// `model` on the process-wide embeddings endpoint (`EMBED_BASE_URL` /
  /// `EMBED_API_KEY`, falling back to the chat ones).
  pub fn with_model(model: impl Into<String>) -> Self {
    Self {
      client: CLIENT.clone(),
      model: model.into(),
    }
  }

  /// The configured embedder: [`config::embed_model`] (`EMBED_MODEL`) on the
  /// process-wide endpoint.
  ///
  /// # Errors
  ///
  /// When `EMBED_MODEL` is not set — there is no sensible default, since the right model
  /// depends entirely on the provider.
  pub fn from_config() -> anyhow::Result<Self> {
    let model = config::embed_model().ok_or_else(|| anyhow::anyhow!("EMBED_MODEL is not set"))?;
    Ok(Self::with_model(model))
  }

  /// Send a single request for one batch from [`plan_batches`].
  async fn embed_batch(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
    let span = tracing::info_span!(
      "embeddings",
      otel.name = %format!("embeddings {}", self.model),
      otel.kind = "client",
      gen_ai.operation.name = "embeddings",
      gen_ai.request.model = %self.model,
      gen_ai.usage.input_tokens = tracing::field::Empty,
      agent.inputs = texts.len() as i64,
    );
    let response = with_retry(
      || async {
        let request = CreateEmbeddingRequestArgs::default()
          .model(&self.model)
          .input(EmbeddingInput::StringArray(texts.to_vec()))
          .build()?;
        anyhow::Ok(self.client.embeddings().create(request).await?)
      },
      is_transient,
    )
    .instrument(span.clone())
    .await?;

    span.record(
      "gen_ai.usage.input_tokens",
      i64::from(response.usage.prompt_tokens),
    );
    usage::charge(
      &self.model,
      TokenUsage {
        prompt_tokens: response.usage.prompt_tokens,
        total_tokens: response.usage.total_tokens,
        ..TokenUsage::default()
      },
    );

    let mut data = response.data;
    data.sort_by_key(|embedding| embedding.index);

    // Callers match embeddings back to their source texts purely by position, so a silent
    // count mismatch would shift every vector after it out of alignment and corrupt
    // search results without ever raising an error.
    anyhow::ensure!(
      data.len() == texts.len(),
      "embedding API returned {} vectors for {} inputs",
      data.len(),
      texts.len()
    );

    Ok(
      data
        .into_iter()
        .map(|embedding| embedding.embedding)
        .collect(),
    )
  }
}

#[async_trait::async_trait]
impl Embedder for OpenAiEmbedder {
  fn model(&self) -> &str {
    &self.model
  }

  async fn embed(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
    let mut embeddings = Vec::with_capacity(texts.len());
    for batch in plan_batches(texts) {
      embeddings.extend(self.embed_batch(&texts[batch]).await?);
    }
    Ok(embeddings)
  }
}

/// Embed `texts` with `model` on the process-wide embeddings endpoint.
///
/// Kept so code written against the earlier free functions still compiles; it is
/// [`OpenAiEmbedder::with_model`] plus [`Embedder::embed`] and nothing more.
#[deprecated(note = "use `OpenAiEmbedder::with_model(model).embed(texts)`")]
pub async fn embed_texts(texts: &[String], model: &str) -> anyhow::Result<Vec<Vec<f32>>> {
  OpenAiEmbedder::with_model(model).embed(texts).await
}

/// Embed one `text` with `model` on the process-wide embeddings endpoint.
///
/// See [`embed_texts`].
#[deprecated(note = "use `OpenAiEmbedder::with_model(model).embed(..)`")]
pub async fn embed_text(text: &str, model: &str) -> anyhow::Result<Vec<f32>> {
  OpenAiEmbedder::with_model(model).embed_one(text).await
}

#[cfg(test)]
mod tests {
  use super::*;

  fn texts_of(sizes: &[usize]) -> Vec<String> {
    sizes.iter().map(|size| "x".repeat(*size)).collect()
  }

  #[test]
  fn nothing_to_embed_plans_no_requests() {
    assert!(plan_batches(&[]).is_empty());
  }

  #[test]
  fn small_inputs_go_in_one_request() {
    assert_eq!(plan_batches(&texts_of(&[10, 20, 30])), vec![0..3]);
  }

  /// The provider's per-request input count.
  #[test]
  fn a_request_never_holds_more_than_the_input_limit() {
    let texts = texts_of(&vec![1; MAX_BATCH_SIZE + 1]);
    assert_eq!(
      plan_batches(&texts),
      [0..MAX_BATCH_SIZE, MAX_BATCH_SIZE..MAX_BATCH_SIZE + 1]
    );
  }

  /// The provider's total-size-per-request limit, which the count limit does not imply.
  #[test]
  fn a_request_never_holds_more_than_the_byte_budget() {
    let half = MAX_BATCH_BYTES / 2;
    let batches = plan_batches(&texts_of(&[half, half, 1, half]));
    assert_eq!(batches, [0..2, 2..4]);
    for batch in batches {
      let total: usize = texts_of(&[half, half, 1, half])[batch]
        .iter()
        .map(String::len)
        .sum();
      assert!(total <= MAX_BATCH_BYTES);
    }
  }

  /// Over budget by itself: still sent, alone, so the provider's own error reaches the
  /// caller rather than a made-up one.
  #[test]
  fn an_oversized_text_gets_a_request_of_its_own() {
    let batches = plan_batches(&texts_of(&[5, MAX_BATCH_BYTES + 1, 5]));
    assert_eq!(batches, [0..1, 1..2, 2..3]);
  }

  /// Every text lands in exactly one request, in order.
  #[test]
  fn the_plan_covers_every_text_exactly_once() {
    let sizes: Vec<usize> = (0..500).map(|n| 1 + (n * 7919) % 9_000).collect();
    let batches = plan_batches(&texts_of(&sizes));
    let mut next = 0;
    for batch in &batches {
      assert_eq!(batch.start, next);
      next = batch.end;
    }
    assert_eq!(next, sizes.len());
  }

  /// A wrong count is refused, not passed through to shift every vector after it.
  #[tokio::test]
  async fn embed_checked_refuses_a_wrong_count() {
    struct Short;
    #[async_trait::async_trait]
    impl Embedder for Short {
      fn model(&self) -> &str {
        "short"
      }
      async fn embed(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(vec![vec![1.0]; texts.len().saturating_sub(1)])
      }
    }
    let err = Short
      .embed_checked(&["a".to_owned(), "b".to_owned()])
      .await
      .unwrap_err()
      .to_string();
    assert!(err.contains("1 vectors for 2 texts"), "got: {err}");
    assert!(Short.embed_one("a").await.is_err());
  }

  /// The compatibility wrappers keep the old contract: nothing to embed, nothing sent.
  #[tokio::test]
  #[allow(deprecated)]
  async fn the_deprecated_free_functions_still_answer_an_empty_request_without_a_call() {
    assert!(embed_texts(&[], "any-model").await.unwrap().is_empty());
  }
}
