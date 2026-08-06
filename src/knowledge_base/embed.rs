use std::sync::LazyLock;

use anyhow::Ok;
use async_openai::{
  Client,
  config::OpenAIConfig,
  types::embeddings::{CreateEmbeddingRequestArgs, EmbeddingInput},
};

use crate::config;

/// Max inputs accepted per embeddings request by OpenAI-compatible APIs. Batches larger
/// than this are split, so one oversized call cannot make the whole batch fail.
const MAX_BATCH_SIZE: usize = 2048;

/// Reuse a single client within the process: avoid rebuilding the connection pool and
/// re-reading env vars on every call (mirrors `crate::llm::client::client`).
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

pub async fn embed_texts(texts: &[String], model: &str) -> anyhow::Result<Vec<Vec<f32>>> {
  if texts.is_empty() {
    return Ok(Vec::new());
  }

  let mut embeddings = Vec::with_capacity(texts.len());
  for batch in texts.chunks(MAX_BATCH_SIZE) {
    embeddings.extend(embed_batch(batch, model).await?);
  }
  Ok(embeddings)
}

/// Send a single request for a batch of at most [`MAX_BATCH_SIZE`] inputs.
async fn embed_batch(texts: &[String], model: &str) -> anyhow::Result<Vec<Vec<f32>>> {
  let request = CreateEmbeddingRequestArgs::default()
    .model(model)
    .input(EmbeddingInput::StringArray(texts.to_vec()))
    .build()?;

  let response = CLIENT.embeddings().create(request).await?;

  let mut data = response.data;
  data.sort_by_key(|embedding| embedding.index);

  // Callers (e.g. `vector_search`) match embeddings back to their source texts purely
  // by position, so a silent count mismatch would shift every vector after it out of
  // alignment and corrupt search results without ever raising an error.
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

pub async fn embed_text(text: &str, model: &str) -> anyhow::Result<Vec<f32>> {
  let owned = [text.to_string()];
  let mut vectors = embed_texts(&owned, model).await?;
  vectors
    .pop()
    .ok_or_else(|| anyhow::anyhow!("embedding API returned no vectors"))
}
