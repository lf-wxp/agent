//! A deterministic [`Embedder`] for tests: no network, and similar texts really do get
//! similar vectors, so ranking can be asserted on.

#[cfg(feature = "sqlite")]
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::{knowledge_base::embed::Embedder, util::fnv1a64};

const DIM: usize = 64;

/// Bag of words: each lowercase word adds 1 to the dimension its hash lands on. Texts
/// sharing words point the same way; texts sharing none are (almost) orthogonal.
pub struct WordEmbedder {
  model: String,
  calls: AtomicUsize,
  texts: AtomicUsize,
}

impl WordEmbedder {
  pub fn new() -> Self {
    Self::named("words-v1")
  }

  /// Another "model" — for asserting that vectors from different ones are kept apart.
  pub fn named(model: &str) -> Self {
    Self {
      model: model.to_owned(),
      calls: AtomicUsize::new(0),
      texts: AtomicUsize::new(0),
    }
  }

  /// Requests made so far.
  pub fn calls(&self) -> usize {
    self.calls.load(Ordering::SeqCst)
  }

  /// Texts embedded so far, across every request. Only the store's tests count these.
  #[cfg(feature = "sqlite")]
  pub fn texts(&self) -> usize {
    self.texts.load(Ordering::SeqCst)
  }

  pub fn vector(text: &str) -> Vec<f32> {
    let mut vector = vec![0.0; DIM];
    for word in text
      .to_lowercase()
      .split(|c: char| !c.is_alphanumeric())
      .filter(|word| !word.is_empty())
    {
      vector[(fnv1a64(word.as_bytes()) % DIM as u64) as usize] += 1.0;
    }
    vector
  }
}

#[async_trait::async_trait]
impl Embedder for WordEmbedder {
  fn model(&self) -> &str {
    &self.model
  }

  async fn embed(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
    self.calls.fetch_add(1, Ordering::SeqCst);
    self.texts.fetch_add(texts.len(), Ordering::SeqCst);
    Ok(texts.iter().map(|text| Self::vector(text)).collect())
  }
}

/// A [`WordEmbedder`] whose endpoint can be taken down and brought back, for asserting how
/// callers behave while it is failing.
#[cfg(feature = "sqlite")]
pub struct Flaky {
  inner: WordEmbedder,
  failing: AtomicBool,
}

#[cfg(feature = "sqlite")]
impl Flaky {
  pub fn new() -> Self {
    Self {
      inner: WordEmbedder::new(),
      failing: AtomicBool::new(false),
    }
  }

  pub fn set_failing(&self, failing: bool) {
    self.failing.store(failing, Ordering::SeqCst);
  }

  /// Requests made so far, failed ones included.
  pub fn calls(&self) -> usize {
    self.inner.calls()
  }
}

#[cfg(feature = "sqlite")]
#[async_trait::async_trait]
impl Embedder for Flaky {
  fn model(&self) -> &str {
    self.inner.model()
  }

  async fn embed(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
    if self.failing.load(Ordering::SeqCst) {
      self.inner.calls.fetch_add(1, Ordering::SeqCst);
      anyhow::bail!("the embedding endpoint is down");
    }
    self.inner.embed(texts).await
  }
}
