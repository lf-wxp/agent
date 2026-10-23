//! Similarity ranking, shared by the throwaway [`vector_search`] and the persistent
//! [`crate::knowledge_base::store::VectorStore`].

use std::{cmp::Ordering, collections::BinaryHeap};

use crate::knowledge_base::embed::{Embedder, OpenAiEmbedder};

#[derive(Debug, Clone)]
pub struct SearchHit {
  pub text: String,
  pub similarity: f32,
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
  let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
  let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
  let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
  if norm_a == 0.0 || norm_b == 0.0 {
    0.0
  } else {
    dot / (norm_a * norm_b)
  }
}

/// `vector` scaled to unit length, or `None` for a zero (or non-finite) vector, which has
/// no direction to compare. Precomputed once per stored vector, so ranking a query
/// against it is a dot product rather than a full [`cosine_similarity`].
///
/// Only the persistent store uses this: [`vector_search`] compares its handful of vectors
/// directly, so without the `sqlite` feature there is nothing to precompute for.
#[cfg(feature = "sqlite")]
pub(crate) fn normalized(vector: &[f32]) -> Option<Vec<f32>> {
  let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
  (norm.is_finite() && norm > 0.0).then(|| vector.iter().map(|x| x / norm).collect())
}

struct ScoredIndex {
  similarity: f32,
  index: usize,
}

impl PartialEq for ScoredIndex {
  fn eq(&self, other: &Self) -> bool {
    self.cmp(other) == Ordering::Equal
  }
}

impl Eq for ScoredIndex {}

/// Reversed, so the [`BinaryHeap`] is a min-heap whose top is the weakest survivor. Ties
/// go to the lower index, which keeps results stable from one search to the next.
impl Ord for ScoredIndex {
  fn cmp(&self, other: &Self) -> Ordering {
    other
      .similarity
      .total_cmp(&self.similarity)
      .then(self.index.cmp(&other.index))
  }
}

impl PartialOrd for ScoredIndex {
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    Some(self.cmp(other))
  }
}

/// The `k` best `(index, similarity)` pairs, best first. Keeps only `k` candidates in
/// memory however many are scored.
pub(crate) fn top_k(scores: impl IntoIterator<Item = (usize, f32)>, k: usize) -> Vec<(usize, f32)> {
  if k == 0 {
    return Vec::new();
  }
  let mut heap: BinaryHeap<ScoredIndex> = BinaryHeap::with_capacity(k + 1);
  for (index, similarity) in scores {
    heap.push(ScoredIndex { similarity, index });
    if heap.len() > k {
      heap.pop();
    }
  }
  let ranked = heap.into_sorted_vec();
  // `into_sorted_vec` is ascending by `Ord`, which is reversed: best first already.
  ranked
    .into_iter()
    .map(|scored| (scored.index, scored.similarity))
    .collect()
}

/// Rank `chunks` against `query` with the configured embedder (`EMBED_MODEL`), embedding
/// both on the spot. For a corpus searched more than once, embed it once into a
/// [`crate::knowledge_base::store::VectorStore`] instead.
pub async fn vector_search(
  query: &str,
  chunks: &[String],
  top_k: usize,
) -> crate::Result<Vec<SearchHit>> {
  if chunks.is_empty() || top_k == 0 {
    return Ok(Vec::new());
  }
  vector_search_with(&OpenAiEmbedder::from_config()?, query, chunks, top_k).await
}

/// [`vector_search`] with a given embedder.
pub async fn vector_search_with(
  embedder: &dyn Embedder,
  query: &str,
  chunks: &[String],
  k: usize,
) -> crate::Result<Vec<SearchHit>> {
  if chunks.is_empty() || k == 0 {
    return Ok(Vec::new());
  }
  let mut texts = Vec::with_capacity(chunks.len() + 1);
  texts.push(query.to_owned());
  texts.extend_from_slice(chunks);
  let mut vectors = embedder.embed_checked(&texts).await?;
  let query_vector = vectors.remove(0);

  let scores = vectors
    .iter()
    .enumerate()
    .map(|(index, vector)| (index, cosine_similarity(&query_vector, vector)));
  Ok(
    top_k(scores, k)
      .into_iter()
      .map(|(index, similarity)| SearchHit {
        text: chunks[index].clone(),
        similarity,
      })
      .collect(),
  )
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::knowledge_base::testing::WordEmbedder;

  #[test]
  fn cosine_similarity_of_identical_vectors_is_one() {
    let similarity = cosine_similarity(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]);
    assert!((similarity - 1.0).abs() < f32::EPSILON, "got {similarity}");
  }

  #[test]
  fn cosine_similarity_of_orthogonal_vectors_is_zero() {
    let similarity = cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]);
    assert!(similarity.abs() < f32::EPSILON, "got {similarity}");
  }

  #[test]
  fn cosine_similarity_of_opposite_vectors_is_negative_one() {
    let similarity = cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]);
    assert!((similarity + 1.0).abs() < f32::EPSILON, "got {similarity}");
  }

  #[test]
  fn cosine_similarity_is_scale_invariant() {
    let a = cosine_similarity(&[1.0, 2.0], &[3.0, 4.0]);
    let b = cosine_similarity(&[2.0, 4.0], &[30.0, 40.0]);
    assert!((a - b).abs() < 1e-6, "got {a} vs {b}");
  }

  #[test]
  fn cosine_similarity_of_a_zero_vector_is_zero_rather_than_nan() {
    // A zero-norm vector would otherwise divide by zero and produce `NaN`, which breaks
    // the `total_cmp`-based ranking in `top_k`.
    assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 2.0]), 0.0);
    assert_eq!(cosine_similarity(&[1.0, 2.0], &[0.0, 0.0]), 0.0);
    assert_eq!(cosine_similarity(&[0.0, 0.0], &[0.0, 0.0]), 0.0);
  }

  #[cfg(feature = "sqlite")]
  #[test]
  fn a_zero_vector_has_no_direction() {
    assert!(normalized(&[0.0, 0.0]).is_none());
    let unit = normalized(&[3.0, 4.0]).unwrap();
    assert!((unit[0] - 0.6).abs() < 1e-6 && (unit[1] - 0.8).abs() < 1e-6);
  }

  #[test]
  fn top_k_keeps_the_best_in_order_and_breaks_ties_by_index() {
    let scores = [(0, 0.1), (1, 0.9), (2, 0.5), (3, 0.9), (4, 0.7)];
    assert_eq!(top_k(scores, 3), [(1, 0.9), (3, 0.9), (4, 0.7)]);
    assert_eq!(top_k(scores, 10).len(), 5);
    assert!(top_k(scores, 0).is_empty());
  }

  #[tokio::test]
  async fn vector_search_of_empty_chunks_returns_no_hits_without_calling_the_embedder() {
    // `top_k > 0` but no chunks: must short-circuit before needing `EMBED_MODEL` to be
    // configured, since there is nothing to embed against.
    let hits = vector_search("query", &[], 5).await.unwrap();
    assert!(hits.is_empty());
  }

  #[tokio::test]
  async fn vector_search_of_zero_top_k_returns_no_hits_without_calling_the_embedder() {
    let hits = vector_search("query", &["a".to_owned()], 0).await.unwrap();
    assert!(hits.is_empty());
  }

  #[tokio::test]
  async fn vector_search_ranks_the_matching_chunk_first() {
    let embedder = WordEmbedder::new();
    let chunks = vec![
      "the cat sat on the mat".to_owned(),
      "rust borrow checker lifetimes".to_owned(),
      "bread flour water yeast".to_owned(),
    ];
    let hits = vector_search_with(&embedder, "rust lifetimes", &chunks, 2)
      .await
      .unwrap();
    assert_eq!(hits[0].text, chunks[1]);
    assert_eq!(embedder.calls(), 1, "query and chunks go in one request");
  }
}
