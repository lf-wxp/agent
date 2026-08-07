use std::{cmp::Ordering, collections::BinaryHeap};

use anyhow::Ok;

use crate::{
  config,
  knowledge_base::embed::{embed_text, embed_texts},
};

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

struct ScoredIndex {
  similarity: f32,
  index: usize,
}

impl PartialEq for ScoredIndex {
  fn eq(&self, other: &Self) -> bool {
    self.similarity == other.similarity
  }
}

impl Eq for ScoredIndex {}
impl Ord for ScoredIndex {
  fn cmp(&self, other: &Self) -> Ordering {
    other.similarity.total_cmp(&self.similarity)
  }
}

impl PartialOrd for ScoredIndex {
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    Some(self.cmp(other))
  }
}

pub async fn vector_search(
  query: &str,
  chunks: &[String],
  top_k: usize,
) -> anyhow::Result<Vec<SearchHit>> {
  if chunks.is_empty() || top_k == 0 {
    return Ok(Vec::new());
  }
  let embed_model =
    config::embed_model().ok_or_else(|| anyhow::anyhow!("EMBED_MODEL is not set"))?;

  let query_embedding = embed_text(query, &embed_model).await?;
  let chunk_embeddings = embed_texts(chunks, &embed_model).await?;

  let mut heap: BinaryHeap<ScoredIndex> = BinaryHeap::with_capacity(top_k + 1);
  for (index, embedding) in chunk_embeddings.iter().enumerate() {
    let similarity = cosine_similarity(&query_embedding, embedding);
    heap.push(ScoredIndex { similarity, index });
    if heap.len() > top_k {
      heap.pop();
    }
  }

  let mut ranked: Vec<ScoredIndex> = heap.into_vec();
  ranked.sort_by(|a, b| b.similarity.total_cmp(&a.similarity));

  Ok(
    ranked
      .into_iter()
      .map(|scored| SearchHit {
        text: chunks[scored.index].clone(),
        similarity: scored.similarity,
      })
      .collect(),
  )
}

#[cfg(test)]
mod tests {
  use super::*;

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
    // the `total_cmp`-based ranking in `vector_search`.
    assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 2.0]), 0.0);
    assert_eq!(cosine_similarity(&[1.0, 2.0], &[0.0, 0.0]), 0.0);
    assert_eq!(cosine_similarity(&[0.0, 0.0], &[0.0, 0.0]), 0.0);
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
}
