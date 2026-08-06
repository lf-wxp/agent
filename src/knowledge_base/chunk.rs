pub fn fixed_length_chunking(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
  if chunk_size == 0 {
    return Vec::new();
  }

  assert!(chunk_size > overlap);

  let chars: Vec<char> = text.chars().collect();

  if chars.is_empty() {
    return Vec::new();
  }

  let mut chunks = Vec::new();
  let mut start = 0;

  while start < chars.len() {
    let end = (start + chunk_size).min(chars.len());

    let piece: String = chars[start..end].iter().collect();

    if !piece.trim().is_empty() {
      chunks.push(piece);
    }

    if end == chars.len() {
      break;
    }

    start = end - overlap;
  }

  chunks
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_basic_chunking() {
    let text = "abcdefghijklmnopqrstuvwxyz";

    let chunks = fixed_length_chunking(text, 10, 2);

    assert_eq!(chunks, vec!["abcdefghij", "ijklmnopqr", "qrstuvwxyz",]);
  }

  #[test]
  fn test_short_text() {
    let text = "hello";

    let chunks = fixed_length_chunking(text, 100, 10);

    assert_eq!(chunks, vec!["hello"]);
  }

  #[test]
  fn test_exact_size_text() {
    let text = "1234567890";

    let chunks = fixed_length_chunking(text, 10, 2);

    assert_eq!(chunks, vec!["1234567890"]);
  }

  #[test]
  fn test_overlap() {
    let text = "abcdefghijklmnop";

    let chunks = fixed_length_chunking(text, 6, 2);

    assert_eq!(chunks, vec!["abcdef", "efghij", "ijklmn", "mnop",]);
  }

  #[test]
  fn test_empty_text() {
    let chunks = fixed_length_chunking("", 10, 2);

    assert!(chunks.is_empty());
  }

  #[test]
  fn test_zero_chunk_size() {
    let chunks = fixed_length_chunking("hello", 0, 0);

    assert!(chunks.is_empty());
  }

  #[test]
  fn test_chinese_utf8_boundary() {
    let text = "你好世界你好世界";

    let chunks = fixed_length_chunking(text, 4, 1);

    for chunk in &chunks {
      // Important:
      // every chunk must be valid UTF-8
      assert!(std::str::from_utf8(chunk.as_bytes()).is_ok());
    }
  }

  #[test]
  fn test_no_replacement_character() {
    let text = "你好世界";

    let chunks = fixed_length_chunking(text, 4, 1);

    for chunk in chunks {
      assert!(
        !chunk.contains('�'),
        "chunk contains invalid UTF-8 replacement character"
      );
    }
  }
}
