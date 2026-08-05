//! Small utilities shared across modules.

/// Truncate on character boundaries to avoid panics from splitting a multi-byte character (e.g. Chinese) in the middle.
///
/// Mainly used for content previews in logs/error messages, to avoid stuffing the entire response body into the error chain.
pub fn truncate_chars(content: &str, max_chars: usize) -> &str {
  match content.char_indices().nth(max_chars) {
    Some((idx, _)) => &content[..idx],
    None => content,
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn keeps_short_content_as_is() {
    assert_eq!(truncate_chars("abc", 10), "abc");
  }

  #[test]
  fn truncates_on_char_boundary() {
    // Byte-based slicing would panic in the middle of the multi-byte character here.
    assert_eq!(truncate_chars("中文内容", 2), "中文");
  }

  #[test]
  fn handles_zero_limit() {
    assert_eq!(truncate_chars("中文", 0), "");
  }
}
