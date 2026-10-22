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

/// FNV-1a, 64-bit. Chosen for being short enough to state in full and fixed forever by
/// this function, unlike [`std::collections::hash_map::DefaultHasher`], which is
/// explicitly not stable across Rust releases — so a hash written to disk today still
/// means the same thing to a future build (see `crate::agent::fingerprint` and
/// `crate::knowledge_base::store`, which both persist one).
///
/// Not a cryptographic hash and not used as one: it detects accidental drift and finds
/// candidates for a lookup, and every caller that acts on a match confirms it some other
/// way where a collision would matter.
pub(crate) fn fnv1a64(bytes: &[u8]) -> u64 {
  const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
  const PRIME: u64 = 0x0000_0100_0000_01b3;

  let mut hash = OFFSET_BASIS;
  for byte in bytes {
    hash ^= u64::from(*byte);
    hash = hash.wrapping_mul(PRIME);
  }
  hash
}

/// Lowercase, zero-padded hex of [`fnv1a64`] over `text`'s bytes.
pub(crate) fn fnv1a64_hex(text: &str) -> String {
  format!("{:016x}", fnv1a64(text.as_bytes()))
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

  /// The published FNV-1a 64 test vectors: a change here would silently invalidate every
  /// hash already on disk.
  #[test]
  fn fnv1a64_matches_the_reference_vectors() {
    assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
    assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
    assert_eq!(fnv1a64(b"foobar"), 0x8594_4171_f739_67e8);
    assert_eq!(fnv1a64_hex("a"), "af63dc4c8601ec8c");
  }
}
