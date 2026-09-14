//! [`RunFingerprint`]: what an interrupted run has to be resumed *by*.
//!
//! # Why a resumed run needs an identity check
//!
//! A run stopped waiting on a human decision can sit there for as long as the human
//! takes, and the process that eventually resumes it is not the one that started it (see
//! [`crate::agent::runtime::AgentRunState`]). In between, the agent's own definition can
//! have changed: a different model, an edited system prompt, a tool that no longer
//! exists. Resuming into that is not a partial failure — the transcript references tool
//! calls by name, was produced under a prompt that no longer applies, and the pending
//! approval was granted against arguments the current tool may interpret differently.
//!
//! So a mismatch refuses to resume rather than proceeding and hoping. The alternative is
//! worse than an error: the model carries on plausibly, and nothing in the output says
//! the run was resumed under different rules than it was suspended under.
//!
//! # Why the digest is computed here rather than with a hasher from the standard library
//!
//! [`std::collections::hash_map::DefaultHasher`] is explicitly not stable across Rust
//! releases. Using it would make "the same instructions" digest differently after a
//! toolchain upgrade, and every suspended run would then refuse to resume — a
//! fail-closed outcome, so nothing unsafe, but a self-inflicted one that looks exactly
//! like a real mismatch. [`fnv1a64`] is fixed by this file instead, so a digest written
//! today still means the same thing to a future build.

use serde::{Deserialize, Serialize};

/// Bumped whenever the serialized shape of a suspended run changes incompatibly.
///
/// Separate from the digests below because it answers a different question: those detect
/// *the agent* having changed, this detects *this crate* having changed. Without it, a
/// state file written by an older layout would be deserialized into the current types —
/// either failing with a serde error that says nothing useful, or, worse, succeeding
/// because the change happened to be additive.
const STATE_VERSION: u32 = 1;

/// Identity of the agent configuration a suspended run was produced by.
///
/// Compared with [`Self::matches`] before a run is resumed; see the module docs for why
/// a mismatch is fatal rather than a warning.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunFingerprint {
  /// See [`STATE_VERSION`].
  pub state_version: u32,
  pub model: String,
  /// Digest rather than the text itself: a system prompt can be long, and it would then
  /// be duplicated into every suspended run's state file — which is also a file this
  /// crate warns is stored as plaintext (see [`crate::agent::session::FileSessionStore`]),
  /// so not copying the prompt into it is the better default.
  pub instructions_digest: String,
  /// Tool names, sorted. Stored in full rather than digested so a mismatch can say
  /// *which* tool went missing; the list is small, and "cannot resume" with no indication
  /// of what changed is a poor error to hand someone whose work is suspended behind it.
  ///
  /// Sorted because registration order is not part of the agent's identity — a tool set
  /// declared in a different order is the same tool set, and treating it as a change
  /// would refuse valid resumes.
  pub tools: Vec<String>,
}

impl RunFingerprint {
  /// Capture the identity of an agent configured with `model`, `instructions` and `tools`.
  pub fn new<'a>(
    model: &str,
    instructions: Option<&str>,
    tools: impl IntoIterator<Item = &'a str>,
  ) -> Self {
    let mut tools: Vec<String> = tools.into_iter().map(str::to_owned).collect();
    tools.sort_unstable();
    Self {
      state_version: STATE_VERSION,
      model: model.to_owned(),
      // An absent prompt and an empty one are deliberately not distinguished: neither
      // contributes anything to the request, so treating them as different identities
      // would refuse a resume over a difference the model never sees.
      instructions_digest: digest(instructions.unwrap_or_default()),
      tools,
    }
  }

  /// Why `self` cannot be resumed by an agent identified by `current`, or `None` when it
  /// can.
  ///
  /// Returns the explanation rather than a bare `bool` so the caller can report what
  /// changed. Checks are ordered most-fundamental first, and only the first difference is
  /// reported: once the state format itself is wrong, comparing the model or tool list is
  /// comparing fields that may not mean the same thing.
  pub fn mismatch(&self, current: &Self) -> Option<String> {
    if self.state_version != current.state_version {
      return Some(format!(
        "saved by an incompatible version of this crate (state format v{}, this build \
         writes v{})",
        self.state_version, current.state_version
      ));
    }
    if self.model != current.model {
      return Some(format!(
        "model changed from `{}` to `{}`",
        self.model, current.model
      ));
    }
    if self.instructions_digest != current.instructions_digest {
      return Some("the agent's instructions changed".to_owned());
    }
    // Reported as "missing" rather than "changed": a tool the transcript already called
    // is the one whose absence actually breaks a resume, whereas a newly *added* tool is
    // harmless — nothing in the suspended run refers to it.
    let missing: Vec<&str> = self
      .tools
      .iter()
      .filter(|name| !current.tools.contains(name))
      .map(String::as_str)
      .collect();
    if !missing.is_empty() {
      return Some(format!(
        "these tools are no longer available: {}",
        missing.join(", ")
      ));
    }
    None
  }

  /// Whether a run carrying `self` can be resumed by an agent identified by `current`.
  pub fn matches(&self, current: &Self) -> bool {
    self.mismatch(current).is_none()
  }
}

/// Lowercase hex of [`fnv1a64`] over `text`'s bytes.
fn digest(text: &str) -> String {
  format!("{:016x}", fnv1a64(text.as_bytes()))
}

/// FNV-1a, 64-bit. Chosen for being short enough to state in full and fixed forever by
/// this function — see the module docs on why a standard-library hasher is not usable
/// here. Not a cryptographic hash and not used as one: this detects accidental drift in a
/// local config, not tampering.
fn fnv1a64(bytes: &[u8]) -> u64 {
  const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
  const PRIME: u64 = 0x0000_0100_0000_01b3;

  let mut hash = OFFSET_BASIS;
  for byte in bytes {
    hash ^= u64::from(*byte);
    hash = hash.wrapping_mul(PRIME);
  }
  hash
}

#[cfg(test)]
mod tests {
  use super::*;

  fn fingerprint() -> RunFingerprint {
    RunFingerprint::new("gpt-4", Some("be helpful"), ["calculator", "delete_file"])
  }

  #[test]
  fn an_identical_configuration_matches() {
    assert!(fingerprint().matches(&fingerprint()));
    assert_eq!(fingerprint().mismatch(&fingerprint()), None);
  }

  #[test]
  fn a_different_model_is_reported() {
    let current = RunFingerprint::new("gpt-5", Some("be helpful"), ["calculator", "delete_file"]);
    let reason = fingerprint().mismatch(&current).expect("should not match");

    assert!(reason.contains("gpt-4"), "got: {reason}");
    assert!(reason.contains("gpt-5"), "got: {reason}");
  }

  #[test]
  fn changed_instructions_are_reported() {
    let current = RunFingerprint::new("gpt-4", Some("be terse"), ["calculator", "delete_file"]);
    assert!(
      fingerprint()
        .mismatch(&current)
        .is_some_and(|reason| reason.contains("instructions"))
    );
  }

  /// The prompt is digested, not stored, so this is really asserting that the digest is
  /// what the comparison rests on.
  #[test]
  fn the_instructions_are_not_stored_verbatim() {
    let fingerprint = RunFingerprint::new("gpt-4", Some("a very secret system prompt"), []);
    let json = serde_json::to_string(&fingerprint).unwrap();

    assert!(
      !json.contains("secret system prompt"),
      "the prompt must not be copied into persisted state: {json}"
    );
  }

  /// A tool the suspended run might have called must not silently disappear.
  #[test]
  fn a_removed_tool_is_named() {
    let current = RunFingerprint::new("gpt-4", Some("be helpful"), ["calculator"]);
    let reason = fingerprint().mismatch(&current).expect("should not match");

    assert!(reason.contains("delete_file"), "got: {reason}");
  }

  /// Adding a tool cannot invalidate a suspended run: nothing in it refers to the new
  /// one. Refusing here would make every tool addition orphan every pending approval.
  #[test]
  fn an_added_tool_still_matches() {
    let current = RunFingerprint::new(
      "gpt-4",
      Some("be helpful"),
      ["calculator", "delete_file", "web_search"],
    );

    assert!(fingerprint().matches(&current));
  }

  /// Registration order is not part of the agent's identity.
  #[test]
  fn tool_order_does_not_matter() {
    let reordered = RunFingerprint::new("gpt-4", Some("be helpful"), ["delete_file", "calculator"]);
    assert!(fingerprint().matches(&reordered));
  }

  /// An incompatible state format is reported ahead of anything else, since the other
  /// fields may not mean the same thing across versions.
  #[test]
  fn an_incompatible_state_version_is_reported_first() {
    let mut saved = fingerprint();
    saved.state_version = STATE_VERSION + 1;
    saved.model = "something else entirely".to_owned();

    let reason = saved.mismatch(&fingerprint()).expect("should not match");
    assert!(reason.contains("state format"), "got: {reason}");
  }

  #[test]
  fn no_instructions_and_empty_instructions_are_the_same_identity() {
    let absent = RunFingerprint::new("gpt-4", None, ["calculator"]);
    let empty = RunFingerprint::new("gpt-4", Some(""), ["calculator"]);

    assert!(absent.matches(&empty));
  }

  #[test]
  fn the_fingerprint_round_trips_through_json() {
    let json = serde_json::to_string(&fingerprint()).unwrap();
    let back: RunFingerprint = serde_json::from_str(&json).unwrap();

    assert_eq!(back, fingerprint());
  }

  /// The digest is fixed by this crate rather than by the toolchain, so these values are
  /// part of the on-disk format. A change here breaks every suspended run, and should
  /// therefore come with a `STATE_VERSION` bump rather than a quiet edit to the test.
  #[test]
  fn the_digest_is_stable_by_construction() {
    assert_eq!(digest(""), "cbf29ce484222325");
    assert_eq!(digest("a"), "af63dc4c8601ec8c");
    assert_eq!(digest("be helpful"), digest("be helpful"));
    assert_ne!(digest("be helpful"), digest("be terse"));
  }
}
