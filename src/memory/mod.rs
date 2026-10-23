//! Long-term memory: short facts the model saves in one conversation and finds again in
//! later ones.
//!
//! [`MemoryStore`] keeps them in a [`VectorStore`] collection of their own, one document
//! per fact, under a *scope* — whoever the memories are about. A single-user front-end
//! (the CLI) uses one scope; a multi-tenant server opens one `MemoryStore` per user, so
//! one user's memories can never surface in another's conversation.
//!
//! Three pieces build on it:
//!
//! - [`tools`] — `remember`, `forget` and `recall`, the model's own handle on it.
//! - [`RecallCallback`] — looks up the memories relevant to each new user message and
//!   puts them in front of the model, so it does not have to think of searching.
//! - `/memory` in the CLI, which lists everything stored, so a human can see what the
//!   model has kept.
//!
//! # What is stored, and what is refused
//!
//! Plain text in a local SQLite file, created owner-read/write only on Unix (see
//! [`VectorStore::open`]; elsewhere it gets what the directory grants) — nothing is
//! encrypted beyond that, so the *file* is protected by file permissions. The *text* is another matter: to be searched by meaning, every
//! fact is sent to the embedding provider when it is saved, and every recall sends the
//! opening of the user's message there as the query (see [`RecallCallback`]). That provider
//! is the chat model's own (embeddings go to `OPENAI_BASE_URL` / `OPENAI_API_KEY`), so it
//! sees nothing it was not already sent. Use a local or self-hosted endpoint when that
//! matters.
//!
//! A fact that looks like a credential (an API key, a private key, a token, a labelled
//! password) is refused outright rather than stored, and the `remember` tool's
//! description tells the model not to save secrets or unconfirmed tool output. That
//! screen is a safety net for the shapes people actually type, not a guarantee. Facts are
//! capped at [`MAX_FACT_CHARS`], and a scope at [`MAX_MEMORIES`], so a model stuck in a
//! loop cannot fill the disk.
//!
//! # Prompt injection
//!
//! A web page read in one conversation can try to plant an instruction to be "remembered"
//! and replayed in every later one. Three things limit that: the tool description asks for
//! facts the user stated or confirmed, every write is a visible tool call in both
//! front-ends, and recalled memories reach the model labelled as reference data — and as
//! possibly untrusted — in an ordinary message, never as a system instruction (see
//! [`RecallCallback`]).
//!
//! `remember` is not gated behind an approval prompt (it is frequent and reversible, and
//! `/memory` shows exactly what it saved). `forget` is: it destroys something the user
//! asked the agent to keep, and a single injected instruction should not be able to empty
//! the store. A front-end wires that up by putting [`tools::GATED`] (today just
//! [`tools::FORGET`]) in its approval list — the `cli` binary does, without a standing
//! "always allow". `forget` takes the memory's text as well as its number and refuses to
//! delete a memory that does not say it ([`MemoryStore::forget_checked`]), so the
//! approval prompt shows what will actually be deleted rather than a bare number.
//!
//! `remember` is refused for the rest of a turn in which the model has read something it
//! cannot vouch for — a web page, a file, an MCP tool's output — until the user has said
//! it in their own words ([`tools::RememberGuard`]); see the module docs there. Together
//! with the labelling of recalled memories this is what stands between a planted
//! instruction and a standing one.

mod recall;
pub mod tools;

use std::{
  path::Path,
  sync::{Arc, LazyLock},
};

use regex::Regex;
use serde_json::json;

pub use recall::{DEFAULT_MIN_SIMILARITY, RecallCallback};

use crate::knowledge_base::{
  embed::Embedder,
  store::{DeleteOutcome, Document, Inserted, VectorStore},
};

/// Longest fact accepted, in characters. A memory is one fact, not a document; anything
/// longer is almost always a pasted page that belongs in a knowledge base instead.
pub const MAX_FACT_CHARS: usize = 500;

/// Most memories one scope may hold before `remember` asks for something to be forgotten.
pub const MAX_MEMORIES: usize = 2000;

/// Collection names are prefixed so a memory scope cannot collide with a document
/// collection in the same file.
const COLLECTION_PREFIX: &str = "memory:";

/// One remembered fact.
#[derive(Debug, Clone, PartialEq)]
pub struct Memory {
  /// What `forget` takes. Stable for the memory's lifetime.
  pub id: i64,
  pub text: String,
  /// RFC 3339, UTC.
  pub created_at: String,
}

impl Memory {
  /// The date part of [`Self::created_at`], for display.
  pub fn date(&self) -> &str {
    self.created_at.get(..10).unwrap_or(&self.created_at)
  }
}

impl From<Document> for Memory {
  fn from(document: Document) -> Self {
    Self {
      id: document.id,
      text: document.text,
      created_at: document.created_at,
    }
  }
}

/// What [`MemoryStore::remember`] did.
#[derive(Debug, Clone, PartialEq)]
pub enum Remembered {
  /// Stored as a new memory.
  New(Memory),
  /// The exact same text was already stored; nothing changed.
  Existing(Memory),
}

/// What [`MemoryStore::forget_checked`] did.
#[derive(Debug, Clone, PartialEq)]
pub enum Forgotten {
  /// The memory was there, said what the caller expected, and is gone.
  Deleted(Memory),
  /// Memory `id` exists but does not say what the caller expected — the number is stale
  /// or mistaken, since numbers are never reused for another memory. Nothing was deleted.
  Mismatch(Memory),
  /// There is no such memory in this scope.
  Missing,
}

/// See the module docs. Cheap to clone.
#[derive(Debug, Clone)]
pub struct MemoryStore {
  store: VectorStore,
  scope: String,
  collection: String,
}

impl MemoryStore {
  /// Memories for `scope`, kept in `store`.
  pub fn new(store: VectorStore, scope: impl Into<String>) -> Self {
    let scope = scope.into();
    Self {
      store,
      collection: format!("{COLLECTION_PREFIX}{scope}"),
      scope,
    }
  }

  /// [`Self::new`] over the SQLite file at `path`.
  pub fn open(
    path: impl AsRef<Path>,
    embedder: Arc<dyn Embedder>,
    scope: impl Into<String>,
  ) -> crate::Result<Self> {
    Ok(Self::new(VectorStore::open(path, embedder)?, scope))
  }

  pub fn scope(&self) -> &str {
    &self.scope
  }

  /// The store underneath, e.g. to share its file with a document collection.
  pub fn vector_store(&self) -> &VectorStore {
    &self.store
  }

  /// Identifies the scope's contents for the purpose of caching something derived from
  /// them: two equal revisions mean nothing this scope holds has changed in between.
  ///
  /// It is the collection's version in the store, which every write bumps in the same
  /// transaction as the change — so a write through this handle, through another handle
  /// on the file, or by another process all move it, and a write to some *other* scope or
  /// collection in the same file does not.
  pub async fn revision(&self) -> crate::Result<i64> {
    self.store.version(&self.collection).await
  }

  /// Save `fact`. Whitespace is trimmed and runs of it collapsed, so the same fact typed
  /// twice is recognized as one.
  ///
  /// # Errors
  ///
  /// When the fact is blank, longer than [`MAX_FACT_CHARS`], looks like a credential, or
  /// the scope already holds [`MAX_MEMORIES`] — each with a message the model can act on.
  pub async fn remember(&self, fact: &str) -> crate::Result<Remembered> {
    let fact = normalize(fact);
    crate::error::ensure!(!fact.is_empty(), invalid_input, "the fact is empty");
    let chars = fact.chars().count();
    crate::error::ensure!(
      chars <= MAX_FACT_CHARS,
      invalid_input,
      "the fact is {chars} characters; keep it to one short fact of at most \
       {MAX_FACT_CHARS}"
    );
    if let Some(kind) = looks_like_a_secret(&fact) {
      crate::error::bail!(
        invalid_input,
        "this looks like {kind}; memories are stored as plain text, so it was not saved"
      );
    }
    let source = format!("memory-{}", uuid::Uuid::new_v4());
    // The duplicate check, the size limit and the insert are one transaction inside the
    // store: two `remember` calls — two processes sharing the file, say — cannot both
    // pass "not there yet, room left". The returned document is what was stored, so the
    // reply and what `/memory` prints later agree on the timestamp too.
    match self
      .store
      .insert_unique(&self.collection, &source, &fact, json!({}), MAX_MEMORIES)
      .await?
    {
      Inserted::New(document) => Ok(Remembered::New(document.into())),
      Inserted::Existing(document) => {
        // A document left over from another embedding model is found by text but
        // invisible to search, so answering `Existing` as it is would claim a fact is
        // remembered that no recall will ever return. Re-embed *this one* — a single
        // short text — and leave migrating the rest to an explicit `--reindex-memory`:
        // a tool call is no place for an unannounced rewrite of every stored vector.
        self
          .store
          .reindex_document(&self.collection, document.id)
          .await?;
        Ok(Remembered::Existing(document.into()))
      }
      Inserted::Full(count) => {
        crate::error::bail!(
          storage,
          "memory is full ({count} facts); forget something outdated first"
        )
      }
    }
  }

  /// Delete memory `id`, returning it; `None` when this scope has no such memory.
  pub async fn forget(&self, id: i64) -> crate::Result<Option<Memory>> {
    Ok(
      self
        .store
        .delete(&self.collection, id)
        .await?
        .map(Memory::from),
    )
  }

  /// Delete memory `id` only if it says what `expected` quotes of it.
  ///
  /// For a caller acting on a number it was given some time ago — the model, from a
  /// listing earlier in the conversation — and, above all, for a human approving the call:
  /// what they were shown is the quoted text, so what is deleted has to be that.
  /// Numbers are never reused for another memory, so a mismatch means the number is
  /// wrong, not that the memory moved.
  ///
  /// `expected` matches when, ignoring case and runs of whitespace, it is the whole fact,
  /// or a quote of at least [`MIN_QUOTE_CHARS`] characters from inside it. (A shorter
  /// fragment would match too many different facts to say anything.)
  pub async fn forget_checked(&self, id: i64, expected: &str) -> crate::Result<Forgotten> {
    let expected = expected.to_owned();
    Ok(
      match self
        .store
        .delete_if(&self.collection, id, move |document| {
          quotes(&document.text, &expected)
        })
        .await?
      {
        DeleteOutcome::Deleted(document) => Forgotten::Deleted(document.into()),
        DeleteOutcome::Refused(document) => Forgotten::Mismatch(document.into()),
        DeleteOutcome::Missing => Forgotten::Missing,
      },
    )
  }

  /// How many memories in this scope were embedded with a model other than the current
  /// one, and so cannot be found by [`Self::recall`] until [`Self::reindex_stale`] runs.
  pub async fn stale_count(&self) -> crate::Result<usize> {
    self.store.stale_count(&self.collection).await
  }

  /// Re-embed the memories left behind by an earlier embedding model, so they become
  /// recallable again. Their numbers are kept, so anything referring to one still works.
  ///
  /// Returns how many were re-embedded — 0, cheaply, when there is nothing to do.
  pub async fn reindex_stale(&self) -> crate::Result<usize> {
    self.store.reindex(&self.collection).await
  }

  /// How many memories this scope holds, whatever model embedded them. Cheap: a count,
  /// not a listing.
  pub async fn count(&self) -> crate::Result<usize> {
    self.store.count(&self.collection).await
  }

  /// The memories most similar to `query`, best first, with their similarity.
  pub async fn recall(
    &self,
    query: &str,
    limit: usize,
    min_similarity: f32,
  ) -> crate::Result<Vec<(Memory, f32)>> {
    Ok(
      self
        .store
        .search(&self.collection, query, limit, min_similarity)
        .await?
        .into_iter()
        .map(|hit| (hit.document.into(), hit.similarity))
        .collect(),
    )
  }

  /// Every memory in this scope, oldest first.
  pub async fn list(&self) -> crate::Result<Vec<Memory>> {
    Ok(
      self
        .store
        .list(&self.collection)
        .await?
        .into_iter()
        .map(Memory::from)
        .collect(),
    )
  }
}

fn normalize(fact: &str) -> String {
  fact.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Shortest quote of a fact that [`MemoryStore::forget_checked`] accepts in place of the
/// whole fact.
pub const MIN_QUOTE_CHARS: usize = 12;

/// Whether `quote` is `fact`, or a long enough piece of it; see
/// [`MemoryStore::forget_checked`].
fn quotes(fact: &str, quote: &str) -> bool {
  let (fact, quote) = (
    normalize(fact).to_lowercase(),
    normalize(quote).to_lowercase(),
  );
  !quote.is_empty()
    && (quote == fact || (quote.chars().count() >= MIN_QUOTE_CHARS && fact.contains(&quote)))
}

/// What kind of credential `text` seems to contain, if any. Deliberately narrow — the
/// shapes of real keys, not every long string — so it never gets in the way of an
/// ordinary fact.
///
/// Labelled values ("`token: …`") are checked in two stages rather than as one regex:
/// for the weaker labels, requiring the value to contain a digit or to be long is what
/// separates `token: hunter2hunter2` from `uses the token bucket algorithm`, and the
/// prose form ("my password is …") needs the value told from an ordinary word. The
/// `regex` crate has no lookarounds, so those tests are written in Rust next to the
/// patterns. A `password:` / `密码：` label needs no such test.
fn looks_like_a_secret(text: &str) -> Option<&'static str> {
  struct Rule {
    pattern: Regex,
    kind: &'static str,
    /// Extra test on the captured value, for rules whose pattern is too broad on its own.
    value_must: fn(&str) -> bool,
  }

  fn any(_: &str) -> bool {
    true
  }

  /// `value` without the punctuation that ends a sentence around it.
  fn bare(value: &str) -> &str {
    value.trim_matches(|c: char| !c.is_alphanumeric())
  }

  /// English words that follow "password is" in ordinary sentences rather than being the
  /// password. Short on purpose: it only has to cover what a fact about *having* a
  /// password says, and a miss costs a refused fact, not a leaked one.
  fn not_a_plain_word(value: &str) -> bool {
    const PLAIN: [&str; 14] = [
      "required",
      "needed",
      "optional",
      "hashed",
      "stored",
      "managed",
      "changed",
      "reset",
      "expired",
      "empty",
      "unknown",
      "weak",
      "strong",
      "encrypted",
    ];
    let value = bare(value).to_ascii_lowercase();
    !PLAIN.contains(&value.as_str())
  }

  /// A value written in ASCII, which a password usually is and a Chinese explanation
  /// ("密码是公司统一管理的") never is.
  fn ascii_value(value: &str) -> bool {
    let value = bare(value);
    value.chars().count() >= 4 && value.is_ascii()
  }

  /// Values that look like a secret rather than a word: long, or carrying a digit.
  fn secret_shaped(value: &str) -> bool {
    let long_enough = value.chars().count() >= 12;
    long_enough || value.chars().any(|c| c.is_ascii_digit())
  }

  /// Where a label may start: the beginning, or after anything that is not a letter or a
  /// digit. Not `\b`, which does not hold between `_` and a letter — `DB_PASSWORD` and
  /// `client_secret` are the labels people actually have — and not "anything but a
  /// letter", which would read the `1Password` product name as a `password` label.
  const LABEL_START: &str = r"(?:^|[^A-Za-z0-9])";

  static RULES: LazyLock<Vec<Rule>> = LazyLock::new(|| {
    // The value, where a rule has one, is the named group `value`.
    let labelled = |labels: &str, separator: &str, value: &str| {
      format!(r"(?i){LABEL_START}(?:{labels})\s*{separator}\s*(?P<value>{value})")
    };
    [
      (
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----".to_owned(),
        "a private key",
        any as fn(&str) -> bool,
      ),
      (r"\bsk-[A-Za-z0-9_-]{16,}".to_owned(), "an API key", any),
      (
        r"\b[sr]k_(?:live|test)_[A-Za-z0-9]{16,}".to_owned(),
        "a Stripe key",
        any,
      ),
      (
        r"\bAIza[0-9A-Za-z_-]{35}".to_owned(),
        "a Google API key",
        any,
      ),
      (r"\bnpm_[A-Za-z0-9]{36}".to_owned(), "an npm token", any),
      (
        r"(?i)\bAKIA[0-9A-Z]{16}\b".to_owned(),
        "an AWS access key",
        any,
      ),
      (
        r"\bgh[pousr]_[A-Za-z0-9]{20,}".to_owned(),
        "a GitHub token",
        any,
      ),
      (
        r"\bxox[abprs]-[A-Za-z0-9-]{10,}".to_owned(),
        "a Slack token",
        any,
      ),
      (
        r"\bglpat-[A-Za-z0-9_-]{16,}".to_owned(),
        "a GitLab token",
        any,
      ),
      (
        r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.".to_owned(),
        "a JSON web token",
        any,
      ),
      (
        r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{20,}=*".to_owned(),
        "a bearer token",
        any,
      ),
      // `scheme://user:password@host`: the password is part of an address that is
      // routinely pasted around as if it were not secret.
      (
        r"(?i)\b[a-z][a-z0-9+.-]*://[^\s/:@]+:(?P<value>[^\s/@]+)@".to_owned(),
        "a URL with a password in it",
        any,
      ),
      // A password label followed by `:` or `=` is a password whatever the value looks
      // like: `password: swordfish` is no less one for being short and all letters, and a
      // refused fact only costs the model a rephrase. `pwd` is only taken with `=`, since
      // `pwd: …` is as often the shell command. Chinese labels likewise.
      (
        labelled("password|passwd|passphrase", "[:=]", r"\S{4,}"),
        "a password",
        any,
      ),
      (labelled("pwd", "=", r"\S{4,}"), "a password", any),
      (
        r"(?P<label>密码|口令)\s*[:=：]\s*(?P<value>\S{3,})".to_owned(),
        "a password",
        any,
      ),
      // The prose form — "my password is …" — is how people actually say it. There the
      // value has to be told from an ordinary word ("the password is required"), which
      // is what `not_a_plain_word` does for English and `ascii_value` for Chinese, where
      // an explanation is written in CJK and a password usually is not.
      (
        format!(
          r"(?i){LABEL_START}(?:password|passwd|passphrase)\s+(?:is|was)\s+(?P<value>\S{{4,}})"
        ),
        "a password",
        not_a_plain_word,
      ),
      (
        r"(?P<label>密码|口令)\s*是\s*(?P<value>\S{4,})".to_owned(),
        "a password",
        ascii_value,
      ),
      // Other labels are more often ordinary words ("the token bucket", "secret sauce"),
      // so the value has to look like a secret; see `secret_shaped`.
      (
        labelled(r"api[_ -]?key|secret|token|credential", "[:=]", r"\S{6,}"),
        "a key or token",
        secret_shaped,
      ),
      (
        r"(?P<label>密钥|令牌|凭证)\s*(?:[:=：]|是)\s*(?P<value>\S{4,})".to_owned(),
        "a key or token",
        secret_shaped,
      ),
    ]
    .into_iter()
    .map(|(pattern, kind, value_must)| Rule {
      pattern: Regex::new(&pattern).expect("a valid pattern"),
      kind,
      value_must,
    })
    .collect()
  });

  RULES.iter().find_map(|rule| {
    // Every match, not just the first: "token: bucket, password: hunter2024" must not be
    // waved through because its first label had an ordinary word after it.
    rule
      .pattern
      .captures_iter(text)
      .any(|captures| {
        // Rules without a value group leave this empty, and `any` covers them.
        let value = captures
          .name("value")
          .map(|value| value.as_str())
          .unwrap_or("");
        (rule.value_must)(value)
      })
      .then_some(rule.kind)
  })
}

#[cfg(test)]
pub(crate) mod test_support {
  use std::sync::Arc;

  use super::MemoryStore;
  use crate::knowledge_base::{embed::Embedder, store::VectorStore, testing::WordEmbedder};

  pub fn memory(scope: &str) -> (MemoryStore, Arc<WordEmbedder>) {
    let embedder = Arc::new(WordEmbedder::new());
    let store = VectorStore::open_in_memory(Arc::clone(&embedder) as Arc<dyn Embedder>).unwrap();
    (MemoryStore::new(store, scope), embedder)
  }
}

#[cfg(test)]
mod tests {
  use super::{test_support::memory, *};
  use crate::knowledge_base::testing::WordEmbedder;

  #[tokio::test]
  async fn a_fact_is_remembered_once_and_recalled_by_meaning() {
    let (memory, _) = memory("alice");
    let first = memory
      .remember("  Alice prefers   tea over coffee ")
      .await
      .unwrap();
    let Remembered::New(saved) = first else {
      panic!("expected a new memory, got {first:?}");
    };
    assert_eq!(saved.text, "Alice prefers tea over coffee");
    assert!(matches!(
      memory.remember("Alice prefers tea over coffee").await.unwrap(),
      Remembered::Existing(existing) if existing.id == saved.id
    ));
    memory
      .remember("The project deadline is Friday")
      .await
      .unwrap();

    let recalled = memory.recall("does she like tea", 1, 0.0).await.unwrap();
    assert_eq!(recalled[0].0.id, saved.id);
    assert_eq!(memory.list().await.unwrap().len(), 2);
  }

  #[tokio::test]
  async fn scopes_do_not_see_each_others_memories() {
    let (alice, _) = memory("alice");
    let bob = MemoryStore::new(alice.vector_store().clone(), "bob");
    let Remembered::New(saved) = alice
      .remember("Alice is allergic to peanuts")
      .await
      .unwrap()
    else {
      panic!("expected a new memory");
    };

    assert!(
      bob
        .recall("peanuts allergy", 5, -1.0)
        .await
        .unwrap()
        .is_empty()
    );
    assert!(bob.list().await.unwrap().is_empty());
    assert!(
      bob.forget(saved.id).await.unwrap().is_none(),
      "an id from another scope must not reach it"
    );
    assert_eq!(alice.list().await.unwrap().len(), 1);
  }

  #[tokio::test]
  async fn forget_removes_exactly_one_memory() {
    let (memory, _) = memory("s");
    let Remembered::New(saved) = memory.remember("likes jazz").await.unwrap() else {
      panic!("expected a new memory");
    };
    assert_eq!(
      memory.forget(saved.id).await.unwrap().unwrap().text,
      "likes jazz"
    );
    assert!(memory.forget(saved.id).await.unwrap().is_none());
    assert!(memory.list().await.unwrap().is_empty());
  }

  #[tokio::test]
  async fn blank_oversized_and_secret_facts_are_refused() {
    let (memory, embedder) = memory("s");
    assert!(memory.remember("   ").await.is_err());
    let long = "x".repeat(MAX_FACT_CHARS + 1);
    assert!(
      memory
        .remember(&long)
        .await
        .unwrap_err()
        .to_string()
        .contains("at most")
    );
    for secret in [
      "my key is sk-abcdefghijklmnopqrstuvwx",
      "password: hunter2hunter2",
      "AKIAABCDEFGHIJKLMNOP is the deploy key",
      "-----BEGIN OPENSSH PRIVATE KEY-----",
    ] {
      let err = memory.remember(secret).await.unwrap_err().to_string();
      assert!(
        err.contains("plain text"),
        "`{secret}` was not refused: {err}"
      );
    }
    assert_eq!(embedder.calls(), 0, "nothing refused is ever embedded");
  }

  #[test]
  fn ordinary_facts_are_not_mistaken_for_secrets() {
    for fact in [
      "Uses 1Password: great app",
      "Server runs at https://example.com:8443/api",
      "Clones with ssh://git@github.com:22/org/repo",
      "Runs pwd: prints the working directory",
      "The user's password manager is 1Password",
      "Uses the token bucket algorithm for rate limiting",
      "Prefers sk-learn over statsmodels",
      "Works at Acme as a security engineer",
      "Sends the auth token in a header",
      "密码由公司统一管理",
      "The password is required for every login",
      "The password policy: rotate every 90 days",
      "密码是公司统一管理的",
    ] {
      assert_eq!(looks_like_a_secret(fact), None, "`{fact}`");
    }
  }

  /// The shapes an ordinary user is likely to actually type, including in Chinese.
  #[test]
  fn secrets_in_other_shapes_are_recognized() {
    for secret in [
      "Bearer abcdefghijklmnopqrstuvwxyz012345",
      "my gitlab token is glpat-abcdefghijklmnop",
      "密码是 Hunter2024",
      "数据库口令：s3cr3tvalue",
      "aws key: AKIAABCDEFGHIJKLMNOP",
      // Short, all letters, no digit: what the length-or-digit test used to wave through.
      "password: swordfish",
      "my password is swordfish",
      "Wifi PASSWORD = opensesame",
      "密码是 swordfish",
      // The labels people really have: a prefix before the word, not the bare word.
      "DB_PASSWORD=hunter2hunter2",
      "client_secret: abcdef123456",
      "export ACCESS_TOKEN=abc123def456ghi",
      "pwd=letmein99",
      // Key shapes of other providers.
      "stripe key sk_live_abcdefghijklmnop1234",
      "maps key AIzaSyA-abcdefghijklmnopqrstuvwxyz01234",
      "npm_abcdefghijklmnopqrstuvwxyz0123456789 is the publish token",
      // A password inside an address.
      "connect with postgres://app:s3cretpw@db.internal/orders",
    ] {
      assert!(
        looks_like_a_secret(secret).is_some(),
        "`{secret}` should have been refused"
      );
    }
  }

  /// The reported bug: a fact saved under one embedding model answered "already
  /// remembered" while no recall could ever return it.
  #[tokio::test]
  async fn a_fact_remembered_under_another_model_is_remembered_again_not_claimed() {
    let dir = crate::tools::testing::TempDir::new("memory-reindex");
    let path = dir.join("memory.db");
    let old = MemoryStore::open(&path, Arc::new(WordEmbedder::named("old-model")), "s").unwrap();
    old.remember("Likes green tea").await.unwrap();

    let new = MemoryStore::open(&path, Arc::new(WordEmbedder::named("new-model")), "s").unwrap();
    assert_eq!(new.stale_count().await.unwrap(), 1);
    assert!(
      new
        .recall("green tea", 5, f32::MIN)
        .await
        .unwrap()
        .is_empty(),
      "an old model's memory is not recallable"
    );

    assert!(matches!(
      new.remember("Likes green tea").await.unwrap(),
      Remembered::Existing(_)
    ));
    assert_eq!(
      new.stale_count().await.unwrap(),
      0,
      "answering `Existing` has to leave it recallable"
    );
    assert_eq!(new.recall("green tea", 5, f32::MIN).await.unwrap().len(), 1);
  }

  /// A new fact is stored with the timestamp it will be listed with, not a second one
  /// taken a few milliseconds later.
  #[tokio::test]
  async fn a_new_memory_reports_the_timestamp_it_was_stored_with() {
    let (memory, _) = memory("s");
    let Remembered::New(saved) = memory.remember("likes jazz").await.unwrap() else {
      panic!("expected a new memory");
    };
    assert_eq!(memory.list().await.unwrap()[0].created_at, saved.created_at);
  }

  #[tokio::test]
  async fn a_write_changes_the_revision() {
    let (memory, _) = memory("s");
    let before = memory.revision().await.unwrap();
    memory.remember("likes jazz").await.unwrap();
    let after = memory.revision().await.unwrap();
    assert_ne!(before, after);

    assert_eq!(
      memory.revision().await.unwrap(),
      after,
      "reading changes nothing"
    );
  }

  /// The revision is the scope's own: another scope in the same file writing does not
  /// make a cache derived from this one look stale, and this one writing does.
  #[tokio::test]
  async fn a_revision_belongs_to_one_scope() {
    let (alice, _) = memory("alice");
    let bob = MemoryStore::new(alice.vector_store().clone(), "bob");
    let before = alice.revision().await.unwrap();

    bob.remember("Bob likes jazz").await.unwrap();
    assert_eq!(alice.revision().await.unwrap(), before, "bob's write");

    alice.remember("Alice likes tea").await.unwrap();
    assert_ne!(alice.revision().await.unwrap(), before, "alice's own write");
  }

  /// Another handle on the same file is noticed too, which is what a process sharing the
  /// database looks like from here.
  #[tokio::test]
  async fn a_write_through_another_handle_moves_the_revision() {
    let dir = crate::tools::testing::TempDir::new("memory-revision");
    let path = dir.join("memory.db");
    let first = MemoryStore::open(&path, Arc::new(WordEmbedder::new()), "s").unwrap();
    let second = MemoryStore::open(&path, Arc::new(WordEmbedder::new()), "s").unwrap();
    let before = first.revision().await.unwrap();

    second.remember("Likes green tea").await.unwrap();

    assert_ne!(first.revision().await.unwrap(), before);
  }

  #[tokio::test]
  async fn forget_checked_deletes_only_a_memory_that_says_what_was_expected() {
    let (memory, _) = memory("s");
    let Remembered::New(tea) = memory.remember("Alice prefers green tea").await.unwrap() else {
      panic!("expected a new memory");
    };

    // Wrong text: reports what is actually there, deletes nothing.
    let mismatch = memory
      .forget_checked(tea.id, "Bob lives in Lisbon")
      .await
      .unwrap();
    assert!(matches!(&mismatch, Forgotten::Mismatch(found) if found.text == tea.text));
    assert_eq!(memory.count().await.unwrap(), 1);

    // Right text, loosely written.
    let deleted = memory
      .forget_checked(tea.id, "  alice PREFERS green   tea ")
      .await
      .unwrap();
    assert!(matches!(&deleted, Forgotten::Deleted(found) if found.id == tea.id));
    assert_eq!(memory.count().await.unwrap(), 0);

    assert_eq!(
      memory
        .forget_checked(tea.id, "Alice prefers green tea")
        .await
        .unwrap(),
      Forgotten::Missing
    );
  }

  #[test]
  fn a_quote_must_be_the_fact_or_a_long_enough_piece_of_it() {
    let fact = "Works on Project Atlas, deploying from the release branch";
    assert!(quotes(fact, fact));
    assert!(quotes(fact, "project atlas, deploying"), "a real quote");
    assert!(quotes(
      fact,
      "  WORKS on   project ATLAS, deploying from the release branch "
    ));

    assert!(!quotes(fact, ""), "nothing");
    assert!(!quotes(fact, "   "), "nothing, spaced");
    assert!(!quotes(fact, "Atlas"), "too short to say anything");
    assert!(!quotes(fact, "release"), "too short to say anything");
    assert!(!quotes(fact, "Works on Project Borealis"), "not in it");
    assert!(
      !quotes("Atlas", "Works on Project Atlas, deploying"),
      "longer than the fact is not a quote of it"
    );
    assert_eq!("exactly 12 c".chars().count(), MIN_QUOTE_CHARS);
    assert!(quotes("say exactly 12 c ok", "exactly 12 c"));
    assert!(!quotes("say exactly 12 c ok", "exactly 12"));
  }

  #[tokio::test]
  async fn count_does_not_load_the_memories() {
    let (memory, _) = memory("s");
    assert_eq!(memory.count().await.unwrap(), 0);
    memory.remember("likes jazz").await.unwrap();
    memory.remember("likes tea").await.unwrap();
    assert_eq!(memory.count().await.unwrap(), 2);
  }

  #[test]
  fn a_memory_shows_its_date() {
    let memory = Memory {
      id: 1,
      text: "t".to_owned(),
      created_at: "2026-10-21T08:00:00+00:00".to_owned(),
    };
    assert_eq!(memory.date(), "2026-10-21");
  }
}
