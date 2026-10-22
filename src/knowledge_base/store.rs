//! [`VectorStore`]: chunks of text and their embeddings, persisted in SQLite, with
//! nearest-neighbour search over them.
//!
//! # Layout
//!
//! One table of documents. A document is one chunk of text with its vector, and belongs
//! to a *collection* (a corpus searched as a unit: `docs`, `memory:local`, ...) and a
//! *source* within it (the file, page or note it came from). A source is the unit of
//! update: [`VectorStore::put_source`] replaces all of a source's chunks at once, in one
//! transaction, so a search never sees half of an old version and half of a new one.
//!
//! A document's `id` is never handed out twice, even after the document is deleted
//! (`AUTOINCREMENT`): a caller that remembers an id — a memory number, most visibly — can
//! never find it pointing at something else later.
//!
//! # Embeddings are computed once
//!
//! Re-indexing a source that barely changed is the normal case — a document edited in
//! one paragraph. Before calling the embedder, [`VectorStore::put_source`] looks every
//! chunk up by its text (any collection, same model) and reuses the vector already
//! stored; only text the store has never seen is sent. [`IndexReport`] says how many of
//! each.
//!
//! # Models
//!
//! Every vector is tagged with the [`Embedder::model`] that produced it, and a search
//! only ever compares vectors with the store's current model — see
//! [`crate::knowledge_base::embed`] for why. Switching `EMBED_MODEL` therefore makes
//! older documents invisible to search until they are re-indexed; they are not deleted,
//! and a warning says how many there are. [`VectorStore::reindex`] does that re-indexing
//! in place, keeping every document's `id`, so whatever refers to one by id (a memory
//! number, most visibly) keeps working across a model switch.
//!
//! # On disk
//!
//! One SQLite file, created owner-read/write only when it does not exist yet — see
//! [`VectorStore::open`]. Its contents are whatever the caller put there, in plain text.
//! A file written by an earlier layout (`schema_version` 1) is upgraded in place on open.
//!
//! # Search
//!
//! Exact (brute-force cosine), not approximate: each collection's vectors are loaded
//! once, normalized, and kept in memory, so a search is one pass of dot products —
//! milliseconds for tens of thousands of chunks. For corpora in the millions, use a
//! dedicated vector database instead.
//!
//! The in-memory copy costs `4 × dimensions` bytes per document — about 6 KB at 1536
//! dimensions, so roughly 300 MB for 50 000 chunks — and is the thing to size by.
//!
//! # Knowing when the copy is stale
//!
//! Every collection has a version: a counter in the `meta` table that each write bumps
//! *inside the transaction that makes the change*. A cached copy is good exactly while its
//! version equals the stored one, which is how a write by this store, by another handle
//! on the file, or by another process is noticed — with nothing else to keep in step, and
//! without a write to one collection throwing away the copies of the others.
//!
//! A copy is read in pages, so a large rebuild never holds the connection (and with it
//! every other call) for long; the version is read before the first page and again after
//! the last, and a copy that a write interleaved with is read again rather than cached.
//! Only one caller rebuilds a collection at a time; the others wait and then use the result.
//!
//! # Concurrency
//!
//! SQLite in WAL mode, so readers do not block the writer and several processes can share
//! one file. Every database call runs on tokio's blocking pool, never on an async worker.

use std::{
  collections::HashMap,
  path::Path,
  sync::{Arc, Mutex, MutexGuard},
  time::Duration,
};

use anyhow::Context;
use rusqlite::{Connection, OptionalExtension, Row, TransactionBehavior, params};
use serde_json::Value;

use crate::{
  knowledge_base::{
    chunk::fixed_length_chunking,
    embed::Embedder,
    search::{normalized, top_k},
  },
  util::fnv1a64_hex,
};

/// Bumped when the table layout changes incompatibly. A file from an older version this
/// build knows how to upgrade is upgraded on open (see [`MIGRATE_V1_TO_V2`]); one from any
/// other version is refused rather than misread.
///
/// 1: `id INTEGER PRIMARY KEY`, so the id of a deleted last row could be handed out again.
/// 2: `AUTOINCREMENT`, and a version counter per collection in `meta`.
const SCHEMA_VERSION: &str = "2";

/// The `documents` table's columns, shared by creating it and by copying rows into the
/// new one during an upgrade.
macro_rules! documents_table {
  () => {
    "(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  collection TEXT NOT NULL,
  source TEXT NOT NULL,
  chunk_index INTEGER NOT NULL,
  text TEXT NOT NULL,
  text_hash TEXT NOT NULL,
  metadata TEXT NOT NULL,
  created_at TEXT NOT NULL,
  model TEXT NOT NULL,
  dim INTEGER NOT NULL,
  embedding BLOB NOT NULL,
  UNIQUE (collection, source, chunk_index)
)"
  };
}

macro_rules! indexes {
  () => {
    "
CREATE INDEX IF NOT EXISTS documents_by_collection ON documents (collection, model);
CREATE INDEX IF NOT EXISTS documents_by_text ON documents (model, text_hash);
"
  };
}

const SCHEMA: &str = concat!(
  "
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents ",
  documents_table!(),
  ";",
  indexes!(),
);

/// Rebuild `documents` with `AUTOINCREMENT`, keeping every row and its id. SQLite cannot
/// add the keyword to an existing table, hence the copy; it runs in the same transaction
/// that reads the version, so a second process opening the file at the same moment waits
/// and then finds it already done.
const MIGRATE_V1_TO_V2: &str = concat!(
  "
DROP INDEX IF EXISTS documents_by_collection;
DROP INDEX IF EXISTS documents_by_text;
ALTER TABLE documents RENAME TO documents_v1;
CREATE TABLE documents ",
  documents_table!(),
  ";
INSERT INTO documents
  (id, collection, source, chunk_index, text, text_hash, metadata, created_at, model, dim,
   embedding)
SELECT id, collection, source, chunk_index, text, text_hash, metadata, created_at, model, dim,
       embedding
FROM documents_v1;
DROP TABLE documents_v1;",
  indexes!(),
);

/// The one statement that adds a document, whoever is adding it.
const INSERT_DOCUMENT: &str = "INSERT INTO documents
  (collection, source, chunk_index, text, text_hash, metadata, created_at, model, dim,
   embedding)
VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)";

/// Rows read per page when a collection's vectors are loaded; see the module docs.
const SNAPSHOT_PAGE: usize = 1024;

/// How often a snapshot is read again because a write landed while it was being read,
/// before the freshest copy read is served without being cached.
const MAX_SNAPSHOT_ATTEMPTS: usize = 3;

const DOCUMENT_COLUMNS: &str = "id, collection, source, chunk_index, text, metadata, created_at";

/// How long a write waits for another process's write to finish before failing.
const BUSY_TIMEOUT: Duration = Duration::from_secs(5);

/// Bring a freshly opened connection to the current layout: WAL, tables, and the schema
/// version — written for a new file, upgraded for an old one, refused for an unknown one.
///
/// Safe to run again after it fails part-way, which [`retry_while_busy`] relies on: every
/// step is idempotent, and the version check and what follows from it are one write
/// transaction, taken up front, so processes opening the same file at the same moment
/// queue there instead of all finding "no version yet" (or "version 1") and all acting on it.
fn initialize(conn: &mut Connection, location: &str) -> anyhow::Result<()> {
  // Returns the mode it ended up in (`memory` for an in-memory database), hence a query.
  conn.query_row("PRAGMA journal_mode = WAL", [], |row| {
    row.get::<_, String>(0)
  })?;
  conn.execute_batch(SCHEMA)?;
  let transaction = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
  let version: Option<String> = transaction
    .query_row(
      "SELECT value FROM meta WHERE key = 'schema_version'",
      [],
      |row| row.get(0),
    )
    .optional()?;
  match version.as_deref() {
    None => {
      transaction.execute(
        "INSERT INTO meta (key, value) VALUES ('schema_version', ?1)",
        [SCHEMA_VERSION],
      )?;
    }
    Some("1") => {
      transaction.execute_batch(MIGRATE_V1_TO_V2)?;
      transaction.execute(
        "UPDATE meta SET value = ?1 WHERE key = 'schema_version'",
        [SCHEMA_VERSION],
      )?;
    }
    Some(version) if version == SCHEMA_VERSION => {}
    Some(version) => anyhow::bail!(
      "the vector store `{location}` has schema version {version}, but this build reads \
       version {SCHEMA_VERSION}"
    ),
  }
  transaction.commit()?;
  Ok(())
}

/// Run `attempt` again for as long as it fails because the database is busy, up to
/// [`BUSY_TIMEOUT`].
///
/// SQLite's own busy handler (`busy_timeout`) covers most contention, but not all of it:
/// switching a file to WAL while another connection is doing the same, and a read
/// upgrading to a write on a snapshot another writer has since moved past, fail at once
/// with "database is locked" however long the timeout is — retrying is the documented
/// answer. Several processes opening one new file together is exactly when it happens.
fn retry_while_busy<T>(mut attempt: impl FnMut() -> anyhow::Result<T>) -> anyhow::Result<T> {
  let deadline = std::time::Instant::now() + BUSY_TIMEOUT;
  loop {
    match attempt() {
      Err(err) if is_busy(&err) && std::time::Instant::now() < deadline => {
        std::thread::sleep(Duration::from_millis(15));
      }
      outcome => return outcome,
    }
  }
}

fn is_busy(err: &anyhow::Error) -> bool {
  err
    .downcast_ref::<rusqlite::Error>()
    .and_then(rusqlite::Error::sqlite_error_code)
    .is_some_and(|code| {
      matches!(
        code,
        rusqlite::ErrorCode::DatabaseBusy | rusqlite::ErrorCode::DatabaseLocked
      )
    })
}

/// One stored chunk.
#[derive(Debug, Clone, PartialEq)]
pub struct Document {
  /// Stable for the document's lifetime, and never handed out again once the document is
  /// gone; a re-indexed source gets new ids.
  pub id: i64,
  pub collection: String,
  pub source: String,
  /// Position within its source, from 0.
  pub chunk_index: usize,
  pub text: String,
  /// Whatever the caller attached when indexing; `{}` when nothing was.
  pub metadata: Value,
  /// RFC 3339, UTC.
  pub created_at: String,
}

/// A [`Document`] and how similar it is to the query, from -1 to 1.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredDocument {
  pub document: Document,
  pub similarity: f32,
}

/// What [`VectorStore::insert_unique`] did.
#[derive(Debug, Clone, PartialEq)]
pub enum Inserted {
  /// Stored as a new document.
  New(Document),
  /// The collection already held exactly this text; nothing changed.
  Existing(Document),
  /// The collection already holds this many documents, the limit; nothing was stored.
  Full(usize),
}

/// What [`VectorStore::delete_if`] did.
#[derive(Debug, Clone, PartialEq)]
pub enum DeleteOutcome {
  /// The document was there, the caller's check accepted it, and it is gone.
  Deleted(Document),
  /// The document is there but the caller's check rejected it; nothing was deleted.
  Refused(Document),
  /// There is no such document in the collection.
  Missing,
}

/// What [`VectorStore::put_source`] did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IndexReport {
  /// Ids of the chunks now stored for the source, in chunk order.
  pub ids: Vec<i64>,
  /// Distinct texts sent to the embedder.
  pub embedded: usize,
  /// Chunks whose vector was already in the store.
  pub reused: usize,
}

/// See the module docs. Cheap to clone: clones share one connection and one cache.
#[derive(Clone)]
pub struct VectorStore {
  inner: Arc<Inner>,
}

impl std::fmt::Debug for VectorStore {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("VectorStore")
      .field("location", &self.inner.location)
      .field("model", &self.inner.embedder.model())
      .finish_non_exhaustive()
  }
}

struct Inner {
  conn: Mutex<Connection>,
  embedder: Arc<dyn Embedder>,
  location: String,
  /// Per collection, the normalized vectors of the current model; see the module docs.
  cache: Mutex<HashMap<String, Arc<Snapshot>>>,
  /// Held while a snapshot is rebuilt, so several callers that all find the cache stale
  /// do the work once instead of each doing it.
  rebuild: Mutex<()>,
}

/// One collection's searchable vectors, as of its `version`.
struct Snapshot {
  version: i64,
  ids: Vec<i64>,
  dim: usize,
  /// `ids.len() * dim` floats, row after row, each row of unit length.
  vectors: Vec<f32>,
}

impl VectorStore {
  /// Open (or create) the store at `path`, embedding with `embedder`. Creates the parent
  /// directory if needed.
  ///
  /// A file created here is owner-read/write only: it holds plain text the caller decided
  /// to keep, which for [`crate::memory`] is what the user asked the agent to remember
  /// about them, and the default umask (commonly `0644`) would make that readable by
  /// every account on the machine. An existing file's permissions are left alone — that
  /// is the owner's decision, and SQLite would have to be told anyway.
  ///
  /// The mode is a Unix one: on other platforms the file gets whatever the directory it
  /// is created in grants, so keep it somewhere only the intended account can read.
  ///
  /// # Errors
  ///
  /// When the file cannot be opened or created, or was written by an incompatible
  /// version of this crate.
  pub fn open(path: impl AsRef<Path>, embedder: Arc<dyn Embedder>) -> anyhow::Result<Self> {
    let path = path.as_ref();
    if let Some(parent) = path
      .parent()
      .filter(|parent| !parent.as_os_str().is_empty())
    {
      std::fs::create_dir_all(parent)
        .with_context(|| format!("failed to create `{}`", parent.display()))?;
    }
    create_private(path).with_context(|| format!("failed to create `{}`", path.display()))?;
    let conn = Connection::open(path)
      .with_context(|| format!("failed to open the vector store `{}`", path.display()))?;
    Self::from_connection(conn, path.display().to_string(), embedder)
  }

  /// A store that lives only as long as this value — for tests and throwaway indexes.
  pub fn open_in_memory(embedder: Arc<dyn Embedder>) -> anyhow::Result<Self> {
    Self::from_connection(
      Connection::open_in_memory()?,
      ":memory:".to_owned(),
      embedder,
    )
  }

  fn from_connection(
    mut conn: Connection,
    location: String,
    embedder: Arc<dyn Embedder>,
  ) -> anyhow::Result<Self> {
    conn.busy_timeout(BUSY_TIMEOUT)?;
    retry_while_busy(|| initialize(&mut conn, &location))?;
    Ok(Self {
      inner: Arc::new(Inner {
        conn: Mutex::new(conn),
        embedder,
        location,
        cache: Mutex::new(HashMap::new()),
        rebuild: Mutex::new(()),
      }),
    })
  }

  /// Where the store lives: a file path, or `:memory:`.
  pub fn location(&self) -> &str {
    &self.inner.location
  }

  /// The embedder documents and queries are embedded with.
  pub fn embedder(&self) -> &Arc<dyn Embedder> {
    &self.inner.embedder
  }

  /// Replace everything stored for `source` in `collection` with `chunks`, each carrying
  /// `metadata`. Blank chunks are skipped; an empty list removes the source.
  ///
  /// Vectors already in the store for the same text and model are reused rather than
  /// recomputed (see the module docs). The replacement is atomic: until it commits,
  /// searches see the old chunks.
  pub async fn put_source(
    &self,
    collection: &str,
    source: &str,
    chunks: Vec<String>,
    metadata: Value,
  ) -> anyhow::Result<IndexReport> {
    let chunks: Vec<String> = chunks
      .into_iter()
      .filter(|chunk| !chunk.trim().is_empty())
      .collect();
    let model = self.inner.embedder.model().to_owned();
    let hashes: Vec<String> = chunks.iter().map(|chunk| fnv1a64_hex(chunk)).collect();

    let mut vectors: Vec<Option<Vec<f32>>> = self
      .blocking({
        let (model, chunks, hashes) = (model.clone(), chunks.clone(), hashes.clone());
        move |inner| inner.known_vectors(&model, &chunks, &hashes)
      })
      .await?;
    let reused = vectors.iter().filter(|vector| vector.is_some()).count();

    // Each distinct unknown text is embedded once, however often it repeats.
    let mut missing: Vec<String> = Vec::new();
    let mut slot_of: HashMap<&str, usize> = HashMap::new();
    for (chunk, vector) in chunks.iter().zip(&vectors) {
      if vector.is_none() && !slot_of.contains_key(chunk.as_str()) {
        slot_of.insert(chunk, missing.len());
        missing.push(chunk.clone());
      }
    }
    if !missing.is_empty() {
      let fresh = self.inner.embedder.embed_checked(&missing).await?;
      for (chunk, vector) in chunks.iter().zip(vectors.iter_mut()) {
        if vector.is_none() {
          *vector = Some(fresh[slot_of[chunk.as_str()]].clone());
        }
      }
    }
    let vectors: Vec<Vec<f32>> = vectors.into_iter().flatten().collect();
    ensure_usable(&vectors)?;

    let metadata = serde_json::to_string(&metadata)?;
    let (collection_owned, source) = (collection.to_owned(), source.to_owned());
    let ids = self
      .blocking(move |inner| {
        inner.replace_source(
          &collection_owned,
          &source,
          &chunks,
          &hashes,
          &vectors,
          &metadata,
          &model,
        )
      })
      .await?;
    Ok(IndexReport {
      ids,
      embedded: missing.len(),
      reused,
    })
  }

  /// Store `text` as a one-chunk document of its own `source`, unless `collection` already
  /// holds exactly that text or already holds `max_documents` documents.
  ///
  /// The duplicate and size checks and the insert are one transaction, taken with
  /// SQLite's write lock up front: two callers — two tool calls, or two processes sharing
  /// the file — cannot both see "not there yet, and room left" and both insert. (Checking
  /// first and inserting later, as separate calls, is exactly that race.) The embedding
  /// is computed before the transaction, not inside it, so the write lock is not held
  /// across a network call.
  ///
  /// The vector already stored for the same text and model is reused, as in
  /// [`Self::put_source`].
  pub async fn insert_unique(
    &self,
    collection: &str,
    source: &str,
    text: &str,
    metadata: Value,
    max_documents: usize,
  ) -> anyhow::Result<Inserted> {
    // Already there: no reason to embed anything.
    if let Some(existing) = self.find_text(collection, text).await? {
      return Ok(Inserted::Existing(existing));
    }
    let model = self.inner.embedder.model().to_owned();
    let hash = fnv1a64_hex(text);
    let known = self
      .blocking({
        let (model, texts, hashes) = (model.clone(), vec![text.to_owned()], vec![hash.clone()]);
        move |inner| inner.known_vectors(&model, &texts, &hashes)
      })
      .await?
      .pop()
      .flatten();
    let vector = match known {
      Some(vector) => vector,
      None => self.inner.embedder.embed_one(text).await?,
    };
    ensure_usable(std::slice::from_ref(&vector))?;

    let metadata = serde_json::to_string(&metadata)?;
    let (collection, source, text) = (collection.to_owned(), source.to_owned(), text.to_owned());
    self
      .blocking(move |inner| {
        inner.insert_unique_row(
          &collection,
          &source,
          &text,
          &hash,
          &vector,
          &metadata,
          &model,
          max_documents,
        )
      })
      .await
  }

  /// [`Self::put_source`] for one long text, cut into `chunk_size`-character chunks that
  /// overlap by `overlap` characters (see
  /// [`crate::knowledge_base::chunk::fixed_length_chunking`]).
  ///
  /// # Errors
  ///
  /// When `overlap` is not smaller than `chunk_size`, or `chunk_size` is 0.
  pub async fn index_text(
    &self,
    collection: &str,
    source: &str,
    text: &str,
    chunk_size: usize,
    overlap: usize,
    metadata: Value,
  ) -> anyhow::Result<IndexReport> {
    anyhow::ensure!(
      chunk_size > 0 && overlap < chunk_size,
      "chunk_size must be positive and larger than overlap"
    );
    let chunks = fixed_length_chunking(text, chunk_size, overlap);
    self.put_source(collection, source, chunks, metadata).await
  }

  /// The `limit` documents of `collection` most similar to `query`, best first, leaving
  /// out any below `min_similarity`. Only documents embedded with the store's current
  /// model are considered (see the module docs).
  pub async fn search(
    &self,
    collection: &str,
    query: &str,
    limit: usize,
    min_similarity: f32,
  ) -> anyhow::Result<Vec<ScoredDocument>> {
    if limit == 0 || query.trim().is_empty() {
      return Ok(Vec::new());
    }
    let snapshot = self.snapshot(collection).await?;
    // Nothing to compare against: no reason to pay for embedding the query.
    if snapshot.ids.is_empty() {
      return Ok(Vec::new());
    }

    let query_vector = self.inner.embedder.embed_one(query).await?;
    anyhow::ensure!(
      query_vector.len() == snapshot.dim,
      "the query embedding has {} dimensions but the stored ones have {} — did the model \
       behind `{}` change without its name changing?",
      query_vector.len(),
      snapshot.dim,
      self.inner.embedder.model()
    );
    let Some(query_vector) = normalized(&query_vector) else {
      return Ok(Vec::new());
    };

    let scores = snapshot
      .vectors
      .chunks_exact(snapshot.dim)
      .enumerate()
      .map(|(row, vector)| {
        let dot: f32 = vector.iter().zip(&query_vector).map(|(a, b)| a * b).sum();
        (row, dot)
      })
      .filter(|(_, similarity)| *similarity >= min_similarity);
    let ranked: Vec<(i64, f32)> = top_k(scores, limit)
      .into_iter()
      .map(|(row, similarity)| (snapshot.ids[row], similarity))
      .collect();

    self
      .blocking(move |inner| {
        let conn = inner.conn();
        let mut statement = conn.prepare_cached(&format!(
          "SELECT {DOCUMENT_COLUMNS} FROM documents WHERE id = ?1"
        ))?;
        let mut hits = Vec::with_capacity(ranked.len());
        for (id, similarity) in ranked {
          // Deleted since the snapshot was taken: skipped rather than reported.
          if let Some(document) = statement.query_row([id], document_from_row).optional()? {
            hits.push(ScoredDocument {
              document,
              similarity,
            });
          }
        }
        Ok(hits)
      })
      .await
  }

  /// Every document in `collection`, oldest first.
  pub async fn list(&self, collection: &str) -> anyhow::Result<Vec<Document>> {
    let collection = collection.to_owned();
    self
      .blocking(move |inner| {
        let conn = inner.conn();
        let mut statement = conn.prepare_cached(&format!(
          "SELECT {DOCUMENT_COLUMNS} FROM documents WHERE collection = ?1 ORDER BY id"
        ))?;
        let rows = statement.query_map([&collection], document_from_row)?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
      })
      .await
  }

  /// How many documents `collection` holds, whatever model embedded them.
  pub async fn count(&self, collection: &str) -> anyhow::Result<usize> {
    let collection = collection.to_owned();
    self
      .blocking(move |inner| {
        let count: i64 = inner.conn().query_row(
          "SELECT COUNT(*) FROM documents WHERE collection = ?1",
          [&collection],
          |row| row.get(0),
        )?;
        Ok(usize::try_from(count).unwrap_or(0))
      })
      .await
  }

  /// The document in `collection` whose text is exactly `text`, if there is one.
  pub async fn find_text(&self, collection: &str, text: &str) -> anyhow::Result<Option<Document>> {
    let (collection, text) = (collection.to_owned(), text.to_owned());
    self
      .blocking(move |inner| {
        let conn = inner.conn();
        Ok(find_by_text(
          &conn,
          &collection,
          &fnv1a64_hex(&text),
          &text,
        )?)
      })
      .await
  }

  /// Delete one document of `collection`, returning it — `None` when there was no such
  /// document *in that collection*, so an id cannot reach across collections.
  pub async fn delete(&self, collection: &str, id: i64) -> anyhow::Result<Option<Document>> {
    Ok(match self.delete_if(collection, id, |_| true).await? {
      DeleteOutcome::Deleted(document) => Some(document),
      DeleteOutcome::Refused(_) | DeleteOutcome::Missing => None,
    })
  }

  /// Delete document `id` of `collection` only if `allow` accepts it.
  ///
  /// Looking the document up, asking `allow`, and deleting it are one transaction, so what
  /// `allow` approved is exactly what is deleted — for a caller that was handed an id some
  /// time ago and wants to be sure it still means what it did (see [`crate::memory`]).
  /// `allow` runs on the blocking pool while the connection is held, so it must be quick.
  pub async fn delete_if(
    &self,
    collection: &str,
    id: i64,
    allow: impl FnOnce(&Document) -> bool + Send + 'static,
  ) -> anyhow::Result<DeleteOutcome> {
    let collection = collection.to_owned();
    self
      .blocking(move |inner| {
        let outcome = {
          let mut conn = inner.conn();
          let transaction = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
          let outcome = match find_by_id(&transaction, &collection, id)? {
            None => DeleteOutcome::Missing,
            Some(document) if !allow(&document) => DeleteOutcome::Refused(document),
            Some(document) => {
              transaction.execute("DELETE FROM documents WHERE id = ?1", [id])?;
              bump_version(&transaction, &collection)?;
              DeleteOutcome::Deleted(document)
            }
          };
          transaction.commit()?;
          outcome
        };
        if matches!(outcome, DeleteOutcome::Deleted(_)) {
          inner.invalidate(&collection);
        }
        Ok(outcome)
      })
      .await
  }

  /// Delete every chunk of `source` in `collection`; returns how many there were.
  pub async fn delete_source(&self, collection: &str, source: &str) -> anyhow::Result<usize> {
    let (collection, source) = (collection.to_owned(), source.to_owned());
    self
      .blocking(move |inner| {
        let removed = {
          let mut conn = inner.conn();
          let transaction = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
          let removed = transaction.execute(
            "DELETE FROM documents WHERE collection = ?1 AND source = ?2",
            params![collection, source],
          )?;
          if removed > 0 {
            bump_version(&transaction, &collection)?;
          }
          transaction.commit()?;
          removed
        };
        if removed > 0 {
          inner.invalidate(&collection);
        }
        Ok(removed)
      })
      .await
  }

  /// The distinct sources that have documents in `collection`, sorted — for a caller that
  /// wants to find the ones that no longer exist where they came from.
  pub async fn sources(&self, collection: &str) -> anyhow::Result<Vec<String>> {
    let collection = collection.to_owned();
    self
      .blocking(move |inner| {
        let conn = inner.conn();
        let mut statement = conn.prepare_cached(
          "SELECT DISTINCT source FROM documents WHERE collection = ?1 ORDER BY source",
        )?;
        let rows = statement.query_map([&collection], |row| row.get::<_, String>(0))?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
      })
      .await
  }

  /// The collection's searchable vectors, rebuilt only when something changed.
  async fn snapshot(&self, collection: &str) -> anyhow::Result<Arc<Snapshot>> {
    let collection = collection.to_owned();
    let model = self.inner.embedder.model().to_owned();
    self
      .blocking(move |inner| inner.snapshot(&collection, &model))
      .await
  }

  /// How many documents of `collection` were embedded with a model other than the one in
  /// use, and so cannot be found by [`Self::search`] until [`Self::reindex`] runs.
  ///
  /// Zero in the ordinary case, which is what makes it cheap enough to check on startup.
  pub async fn stale_count(&self, collection: &str) -> anyhow::Result<usize> {
    let (collection, model) = (
      collection.to_owned(),
      self.inner.embedder.model().to_owned(),
    );
    self
      .blocking(move |inner| {
        let count: i64 = inner.conn().query_row(
          "SELECT COUNT(*) FROM documents WHERE collection = ?1 AND model != ?2",
          params![collection, model],
          |row| row.get(0),
        )?;
        Ok(usize::try_from(count).unwrap_or(0))
      })
      .await
  }

  /// Re-embed every document of `collection` that was embedded with another model, so
  /// that it becomes searchable again under the one in use now.
  ///
  /// Each document keeps its `id`: a re-index through [`Self::put_source`] would delete
  /// and re-insert the whole source, and every id the caller has handed out (a memory
  /// number, say) would dangle. Documents already on the current model are untouched, so
  /// calling this when there is nothing stale costs one query and no model calls.
  ///
  /// # Errors
  ///
  /// When the embedder fails, or returns a different number of vectors than it was given
  /// texts, or vectors whose length disagrees with each other. Nothing is written in that
  /// case: the collection is either fully migrated or left exactly as it was.
  pub async fn reindex(&self, collection: &str) -> anyhow::Result<usize> {
    let model = self.inner.embedder.model().to_owned();
    let pending = {
      let collection = collection.to_owned();
      let model = model.clone();
      self
        .blocking(move |inner| inner.stale_documents(&collection, &model))
        .await?
    };
    if pending.is_empty() {
      return Ok(0);
    }

    // One vector per distinct text, as in `put_source`: a collection can hold the same
    // text under several sources, and it is one vector either way.
    let mut distinct: Vec<String> = Vec::new();
    let mut slot_of: HashMap<&str, usize> = HashMap::new();
    for (_, text) in &pending {
      if !slot_of.contains_key(text.as_str()) {
        slot_of.insert(text, distinct.len());
        distinct.push(text.clone());
      }
    }
    let fresh = self.inner.embedder.embed_checked(&distinct).await?;
    ensure_usable(&fresh)?;

    // Each document's vector, looked up once here so the write below owns everything it
    // touches: `slot_of` borrows `pending`, which cannot cross into the blocking pool.
    let updates: Vec<(i64, Vec<f32>)> = pending
      .iter()
      .map(|(id, text)| (*id, fresh[slot_of[text.as_str()]].clone()))
      .collect();
    let reindexed = updates.len();
    let collection_owned = collection.to_owned();
    self
      .blocking(move |inner| inner.replace_embeddings(&collection_owned, &updates, &model))
      .await?;
    Ok(reindexed)
  }

  /// [`Self::reindex`] for one document: re-embed it if it was embedded with another
  /// model, and say whether that was needed. A document already on the current model, or
  /// not in `collection`, is left alone and `false` is returned.
  ///
  /// For a caller that has found one particular document and needs *it* to be searchable
  /// (a repeated `remember`) without paying for — or silently doing — a migration of
  /// everything else.
  pub async fn reindex_document(&self, collection: &str, id: i64) -> anyhow::Result<bool> {
    let model = self.inner.embedder.model().to_owned();
    let (collection_owned, model_owned) = (collection.to_owned(), model.clone());
    let stale_text = self
      .blocking(move |inner| {
        Ok(
          inner
            .conn()
            .query_row(
              "SELECT text FROM documents WHERE collection = ?1 AND id = ?2 AND model != ?3",
              params![collection_owned, id, model_owned],
              |row| row.get::<_, String>(0),
            )
            .optional()?,
        )
      })
      .await?;
    let Some(text) = stale_text else {
      return Ok(false);
    };

    let vector = self.inner.embedder.embed_one(&text).await?;
    ensure_usable(std::slice::from_ref(&vector))?;
    let collection = collection.to_owned();
    self
      .blocking(move |inner| inner.replace_embeddings(&collection, &[(id, vector)], &model))
      .await?;
    Ok(true)
  }

  /// The version of `collection`: a counter that goes up with every write to it, whoever
  /// makes the write (this handle, another handle on the file, another process). `0` for a
  /// collection that has never been written.
  ///
  /// Exposed so a caller that caches something derived from a collection
  /// ([`crate::memory`] does) can tell when it changed underneath — see the module docs for
  /// how the store itself uses it.
  pub async fn version(&self, collection: &str) -> anyhow::Result<i64> {
    let collection = collection.to_owned();
    self
      .blocking(move |inner| Ok(read_version(&inner.conn(), &collection)?))
      .await
  }

  async fn blocking<T, F>(&self, f: F) -> anyhow::Result<T>
  where
    T: Send + 'static,
    F: FnOnce(&Inner) -> anyhow::Result<T> + Send + 'static,
  {
    let inner = Arc::clone(&self.inner);
    tokio::task::spawn_blocking(move || f(&inner))
      .await
      .context("a vector store task panicked")?
  }
}

impl Inner {
  /// A poisoned lock still guards a valid connection: SQLite rolls back a transaction a
  /// panic interrupted.
  fn conn(&self) -> MutexGuard<'_, Connection> {
    self
      .conn
      .lock()
      .unwrap_or_else(|poisoned| poisoned.into_inner())
  }

  fn cache(&self) -> MutexGuard<'_, HashMap<String, Arc<Snapshot>>> {
    self
      .cache
      .lock()
      .unwrap_or_else(|poisoned| poisoned.into_inner())
  }

  /// Drop the cached copy of `collection` after a write through this store. Not needed for
  /// correctness — the version bumped by the write already makes the copy stale — but there
  /// is no reason to keep megabytes of vectors around that nothing can use.
  fn invalidate(&self, collection: &str) {
    self.cache().remove(collection);
  }

  /// For each chunk, the vector already stored for the same text under `model`, if any.
  fn known_vectors(
    &self,
    model: &str,
    chunks: &[String],
    hashes: &[String],
  ) -> anyhow::Result<Vec<Option<Vec<f32>>>> {
    let conn = self.conn();
    let mut statement = conn.prepare_cached(
      "SELECT text, embedding FROM documents WHERE model = ?1 AND text_hash = ?2 LIMIT 8",
    )?;
    let mut known = Vec::with_capacity(chunks.len());
    for (chunk, hash) in chunks.iter().zip(hashes) {
      let candidates = statement.query_map(params![model, hash], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?))
      })?;
      let mut found = None;
      for candidate in candidates {
        // The hash only finds candidates; the text itself decides.
        let (text, blob) = candidate?;
        if text == *chunk {
          found = decode(&blob);
          break;
        }
      }
      known.push(found);
    }
    Ok(known)
  }

  /// The transaction behind [`VectorStore::insert_unique`].
  #[allow(clippy::too_many_arguments)]
  fn insert_unique_row(
    &self,
    collection: &str,
    source: &str,
    text: &str,
    hash: &str,
    vector: &[f32],
    metadata: &str,
    model: &str,
    max_documents: usize,
  ) -> anyhow::Result<Inserted> {
    let created_at = chrono::Utc::now().to_rfc3339();
    let outcome = {
      let mut conn = self.conn();
      // `Immediate` takes the write lock now, so the checks below cannot go stale before
      // the insert: another writer waits (up to the busy timeout) rather than interleave.
      let transaction = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
      let outcome = if let Some(existing) = find_by_text(&transaction, collection, hash, text)? {
        Inserted::Existing(existing)
      } else {
        let count: i64 = transaction.query_row(
          "SELECT COUNT(*) FROM documents WHERE collection = ?1",
          [collection],
          |row| row.get(0),
        )?;
        let count = usize::try_from(count).unwrap_or(usize::MAX);
        if count >= max_documents {
          Inserted::Full(count)
        } else {
          let id = insert_document(
            &transaction,
            &NewDocument {
              collection,
              source,
              chunk_index: 0,
              text,
              hash,
              metadata,
              created_at: &created_at,
              model,
              vector,
            },
          )?;
          bump_version(&transaction, collection)?;
          Inserted::New(Document {
            id,
            collection: collection.to_owned(),
            source: source.to_owned(),
            chunk_index: 0,
            text: text.to_owned(),
            metadata: parse_metadata(metadata),
            created_at,
          })
        }
      };
      transaction.commit()?;
      outcome
    };
    if matches!(outcome, Inserted::New(_)) {
      self.invalidate(collection);
    }
    Ok(outcome)
  }

  #[allow(clippy::too_many_arguments)]
  fn replace_source(
    &self,
    collection: &str,
    source: &str,
    chunks: &[String],
    hashes: &[String],
    vectors: &[Vec<f32>],
    metadata: &str,
    model: &str,
  ) -> anyhow::Result<Vec<i64>> {
    let created_at = chrono::Utc::now().to_rfc3339();
    let ids = {
      let mut conn = self.conn();
      let transaction = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
      transaction.execute(
        "DELETE FROM documents WHERE collection = ?1 AND source = ?2",
        params![collection, source],
      )?;
      let mut ids = Vec::with_capacity(chunks.len());
      for (index, ((text, hash), vector)) in chunks.iter().zip(hashes).zip(vectors).enumerate() {
        ids.push(insert_document(
          &transaction,
          &NewDocument {
            collection,
            source,
            chunk_index: index,
            text,
            hash,
            metadata,
            created_at: &created_at,
            model,
            vector,
          },
        )?);
      }
      bump_version(&transaction, collection)?;
      transaction.commit()?;
      ids
    };
    self.invalidate(collection);
    Ok(ids)
  }

  /// The documents embedded with a model other than `model`, with the text to embed
  /// again — the input of [`VectorStore::reindex`].
  fn stale_documents(&self, collection: &str, model: &str) -> anyhow::Result<Vec<(i64, String)>> {
    let conn = self.conn();
    let mut statement = conn.prepare_cached(
      "SELECT id, text FROM documents WHERE collection = ?1 AND model != ?2 ORDER BY id",
    )?;
    let rows = statement.query_map(params![collection, model], |row| {
      Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
    })?;
    Ok(rows.collect::<Result<Vec<_>, _>>()?)
  }

  /// Point already-stored documents at new vectors and `model`, in one transaction.
  fn replace_embeddings(
    &self,
    collection: &str,
    updates: &[(i64, Vec<f32>)],
    model: &str,
  ) -> anyhow::Result<()> {
    {
      let mut conn = self.conn();
      let transaction = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
      {
        let mut update = transaction.prepare_cached(
          "UPDATE documents SET model = ?1, dim = ?2, embedding = ?3
           WHERE collection = ?4 AND id = ?5",
        )?;
        for (id, vector) in updates {
          update.execute(params![
            model,
            vector.len() as i64,
            encode(vector),
            collection,
            id
          ])?;
        }
      }
      bump_version(&transaction, collection)?;
      transaction.commit()?;
    }
    self.invalidate(collection);
    Ok(())
  }

  /// The collection's searchable vectors, rebuilt only when something changed.
  fn snapshot(&self, collection: &str, model: &str) -> anyhow::Result<Arc<Snapshot>> {
    self.snapshot_paged(collection, model, SNAPSHOT_PAGE, &mut || {})
  }

  /// The cached copy of `collection`, if it is still the current version.
  fn current_snapshot(&self, collection: &str) -> anyhow::Result<Option<Arc<Snapshot>>> {
    let version = read_version(&self.conn(), collection)?;
    Ok(
      self
        .cache()
        .get(collection)
        .filter(|snapshot| snapshot.version == version)
        .cloned(),
    )
  }

  /// [`Self::snapshot`] reading `page` rows at a time. `between` runs after every page
  /// has been read, with no lock held — the seam a test uses to land a write in the
  /// middle of a rebuild.
  ///
  /// The version is read before the first page and after the last. Equal means no write
  /// committed in between (every write bumps it in its own transaction), so the pages add
  /// up to one consistent state and are cached under that version. Different means the
  /// pages may mix two states, so they are read again — and if writers keep winning, the
  /// freshest copy read is served once without being cached, rather than looping.
  fn snapshot_paged(
    &self,
    collection: &str,
    model: &str,
    page: usize,
    between: &mut dyn FnMut(),
  ) -> anyhow::Result<Arc<Snapshot>> {
    if let Some(current) = self.current_snapshot(collection)? {
      return Ok(current);
    }
    // One rebuild at a time: callers that all found the cache stale queue here, and the
    // ones after the first find its work already done.
    let _rebuilding = self
      .rebuild
      .lock()
      .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(current) = self.current_snapshot(collection)? {
      return Ok(current);
    }

    let mut attempt = 1;
    loop {
      let before = read_version(&self.conn(), collection)?;
      let snapshot = Arc::new(self.read_snapshot(collection, model, before, page, between)?);
      let after = read_version(&self.conn(), collection)?;
      if before == after {
        self
          .cache()
          .insert(collection.to_owned(), Arc::clone(&snapshot));
        return Ok(snapshot);
      }
      if attempt >= MAX_SNAPSHOT_ATTEMPTS {
        tracing::debug!(
          collection,
          "collection kept changing while it was loaded; not caching"
        );
        return Ok(snapshot);
      }
      attempt += 1;
    }
  }

  /// Read, decode and normalize `collection`'s vectors for `model`, a page at a time.
  /// Only the SQL for a page runs under the connection lock; decoding — the part that
  /// grows with the corpus — happens after it is released, so a large rebuild does not
  /// hold up every other call (a memory recall has a 3-second budget).
  fn read_snapshot(
    &self,
    collection: &str,
    model: &str,
    version: i64,
    page: usize,
    between: &mut dyn FnMut(),
  ) -> anyhow::Result<Snapshot> {
    let expected: i64 = self.conn().query_row(
      "SELECT COUNT(*) FROM documents WHERE collection = ?1 AND model = ?2",
      params![collection, model],
      |row| row.get(0),
    )?;
    let expected = usize::try_from(expected).unwrap_or(0);
    let limit = i64::try_from(page).unwrap_or(i64::MAX);

    let mut ids = Vec::with_capacity(expected);
    let (mut vectors, mut dim, mut unusable) = (Vec::<f32>::new(), 0usize, 0usize);
    let mut after_id = i64::MIN;
    loop {
      let rows = {
        let conn = self.conn();
        let mut statement = conn.prepare_cached(
          "SELECT id, dim, embedding FROM documents
           WHERE collection = ?1 AND model = ?2 AND id > ?3
           ORDER BY id LIMIT ?4",
        )?;
        statement
          .query_map(params![collection, model, after_id, limit], |row| {
            Ok((
              row.get::<_, i64>(0)?,
              row.get::<_, i64>(1)?,
              row.get::<_, Vec<u8>>(2)?,
            ))
          })?
          .collect::<Result<Vec<_>, _>>()?
      };
      between();
      let Some(&(last_id, _, _)) = rows.last() else {
        break;
      };
      after_id = last_id;
      let full_page = rows.len() >= page;

      for (id, row_dim, blob) in rows {
        let vector = decode(&blob).filter(|vector| vector.len() as i64 == row_dim);
        let Some(unit) = vector.as_deref().and_then(normalized) else {
          unusable += 1;
          continue;
        };
        if dim == 0 {
          dim = unit.len();
          // Known now, so the rest of the corpus lands in one allocation rather than
          // doubling its way there (which briefly holds twice the memory).
          vectors.reserve_exact(expected.saturating_mul(dim));
        }
        if unit.len() != dim {
          unusable += 1;
          continue;
        }
        ids.push(id);
        vectors.extend(unit);
      }
      if !full_page {
        break;
      }
    }

    let other_models: i64 = self.conn().query_row(
      "SELECT COUNT(*) FROM documents WHERE collection = ?1 AND model != ?2",
      params![collection, model],
      |row| row.get(0),
    )?;
    if unusable > 0 {
      tracing::warn!(
        collection,
        unusable,
        "skipped stored vectors that are empty or have the wrong length"
      );
    }
    if other_models > 0 {
      tracing::warn!(
        collection,
        model,
        other_models,
        "documents embedded with another model are not searchable until re-indexed"
      );
    }

    Ok(Snapshot {
      version,
      ids,
      dim,
      vectors,
    })
  }
}

/// A document about to be stored; see [`insert_document`].
struct NewDocument<'a> {
  collection: &'a str,
  source: &'a str,
  chunk_index: usize,
  text: &'a str,
  hash: &'a str,
  metadata: &'a str,
  created_at: &'a str,
  model: &'a str,
  vector: &'a [f32],
}

/// Add `document`, returning its id. The one place that knows the column order of
/// [`INSERT_DOCUMENT`]; `connection` is usually a transaction's.
fn insert_document(connection: &Connection, document: &NewDocument<'_>) -> rusqlite::Result<i64> {
  connection
    .prepare_cached(INSERT_DOCUMENT)?
    .execute(params![
      document.collection,
      document.source,
      document.chunk_index as i64,
      document.text,
      document.hash,
      document.metadata,
      document.created_at,
      document.model,
      document.vector.len() as i64,
      encode(document.vector),
    ])?;
  Ok(connection.last_insert_rowid())
}

/// The document in `collection` whose text is exactly `text` (`hash` being its
/// [`fnv1a64_hex`], which only narrows the search — the text decides).
fn find_by_text(
  connection: &Connection,
  collection: &str,
  hash: &str,
  text: &str,
) -> rusqlite::Result<Option<Document>> {
  connection
    .prepare_cached(&format!(
      "SELECT {DOCUMENT_COLUMNS} FROM documents
       WHERE collection = ?1 AND text_hash = ?2 AND text = ?3
       ORDER BY id LIMIT 1"
    ))?
    .query_row(params![collection, hash, text], document_from_row)
    .optional()
}

/// Document `id`, but only if it is in `collection`.
fn find_by_id(
  connection: &Connection,
  collection: &str,
  id: i64,
) -> rusqlite::Result<Option<Document>> {
  connection
    .prepare_cached(&format!(
      "SELECT {DOCUMENT_COLUMNS} FROM documents WHERE collection = ?1 AND id = ?2"
    ))?
    .query_row(params![collection, id], document_from_row)
    .optional()
}

/// The `meta` key holding a collection's version.
fn version_key(collection: &str) -> String {
  format!("version:{collection}")
}

/// Count one more write to `collection`. Called inside the transaction that makes the
/// write, so the version and the data change together or not at all.
fn bump_version(connection: &Connection, collection: &str) -> rusqlite::Result<()> {
  connection.execute(
    "INSERT INTO meta (key, value) VALUES (?1, '1')
     ON CONFLICT (key) DO UPDATE SET value = CAST(value AS INTEGER) + 1",
    [version_key(collection)],
  )?;
  Ok(())
}

fn read_version(connection: &Connection, collection: &str) -> rusqlite::Result<i64> {
  let stored: Option<String> = connection
    .query_row(
      "SELECT value FROM meta WHERE key = ?1",
      [version_key(collection)],
      |row| row.get(0),
    )
    .optional()?;
  Ok(stored.and_then(|value| value.parse().ok()).unwrap_or(0))
}

/// What a vector set has to be before it is stored: non-empty vectors, all one length.
/// A misbehaving embedder is caught here, rather than as a search that errors for every
/// query afterwards.
fn ensure_usable(vectors: &[Vec<f32>]) -> anyhow::Result<()> {
  if let Some(first) = vectors.first() {
    anyhow::ensure!(!first.is_empty(), "the embedder returned an empty vector");
    anyhow::ensure!(
      vectors.iter().all(|vector| vector.len() == first.len()),
      "the embedder returned vectors of different lengths"
    );
  }
  Ok(())
}

/// Written by this module as JSON; anything else is reported as no metadata rather than
/// failing a search over it.
fn parse_metadata(metadata: &str) -> Value {
  serde_json::from_str(metadata).unwrap_or_else(|_| Value::Object(Default::default()))
}

fn document_from_row(row: &Row<'_>) -> rusqlite::Result<Document> {
  let metadata: String = row.get(5)?;
  Ok(Document {
    id: row.get(0)?,
    collection: row.get(1)?,
    source: row.get(2)?,
    chunk_index: usize::try_from(row.get::<_, i64>(3)?).unwrap_or(0),
    text: row.get(4)?,
    metadata: parse_metadata(&metadata),
    created_at: row.get(6)?,
  })
}

/// Create `path` owner-read/write only if nothing is there yet; an existing file is left
/// exactly as it is.
///
/// Done before SQLite opens it rather than after, so there is no window in which the file
/// exists with the process umask's permissions. SQLite gives its `-wal`/`-shm` companions
/// the same mode as the database file, so this covers them too.
fn create_private(path: &Path) -> std::io::Result<()> {
  let mut options = std::fs::OpenOptions::new();
  options.write(true).create_new(true);
  #[cfg(unix)]
  {
    use std::os::unix::fs::OpenOptionsExt;
    options.mode(0o600);
  }
  match options.open(path) {
    Ok(_) => Ok(()),
    // Already there: the owner's permissions, not ours.
    Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
    Err(err) => Err(err),
  }
}

/// Little-endian `f32`s: fixed by this function, whatever the machine.
fn encode(vector: &[f32]) -> Vec<u8> {
  vector
    .iter()
    .flat_map(|value| value.to_le_bytes())
    .collect()
}

fn decode(blob: &[u8]) -> Option<Vec<f32>> {
  let (words, rest) = blob.as_chunks::<4>();
  rest.is_empty().then(|| {
    words
      .iter()
      .map(|bytes| f32::from_le_bytes(*bytes))
      .collect()
  })
}

#[cfg(test)]
mod tests {
  use serde_json::json;

  use super::*;
  use crate::{knowledge_base::testing::WordEmbedder, tools::testing::TempDir};

  fn store_with(embedder: &Arc<WordEmbedder>) -> VectorStore {
    VectorStore::open_in_memory(Arc::clone(embedder) as Arc<dyn Embedder>).unwrap()
  }

  fn strings(texts: &[&str]) -> Vec<String> {
    texts.iter().map(|text| (*text).to_owned()).collect()
  }

  /// A database file in a directory removed again when the guard drops — even when an
  /// assertion fails first.
  fn temp_db() -> (TempDir, std::path::PathBuf) {
    let dir = TempDir::new("vectors");
    let path = dir.join("nested/vectors.db");
    (dir, path)
  }

  #[tokio::test]
  async fn search_finds_the_relevant_chunk_and_carries_its_source() {
    let embedder = Arc::new(WordEmbedder::new());
    let store = store_with(&embedder);
    store
      .put_source(
        "docs",
        "pets.md",
        strings(&["cats purr and sleep", "dogs bark at the mailman"]),
        json!({ "lang": "en" }),
      )
      .await
      .unwrap();
    store
      .put_source(
        "docs",
        "rust.md",
        strings(&["rust lifetimes and borrowing"]),
        json!({}),
      )
      .await
      .unwrap();

    let hits = store
      .search("docs", "why do dogs bark", 2, 0.0)
      .await
      .unwrap();

    assert_eq!(hits[0].document.text, "dogs bark at the mailman");
    assert_eq!(hits[0].document.source, "pets.md");
    assert_eq!(hits[0].document.chunk_index, 1);
    assert_eq!(hits[0].document.metadata, json!({ "lang": "en" }));
    assert!(hits[0].similarity > hits[1].similarity);
  }

  #[tokio::test]
  async fn min_similarity_drops_weak_matches() {
    let embedder = Arc::new(WordEmbedder::new());
    let store = store_with(&embedder);
    store
      .put_source("docs", "a", strings(&["bread flour yeast"]), json!({}))
      .await
      .unwrap();
    assert!(
      store
        .search("docs", "quantum chromodynamics", 5, 0.5)
        .await
        .unwrap()
        .is_empty()
    );
  }

  /// The point of persisting: re-indexing unchanged text costs no embedding calls.
  #[tokio::test]
  async fn re_indexing_reuses_the_vectors_it_already_has() {
    let embedder = Arc::new(WordEmbedder::new());
    let store = store_with(&embedder);
    let first = store
      .put_source("docs", "a", strings(&["one two", "three four"]), json!({}))
      .await
      .unwrap();
    assert_eq!((first.embedded, first.reused), (2, 0));

    let second = store
      .put_source(
        "docs",
        "a",
        strings(&["one two", "five six", "five six"]),
        json!({}),
      )
      .await
      .unwrap();
    assert_eq!(second.reused, 1, "the unchanged chunk");
    assert_eq!(second.embedded, 1, "a repeated new chunk is embedded once");
    assert_eq!(embedder.texts(), 3);
    assert_eq!(
      store.count("docs").await.unwrap(),
      3,
      "the old chunks are gone"
    );
  }

  #[tokio::test]
  async fn put_source_replaces_and_an_empty_list_removes() {
    let embedder = Arc::new(WordEmbedder::new());
    let store = store_with(&embedder);
    store
      .put_source("docs", "a", strings(&["old text"]), json!({}))
      .await
      .unwrap();
    store
      .put_source("docs", "a", strings(&["new text", "  "]), json!({}))
      .await
      .unwrap();
    let docs = store.list("docs").await.unwrap();
    assert_eq!(docs.len(), 1, "blank chunks are skipped");
    assert_eq!(docs[0].text, "new text");

    store
      .put_source("docs", "a", Vec::new(), json!({}))
      .await
      .unwrap();
    assert_eq!(store.count("docs").await.unwrap(), 0);
  }

  /// A search right after a write sees it: the in-memory copy is rebuilt.
  #[tokio::test]
  async fn search_sees_writes_and_deletes_immediately() {
    let embedder = Arc::new(WordEmbedder::new());
    let store = store_with(&embedder);
    assert!(
      store
        .search("docs", "apple", 5, 0.0)
        .await
        .unwrap()
        .is_empty()
    );
    assert_eq!(
      embedder.calls(),
      0,
      "nothing stored, so the query is not embedded"
    );

    let report = store
      .put_source("docs", "a", strings(&["apple pie"]), json!({}))
      .await
      .unwrap();
    assert_eq!(
      store.search("docs", "apple", 5, 0.0).await.unwrap().len(),
      1
    );

    let deleted = store.delete("docs", report.ids[0]).await.unwrap();
    assert_eq!(deleted.unwrap().text, "apple pie");
    assert!(
      store
        .search("docs", "apple", 5, 0.0)
        .await
        .unwrap()
        .is_empty()
    );
  }

  #[tokio::test]
  async fn collections_are_kept_apart_and_ids_cannot_cross_them() {
    let embedder = Arc::new(WordEmbedder::new());
    let store = store_with(&embedder);
    let report = store
      .put_source("secret", "a", strings(&["launch codes"]), json!({}))
      .await
      .unwrap();

    assert!(
      store
        .search("docs", "launch codes", 5, 0.0)
        .await
        .unwrap()
        .is_empty()
    );
    assert!(
      store.delete("docs", report.ids[0]).await.unwrap().is_none(),
      "deleting through another collection must not reach it"
    );
    assert_eq!(store.count("secret").await.unwrap(), 1);
  }

  /// Vectors from another model are never compared with the current one's.
  #[tokio::test]
  async fn documents_from_another_model_are_not_searched() {
    let (_dir, path) = temp_db();
    let old = Arc::new(WordEmbedder::named("old-model"));
    VectorStore::open(&path, Arc::clone(&old) as Arc<dyn Embedder>)
      .unwrap()
      .put_source("docs", "a", strings(&["apple pie"]), json!({}))
      .await
      .unwrap();

    let new = Arc::new(WordEmbedder::named("new-model"));
    let store = VectorStore::open(&path, Arc::clone(&new) as Arc<dyn Embedder>).unwrap();
    assert!(
      store
        .search("docs", "apple pie", 5, 0.0)
        .await
        .unwrap()
        .is_empty()
    );
    assert_eq!(
      store.count("docs").await.unwrap(),
      1,
      "kept, just not searched"
    );

    let report = store
      .put_source("docs", "a", strings(&["apple pie"]), json!({}))
      .await
      .unwrap();
    assert_eq!(
      report.reused, 0,
      "another model's vector is not reused either"
    );
  }

  /// Persisted, and a second process (here: a second connection) writing to the same
  /// file is seen by the first one's next search.
  #[tokio::test]
  async fn the_store_survives_reopening_and_sees_other_writers() {
    let (_dir, path) = temp_db();
    let embedder = Arc::new(WordEmbedder::new());
    let first = VectorStore::open(&path, Arc::clone(&embedder) as Arc<dyn Embedder>).unwrap();
    first
      .put_source("docs", "a", strings(&["apple pie"]), json!({}))
      .await
      .unwrap();
    assert_eq!(
      first.search("docs", "pear", 5, -1.0).await.unwrap().len(),
      1
    );

    let second = VectorStore::open(&path, Arc::clone(&embedder) as Arc<dyn Embedder>).unwrap();
    second
      .put_source("docs", "b", strings(&["pear tart"]), json!({}))
      .await
      .unwrap();

    let hits = first.search("docs", "pear", 5, 0.1).await.unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].document.text, "pear tart");
  }

  #[tokio::test]
  async fn find_text_matches_exactly() {
    let embedder = Arc::new(WordEmbedder::new());
    let store = store_with(&embedder);
    store
      .put_source("notes", "a", strings(&["likes tea"]), json!({}))
      .await
      .unwrap();
    assert!(
      store
        .find_text("notes", "likes tea")
        .await
        .unwrap()
        .is_some()
    );
    assert!(
      store
        .find_text("notes", "likes tea!")
        .await
        .unwrap()
        .is_none()
    );
    assert!(
      store
        .find_text("other", "likes tea")
        .await
        .unwrap()
        .is_none()
    );
  }

  #[tokio::test]
  async fn delete_source_removes_every_chunk() {
    let embedder = Arc::new(WordEmbedder::new());
    let store = store_with(&embedder);
    store
      .index_text("docs", "long.md", &"word ".repeat(100), 50, 10, json!({}))
      .await
      .unwrap();
    let chunks = store.count("docs").await.unwrap();
    assert!(chunks > 1);
    assert_eq!(
      store.delete_source("docs", "long.md").await.unwrap(),
      chunks
    );
    assert_eq!(store.count("docs").await.unwrap(), 0);
    assert_eq!(
      store.delete_source("docs", "long.md").await.unwrap(),
      0,
      "nothing left to delete"
    );
  }

  #[tokio::test]
  async fn index_text_refuses_an_overlap_that_is_not_smaller_than_the_chunk() {
    let embedder = Arc::new(WordEmbedder::new());
    let store = store_with(&embedder);
    for (size, overlap) in [(10, 10), (10, 11), (0, 0)] {
      assert!(
        store
          .index_text("docs", "x", "y", size, overlap, json!({}))
          .await
          .is_err(),
        "chunk {size}, overlap {overlap}"
      );
    }
    assert_eq!(embedder.calls(), 0, "refused before anything is embedded");
  }

  #[test]
  fn a_store_from_another_schema_version_is_refused() {
    let (_dir, path) = temp_db();
    let embedder: Arc<dyn Embedder> = Arc::new(WordEmbedder::new());
    drop(VectorStore::open(&path, Arc::clone(&embedder)).unwrap());
    Connection::open(&path)
      .unwrap()
      .execute(
        "UPDATE meta SET value = '99' WHERE key = 'schema_version'",
        [],
      )
      .unwrap();
    let err = VectorStore::open(&path, embedder).unwrap_err().to_string();
    assert!(err.contains("schema version 99"), "got: {err}");
  }

  #[test]
  fn vectors_round_trip_through_their_encoding() {
    let vector = vec![0.5, -1.25, f32::MIN_POSITIVE, 3.0e7];
    assert_eq!(decode(&encode(&vector)).unwrap(), vector);
    assert!(decode(&[1, 2, 3]).is_none());
  }

  /// The id a caller was given (a memory number) has to survive a model switch, since
  /// `forget` takes it: re-indexing the source would mint new ones.
  #[tokio::test]
  async fn reindex_makes_another_models_documents_searchable_again_with_the_same_ids() {
    let (_dir, path) = temp_db();
    let old: Arc<dyn Embedder> = Arc::new(WordEmbedder::named("old-model"));
    let before = VectorStore::open(&path, old).unwrap();
    let report = before
      .put_source(
        "memory:s",
        "a",
        strings(&["apple pie", "pear tart"]),
        json!({}),
      )
      .await
      .unwrap();
    assert_eq!(before.stale_count("memory:s").await.unwrap(), 0);

    let new: Arc<dyn Embedder> = Arc::new(WordEmbedder::named("new-model"));
    let store = VectorStore::open(&path, new).unwrap();
    assert_eq!(
      store.stale_count("memory:s").await.unwrap(),
      2,
      "both documents were embedded with the other model"
    );
    assert!(
      store
        .search("memory:s", "apple", 5, 0.0)
        .await
        .unwrap()
        .is_empty(),
      "and are invisible until re-indexed"
    );

    assert_eq!(store.reindex("memory:s").await.unwrap(), 2);
    assert_eq!(store.stale_count("memory:s").await.unwrap(), 0);

    let hits = store.search("memory:s", "apple", 5, 0.0).await.unwrap();
    assert_eq!(hits[0].document.text, "apple pie");
    assert_eq!(
      hits[0].document.id, report.ids[0],
      "the id `forget` takes has to survive"
    );

    assert_eq!(
      store.reindex("memory:s").await.unwrap(),
      0,
      "nothing stale, nothing to do"
    );
  }

  /// One stale document is migrated without touching the others: what a repeated
  /// `remember` needs, as opposed to a whole-collection rewrite nobody asked for.
  #[tokio::test]
  async fn reindex_document_migrates_only_the_one_it_is_given() {
    let (_dir, path) = temp_db();
    let old: Arc<dyn Embedder> = Arc::new(WordEmbedder::named("old-model"));
    let before = VectorStore::open(&path, old).unwrap();
    let report = before
      .put_source(
        "memory:s",
        "a",
        strings(&["apple pie", "pear tart"]),
        json!({}),
      )
      .await
      .unwrap();

    let new = Arc::new(WordEmbedder::named("new-model"));
    let store = VectorStore::open(&path, Arc::clone(&new) as Arc<dyn Embedder>).unwrap();
    assert!(
      store
        .reindex_document("memory:s", report.ids[0])
        .await
        .unwrap()
    );
    assert_eq!(new.texts(), 1, "one text sent, not the whole collection");
    assert_eq!(store.stale_count("memory:s").await.unwrap(), 1);

    assert!(
      !store
        .reindex_document("memory:s", report.ids[0])
        .await
        .unwrap(),
      "already current"
    );
    assert!(
      !store
        .reindex_document("other", report.ids[1])
        .await
        .unwrap(),
      "an id does not reach across collections"
    );
    assert_eq!(new.texts(), 1);
  }

  #[tokio::test]
  async fn insert_unique_stores_once_and_stops_at_the_limit() {
    let embedder = Arc::new(WordEmbedder::new());
    let store = store_with(&embedder);

    let Inserted::New(first) = store
      .insert_unique("c", "s1", "tea over coffee", json!({}), 2)
      .await
      .unwrap()
    else {
      panic!("expected a new document");
    };
    assert_eq!(first.text, "tea over coffee");
    assert_eq!(
      store.list("c").await.unwrap()[0].created_at,
      first.created_at,
      "the returned document is what was stored"
    );

    assert!(matches!(
      store
        .insert_unique("c", "s2", "tea over coffee", json!({}), 2)
        .await
        .unwrap(),
      Inserted::Existing(existing) if existing.id == first.id
    ));
    assert_eq!(embedder.texts(), 1, "a duplicate is not embedded again");

    store
      .insert_unique("c", "s3", "metric units", json!({}), 2)
      .await
      .unwrap();
    assert_eq!(
      store
        .insert_unique("c", "s4", "a third fact", json!({}), 2)
        .await
        .unwrap(),
      Inserted::Full(2)
    );
    assert_eq!(store.count("c").await.unwrap(), 2);
    assert!(
      !store
        .search("c", "tea coffee", 1, 0.0)
        .await
        .unwrap()
        .is_empty(),
      "an insert is visible to search straight away"
    );
  }

  /// The reason `insert_unique` exists: with the check and the insert as separate calls,
  /// every one of these would see "not there yet, room left".
  #[tokio::test]
  async fn concurrent_insert_unique_calls_cannot_duplicate_or_overfill() {
    let (_dir, path) = temp_db();
    let embedder: Arc<dyn Embedder> = Arc::new(WordEmbedder::new());
    // Two handles on one file, as two processes would be.
    let stores = [
      VectorStore::open(&path, Arc::clone(&embedder)).unwrap(),
      VectorStore::open(&path, embedder).unwrap(),
    ];

    let mut tasks = Vec::new();
    for round in 0..8 {
      for (which, store) in stores.iter().enumerate() {
        let store = store.clone();
        tasks.push(tokio::spawn(async move {
          // Four distinct facts, each attempted several times; room for only three.
          let text = format!("fact number {}", round % 4);
          store
            .insert_unique("c", &format!("s{which}-{round}"), &text, json!({}), 3)
            .await
        }));
      }
    }
    for task in tasks {
      task.await.unwrap().unwrap();
    }

    let documents = stores[0].list("c").await.unwrap();
    let mut texts: Vec<_> = documents.iter().map(|d| d.text.as_str()).collect();
    texts.sort_unstable();
    texts.dedup();
    assert_eq!(documents.len(), 3, "never over the limit");
    assert_eq!(texts.len(), 3, "never the same text twice");
  }

  /// Rows for [`Inner::replace_source`] without going through the embedder: a vector the
  /// word model would give `text`.
  fn write_directly(inner: &Inner, collection: &str, source: &str, text: &str) {
    inner
      .replace_source(
        collection,
        source,
        &[text.to_owned()],
        &[fnv1a64_hex(text)],
        &[WordEmbedder::vector(text)],
        "{}",
        "words-v1",
      )
      .unwrap();
  }

  fn seeded(texts: &[&str]) -> VectorStore {
    let embedder = Arc::new(WordEmbedder::new());
    let store = store_with(&embedder);
    for (index, text) in texts.iter().enumerate() {
      write_directly(&store.inner, "c", &format!("s{index}"), text);
    }
    store
  }

  /// The race this design exists for, constructed rather than argued: a write lands
  /// between two pages of a rebuild. The copy read so far mixes two states, so it must not
  /// be cached, and the rebuild must be redone until it is consistent.
  #[test]
  fn a_write_between_pages_of_a_rebuild_is_not_lost_to_the_cache() {
    let store = seeded(&["one", "two", "three", "four", "five"]);
    let inner = &store.inner;
    let mut landed = false;

    let snapshot = inner
      .snapshot_paged("c", "words-v1", 2, &mut || {
        if !landed {
          landed = true;
          write_directly(inner, "c", "late", "a late arrival");
        }
      })
      .unwrap();

    assert_eq!(
      snapshot.ids.len(),
      6,
      "the rebuild was redone and sees the write"
    );
    assert_eq!(
      snapshot.version,
      read_version(&inner.conn(), "c").unwrap(),
      "and is what the collection's version says"
    );
    let cached = inner
      .cache()
      .get("c")
      .cloned()
      .expect("a consistent copy is cached");
    assert_eq!(cached.ids, snapshot.ids);
  }

  /// If writers keep winning the race, a rebuild gives up on caching rather than looping,
  /// serves what it read, and the next caller starts from the truth again.
  #[test]
  fn a_rebuild_that_keeps_losing_the_race_serves_but_does_not_cache() {
    let store = seeded(&["one", "two", "three"]);
    let inner = &store.inner;
    let mut writes = 0;

    let served = inner
      .snapshot_paged("c", "words-v1", 2, &mut || {
        writes += 1;
        write_directly(
          inner,
          "c",
          &format!("churn-{writes}"),
          &format!("churn {writes}"),
        );
      })
      .unwrap();

    assert!(served.ids.len() >= 3, "something usable was served");
    assert!(
      inner
        .cache()
        .get("c")
        .is_none_or(|cached| cached.version == read_version(&inner.conn(), "c").unwrap()),
      "whatever is cached is current"
    );
    let next = inner.snapshot("c", "words-v1").unwrap();
    assert_eq!(
      next.ids.len(),
      3 + writes,
      "the next caller sees every write"
    );
  }

  /// Pages are an implementation detail: any page size gives the same rows in the same
  /// order, including one that does not divide the row count and one larger than it.
  #[test]
  fn paging_does_not_change_what_is_loaded() {
    let store = seeded(&["a1", "b2", "c3", "d4", "e5"]);
    let inner = &store.inner;
    let whole = inner
      .snapshot_paged("c", "words-v1", 1000, &mut || {})
      .unwrap();
    for page in [1, 2, 3, 5, 6] {
      inner.cache().clear();
      let paged = inner
        .snapshot_paged("c", "words-v1", page, &mut || {})
        .unwrap();
      assert_eq!(paged.ids, whole.ids, "page size {page}");
      assert_eq!(paged.vectors, whole.vectors, "page size {page}");
    }
    assert_eq!(whole.ids.len(), 5);
  }

  /// Callers that all find the cache stale rebuild it once between them.
  #[test]
  fn concurrent_callers_share_one_rebuild() {
    let store = seeded(&["one", "two", "three"]);
    let rebuilds = std::sync::atomic::AtomicUsize::new(0);
    let inner = &store.inner;

    std::thread::scope(|scope| {
      for _ in 0..8 {
        scope.spawn(|| {
          inner
            .snapshot_paged("c", "words-v1", 2, &mut || {
              rebuilds.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            })
            .unwrap();
        });
      }
    });

    // Two pages for three rows, plus the empty read that ends the scan when the last page
    // is full: a single rebuild calls `between` a bounded, small number of times.
    assert!(
      rebuilds.load(std::sync::atomic::Ordering::SeqCst) <= 2,
      "more than one rebuild: {rebuilds:?}"
    );
  }

  /// A write to one collection says nothing about another: neither its version nor its
  /// cached copy changes.
  #[tokio::test]
  async fn a_write_to_one_collection_leaves_the_others_alone() {
    let embedder = Arc::new(WordEmbedder::new());
    let store = store_with(&embedder);
    store
      .put_source("docs", "a", strings(&["apple pie"]), json!({}))
      .await
      .unwrap();
    store.search("docs", "apple", 1, 0.0).await.unwrap();
    let docs_before = store.version("docs").await.unwrap();
    let cached_before = store.inner.cache().get("docs").cloned().unwrap();

    store
      .put_source("memory:s", "m", strings(&["likes tea"]), json!({}))
      .await
      .unwrap();

    assert_eq!(store.version("docs").await.unwrap(), docs_before);
    assert_eq!(store.version("nothing-here").await.unwrap(), 0);
    let cached_after = store.inner.cache().get("docs").cloned().unwrap();
    assert!(
      Arc::ptr_eq(&cached_before, &cached_after),
      "the other collection's copy was kept, not rebuilt"
    );
  }

  /// Every kind of write moves the version of the collection it touched.
  #[tokio::test]
  async fn every_write_moves_the_version() {
    let embedder = Arc::new(WordEmbedder::new());
    let store = store_with(&embedder);
    let mut last = store.version("c").await.unwrap();
    let mut moved = |label: &'static str, now: i64| {
      assert!(now > last, "{label}: {last} -> {now}");
      last = now;
    };

    let report = store
      .put_source("c", "a", strings(&["apple pie", "pear tart"]), json!({}))
      .await
      .unwrap();
    moved("put_source", store.version("c").await.unwrap());
    store
      .insert_unique("c", "b", "fig jam", json!({}), 10)
      .await
      .unwrap();
    moved("insert_unique", store.version("c").await.unwrap());
    store.delete("c", report.ids[0]).await.unwrap();
    moved("delete", store.version("c").await.unwrap());
    store.delete_source("c", "a").await.unwrap();
    moved("delete_source", store.version("c").await.unwrap());

    let before = store.version("c").await.unwrap();
    store.delete("c", 424_242).await.unwrap();
    store.delete_source("c", "no-such-source").await.unwrap();
    assert_eq!(
      store.version("c").await.unwrap(),
      before,
      "a delete that found nothing changed nothing"
    );
  }

  // ---- ids are never handed out twice ---------------------------------------------------

  /// The reported hole: with a plain `INTEGER PRIMARY KEY`, deleting the newest row freed
  /// its id for the next insert, so a `forget #12` from earlier in a conversation could
  /// delete a memory saved after it.
  #[tokio::test]
  async fn a_deleted_documents_id_is_never_handed_out_again() {
    let embedder = Arc::new(WordEmbedder::new());
    let store = store_with(&embedder);
    let report = store
      .put_source("c", "a", strings(&["first", "second"]), json!({}))
      .await
      .unwrap();
    let newest = *report.ids.last().unwrap();
    store.delete("c", newest).await.unwrap();

    let Inserted::New(next) = store
      .insert_unique("c", "b", "third", json!({}), 10)
      .await
      .unwrap()
    else {
      panic!("expected a new document");
    };

    assert!(next.id > newest, "{} was reused for {}", newest, next.id);
    assert!(store.delete("c", newest).await.unwrap().is_none());
    assert_eq!(
      store.count("c").await.unwrap(),
      2,
      "the stale id reached nothing"
    );
  }

  /// Re-indexing a source mints new ids too, and none of them is an old one.
  #[tokio::test]
  async fn re_indexing_never_reuses_an_id() {
    let embedder = Arc::new(WordEmbedder::new());
    let store = store_with(&embedder);
    let first = store
      .put_source("c", "a", strings(&["one", "two"]), json!({}))
      .await
      .unwrap();
    let second = store
      .put_source("c", "a", strings(&["one", "two"]), json!({}))
      .await
      .unwrap();
    assert!(second.ids.iter().all(|id| !first.ids.contains(id)));
    assert!(second.ids.iter().min() > first.ids.iter().max());
  }

  // ---- upgrading a file written by the first layout ----------------------------------------

  /// A file as the first version of this crate wrote it: no `AUTOINCREMENT`, schema 1.
  fn write_v1_file(path: &std::path::Path, rows: &[(i64, &str)]) {
    let conn = Connection::open(path).unwrap();
    conn
      .execute_batch(
        "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
         CREATE TABLE documents (
           id INTEGER PRIMARY KEY, collection TEXT NOT NULL, source TEXT NOT NULL,
           chunk_index INTEGER NOT NULL, text TEXT NOT NULL, text_hash TEXT NOT NULL,
           metadata TEXT NOT NULL, created_at TEXT NOT NULL, model TEXT NOT NULL,
           dim INTEGER NOT NULL, embedding BLOB NOT NULL,
           UNIQUE (collection, source, chunk_index));
         CREATE INDEX documents_by_collection ON documents (collection, model);
         CREATE INDEX documents_by_text ON documents (model, text_hash);
         INSERT INTO meta (key, value) VALUES ('schema_version', '1');",
      )
      .unwrap();
    for (id, text) in rows {
      conn
        .execute(
          "INSERT INTO documents (id, collection, source, chunk_index, text, text_hash,
             metadata, created_at, model, dim, embedding)
           VALUES (?1, 'memory:s', ?2, 0, ?3, ?4, '{}', '2026-01-01T00:00:00+00:00',
             'words-v1', 64, ?5)",
          params![
            id,
            format!("src-{id}"),
            text,
            fnv1a64_hex(text),
            encode(&WordEmbedder::vector(text))
          ],
        )
        .unwrap();
    }
  }

  #[tokio::test]
  async fn a_first_layout_file_is_upgraded_in_place_keeping_every_id() {
    let (_dir, path) = temp_db();
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    write_v1_file(&path, &[(3, "likes tea"), (7, "works on atlas")]);

    let embedder: Arc<dyn Embedder> = Arc::new(WordEmbedder::new());
    let store = VectorStore::open(&path, Arc::clone(&embedder)).unwrap();

    let ids: Vec<i64> = store
      .list("memory:s")
      .await
      .unwrap()
      .iter()
      .map(|d| d.id)
      .collect();
    assert_eq!(
      ids,
      [3, 7],
      "ids are what `forget` takes; they must survive"
    );
    let hits = store.search("memory:s", "atlas", 1, 0.0).await.unwrap();
    assert_eq!(hits[0].document.id, 7, "and the vectors came with them");

    // The upgrade is what makes ids stop being reused.
    store.delete("memory:s", 7).await.unwrap();
    let Inserted::New(next) = store
      .insert_unique("memory:s", "new", "fresh fact", json!({}), 10)
      .await
      .unwrap()
    else {
      panic!("expected a new document");
    };
    assert!(next.id > 7, "got {}", next.id);

    let stored: String = Connection::open(&path)
      .unwrap()
      .query_row(
        "SELECT value FROM meta WHERE key = 'schema_version'",
        [],
        |r| r.get(0),
      )
      .unwrap();
    assert_eq!(stored, SCHEMA_VERSION);
    drop(store);
    VectorStore::open(&path, embedder).expect("opening an upgraded file again is a no-op");
  }

  /// Several processes opening one new (or old) file at the same moment all succeed: the
  /// version is read and written in one write transaction rather than check-then-insert.
  #[test]
  fn opening_one_file_from_many_threads_at_once_succeeds() {
    for upgrade in [false, true] {
      let (_dir, path) = temp_db();
      if upgrade {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        write_v1_file(&path, &[(1, "likes tea")]);
      }
      let barrier = Arc::new(std::sync::Barrier::new(6));
      let handles: Vec<_> = (0..6)
        .map(|_| {
          let (path, barrier) = (path.clone(), Arc::clone(&barrier));
          std::thread::spawn(move || {
            barrier.wait();
            let embedder: Arc<dyn Embedder> = Arc::new(WordEmbedder::new());
            VectorStore::open(&path, embedder).map(|_| ())
          })
        })
        .collect();
      for handle in handles {
        handle
          .join()
          .unwrap()
          .unwrap_or_else(|err| panic!("a concurrent open failed (upgrade: {upgrade}): {err:#}"));
      }
    }
  }

  // ---- delete_if / sources --------------------------------------------------------------

  #[tokio::test]
  async fn delete_if_deletes_only_what_the_check_accepts() {
    let embedder = Arc::new(WordEmbedder::new());
    let store = store_with(&embedder);
    let report = store
      .put_source("c", "a", strings(&["keep me", "drop me"]), json!({}))
      .await
      .unwrap();

    let refused = store
      .delete_if("c", report.ids[0], |document| document.text == "drop me")
      .await
      .unwrap();
    assert!(matches!(&refused, DeleteOutcome::Refused(d) if d.text == "keep me"));
    assert_eq!(
      store.count("c").await.unwrap(),
      2,
      "refused deletes nothing"
    );

    let deleted = store
      .delete_if("c", report.ids[1], |document| document.text == "drop me")
      .await
      .unwrap();
    assert!(matches!(&deleted, DeleteOutcome::Deleted(d) if d.text == "drop me"));
    assert_eq!(store.count("c").await.unwrap(), 1);

    assert_eq!(
      store.delete_if("c", report.ids[1], |_| true).await.unwrap(),
      DeleteOutcome::Missing
    );
    assert_eq!(
      store
        .delete_if("other", report.ids[0], |_| true)
        .await
        .unwrap(),
      DeleteOutcome::Missing,
      "an id does not reach across collections"
    );
  }

  #[tokio::test]
  async fn sources_lists_each_once_sorted() {
    let embedder = Arc::new(WordEmbedder::new());
    let store = store_with(&embedder);
    store
      .put_source("c", "b.md", strings(&["one", "two"]), json!({}))
      .await
      .unwrap();
    store
      .put_source("c", "a.md", strings(&["three"]), json!({}))
      .await
      .unwrap();
    store
      .put_source("other", "z.md", strings(&["four"]), json!({}))
      .await
      .unwrap();
    assert_eq!(store.sources("c").await.unwrap(), ["a.md", "b.md"]);
    assert!(store.sources("empty").await.unwrap().is_empty());
  }

  #[cfg(unix)]
  #[tokio::test]
  async fn a_new_database_file_is_not_readable_by_other_accounts() {
    use std::os::unix::fs::PermissionsExt;

    let (_dir, path) = temp_db();
    let embedder: Arc<dyn Embedder> = Arc::new(WordEmbedder::new());
    let store = VectorStore::open(&path, embedder).unwrap();
    store
      .put_source("docs", "a", strings(&["apple pie"]), json!({}))
      .await
      .unwrap();

    let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o600, "got {mode:o}");
  }
}
