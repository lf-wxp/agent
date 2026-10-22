# Changelog

Notable changes to the `agent` crate. Entries under *Unreleased* ship with the next version.

## Unreleased

### Added

- `knowledge_base::store::VectorStore`: a persistent SQLite vector store (collections,
  per-source atomic re-indexing, vectors reused for unchanged text, exact cosine search).
  `insert_unique` stores one text unless it is already there or the collection is full,
  as a single transaction; `reindex` / `reindex_document` re-embed documents left behind
  by another `EMBED_MODEL`; `delete_if` deletes a document only if a check on its current
  content passes. Document ids are never handed out twice. Every collection has a version
  (`version`) that each write bumps in its own transaction. A file written by an earlier
  build is upgraded in place on first open.
- `knowledge_base::embed::{Embedder, OpenAiEmbedder}`: embeddings behind a trait, with
  retry, an `embeddings` tracing span and usage charged to the surrounding agent run.
- `memory` (`MemoryStore`, `RecallCallback`, `memory::tools`): long-term memory across
  conversations. `memory::tools::GATED` lists the tools a front-end should put behind an
  approval prompt (`forget`). `forget` takes the memory's text as well as its number and
  deletes nothing that does not say it. `memory::tools::RememberGuard` declines `remember`
  in a turn that used a tool whose output a third party could have written.
- `tools::retrieve::RetrieveTool`: let the model search a `VectorStore` collection.
- `cli`: `--memory`, `--reindex-memory`, `/memory`, and `AGENT_CLI_MEMORY_PATH`.
- `knowledge_base::embed::Embedder::{embed_checked, embed_one}`, and `tools::schema_of`.
- `examples/rag.rs`.

### Changed

- **Build:** a new default feature, `sqlite`, pulls in `rusqlite` with `bundled` SQLite.
  The first build therefore needs a C compiler and is slower. Opt out with
  `default-features = false`; the `cli` binary and the `rag` example require the feature.
- `knowledge_base::embed::embed_text` and `embed_texts` are deprecated in favour of
  `OpenAiEmbedder`; they still work.
- `knowledge_base::search::vector_search` embeds the query and the chunks in one request
  (`vector_search_with` takes any `Embedder`).
- `OpenAiEmbedder` splits a large input into requests of at most 2048 texts **and** about
  200 KB, not by count alone.
- **Wire protocol:** `ChatEvent::ApprovalRequired` and `PendingApprovalView` gained a
  `once_only` field (`#[serde(default)]`, so an older bundle or server still reads). It
  is set for a tool that does not remember "always allow" (`run_command`, `forget`), and
  the terminal and web prompts leave that option out. `callback::dual_approval::ApprovalMeta`
  gained the same field, so code that builds one with a struct literal needs `once_only`.
  Rebuild the web bundle (`cargo make web`) to pick it up.
- `crate::agent::fingerprint`'s FNV-1a hash moved to `crate::util`; the digests it
  produces are unchanged.

### Privacy

- Memories are sent to the embedding provider when saved, and the opening of each user
  message is sent as the recall query. With `EMBED_BASE_URL` unset that is the chat
  provider; set it to a different provider and that one sees them too.
