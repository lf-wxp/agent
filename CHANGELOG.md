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
- `tools::FnTool`: a tool built from an argument struct (`Deserialize + JsonSchema`) and
  an async closure, with `with_timeout` / `read_only`. Unparseable arguments are reported
  to the model with serde's reason; empty arguments are read as `{}`.
- `tools::ToolDefinition` (name, description, JSON Schema), `tools::tool::validate_tool_name`
  and `MAX_TOOL_NAME_CHARS`.
- `callback::context_optimizer::Encoding` (`cl100k_base` / `o200k_base`, chosen by
  `Encoding::for_model`), `ContextOptimizer::{for_model, with_encoding, encoding}` and
  `tokens::Ledger::{with_encoding, encoding}`.
- `Agent::context_encoding`: what the optimizer `Agent::new` installed counts the history
  budget in, so a caller can report it and a test can check the choice (the installed
  optimizer is a trait object and cannot be read back otherwise).

### Changed

- **Token counting follows the model.** The `ContextOptimizer` that `Agent::new` installs
  counts the budget with `o200k_base` when the agent's model uses it (GPT-4o, GPT-4.1,
  GPT-5, the `o` series, `gpt-oss`), by `tiktoken-rs`'s model table; every other model is
  still counted with `cl100k_base`. For those OpenAI models a conversation usually counts
  as fewer tokens than before — markedly so for Chinese — so trimming starts later. A
  hand-built `ContextOptimizer::new(n)`, and the free functions in
  `context_optimizer::tokens`, keep counting with `cl100k_base`; add `.for_model(model)`
  to match.
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
- **Build:** three more default features, so the library can be built without what only
  the CLI or one tool needs. With `default-features = false` the crate depends on about
  180 crates instead of about 260.
  - `cli`: `axum`, `tower-http`, `reedline`, `crossterm`, the `shared` wire types and
    tokio's `net` / `signal`. The `cli` binary now requires this feature (it turns on
    `mcp` and `sqlite` itself).
  - `mcp`: `tools::mcp`, `ToolRegistry::{with_mcp, with_mcp_over}` and
    `callback::mcp_guard`. `rmcp` is now built with its client features only; the server
    side the `mcp_*` examples use moved to `[dev-dependencies]`.
  - `zip`: the `unzip_file` tool (`tools::file_upzip`). Without it the tool is missing
    from `ToolRegistry::builtin` / `builtin_all` and `select` rejects its name. The
    `gaia` binary requires it.

  A crate that already sets `default-features = false` (to drop `sqlite`) now also loses
  these; add back the ones it uses.
- `examples/file_tools`, `mcp_agent`, `mcp_config`, `mcp_http_agent` and `mcp_inspect`
  declare `required-features`, so they are skipped, not broken, in a build without them.
- **Breaking — tools no longer expose `async-openai` types.** A tool is described by
  `ToolDefinition`, converted to the chat-completions shape only in
  `llm::client::request_builder`; the JSON sent on the wire is unchanged.
  - `Tool::definition` returns `ToolDefinition` and cannot fail (it was
    `anyhow::Result<ChatCompletionTools>`). Nothing to change unless a tool overrode it.
  - `ToolRegistry::definitions` returns `&[ToolDefinition]`; read `.name`,
    `.description`, `.parameters` directly instead of matching `ChatCompletionTools`.
  - `llm::client::request_builder` takes `&[ToolDefinition]`.
  - `llm::schema::final_answer_tool` returns `agent::Result<ToolDefinition>`, and
    `final_answer_tool_from_schema` returns `ToolDefinition` (no longer a `Result`).
- **Breaking — a tool name must be one chat APIs accept.** `ToolRegistry::add` now refuses
  a name outside 1–64 ASCII letters, digits, `_` or `-`, instead of letting every request
  carrying it fail later with an error that does not say which tool is to blame. Built-in,
  MCP and sub-agent tools already meet this. Note that the rule is the chat-completions
  one, applied in the provider-neutral registry: a `ChatBackend` whose own naming rule is
  *wider* (dotted or namespaced names) cannot register such a tool, and a `Tool` or
  `AgentTool` that used one now fails at registration. `FnTool::new` and `AgentTool::new`
  also reject a blank description, which previously only `AgentTool` did.
- **Breaking — typed errors.** Every fallible public function returns `agent::Result<T>`
  (`Result<T, agent::Error>`) instead of `anyhow::Result<T>`, and the library no longer
  uses `anyhow` internally — the binaries, examples and integration tests still do, so
  `anyhow` stays a normal dependency. `agent::Error` is a `#[non_exhaustive]` enum:
  `Cancelled`, `BudgetExhausted`, `TruncatedOutput`, `Provider`, `InvalidInput`, `Config`,
  `Tool`, `Mcp`, `Storage`, `Io`, `Other` (see `agent::error`). `{err}` shows the outer
  message and `{err:#}` the whole cause chain, as before; `?` still converts into
  `anyhow`.
  - Match instead of downcasting: `Err(agent::Error::Cancelled(c))` instead of
    `err.downcast_ref::<Cancelled>()`; `Error::downcast_ref` remains for reaching a cause
    (an `std::io::Error`, a `rusqlite::Error`). One level is *not* on the `source()`
    chain: a cause that arrived through `?` from another crate is transparent, so it is
    reached with `downcast_ref` rather than by walking `source()`.
  - Implementations of `Tool::execute`, `ChatBackend::{create, create_stream}` and
    `Embedder::embed` return `agent::Result`. Build errors with `Error::tool(..)` and
    friends, wrap foreign ones with `agent::error::ResultExt::wrap_err` / `err_as`. The
    closure given to `FnTool` returns `agent::Result<R>`.
  - `wrap_err` / `wrap_err_with` / `err_as` only reclassify another crate's error: an
    `agent::Error` keeps its variant (with the message added), so a `Cancelled` stays one
    and an unretryable `Provider` stays unretryable.
  - `llm::retry::Permanent` is gone. A backend reports provider failures as
    `agent::error::ProviderError::transient` / `::permanent` (an `OpenAIError` converts
    with `?`, classified by HTTP status as before).
  - `llm::retry::is_transient` takes `&agent::Error` and no longer retries
    `TruncatedOutput`, `BudgetExhausted`, `InvalidInput`, `Config` or `Cancelled`;
    `with_retry` takes and returns `agent::Result`.
  - `llm::tool_loop::BudgetExhausted` now wraps the failure it qualifies
    (`BudgetExhausted::inner`) instead of being a context marker; it and
    `llm::structured::TruncatedOutput` live in `agent::error` and are re-exported from
    their old paths. `agent::Cancelled` likewise.
  - The `FromStr` impls of `StructuredMode` and `TokenLimitField` return `agent::Error`.
  - A trace's `error.type` for a non-provider failure is now the error's kind (`tool`,
    `storage`, ...) instead of `_OTHER`.

### Privacy

- Memories are sent to the embedding provider when saved, and the opening of each user
  message is sent as the recall query. With `EMBED_BASE_URL` unset that is the chat
  provider; set it to a different provider and that one sees them too.
