//! Demonstrates retrieval and long-term memory.
//!
//! - **Retrieval (RAG)**: a directory of text files is chunked, embedded and stored in a
//!   SQLite-backed [`VectorStore`]; the agent gets a `retrieve` tool over it and answers a
//!   question from the passages it finds. Re-running the example re-indexes only what
//!   changed — unchanged chunks are never sent to the embedder again — and drops what is
//!   gone: the index mirrors the directories you name, so a file deleted since the last
//!   run (or a directory you stopped naming) is no longer searchable.
//! - **Memory**: two separate conversations, the first saving a fact with the `remember`
//!   tool, the second recalling it automatically — the point of a *cross-session* memory.
//!
//! Both need an embedding model (`EMBED_MODEL`, served from the chat provider's endpoint)
//! and a chat model (`LLM_MODEL`).
//!
//! ```sh
//! cargo run --example rag                 # indexes this crate's own `docs/`
//! cargo run --example rag -- notes/ more/ # or the directories you name
//! ```
//!
//! Everything under the directories you name is read and **sent to the embedding
//! provider**, so point it at documents you are happy to send there. Hidden files,
//! `.env*`, key and certificate files, and anything named like a secret (`id_rsa`,
//! `credentials`, `secrets.yaml`, …) are skipped, but that is a safety net, not a
//! substitute for choosing the directory.
//!
//! The memories this run saves are removed again when it ends — normally, on an error, and
//! on Ctrl-C. (Only a `kill -9` leaves them behind, under a scope nothing else reads.)

use std::{
  collections::HashSet,
  path::{Path, PathBuf},
  sync::Arc,
};

use agent::{
  Agent,
  callback::approval::ApprovalCallback,
  config,
  knowledge_base::{
    embed::{Embedder, OpenAiEmbedder},
    store::VectorStore,
  },
  llm::provider::Provider,
  memory::{
    MemoryStore, RecallCallback,
    tools::{GATED, RememberGuard, memory_tools},
  },
  telemetry,
  tools::{
    ToolRegistry,
    retrieve::{self, RetrieveTool},
  },
};
use walkdir::WalkDir;

/// Collection name for the indexed documents.
const DOCS: &str = "docs";

/// Characters per chunk, and how much neighbouring chunks overlap.
const CHUNK_SIZE: usize = 800;
const CHUNK_OVERLAP: usize = 120;

/// Largest file indexed, in bytes. A stray binary or a huge log is skipped rather than
/// embedded.
const MAX_FILE_BYTES: u64 = 1 << 20;

const QUESTION: &str = "How do I give an agent its own tools, and what happens if a tool \
                        call raises an approval prompt?";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
  let _telemetry = telemetry::init()?;

  let roots: Vec<PathBuf> = std::env::args().skip(1).map(PathBuf::from).collect();
  let roots = if roots.is_empty() {
    vec![default_docs()]
  } else {
    roots
  };

  let embedder: Arc<dyn Embedder> = Arc::new(OpenAiEmbedder::from_config()?);
  // Every file under the roots goes to the embedding provider, so a setting that used to
  // send them elsewhere (`EMBED_BASE_URL`, now ignored) is reported before they are sent,
  // not after.
  if let Some(warning) = config::removed_embed_overrides_warning() {
    println!("{warning}\n");
  }
  // One file holds both collections: the indexed documents and the memories. That is
  // fine — a collection is a namespace, not a file. Under `.agent/`, which the repository
  // ignores.
  let store = VectorStore::open(".agent/rag-example.sqlite", Arc::clone(&embedder))?;
  // A scope of this run's own: the second conversation has to show a fact being *saved*
  // rather than found from an earlier run, and the way to get that without deleting
  // anything this example did not create is to start from a scope nothing else uses. What
  // the run saved is forgotten again at the end.
  let scope = format!("example-{}", uuid::Uuid::new_v4().simple());
  let memory = MemoryStore::new(store.clone(), scope);

  index(&store, &roots).await?;

  let toolbox = retrieval_toolbox(store, &memory);

  // The conversations are one step and the cleanup another, so that the memories this run
  // saved are removed whichever way the first ends: an error from the model, or Ctrl-C.
  // (Skipping the cleanup because a `?` returned early is how a "scratch" scope turns into
  // a pile of facts nobody can see.)
  let outcome = tokio::select! {
    result = conversations(&toolbox, &memory) => result,
    _ = tokio::signal::ctrl_c() => Err(anyhow::anyhow!("interrupted")),
  };
  let cleanup = forget_everything(&memory).await;
  outcome?;
  cleanup?;
  Ok(())
}

/// The agent both conversations use: it recalls memories before each turn, asks before
/// `forget`, and will not `remember` on the strength of something it only read.
fn assistant(
  name: &str,
  instructions: &str,
  toolbox: &Arc<ToolRegistry>,
  memory: &MemoryStore,
) -> Agent {
  Agent::new(
    Provider::shared().clone(),
    config::model(),
    Some(instructions),
    Arc::clone(toolbox),
  )
  .with_before_llm_callback(Arc::new(RecallCallback::new(memory.clone())))
  // `forget` destroys something the user asked to keep, so it asks first — `GATED` is the
  // list of memory tools to treat that way. (It prompts on stdin, like
  // `callback_approval`; this example's conversations are not expected to reach it.)
  .with_before_tool_callback(Arc::new(ApprovalCallback::new(GATED.iter().copied())))
  // A memory outlives the conversation, so `remember` is declined in a turn that has
  // used a tool whose output a third party could have written. The indexed documents are
  // the ones you pointed this example at, so `retrieve` is vouched for here; a deployment
  // indexing pages it does not control should not do the same.
  .with_before_tool_callback(Arc::new(RememberGuard::new().trusting(retrieve::NAME)))
  .with_name(name)
}

/// The three conversations of the demo: retrieval, a fact being saved, and the same fact
/// found in a fresh conversation.
async fn conversations(toolbox: &Arc<ToolRegistry>, memory: &MemoryStore) -> anyhow::Result<()> {
  println!("\n=== Retrieval ===");
  let agent = assistant(
    "docs-assistant",
    "Answer from the retrieved passages, and say where each answer came from.",
    toolbox,
    memory,
  );
  let answer = agent.run(QUESTION).await?;
  println!("\nQ: {QUESTION}\nA: {}", answer.output);
  println!("usage: {:?}", answer.context.usage);

  println!("\n=== Memory, first conversation ===");
  let first = agent
    .run("For the record: I am working on the Atlas migration, and I prefer answers in Chinese.")
    .await?;
  println!("A: {}", first.output);
  for saved in memory.list().await? {
    println!("  memory #{} ({}): {}", saved.id, saved.date(), saved.text);
  }

  println!("\n=== Memory, second conversation (fresh, nothing carried over) ===");
  // A brand-new agent with an empty transcript: only the memory connects the two.
  let second = assistant("memory-assistant", "Be brief.", toolbox, memory);
  let answer = second
    .run("What am I working on at the moment, and how should you answer me?")
    .await?;
  println!("A: {}", answer.output);
  Ok(())
}

/// Remove every memory in this run's scope, returning how many there were. The scope is
/// not reused, so what is left would only accumulate in the file.
async fn forget_everything(memory: &MemoryStore) -> anyhow::Result<usize> {
  let saved = memory.list().await?;
  for memory_item in &saved {
    memory.forget(memory_item.id).await?;
  }
  Ok(saved.len())
}

/// `retrieve` over the documents, plus the three memory tools.
fn retrieval_toolbox(store: VectorStore, memory: &MemoryStore) -> Arc<ToolRegistry> {
  let mut toolbox = ToolRegistry::builtin().expect("the built-in tools register");
  toolbox
    .add(Arc::new(RetrieveTool::new(store, DOCS).with_description(
      "Search the example documents for passages relevant to a question.",
    )))
    .expect("`retrieve` is free");
  for tool in memory_tools(memory) {
    toolbox.add(tool).expect("memory tool names are free");
  }
  Arc::new(toolbox)
}

/// With no arguments, index this crate's own `docs/` — something worth searching, and
/// already on disk when run from the repository root.
fn default_docs() -> PathBuf {
  PathBuf::from("docs")
}

/// Index every text file under `roots`, reporting what was embedded and what was reused,
/// and drop what the roots no longer hold.
async fn index(store: &VectorStore, roots: &[PathBuf]) -> anyhow::Result<()> {
  let (mut files, mut embedded, mut reused) = (0, 0, 0);
  let mut live = HashSet::new();
  for path in roots.iter().flat_map(|root| walk(root)) {
    let Ok(text) = std::fs::read_to_string(&path) else {
      continue; // Not UTF-8: not a document.
    };
    if text.trim().is_empty() {
      continue;
    }
    let source = path.display().to_string();
    let report = store
      .index_text(
        DOCS,
        &source,
        &text,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        serde_json::json!({}),
      )
      .await?;
    if report.ids.is_empty() {
      continue;
    }
    live.insert(source);
    files += 1;
    embedded += report.embedded;
    reused += report.reused;
  }
  let dropped = drop_missing_sources(store, &live).await?;
  println!(
    "Indexed {files} files into `{}`: {embedded} chunks embedded, {reused} reused from a \
     previous run, {dropped} files from earlier runs dropped.",
    store.location()
  );
  Ok(())
}

/// Delete the indexed sources that are not in `live`: files removed, emptied or renamed
/// since they were indexed, and directories no longer named. Left in, they would keep
/// being returned by `retrieve` as passages of documents that are not there — or that the
/// user took out precisely because they should not be found.
async fn drop_missing_sources(
  store: &VectorStore,
  live: &HashSet<String>,
) -> anyhow::Result<usize> {
  let mut dropped = 0;
  for source in store.sources(DOCS).await? {
    if !live.contains(&source) {
      store.delete_source(DOCS, &source).await?;
      dropped += 1;
    }
  }
  Ok(dropped)
}

/// Whether a file is named like something that holds credentials. Judged by name because
/// the contents of a private key file have nothing else to recognize them by — no
/// extension on `id_rsa`, none on `credentials` — and by substring because the names that
/// matter are `prod-secrets.yaml`, `old_passwords.txt`, `private-notes`.
///
/// Over-inclusive on purpose, as the rest of [`walk`]: a document wrongly left out is one
/// `--` argument away from being renamed, and a key wrongly sent is not coming back.
fn named_like_a_secret(file_name: &str) -> bool {
  const MARKERS: [&str; 6] = [
    "secret",
    "credential",
    "password",
    "passwd",
    "private",
    "token",
  ];
  let name = file_name.to_ascii_lowercase();
  name.starts_with("id_") || MARKERS.iter().any(|marker| name.contains(marker))
}

/// Text files under `root`, skipping what is never worth embedding — and, more to the
/// point, what must not leave the machine: hidden files and directories (`.env`, `.git`,
/// `.ssh`), build output, key and certificate files, and anything over
/// [`MAX_FILE_BYTES`].
///
/// Whatever this returns is sent to the embedding provider and stored in plain text in
/// the SQLite file, so the exclusions err on the side of leaving things out.
fn walk(root: &Path) -> Vec<PathBuf> {
  const SKIPPED_DIRS: [&str; 3] = ["target", "node_modules", "dist"];
  const SKIPPED_EXTENSIONS: [&str; 12] = [
    "pem", "key", "p12", "pfx", "crt", "cer", "keystore", "jks", "ppk", "kdbx", "p8", "gpg",
  ];

  WalkDir::new(root)
    .into_iter()
    .filter_entry(|entry| {
      // The root itself is whatever the user named, even if it is spelled `.` or `..`.
      if entry.depth() == 0 {
        return true;
      }
      let name = entry.file_name().to_string_lossy();
      // Hidden files as well as directories: `docs/.env` is as much a secret as `./.env`.
      !(name.starts_with('.')
        || (entry.file_type().is_dir() && SKIPPED_DIRS.contains(&name.as_ref())))
    })
    .filter_map(Result::ok)
    .filter(|entry| entry.file_type().is_file())
    .filter(|entry| !named_like_a_secret(&entry.file_name().to_string_lossy()))
    .filter(|entry| {
      entry
        .path()
        .extension()
        .and_then(|extension| extension.to_str())
        .is_none_or(|extension| {
          !SKIPPED_EXTENSIONS
            .iter()
            .any(|skipped| extension.eq_ignore_ascii_case(skipped))
        })
    })
    .filter(|entry| {
      entry
        .metadata()
        .map(|metadata| metadata.len() <= MAX_FILE_BYTES)
        .unwrap_or(false)
    })
    .map(|entry| entry.into_path())
    .collect()
}

#[cfg(test)]
mod tests {
  use agent::knowledge_base::embed::Embedder;
  use serde_json::json;

  use super::*;

  /// Every text is the same vector: these tests are about what is kept and removed, not
  /// about ranking.
  struct Flat;

  #[async_trait::async_trait]
  impl Embedder for Flat {
    fn model(&self) -> &str {
      "flat"
    }
    async fn embed(&self, texts: &[String]) -> agent::Result<Vec<Vec<f32>>> {
      Ok(texts.iter().map(|_| vec![1.0, 0.0]).collect())
    }
  }

  fn in_memory_store() -> VectorStore {
    VectorStore::open_in_memory(Arc::new(Flat)).unwrap()
  }

  /// A scratch directory removed when dropped, so it is cleaned up when an assertion fails
  /// as well as when the test passes.
  struct ScratchDir(PathBuf);

  impl ScratchDir {
    fn new() -> Self {
      Self(std::env::temp_dir().join(format!("rag-walk-{}", uuid::Uuid::new_v4())))
    }
  }

  impl Drop for ScratchDir {
    fn drop(&mut self) {
      let _ = std::fs::remove_dir_all(&self.0);
    }
  }

  /// The reported hole: only hidden *directories* were skipped, so a `.env` sitting in a
  /// docs folder was read, sent to the embedding provider and stored.
  #[test]
  fn secrets_and_hidden_files_are_never_collected() {
    let guard = ScratchDir::new();
    let root = guard.0.clone();
    std::fs::create_dir_all(root.join("sub")).unwrap();
    std::fs::create_dir_all(root.join(".git")).unwrap();
    for (path, body) in [
      ("guide.md", "ordinary"),
      ("sub/notes.txt", "ordinary"),
      (".env", "API_KEY=sk-secret"),
      ("sub/.env.local", "TOKEN=secret"),
      (".git/config", "[core]"),
      ("server.pem", "-----BEGIN CERTIFICATE-----"),
      ("sub/id.KEY", "-----BEGIN PRIVATE KEY-----"),
    ] {
      std::fs::write(root.join(path), body).unwrap();
    }

    let mut found: Vec<String> = walk(&root)
      .into_iter()
      .map(|path| path.strip_prefix(&root).unwrap().display().to_string())
      .collect();
    found.sort();

    assert_eq!(found, ["guide.md", "sub/notes.txt"]);
  }

  /// Key files are recognized by name: `id_rsa` has no extension, and neither has
  /// `credentials`.
  #[test]
  fn files_named_like_secrets_are_never_collected() {
    let guard = ScratchDir::new();
    let root = guard.0.clone();
    std::fs::create_dir_all(&root).unwrap();
    for name in [
      "guide.md",
      "id_rsa",
      "id_ed25519",
      "credentials",
      "prod-secrets.yaml",
      "Old_Passwords.txt",
      "private-notes.md",
      "api_token.txt",
      "release.kdbx",
    ] {
      std::fs::write(root.join(name), "body").unwrap();
    }

    let found: Vec<String> = walk(&root)
      .into_iter()
      .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
      .collect();

    assert_eq!(found, ["guide.md"]);
  }

  #[test]
  fn ordinary_names_are_not_mistaken_for_secrets() {
    for name in [
      "README.md",
      "design.md",
      "identity.md",
      "idea.txt",
      "notes_2024.txt",
    ] {
      assert!(!named_like_a_secret(name), "`{name}`");
    }
  }

  /// The index mirrors the directories named: what is no longer there stops being found.
  #[tokio::test]
  async fn a_source_that_is_gone_is_dropped_and_the_others_are_kept() {
    let store = in_memory_store();
    for source in ["docs/kept.md", "docs/deleted.md", "old-root/a.md"] {
      store
        .put_source(DOCS, source, vec![format!("text of {source}")], json!({}))
        .await
        .unwrap();
    }
    store
      .put_source(
        "other",
        "elsewhere.md",
        vec!["not ours".to_owned()],
        json!({}),
      )
      .await
      .unwrap();
    let live: HashSet<String> = ["docs/kept.md".to_owned()].into();

    let dropped = drop_missing_sources(&store, &live).await.unwrap();

    assert_eq!(dropped, 2);
    assert_eq!(store.sources(DOCS).await.unwrap(), ["docs/kept.md"]);
    assert_eq!(
      store.sources("other").await.unwrap(),
      ["elsewhere.md"],
      "another collection is none of its business"
    );
    assert_eq!(drop_missing_sources(&store, &live).await.unwrap(), 0);
  }

  /// The memories a run saved are removed after it, and only that scope's.
  #[tokio::test]
  async fn the_cleanup_removes_this_runs_memories_and_no_others() {
    let store = in_memory_store();
    let ours = MemoryStore::new(store.clone(), "example-run");
    let theirs = MemoryStore::new(store, "someone-else");
    ours.remember("I work on Atlas").await.unwrap();
    ours.remember("I prefer Chinese").await.unwrap();
    theirs.remember("Unrelated fact").await.unwrap();

    assert_eq!(forget_everything(&ours).await.unwrap(), 2);

    assert!(ours.list().await.unwrap().is_empty());
    assert_eq!(theirs.list().await.unwrap().len(), 1);
    assert_eq!(
      forget_everything(&ours).await.unwrap(),
      0,
      "and it is repeatable"
    );
  }
}
