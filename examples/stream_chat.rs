use agent::{
  config,
  llm::{provider::Provider, stream::chat_stream_with_retry},
  telemetry,
  tools::ToolRegistry,
};
use tokio::task::JoinSet;
use tracing::Instrument;

const SYSTEM_PROMPT: &str = "You are a general-purpose assistant";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
  telemetry::init()?;

  let prompts = [
    "Explain Rust's ownership model in three sentences",
    "What is async programming, and how does it differ from multithreading",
    "Explain the TCP three-way handshake",
    "Explain what a large language model is in simple terms",
    "What is the difference between Arc and Rc in Rust",
    "What is RAG, and why is it commonly used in AI applications",
    "What is a deadlock, and how can it be avoided",
    "Explain recursion using an everyday analogy",
    "Why is Rust memory-safe even though it has no GC",
  ];

  // Fetch once up front: `config::model()` returns a 'static value, so it can be moved into each task directly.
  let model = config::model();
  // Concurrency is enforced inside `chat_stream_with_retry` itself now (see
  // `Provider::acquire`), so tasks no longer need to acquire a permit by hand.
  let provider = Provider::shared();

  let mut set = JoinSet::new();
  for prompt in prompts {
    let span = tracing::info_span!("chat", prompt);
    set.spawn(
      async move {
        let output = chat_stream_with_retry(
          provider,
          model,
          Some(SYSTEM_PROMPT),
          prompt,
          &ToolRegistry::empty(),
        )
        .await?;
        Ok::<_, anyhow::Error>((prompt, output))
      }
      .instrument(span),
    );
  }

  while let Some(joined) = set.join_next().await {
    match joined {
      Ok(Ok((prompt, output))) => tracing::info!("\n{prompt}\n{output}"),
      Ok(Err(err)) => tracing::error!("task failed: {err:#}"),
      Err(err) => tracing::error!("task panicked: {err}"),
    }
  }

  Ok(())
}
