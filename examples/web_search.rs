use agent::{config, llm::complete::chat_complete, telemetry, tools::ToolRegistry};

const SYSTEM_PROMPT: &str = "You are a general-purpose assistant. When you use search \
                             results, cite the source URL.";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
  telemetry::init()?;

  // Needs fresh information, so the model has to reach for the web_search tool.
  let answer = chat_complete(
    config::model(),
    Some(SYSTEM_PROMPT),
    "What is the latest stable Rust version, and what did it change?",
    &ToolRegistry::builtin()?,
  )
  .await?;

  tracing::info!("Answer: {answer}");

  Ok(())
}
