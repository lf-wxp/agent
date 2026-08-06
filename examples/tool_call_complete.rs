use agent::{config, llm::complete::chat_complete, telemetry, tools::ToolRegistry};

const SYSTEM_PROMPT: &str = "You are a general-purpose assistant";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
  telemetry::init()?;

  let model = config::model();
  let registry = ToolRegistry::builtin()?;

  // No tool needed: the model should answer directly.
  let answer = chat_complete(model, Some(SYSTEM_PROMPT), "尼泊尔的首都是哪里", &registry).await?;
  tracing::info!("Response one: {answer}");

  // Arithmetic: the model is expected to call the calculator tool.
  let answer = chat_complete(
    model,
    Some(SYSTEM_PROMPT),
    "5875 乘以 467 是多少",
    &registry,
  )
  .await?;
  tracing::info!("Response two: {answer}");

  Ok(())
}
