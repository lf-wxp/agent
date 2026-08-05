use agent::{config, llm::complete::chat_complete, telemetry, tools::tools};

const SYSTEM_PROMPT: &str = "You are a general-purpose assistant";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
  telemetry::init()?;

  let model = config::model();
  let tools = tools();

  // No tool needed: the model should answer directly.
  let answer = chat_complete(model, Some(SYSTEM_PROMPT), "尼泊尔的首都是哪里", tools).await?;
  tracing::info!("Response one: {answer}");

  // Arithmetic: the model is expected to call the calculator tool.
  let answer = chat_complete(model, Some(SYSTEM_PROMPT), "5875 乘以 467 是多少", tools).await?;
  tracing::info!("Response two: {answer}");

  Ok(())
}
