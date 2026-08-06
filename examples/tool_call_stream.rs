use agent::{
  config,
  llm::{provider::Provider, stream::chat_stream_with_retry},
  telemetry,
  tools::ToolRegistry,
};

const SYSTEM_PROMPT: &str = "You are a general-purpose assistant";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
  telemetry::init()?;

  let model = config::model();
  let registry = ToolRegistry::builtin()?;

  // Streamed tool call: the calculator runs mid-stream and the model continues with the result.
  let output = chat_stream_with_retry(
    Provider::shared(),
    model,
    Some(SYSTEM_PROMPT),
    "先用计算器算 5875乘以 467，再把结果除以 5，只给出最终数字",
    &registry,
  )
  .await?;

  tracing::info!("Streamed answer: {output}");

  Ok(())
}
