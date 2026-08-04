use anyhow::Ok;
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

use crate::llm::{complete::chat_complete, structured_ds::chat_complete_structured_ds};

mod llm;
mod models;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
  dotenvy::dotenv()?;
  let key = std::env::var("OPENAI_BASE_URL")?;
  println!("key: {}", key);
  let subscriber = FmtSubscriber::builder()
    .with_max_level(Level::INFO)
    .finish();
  tracing::subscriber::set_global_default(subscriber)?;

  // let content = chat_complete(
  //   "deepseek-v4-flash",
  //   Some("你是一个全能的助手"),
  //   "中国的首都是哪里",
  // )
  // .await?;

  let plan = chat_complete_structured_ds(
    "deepseek-v4-flash",
    Some("你是一个全能的助手"),
    "中国的首都是哪里",
  )
  .await?;

  // println!("Response: {content}");
  println!("Response: {plan:#?}");

  Ok(())
}
