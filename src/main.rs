use agent::{
  config, llm::structured::chat_complete_structured, models::action_plan::ActionPlan, telemetry,
  tools::ToolRegistry,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
  telemetry::init()?;

  let plan = chat_complete_structured::<ActionPlan>(
    config::model(),
    Some("You are a general-purpose assistant"),
    "Help me plan a three-day trip to Hangzhou",
    &ToolRegistry::empty(),
  )
  .await?;

  println!("Response: {plan:#?}");

  Ok(())
}
