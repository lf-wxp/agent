use agent::{
  config, llm::structured::chat_complete_structured, models::action_plan::ActionPlan, telemetry,
  tools::ToolRegistry,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
  telemetry::init()?;

  let model = config::model();
  let plan = chat_complete_structured::<ActionPlan>(
    model,
    Some("You are a general-purpose assistant"),
    "I'm going to the USA, Canada, and Mexico for the World Cup; how should I plan the trip?",
    &ToolRegistry::empty(),
  )
  .await?;

  println!("Response: {plan:#?}");

  Ok(())
}
