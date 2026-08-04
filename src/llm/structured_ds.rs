use async_openai::types::chat::{
  ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
  CreateChatCompletionRequestArgs, ResponseFormat,
};

use crate::models::action_plan::ActionPlan;

pub async fn chat_complete_structured_ds(
  model: &str,
  system: Option<&str>,
  prompt: &str,
) -> anyhow::Result<ActionPlan> {
  let client = async_openai::Client::new();

  // DeepSeek 仅支持 response_format = json_object，不支持 json_schema。
  // 因此将 JSON Schema 注入到 system 提示中，引导模型输出符合结构的 JSON。
  let schema = schemars::schema_for!(ActionPlan);
  let schema_json = serde_json::to_string_pretty(&schema)?;
  let schema_instruction = format!(
    "You must reply with a single valid JSON object that conforms to the following JSON Schema. \
     Do not include any explanation, markdown code fences, or extra text.\n\nJSON Schema:\n{schema_json}"
  );
  let system_content = match system {
    Some(s) => format!("{s}\n\n{schema_instruction}"),
    None => schema_instruction,
  };

  let mut messages = vec![];
  messages.push(
    ChatCompletionRequestSystemMessageArgs::default()
      .content(system_content)
      .build()?
      .into(),
  );
  messages.push(
    ChatCompletionRequestUserMessageArgs::default()
      .content(prompt)
      .build()?
      .into(),
  );

  let request = CreateChatCompletionRequestArgs::default()
    .model(model)
    .messages(messages)
    .response_format(ResponseFormat::JsonObject)
    .max_tokens(2048u32)
    .build()?;
  let response = client.chat().create(request).await?;

  tracing::info!("Response {:#?}", response);

  let plan = response
    .choices
    .into_iter()
    .next()
    .and_then(|c| c.message.content)
    .ok_or_else(|| anyhow::anyhow!("No content in response"))
    .and_then(|s| serde_json::from_str(&s).map_err(Into::into))?;

  Ok(plan)
}
