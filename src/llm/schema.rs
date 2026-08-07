//! Parts shared by both structured-output strategies that are independent of control flow.
//!
//! Structured output currently has two paths, each with its own reason to exist and cannot be
//! merged into one:
//! - [`crate::llm::structured`]: stateless, uses `response_format` (native json_schema /
//!   json_object); the server enforces the structure;
//! - [`crate::agent::runtime::Agent::run_structured`]: stateful, needs to wrap up **mid-tool-loop**,
//!   which `response_format` cannot express, so a synthetic `final_answer` tool carries `T`'s
//!   JSON Schema, and the model calling it signals the end.
//!
//! But both paths must derive an OpenAI naming-constraint-compliant schema name from `T`. This
//! logic used to be written in two places (and the agent side missed the name sanitize); it is now
//! unified here to avoid logic drift.

use async_openai::types::chat::{
  ChatCompletionTool, ChatCompletionTools, FunctionObjectArgs, ResponseFormat,
  ResponseFormatJsonSchema,
};
use serde_json::Value;

/// The synthetic tool name the model uses to submit the final answer in a structured agent loop.
///
/// Extracted as a constant: it both enters the definition of [`final_answer_tool`] and must be
/// matched by name when a tool call arrives, so scattering it as string literals risks updating
/// one spot and forgetting the other.
pub const FINAL_ANSWER_TOOL_NAME: &str = "final_answer";

/// Derive the schema name from a type name.
///
/// OpenAI requires `name` to contain only `[a-zA-Z0-9_-]`, so strip the module path and generic
/// parameters. For example `crate::models::ActionPlan` → `ActionPlan`, `Vec<ActionPlan>` → `Vec`.
pub fn schema_name<T: ?Sized>() -> String {
  let full = std::any::type_name::<T>();
  let without_generics = full.split('<').next().unwrap_or(full);
  without_generics
    .rsplit("::")
    .next()
    .unwrap_or(without_generics)
    .to_owned()
}

/// Validate a schema `name` supplied at runtime (as opposed to [`schema_name`], which derives
/// one from a compile-time type and is therefore always valid).
///
/// OpenAI's constraint on `function.name` / `response_format.json_schema.name`: 1-64 characters,
/// `[a-zA-Z0-9_-]` only. A name that violates this reaches this crate from the outside — e.g. a
/// `response_schema` field on an HTTP request — so it must be rejected here, with a message
/// naming the actual problem, rather than left to fail as an opaque error from the model API.
pub fn validate_schema_name(name: &str) -> anyhow::Result<()> {
  if name.is_empty() || name.len() > 64 {
    anyhow::bail!(
      "schema name must be 1-64 characters long, got {} ({name:?})",
      name.len()
    );
  }
  if !name
    .chars()
    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
  {
    anyhow::bail!("schema name must only contain [a-zA-Z0-9_-], got {name:?}");
  }
  Ok(())
}

/// Build the `final_answer` tool: carries `T`'s JSON Schema for a structured agent loop to end.
///
/// Used by [`crate::agent::runtime::Agent::run_structured`]. The name is fixed to
/// [`FINAL_ANSWER_TOOL_NAME`], and the description includes the schema name so both logs and the
/// model can tell what it is meant to produce.
///
/// # Errors
///
/// Returns `Err` when name / description / parameters cannot be assembled into an API-accepted
/// definition; this is a programming error, not a runtime condition.
pub fn final_answer_tool<T: schemars::JsonSchema>() -> anyhow::Result<ChatCompletionTools> {
  let schema_json = serde_json::to_value(schemars::schema_for!(T))?;
  final_answer_tool_from_schema(&schema_name::<T>(), schema_json)
}

/// Build the `final_answer` tool from a caller-supplied JSON Schema `Value` rather than a
/// compile-time Rust type, for [`crate::agent::runtime::Agent::run_structured_raw`] — the
/// counterpart to [`final_answer_tool`] for callers (e.g. the HTTP API) that only have a schema
/// at runtime.
///
/// `name` is used only in the tool's description (the tool itself is always named
/// [`FINAL_ANSWER_TOOL_NAME`]); callers that accept it from the outside should validate it with
/// [`validate_schema_name`] first.
///
/// # Errors
///
/// Returns `Err` when the description / parameters cannot be assembled into an API-accepted
/// definition.
pub fn final_answer_tool_from_schema(
  name: &str,
  schema: Value,
) -> anyhow::Result<ChatCompletionTools> {
  let function = FunctionObjectArgs::default()
    .name(FINAL_ANSWER_TOOL_NAME)
    .description(format!(
      "Return the final answer as a `{name}` object matching the required schema."
    ))
    .parameters(schema)
    .build()
    .map_err(|e| anyhow::anyhow!("Failed to build `{FINAL_ANSWER_TOOL_NAME}` tool: {e}"))?;

  Ok(ChatCompletionTools::Function(ChatCompletionTool {
    function,
  }))
}

/// Build the native `json_schema` response format for `T`.
///
/// `strict = true` requires the schema to satisfy OpenAI's strict subset (no unknown
/// fields, all properties `required`); the target type should add
/// `#[schemars(deny_unknown_fields)]` and avoid `Option` fields.
pub fn native_schema_format<T: schemars::JsonSchema>() -> ResponseFormat {
  native_schema_format_from_value(
    &schema_name::<T>(),
    schemars::schema_for!(T).as_value().clone(),
  )
}

/// Build the native `json_schema` response format from a caller-supplied JSON Schema `Value`,
/// the counterpart to [`native_schema_format`] for [`crate::agent::runtime::Agent::run_structured_raw`].
///
/// Same `strict = true` caveat as [`native_schema_format`], except the caller — not this crate —
/// is responsible for the schema satisfying OpenAI's strict subset, since there is no Rust type
/// to add `#[schemars(deny_unknown_fields)]` to; a schema that does not will fail at the model
/// API rather than at this call.
pub fn native_schema_format_from_value(name: &str, schema: Value) -> ResponseFormat {
  ResponseFormat::JsonSchema {
    json_schema: ResponseFormatJsonSchema {
      description: None,
      name: name.to_owned(),
      schema,
      strict: Some(true),
    },
  }
}

/// Render `T`'s JSON Schema as a system instruction, for services that only support
/// `json_object` (structure is guided by the prompt rather than server-enforced).
///
/// A `static` inside a generic fn is shared across monomorphizations and cannot be cached
/// per type, so this regenerates each call — negligible next to one network request.
pub fn schema_instruction<T: schemars::JsonSchema>() -> String {
  let schema = schemars::schema_for!(T);
  schema_instruction_from_value(schema.as_value())
}

/// Render a caller-supplied JSON Schema `Value` as a system instruction, the counterpart to
/// [`schema_instruction`] for [`crate::agent::runtime::Agent::run_structured_raw`].
pub fn schema_instruction_from_value(schema: &Value) -> String {
  let schema_json = serde_json::to_string_pretty(schema).unwrap_or_else(|_| schema.to_string());
  format!(
    "You must reply with a single valid JSON object that conforms to the following JSON Schema. \
     Do not include any explanation, markdown code fences, or extra text.\n\nJSON Schema:\n\
     {schema_json}"
  )
}

/// Concatenate the caller's system prompt with an internal instruction; a blank system is
/// treated as absent.
pub fn merge_system(system: Option<&str>, instruction: &str) -> String {
  match system.map(str::trim).filter(|s| !s.is_empty()) {
    Some(system) => format!("{system}\n\n{instruction}"),
    None => instruction.to_owned(),
  }
}

/// Strip a surrounding ```` ```json ... ``` ```` fence, if present.
///
/// Even when the prompt forbids it, models still occasionally wrap JSON in a code fence;
/// tolerate it rather than fail to parse.
pub fn strip_code_fence(content: &str) -> &str {
  let trimmed = content.trim();
  let Some(rest) = trimmed.strip_prefix("```") else {
    return trimmed;
  };
  // Drop the language-tag line immediately following the opening fence (e.g. ```json).
  let body = rest.split_once('\n').map_or(rest, |(_, body)| body);
  body.trim_end().strip_suffix("```").unwrap_or(body).trim()
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::models::action_plan::ActionPlan;

  #[test]
  fn schema_name_strips_module_path() {
    assert_eq!(schema_name::<ActionPlan>(), "ActionPlan");
  }

  #[test]
  fn schema_name_strips_generics() {
    assert_eq!(schema_name::<Vec<ActionPlan>>(), "Vec");
  }

  #[test]
  fn final_answer_tool_is_named_and_describes_the_schema() {
    let ChatCompletionTools::Function(function) = final_answer_tool::<ActionPlan>().unwrap() else {
      panic!("expected a function tool");
    };
    assert_eq!(function.function.name, FINAL_ANSWER_TOOL_NAME);
    let description = function.function.description.unwrap_or_default();
    assert!(description.contains("ActionPlan"), "got: {description}");
  }

  #[test]
  fn validate_schema_name_accepts_the_openai_charset() {
    assert!(validate_schema_name("Action_Plan-1").is_ok());
  }

  #[test]
  fn validate_schema_name_rejects_empty_and_oversized_names() {
    assert!(validate_schema_name("").is_err());
    assert!(validate_schema_name(&"a".repeat(65)).is_err());
    assert!(validate_schema_name(&"a".repeat(64)).is_ok());
  }

  #[test]
  fn validate_schema_name_rejects_disallowed_characters() {
    assert!(validate_schema_name("action plan").is_err());
    assert!(validate_schema_name("action.plan").is_err());
  }

  #[test]
  fn final_answer_tool_from_schema_uses_the_given_name_and_schema() {
    let schema = serde_json::json!({"type": "object", "properties": {"x": {"type": "integer"}}});
    let ChatCompletionTools::Function(function) =
      final_answer_tool_from_schema("Custom", schema.clone()).unwrap()
    else {
      panic!("expected a function tool");
    };
    assert_eq!(function.function.name, FINAL_ANSWER_TOOL_NAME);
    assert!(
      function
        .function
        .description
        .unwrap_or_default()
        .contains("Custom")
    );
    assert_eq!(function.function.parameters, Some(schema));
  }

  #[test]
  fn native_schema_format_from_value_carries_the_given_name_and_schema() {
    let schema = serde_json::json!({"type": "object"});
    let ResponseFormat::JsonSchema { json_schema } =
      native_schema_format_from_value("Custom", schema.clone())
    else {
      panic!("expected a json_schema response format");
    };
    assert_eq!(json_schema.name, "Custom");
    assert_eq!(json_schema.schema, schema);
    assert_eq!(json_schema.strict, Some(true));
  }

  #[test]
  fn native_schema_format_delegates_to_the_value_based_builder() {
    let ResponseFormat::JsonSchema { json_schema } = native_schema_format::<ActionPlan>() else {
      panic!("expected a json_schema response format");
    };
    assert_eq!(json_schema.name, "ActionPlan");
  }

  #[test]
  fn schema_instruction_from_value_embeds_the_schema() {
    let schema = serde_json::json!({"type": "object"});
    let instruction = schema_instruction_from_value(&schema);
    assert!(instruction.contains("JSON Schema"));
    assert!(instruction.contains("\"type\""));
  }

  #[test]
  fn merge_system_joins_or_falls_back() {
    assert_eq!(merge_system(Some("role"), "rule"), "role\n\nrule");
    assert_eq!(merge_system(Some("  "), "rule"), "rule");
    assert_eq!(merge_system(None, "rule"), "rule");
  }

  #[test]
  fn strip_code_fence_unwraps_fenced_json() {
    assert_eq!(strip_code_fence("```json\n{\"a\":1}\n```"), "{\"a\":1}");
    assert_eq!(strip_code_fence("```\n{\"a\":1}\n```"), "{\"a\":1}");
    assert_eq!(strip_code_fence("  {\"a\":1}  "), "{\"a\":1}");
  }
}
