use std::collections::HashMap;

use serde_json::Value;
use uuid::Uuid;

use super::event::Event;

#[derive(Debug, Clone, Copy, Default)]
pub struct TokenUsage {
  pub prompt_tokens: u32,
  pub completion_tokens: u32,
  pub total_tokens: u32,
}

impl TokenUsage {
  pub fn add(&mut self, prompt_tokens: u32, completion_tokens: u32, total_tokens: u32) {
    self.prompt_tokens += prompt_tokens;
    self.completion_tokens += completion_tokens;
    self.total_tokens += total_tokens;
  }
}

#[derive(Debug)]
pub struct ExecutionContext {
  pub execution_id: String,
  pub events: Vec<Event>,
  pub current_step: u32,
  pub state: HashMap<String, Value>,
  pub final_result: Option<String>,
  pub usage: TokenUsage,
}

impl ExecutionContext {
  pub fn new() -> Self {
    Self {
      execution_id: Uuid::new_v4().to_string(),
      events: Vec::new(),
      current_step: 0,
      state: HashMap::new(),
      final_result: None,
      usage: TokenUsage::default(),
    }
  }

  pub fn add_event(&mut self, event: Event) {
    self.events.push(event);
  }

  pub fn increment_step(&mut self) {
    self.current_step += 1;
  }
}

impl Default for ExecutionContext {
  fn default() -> Self {
    Self::new()
  }
}
