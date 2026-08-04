//! agent 库入口。
//!
//! 对外暴露 LLM 调用能力与数据模型，供 bin target（`src/main.rs`）与
//! `examples/` 下的示例共同复用，避免同一份源码被多次编译。
pub mod llm;
pub mod models;
