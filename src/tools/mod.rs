//! Tool registry and the tools themselves.

pub mod calculator;
pub mod mcp;
pub mod registry;
pub mod tool;
pub mod web_search;

pub use registry::ToolRegistry;
pub use tool::Tool;
