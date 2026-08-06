//! Model Context Protocol integration.
//!
//! An MCP server's tools are adapted to the local [`crate::tools::Tool`] trait, so the
//! rest of the agent cannot tell a remote tool from a built-in one.
//!
//! Servers are declared in an `mcp.json` file; see [`config`].

pub mod client;
pub mod config;

pub use client::{McpConnection, McpTool};
pub use config::{McpConfig, ServerConfig, TransportKind};
