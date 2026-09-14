#![deny(clippy::correctness)]
#![warn(clippy::suspicious, clippy::style, clippy::complexity, clippy::perf)]
// These docs explain how things work, so a public item's documentation routinely points
// at the private helper that implements the step it is describing. `rustdoc` flags that
// on the assumption the reader cannot follow such a link — but this workspace builds its
// docs with `--document-private-items` (see `.cargo/config.toml`), so the targets *are*
// documented and the links resolve. Keeping the lint on would mean warning about links
// that demonstrably work.
#![allow(rustdoc::private_intra_doc_links)]

pub mod agent;
pub mod callback;
pub mod config;
pub mod gaia;
pub mod http;
pub mod knowledge_base;
pub mod llm;
pub mod models;
pub mod telemetry;
pub mod tools;
pub mod util;

pub use agent::session;
pub use agent::{
  Agent, AgentOutcome, AgentResult, AgentRunState, AgentStreamEvent, GiveUp, ResumedDecision,
  RunCheckpoint, StopReason, StructuredAgentResult, SuspendedToolCall,
};
