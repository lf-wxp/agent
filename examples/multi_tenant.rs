//! Demonstrates serving multiple tenants from one process with [`agent::llm::provider::Provider`].
//!
//! Each tenant gets its own credentials and its own concurrency budget: tenant `a`'s
//! prompts never share a rate-limit slot with tenant `b`'s, and a wrong/expired key for
//! one tenant cannot affect the other. This is the building block for running an
//! `agent`-based service on behalf of several independent customers/API keys at once.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example multi_tenant
//! ```
//!
//! Both tenants read the *same* `OPENAI_API_KEY`/`OPENAI_BASE_URL` here for the example to
//! run out of the box; in a real multi-tenant setup each `Provider` would be built with a
//! distinct key (e.g. `OpenAIConfig::new().with_api_key(tenant.api_key)`), typically looked
//! up from wherever tenant records are stored.

use std::sync::Arc;

use agent::{Agent, config, llm::provider::Provider, telemetry, tools::ToolRegistry};
use async_openai::config::OpenAIConfig;
use tokio::task::JoinSet;
use tracing::Instrument;

const SYSTEM_PROMPT: &str = "You are a general-purpose assistant.";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
  telemetry::init()?;

  let toolbox = Arc::new(ToolRegistry::empty());

  // Two tenants, two independent concurrency budgets. `tenant_a` is allowed 2 requests
  // in flight at once, `tenant_b` only 1 — a burst from `a` cannot starve `b`'s single
  // slot, and vice versa, because they never share a semaphore.
  let tenant_a = (
    "tenant-a",
    Provider::new(OpenAIConfig::new(), 2),
    vec![
      "What is the capital of Nepal?",
      "Explain recursion using an everyday analogy.",
    ],
  );
  let tenant_b = (
    "tenant-b",
    Provider::new(OpenAIConfig::new(), 1),
    vec![
      "What is the difference between Arc and Rc in Rust?",
      "What is a deadlock, and how can it be avoided?",
    ],
  );

  let mut set = JoinSet::new();
  for (tenant, provider, prompts) in [tenant_a, tenant_b] {
    let agent = Agent::new(
      provider,
      config::model(),
      Some(SYSTEM_PROMPT),
      Arc::clone(&toolbox),
    );
    let agent = Arc::new(agent);

    for prompt in prompts {
      let agent = Arc::clone(&agent);
      let span = tracing::info_span!("chat", tenant, prompt);
      set.spawn(
        async move {
          let result = agent.run(prompt).await?;
          Ok::<_, anyhow::Error>((tenant, prompt, result.output))
        }
        .instrument(span),
      );
    }
  }

  while let Some(joined) = set.join_next().await {
    match joined {
      Ok(Ok((tenant, prompt, output))) => tracing::info!("[{tenant}] {prompt}\n{output}"),
      Ok(Err(err)) => tracing::error!("task failed: {err:#}"),
      Err(err) => tracing::error!("task panicked: {err}"),
    }
  }

  Ok(())
}
