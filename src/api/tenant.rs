//! Tenant config for the HTTP agent server: which bearer token maps to which
//! [`Provider`] (credentials + concurrency budget), loaded from a JSON file.
//!
//! There is no session/account system here on purpose: a "tenant" is just a mapping from
//! one opaque bearer token to one set of upstream LLM credentials. Rotating a tenant's
//! token, changing its API key, or adjusting its concurrency budget only means editing
//! this file and restarting the server.
//!
//! ```json
//! {
//!   "tenants": {
//!     "${env:TENANT_A_TOKEN}": {
//!       "label": "tenant-a",
//!       "apiKey": "${env:TENANT_A_API_KEY}",
//!       "maxConcurrency": 5
//!     },
//!     "${env:TENANT_B_TOKEN}": {
//!       "label": "tenant-b",
//!       "apiKey": "${env:TENANT_B_API_KEY}",
//!       "baseUrl": "https://tenant-b.example.com/v1",
//!       "maxConcurrency": 1,
//!       "defaultModel": "gpt-4o-mini"
//!     }
//!   }
//! }
//! ```
//!
//! ## Trust boundary
//!
//! Like `mcp.json` (see [`crate::tools::mcp::config`]), this file and the path override
//! (`AGENT_TENANTS_PATH`) are meant to be set by whoever operates the server, not by
//! end users: anyone who can edit it can mint themselves a valid bearer token.

use std::{collections::BTreeMap, collections::HashMap, path::Path};

use anyhow::Context;
use async_openai::config::OpenAIConfig;
use serde::Deserialize;

use crate::{config, llm::provider::Provider, tools::mcp::config::expand};

/// One tenant: who they are (for logging) and what they may talk to.
#[derive(Clone, Debug)]
pub struct Tenant {
  pub label: String,
  pub provider: Provider,
  /// Falls back to [`config::model`] when a request does not specify one.
  pub default_model: Option<String>,
}

/// Bearer token -> [`Tenant`] lookup table.
#[derive(Default, Debug)]
pub struct TenantRegistry {
  tenants: HashMap<String, Tenant>,
}

impl TenantRegistry {
  /// Read and parse a tenant config file. A missing file is not an error: it just means
  /// no tenant can authenticate, which the caller (the server binary) should treat as
  /// "misconfigured" rather than crash on, since a `--help`-style dry run should still work.
  pub async fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
    let path = path.as_ref();
    let text = tokio::fs::read_to_string(path)
      .await
      .with_context(|| format!("failed to read tenant config `{}`", path.display()))?;

    Self::from_json(&text)
  }

  /// Parse from JSON text.
  pub fn from_json(text: &str) -> anyhow::Result<Self> {
    let raw: RawTenants = serde_json::from_str(text).context("failed to parse tenant config")?;
    let mut tenants = HashMap::with_capacity(raw.tenants.len());

    for (token, entry) in raw.tenants {
      let token = expand(&token)?;
      let api_key = expand(&entry.api_key)?;
      let base_url = entry.base_url.as_deref().map(expand).transpose()?;
      let max_concurrency = entry
        .max_concurrency
        .unwrap_or_else(config::max_concurrency);

      let mut openai_config = OpenAIConfig::new().with_api_key(api_key);
      if let Some(base_url) = base_url {
        openai_config = openai_config.with_api_base(base_url);
      }

      let label = match entry.label {
        Some(label) => expand(&label)?,
        None => token.clone(),
      };
      let tenant = Tenant {
        label,
        provider: Provider::new(openai_config, max_concurrency),
        default_model: entry.default_model,
      };

      anyhow::ensure!(
        tenants.insert(token, tenant).is_none(),
        "duplicate tenant token in config (after `${{...}}` expansion)"
      );
    }

    Ok(Self { tenants })
  }

  /// Resolve a bearer token to its tenant, if any.
  pub fn lookup(&self, token: &str) -> Option<&Tenant> {
    self.tenants.get(token)
  }

  pub fn len(&self) -> usize {
    self.tenants.len()
  }

  pub fn is_empty(&self) -> bool {
    self.tenants.is_empty()
  }
}

#[derive(Debug, Default, Deserialize)]
struct RawTenants {
  #[serde(default)]
  tenants: BTreeMap<String, RawTenant>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawTenant {
  #[serde(default)]
  label: Option<String>,
  api_key: String,
  #[serde(default)]
  base_url: Option<String>,
  #[serde(default)]
  max_concurrency: Option<usize>,
  #[serde(default)]
  default_model: Option<String>,
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn resolves_a_token_to_its_tenant() {
    let registry = TenantRegistry::from_json(
      r#"{"tenants": {"secret-a": {"label": "tenant-a", "apiKey": "sk-a", "maxConcurrency": 2}}}"#,
    )
    .unwrap();

    let tenant = registry.lookup("secret-a").unwrap();
    assert_eq!(tenant.label, "tenant-a");
    assert!(registry.lookup("nope").is_none());
  }

  #[test]
  fn label_defaults_to_the_token_itself() {
    let registry =
      TenantRegistry::from_json(r#"{"tenants": {"secret-b": {"apiKey": "sk-b"}}}"#).unwrap();

    assert_eq!(registry.lookup("secret-b").unwrap().label, "secret-b");
  }

  #[test]
  fn max_concurrency_falls_back_to_the_shared_default() {
    let registry =
      TenantRegistry::from_json(r#"{"tenants": {"secret-c": {"apiKey": "sk-c"}}}"#).unwrap();

    assert!(registry.lookup("secret-c").is_some());
  }

  #[test]
  fn expands_environment_variables_in_the_token_and_api_key() {
    // SAFETY: single-threaded test, and the names are unique to this test.
    unsafe {
      std::env::set_var("AGENT_TEST_TENANT_TOKEN", "expanded-token");
      std::env::set_var("AGENT_TEST_TENANT_KEY", "sk-expanded");
    }

    let registry = TenantRegistry::from_json(
      r#"{"tenants": {"${env:AGENT_TEST_TENANT_TOKEN}": {"apiKey": "${env:AGENT_TEST_TENANT_KEY}"}}}"#,
    )
    .unwrap();

    assert!(registry.lookup("expanded-token").is_some());
    assert!(registry.lookup("${env:AGENT_TEST_TENANT_TOKEN}").is_none());
  }

  #[test]
  fn rejects_duplicate_tokens_after_expansion() {
    // SAFETY: single-threaded test, and the name is unique to this test.
    unsafe { std::env::set_var("AGENT_TEST_DUP_TOKEN", "same-token") };

    let err = TenantRegistry::from_json(
      r#"{"tenants": {
        "${env:AGENT_TEST_DUP_TOKEN}": {"apiKey": "sk-a"},
        "same-token": {"apiKey": "sk-b"}
      }}"#,
    )
    .unwrap_err();

    assert!(err.to_string().contains("duplicate tenant token"));
  }

  #[test]
  fn defaults_to_no_tenants() {
    assert!(TenantRegistry::from_json("{}").unwrap().is_empty());
  }
}
