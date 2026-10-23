//! Estimating what a run cost, from its token usage and a price list.
//!
//! No prices are built in. They change, differ by provider, region and contract, and a
//! wrong number presented as a cost is worse than none — so they are configured, in a
//! JSON file named by `LLM_PRICING_FILE` (see [`PriceTable::from_config`]):
//!
//! ```json
//! {
//!   "currency": "USD",
//!   "models": {
//!     "deepseek-v4-flash": { "input": 0.27, "cached_input": 0.07, "output": 1.10 },
//!     "gpt-4o-mini": { "input": 0.15, "output": 0.60 }
//!   }
//! }
//! ```
//!
//! Prices are per million tokens. `cached_input` — what a prompt-cache hit costs — is
//! optional and defaults to `input`. Model names are matched exactly against the names
//! requests were sent with (the names in [`ExecutionContext::usage_by_model`]).
//!
//! [`ExecutionContext::usage_by_model`]: crate::agent::ExecutionContext::usage_by_model

use std::{
  collections::{BTreeMap, HashMap},
  path::Path,
  sync::LazyLock,
};

use serde::Deserialize;

use crate::{agent::TokenUsage, config};

const TOKENS_PER_PRICE_UNIT: f64 = 1_000_000.0;

/// What one model charges, per million tokens.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelPrice {
  pub input: f64,
  /// What a prompt token served from the provider's cache costs; `input` when unset.
  #[serde(default)]
  pub cached_input: Option<f64>,
  pub output: f64,
}

impl ModelPrice {
  /// What `usage` costs at these prices. Reasoning tokens are part of the completion
  /// tokens and billed as output, so they need no term of their own.
  pub fn cost(&self, usage: &TokenUsage) -> f64 {
    let cached = usage.cached_prompt_tokens.min(usage.prompt_tokens);
    let uncached = usage.prompt_tokens - cached;
    let cached_price = self.cached_input.unwrap_or(self.input);
    (f64::from(uncached) * self.input
      + f64::from(cached) * cached_price
      + f64::from(usage.completion_tokens) * self.output)
      / TOKENS_PER_PRICE_UNIT
  }

  fn validate(&self, model: &str) -> crate::Result<()> {
    let prices = [Some(self.input), self.cached_input, Some(self.output)];
    crate::error::ensure!(
      prices
        .into_iter()
        .flatten()
        .all(|price| price.is_finite() && price >= 0.0),
      config,
      "the prices for `{model}` must be finite, non-negative numbers"
    );
    Ok(())
  }
}

/// A run's estimated cost.
#[derive(Debug, Clone, PartialEq)]
pub struct CostEstimate {
  pub amount: f64,
  pub currency: String,
  /// Models that spent tokens but have no price. Non-empty means `amount` is a lower
  /// bound, not the whole cost.
  pub unpriced_models: Vec<String>,
}

/// Prices by model; see the module docs for the file format.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PriceTable {
  #[serde(default = "default_currency")]
  pub currency: String,
  #[serde(default)]
  models: HashMap<String, ModelPrice>,
}

fn default_currency() -> String {
  "USD".to_owned()
}

impl Default for PriceTable {
  fn default() -> Self {
    Self::new(default_currency())
  }
}

impl PriceTable {
  /// An empty table quoting costs in `currency` (any label: `USD`, `CNY`, `credits`).
  pub fn new(currency: impl Into<String>) -> Self {
    Self {
      currency: currency.into(),
      models: HashMap::new(),
    }
  }

  /// Add (or replace) the price of `model`.
  #[must_use]
  pub fn with_price(mut self, model: impl Into<String>, price: ModelPrice) -> Self {
    self.models.insert(model.into(), price);
    self
  }

  /// Read a price file; see the module docs for the format.
  pub fn load(path: impl AsRef<Path>) -> crate::Result<Self> {
    use crate::{Error, error::ResultExt};

    let path = path.as_ref();
    let text = std::fs::read_to_string(path).wrap_err_with(Error::config, || {
      format!("failed to read the price file `{}`", path.display())
    })?;
    let table: Self = serde_json::from_str(&text).wrap_err_with(Error::config, || {
      format!("`{}` is not a valid price file", path.display())
    })?;
    for (model, price) in &table.models {
      price.validate(model)?;
    }
    Ok(table)
  }

  /// The table named by `LLM_PRICING_FILE`, read once per process. Empty when the
  /// variable is unset; also empty — with a warning, rather than an error — when the
  /// file cannot be read or parsed, since a broken price list should cost the cost
  /// estimate, not the run.
  pub fn from_config() -> &'static Self {
    static TABLE: LazyLock<PriceTable> = LazyLock::new(|| match config::pricing_file() {
      None => PriceTable::default(),
      Some(path) => PriceTable::load(&path).unwrap_or_else(|err| {
        tracing::warn!("ignoring LLM_PRICING_FILE: {err:#}");
        PriceTable::default()
      }),
    });
    &TABLE
  }

  pub fn is_empty(&self) -> bool {
    self.models.is_empty()
  }

  pub fn price(&self, model: &str) -> Option<&ModelPrice> {
    self.models.get(model)
  }

  /// What `usage_by_model` cost, or `None` when not one model that spent anything has a
  /// price — there is nothing to estimate then, and a `0` would read as "free".
  pub fn estimate(&self, usage_by_model: &BTreeMap<String, TokenUsage>) -> Option<CostEstimate> {
    let mut amount = 0.0;
    let mut priced_any = false;
    let mut unpriced_models = Vec::new();
    for (model, usage) in usage_by_model.iter().filter(|(_, usage)| !usage.is_empty()) {
      match self.price(model) {
        Some(price) => {
          amount += price.cost(usage);
          priced_any = true;
        }
        None => unpriced_models.push(model.clone()),
      }
    }
    priced_any.then(|| CostEstimate {
      amount,
      currency: self.currency.clone(),
      unpriced_models,
    })
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn usage(prompt: u32, cached: u32, completion: u32) -> TokenUsage {
    TokenUsage {
      prompt_tokens: prompt,
      completion_tokens: completion,
      total_tokens: prompt + completion,
      cached_prompt_tokens: cached,
      reasoning_tokens: 0,
    }
  }

  const PRICE: ModelPrice = ModelPrice {
    input: 2.0,
    cached_input: Some(0.5),
    output: 8.0,
  };

  #[test]
  fn cached_prompt_tokens_are_billed_at_the_cached_price() {
    // 600k uncached at 2.0, 400k cached at 0.5, 100k output at 8.0 per million.
    let cost = PRICE.cost(&usage(1_000_000, 400_000, 100_000));
    assert!((cost - (1.2 + 0.2 + 0.8)).abs() < 1e-9, "got {cost}");
  }

  #[test]
  fn without_a_cached_price_a_cache_hit_costs_the_full_input_price() {
    let price = ModelPrice {
      cached_input: None,
      ..PRICE
    };
    assert!((price.cost(&usage(1_000_000, 1_000_000, 0)) - 2.0).abs() < 1e-9);
  }

  #[test]
  fn an_estimate_covers_every_priced_model_and_names_the_rest() {
    let table = PriceTable::new("USD").with_price("big", PRICE);
    let spent = BTreeMap::from([
      ("big".to_owned(), usage(1_000_000, 0, 0)),
      ("mystery".to_owned(), usage(10, 0, 10)),
      ("idle".to_owned(), TokenUsage::default()),
    ]);

    let estimate = table.estimate(&spent).unwrap();
    assert!((estimate.amount - 2.0).abs() < 1e-9);
    assert_eq!(estimate.currency, "USD");
    assert_eq!(
      estimate.unpriced_models,
      ["mystery"],
      "an idle model is not missing a price"
    );
  }

  /// Nothing priced is not the same as free.
  #[test]
  fn no_estimate_without_a_single_priced_model() {
    let spent = BTreeMap::from([("mystery".to_owned(), usage(10, 0, 10))]);
    assert!(PriceTable::default().estimate(&spent).is_none());
    assert!(
      PriceTable::new("USD")
        .with_price("other", PRICE)
        .estimate(&spent)
        .is_none()
    );
  }

  #[test]
  fn loads_the_documented_file_format() {
    let dir = std::env::temp_dir().join(format!("agent-pricing-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("prices.json");
    std::fs::write(
      &path,
      r#"{ "currency": "CNY", "models": {
           "a": { "input": 1, "cached_input": 0.1, "output": 4 },
           "b": { "input": 2, "output": 8 } } }"#,
    )
    .unwrap();

    let table = PriceTable::load(&path).unwrap();
    assert_eq!(table.currency, "CNY");
    assert_eq!(table.price("a").unwrap().cached_input, Some(0.1));
    assert_eq!(table.price("b").unwrap().cached_input, None);
    let _ = std::fs::remove_dir_all(dir);
  }

  /// A typo in a field name must not silently price a model at zero.
  #[test]
  fn a_malformed_price_file_is_rejected() {
    let dir = std::env::temp_dir().join(format!("agent-pricing-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    for (name, body) in [
      (
        "typo",
        // cspell:disable-next-line — a misspelt field is the point of this case.
        r#"{ "models": { "a": { "input": 1, "ouput": 4 } } }"#,
      ),
      (
        "negative",
        r#"{ "models": { "a": { "input": -1, "output": 4 } } }"#,
      ),
      ("not-json", "input = 1"),
    ] {
      let path = dir.join(format!("{name}.json"));
      std::fs::write(&path, body).unwrap();
      assert!(PriceTable::load(&path).is_err(), "`{name}` was accepted");
    }
    let _ = std::fs::remove_dir_all(dir);
  }
}
