use reqwest::StatusCode;

use crate::{
  config,
  gaia::models::{GaiaRow, HfResponse},
  http,
  util::truncate_chars,
};

/// HF datasets-server rows query endpoint.
const HF_ROWS_ENDPOINT: &str = "https://datasets-server.huggingface.co/rows";

/// GAIA dataset repo ID (a gated repo; accept the access terms on the web page first).
const GAIA_DATASET: &str = "gaia-benchmark/GAIA";

/// Max rows per single request to the /rows endpoint.
const MAX_ROWS_PER_REQUEST: usize = 100;

/// Length (in chars) of the response body kept in error messages, to avoid log bloat.
const BODY_PREVIEW_CHARS: usize = 512;

/// Fetch some rows of the given GAIA config/split from the HF datasets-server.
pub async fn load_gaia_rows(
  config: &str,
  split: &str,
  offset: usize,
  length: usize,
) -> anyhow::Result<Vec<GaiaRow>> {
  anyhow::ensure!(
    (1..=MAX_ROWS_PER_REQUEST).contains(&length),
    "length must be in 1..={MAX_ROWS_PER_REQUEST}, got {length}"
  );

  let token = config::hf_token().ok_or_else(|| {
    anyhow::anyhow!(
      "HF_TOKEN is not set; create a read token at https://huggingface.co/settings/tokens"
    )
  })?;

  let offset = offset.to_string();
  let length = length.to_string();
  let response = http::client()
    .get(HF_ROWS_ENDPOINT)
    .query(&[
      ("dataset", GAIA_DATASET),
      ("config", config),
      ("split", split),
      ("offset", offset.as_str()),
      ("length", length.as_str()),
    ])
    .bearer_auth(token)
    .send()
    .await?;

  let status = response.status();
  // Fetch the text first, then parse: on failure this lets us carry the server's `{"error": ...}` into the error,
  // otherwise a direct `.json::<HfResponse>()` would report "missing field rows" and hide the real cause.
  let body = response.text().await?;

  if !status.is_success() {
    // For unauthorized gated repos, datasets-server uniformly returns 404 (not exposing whether the repo exists) instead of 403.
    if status == StatusCode::NOT_FOUND {
      anyhow::bail!(
        "HF datasets-server returned 404 for `{GAIA_DATASET}`. It is a gated dataset: open \
         https://huggingface.co/datasets/{GAIA_DATASET} and accept the access conditions with the \
         same account that owns HF_TOKEN (unauthorized gated repos are reported as 404, not 403). \
         Also verify config=`{config}` / split=`{split}` exist. Response: {}",
        truncate_chars(&body, BODY_PREVIEW_CHARS)
      );
    }
    anyhow::bail!(
      "HF datasets-server request failed with {status}. Response: {}",
      truncate_chars(&body, BODY_PREVIEW_CHARS)
    );
  }

  let payload = serde_json::from_str::<HfResponse>(&body).map_err(|err| {
    anyhow::anyhow!(
      "Failed to parse HF rows response: {err}; body: {}",
      truncate_chars(&body, BODY_PREVIEW_CHARS)
    )
  })?;

  tracing::debug!(config, split, rows = payload.rows.len(), "loaded GAIA rows");

  Ok(payload.rows.into_iter().map(|r| r.row).collect())
}

/// Load GAIA level 1 data.
///
/// Note: the `test` split does not expose `Final answer` (returns an empty string) and cannot be scored,
/// so the `validation` split is used here.
pub async fn load_gaia_level1(length: usize) -> anyhow::Result<Vec<GaiaRow>> {
  load_gaia_rows("2023_level1", "validation", 0, length).await
}
