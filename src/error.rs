//! [`Error`]: what every fallible function in this crate returns, as [`Result`].
//!
//! One enum, split by what a caller can *do* about a failure rather than by which module
//! raised it:
//!
//! | Variant | Meaning | Retrying helps? |
//! |---|---|---|
//! | [`Error::Cancelled`] | the run's cancellation token fired; carries the transcript so far | no |
//! | [`Error::BudgetExhausted`] | failed after spending the whole tool-round budget | no |
//! | [`Error::TruncatedOutput`] | the answer was cut off by `max_tokens` | no |
//! | [`Error::Provider`] | the model provider failed or answered with something unusable | [`ProviderError::is_retryable`] |
//! | [`Error::InvalidInput`] | an argument was rejected before anything was sent | no |
//! | [`Error::Config`] | configuration is missing or malformed | no |
//! | [`Error::Tool`] | a tool failed | depends on the tool |
//! | [`Error::Mcp`] | an MCP server could not be started, reached or used | maybe |
//! | [`Error::Storage`] | the vector store, memory or another persistent store failed | maybe |
//! | [`Error::Io`] | a filesystem or other I/O operation failed | maybe |
//! | [`Error::Other`] | anything else | maybe |
//!
//! [`crate::llm::retry::is_transient`] encodes the last column.
//!
//! # Messages and causes
//!
//! Every error has a message and may have a cause, like `anyhow`'s context chain:
//!
//! - `{err}` prints the outermost message only — what the model is shown when a tool
//!   fails, so a tool's message should say what to do differently.
//! - `{err:#}` prints the whole chain, `outer: inner: innermost`.
//! - [`std::error::Error::source`] walks the same chain, so `anyhow` (which the binaries
//!   use) prints it too.
//!
//! One level is skipped by `source()`: a [`Detail`] that only carries a cause — what `?`
//! on a [`std::io::Error`] produces — *is* that cause, so it is not repeated. An
//! [`std::io::Error`] behind an [`Error::Io`] therefore does not appear in the `source()`
//! walk; reach it with [`Error::downcast_ref`] instead, e.g. to read its `kind()`.
//!
//! [`Error::context`] adds a message in front of an error without changing its variant,
//! so a [`Error::Provider`] stays one (and stays retryable or not) however many layers
//! describe where it happened. [`ResultExt::wrap_err`] turns a foreign error into one of
//! these, with a message saying what was being done; an error that is already one of
//! these keeps its variant.
//!
//! # Writing a tool or a backend
//!
//! A [`crate::tools::Tool`] returns [`Error::tool`] for its own failures, with a cause
//! attached when there is one:
//!
//! ```
//! use agent::error::{Error, ResultExt};
//!
//! fn parse_port(raw: &str) -> agent::Result<u16> {
//!   raw.parse().wrap_err(Error::tool, format!("`{raw}` is not a port number"))
//! }
//! assert!(matches!(parse_port("x"), Err(Error::Tool(_))));
//! ```
//!
//! A [`crate::llm::backend::ChatBackend`] or [`crate::knowledge_base::embed::Embedder`]
//! reports provider failures as [`ProviderError`], saying whether the same request could
//! succeed if sent again.

use std::{error::Error as StdError, fmt};

use crate::agent::ExecutionContext;

/// `Result` with this crate's [`Error`] as the default error type.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Any error, as a cause: what [`Error::with_source`] and [`ResultExt::wrap_err`] take.
pub type BoxError = Box<dyn StdError + Send + Sync + 'static>;

/// See the module docs.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
  /// The run was cancelled through its token. Carries the transcript up to that point,
  /// so the caller can persist the partial turn.
  #[error(transparent)]
  Cancelled(#[from] Cancelled),
  /// A failure that came after the tool-round budget had run out: a retry would spend
  /// the whole budget again before failing the same way. The failure itself is the
  /// cause ([`BudgetExhausted::inner`]).
  #[error(transparent)]
  BudgetExhausted(#[from] BudgetExhausted),
  /// The answer was cut off by `max_tokens`, so it is empty or incomplete.
  #[error(transparent)]
  TruncatedOutput(#[from] TruncatedOutput),
  /// The model provider failed, or answered with something that could not be used.
  #[error(transparent)]
  Provider(#[from] ProviderError),
  /// An argument was rejected before anything was sent: a tool name the API would not
  /// accept, an empty prompt, a malformed schema name.
  #[error(transparent)]
  InvalidInput(Detail),
  /// Configuration is missing or malformed: an unset `EMBED_MODEL`, a broken `mcp.json`
  /// or price list.
  #[error(transparent)]
  Config(Detail),
  /// A tool failed. The message is what the model is shown.
  #[error(transparent)]
  Tool(Detail),
  /// An MCP server could not be started, reached, or used.
  #[error(transparent)]
  Mcp(Detail),
  /// A persistent store failed: the vector store, long-term memory.
  #[error(transparent)]
  Storage(Detail),
  /// A filesystem or other I/O operation failed. The [`std::io::Error`] is the cause.
  #[error(transparent)]
  Io(Detail),
  /// Anything else.
  #[error(transparent)]
  Other(Detail),
}

impl Error {
  /// An [`Error::InvalidInput`] saying `message`.
  pub fn invalid_input(message: impl Into<String>) -> Self {
    Self::InvalidInput(Detail::new(message))
  }

  /// An [`Error::Config`] saying `message`.
  pub fn config(message: impl Into<String>) -> Self {
    Self::Config(Detail::new(message))
  }

  /// An [`Error::Tool`] saying `message`.
  pub fn tool(message: impl Into<String>) -> Self {
    Self::Tool(Detail::new(message))
  }

  /// An [`Error::Mcp`] saying `message`.
  pub fn mcp(message: impl Into<String>) -> Self {
    Self::Mcp(Detail::new(message))
  }

  /// An [`Error::Storage`] saying `message`.
  pub fn storage(message: impl Into<String>) -> Self {
    Self::Storage(Detail::new(message))
  }

  /// An [`Error::Io`] saying `message`; attach the [`std::io::Error`] with
  /// [`Self::with_source`], or use `?` on one directly.
  pub fn io(message: impl Into<String>) -> Self {
    Self::Io(Detail::new(message))
  }

  /// An [`Error::Other`] saying `message`.
  pub fn other(message: impl Into<String>) -> Self {
    Self::Other(Detail::new(message))
  }

  /// Attach `source` as the cause of this error, replacing any it had.
  ///
  /// Only the message-and-cause variants (and [`Error::Provider`]) have a cause to set;
  /// the others — [`Error::Cancelled`], [`Error::BudgetExhausted`],
  /// [`Error::TruncatedOutput`] — are returned unchanged.
  #[must_use]
  pub fn with_source(self, source: impl Into<BoxError>) -> Self {
    let source = source.into();
    match self {
      Self::Provider(provider) => Self::Provider(provider.with_source(source)),
      other => other.map_detail(|detail| detail.with_source(source)),
    }
  }

  /// Put `message` in front of this error, keeping its variant: the old error becomes
  /// the cause, so `{err}` shows `message` and `{err:#}` shows `message: old`.
  ///
  /// [`Error::Cancelled`], [`Error::BudgetExhausted`] and [`Error::TruncatedOutput`]
  /// carry data a caller matches on, so they are returned unchanged rather than buried
  /// under a message.
  #[must_use]
  pub fn context(self, message: impl Into<String>) -> Self {
    let message = message.into();
    match self {
      Self::Provider(provider) => Self::Provider(provider.context(message)),
      Self::InvalidInput(detail) => Self::InvalidInput(detail.context(message)),
      Self::Config(detail) => Self::Config(detail.context(message)),
      Self::Tool(detail) => Self::Tool(detail.context(message)),
      Self::Mcp(detail) => Self::Mcp(detail.context(message)),
      Self::Storage(detail) => Self::Storage(detail.context(message)),
      Self::Io(detail) => Self::Io(detail.context(message)),
      Self::Other(detail) => Self::Other(detail.context(message)),
      unchanged @ (Self::Cancelled(_) | Self::BudgetExhausted(_) | Self::TruncatedOutput(_)) => {
        unchanged
      }
    }
  }

  /// The [`ProviderError`] this is, if it is one.
  pub fn as_provider(&self) -> Option<&ProviderError> {
    match self {
      Self::Provider(provider) => Some(provider),
      _ => None,
    }
  }

  /// A short, stable name for the variant (`"provider"`, `"tool"`, ...), e.g. for a
  /// metric label or a trace's `error.type`.
  pub fn kind(&self) -> &'static str {
    match self {
      Self::Cancelled(_) => "cancelled",
      Self::BudgetExhausted(_) => "budget_exhausted",
      Self::TruncatedOutput(_) => "truncated_output",
      Self::Provider(_) => "provider",
      Self::InvalidInput(_) => "invalid_input",
      Self::Config(_) => "config",
      Self::Tool(_) => "tool",
      Self::Mcp(_) => "mcp",
      Self::Storage(_) => "storage",
      Self::Io(_) => "io",
      Self::Other(_) => "other",
    }
  }

  /// The first error of type `E` in this error's chain — the payload itself, then every
  /// cause, including one this error displays transparently (what `?` on an
  /// [`std::io::Error`] produces):
  ///
  /// ```
  /// let err = agent::Error::from(std::io::Error::from(std::io::ErrorKind::NotFound));
  /// let io = err.downcast_ref::<std::io::Error>().unwrap();
  /// assert_eq!(io.kind(), std::io::ErrorKind::NotFound);
  /// ```
  pub fn downcast_ref<E: StdError + 'static>(&self) -> Option<&E> {
    let mut next = Some(self.payload());
    while let Some(err) = next {
      if let Some(found) = err.downcast_ref::<E>() {
        return Some(found);
      }
      next = inner_of(err);
    }
    None
  }

  fn payload(&self) -> &(dyn StdError + 'static) {
    match self {
      Self::Cancelled(inner) => inner,
      Self::BudgetExhausted(inner) => inner,
      Self::TruncatedOutput(inner) => inner,
      Self::Provider(inner) => inner,
      Self::InvalidInput(detail)
      | Self::Config(detail)
      | Self::Tool(detail)
      | Self::Mcp(detail)
      | Self::Storage(detail)
      | Self::Io(detail)
      | Self::Other(detail) => detail,
    }
  }

  fn map_detail(self, f: impl FnOnce(Detail) -> Detail) -> Self {
    match self {
      Self::InvalidInput(detail) => Self::InvalidInput(f(detail)),
      Self::Config(detail) => Self::Config(f(detail)),
      Self::Tool(detail) => Self::Tool(f(detail)),
      Self::Mcp(detail) => Self::Mcp(f(detail)),
      Self::Storage(detail) => Self::Storage(f(detail)),
      Self::Io(detail) => Self::Io(f(detail)),
      Self::Other(detail) => Self::Other(f(detail)),
      unchanged => unchanged,
    }
  }
}

impl From<std::io::Error> for Error {
  /// An [`Error::Io`] showing the I/O error's own message, with the I/O error as the
  /// cause (so its `kind()` can still be read).
  fn from(err: std::io::Error) -> Self {
    Self::Io(Detail::transparent(Box::new(err)))
  }
}

impl From<async_openai::error::OpenAIError> for Error {
  /// A request the client refused to build is [`Error::InvalidInput`]; everything else
  /// is a [`ProviderError`], classified by [`ProviderError::from`].
  fn from(err: async_openai::error::OpenAIError) -> Self {
    match err {
      async_openai::error::OpenAIError::InvalidArgument(_) => {
        Self::InvalidInput(Detail::transparent(Box::new(err)))
      }
      other => Self::Provider(other.into()),
    }
  }
}

#[cfg(feature = "sqlite")]
impl From<rusqlite::Error> for Error {
  /// An [`Error::Storage`] showing SQLite's own message, with the SQLite error as the
  /// cause (reach it with [`Error::downcast_ref`]).
  fn from(err: rusqlite::Error) -> Self {
    Self::Storage(Detail::transparent(Box::new(err)))
  }
}

/// The next error down from `err`, seeing through this module's own wrappers: a
/// transparent [`Detail`] or [`ProviderError`] reports its cause's cause as its
/// `source()`, which would skip the cause itself.
fn inner_of<'a>(err: &'a (dyn StdError + 'static)) -> Option<&'a (dyn StdError + 'static)> {
  if let Some(detail) = err.downcast_ref::<Detail>() {
    return detail
      .source
      .as_deref()
      .map(|source| source as &(dyn StdError + 'static));
  }
  if let Some(provider) = err.downcast_ref::<ProviderError>() {
    return provider
      .source
      .as_deref()
      .map(|source| source as &(dyn StdError + 'static));
  }
  if let Some(error) = err.downcast_ref::<Error>() {
    return Some(error.payload());
  }
  err.source()
}

/// A message, a cause, or both: the payload of the message-and-cause variants of
/// [`Error`].
///
/// With only a cause it is *transparent*: it displays as the cause and its
/// [`source`](StdError::source) is the cause's own, so wrapping adds no line to the
/// chain. That is what `?` on a [`std::io::Error`] produces.
#[derive(Debug)]
pub struct Detail {
  message: Option<String>,
  source: Option<BoxError>,
}

impl Detail {
  fn new(message: impl Into<String>) -> Self {
    Self {
      message: Some(message.into()),
      source: None,
    }
  }

  fn transparent(source: BoxError) -> Self {
    Self {
      message: None,
      source: Some(source),
    }
  }

  fn with_source(mut self, source: BoxError) -> Self {
    self.source = Some(source);
    self
  }

  fn context(self, message: String) -> Self {
    Self {
      message: Some(message),
      source: Some(Box::new(self)),
    }
  }

  /// The message this level adds, if it adds one (a transparent wrapper does not).
  pub fn message(&self) -> Option<&str> {
    self.message.as_deref()
  }
}

impl fmt::Display for Detail {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match (&self.message, &self.source) {
      (Some(message), _) => f.write_str(message)?,
      (None, Some(source)) => write!(f, "{source}")?,
      (None, None) => f.write_str("unknown error")?,
    }
    write_causes(f, self.source())
  }
}

impl StdError for Detail {
  fn source(&self) -> Option<&(dyn StdError + 'static)> {
    let source = self.source.as_deref()?;
    if self.message.is_some() {
      Some(source)
    } else {
      source.source()
    }
  }
}

/// A failure from the model provider. See [`Error::Provider`].
#[derive(Debug)]
pub struct ProviderError {
  message: Option<String>,
  status: Option<u16>,
  retryable: bool,
  class: &'static str,
  source: Option<BoxError>,
}

impl ProviderError {
  /// A failure the same request could plausibly survive if sent again: a dropped
  /// connection, rate limiting, a 5xx, a garbled response.
  pub fn transient(message: impl Into<String>) -> Self {
    Self::new(Some(message.into()), true)
  }

  /// A failure the same request would meet again: a rejected key, an unknown model, a
  /// request the API calls malformed. Retrying only delays surfacing it.
  pub fn permanent(message: impl Into<String>) -> Self {
    Self::new(Some(message.into()), false)
  }

  fn new(message: Option<String>, retryable: bool) -> Self {
    Self {
      message,
      status: None,
      retryable,
      class: "provider",
      source: None,
    }
  }

  /// Record the HTTP status the provider answered with.
  #[must_use]
  pub fn with_status(mut self, status: u16) -> Self {
    self.status = Some(status);
    self
  }

  /// Attach the underlying error as the cause.
  #[must_use]
  pub fn with_source(mut self, source: impl Into<BoxError>) -> Self {
    self.source = Some(source.into());
    self
  }

  fn context(self, message: String) -> Self {
    Self {
      message: Some(message),
      status: self.status,
      retryable: self.retryable,
      class: self.class,
      source: Some(Box::new(self)),
    }
  }

  /// The HTTP status the provider answered with, when it answered at all.
  pub fn status(&self) -> Option<u16> {
    self.status
  }

  /// Whether sending the same request again could succeed.
  pub fn is_retryable(&self) -> bool {
    self.retryable
  }

  /// What kind of failure this was, for a trace's `error.type`: the status code when
  /// there is one, else `transport`, `stream`, `invalid_response` or `provider`.
  pub(crate) fn class(&self) -> String {
    match self.status {
      Some(status) => status.to_string(),
      None => self.class.to_owned(),
    }
  }
}

impl fmt::Display for ProviderError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match (&self.message, &self.source) {
      (Some(message), _) => f.write_str(message)?,
      (None, Some(source)) => write!(f, "{source}")?,
      (None, None) => f.write_str("the model provider failed")?,
    }
    write_causes(f, self.source())
  }
}

impl StdError for ProviderError {
  fn source(&self) -> Option<&(dyn StdError + 'static)> {
    let source = self.source.as_deref()?;
    if self.message.is_some() {
      Some(source)
    } else {
      source.source()
    }
  }
}

impl From<async_openai::error::OpenAIError> for ProviderError {
  /// Retryable: transport errors, a broken stream, a response that did not parse,
  /// rate limiting (429), a request timeout (408), a conflict (409), and any 5xx. Every
  /// other status the API answers with — a malformed request, a rejected key, a model
  /// the tenant cannot reach — is not: the identical request fails identically.
  fn from(err: async_openai::error::OpenAIError) -> Self {
    use async_openai::error::OpenAIError;

    let (status, retryable, class) = match &err {
      OpenAIError::ApiError(response) => {
        let status = response.status_code;
        let retryable = status.is_server_error() || matches!(status.as_u16(), 408 | 409 | 429);
        (Some(status.as_u16()), retryable, "provider")
      }
      OpenAIError::Reqwest(inner) => (inner.status().map(|s| s.as_u16()), true, "transport"),
      OpenAIError::StreamError(_) => (None, true, "stream"),
      OpenAIError::JSONDeserialize(..) => (None, true, "invalid_response"),
      OpenAIError::InvalidArgument(_) => (None, false, "invalid_request"),
      #[allow(unreachable_patterns)]
      _ => (None, true, "provider"),
    };
    Self {
      message: None,
      status,
      retryable,
      class,
      source: Some(Box::new(err)),
    }
  }
}

/// The run was stopped through its [`ExecutionContext::cancellation`] token. See
/// [`Error::Cancelled`].
///
/// An error because the run did not finish, but it carries the transcript up to that
/// point: every completed round, plus a result for each call that was cut short saying
/// so. A caller that persists history can file [`Self::context`] and the conversation
/// carries on from there; one that does not care just sees "the run was cancelled".
///
/// Only tools are cut short. A before-tool hook already waiting — typically an approval
/// prompt — is left to its own timeout rather than dropped mid-wait, since dropping it
/// would leave the prompt it raised dangling in every front-end showing it.
#[derive(Debug)]
pub struct Cancelled {
  /// The transcript up to the cancellation: what ran is recorded, a tool cut short is
  /// recorded as cancelled.
  pub context: Box<ExecutionContext>,
}

impl Cancelled {
  pub(crate) fn new(context: ExecutionContext) -> Self {
    Self {
      context: Box::new(context),
    }
  }
}

impl fmt::Display for Cancelled {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("the run was cancelled")
  }
}

impl StdError for Cancelled {}

/// A failure that came after the tool-round budget ran out. See
/// [`Error::BudgetExhausted`].
#[derive(Debug)]
pub struct BudgetExhausted {
  inner: Box<Error>,
}

impl BudgetExhausted {
  /// Mark `err` as having happened after the whole tool-round budget was spent.
  pub fn new(err: Error) -> Self {
    Self {
      inner: Box::new(err),
    }
  }

  /// The failure itself.
  pub fn inner(&self) -> &Error {
    &self.inner
  }

  /// The failure itself, by value.
  pub fn into_inner(self) -> Error {
    *self.inner
  }
}

impl fmt::Display for BudgetExhausted {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str(
      "failed after the tool round budget was spent, so retrying would repeat the full budget",
    )?;
    write_causes(f, self.source())
  }
}

impl StdError for BudgetExhausted {
  fn source(&self) -> Option<&(dyn StdError + 'static)> {
    Some(self.inner.as_ref())
  }
}

/// The answer was cut off by `max_tokens`. See [`Error::TruncatedOutput`].
#[derive(Debug)]
pub struct TruncatedOutput {
  /// The output cap the request was sent with.
  pub limit: u32,
}

impl fmt::Display for TruncatedOutput {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(
      f,
      "output truncated by max_tokens ({}): the reasoning chain consumed the whole budget, so the \
       JSON body is empty or incomplete",
      self.limit
    )
  }
}

impl StdError for TruncatedOutput {}

/// In alternate mode (`{:#}`), append `: cause` for every cause in the chain.
///
/// Every level writes only its own message in normal mode, so a printer that walks the
/// chain itself (`anyhow`'s `{:#}`, a logger) does not see a cause twice.
fn write_causes(
  f: &mut fmt::Formatter<'_>,
  mut next: Option<&(dyn StdError + 'static)>,
) -> fmt::Result {
  if !f.alternate() {
    return Ok(());
  }
  while let Some(cause) = next {
    write!(f, ": {cause}")?;
    next = cause.source();
  }
  Ok(())
}

/// Turn a foreign error into an [`Error`] of a chosen variant, with a message saying
/// what was being done:
///
/// ```
/// use agent::error::{Error, ResultExt};
///
/// let read = std::fs::read_to_string("/no/such/file")
///   .wrap_err(Error::config, "failed to read the price list");
/// let err = read.unwrap_err();
/// assert!(matches!(err, Error::Config(_)));
/// assert!(format!("{err:#}").starts_with("failed to read the price list: "));
/// ```
///
/// `kind` is one of [`Error`]'s message constructors ([`Error::tool`],
/// [`Error::storage`], ...). The original error becomes the cause.
///
/// An error that is *already* an [`Error`] keeps its variant: `kind` only applies to
/// errors from other crates. A blanket `E: Into<BoxError>` would otherwise turn the run's
/// [`Error::Cancelled`] into whatever the call site passed, losing the payload a caller
/// matches on — so wrapping one adds only the message.
pub trait ResultExt<T> {
  /// See the trait docs.
  fn wrap_err(self, kind: fn(String) -> Error, message: impl Into<String>) -> Result<T>;

  /// [`Self::wrap_err`], building the message only on failure.
  fn wrap_err_with<M: Into<String>>(
    self,
    kind: fn(String) -> Error,
    message: impl FnOnce() -> M,
  ) -> Result<T>;

  /// Convert to `kind`'s variant without adding a message: the error displays as itself,
  /// and stays reachable as the cause. An [`Error`] again keeps its own variant.
  ///
  /// ```
  /// use agent::error::{Error, ResultExt};
  ///
  /// let err = "x".parse::<u16>().err_as(Error::tool).unwrap_err();
  /// assert!(matches!(err, Error::Tool(_)));
  /// assert_eq!(err.to_string(), "invalid digit found in string");
  /// ```
  fn err_as(self, kind: fn(String) -> Error) -> Result<T>;
}

/// The shared body of [`ResultExt`]: reclassify a foreign error, keep one of ours.
fn adopt(err: BoxError, kind: fn(String) -> Error, message: Option<String>) -> Error {
  match err.downcast::<Error>() {
    Ok(own) => match message {
      Some(message) => own.context(message),
      None => *own,
    },
    Err(foreign) => match message {
      Some(message) => kind(message).with_source(foreign),
      None => kind(String::new()).map_detail(|_| Detail::transparent(foreign)),
    },
  }
}

impl<T, E: Into<BoxError>> ResultExt<T> for std::result::Result<T, E> {
  fn wrap_err(self, kind: fn(String) -> Error, message: impl Into<String>) -> Result<T> {
    self.map_err(|err| adopt(err.into(), kind, Some(message.into())))
  }

  fn wrap_err_with<M: Into<String>>(
    self,
    kind: fn(String) -> Error,
    message: impl FnOnce() -> M,
  ) -> Result<T> {
    self.map_err(|err| adopt(err.into(), kind, Some(message().into())))
  }

  fn err_as(self, kind: fn(String) -> Error) -> Result<T> {
    self.map_err(|err| adopt(err.into(), kind, None))
  }
}

/// [`Error::context`] on a `Result`: put a message in front of the error, keeping its
/// variant.
pub trait Context<T> {
  /// ```rust
  /// use agent::error::{Context, Error};
  ///
  /// let err = Err::<(), _>(Error::from(agent::error::ProviderError::permanent("401")))
  ///   .context("calling the backend")
  ///   .unwrap_err();
  /// assert!(err.as_provider().is_some(), "the variant survives");
  /// assert_eq!(err.to_string(), "calling the backend");
  /// ```
  fn context(self, message: impl Into<String>) -> Result<T>;

  /// [`Self::context`], building the message only on failure.
  fn with_context<M: Into<String>>(self, message: impl FnOnce() -> M) -> Result<T>;
}

impl<T> Context<T> for Result<T> {
  fn context(self, message: impl Into<String>) -> Result<T> {
    self.map_err(|err| err.context(message))
  }

  fn with_context<M: Into<String>>(self, message: impl FnOnce() -> M) -> Result<T> {
    self.map_err(|err| err.context(message()))
  }
}

/// `return Err(Error::$kind(format!(...)))`, e.g. `bail!(tool, "no such file `{path}`")`.
macro_rules! bail {
  ($kind:ident, $($arg:tt)+) => {
    return Err($crate::error::Error::$kind(format!($($arg)+)))
  };
}

/// `if !cond { bail!(...) }`, e.g. `ensure!(n > 0, invalid_input, "n must be positive")`.
macro_rules! ensure {
  ($cond:expr, $kind:ident, $($arg:tt)+) => {
    if !$cond {
      $crate::error::bail!($kind, $($arg)+);
    }
  };
}

pub(crate) use {bail, ensure};

#[cfg(test)]
mod tests {
  use super::*;

  fn io_error() -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::NotFound, "no such file")
  }

  #[test]
  fn display_shows_the_outer_message_and_alternate_shows_the_chain() {
    let err = Error::storage("failed to open the store")
      .with_source(io_error())
      .context("could not start");
    assert_eq!(err.to_string(), "could not start");
    assert_eq!(
      format!("{err:#}"),
      "could not start: failed to open the store: no such file"
    );
    assert!(
      matches!(err, Error::Storage(_)),
      "context keeps the variant"
    );
  }

  /// `?` on an I/O error adds no line of its own, and the I/O error stays reachable.
  #[test]
  fn an_io_error_converts_transparently() {
    let err: Error = io_error().into();
    assert_eq!(err.to_string(), "no such file");
    assert_eq!(format!("{err:#}"), "no such file");
    assert!(StdError::source(&err).is_none());

    let wrapped = Err::<(), _>(io_error())
      .wrap_err(Error::config, "failed to read `prices.json`")
      .unwrap_err();
    let cause = StdError::source(&wrapped)
      .and_then(|cause| cause.downcast_ref::<std::io::Error>())
      .expect("the io::Error is the cause");
    assert_eq!(cause.kind(), std::io::ErrorKind::NotFound);
  }

  /// `anyhow` walks `source()` itself, so it must see each message exactly once.
  #[test]
  fn anyhow_prints_the_same_chain_without_repeats() {
    let err = Error::tool("the tool failed")
      .with_source(io_error())
      .context("outer");
    let anyhow_err = anyhow::Error::from(err);
    assert_eq!(
      format!("{anyhow_err:#}"),
      "outer: the tool failed: no such file"
    );
  }

  #[test]
  fn a_provider_error_keeps_its_status_and_retryability_through_context() {
    let err = Error::from(
      ProviderError::permanent("401 invalid api key")
        .with_status(401)
        .with_source(io_error()),
    )
    .context("embedding request failed");
    let provider = err.as_provider().expect("still a provider error");
    assert_eq!(provider.status(), Some(401));
    assert!(!provider.is_retryable());
    assert_eq!(err.to_string(), "embedding request failed");
    assert_eq!(
      format!("{err:#}"),
      "embedding request failed: 401 invalid api key: no such file"
    );
  }

  #[test]
  fn openai_errors_are_classified_by_status() {
    use async_openai::error::{ApiError, ApiErrorResponse, OpenAIError};

    let api = |status: u16| {
      Error::from(OpenAIError::ApiError(ApiErrorResponse {
        status_code: reqwest::StatusCode::from_u16(status).unwrap(),
        api_error: ApiError {
          message: "nope".to_owned(),
          r#type: None,
          param: None,
          code: None,
          misalignment: None,
        },
      }))
    };
    for (status, retryable) in [
      (429, true),
      (500, true),
      (503, true),
      (408, true),
      (400, false),
      (401, false),
      (404, false),
    ] {
      let err = api(status);
      let provider = err.as_provider().unwrap();
      assert_eq!(provider.status(), Some(status));
      assert_eq!(provider.is_retryable(), retryable, "status {status}");
      assert_eq!(provider.class(), status.to_string());
    }

    let invalid = Error::from(OpenAIError::InvalidArgument("bad".to_owned()));
    assert!(matches!(invalid, Error::InvalidInput(_)), "got {invalid:?}");
  }

  #[test]
  fn special_variants_pass_through_context_and_with_source_unchanged() {
    let err = Error::from(TruncatedOutput { limit: 16 }).context("ignored");
    assert!(matches!(
      err,
      Error::TruncatedOutput(TruncatedOutput { limit: 16 })
    ));

    let err = Error::from(BudgetExhausted::new(Error::other("parse failed")));
    assert_eq!(
      format!("{err:#}"),
      "failed after the tool round budget was spent, so retrying would repeat the full \
       budget: parse failed"
    );
    let Error::BudgetExhausted(budget) = err else {
      panic!("expected BudgetExhausted");
    };
    assert!(matches!(budget.inner(), Error::Other(_)));
  }

  #[test]
  fn the_macros_build_the_named_variant() {
    fn check(n: i32) -> Result<i32> {
      ensure!(n >= 0, invalid_input, "n must not be negative, got {n}");
      if n > 10 {
        bail!(tool, "{n} is too large");
      }
      Ok(n)
    }
    assert_eq!(check(3).unwrap(), 3);
    let err = check(-1).unwrap_err();
    assert!(matches!(err, Error::InvalidInput(_)));
    assert_eq!(err.to_string(), "n must not be negative, got -1");
    assert!(matches!(check(11), Err(Error::Tool(_))));
  }

  #[test]
  fn kinds_are_stable_names() {
    assert_eq!(Error::tool("x").kind(), "tool");
    assert_eq!(Error::from(io_error()).kind(), "io");
    assert_eq!(
      Error::from(ProviderError::transient("x")).kind(),
      "provider"
    );
  }

  fn cancelled() -> Error {
    Error::from(Cancelled::new(crate::agent::ExecutionContext::new()))
  }

  /// `ResultExt` exists to reclassify *another crate's* error. A blanket
  /// `E: Into<BoxError>` also matches this crate's own `Error`, and reclassifying that
  /// would lose the payload a caller matches on — a run that ended `Cancelled` would
  /// read as a tool failure, and an unretryable 401 would become retryable again.
  #[test]
  fn wrapping_keeps_one_of_our_errors_as_what_it_was() {
    // `Cancelled` carries data and, per `Error::context`, passes through unchanged with
    // `kind` ignored. What matters is that the variant survives at all.
    let err = Err::<(), _>(cancelled())
      .wrap_err(Error::tool, "while calling the tool")
      .unwrap_err();
    assert!(matches!(err, Error::Cancelled(_)), "got {err:?}");
    assert_eq!(err.to_string(), "the run was cancelled");

    let err = Err::<(), _>(cancelled()).err_as(Error::config).unwrap_err();
    assert!(matches!(err, Error::Cancelled(_)), "got {err:?}");

    // A provider error does take the message — and keeps its status and retryability.
    let err = Err::<(), _>(Error::from(
      ProviderError::permanent("401").with_status(401),
    ))
    .wrap_err(Error::storage, "saving")
    .unwrap_err();
    assert!(matches!(err, Error::Provider(_)), "got {err:?}");
    let provider = err.as_provider().unwrap();
    assert_eq!(provider.status(), Some(401));
    assert!(!provider.is_retryable());
    assert_eq!(err.to_string(), "saving");

    // A foreign one is reclassified, as before.
    let err = Err::<(), _>(io_error())
      .wrap_err(Error::tool, "reading")
      .unwrap_err();
    assert!(matches!(err, Error::Tool(_)), "got {err:?}");
  }

  /// The `Result`-side companion of [`Error::context`], which nothing else exercised.
  #[test]
  fn the_context_trait_keeps_the_variant_and_builds_the_message_lazily() {
    let mut built = false;
    let err = Ok::<(), _>(())
      .context("never used")
      .and_then(|()| {
        Err::<(), _>(Error::from(
          ProviderError::permanent("401").with_status(401),
        ))
      })
      .with_context(|| {
        built = true;
        "calling the backend"
      })
      .unwrap_err();
    assert!(built, "built on failure");
    assert_eq!(err.as_provider().unwrap().status(), Some(401));
    assert_eq!(err.to_string(), "calling the backend");
  }

  /// Causes stay reachable by type however many layers of `context` sit in between —
  /// `store.rs` finds SQLite's "busy" code this way.
  #[test]
  fn downcast_reaches_a_cause_through_context_layers() {
    let err = Error::from(ProviderError::transient("the request failed").with_source(io_error()))
      .context("while embedding")
      .context("outermost");
    assert_eq!(err.to_string(), "outermost");
    let io = err.downcast_ref::<std::io::Error>().expect("the io::Error");
    assert_eq!(io.kind(), std::io::ErrorKind::NotFound);
    assert!(err.downcast_ref::<ProviderError>().is_some());

    // A transparent wrapper is seen through rather than reported as the cause.
    let err: Error = io_error().into();
    assert!(err.downcast_ref::<std::io::Error>().is_some());
  }

  /// What `ProviderError::from` decides for the variants that carry no HTTP status.
  #[test]
  fn transport_and_parse_failures_are_classified_without_a_status() {
    let reqwest = reqwest::Client::new()
      .get("not a url")
      .build()
      .expect_err("an invalid URL fails to build");
    let provider = ProviderError::from(async_openai::error::OpenAIError::Reqwest(reqwest));
    assert_eq!(provider.status(), None);
    assert!(provider.is_retryable());
    assert_eq!(provider.class(), "transport");

    let parse = ProviderError::from(async_openai::error::OpenAIError::JSONDeserialize(
      serde_json::from_str::<i32>("nope").unwrap_err(),
      "{}".to_owned(),
    ));
    assert_eq!(parse.class(), "invalid_response");
    assert!(parse.is_retryable());

    let stream = ProviderError::from(async_openai::error::OpenAIError::StreamError(Box::new(
      async_openai::error::StreamError::EventStream("connection reset".to_owned()),
    )));
    assert_eq!(stream.class(), "stream");
    assert!(stream.is_retryable());
  }

  /// The binaries convert with `?` into `anyhow`, which needs these.
  #[test]
  fn the_error_is_send_sync_and_static() {
    fn assert_bounds<E: StdError + Send + Sync + 'static>() {}
    assert_bounds::<Error>();
  }
}
