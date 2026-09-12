//! [`LlmRequest`] — the mutable, per-round copy of everything that will be sent to the
//! model, handed to [`crate::agent::BeforeLlmCallback`] hooks before it is converted into
//! API messages.
//!
//! [`ExecutionContext::events`](crate::agent::ExecutionContext::events) is the
//! authoritative, append-only transcript and is never
//! modified through this path. `LlmRequest` is rebuilt from those events on every round,
//! so whatever a callback does to it stays local to the single API call being prepared —
//! which is what makes "trim the prompt" and "keep the full history on disk" coexist (see
//! [`crate::agent::session`]).
//!
//! The request is *complete*: it carries the agent's system prompt alongside the
//! conversation, rather than leaving it to be prepended afterwards. A hook that measures
//! the request therefore measures what is actually sent — without that, a token budget
//! systematically undercounts by the length of the system prompt, which is exactly the
//! part a caller is most likely to have made large.
//!
//! This module deliberately holds only the data shape; the hook contract lives with the
//! other callback traits in [`crate::agent::callback`], and concrete strategies live in
//! [`crate::callback::context_optimizer`].

use crate::agent::{ContentItem, Event};

/// What will be sent to the model this round, in a form callbacks can still edit.
///
/// Emitted in field order: [`Self::instructions`], then [`Self::contents`].
#[derive(Debug, Clone, Default)]
pub struct LlmRequest {
  /// System-level directives, emitted in order ahead of the conversation.
  ///
  /// The agent's own standing prompt, when it has one, is simply the first entry —
  /// there is no separate field for it, because on the wire there is no separate thing:
  /// every entry here becomes a system message. Hooks append their own (a recap of
  /// dropped history, a schema hint) with [`Self::push_instruction`], and are free to
  /// rewrite or drop what is already there; as with everything in this struct, that
  /// affects one request and never the agent itself.
  ///
  /// Because every entry lands in a system message, a hook that pushes text *derived from
  /// untrusted material* — a recap of fetched pages, a retrieved snippet — is putting
  /// that material on the request's highest-authority channel. That can be the right
  /// call: it is also the only part of the request
  /// [`crate::callback::context_optimizer::eviction`] cannot trim, which is precisely why
  /// a recap standing in for already-dropped history belongs here. But it is a trade, not
  /// a free choice, and such a hook owes the reader a label marking the content as
  /// reference data rather than direction (see
  /// [`crate::callback::context_optimizer::Summarization`] for the shape that takes).
  /// Anything that does *not* need to survive trimming should be a
  /// [`ContentItem::Message`] instead.
  pub instructions: Vec<String>,
  /// Flat sequence of [`ContentItem`]s derived from
  /// [`ExecutionContext::events`](crate::agent::ExecutionContext::events) — the
  /// actual conversation to send.
  pub contents: Vec<ContentItem>,
}

impl LlmRequest {
  /// Build the per-round request from the agent's system prompt and its transcript.
  ///
  /// This is a copy by necessity — callbacks mutate it freely and must not be able to
  /// reach back into the transcript — so the destination is sized up front rather than
  /// grown item by item.
  pub fn new(system: Option<String>, events: &[Event]) -> Self {
    let mut contents = Vec::with_capacity(events.iter().map(|event| event.content.len()).sum());
    for event in events {
      contents.extend(event.content.iter().cloned());
    }
    Self {
      instructions: system.into_iter().collect(),
      contents,
    }
  }

  /// Append one more system-level directive for this round only, after those already
  /// present.
  pub fn push_instruction(&mut self, instruction: impl Into<String>) {
    self.instructions.push(instruction.into());
  }
}
