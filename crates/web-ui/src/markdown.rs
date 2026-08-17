//! Renders assistant/user message text as Markdown, via [`render_markdown`].
//!
//! Parses with [`pulldown_cmark`] (a pure-Rust CommonMark implementation, so it needs no
//! OS/filesystem access and compiles fine for `wasm32-unknown-unknown`) and walks the
//! resulting event stream into real Leptos views directly — **not** by turning it into
//! an HTML string (`pulldown_cmark::html::push_html`) and injecting it via
//! `inner_html`/`dangerously_set_inner_html`. That distinction is a deliberate security
//! boundary, not a style preference: the text handed to [`render_markdown`] is ultimately
//! model output, which can itself be shaped by whatever a tool call fed back into the
//! conversation (a `web_search` result, a file's contents, ...) — content this page does
//! not otherwise trust. Building typed Leptos elements from the parsed *structure*
//! (headings, lists, code spans, ...) means a literal `<script>` appearing in that text
//! is just text, rendered as the characters `<script>`, exactly like any other character
//! sequence — never parsed as markup, because nothing here ever asks the DOM to parse
//! anything. [`pulldown_cmark::Event::Html`]/[`pulldown_cmark::Event::InlineHtml`] (raw
//! HTML embedded in markdown) get the same treatment: printed as literal text, never
//! executed.

use leptos::prelude::*;
use pulldown_cmark::{Alignment, CodeBlockKind, Event, HeadingLevel, Options, Parser, Tag};
use wasm_bindgen_futures::spawn_local;

use crate::i18n::{self, Key, t};

/// Parse `text` as CommonMark (+ GFM tables/strikethrough/task lists) and render it as a
/// tree of real elements, wrapped in one `.markdown-body` container (see `style.css` for
/// how each element type underneath it is styled).
///
/// Single newlines *within* a paragraph (CommonMark "soft breaks") render as an actual
/// line break (`<br>`) here rather than CommonMark's default of collapsing to a single
/// space — deliberately: model output routinely uses a bare `\n` to mean "new line" the
/// way a human writing prose would, not "still the same line, wrap however the viewport
/// wants" — treating it as a real break is what every popular chat UI rendering LLM
/// output does, and collapsing it to a space instead reads as a bug (words visibly
/// running together) rather than a formatting choice a user would recognize as
/// intentional.
///
/// Returns an owned view with `use<>`: nothing in the tree actually borrows from `text`
/// past this call — every [`pulldown_cmark::CowStr`] pulled out of the parser is turned
/// into an owned [`String`] via `.into_string()`/[`plain_text`] before being handed to
/// any view macro — but Rust 2024's default return-position-`impl Trait` lifetime
/// capture would otherwise tie the result to `text`'s lifetime anyway (RPITIT/RPIT
/// capture *every* in-scope lifetime unless told not to), which would make this
/// unusable from a reactive closure that needs to return an owned value with no
/// borrowed `&str` still alive by the time it runs again.
pub fn render_markdown(text: &str) -> impl IntoView + use<> {
  let options = Options::ENABLE_STRIKETHROUGH | Options::ENABLE_TABLES | Options::ENABLE_TASKLISTS;
  let events: Vec<Event> = Parser::new_ext(text, options).collect();
  let mut iter = events.into_iter();
  let nodes = render_events(&mut iter);
  view! { <div class="markdown-body">{nodes}</div> }
}

/// Consumes events from `iter` until it is exhausted, converting each into zero or more
/// views. For a [`Event::Start`], this recurses into [`take_matching_end`] to isolate
/// exactly the events nested inside that tag (see that function's docs) before handing
/// them to [`render_tag`], which recurses back into this function for the tag's own
/// children — the mutual recursion is what lets an arbitrarily nested tree (a list
/// inside a blockquote inside a list item, say) come out right without this needing to
/// track that nesting itself; [`take_matching_end`] already did.
fn render_events<'a>(iter: &mut impl Iterator<Item = Event<'a>>) -> Vec<AnyView> {
  let mut out = Vec::new();
  while let Some(event) = iter.next() {
    match event {
      Event::Start(tag) => {
        let children_events = take_matching_end(iter);
        out.push(render_tag(tag, children_events));
      }
      // Reachable only if a stream ever has an unmatched `End` (should not happen for
      // well-formed input) — [`take_matching_end`] is what consumes every `End` that
      // actually pairs with a `Start` seen above. Ignored rather than treated as an
      // error: a malformed sequence here should degrade to "missing a bit of
      // formatting", not break rendering entirely.
      Event::End(_) => {}
      Event::Text(text) => out.push(text.into_string().into_any()),
      Event::Code(code) => {
        out.push(view! { <code class="md-code">{code.into_string()}</code> }.into_any())
      }
      // See [`render_markdown`]'s docs for why a soft break becomes a real line break
      // rather than CommonMark's default (a single space).
      Event::SoftBreak => out.push(view! { <br /> }.into_any()),
      Event::HardBreak => out.push(view! { <br /> }.into_any()),
      Event::Rule => out.push(view! { <hr class="md-hr" /> }.into_any()),
      Event::TaskListMarker(checked) => out.push(
        view! {
          <input type="checkbox" class="md-task" checked=checked disabled=true />
        }
        .into_any(),
      ),
      // Raw HTML embedded in the markdown — rendered as the literal source text (see
      // the module docs on why this never reaches `inner_html`), not interpreted.
      Event::Html(html) | Event::InlineHtml(html) => out.push(html.into_string().into_any()),
      // Footnote definitions/references are rare enough in model output, and add
      // enough own complexity (backreferences, a definition list elsewhere in the
      // document), that silently dropping them is a better trade than either
      // implementing them or letting an `_ => {}` catch-all hide *other*, future event
      // variants this should have handled.
      Event::FootnoteReference(_) => {}
      Event::InlineMath(_) | Event::DisplayMath(_) => {}
    }
  }
  out
}

/// Collects every event between a just-seen [`Event::Start`] and its matching
/// [`Event::End`] (exclusive of both), leaving `iter` positioned just past that `End`.
///
/// Matching is done by depth, not by tag identity: every [`Event::Start`] increments a
/// counter and every [`Event::End`] decrements it, and the matching `End` is whichever
/// one brings the counter back to zero. This works regardless of *what* tag is nested
/// inside itself (a list inside a list item inside a list, say) because
/// `pulldown_cmark`'s event stream is guaranteed well-formed — `Start`/`End` always
/// nest like parentheses — so depth alone is sufficient; comparing tag *kinds* would
/// only add complexity for no added correctness.
fn take_matching_end<'a>(iter: &mut impl Iterator<Item = Event<'a>>) -> Vec<Event<'a>> {
  let mut depth = 1u32;
  let mut out = Vec::new();
  for event in iter {
    match &event {
      Event::Start(_) => depth += 1,
      Event::End(_) => {
        depth -= 1;
        if depth == 0 {
          break;
        }
      }
      _ => {}
    }
    out.push(event);
  }
  out
}

/// Renders one container tag given its already-isolated child events (see
/// [`take_matching_end`]). [`Tag::Table`] is handled by a dedicated [`render_table`]
/// rather than falling through to a generic "wrap `children` in an element" case: a
/// table's immediate children ([`Tag::TableHead`]/[`Tag::TableRow`]) need to be told
/// whether they are the header row (to render `<th>` instead of `<td>`) and need access
/// to `alignments`, neither of which the uniform recursion below has anywhere to pass
/// through.
fn render_tag(tag: Tag<'_>, children_events: Vec<Event<'_>>) -> AnyView {
  match tag {
    Tag::Paragraph => {
      let children = render_events(&mut children_events.into_iter());
      view! { <p class="md-p">{children}</p> }.into_any()
    }
    Tag::Heading { level, .. } => {
      let children = render_events(&mut children_events.into_iter());
      render_heading(level, children)
    }
    Tag::BlockQuote(_) => {
      let children = render_events(&mut children_events.into_iter());
      view! { <blockquote class="md-blockquote">{children}</blockquote> }.into_any()
    }
    Tag::CodeBlock(kind) => render_code_block(kind, children_events),
    Tag::List(Some(start)) => {
      let children = render_events(&mut children_events.into_iter());
      view! { <ol class="md-list" start=start as i32>{children}</ol> }.into_any()
    }
    Tag::List(None) => {
      let children = render_events(&mut children_events.into_iter());
      view! { <ul class="md-list">{children}</ul> }.into_any()
    }
    Tag::Item => {
      let children = render_events(&mut children_events.into_iter());
      view! { <li class="md-li">{children}</li> }.into_any()
    }
    Tag::Emphasis => {
      let children = render_events(&mut children_events.into_iter());
      view! { <em class="md-em">{children}</em> }.into_any()
    }
    Tag::Strong => {
      let children = render_events(&mut children_events.into_iter());
      view! { <strong class="md-strong">{children}</strong> }.into_any()
    }
    Tag::Strikethrough => {
      let children = render_events(&mut children_events.into_iter());
      view! { <del class="md-del">{children}</del> }.into_any()
    }
    Tag::Link {
      dest_url, title, ..
    } => {
      let children = render_events(&mut children_events.into_iter());
      let title = title.into_string();
      view! {
        <a
          class="md-link"
          href=dest_url.into_string()
          title=(!title.is_empty()).then_some(title)
          target="_blank"
          rel="noopener noreferrer"
        >
          {children}
        </a>
      }
      .into_any()
    }
    Tag::Image {
      dest_url, title, ..
    } => {
      // An image's own children are the alt-text events (plain runs of `Text`, per
      // CommonMark) rather than something to render as nested elements — flattened
      // back into a single string for the `alt` attribute instead of being recursed
      // into via `render_events`, which would try to build child *views* for something
      // that is not a rendering context at all.
      let alt = plain_text(&children_events);
      let title = title.into_string();
      view! {
        <img
          class="md-img"
          src=dest_url.into_string()
          alt=alt
          title=(!title.is_empty()).then_some(title)
          loading="lazy"
        />
      }
      .into_any()
    }
    Tag::Table(alignments) => render_table(children_events, &alignments),
    // Reached only if a `TableHead`/`TableRow`/`TableCell` shows up outside of a
    // `Table` — malformed input, not something `pulldown_cmark` itself produces from
    // valid markdown. Rendered as a plain, unstyled fragment of its children instead
    // of being dropped, so a bug here loses formatting rather than losing content.
    Tag::TableHead | Tag::TableRow | Tag::TableCell => {
      let children = render_events(&mut children_events.into_iter());
      view! { <>{children}</> }.into_any()
    }
    // Everything else falls through to a plain, unstyled fragment of its children: some
    // of these (`DefinitionList*`, `Superscript`/`Subscript`) are only ever produced
    // when the corresponding `Options::ENABLE_*` flag is on, which [`render_markdown`]
    // does not set, so they cannot actually occur here today; the rest
    // (`FootnoteDefinition`, `HtmlBlock`, `MetadataBlock`) are handled this generically
    // because giving up specific styling for them is an acceptable trade against the
    // extra code a dedicated case each would take, for tags this unlikely to show up in
    // chat-style model output.
    Tag::FootnoteDefinition(_)
    | Tag::HtmlBlock
    | Tag::MetadataBlock(_)
    | Tag::DefinitionList
    | Tag::DefinitionListTitle
    | Tag::DefinitionListDefinition
    | Tag::Superscript
    | Tag::Subscript => {
      let children = render_events(&mut children_events.into_iter());
      view! { <>{children}</> }.into_any()
    }
  }
}

fn render_heading(level: HeadingLevel, children: Vec<AnyView>) -> AnyView {
  match level {
    HeadingLevel::H1 => view! { <h1 class="md-h1">{children}</h1> }.into_any(),
    HeadingLevel::H2 => view! { <h2 class="md-h2">{children}</h2> }.into_any(),
    HeadingLevel::H3 => view! { <h3 class="md-h3">{children}</h3> }.into_any(),
    HeadingLevel::H4 => view! { <h4 class="md-h4">{children}</h4> }.into_any(),
    // H5/H6 share one style: by the time model output nests headings this deep, the
    // visual distinction from H4 stops mattering and is not worth a whole extra pair of
    // CSS rules to preserve.
    HeadingLevel::H5 | HeadingLevel::H6 => view! { <h5 class="md-h4">{children}</h5> }.into_any(),
  }
}

/// Flattens a run of events down to their text content — used for an image's `alt`
/// (see [`render_tag`]'s [`Tag::Image`] arm), which CommonMark defines as plain text
/// even though it is technically parsed as a nested inline sequence.
fn plain_text(events: &[Event<'_>]) -> String {
  let mut out = String::new();
  for event in events {
    match event {
      Event::Text(text) | Event::Code(text) => out.push_str(text),
      Event::SoftBreak | Event::HardBreak => out.push(' '),
      _ => {}
    }
  }
  out
}

/// Renders a fenced/indented code block with a language-tagged header and a copy
/// button. `children_events` is expected to be a run of [`Event::Text`] (what
/// `pulldown_cmark` emits for a code block's literal content) — flattened via
/// [`plain_text`] rather than recursed through [`render_events`], since code block
/// content is never further parsed as markdown (a `*` inside a fenced block is a
/// literal asterisk, not the start of emphasis).
fn render_code_block(kind: CodeBlockKind<'_>, children_events: Vec<Event<'_>>) -> AnyView {
  let language = match kind {
    CodeBlockKind::Fenced(info) => {
      // The info string is `<language> <anything else, ignored>` per CommonMark (e.g.
      // "rust,no_run" in some dialects) — only the first token is a language name
      // worth showing in the header.
      info.split_whitespace().next().unwrap_or("").to_owned()
    }
    CodeBlockKind::Indented => String::new(),
  };
  let code = plain_text(&children_events);
  let copied = RwSignal::new(false);
  let code_for_copy = code.clone();
  let lang = i18n::current_lang();

  let on_copy = move |_| {
    let code = code_for_copy.clone();
    spawn_local(async move {
      if copy_to_clipboard(&code).await {
        copied.set(true);
      }
    });
  };
  // Reverts the "Copied" label back once the pointer leaves rather than on a timer: no
  // extra timer/`Closure` bookkeeping needed, and a copy button a user is not even
  // looking at anymore has no reason to still be announcing a stale confirmation.
  let on_leave = move |_| copied.set(false);

  view! {
    <div class="md-codeblock">
      <div class="md-codeblock-head">
        <span class="md-codeblock-lang">{(!language.is_empty()).then_some(language)}</span>
        <button
          type="button"
          class="md-copy-btn"
          on:click=on_copy
          on:mouseleave=on_leave
        >
          {move || t(lang.get(), if copied.get() { Key::CopiedLabel } else { Key::CopyLabel })}
        </button>
      </div>
      <pre class="md-codeblock-body"><code>{code}</code></pre>
    </div>
  }
  .into_any()
}

/// Writes `text` to the system clipboard via the [`web_sys::Clipboard`] API, returning
/// whether it succeeded. Failure (permission denied, an insecure context lacking the
/// API at all, ...) is swallowed here rather than surfaced anywhere — a copy button that
/// silently does nothing on failure is an acceptable degradation for what is a
/// convenience feature, not a critical path.
async fn copy_to_clipboard(text: &str) -> bool {
  let Some(window) = web_sys::window() else {
    return false;
  };
  let clipboard = window.navigator().clipboard();
  wasm_bindgen_futures::JsFuture::from(clipboard.write_text(text))
    .await
    .is_ok()
}

fn alignment_class(alignment: Alignment) -> &'static str {
  match alignment {
    Alignment::None => "",
    Alignment::Left => "md-align-left",
    Alignment::Center => "md-align-center",
    Alignment::Right => "md-align-right",
  }
}

/// Renders a [`Tag::Table`]'s contents. `children_events` is a flat sibling sequence of
/// exactly one optional [`Tag::TableHead`] followed by zero or more [`Tag::TableRow`]
/// (per CommonMark's table extension grammar) — walked here directly, rather than via
/// the generic [`render_events`]/[`render_tag`] recursion, because turning a header row
/// into `<th>` cells instead of `<td>` needs to be decided per-row, information the
/// generic per-tag dispatch has no way to carry down into a [`Tag::TableCell`].
fn render_table(children_events: Vec<Event<'_>>, alignments: &[Alignment]) -> AnyView {
  let mut iter = children_events.into_iter();
  let mut head: Option<AnyView> = None;
  let mut rows: Vec<AnyView> = Vec::new();

  while let Some(event) = iter.next() {
    match event {
      Event::Start(Tag::TableHead) => {
        let row_events = take_matching_end(&mut iter);
        head = Some(render_table_row(row_events, alignments, true));
      }
      Event::Start(Tag::TableRow) => {
        let row_events = take_matching_end(&mut iter);
        rows.push(render_table_row(row_events, alignments, false));
      }
      _ => {}
    }
  }

  view! {
    <div class="md-table-wrap">
      <table class="md-table">
        {head.map(|head| view! { <thead>{head}</thead> })}
        <tbody>{rows}</tbody>
      </table>
    </div>
  }
  .into_any()
}

/// Renders one table row's cells, given the events between its `TableHead`/`TableRow`
/// start and end (see [`render_table`]). `is_head` picks `<th>` vs. `<td>` per cell —
/// CommonMark's table grammar has exactly one header row, always a `TableHead`, so this
/// is constant for every cell in a given row rather than something decided per cell.
fn render_table_row(
  row_events: Vec<Event<'_>>,
  alignments: &[Alignment],
  is_head: bool,
) -> AnyView {
  let mut iter = row_events.into_iter();
  let mut cells = Vec::new();
  let mut index = 0usize;

  while let Some(event) = iter.next() {
    if let Event::Start(Tag::TableCell) = event {
      let cell_events = take_matching_end(&mut iter);
      let content = render_events(&mut cell_events.into_iter());
      let align_class = alignment_class(alignments.get(index).copied().unwrap_or(Alignment::None));
      let class = format!("md-cell {align_class}");
      cells.push(if is_head {
        view! { <th class=class>{content}</th> }.into_any()
      } else {
        view! { <td class=class>{content}</td> }.into_any()
      });
      index += 1;
    }
  }

  view! { <tr>{cells}</tr> }.into_any()
}
