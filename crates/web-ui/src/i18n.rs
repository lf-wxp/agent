//! UI text in Chinese, English, and Spanish, plus the machinery to pick one on load and
//! switch between them at runtime.
//!
//! Every user-visible string is a [`Key`] variant, translated by [`t`] — a closed enum
//! rather than a stringly-typed lookup (`&str` key into a `HashMap`, say) specifically so
//! a missing translation is a **compile error** (a non-exhaustive `match` in [`t`]) the
//! moment a new [`Key`] is added, instead of a silent runtime fallback to English/a blank
//! string discovered only by clicking around in Spanish. A few strings need an
//! interpolated value (an error message, say) and cannot be a `Key -> &'static str`
//! lookup at all — those are their own small functions below `t` instead
//! ([`request_failed`], [`approval_submit_failed`], [`server_error_status`]).
//!
//! [`Lang`] is held in one [`leptos::prelude::RwSignal`] created once in `main.rs`'s
//! `App` and shared via [`leptos::prelude::provide_context`]/`expect_context` rather than
//! threaded as a parameter through every rendering function — see `main.rs`'s call sites
//! for how each one picks it up. Every place a translated string appears in the page
//! must read it from *inside* a `move || ...` closure (`{move || t(lang.get(), Key::X)}`,
//! not `let text = t(lang.get(), Key::X);` used as a plain value) so that switching
//! languages after some content has already rendered — most of the timeline could be
//! from ten turns ago by the time a user opens the language picker — still updates that
//! already-mounted chrome (badges, buttons, the composer's placeholder, ...) instead of
//! only affecting whatever renders *after* the switch.
//!
//! The one deliberate exception is free-form text that was never part of this page's own
//! vocabulary to begin with: a [`shared::ChatEvent::Error`]'s message, or the
//! `BudgetExhausted` notice pushed once into the timeline history. Those are formatted
//! once, in whichever language was active at the moment they occurred, and — like the
//! assistant's own message text, or the user's own input — never retranslated after the
//! fact. Retranslating an assistant reply after it already streamed in makes no sense at
//! all (it is not this page's text to translate); treating a same-process, one-off status
//! notice the same way keeps that rule simple rather than carving out a special case for
//! "well, this one's ours, so translate it retroactively but nothing else."

use leptos::prelude::*;
use shared::commands::Command;

/// Supported UI languages. `Copy` because a [`RwSignal<Lang>`] is read constantly from
/// inside reactive closures scattered across every rendering function (see the module
/// docs) — cloning being free removes any reason to write those as borrows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lang {
  Zh,
  En,
  Es,
}

/// Every supported language, in the order the header's picker lists them.
pub const ALL_LANGS: [Lang; 3] = [Lang::Zh, Lang::En, Lang::Es];

const STORAGE_KEY: &str = "agent-web-lang";

impl Lang {
  /// BCP 47 code, used both for `<html lang>` (see `main.rs`'s `App`) and as the value
  /// [`Self::store`]/[`stored_lang`] round-trip through `localStorage`.
  pub fn code(self) -> &'static str {
    match self {
      Lang::Zh => "zh-CN",
      Lang::En => "en",
      Lang::Es => "es",
    }
  }

  /// A short label for the picker button itself — kept to two or three characters (not
  /// each language's full name) so three of these plus the connection indicator still
  /// fit in the header on a narrow phone screen; [`Self::native_name`] carries the full
  /// name instead, as that button's tooltip.
  pub fn short_label(self) -> &'static str {
    match self {
      Lang::Zh => "中",
      Lang::En => "EN",
      Lang::Es => "ES",
    }
  }

  /// This language's own name, written *in* that language — every picker button shows
  /// this as its `title`, deliberately not translated into whichever language is
  /// currently active: a Spanish speaker who landed on the Chinese UI by accident needs
  /// to recognize "Español" without first being able to read Chinese.
  pub fn native_name(self) -> &'static str {
    match self {
      Lang::Zh => "中文",
      Lang::En => "English",
      Lang::Es => "Español",
    }
  }

  /// Matches on the primary subtag only (`"zh"`, `"en"`, `"es"`, case-insensitively) so
  /// a browser reporting `"zh-Hans-CN"` or a stored `"en"` both resolve the same way a
  /// full BCP 47 parser would for the purposes this needs — a full parse is not worth
  /// pulling in a dependency for over three prefixes.
  fn from_code_prefix(code: &str) -> Option<Lang> {
    let lower = code.to_ascii_lowercase();
    if lower.starts_with("zh") {
      Some(Lang::Zh)
    } else if lower.starts_with("es") {
      Some(Lang::Es)
    } else if lower.starts_with("en") {
      Some(Lang::En)
    } else {
      None
    }
  }

  /// Picks the language this page should open in: an explicit earlier choice saved to
  /// `localStorage` (see [`Self::store`]) wins over the browser's own
  /// `navigator.language`, which in turn wins over [`Lang::Zh`] as the last-resort
  /// default — matching this page's language before this module existed, for a visitor
  /// whose browser reports something this cannot recognize at all (`"ja"`, `"fr"`, ...).
  pub fn detect() -> Lang {
    stored_lang().or_else(browser_lang).unwrap_or(Lang::Zh)
  }

  /// Saves this choice to `localStorage` so a later visit (see [`Self::detect`]) opens
  /// directly in it instead of re-deriving it from the browser's language again.
  /// Failure (a browser with `localStorage` disabled, a `SecurityError` in some private-
  /// browsing modes, ...) is swallowed: the picker still switches the *current* page's
  /// language regardless, it just will not be remembered next visit — degrading to "not
  /// persisted" is preferable to the switch itself silently failing.
  pub fn store(self) {
    if let Some(storage) = local_storage() {
      let _ = storage.set_item(STORAGE_KEY, self.code());
    }
  }
}

fn local_storage() -> Option<web_sys::Storage> {
  web_sys::window()?.local_storage().ok()?
}

fn stored_lang() -> Option<Lang> {
  let value = local_storage()?.get_item(STORAGE_KEY).ok()??;
  Lang::from_code_prefix(&value)
}

fn browser_lang() -> Option<Lang> {
  let language = web_sys::window()?.navigator().language()?;
  Lang::from_code_prefix(&language)
}

/// Every fixed (non-interpolated) user-visible string in this crate — see the module
/// docs for why this is a closed enum rather than a stringly-typed lookup, and for the
/// handful of strings that need an interpolated value and so live outside it instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Key {
  BrandTag,
  ConnConnecting,
  ConnOpen,
  ConnRetrying,
  EmptyTitle,
  EmptyHint,
  ComposerPlaceholder,
  ComposerHint,
  SendAria,
  RoleYou,
  RoleAgent,
  ToolCallBadge,
  ToolDoneBadge,
  ToolFailedBadge,
  ToolExpand,
  ToolCollapse,
  ApprovalPrompt,
  ApprovalApprove,
  ApprovalDeny,
  ApprovalAlways,
  ApprovalNever,
  ApprovalStickyHint,
  ApprovalReasonPlaceholder,
  ApprovalApproved,
  ApprovalDenied,
  ThinkingAria,
  CopyLabel,
  CopiedLabel,
  BudgetExhausted,
  CommandMenuAria,
  CommandMenuHint,
}

pub fn t(lang: Lang, key: Key) -> &'static str {
  use Key::*;
  use Lang::*;
  match (key, lang) {
    (BrandTag, Zh) => "本地会话 · 与终端实时同步",
    (BrandTag, En) => "Local session · live-synced with the terminal",
    (BrandTag, Es) => "Sesión local · sincronizada en vivo con la terminal",

    (ConnConnecting, Zh) => "连接中",
    (ConnConnecting, En) => "Connecting",
    (ConnConnecting, Es) => "Conectando",

    (ConnOpen, Zh) => "实时同步",
    (ConnOpen, En) => "Live",
    (ConnOpen, Es) => "En vivo",

    (ConnRetrying, Zh) => "重新连接中",
    (ConnRetrying, En) => "Reconnecting",
    (ConnRetrying, Es) => "Reconectando",

    (EmptyTitle, Zh) => "暂无消息",
    (EmptyTitle, En) => "No messages yet",
    (EmptyTitle, Es) => "Sin mensajes todavía",

    (EmptyHint, Zh) => "在下方输入，或在终端里继续这段对话",
    (EmptyHint, En) => "Type below, or continue this conversation in the terminal",
    (EmptyHint, Es) => "Escribe abajo, o continúa esta conversación en la terminal",

    (ComposerPlaceholder, Zh) => "输入消息…",
    (ComposerPlaceholder, En) => "Type a message…",
    (ComposerPlaceholder, Es) => "Escribe un mensaje…",

    (ComposerHint, Zh) => "Enter 发送 · Shift+Enter 换行 · / 查看命令",
    (ComposerHint, En) => "Enter to send · Shift+Enter for a new line · / for commands",
    (ComposerHint, Es) => "Enter para enviar · Shift+Enter para salto de línea · / para comandos",

    (SendAria, Zh) => "发送",
    (SendAria, En) => "Send",
    (SendAria, Es) => "Enviar",

    (RoleYou, Zh) => "你",
    (RoleYou, En) => "You",
    (RoleYou, Es) => "Tú",

    // The product's own name, kept identical in every language rather than translated
    // or transliterated — the same reasoning `Lang::native_name` docs above apply to a
    // brand name specifically.
    (RoleAgent, Zh | En | Es) => "agent",

    (ToolCallBadge, Zh) => "调用",
    (ToolCallBadge, En) => "Call",
    (ToolCallBadge, Es) => "Llamada",

    (ToolDoneBadge, Zh) => "完成",
    (ToolDoneBadge, En) => "Done",
    (ToolDoneBadge, Es) => "Listo",

    (ToolFailedBadge, Zh) => "失败",
    (ToolFailedBadge, En) => "Failed",
    (ToolFailedBadge, Es) => "Error",

    (ToolExpand, Zh) => "展开",
    (ToolExpand, En) => "Expand",
    (ToolExpand, Es) => "Expandir",

    (ToolCollapse, Zh) => "收起",
    (ToolCollapse, En) => "Collapse",
    (ToolCollapse, Es) => "Contraer",

    (ApprovalPrompt, Zh) => "即将执行高危操作",
    (ApprovalPrompt, En) => "About to run a dangerous operation",
    (ApprovalPrompt, Es) => "Se va a ejecutar una operación peligrosa",

    (ApprovalApprove, Zh) => "批准",
    (ApprovalApprove, En) => "Approve",
    (ApprovalApprove, Es) => "Aprobar",

    (ApprovalDeny, Zh) => "拒绝",
    (ApprovalDeny, En) => "Deny",
    (ApprovalDeny, Es) => "Rechazar",

    (ApprovalAlways, Zh) => "总是允许",
    (ApprovalAlways, En) => "Always allow",
    (ApprovalAlways, Es) => "Permitir siempre",

    (ApprovalNever, Zh) => "总是拒绝",
    (ApprovalNever, En) => "Always deny",
    (ApprovalNever, Es) => "Rechazar siempre",

    (ApprovalStickyHint, Zh) => "「总是」适用于本会话内该工具的后续调用，/reset 后失效",
    (ApprovalStickyHint, En) => {
      "\"Always\" applies to later calls of this tool in this session, until /reset"
    }
    (ApprovalStickyHint, Es) => {
      "\"Siempre\" se aplica a las siguientes llamadas de esta herramienta en esta sesión, hasta /reset"
    }

    (ApprovalReasonPlaceholder, Zh) => "拒绝理由（可选，会告知模型）",
    (ApprovalReasonPlaceholder, En) => "Reason for denying (optional, shown to the model)",
    (ApprovalReasonPlaceholder, Es) => "Motivo del rechazo (opcional, se muestra al modelo)",

    (ApprovalApproved, Zh) => "✓ 已批准",
    (ApprovalApproved, En) => "✓ Approved",
    (ApprovalApproved, Es) => "✓ Aprobado",

    (ApprovalDenied, Zh) => "✕ 已拒绝",
    (ApprovalDenied, En) => "✕ Denied",
    (ApprovalDenied, Es) => "✕ Rechazado",

    (ThinkingAria, Zh) => "agent 正在处理",
    (ThinkingAria, En) => "agent is processing",
    (ThinkingAria, Es) => "agent está procesando",

    (CopyLabel, Zh) => "复制",
    (CopyLabel, En) => "Copy",
    (CopyLabel, Es) => "Copiar",

    (CopiedLabel, Zh) => "已复制",
    (CopiedLabel, En) => "Copied",
    (CopiedLabel, Es) => "Copiado",

    (BudgetExhausted, Zh) => "工具调用轮次预算已用尽，回答可能基于部分结果。",
    (BudgetExhausted, En) => {
      "Tool-call round budget exhausted; the answer may be based on partial results."
    }
    (BudgetExhausted, Es) => {
      "Se agotó el presupuesto de rondas de herramientas; la respuesta puede basarse en \
       resultados parciales."
    }

    (CommandMenuAria, Zh) => "可用命令",
    (CommandMenuAria, En) => "Available commands",
    (CommandMenuAria, Es) => "Comandos disponibles",

    (CommandMenuHint, Zh) => "↑↓ 选择 · Enter 确认 · Esc 关闭",
    (CommandMenuHint, En) => "↑↓ to choose · Enter to confirm · Esc to dismiss",
    (CommandMenuHint, Es) => "↑↓ para elegir · Enter para confirmar · Esc para cerrar",
  }
}

/// What one row of the `/`-triggered command menu says the command does.
///
/// Kept here rather than read off [`shared::commands::CommandSpec::summary`], which is
/// Chinese-only: that string is written for `//help`'s plain-text output and for the
/// terminal, neither of which is translated, while this menu is part of *this* page's
/// chrome and has to follow the language picker like every other [`Key`] does.
///
/// Matching on the [`Command`] enum rather than on the summary text keeps the guarantee
/// the rest of this module is built on (see the module docs): adding a command to
/// [`shared::commands::COMMANDS`] makes this `match` non-exhaustive, so a new command
/// cannot reach the menu without its description being translated first.
pub fn command_summary(lang: Lang, command: Command) -> &'static str {
  match (command, lang) {
    (Command::Help, Lang::Zh) => "显示可用命令",
    (Command::Help, Lang::En) => "Show the available commands",
    (Command::Help, Lang::Es) => "Mostrar los comandos disponibles",

    (Command::Reset, Lang::Zh) => "清空当前会话的历史记录",
    (Command::Reset, Lang::En) => "Clear this session's history",
    (Command::Reset, Lang::Es) => "Borrar el historial de esta sesión",

    // Never actually listed in this menu (`available_on_web` filters it out before it
    // gets here), but translated anyway rather than left to a fallback arm: a catch-all
    // is exactly what would let the *next* command through undescribed.
    (Command::Exit, Lang::Zh) => "退出（仅命令行）",
    (Command::Exit, Lang::En) => "Leave the chat (terminal only)",
    (Command::Exit, Lang::Es) => "Salir del chat (solo terminal)",
  }
}

/// `POST /api/chat`/`/api/approve`'s error path (`main.rs`'s `send`) — kept out of
/// [`Key`]/[`t`] because it interpolates `err`, which a `Key -> &'static str` lookup has
/// no room for.
pub fn request_failed(lang: Lang, err: &str) -> String {
  match lang {
    Lang::Zh => format!("请求失败：{err}"),
    Lang::En => format!("Request failed: {err}"),
    Lang::Es => format!("Error en la solicitud: {err}"),
  }
}

/// An approval decision's own submission failure (`main.rs`'s `render_approval`) — same
/// reason as [`request_failed`] for not being a [`Key`].
pub fn approval_submit_failed(lang: Lang, err: &str) -> String {
  match lang {
    Lang::Zh => format!("提交决策失败，请重试：{err}"),
    Lang::En => format!("Failed to submit decision, please retry: {err}"),
    Lang::Es => format!("No se pudo enviar la decisión, inténtalo de nuevo: {err}"),
  }
}

/// A non-2xx HTTP response's message (`main.rs`'s `error_status`) — same reason as
/// [`request_failed`] for not being a [`Key`].
pub fn server_error_status(lang: Lang, status: u16, status_text: &str) -> String {
  match lang {
    Lang::Zh => format!("服务端返回 {status} {status_text}"),
    Lang::En => format!("Server returned {status} {status_text}"),
    Lang::Es => format!("El servidor devolvió {status} {status_text}"),
  }
}

/// The [`RwSignal<Lang>`] every rendering function reads via
/// [`leptos::prelude::expect_context`] — see the module docs for why this is threaded
/// through context rather than as a parameter. A thin wrapper purely so call sites read
/// `i18n::current_lang()` rather than repeating the same `expect_context::<RwSignal<Lang>>()`
/// (and its exact panic message, if the context is ever missing — a programmer error,
/// since `main.rs`'s `App` always provides it before any of these render) at every one
/// of them.
pub fn current_lang() -> RwSignal<Lang> {
  expect_context::<RwSignal<Lang>>()
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn from_code_prefix_matches_case_insensitively_and_ignores_region_subtags() {
    assert_eq!(Lang::from_code_prefix("zh-CN"), Some(Lang::Zh));
    assert_eq!(Lang::from_code_prefix("ZH-hans-cn"), Some(Lang::Zh));
    assert_eq!(Lang::from_code_prefix("en-US"), Some(Lang::En));
    assert_eq!(Lang::from_code_prefix("es-MX"), Some(Lang::Es));
    assert_eq!(Lang::from_code_prefix("fr-FR"), None);
  }

  #[test]
  fn every_key_has_a_translation_in_every_language() {
    // `t` is a `match` with no wildcard arm, so this loop is mostly documentation of
    // intent by this point (a missing case is already a compile error) — kept anyway
    // as a guard against a future refactor that widens `t`'s match with a catch-all,
    // which would silently reintroduce exactly the gap this module's design avoids.
    let keys = [
      Key::BrandTag,
      Key::ConnConnecting,
      Key::ConnOpen,
      Key::ConnRetrying,
      Key::EmptyTitle,
      Key::EmptyHint,
      Key::ComposerPlaceholder,
      Key::ComposerHint,
      Key::SendAria,
      Key::RoleYou,
      Key::RoleAgent,
      Key::ToolCallBadge,
      Key::ToolDoneBadge,
      Key::ToolFailedBadge,
      Key::ToolExpand,
      Key::ToolCollapse,
      Key::ApprovalPrompt,
      Key::ApprovalApprove,
      Key::ApprovalDeny,
      Key::ApprovalAlways,
      Key::ApprovalNever,
      Key::ApprovalStickyHint,
      Key::ApprovalReasonPlaceholder,
      Key::ApprovalApproved,
      Key::ApprovalDenied,
      Key::ThinkingAria,
      Key::CopyLabel,
      Key::CopiedLabel,
      Key::BudgetExhausted,
      Key::CommandMenuAria,
      Key::CommandMenuHint,
    ];
    for key in keys {
      for lang in ALL_LANGS {
        assert!(!t(lang, key).is_empty());
      }
    }
  }

  /// Same guarantee as above, for the menu's own descriptions: every command the table
  /// defines must be describable in every language, including the ones this page filters
  /// out of the menu.
  #[test]
  fn every_command_has_a_summary_in_every_language() {
    for spec in shared::commands::COMMANDS {
      for lang in ALL_LANGS {
        assert!(
          !command_summary(lang, spec.command).is_empty(),
          "{:?} has no summary in {}",
          spec.command,
          lang.code()
        );
      }
    }
  }

  /// The menu is opened by typing `/`, so the hint pointing at it has to name that and
  /// not the older `/help` route to the same list.
  #[test]
  fn the_composer_hint_points_at_the_slash_menu() {
    for lang in ALL_LANGS {
      let hint = t(lang, Key::ComposerHint);
      assert!(hint.contains('/'), "{}", lang.code());
      assert!(!hint.contains("/help"), "{}", lang.code());
    }
  }
}
