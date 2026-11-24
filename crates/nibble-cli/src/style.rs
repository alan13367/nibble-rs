//! ANSI colours for the CLI's messages on stderr.
//!
//! Colour is used only when stderr is a terminal that understands ANSI
//! escapes, and never when `NO_COLOR` is set (<https://no-color.org>).
//! `CLICOLOR_FORCE` forces it on, e.g. when the output goes through a pager.

use std::env;
use std::io::IsTerminal;

/// Paints text for stderr, or passes it through unchanged.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Style {
    enabled: bool,
}

impl Style {
    /// The style for stderr in this environment.
    pub(crate) fn stderr() -> Self {
        let set = |name: &str| env::var_os(name).is_some_and(|v| !v.is_empty() && v != "0");
        let enabled = if set("NO_COLOR") {
            false
        } else if set("CLICOLOR_FORCE") {
            true
        } else {
            std::io::stderr().is_terminal()
                && env::var_os("TERM").is_none_or(|t| t != "dumb")
                && ansi_console()
        };
        Self { enabled }
    }

    /// A style that never adds escapes.
    #[cfg(test)]
    pub(crate) const fn plain() -> Self {
        Self { enabled: false }
    }

    /// Whether this style adds escapes.
    pub(crate) const fn enabled(self) -> bool {
        self.enabled
    }

    fn paint(self, codes: &str, text: &str) -> String {
        if self.enabled {
            format!("\x1b[{codes}m{text}\x1b[0m")
        } else {
            text.to_owned()
        }
    }

    pub(crate) fn bold(self, text: &str) -> String {
        self.paint("1", text)
    }

    pub(crate) fn dim(self, text: &str) -> String {
        self.paint("2", text)
    }

    /// The program name and other headings.
    pub(crate) fn title(self, text: &str) -> String {
        self.paint("1;32", text)
    }

    /// Addresses the user can open or copy.
    pub(crate) fn link(self, text: &str) -> String {
        self.paint("1;4;36", text)
    }

    /// HTTP methods and other keywords.
    pub(crate) fn keyword(self, text: &str) -> String {
        self.paint("35", text)
    }

    /// Something that works but deserves attention.
    pub(crate) fn warning(self, text: &str) -> String {
        self.paint("1;33", text)
    }
}

/// Whether the console interprets ANSI escapes. Every Unix terminal does.
/// On Windows the classic console shows them as text unless the program
/// enables them, so colour is used only in terminals known to handle them.
fn ansi_console() -> bool {
    if cfg!(windows) {
        ["WT_SESSION", "TERM", "TERM_PROGRAM", "ANSICON"]
            .iter()
            .any(|v| env::var_os(v).is_some())
            || env::var_os("ConEmuANSI").is_some_and(|v| v == "ON")
    } else {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_style_adds_nothing() {
        let s = Style::plain();
        assert_eq!(s.bold("a"), "a");
        assert_eq!(s.link("http://x"), "http://x");
        assert!(!s.enabled());
    }

    #[test]
    fn enabled_style_wraps_and_resets() {
        let s = Style { enabled: true };
        assert_eq!(s.title("nibble"), "\x1b[1;32mnibble\x1b[0m");
        assert_eq!(s.dim("x"), "\x1b[2mx\x1b[0m");
    }
}
