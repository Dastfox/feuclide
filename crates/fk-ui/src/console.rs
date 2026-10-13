//! A text console over the image, for debugging: a line typed and entered, earlier lines
//! recalled, commands completed, output scrolled, drawn in `fk-render`'s [`TextPanel`] (the
//! counterpart of `bevy_console`). The app parses what is entered and prints its answers.

use fk_app::{KeyCode, WindowSize};
use fk_math::nalgebra::Vector2;
use fk_render::{Color, TextPanel};

/// Lines of output kept to scroll back to.
const KEPT: usize = 500;

/// A console: open or not, the line being typed, what it printed, what was entered.
#[derive(Clone, Debug, Default)]
pub struct Console {
    /// Whether it shows and takes the keyboard.
    pub open: bool,
    /// The line being typed.
    pub line: String,
    /// What it printed, oldest first.
    pub output: Vec<String>,
    /// Lines entered, oldest first, and which one Up has recalled.
    past: Vec<String>,
    recalled: Option<usize>,
    /// Lines scrolled back from the last.
    pub scrolled: usize,
}

/// A line's words, lowercase and apart by single spaces, without a slash in front.
pub fn words(line: &str) -> String {
    let line = line.trim_start();
    let line = line.strip_prefix('/').unwrap_or(line);
    let words: Vec<&str> = line.split_whitespace().collect();
    words.join(" ").to_lowercase()
}

impl Console {
    /// Prints `text`, cut into lines the text panel can hold.
    pub fn print(&mut self, text: &str) {
        let width = TextPanel::MAX_COLUMNS - 1;
        for line in text.lines() {
            let chars: Vec<char> = line.chars().collect();
            if chars.is_empty() {
                self.output.push(String::new());
            }
            for piece in chars.chunks(width) {
                self.output.push(piece.iter().collect());
            }
        }
        let over = self.output.len().saturating_sub(KEPT);
        self.output.drain(..over);
        self.scrolled = 0;
    }

    /// Takes a frame's typing, open: the text (what the font cannot draw left out), then the
    /// editing keys in order (Enter, Backspace, Escape closes, Up and Down recall, Tab
    /// completes from `commands`, Page Up and Page Down scroll). Returns the lines entered.
    pub fn edit<S: AsRef<str>>(
        &mut self,
        typed: &str,
        keys: &[KeyCode],
        commands: impl Fn() -> Vec<S>,
    ) -> Vec<String> {
        self.line
            .extend(typed.chars().filter(|c| c.is_ascii_graphic() || *c == ' '));
        let mut entered = Vec::new();
        for &key in keys {
            match key {
                KeyCode::Enter | KeyCode::NumpadEnter => {
                    let line = std::mem::take(&mut self.line);
                    if !line.trim().is_empty() && self.past.last() != Some(&line) {
                        self.past.push(line.clone());
                    }
                    self.recalled = None;
                    entered.push(line);
                }
                KeyCode::Backspace => {
                    self.line.pop();
                }
                KeyCode::Escape => self.open = false,
                KeyCode::ArrowUp | KeyCode::ArrowDown => self.recall(key == KeyCode::ArrowUp),
                KeyCode::Tab => self.complete(commands()),
                KeyCode::PageUp => self.scrolled += 10,
                KeyCode::PageDown => self.scrolled = self.scrolled.saturating_sub(10),
                _ => {}
            }
        }
        entered
    }

    /// Up an earlier line into the line, or down a later one (past the last, an empty line).
    pub fn recall(&mut self, earlier: bool) {
        let at = match (self.recalled, earlier) {
            (None, true) => self.past.len().checked_sub(1),
            (None, false) => None,
            (Some(at), true) => Some(at.saturating_sub(1)),
            (Some(at), false) => Some(at + 1).filter(|&at| at < self.past.len()),
        };
        self.recalled = at;
        self.line = at.map_or_else(String::new, |at| self.past[at].clone());
    }

    /// Completes the command begun on the line, out of `commands` (their words, without the
    /// slash), as far as every one it may be agrees; with several left, prints them.
    pub fn complete<S: AsRef<str>>(&mut self, commands: impl IntoIterator<Item = S>) {
        let begun = words(&self.line);
        let commands: Vec<S> = commands.into_iter().collect();
        let fits: Vec<&str> = commands
            .iter()
            .map(AsRef::as_ref)
            .filter(|name| name.starts_with(&begun))
            .collect();
        let Some(first) = fits.first() else { return };
        let common = fits.iter().fold(first.len(), |common, name| {
            first
                .bytes()
                .zip(name.bytes())
                .take(common)
                .take_while(|(a, b)| a == b)
                .count()
        });
        let agreed = &first[..common];
        self.line = format!("/{agreed}");
        if fits.len() == 1 {
            self.line.push(' ');
        } else if words(agreed) == begun {
            let all: Vec<String> = fits.iter().map(|name| format!("/{name}")).collect();
            self.print(&all.join("  "));
        }
    }

    /// The console as a text panel over the top of a window `window` big: as much of its output
    /// as fits two thirds of the window, and the line being typed, its cursor shown if `cursor`.
    pub fn panel(&self, window: WindowSize, cursor: bool) -> TextPanel {
        // In whole pixels, 8 by 10 font pixels a character, as wide as the window lets a full
        // line.
        let columns = TextPanel::MAX_COLUMNS as u32;
        let scale = (window.width.saturating_sub(32) / (8 * columns + 8)).clamp(1, 3);
        let rows = ((window.height * 2 / 3) / (10 * scale)).clamp(4, TextPanel::MAX_LINES as u32);
        let shown_output = rows as usize - 1;
        let end = self.output.len().saturating_sub(
            self.scrolled
                .min(self.output.len().saturating_sub(shown_output)),
        );
        let start = end.saturating_sub(shown_output);
        let mut lines: Vec<String> = self.output[start..end].to_vec();
        let prompt = format!("> {}{}", self.line, if cursor { "_" } else { " " });
        // The line cut from the left as it grows past the panel, padded so the panel keeps its
        // width.
        let width = TextPanel::MAX_COLUMNS - 1;
        let skip = prompt.len().saturating_sub(width);
        lines.push(format!("{:<width$}", &prompt[skip..]));
        TextPanel {
            lines,
            origin: Vector2::new(8.0, 8.0),
            scale,
            color: Color::srgb(0.85, 0.85, 0.8),
            background: Color::BLACK,
            opacity: 0.8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const COMMANDS: [&str; 4] = ["fly", "void call", "void reach", "screenshot"];

    #[test]
    fn typing_edits_enters_recalls_and_completes() {
        let mut console = Console {
            open: true,
            ..Console::default()
        };
        let commands = || COMMANDS.to_vec();
        assert!(
            console
                .edit("/flyx", &[KeyCode::Backspace], commands)
                .is_empty()
        );
        assert_eq!(
            console.edit("", &[KeyCode::Enter], commands),
            vec!["/fly".to_owned()]
        );
        console.edit("/ink", &[KeyCode::Enter], commands);
        console.edit("", &[KeyCode::ArrowUp, KeyCode::ArrowUp], commands);
        assert_eq!(console.line, "/fly");
        console.line = "/vo".to_owned();
        console.complete(COMMANDS);
        assert_eq!(console.line, "/void ");
        console.complete(COMMANDS);
        assert!(
            console
                .output
                .last()
                .is_some_and(|l| l.contains("/void reach"))
        );
        console.edit("é²", &[KeyCode::Escape], commands);
        assert!(!console.open);
        console.print(&"x".repeat(200));
        assert!(
            console
                .output
                .iter()
                .all(|l| l.len() < TextPanel::MAX_COLUMNS)
        );
    }
}
