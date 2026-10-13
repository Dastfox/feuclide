//! Menu pages as a model the app draws: a [`Sheet`] (a heading, tabs, lines, the one chosen),
//! what the pointer over it asks of the page ([`Target`], [`pointed`], [`PageInput`]), and a
//! page of keys rebound through the key map ([`KeysPage`]). How a sheet looks is the app's.

use fk_app::keymap::{Action, Binding, KeyMap, Part, Rebind, RebindTarget, Resolution};
use fk_math::Real;

/// A page as it is drawn.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Sheet {
    /// What it is called, over its lines.
    pub heading: String,
    /// Whether it is drawn as the title's menu is, larger and further apart (the game's menu's
    /// main page), rather than as a page of the settings.
    pub menu: bool,
    /// Its tabs, left to right, if it has any.
    pub tabs: Vec<&'static str>,
    /// The tab shown.
    pub tab: usize,
    /// Whether the tabs are chosen rather than a line.
    pub on_tabs: bool,
    /// Its lines, top to bottom.
    pub rows: Vec<Row>,
    /// The line chosen.
    pub at: usize,
    /// When the line chosen last changed, seconds of the clock it is drawn at: its marker grows
    /// in from then.
    pub moved_at: Real,
    /// When the tab shown last changed, on the same clock, and the tab it came from: the
    /// underline glides across and the lines come in from that side.
    pub tabbed_at: Real,
    /// See [`tabbed_at`](Self::tabbed_at).
    pub from_tab: usize,
}

/// A line of a [`Sheet`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Row {
    /// The group it begins, its name drawn over it (empty: a gap, unnamed).
    pub group: Option<&'static str>,
    /// What it is called.
    pub label: String,
    /// Its value, right of the label.
    pub value: Option<String>,
    /// Whether left and right change the value: arrows flank it, chosen.
    pub changes: bool,
    /// Said after it, softer (New game asked once: Enter again).
    pub note: Option<String>,
    /// Whether it waits for something (a key pressed): its value is drawn as a question.
    pub waiting: bool,
    /// Whether it cannot be chosen.
    pub closed: bool,
}

impl Row {
    /// A line called `label`, nothing more.
    pub fn plain(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            ..Self::default()
        }
    }
}

/// What the player asks of a page in a frame: lines to move down (negative up), steps to change
/// the value (negative less), whether they chose, and where the pointer took them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PageInput {
    /// Lines down, or up if negative.
    pub step: isize,
    /// Steps more, or less if negative.
    pub change: i32,
    /// Whether the line was chosen (Enter, or a click on it).
    pub confirm: bool,
    /// Where the pointer took the page, before the rest.
    pub point: Option<Point>,
}

impl PageInput {
    /// This, and `other` with it: steps and changes added, either's choice, `other`'s pointing
    /// if it points.
    pub fn and(self, other: Self) -> Self {
        Self {
            step: self.step + other.step,
            change: self.change + other.change,
            confirm: self.confirm || other.confirm,
            point: other.point.or(self.point),
        }
    }
}

/// Where the pointer takes a page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Point {
    /// Onto a line, by its place among the lines shown.
    Row(usize),
    /// Onto the tabs.
    Tabs,
    /// Into a tab, clicked, by its place.
    Tab(usize),
}

/// What the pointer is over on a [`Sheet`] as drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    /// A tab, by its place.
    Tab(usize),
    /// A line, by its place.
    Row(usize),
    /// The arrow either side of the chosen line's value: less (-1) or more (1).
    Change(usize, i32),
}

/// What the pointer asks of a page: over `target`, having `moved` this frame, `clicked` or not.
/// Moving onto a line chooses it, onto the tabs chooses them; a click on a line is Enter, on a
/// tab opens it, on an arrow changes the value.
pub fn pointed(target: Option<Target>, moved: bool, clicked: bool) -> PageInput {
    let mut input = PageInput::default();
    match target {
        Some(Target::Row(row)) => {
            if moved || clicked {
                input.point = Some(Point::Row(row));
            }
            input.confirm = clicked;
        }
        Some(Target::Tab(tab)) => {
            if clicked {
                input.point = Some(Point::Tab(tab));
            } else if moved {
                input.point = Some(Point::Tabs);
            }
        }
        Some(Target::Change(row, change)) if clicked => {
            input.point = Some(Point::Row(row));
            input.change = change;
        }
        Some(Target::Change(..)) | None => {}
    }
    input
}

/// One line of a keys page: what it is called, the action and which of its keys Enter
/// rebinds (none: shown, not rebound there).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KeyRow<A> {
    /// What it is called.
    pub label: &'static str,
    /// The action.
    pub action: A,
    /// Which of its keys Enter rebinds; none, shown only.
    pub target: Option<RebindTarget>,
}

/// The key of `binding` a rebind of `target` would replace, or all of it.
pub fn key_of(binding: &Binding, target: Option<RebindTarget>) -> String {
    match (binding, target) {
        (
            Binding::ButtonQuad {
                up,
                down,
                left,
                right,
            },
            Some(RebindTarget::Part { part, .. }),
        ) => match part {
            Part::Up => up.to_string(),
            Part::Down => down.to_string(),
            Part::Left => left.to_string(),
            _ => right.to_string(),
        },
        _ => binding.to_string(),
    }
}

/// A keys page: every action of its table ([`KeyRow`]s) and its keys; Enter on one listens
/// for its next key, the key map's own rebind flow (`fk_app::keymap::Rebind`), taking the key
/// from any other action that had it. Holds only the rebind under way; the line chosen is the
/// page's around it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KeysPage {
    /// Whether the key map was listening last frame.
    listened: bool,
}

impl KeysPage {
    /// How many lines it has for `rows`, Back included.
    pub fn lines<A>(rows: &[KeyRow<A>]) -> usize {
        rows.len() + 1
    }

    /// Settles a rebind under way, before anything else reads the frame: a key taken from
    /// another action goes to the one it was asked for, and `back` (Esc) keeps the key there
    /// was. Returns whether the frame belongs to the rebind (listening, or the key just
    /// pressed), and so is not the page's.
    pub fn rebinding<A: Action>(&mut self, keys: &mut KeyMap<A>, back: bool) -> bool {
        if let Rebind::Conflict { .. } = keys.rebind_state() {
            keys.resolve(Resolution::Steal);
        }
        let listening = matches!(keys.rebind_state(), Rebind::Listening { .. });
        if listening && back {
            keys.cancel_rebind();
        }
        // The frame the key is taken in is the rebind's too: an Enter bound is no choice.
        let held = listening || self.listened;
        self.listened = listening && !back;
        held
    }

    /// Listens for the next key of line `row`'s action in `rows`, if it is one rebound there.
    pub fn listen<A: Action>(&mut self, keys: &mut KeyMap<A>, rows: &[KeyRow<A>], row: usize) {
        if let Some(KeyRow {
            action,
            target: Some(target),
            ..
        }) = rows.get(row)
        {
            if let Err(error) = keys.listen(*action, *target) {
                tracing::warn!("cannot rebind {}: {error}", action.name());
            }
            self.listened = true;
        }
    }

    /// The tab's lines as drawn: each action and its keys, the one listening asking for its
    /// key, and Back.
    pub fn rows<A: Action>(keys: &KeyMap<A>, rows: &[KeyRow<A>]) -> Vec<Row> {
        let listening = match keys.rebind_state() {
            Rebind::Listening { action, target } => Some((*action, *target)),
            _ => None,
        };
        let mut lines: Vec<Row> = rows
            .iter()
            .map(|row| {
                let this = row.target.map(|target| (row.action, target));
                let waiting = listening.is_some() && listening == this;
                let bindings = keys.bindings().get(row.action);
                let value = if waiting {
                    "press a key (Esc: keep it)".to_owned()
                } else {
                    match row.target {
                        Some(RebindTarget::Part { .. }) => bindings
                            .first()
                            .map_or_else(|| "-".to_owned(), |b| key_of(b, row.target)),
                        _ => bindings
                            .iter()
                            .map(ToString::to_string)
                            .collect::<Vec<_>>()
                            .join(", "),
                    }
                };
                Row {
                    value: Some(value),
                    waiting,
                    ..Row::plain(row.label)
                }
            })
            .collect();
        lines.push(Row {
            group: Some(""),
            ..Row::plain("Back")
        });
        lines
    }
}
