//! A button action a stick pressed in latches, the way many pads run: pressed once, it holds
//! until another action (the walk) comes back to rest, or it is pressed again.

use crate::input::{ButtonInput, PadButton, RawInput};

use super::{Action, Binding, KeyMap};

/// An action held while a key or a button bound to it is held, or latched by a stick bound to
/// it pressed in until the action it goes with is released (the walk's stick back at rest).
/// The thumb never holds a stick in while it pushes it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StickLatch {
    latched: bool,
    held: bool,
}

/// Whether `button` is a stick pressed in.
fn is_click(button: ButtonInput) -> bool {
    matches!(
        button,
        ButtonInput::Pad(PadButton::LeftStick | PadButton::RightStick)
    )
}

impl StickLatch {
    /// Whether the action is held, or latched.
    pub fn held(&self) -> bool {
        self.held
    }

    /// Steps the latch: `clicked` a stick bound to the action was pressed in this frame, `held`
    /// a key or another button bound to it is held, `released` the action it goes with came back
    /// to rest this frame.
    pub fn step(&mut self, clicked: bool, held: bool, released: bool) {
        if clicked {
            self.latched = !self.latched;
        } else if released {
            self.latched = false;
        }
        self.held = self.latched || held;
    }

    /// Steps the latch of `action` from `map` and `input`, until `until` is released: read
    /// it each frame after the key maps, in place of `map.held(action)`.
    pub fn read<A: Action>(&mut self, map: &KeyMap<A>, input: &RawInput, action: A, until: A) {
        let buttons = || {
            map.bindings()
                .get(action)
                .iter()
                .filter_map(|binding| match binding {
                    Binding::Button(button) => Some(*button),
                    _ => None,
                })
        };
        let clicked = buttons().any(|button| is_click(button) && input.just_pressed(button));
        // A stick pressed in latches; the key map's own hold of it is no hold.
        let held =
            map.held(action) && buttons().any(|button| !is_click(button) && input.held(button));
        self.step(clicked, held, map.released(until));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_click_holds_until_the_other_action_rests() {
        let mut latch = StickLatch::default();
        latch.step(true, false, false);
        assert!(latch.held());
        latch.step(false, false, false);
        assert!(latch.held(), "latched");
        latch.step(false, false, true);
        assert!(!latch.held(), "the stick back at rest");
        latch.step(true, false, false);
        latch.step(true, false, false);
        assert!(!latch.held(), "pressed again, let go");
        latch.step(false, true, false);
        assert!(latch.held(), "a key holds while held");
        latch.step(false, false, false);
        assert!(!latch.held());
    }
}
