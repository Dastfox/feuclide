//! Gamepads, through gilrs: the runner reads every pad connected into [`RawInput`] before each
//! frame, all of them driving the same buttons and sticks.

use fk_math::Real;
use fk_math::nalgebra::Vector2;
use gilrs::{Axis, Button, EventType, Gamepad, Gilrs};

use crate::input::{PadButton, RawInput, Stick};

/// How far down a trigger goes before it is pressed, a share of its travel...
const TRIGGER_DOWN: Real = 0.4;
/// ...and how far it comes back up before it is released, so that a trigger held near the
/// threshold does not flutter.
const TRIGGER_UP: Real = 0.3;

/// The gamepads, while the platform has them.
pub(crate) struct Pads {
    gilrs: Option<Gilrs>,
}

impl Pads {
    /// Opens the platform's gamepads, logging those already connected; none, and a warning, if
    /// it cannot.
    pub(crate) fn new() -> Self {
        let gilrs = match Gilrs::new() {
            Ok(gilrs) => Some(gilrs),
            Err(gilrs::Error::NotImplemented(_)) => {
                tracing::info!("no gamepads on this platform");
                None
            }
            Err(error) => {
                tracing::warn!("gamepads unavailable: {error}");
                None
            }
        };
        if let Some(gilrs) = &gilrs {
            for (_, pad) in gilrs.gamepads() {
                tracing::info!("gamepad: {}", pad.name());
            }
        }
        Self { gilrs }
    }

    /// Reads the pads into `input`. Unfocused, their events are drained and nothing is read (the
    /// runner released everything as the focus went).
    pub(crate) fn poll(&mut self, input: &mut RawInput, focused: bool) {
        let Some(gilrs) = &mut self.gilrs else { return };
        while let Some(event) = gilrs.next_event() {
            match event.event {
                EventType::Connected => {
                    tracing::info!("gamepad connected: {}", gilrs.gamepad(event.id).name());
                }
                EventType::Disconnected => {
                    tracing::info!("gamepad disconnected: {}", gilrs.gamepad(event.id).name());
                }
                _ => {}
            }
        }
        if !focused {
            return;
        }
        let mut held = [false; PadButton::ALL.len()];
        let mut sticks = [Vector2::<Real>::zeros(); 2];
        for (_, pad) in gilrs.gamepads() {
            for (down, &button) in held.iter_mut().zip(&PadButton::ALL) {
                *down |= match button {
                    PadButton::LeftTrigger | PadButton::RightTrigger => {
                        let threshold = if input.pad_buttons.held(button) {
                            TRIGGER_UP
                        } else {
                            TRIGGER_DOWN
                        };
                        trigger(&pad, gilrs_button(button)) > threshold
                    }
                    _ => pad.is_pressed(gilrs_button(button)),
                };
            }
            // The pad pushed furthest has each stick.
            for (stick, (x, y)) in sticks.iter_mut().zip([
                (Axis::LeftStickX, Axis::LeftStickY),
                (Axis::RightStickX, Axis::RightStickY),
            ]) {
                let read = Vector2::new(Real::from(pad.value(x)), Real::from(pad.value(y)));
                if read.norm() > stick.norm() {
                    *stick = read;
                }
            }
        }
        for (&down, &button) in held.iter().zip(&PadButton::ALL) {
            if down {
                input.press(button);
            } else {
                input.release(button);
            }
        }
        input.set_stick(Stick::Left, sticks[0]);
        input.set_stick(Stick::Right, sticks[1]);
    }
}

/// How far down a trigger is, 0 to 1 (a trigger that is only a switch is all or nothing).
fn trigger(pad: &Gamepad<'_>, button: Button) -> Real {
    Real::from(pad.button_data(button).map_or(0.0, |data| data.value()))
}

/// gilrs's name for a pad button. Its triggers are the second pair (`LeftTrigger2`), its
/// bumpers the first.
fn gilrs_button(button: PadButton) -> Button {
    match button {
        PadButton::South => Button::South,
        PadButton::East => Button::East,
        PadButton::West => Button::West,
        PadButton::North => Button::North,
        PadButton::LeftBumper => Button::LeftTrigger,
        PadButton::RightBumper => Button::RightTrigger,
        PadButton::LeftTrigger => Button::LeftTrigger2,
        PadButton::RightTrigger => Button::RightTrigger2,
        PadButton::Select => Button::Select,
        PadButton::Start => Button::Start,
        PadButton::LeftStick => Button::LeftThumb,
        PadButton::RightStick => Button::RightThumb,
        PadButton::DPadUp => Button::DPadUp,
        PadButton::DPadDown => Button::DPadDown,
        PadButton::DPadLeft => Button::DPadLeft,
        PadButton::DPadRight => Button::DPadRight,
    }
}
