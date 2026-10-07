//! The rebind flow: listen for the next input for one action, detect conflicts, apply.

use std::fmt;

use fk_math::Real;
use fk_math::nalgebra::Vector2;

use super::{Action, ActionKind, Binding, Input, KeyMap};
use crate::input::{ButtonInput, RawInput};

/// Mouse motion, in pixels within one frame, that counts as "the player moved the mouse" when
/// listening for an axis.
const MOTION_THRESHOLD: Real = 8.0;

/// Which binding of an action a rebind replaces.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RebindTarget {
    /// Binding number `index` of the action, or a new binding when `index` is the number of
    /// bindings. A button action takes the next key or mouse button; a 2D axis the next mouse
    /// motion or wheel motion; a 1D axis the next wheel motion.
    Binding(usize),
    /// One button of the composite binding number `index` (a [`Binding::ButtonPair`] or
    /// [`Binding::ButtonQuad`]). Takes the next key or mouse button.
    Part {
        /// Which binding of the action.
        index: usize,
        /// Which of its buttons.
        part: Part,
    },
}

/// One button of a composite binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Part {
    /// [`Binding::ButtonPair::negative`].
    Negative,
    /// [`Binding::ButtonPair::positive`].
    Positive,
    /// [`Binding::ButtonQuad::up`].
    Up,
    /// [`Binding::ButtonQuad::down`].
    Down,
    /// [`Binding::ButtonQuad::left`].
    Left,
    /// [`Binding::ButtonQuad::right`].
    Right,
}

/// Where the rebind flow is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Rebind<A> {
    /// No rebind in progress.
    Idle,
    /// Waiting for the next input.
    Listening {
        /// The action being rebound.
        action: A,
        /// Which of its bindings.
        target: RebindTarget,
    },
    /// The captured input is already used by other actions; waiting for
    /// [`KeyMap::resolve`] or [`KeyMap::cancel_rebind`].
    Conflict {
        /// The action being rebound.
        action: A,
        /// Which of its bindings.
        target: RebindTarget,
        /// The binding that will be applied.
        binding: Binding,
        /// The inputs the rebind captured.
        inputs: Vec<Input>,
        /// The other actions that use them.
        conflicts: Vec<A>,
    },
}

/// What to do with a conflicting rebind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Resolution {
    /// Apply it; the input then drives both actions.
    Keep,
    /// Apply it and remove from the other actions every binding that uses the input.
    Steal,
}

/// Why [`KeyMap::listen`] refused a target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RebindError {
    /// The action has no binding number `index`.
    NoSuchBinding {
        /// The index asked for.
        index: usize,
        /// How many bindings the action has.
        len: usize,
    },
    /// The binding has no such part (for example `Up` on a pair, or any part on a button).
    NoSuchPart(Part),
}

impl fmt::Display for RebindError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoSuchBinding { index, len } => {
                write!(f, "no binding {index}, the action has {len}")
            }
            Self::NoSuchPart(part) => write!(f, "the binding has no {part:?} part"),
        }
    }
}

impl std::error::Error for RebindError {}

impl Binding {
    /// This composite binding with `part` replaced by `button`, or `None` if it has no such part.
    pub fn with_part(mut self, part: Part, button: ButtonInput) -> Option<Self> {
        let slot = match (&mut self, part) {
            (Self::ButtonPair { negative, .. }, Part::Negative) => negative,
            (Self::ButtonPair { positive, .. }, Part::Positive) => positive,
            (Self::ButtonQuad { up, .. }, Part::Up) => up,
            (Self::ButtonQuad { down, .. }, Part::Down) => down,
            (Self::ButtonQuad { left, .. }, Part::Left) => left,
            (Self::ButtonQuad { right, .. }, Part::Right) => right,
            _ => return None,
        };
        *slot = button;
        Some(self)
    }
}

impl<A: Action> KeyMap<A> {
    /// Where the rebind flow is.
    pub fn rebind_state(&self) -> &Rebind<A> {
        &self.rebind
    }

    /// Starts listening for the next input to bind to `target` of `action`, replacing any rebind
    /// in progress. Until the rebind ends, every action reads as released.
    pub fn listen(&mut self, action: A, target: RebindTarget) -> Result<(), RebindError> {
        self.check_target(action, target)?;
        self.rebind = Rebind::Listening { action, target };
        Ok(())
    }

    /// Abandons the rebind in progress, if any, leaving the bindings unchanged.
    pub fn cancel_rebind(&mut self) {
        self.rebind = Rebind::Idle;
    }

    /// Applies a rebind waiting in [`Rebind::Conflict`]. Does nothing in any other state.
    pub fn resolve(&mut self, resolution: Resolution) {
        let Rebind::Conflict {
            action,
            target,
            binding,
            inputs,
            ..
        } = std::mem::replace(&mut self.rebind, Rebind::Idle)
        else {
            return;
        };
        if resolution == Resolution::Steal {
            for &other in A::ALL.iter().filter(|&&other| other != action) {
                self.bindings
                    .get_mut(other)
                    .retain(|b| !inputs.iter().any(|&input| b.uses(input)));
            }
        }
        self.apply(action, target, binding);
    }

    fn check_target(&self, action: A, target: RebindTarget) -> Result<(), RebindError> {
        let bindings = self.bindings.get(action);
        let len = bindings.len();
        match target {
            RebindTarget::Binding(index) if index > len => {
                Err(RebindError::NoSuchBinding { index, len })
            }
            RebindTarget::Binding(_) => Ok(()),
            RebindTarget::Part { index, part } => {
                let binding = bindings
                    .get(index)
                    .ok_or(RebindError::NoSuchBinding { index, len })?;
                // Any button stands in for the one the player will press.
                let probe = ButtonInput::Key(crate::KeyCode::Space);
                binding
                    .with_part(part, probe)
                    .map(|_| ())
                    .ok_or(RebindError::NoSuchPart(part))
            }
        }
    }

    /// Looks for the input a listening rebind is waiting for. Called by [`KeyMap::update`] while
    /// a rebind is in progress.
    pub(super) fn capture(&mut self, input: &RawInput) {
        let pressed = input
            .iter_just_pressed()
            .find(|button| !self.suppressed.contains(button));
        // Everything held now stays ignored until it goes up, so the rebind does not leak a press
        // into the actions when it ends.
        self.suppressed.extend(input.iter_held());

        let Rebind::Listening { action, target } = self.rebind else {
            return;
        };
        if let Err(error) = self.check_target(action, target) {
            tracing::warn!("rebind of {} cancelled: {error}", action.name());
            self.rebind = Rebind::Idle;
            return;
        }
        let captured = match target {
            RebindTarget::Binding(_) => match action.kind() {
                ActionKind::Button => pressed.map(Binding::Button),
                ActionKind::Axis1 => (input.wheel().y != 0.0).then_some(Binding::Wheel),
                ActionKind::Axis2 if input.mouse_motion().norm() >= MOTION_THRESHOLD => {
                    Some(Binding::MouseMotion)
                }
                ActionKind::Axis2 => (input.wheel() != Vector2::zeros()).then_some(Binding::Wheel),
            }
            .map(|binding| (binding, binding.inputs())),
            RebindTarget::Part { index, part } => pressed.and_then(|button| {
                let binding = self.bindings.get(action)[index].with_part(part, button)?;
                Some((binding, vec![Input::Button(button)]))
            }),
        };
        let Some((binding, inputs)) = captured else {
            return;
        };

        let mut conflicts = Vec::new();
        for &input in &inputs {
            for other in self.bindings.actions_using(input, Some(action)) {
                if !conflicts.contains(&other) {
                    conflicts.push(other);
                }
            }
        }
        if conflicts.is_empty() {
            self.rebind = Rebind::Idle;
            self.apply(action, target, binding);
        } else {
            self.rebind = Rebind::Conflict {
                action,
                target,
                binding,
                inputs,
                conflicts,
            };
        }
    }

    fn apply(&mut self, action: A, target: RebindTarget, binding: Binding) {
        let bindings = self.bindings.get_mut(action);
        let index = match target {
            RebindTarget::Binding(index) | RebindTarget::Part { index, .. } => index,
        };
        if index < bindings.len() {
            bindings[index] = binding;
        } else {
            bindings.push(binding);
        }
        tracing::info!("{} rebound to {binding:?}", action.name());
        self.dirty = true;
    }
}
