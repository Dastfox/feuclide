//! The key mapper: actions a game declares, bound to physical inputs the player can change.
//!
//! A game lists its actions as a type implementing [`Action`]. Each action is a button (pressed,
//! held, released) or a 1D or 2D axis, and has a list of [`Binding`]s; it reads as the
//! combination of all of them. The game supplies the default [`Bindings`]; [`KeyMapPlugin`] keeps
//! the player's bindings in a TOML file, reloads it when it changes on disk and saves it when
//! bindings change in game. [`KeyMap::listen`] starts the rebind flow for one action.
//!
//! Bindings are on physical keys ([`KeyCode`](crate::KeyCode)), so WASD on QWERTY and ZQSD on
//! AZERTY are one and the same binding.

mod config;
mod rebind;

use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;
use std::path::{Path, PathBuf};

use bevy_ecs::resource::Resource;
use bevy_ecs::schedule::{IntoScheduleConfigs, SystemSet};
use bevy_ecs::system::{Res, ResMut};
use fk_math::Real;
use fk_math::nalgebra::Vector2;

use crate::input::{ButtonInput, RawInput};
use crate::{App, Plugin, PreUpdate, Time};

pub use rebind::{Part, Rebind, RebindError, RebindTarget, Resolution};

/// The actions of a game.
///
/// Usually a fieldless enum. [`Action::name`] is the action's key in the bindings file, so it
/// must be a TOML bare key (letters, digits, `_`, `-`) and must not change between versions.
pub trait Action: Copy + Eq + Hash + Debug + Send + Sync + 'static {
    /// Every action, in the order the bindings file lists them.
    const ALL: &'static [Self];

    /// The action's name in the bindings file.
    fn name(self) -> &'static str;

    /// Whether the action is a button or an axis.
    fn kind(self) -> ActionKind;

    /// The action called `name`, if any.
    fn from_name(name: &str) -> Option<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|action| action.name() == name)
    }
}

/// What an action reads as.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ActionKind {
    /// Down or up, with how long it has been down.
    Button,
    /// A number, read with [`KeyMap::axis1`].
    Axis1,
    /// A 2D vector, read with [`KeyMap::axis2`].
    Axis2,
}

/// One way to drive an action.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Binding {
    /// A key or mouse button. Binds a button.
    Button(ButtonInput),
    /// −1 while `negative` is down, +1 while `positive` is. Binds a 1D axis.
    ButtonPair {
        /// The button for −1.
        negative: ButtonInput,
        /// The button for +1.
        positive: ButtonInput,
    },
    /// Four buttons for the four directions, x to the right and y up (forward). Binds a 2D axis;
    /// the button part of the axis is clamped to the unit disc, so diagonals are not faster.
    ButtonQuad {
        /// The button for +y.
        up: ButtonInput,
        /// The button for −y.
        down: ButtonInput,
        /// The button for −x.
        left: ButtonInput,
        /// The button for +x.
        right: ButtonInput,
    },
    /// Raw mouse motion in pixels this frame, x to the right, y down. Binds a 2D axis.
    MouseMotion,
    /// Wheel motion in lines this frame. Binds a 2D axis, or its vertical part a 1D axis.
    Wheel,
}

/// Something a binding listens to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Input {
    /// A key or mouse button.
    Button(ButtonInput),
    /// Mouse motion.
    MouseMotion,
    /// The wheel.
    Wheel,
}

impl Binding {
    /// Whether this binding can drive an action of kind `kind`.
    pub fn fits(&self, kind: ActionKind) -> bool {
        matches!(
            (self, kind),
            (Self::Button(_), ActionKind::Button)
                | (Self::ButtonPair { .. }, ActionKind::Axis1)
                | (Self::ButtonQuad { .. }, ActionKind::Axis2)
                | (Self::MouseMotion, ActionKind::Axis2)
                | (Self::Wheel, ActionKind::Axis1 | ActionKind::Axis2)
        )
    }

    /// Everything the binding listens to.
    pub fn inputs(&self) -> Vec<Input> {
        match *self {
            Self::Button(button) => vec![Input::Button(button)],
            Self::ButtonPair { negative, positive } => {
                vec![Input::Button(negative), Input::Button(positive)]
            }
            Self::ButtonQuad {
                up,
                down,
                left,
                right,
            } => [up, down, left, right].map(Input::Button).to_vec(),
            Self::MouseMotion => vec![Input::MouseMotion],
            Self::Wheel => vec![Input::Wheel],
        }
    }

    /// Whether the binding listens to `input`.
    pub fn uses(&self, input: Input) -> bool {
        self.inputs().contains(&input)
    }
}

/// A short name for a player: the buttons as [`ButtonInput`] names them, a pair as
/// `negative/positive`, four directions as `up/left/down/right` (`W/A/S/D`).
impl std::fmt::Display for Binding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Button(button) => write!(f, "{button}"),
            Self::ButtonPair { negative, positive } => write!(f, "{negative}/{positive}"),
            Self::ButtonQuad {
                up,
                down,
                left,
                right,
            } => write!(f, "{up}/{left}/{down}/{right}"),
            Self::MouseMotion => f.write_str("Mouse"),
            Self::Wheel => f.write_str("Wheel"),
        }
    }
}

/// Error from [`Bindings::set`]: a binding does not fit the action's kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MisfitBinding<A> {
    /// The action.
    pub action: A,
    /// The binding that does not fit it.
    pub binding: Binding,
}

impl<A: Action> std::fmt::Display for MisfitBinding<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?} cannot drive {} ({:?})",
            self.binding,
            self.action.name(),
            self.action.kind()
        )
    }
}

impl<A: Action> std::error::Error for MisfitBinding<A> {}

/// The bindings of every action.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bindings<A: Action> {
    map: HashMap<A, Vec<Binding>>,
}

impl<A: Action> Bindings<A> {
    /// Every action unbound.
    pub fn new() -> Self {
        Self {
            map: A::ALL.iter().map(|&action| (action, Vec::new())).collect(),
        }
    }

    /// Adds `binding` to `action`, for building a default set.
    ///
    /// # Panics
    ///
    /// If the binding does not fit the action's kind.
    pub fn bind(mut self, action: A, binding: Binding) -> Self {
        assert!(
            binding.fits(action.kind()),
            "{}",
            MisfitBinding { action, binding }
        );
        self.map.entry(action).or_default().push(binding);
        self
    }

    /// The bindings of `action`.
    pub fn get(&self, action: A) -> &[Binding] {
        self.map.get(&action).map_or(&[], Vec::as_slice)
    }

    /// Replaces the bindings of `action`.
    pub fn set(&mut self, action: A, bindings: Vec<Binding>) -> Result<(), MisfitBinding<A>> {
        if let Some(&binding) = bindings.iter().find(|b| !b.fits(action.kind())) {
            return Err(MisfitBinding { action, binding });
        }
        self.map.insert(action, bindings);
        Ok(())
    }

    /// The actions, other than `except`, with a binding that listens to `input`.
    pub fn actions_using(&self, input: Input, except: Option<A>) -> Vec<A> {
        A::ALL
            .iter()
            .copied()
            .filter(|&action| Some(action) != except)
            .filter(|&action| self.get(action).iter().any(|b| b.uses(input)))
            .collect()
    }

    fn get_mut(&mut self, action: A) -> &mut Vec<Binding> {
        self.map.entry(action).or_default()
    }
}

impl<A: Action> Default for Bindings<A> {
    fn default() -> Self {
        Self::new()
    }
}

/// The state of one action this frame.
///
/// An axis counts as held while its value is not zero, so axes have press and release edges and
/// a hold time too.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ActionState {
    held: bool,
    pressed: bool,
    released: bool,
    held_for: Real,
    released_after: Option<Real>,
    value: Vector2<Real>,
}

impl ActionState {
    /// Whether the action is down.
    pub fn held(&self) -> bool {
        self.held
    }

    /// Whether the action went down this frame.
    pub fn pressed(&self) -> bool {
        self.pressed
    }

    /// Whether the action went up this frame. A press and release within one frame reports both
    /// [`pressed`](Self::pressed) and `released`, and is not [`held`](Self::held).
    pub fn released(&self) -> bool {
        self.released
    }

    /// Seconds the action has been down, 0 on the frame it went down and while it is up.
    pub fn held_for(&self) -> Real {
        self.held_for
    }

    /// On the frame the action went up, how many seconds it had been down.
    pub fn released_after(&self) -> Option<Real> {
        self.released_after
    }

    /// Value of a 1D axis; 1 while a button is held.
    pub fn axis1(&self) -> Real {
        self.value.x
    }

    /// Value of a 2D axis.
    pub fn axis2(&self) -> Vector2<Real> {
        self.value
    }

    fn step(&mut self, down: bool, tapped: bool, value: Vector2<Real>, dt: Real) {
        let was = self.held;
        let tap = tapped && !was && !down;
        self.pressed = (down && !was) || tap;
        self.released = (was && !down) || tap;
        self.released_after = match (self.released, was) {
            (false, _) => None,
            (true, true) => Some(self.held_for + dt),
            (true, false) => Some(0.0),
        };
        self.held_for = if down && was { self.held_for + dt } else { 0.0 };
        self.held = down;
        self.value = value;
    }
}

/// The current bindings and the state of every action, as a resource.
#[derive(Resource)]
pub struct KeyMap<A: Action> {
    defaults: Bindings<A>,
    bindings: Bindings<A>,
    states: HashMap<A, ActionState>,
    rebind: Rebind<A>,
    /// Buttons held when a rebind started or captured, ignored until they go up.
    suppressed: HashSet<ButtonInput>,
    /// Actions to press at the next update whatever their keys say ([`KeyMap::trigger`]).
    triggered: HashSet<A>,
    file: Option<config::ConfigFile>,
    dirty: bool,
}

impl<A: Action> KeyMap<A> {
    /// A key map using `defaults`, with no bindings file.
    pub fn new(defaults: Bindings<A>) -> Self {
        Self {
            bindings: defaults.clone(),
            defaults,
            states: A::ALL
                .iter()
                .map(|&a| (a, ActionState::default()))
                .collect(),
            rebind: Rebind::Idle,
            suppressed: HashSet::new(),
            triggered: HashSet::new(),
            file: None,
            dirty: false,
        }
    }

    /// A key map whose bindings live in the file at `path`.
    ///
    /// The file is read now. Actions it does not list, or lists with bindings that cannot be
    /// read, keep their defaults. A missing file is created with the defaults; a file that is not
    /// valid TOML is left alone and the defaults are used until it is fixed.
    pub fn with_file(defaults: Bindings<A>, path: impl Into<PathBuf>) -> Self {
        let mut map = Self::new(defaults);
        map.file = Some(config::ConfigFile::new(path.into()));
        map.reload();
        map
    }

    /// The bindings file, if there is one.
    pub fn file_path(&self) -> Option<&Path> {
        self.file.as_ref().map(|file| file.path.as_path())
    }

    /// The current bindings.
    pub fn bindings(&self) -> &Bindings<A> {
        &self.bindings
    }

    /// The default bindings the game supplied.
    pub fn defaults(&self) -> &Bindings<A> {
        &self.defaults
    }

    /// Replaces the bindings of `action`; they are saved to the bindings file this frame.
    pub fn set_bindings(
        &mut self,
        action: A,
        bindings: Vec<Binding>,
    ) -> Result<(), MisfitBinding<A>> {
        self.bindings.set(action, bindings)?;
        self.dirty = true;
        Ok(())
    }

    /// Puts every action back on its default bindings and cancels any rebind.
    pub fn reset_to_defaults(&mut self) {
        self.bindings = self.defaults.clone();
        self.rebind = Rebind::Idle;
        self.dirty = true;
    }

    /// Puts `action` back on its default bindings.
    pub fn reset_action(&mut self, action: A) {
        let defaults = self.defaults.get(action).to_vec();
        *self.bindings.get_mut(action) = defaults;
        self.dirty = true;
    }

    /// The state of `action` this frame.
    pub fn state(&self, action: A) -> ActionState {
        self.states.get(&action).copied().unwrap_or_default()
    }

    /// Whether `action` is down.
    pub fn held(&self, action: A) -> bool {
        self.state(action).held()
    }

    /// Whether `action` went down this frame.
    pub fn pressed(&self, action: A) -> bool {
        self.state(action).pressed()
    }

    /// Whether `action` went up this frame.
    pub fn released(&self, action: A) -> bool {
        self.state(action).released()
    }

    /// Seconds `action` has been down.
    pub fn held_for(&self, action: A) -> Real {
        self.state(action).held_for()
    }

    /// On the frame `action` went up, how many seconds it had been down.
    pub fn released_after(&self, action: A) -> Option<Real> {
        self.state(action).released_after()
    }

    /// Value of the 1D axis `action`.
    pub fn axis1(&self, action: A) -> Real {
        self.state(action).axis1()
    }

    /// Value of the 2D axis `action`.
    pub fn axis2(&self, action: A) -> Vector2<Real> {
        self.state(action).axis2()
    }

    /// Presses `action` at the next [`update`](Self::update), as if a key bound to it had been
    /// tapped: pressed and released that frame. For a console or a script; a rebind in progress
    /// drops it.
    pub fn trigger(&mut self, action: A) {
        self.triggered.insert(action);
    }

    /// Reads this frame's input into the actions. `dt` is the frame duration.
    ///
    /// While a rebind is in progress every action reads as released, so the input being captured
    /// does not also trigger what it is currently bound to.
    pub fn update(&mut self, input: &RawInput, dt: Real) {
        self.suppressed.retain(|&button| input.held(button));
        if !matches!(self.rebind, Rebind::Idle) {
            self.capture(input);
        }
        let frozen = !matches!(self.rebind, Rebind::Idle);
        for &action in A::ALL {
            let triggered = self.triggered.remove(&action);
            let (down, tapped, value) = if frozen {
                (false, false, Vector2::zeros())
            } else {
                let (down, tapped, value) = self.read(action, input);
                (down, tapped || triggered, value)
            };
            if let Some(state) = self.states.get_mut(&action) {
                state.step(down, tapped, value, dt);
            }
        }
    }

    /// Whether `button` is down and not suppressed, and whether it went down this frame.
    fn button(&self, input: &RawInput, button: ButtonInput) -> (bool, bool) {
        if self.suppressed.contains(&button) {
            (false, false)
        } else {
            (input.held(button), input.just_pressed(button))
        }
    }

    fn read(&self, action: A, input: &RawInput) -> (bool, bool, Vector2<Real>) {
        let level = |button| Real::from(u8::from(self.button(input, button).0));
        let mut down = false;
        let mut tapped = false;
        let mut buttons = Vector2::zeros();
        let mut analog = Vector2::zeros();
        for binding in self.bindings.get(action) {
            match *binding {
                Binding::Button(button) => {
                    let (held, pressed) = self.button(input, button);
                    down |= held;
                    tapped |= pressed;
                }
                Binding::ButtonPair { negative, positive } => {
                    buttons.x += level(positive) - level(negative);
                }
                Binding::ButtonQuad {
                    up,
                    down: back,
                    left,
                    right,
                } => {
                    buttons += Vector2::new(level(right) - level(left), level(up) - level(back));
                }
                Binding::MouseMotion => analog += input.mouse_motion(),
                Binding::Wheel => match action.kind() {
                    ActionKind::Axis1 => analog.x += input.wheel().y,
                    _ => analog += input.wheel(),
                },
            }
        }
        match action.kind() {
            ActionKind::Button => {
                let value = Vector2::new(Real::from(u8::from(down)), 0.0);
                (down, tapped, value)
            }
            ActionKind::Axis1 => {
                let value = Vector2::new(buttons.x.clamp(-1.0, 1.0) + analog.x, 0.0);
                (value.x != 0.0, false, value)
            }
            ActionKind::Axis2 => {
                let norm = buttons.norm();
                if norm > 1.0 {
                    buttons /= norm;
                }
                let value = buttons + analog;
                (value != Vector2::zeros(), false, value)
            }
        }
    }

    /// Writes the bindings file now if bindings changed since it was last written.
    pub fn save_if_dirty(&mut self) {
        if !std::mem::take(&mut self.dirty) {
            return;
        }
        if let Some(file) = &mut self.file
            && let Err(error) = file.save(&self.bindings)
        {
            tracing::warn!(
                "could not save bindings to {}: {error}",
                file.path.display()
            );
        }
    }

    /// Reloads the bindings file if it changed on disk; checks at most once per second of `now`.
    pub fn poll_file(&mut self, now: Real) {
        if self.file.as_mut().is_some_and(|file| file.changed(now)) {
            self.reload();
        }
    }

    fn reload(&mut self) {
        let Some(file) = &mut self.file else { return };
        match file.load(&self.defaults) {
            config::Loaded::Bindings(bindings) => {
                if bindings != self.bindings {
                    tracing::info!("bindings loaded from {}", file.path.display());
                }
                self.bindings = bindings;
            }
            config::Loaded::Missing => {
                self.bindings = self.defaults.clone();
                self.dirty = true;
                self.save_if_dirty();
            }
            config::Loaded::Unreadable => self.bindings = self.defaults.clone(),
        }
    }
}

/// The system set that updates every [`KeyMap`], in [`PreUpdate`]. Order `PreUpdate` systems
/// that read actions after it.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KeyMapSystems;

/// Adds a [`KeyMap<A>`] resource, updated every frame in [`PreUpdate`].
pub struct KeyMapPlugin<A: Action> {
    defaults: Bindings<A>,
    file: Option<PathBuf>,
}

impl<A: Action> KeyMapPlugin<A> {
    /// A key map with the game's default bindings and no bindings file.
    pub fn new(defaults: Bindings<A>) -> Self {
        Self {
            defaults,
            file: None,
        }
    }

    /// Keeps the player's bindings in the file at `path`; see [`KeyMap::with_file`].
    pub fn with_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.file = Some(path.into());
        self
    }
}

impl<A: Action> Plugin for KeyMapPlugin<A> {
    fn build(&self, app: &mut App) {
        let defaults = self.defaults.clone();
        let map = match &self.file {
            Some(path) => KeyMap::with_file(defaults, path),
            None => KeyMap::new(defaults),
        };
        app.insert_resource(map)
            .add_systems(PreUpdate, update_key_map::<A>.in_set(KeyMapSystems));
    }
}

fn update_key_map<A: Action>(input: Res<RawInput>, time: Res<Time>, mut map: ResMut<KeyMap<A>>) {
    map.poll_file(time.elapsed());
    map.update(&input, time.delta());
    map.save_if_dirty();
}

#[cfg(test)]
mod tests;
