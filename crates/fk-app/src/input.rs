//! Raw input, on physical keys, the mouse and gamepads, and the cursor grab.
//!
//! Keys are [`KeyCode`]s: positions on the keyboard, named after the key at that position on a
//! US QWERTY layout, whatever the active layout prints on it. `KeyCode::KeyW` is the key labelled
//! W on QWERTY and Z on AZERTY, so a binding on it is the same binding for both.
//!
//! Gamepads are [`PadButton`]s and [`Stick`]s, named by their place on the pad whatever it prints
//! on them; every pad connected drives the same ones.

use std::collections::HashSet;
use std::hash::Hash;

use bevy_ecs::resource::Resource;
use fk_math::Real;
use fk_math::nalgebra::Vector2;

pub use winit::event::MouseButton;
pub use winit::keyboard::KeyCode;

/// A button of a gamepad, by its place: `South` is A on an Xbox pad and Cross on a PlayStation
/// one. The triggers are buttons too, down past 40% of their travel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PadButton {
    /// The bottom face button: A, Cross.
    South,
    /// The right face button: B, Circle.
    East,
    /// The left face button: X, Square.
    West,
    /// The top face button: Y, Triangle.
    North,
    /// The left shoulder button: LB, L1.
    LeftBumper,
    /// The right shoulder button: RB, R1.
    RightBumper,
    /// The left trigger: LT, L2.
    LeftTrigger,
    /// The right trigger: RT, R2.
    RightTrigger,
    /// The left middle button: View or Back, Share.
    Select,
    /// The right middle button: Menu or Start, Options.
    Start,
    /// The left stick pressed in: LS, L3.
    LeftStick,
    /// The right stick pressed in: RS, R3.
    RightStick,
    /// The D-pad, up.
    DPadUp,
    /// The D-pad, down.
    DPadDown,
    /// The D-pad, left.
    DPadLeft,
    /// The D-pad, right.
    DPadRight,
}

impl PadButton {
    /// Every pad button.
    pub const ALL: [Self; 16] = [
        Self::South,
        Self::East,
        Self::West,
        Self::North,
        Self::LeftBumper,
        Self::RightBumper,
        Self::LeftTrigger,
        Self::RightTrigger,
        Self::Select,
        Self::Start,
        Self::LeftStick,
        Self::RightStick,
        Self::DPadUp,
        Self::DPadDown,
        Self::DPadLeft,
        Self::DPadRight,
    ];

    /// Its short name on an Xbox pad, for a player: `A`, `LT`, `LS`, `D-pad up`.
    pub fn label(self) -> &'static str {
        match self {
            Self::South => "A",
            Self::East => "B",
            Self::West => "X",
            Self::North => "Y",
            Self::LeftBumper => "LB",
            Self::RightBumper => "RB",
            Self::LeftTrigger => "LT",
            Self::RightTrigger => "RT",
            Self::Select => "View",
            Self::Start => "Menu",
            Self::LeftStick => "LS",
            Self::RightStick => "RS",
            Self::DPadUp => "D-pad up",
            Self::DPadDown => "D-pad down",
            Self::DPadLeft => "D-pad left",
            Self::DPadRight => "D-pad right",
        }
    }
}

/// One of a gamepad's two sticks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stick {
    /// The left stick.
    Left,
    /// The right stick.
    Right,
}

impl Stick {
    /// Its name for a player.
    pub fn label(self) -> &'static str {
        match self {
            Self::Left => "Left stick",
            Self::Right => "Right stick",
        }
    }
}

/// A way a stick is pushed, y up.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Direction {
    /// Up, away from the player.
    Up,
    /// Down, towards the player.
    Down,
    /// To the left.
    Left,
    /// To the right.
    Right,
}

impl Direction {
    /// Its name for a player.
    pub fn label(self) -> &'static str {
        match self {
            Self::Up => "up",
            Self::Down => "down",
            Self::Left => "left",
            Self::Right => "right",
        }
    }

    /// The unit push its way, x to the right and y up.
    pub fn vector(self) -> Vector2<Real> {
        match self {
            Self::Up => Vector2::new(0.0, 1.0),
            Self::Down => Vector2::new(0.0, -1.0),
            Self::Left => Vector2::new(-1.0, 0.0),
            Self::Right => Vector2::new(1.0, 0.0),
        }
    }
}

/// A key, a mouse button or a pad button: an input that is either down or up.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ButtonInput {
    /// A physical key.
    Key(KeyCode),
    /// A mouse button.
    Mouse(MouseButton),
    /// A gamepad button.
    Pad(PadButton),
}

/// A short name for a player, after the US QWERTY key at that position: `KeyW` is `W`,
/// `Digit1` is `1`, `ShiftLeft` is `Shift`, the left mouse button is `Mouse left`.
impl std::fmt::Display for ButtonInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Key(key) => {
                // KeyCode's variants are all fieldless, so Debug prints the variant's name.
                let name = format!("{key:?}");
                let short = match *key {
                    KeyCode::Comma => ",",
                    KeyCode::Period => ".",
                    KeyCode::Minus => "-",
                    KeyCode::Equal => "=",
                    KeyCode::Slash => "/",
                    KeyCode::Semicolon => ";",
                    KeyCode::Quote => "'",
                    KeyCode::BracketLeft => "[",
                    KeyCode::BracketRight => "]",
                    KeyCode::Backslash => "\\",
                    KeyCode::Backquote => "`",
                    KeyCode::ArrowUp => "Up",
                    KeyCode::ArrowDown => "Down",
                    KeyCode::ArrowLeft => "Left",
                    KeyCode::ArrowRight => "Right",
                    KeyCode::ShiftLeft | KeyCode::ShiftRight => "Shift",
                    KeyCode::ControlLeft | KeyCode::ControlRight => "Ctrl",
                    KeyCode::AltLeft | KeyCode::AltRight => "Alt",
                    _ => {
                        let stripped = name
                            .strip_prefix("Key")
                            .or_else(|| name.strip_prefix("Digit"))
                            .unwrap_or(&name);
                        return f.write_str(stripped);
                    }
                };
                f.write_str(short)
            }
            Self::Mouse(MouseButton::Left) => f.write_str("Mouse left"),
            Self::Mouse(MouseButton::Right) => f.write_str("Mouse right"),
            Self::Mouse(MouseButton::Middle) => f.write_str("Mouse middle"),
            Self::Mouse(MouseButton::Back) => f.write_str("Mouse back"),
            Self::Mouse(MouseButton::Forward) => f.write_str("Mouse forward"),
            Self::Mouse(MouseButton::Other(n)) => write!(f, "Mouse {n}"),
            Self::Pad(button) => f.write_str(button.label()),
        }
    }
}

impl From<KeyCode> for ButtonInput {
    fn from(key: KeyCode) -> Self {
        Self::Key(key)
    }
}

impl From<MouseButton> for ButtonInput {
    fn from(button: MouseButton) -> Self {
        Self::Mouse(button)
    }
}

impl From<PadButton> for ButtonInput {
    fn from(button: PadButton) -> Self {
        Self::Pad(button)
    }
}

/// The down/up state of a set of buttons, with the changes of the current frame.
#[derive(Clone, Debug)]
pub struct ButtonSet<T> {
    held: HashSet<T>,
    pressed: HashSet<T>,
    released: HashSet<T>,
}

impl<T: Copy + Eq + Hash> ButtonSet<T> {
    /// Whether `button` is down.
    pub fn held(&self, button: T) -> bool {
        self.held.contains(&button)
    }

    /// Whether `button` went down this frame. A button can be pressed and released in the same
    /// frame, in which case it is not [`held`](Self::held).
    pub fn just_pressed(&self, button: T) -> bool {
        self.pressed.contains(&button)
    }

    /// Whether `button` went up this frame.
    pub fn just_released(&self, button: T) -> bool {
        self.released.contains(&button)
    }

    /// The buttons that are down.
    pub fn iter_held(&self) -> impl Iterator<Item = T> + '_ {
        self.held.iter().copied()
    }

    /// The buttons that went down this frame.
    pub fn iter_just_pressed(&self) -> impl Iterator<Item = T> + '_ {
        self.pressed.iter().copied()
    }

    /// Records `button` going down. Does nothing if it is already down (key repeat).
    pub fn press(&mut self, button: T) {
        if self.held.insert(button) {
            self.pressed.insert(button);
        }
    }

    /// Records `button` going up. Does nothing if it is not down.
    pub fn release(&mut self, button: T) {
        if self.held.remove(&button) {
            self.released.insert(button);
        }
    }

    /// Releases every button that is down.
    pub fn release_all(&mut self) {
        self.released.extend(self.held.drain());
    }

    fn end_frame(&mut self) {
        self.pressed.clear();
        self.released.clear();
    }
}

impl<T> Default for ButtonSet<T> {
    fn default() -> Self {
        Self {
            held: HashSet::new(),
            pressed: HashSet::new(),
            released: HashSet::new(),
        }
    }
}

/// Everything the player did with the keyboard, the mouse and the gamepads since the previous
/// frame.
///
/// The app fills it from window events and the pads before each frame and clears the per-frame
/// parts after it. Losing the window focus releases every key and button and lets the sticks go.
#[derive(Resource, Clone, Debug, Default)]
pub struct RawInput {
    /// Physical keys.
    pub keys: ButtonSet<KeyCode>,
    /// Mouse buttons.
    pub mouse_buttons: ButtonSet<MouseButton>,
    /// Gamepad buttons, of every pad connected.
    pub pad_buttons: ButtonSet<PadButton>,
    sticks: [Vector2<Real>; 2],
    pad_last: bool,
    mouse_motion: Vector2<Real>,
    wheel: Vector2<Real>,
    typed: String,
    typed_keys: Vec<KeyCode>,
}

impl RawInput {
    /// Pixels of a smooth (touchpad) scroll counted as one wheel line.
    pub const PIXELS_PER_LINE: Real = 40.0;

    /// Whether a key, a mouse button or a pad button is down.
    pub fn held(&self, button: ButtonInput) -> bool {
        match button {
            ButtonInput::Key(key) => self.keys.held(key),
            ButtonInput::Mouse(button) => self.mouse_buttons.held(button),
            ButtonInput::Pad(button) => self.pad_buttons.held(button),
        }
    }

    /// Whether a key, a mouse button or a pad button went down this frame.
    pub fn just_pressed(&self, button: ButtonInput) -> bool {
        match button {
            ButtonInput::Key(key) => self.keys.just_pressed(key),
            ButtonInput::Mouse(button) => self.mouse_buttons.just_pressed(button),
            ButtonInput::Pad(button) => self.pad_buttons.just_pressed(button),
        }
    }

    /// Whether a key, a mouse button or a pad button went up this frame.
    pub fn just_released(&self, button: ButtonInput) -> bool {
        match button {
            ButtonInput::Key(key) => self.keys.just_released(key),
            ButtonInput::Mouse(button) => self.mouse_buttons.just_released(button),
            ButtonInput::Pad(button) => self.pad_buttons.just_released(button),
        }
    }

    /// The keys, mouse buttons and pad buttons that are down.
    pub fn iter_held(&self) -> impl Iterator<Item = ButtonInput> + '_ {
        let keys = self.keys.iter_held().map(ButtonInput::Key);
        keys.chain(self.mouse_buttons.iter_held().map(ButtonInput::Mouse))
            .chain(self.pad_buttons.iter_held().map(ButtonInput::Pad))
    }

    /// The keys, mouse buttons and pad buttons that went down this frame.
    pub fn iter_just_pressed(&self) -> impl Iterator<Item = ButtonInput> + '_ {
        let keys = self.keys.iter_just_pressed().map(ButtonInput::Key);
        keys.chain(
            self.mouse_buttons
                .iter_just_pressed()
                .map(ButtonInput::Mouse),
        )
        .chain(self.pad_buttons.iter_just_pressed().map(ButtonInput::Pad))
    }

    /// Records a key, a mouse button or a pad button going down.
    pub fn press(&mut self, button: impl Into<ButtonInput>) {
        let button = button.into();
        if !self.held(button) {
            self.pad_last = matches!(button, ButtonInput::Pad(_));
        }
        match button {
            ButtonInput::Key(key) => self.keys.press(key),
            ButtonInput::Mouse(button) => self.mouse_buttons.press(button),
            ButtonInput::Pad(button) => self.pad_buttons.press(button),
        }
    }

    /// Whether a gamepad was the last thing the player used, rather than the keyboard or the
    /// mouse: for what a screen shows to press.
    pub fn pad_last(&self) -> bool {
        self.pad_last
    }

    /// Records a key, a mouse button or a pad button going up.
    pub fn release(&mut self, button: impl Into<ButtonInput>) {
        match button.into() {
            ButtonInput::Key(key) => self.keys.release(key),
            ButtonInput::Mouse(button) => self.mouse_buttons.release(button),
            ButtonInput::Pad(button) => self.pad_buttons.release(button),
        }
    }

    /// Releases every key, mouse button and pad button, and lets the sticks go.
    pub fn release_all(&mut self) {
        self.keys.release_all();
        self.mouse_buttons.release_all();
        self.pad_buttons.release_all();
        self.sticks = [Vector2::zeros(); 2];
    }

    /// Where `stick` is pushed, within the unit disc, x to the right and y up: zero in its dead
    /// zone, and whole a little before the stick's rim.
    pub fn stick(&self, stick: Stick) -> Vector2<Real> {
        self.sticks[stick as usize]
    }

    /// Whether `stick` is pushed `direction`, at least `threshold` of its whole push that way
    /// and more that way than across it, so that a diagonal pushes one way only.
    pub fn stick_pushed(&self, stick: Stick, direction: Direction, threshold: Real) -> bool {
        let push = self.stick(stick);
        let way = direction.vector();
        let along = push.dot(&way);
        let across = push.x * way.y - push.y * way.x;
        along >= threshold && along >= across.abs()
    }

    /// Records where `stick` is as the pad reads it, each axis from −1 to 1, y up; the dead zone
    /// is taken out here.
    pub fn set_stick(&mut self, stick: Stick, raw: Vector2<Real>) {
        let pushed = dead_zone(raw);
        if pushed != Vector2::zeros() && self.sticks[stick as usize] == Vector2::zeros() {
            self.pad_last = true;
        }
        self.sticks[stick as usize] = pushed;
    }

    /// Raw mouse motion this frame in device pixels, x to the right, y down. Not affected by
    /// pointer acceleration or by the cursor being stopped at the window edge.
    pub fn mouse_motion(&self) -> Vector2<Real> {
        self.mouse_motion
    }

    /// Wheel motion this frame in lines, x to the right, y away from the player.
    pub fn wheel(&self) -> Vector2<Real> {
        self.wheel
    }

    /// Records raw mouse motion.
    pub fn add_mouse_motion(&mut self, delta: Vector2<Real>) {
        /// Pixels the mouse moves within a frame to count as used: not a knock of the desk.
        const MOVED: Real = 6.0;
        self.mouse_motion += delta;
        if self.mouse_motion.norm() >= MOVED {
            self.pad_last = false;
        }
    }

    /// Records wheel motion, in lines.
    pub fn add_wheel(&mut self, delta: Vector2<Real>) {
        self.wheel += delta;
    }

    /// The text typed this frame, in the active layout, key repeats included: what a text field
    /// takes. Control characters (Backspace, Enter, Escape, …) are left out: read those from
    /// [`typed_keys`](Self::typed_keys).
    pub fn typed(&self) -> &str {
        &self.typed
    }

    /// The keys that went down this frame, key repeats included, in order: what a text field's
    /// editing keys take.
    pub fn typed_keys(&self) -> &[KeyCode] {
        &self.typed_keys
    }

    /// Records a key typed, `text` what the layout made of it, once more for every key repeat.
    pub fn add_typed(&mut self, key: KeyCode, text: Option<&str>) {
        self.typed_keys.push(key);
        if let Some(text) = text {
            self.typed.extend(text.chars().filter(|c| !c.is_control()));
        }
    }

    /// Forgets every key, held or typed, until each goes down again, as if the keyboard had not
    /// been touched: for a text field that takes the keyboard from whatever is bound to it.
    pub fn take_keyboard(&mut self) {
        self.keys = ButtonSet::default();
        self.typed.clear();
        self.typed_keys.clear();
    }

    /// Clears what happened this frame, keeping what is held.
    pub(crate) fn end_frame(&mut self) {
        self.keys.end_frame();
        self.mouse_buttons.end_frame();
        self.mouse_motion = Vector2::zeros();
        self.wheel = Vector2::zeros();
        self.typed.clear();
        self.typed_keys.clear();
    }
}

/// A stick's position with its dead zone taken out: radial, so the diagonals are not cut, and
/// rescaled, so the push still starts from zero at its edge and is whole before the rim.
fn dead_zone(raw: Vector2<Real>) -> Vector2<Real> {
    /// Below this the stick is let go: worn sticks rest a little off centre.
    const INNER: Real = 0.15;
    /// Past this the push is whole: few sticks reach their rim on every side.
    const OUTER: Real = 0.95;
    let norm = raw.norm();
    if norm <= INNER {
        return Vector2::zeros();
    }
    raw * (((norm - INNER) / (OUTER - INNER)).min(1.0) / norm)
}

/// Whether the cursor is grabbed: hidden and held in the window so the mouse only turns the view.
///
/// Clicking in the window grabs it (that click reaches no action). Call [`CursorGrab::release`]
/// to let it go; losing the window focus lets it go too.
#[derive(Resource, Clone, Debug)]
pub struct CursorGrab {
    grabbed: bool,
    requested: Option<bool>,
    /// Whether a click in the window grabs the cursor.
    pub grab_on_click: bool,
    /// Whether the window gaining the focus grabs the cursor (opening focused counts, and so
    /// does being handed the window focused), so that the mouse moves the view at once.
    pub grab_on_focus: bool,
    /// Whether the cursor stays hidden over the window even when it is not grabbed.
    pub hidden: bool,
}

impl CursorGrab {
    /// Whether the cursor is grabbed.
    pub fn is_grabbed(&self) -> bool {
        self.grabbed
    }

    /// Asks for the cursor to be grabbed after this frame, if the window has the focus then.
    pub fn grab(&mut self) {
        self.requested = Some(true);
    }

    /// Asks for the cursor to be released after this frame.
    pub fn release(&mut self) {
        self.requested = Some(false);
    }

    pub(crate) fn take_request(&mut self) -> Option<bool> {
        self.requested.take().filter(|&want| want != self.grabbed)
    }

    pub(crate) fn set_grabbed(&mut self, grabbed: bool) {
        self.grabbed = grabbed;
    }
}

impl Default for CursorGrab {
    fn default() -> Self {
        Self {
            grabbed: false,
            requested: None,
            grab_on_click: true,
            grab_on_focus: false,
            hidden: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edges_last_one_frame() {
        let mut input = RawInput::default();
        input.press(KeyCode::KeyW);
        assert!(input.held(KeyCode::KeyW.into()) && input.just_pressed(KeyCode::KeyW.into()));
        input.end_frame();
        assert!(input.held(KeyCode::KeyW.into()) && !input.just_pressed(KeyCode::KeyW.into()));
        input.release(KeyCode::KeyW);
        assert!(!input.held(KeyCode::KeyW.into()) && input.just_released(KeyCode::KeyW.into()));
    }

    #[test]
    fn key_repeat_is_not_a_new_press() {
        let mut input = RawInput::default();
        input.press(KeyCode::Space);
        input.end_frame();
        input.press(KeyCode::Space);
        assert!(!input.just_pressed(KeyCode::Space.into()));
    }

    #[test]
    fn release_all_releases_keys_and_buttons() {
        let mut input = RawInput::default();
        input.press(KeyCode::KeyA);
        input.press(MouseButton::Right);
        input.end_frame();
        input.release_all();
        assert!(input.just_released(KeyCode::KeyA.into()));
        assert!(input.just_released(MouseButton::Right.into()));
        assert!(!input.held(MouseButton::Right.into()));
    }

    #[test]
    fn a_stick_rests_in_its_dead_zone_and_is_whole_before_its_rim() {
        let mut input = RawInput::default();
        input.set_stick(Stick::Left, Vector2::new(0.1, -0.08));
        assert_eq!(input.stick(Stick::Left), Vector2::zeros());
        input.set_stick(Stick::Left, Vector2::new(0.0, 0.97));
        assert!((input.stick(Stick::Left) - Vector2::new(0.0, 1.0)).norm() < 1e-12);
        input.set_stick(Stick::Right, Vector2::new(0.55, 0.0));
        assert!((input.stick(Stick::Right).x - 0.5).abs() < 1e-12);
        input.release_all();
        assert_eq!(input.stick(Stick::Right), Vector2::zeros());
    }

    #[test]
    fn the_last_thing_used_is_known() {
        let mut input = RawInput::default();
        assert!(!input.pad_last());
        input.set_stick(Stick::Left, Vector2::new(0.0, 0.9));
        assert!(input.pad_last());
        input.add_mouse_motion(Vector2::new(1.0, 1.0));
        assert!(input.pad_last(), "a knock of the desk");
        input.press(KeyCode::KeyW);
        assert!(!input.pad_last());
        // Held, the pad is read again every frame: no new use.
        input.set_stick(Stick::Left, Vector2::new(0.0, 0.8));
        assert!(!input.pad_last());
        input.press(PadButton::South);
        assert!(input.pad_last());
    }

    #[test]
    fn a_stick_pushes_one_way_at_a_time() {
        let mut input = RawInput::default();
        input.set_stick(Stick::Left, Vector2::new(0.5, 0.8));
        assert!(input.stick_pushed(Stick::Left, Direction::Up, 0.5));
        assert!(!input.stick_pushed(Stick::Left, Direction::Right, 0.5));
        assert!(!input.stick_pushed(Stick::Left, Direction::Down, 0.5));
        assert!(!input.stick_pushed(Stick::Right, Direction::Up, 0.5));
    }

    #[test]
    fn motion_accumulates_within_a_frame() {
        let mut input = RawInput::default();
        input.add_mouse_motion(Vector2::new(1.0, 2.0));
        input.add_mouse_motion(Vector2::new(3.0, -1.0));
        assert_eq!(input.mouse_motion(), Vector2::new(4.0, 1.0));
        input.end_frame();
        assert_eq!(input.mouse_motion(), Vector2::zeros());
    }
}
