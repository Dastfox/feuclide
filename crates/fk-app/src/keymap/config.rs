//! The bindings file: TOML, one key per action, each a list of bindings.
//!
//! ```toml
//! Move = [{ up = "KeyW", down = "KeyS", left = "KeyA", right = "KeyD" }]
//! Look = ["MouseMotion"]
//! Jump = ["Space"]
//! Inhale = ["MouseLeft", "PadLT"]
//! ```

use std::fmt::Write as _;
use std::io;
use std::path::PathBuf;
use std::time::SystemTime;

use fk_math::Real;
use serde::Deserialize;
use serde::de::IntoDeserializer;
use serde::de::value::Error as ValueError;

use super::{Action, Binding, Bindings};
use crate::input::{ButtonInput, Direction, KeyCode, MouseButton, PadButton, Stick};

const HEADER: &str = "\
# Key bindings. Each action takes a list of bindings; an empty list leaves it unbound.
#
# Keys are physical positions, named after the key at that position on a US QWERTY
# keyboard: \"KeyW\" is the key labelled Z on AZERTY. Names are those of winit's KeyCode
# (\"KeyA\", \"Digit1\", \"Space\", \"ShiftLeft\", \"ArrowUp\", \"F1\", ...). Mouse buttons
# are \"MouseLeft\", \"MouseRight\", \"MouseMiddle\", \"MouseBack\", \"MouseForward\".
# Gamepad buttons, by their place on an Xbox pad: \"PadA\", \"PadB\", \"PadX\", \"PadY\",
# \"PadLB\", \"PadRB\", \"PadLT\", \"PadRT\", \"PadView\", \"PadMenu\", \"PadLS\", \"PadRS\"
# (the sticks pressed in), \"PadUp\", \"PadDown\", \"PadLeft\", \"PadRight\" (the D-pad), and a
# stick pushed one way, \"LeftStickUp\", \"RightStickLeft\", ...
#
# Axes take { negative = .., positive = .. } (1D), { up = .., down = .., left = ..,
# right = .. } (2D), \"MouseMotion\" (2D), \"Wheel\", \"LeftStick\" and \"RightStick\" (2D,
# where the stick is) and \"LeftStickMotion\" and \"RightStickMotion\" (2D, the stick
# turning as the mouse does).
#
# Changes are picked up while the game runs. Delete this file to restore the defaults.

";

/// Seconds between two checks of the file's modification time.
const POLL_INTERVAL: Real = 1.0;

#[derive(Deserialize)]
#[serde(untagged)]
enum BindingRepr {
    Name(String),
    Pair {
        negative: String,
        positive: String,
    },
    Quad {
        up: String,
        down: String,
        left: String,
        right: String,
    },
}

pub(super) enum Loaded<A: Action> {
    /// The bindings read, and whether the defaults' pad bindings were added to them.
    Bindings(Bindings<A>, bool),
    Missing,
    Unreadable,
}

pub(super) struct ConfigFile {
    pub(super) path: PathBuf,
    modified: Option<SystemTime>,
    next_check: Real,
}

impl ConfigFile {
    pub(super) fn new(path: PathBuf) -> Self {
        Self {
            path,
            modified: None,
            next_check: 0.0,
        }
    }

    fn modified_on_disk(&self) -> Option<SystemTime> {
        std::fs::metadata(&self.path)
            .and_then(|m| m.modified())
            .ok()
    }

    /// Whether the file changed on disk since it was last loaded or saved.
    pub(super) fn changed(&mut self, now: Real) -> bool {
        if now < self.next_check {
            return false;
        }
        self.next_check = now + POLL_INTERVAL;
        self.modified_on_disk() != self.modified
    }

    pub(super) fn load<A: Action>(&mut self, defaults: &Bindings<A>) -> Loaded<A> {
        self.modified = self.modified_on_disk();
        let text = match std::fs::read_to_string(&self.path) {
            Ok(text) => text,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Loaded::Missing,
            Err(error) => {
                tracing::warn!("could not read {}: {error}", self.path.display());
                return Loaded::Unreadable;
            }
        };
        match parse_upgrading(&text, defaults) {
            Ok((bindings, upgraded)) => Loaded::Bindings(bindings, upgraded),
            Err(error) => {
                tracing::warn!(
                    "{} is not valid TOML, using the default bindings until it is fixed: {error}",
                    self.path.display()
                );
                Loaded::Unreadable
            }
        }
    }

    pub(super) fn save<A: Action>(&mut self, bindings: &Bindings<A>) -> io::Result<()> {
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let temporary = self.path.with_extension("toml.tmp");
        std::fs::write(&temporary, write(bindings))?;
        std::fs::rename(&temporary, &self.path)?;
        self.modified = self.modified_on_disk();
        Ok(())
    }
}

/// Reads a bindings file over `defaults`. Fails only if the text is not a TOML table; entries
/// that cannot be read are logged and leave their action on its defaults.
#[cfg(test)]
pub(super) fn parse<A: Action>(
    text: &str,
    defaults: &Bindings<A>,
) -> Result<Bindings<A>, toml::de::Error> {
    parse_upgrading(text, defaults).map(|(bindings, _)| bindings)
}

/// [`parse`], and if the file binds nothing on a gamepad (written before the game had them),
/// the defaults' pad bindings added to every action it lists; says whether they were.
pub(super) fn parse_upgrading<A: Action>(
    text: &str,
    defaults: &Bindings<A>,
) -> Result<(Bindings<A>, bool), toml::de::Error> {
    let table: toml::Table = toml::from_str(text)?;
    let mut bindings = defaults.clone();
    let mut listed = Vec::new();
    for (name, value) in table {
        let Some(action) = A::from_name(&name) else {
            tracing::warn!("bindings file: unknown action {name:?}, ignored");
            continue;
        };
        match parse_action(action, value) {
            Ok(list) => {
                bindings
                    .set(action, list)
                    .expect("parse_action checks every binding fits");
                listed.push(action);
            }
            Err(error) => {
                tracing::warn!("bindings file: {name}: {error}; keeping its default bindings");
            }
        }
    }
    let padless = !listed
        .iter()
        .any(|&action| bindings.get(action).iter().any(Binding::is_pad));
    let mut upgraded = false;
    if padless {
        for &action in &listed {
            let pads = defaults.get(action).iter().filter(|b| b.is_pad());
            let mut list = bindings.get(action).to_vec();
            let before = list.len();
            list.extend(pads);
            upgraded |= list.len() > before;
            bindings
                .set(action, list)
                .expect("the defaults fit their actions");
        }
    }
    Ok((bindings, upgraded))
}

fn parse_action<A: Action>(action: A, value: toml::Value) -> Result<Vec<Binding>, String> {
    let reprs: Vec<BindingRepr> = value
        .try_into()
        .map_err(|_| "expected a list of bindings".to_owned())?;
    reprs
        .into_iter()
        .map(|repr| {
            let binding = parse_binding(repr)?;
            if binding.fits(action.kind()) {
                Ok(binding)
            } else {
                Err(format!("{binding:?} cannot drive a {:?}", action.kind()))
            }
        })
        .collect()
}

fn parse_binding(repr: BindingRepr) -> Result<Binding, String> {
    Ok(match repr {
        BindingRepr::Name(name) => match name.as_str() {
            "MouseMotion" => Binding::MouseMotion,
            "Wheel" => Binding::Wheel,
            "LeftStick" => Binding::Stick(Stick::Left),
            "RightStick" => Binding::Stick(Stick::Right),
            "LeftStickMotion" => Binding::StickMotion(Stick::Left),
            "RightStickMotion" => Binding::StickMotion(Stick::Right),
            _ => match stick_pushes().find(|(_, pushed)| *pushed == name) {
                Some((binding, _)) => binding,
                None => Binding::Button(parse_button(&name)?),
            },
        },
        BindingRepr::Pair { negative, positive } => Binding::ButtonPair {
            negative: parse_button(&negative)?,
            positive: parse_button(&positive)?,
        },
        BindingRepr::Quad {
            up,
            down,
            left,
            right,
        } => Binding::ButtonQuad {
            up: parse_button(&up)?,
            down: parse_button(&down)?,
            left: parse_button(&left)?,
            right: parse_button(&right)?,
        },
    })
}

fn parse_button(name: &str) -> Result<ButtonInput, String> {
    let mouse = match name {
        "MouseLeft" => Some(MouseButton::Left),
        "MouseRight" => Some(MouseButton::Right),
        "MouseMiddle" => Some(MouseButton::Middle),
        "MouseBack" => Some(MouseButton::Back),
        "MouseForward" => Some(MouseButton::Forward),
        _ => name
            .strip_prefix("Mouse")
            .and_then(|n| n.parse().ok())
            .map(MouseButton::Other),
    };
    if let Some(button) = mouse {
        return Ok(ButtonInput::Mouse(button));
    }
    if let Some(&pad) = PadButton::ALL.iter().find(|&&pad| pad_name(pad) == name) {
        return Ok(ButtonInput::Pad(pad));
    }
    let deserializer: serde::de::value::StrDeserializer<'_, ValueError> = name.into_deserializer();
    KeyCode::deserialize(deserializer)
        .map(ButtonInput::Key)
        .map_err(|_| format!("unknown key {name:?}"))
}

fn button_name(button: ButtonInput) -> String {
    match button {
        // KeyCode's variants are all fieldless, so Debug prints the serde name.
        ButtonInput::Key(key) => format!("{key:?}"),
        ButtonInput::Mouse(MouseButton::Left) => "MouseLeft".to_owned(),
        ButtonInput::Mouse(MouseButton::Right) => "MouseRight".to_owned(),
        ButtonInput::Mouse(MouseButton::Middle) => "MouseMiddle".to_owned(),
        ButtonInput::Mouse(MouseButton::Back) => "MouseBack".to_owned(),
        ButtonInput::Mouse(MouseButton::Forward) => "MouseForward".to_owned(),
        ButtonInput::Mouse(MouseButton::Other(n)) => format!("Mouse{n}"),
        ButtonInput::Pad(pad) => pad_name(pad).to_owned(),
    }
}

/// Every stick pushed one way, and its name in the file: `LeftStickUp`.
fn stick_pushes() -> impl Iterator<Item = (Binding, String)> {
    [(Stick::Left, "Left"), (Stick::Right, "Right")]
        .into_iter()
        .flat_map(|(stick, side)| {
            [
                (Direction::Up, "Up"),
                (Direction::Down, "Down"),
                (Direction::Left, "Left"),
                (Direction::Right, "Right"),
            ]
            .map(|(direction, way)| {
                (
                    Binding::StickPush(stick, direction),
                    format!("{side}Stick{way}"),
                )
            })
        })
}

/// A pad button's name in the file: `Pad` and its place on an Xbox pad.
fn pad_name(button: PadButton) -> &'static str {
    match button {
        PadButton::South => "PadA",
        PadButton::East => "PadB",
        PadButton::West => "PadX",
        PadButton::North => "PadY",
        PadButton::LeftBumper => "PadLB",
        PadButton::RightBumper => "PadRB",
        PadButton::LeftTrigger => "PadLT",
        PadButton::RightTrigger => "PadRT",
        PadButton::Select => "PadView",
        PadButton::Start => "PadMenu",
        PadButton::LeftStick => "PadLS",
        PadButton::RightStick => "PadRS",
        PadButton::DPadUp => "PadUp",
        PadButton::DPadDown => "PadDown",
        PadButton::DPadLeft => "PadLeft",
        PadButton::DPadRight => "PadRight",
    }
}

fn binding_text(binding: &Binding) -> String {
    let name = |button| format!("\"{}\"", button_name(button));
    match *binding {
        Binding::Button(button) => name(button),
        Binding::ButtonPair { negative, positive } => {
            format!(
                "{{ negative = {}, positive = {} }}",
                name(negative),
                name(positive)
            )
        }
        Binding::ButtonQuad {
            up,
            down,
            left,
            right,
        } => format!(
            "{{ up = {}, down = {}, left = {}, right = {} }}",
            name(up),
            name(down),
            name(left),
            name(right)
        ),
        Binding::MouseMotion => "\"MouseMotion\"".to_owned(),
        Binding::Wheel => "\"Wheel\"".to_owned(),
        Binding::Stick(Stick::Left) => "\"LeftStick\"".to_owned(),
        Binding::Stick(Stick::Right) => "\"RightStick\"".to_owned(),
        Binding::StickMotion(Stick::Left) => "\"LeftStickMotion\"".to_owned(),
        Binding::StickMotion(Stick::Right) => "\"RightStickMotion\"".to_owned(),
        Binding::StickPush(..) => stick_pushes()
            .find(|(push, _)| push == binding)
            .map(|(_, name)| format!("\"{name}\""))
            .expect("every stick push has a name"),
    }
}

/// Writes every action's bindings, in [`Action::ALL`] order, under an explanatory header.
pub(super) fn write<A: Action>(bindings: &Bindings<A>) -> String {
    let mut text = HEADER.to_owned();
    for &action in A::ALL {
        let list: Vec<String> = bindings.get(action).iter().map(binding_text).collect();
        writeln!(text, "{} = [{}]", action.name(), list.join(", ")).expect("writing to a String");
    }
    text
}
