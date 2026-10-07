use std::path::PathBuf;

use fk_math::nalgebra::Vector2;

use super::*;
use crate::input::{KeyCode, MouseButton};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Act {
    Move,
    Look,
    Zoom,
    Jump,
    Inhale,
}

impl Action for Act {
    const ALL: &'static [Self] = &[Act::Move, Act::Look, Act::Zoom, Act::Jump, Act::Inhale];

    fn name(self) -> &'static str {
        match self {
            Act::Move => "Move",
            Act::Look => "Look",
            Act::Zoom => "Zoom",
            Act::Jump => "Jump",
            Act::Inhale => "Inhale",
        }
    }

    fn kind(self) -> ActionKind {
        match self {
            Act::Move | Act::Look => ActionKind::Axis2,
            Act::Zoom => ActionKind::Axis1,
            Act::Jump | Act::Inhale => ActionKind::Button,
        }
    }
}

fn key(code: KeyCode) -> ButtonInput {
    ButtonInput::Key(code)
}

fn wasd() -> Binding {
    Binding::ButtonQuad {
        up: key(KeyCode::KeyW),
        down: key(KeyCode::KeyS),
        left: key(KeyCode::KeyA),
        right: key(KeyCode::KeyD),
    }
}

fn defaults() -> Bindings<Act> {
    Bindings::new()
        .bind(Act::Move, wasd())
        .bind(Act::Look, Binding::MouseMotion)
        .bind(Act::Zoom, Binding::Wheel)
        .bind(Act::Jump, Binding::Button(key(KeyCode::Space)))
        .bind(
            Act::Inhale,
            Binding::Button(ButtonInput::Mouse(MouseButton::Left)),
        )
}

/// Runs one frame: `edit` changes the input, the map reads it, then the frame's edges clear.
fn frame(map: &mut KeyMap<Act>, input: &mut RawInput, edit: impl FnOnce(&mut RawInput)) {
    edit(input);
    map.update(input, 0.5);
    input.end_frame();
}

fn temp_file(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("fk-app-keymap-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir.join("bindings.toml")
}

#[test]
fn physical_wasd_moves() {
    let mut map = KeyMap::new(defaults());
    let mut input = RawInput::default();
    // KeyW is the key labelled Z on AZERTY: one binding for both layouts.
    frame(&mut map, &mut input, |i| i.press(KeyCode::KeyW));
    assert_eq!(map.axis2(Act::Move), Vector2::new(0.0, 1.0));
    assert!(map.pressed(Act::Move));
    frame(&mut map, &mut input, |i| i.press(KeyCode::KeyD));
    let diagonal = map.axis2(Act::Move);
    assert!((diagonal.norm() - 1.0).abs() < 1e-12 && diagonal.x > 0.0 && diagonal.y > 0.0);
    frame(&mut map, &mut input, |i| i.release_all());
    assert_eq!(map.axis2(Act::Move), Vector2::zeros());
    assert!(map.released(Act::Move));
}

#[test]
fn buttons_report_hold_time() {
    let mut map = KeyMap::new(defaults());
    let mut input = RawInput::default();
    frame(&mut map, &mut input, |i| i.press(MouseButton::Left));
    assert!(map.pressed(Act::Inhale) && map.held(Act::Inhale));
    assert_eq!(map.held_for(Act::Inhale), 0.0);
    frame(&mut map, &mut input, |_| {});
    frame(&mut map, &mut input, |_| {});
    assert!(!map.pressed(Act::Inhale));
    assert_eq!(map.held_for(Act::Inhale), 1.0);
    frame(&mut map, &mut input, |i| i.release(MouseButton::Left));
    assert!(map.released(Act::Inhale) && !map.held(Act::Inhale));
    assert_eq!(map.released_after(Act::Inhale), Some(1.5));
    assert_eq!(map.held_for(Act::Inhale), 0.0);
    frame(&mut map, &mut input, |_| {});
    assert_eq!(map.released_after(Act::Inhale), None);
}

#[test]
fn a_tap_within_one_frame_presses_and_releases() {
    let mut map = KeyMap::new(defaults());
    let mut input = RawInput::default();
    frame(&mut map, &mut input, |i| {
        i.press(KeyCode::Space);
        i.release(KeyCode::Space);
    });
    assert!(map.pressed(Act::Jump) && map.released(Act::Jump) && !map.held(Act::Jump));
}

#[test]
fn a_trigger_taps_once_even_unbound() {
    let mut map = KeyMap::new(Bindings::new());
    let mut input = RawInput::default();
    map.trigger(Act::Jump);
    frame(&mut map, &mut input, |_| {});
    assert!(map.pressed(Act::Jump) && map.released(Act::Jump) && !map.held(Act::Jump));
    frame(&mut map, &mut input, |_| {});
    assert!(!map.pressed(Act::Jump));
}

#[test]
fn a_taken_keyboard_presses_nothing() {
    let mut map = KeyMap::new(defaults());
    let mut input = RawInput::default();
    frame(&mut map, &mut input, |i| {
        i.press(KeyCode::Space);
        i.add_typed(KeyCode::Space, Some(" "));
        i.take_keyboard();
    });
    assert!(!map.pressed(Act::Jump) && !map.held(Act::Jump));
    assert!(input.typed().is_empty() && input.typed_keys().is_empty());
    // Let go while taken, it is not released either.
    frame(&mut map, &mut input, |i| i.release(KeyCode::Space));
    assert!(!map.released(Act::Jump));
}

#[test]
fn analog_axes_read_mouse_and_wheel() {
    let mut map = KeyMap::new(defaults());
    let mut input = RawInput::default();
    frame(&mut map, &mut input, |i| {
        i.add_mouse_motion(Vector2::new(3.0, -2.0));
        i.add_wheel(Vector2::new(0.5, 2.0));
    });
    assert_eq!(map.axis2(Act::Look), Vector2::new(3.0, -2.0));
    assert_eq!(map.axis1(Act::Zoom), 2.0);
    frame(&mut map, &mut input, |_| {});
    assert!(map.released(Act::Look) && map.axis2(Act::Look) == Vector2::zeros());
}

#[test]
fn misfit_bindings_are_refused() {
    let mut bindings = defaults();
    let error = bindings
        .set(Act::Jump, vec![Binding::MouseMotion])
        .unwrap_err();
    assert_eq!(error.action, Act::Jump);
    assert_eq!(bindings.get(Act::Jump), defaults().get(Act::Jump));
}

#[test]
fn file_round_trips() {
    let bindings = defaults()
        .bind(
            Act::Zoom,
            Binding::ButtonPair {
                negative: key(KeyCode::Minus),
                positive: ButtonInput::Mouse(MouseButton::Other(7)),
            },
        )
        .bind(
            Act::Inhale,
            Binding::Button(ButtonInput::Mouse(MouseButton::Back)),
        );
    let text = config::write(&bindings);
    assert!(text.contains("Move = [{ up = \"KeyW\""), "{text}");
    assert_eq!(config::parse(&text, &Bindings::new()).unwrap(), bindings);
}

#[test]
fn file_overrides_only_what_it_lists_and_reads() {
    let text = r#"
        Jump = ["KeyE"]
        Inhale = ["NotAKey"]
        Look = ["Space"]
        Zoom = []
        Fly = ["KeyF"]
    "#;
    let bindings = config::parse(text, &defaults()).unwrap();
    assert_eq!(
        bindings.get(Act::Jump),
        &[Binding::Button(key(KeyCode::KeyE))]
    );
    assert_eq!(bindings.get(Act::Inhale), defaults().get(Act::Inhale));
    assert_eq!(bindings.get(Act::Look), defaults().get(Act::Look));
    assert_eq!(bindings.get(Act::Zoom), &[]);
    assert_eq!(bindings.get(Act::Move), defaults().get(Act::Move));
    assert!(config::parse("Jump = [", &defaults()).is_err());
}

#[test]
fn missing_file_is_created_with_defaults() {
    let path = temp_file("missing");
    let map = KeyMap::with_file(defaults(), &path);
    assert_eq!(map.bindings(), &defaults());
    let text = std::fs::read_to_string(&path).unwrap();
    assert_eq!(config::parse(&text, &Bindings::new()).unwrap(), defaults());
}

#[test]
fn invalid_file_is_left_alone() {
    let path = temp_file("invalid");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, "Jump = [").unwrap();
    let mut map = KeyMap::with_file(defaults(), &path);
    assert_eq!(map.bindings(), &defaults());
    map.save_if_dirty();
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "Jump = [");
}

#[test]
fn editing_the_file_rebinds() {
    let path = temp_file("edit");
    let mut map = KeyMap::with_file(defaults(), &path);
    // Make sure the new modification time differs on coarse-grained filesystems.
    std::thread::sleep(std::time::Duration::from_millis(20));
    std::fs::write(&path, "Jump = [\"KeyJ\"]\n").unwrap();
    map.poll_file(10.0);
    assert_eq!(
        map.bindings().get(Act::Jump),
        &[Binding::Button(key(KeyCode::KeyJ))]
    );
}

#[test]
fn changes_in_game_are_saved() {
    let path = temp_file("save");
    let mut map = KeyMap::with_file(defaults(), &path);
    map.set_bindings(Act::Jump, vec![Binding::Button(key(KeyCode::KeyK))])
        .unwrap();
    map.save_if_dirty();
    let reloaded = KeyMap::with_file(defaults(), &path);
    assert_eq!(
        reloaded.bindings().get(Act::Jump),
        &[Binding::Button(key(KeyCode::KeyK))]
    );
}

#[test]
fn rebind_captures_the_next_press_without_triggering_actions() {
    let mut map = KeyMap::new(defaults());
    let mut input = RawInput::default();
    map.listen(Act::Inhale, RebindTarget::Binding(0)).unwrap();
    frame(&mut map, &mut input, |i| i.press(KeyCode::KeyF));
    assert_eq!(map.rebind_state(), &Rebind::Idle);
    assert_eq!(
        map.bindings().get(Act::Inhale),
        &[Binding::Button(key(KeyCode::KeyF))]
    );
    // KeyF is still down, but the rebind swallowed it.
    frame(&mut map, &mut input, |_| {});
    assert!(!map.held(Act::Inhale));
    frame(&mut map, &mut input, |i| i.release(KeyCode::KeyF));
    frame(&mut map, &mut input, |i| i.press(KeyCode::KeyF));
    assert!(map.pressed(Act::Inhale));
}

#[test]
fn rebind_appends_and_replaces_parts() {
    let mut map = KeyMap::new(defaults());
    let mut input = RawInput::default();
    map.listen(Act::Jump, RebindTarget::Binding(1)).unwrap();
    frame(&mut map, &mut input, |i| i.press(KeyCode::KeyJ));
    assert_eq!(map.bindings().get(Act::Jump).len(), 2);

    let up = RebindTarget::Part {
        index: 0,
        part: Part::Up,
    };
    map.listen(Act::Move, up).unwrap();
    frame(&mut map, &mut input, |i| i.press(KeyCode::ArrowUp));
    let expected = wasd().with_part(Part::Up, key(KeyCode::ArrowUp)).unwrap();
    assert_eq!(map.bindings().get(Act::Move), &[expected]);
}

#[test]
fn rebind_waits_for_the_right_kind_of_input() {
    let mut map = KeyMap::new(defaults());
    let mut input = RawInput::default();
    map.listen(Act::Look, RebindTarget::Binding(0)).unwrap();
    frame(&mut map, &mut input, |i| i.press(KeyCode::KeyQ));
    frame(&mut map, &mut input, |i| {
        i.add_mouse_motion(Vector2::new(1.0, 0.0))
    });
    assert!(matches!(map.rebind_state(), Rebind::Listening { .. }));
    frame(&mut map, &mut input, |i| {
        i.add_wheel(Vector2::new(0.0, 1.0))
    });
    // Zoom already uses the wheel.
    assert!(matches!(
        map.rebind_state(),
        Rebind::Conflict { conflicts, .. } if conflicts == &[Act::Zoom]
    ));
}

#[test]
fn conflicts_are_kept_or_stolen() {
    let mut keep = KeyMap::new(defaults());
    let mut input = RawInput::default();
    keep.listen(Act::Jump, RebindTarget::Binding(0)).unwrap();
    frame(&mut keep, &mut input, |i| i.press(KeyCode::KeyW));
    let Rebind::Conflict { conflicts, .. } = keep.rebind_state() else {
        panic!("expected a conflict, got {:?}", keep.rebind_state());
    };
    assert_eq!(conflicts, &[Act::Move]);
    let mut steal = KeyMap::new(defaults());
    steal.rebind = keep.rebind.clone();

    keep.resolve(Resolution::Keep);
    assert_eq!(
        keep.bindings().get(Act::Jump),
        &[Binding::Button(key(KeyCode::KeyW))]
    );
    assert_eq!(keep.bindings().get(Act::Move), &[wasd()]);

    steal.resolve(Resolution::Steal);
    assert_eq!(
        steal.bindings().get(Act::Jump),
        &[Binding::Button(key(KeyCode::KeyW))]
    );
    assert_eq!(steal.bindings().get(Act::Move), &[]);
}

#[test]
fn cancel_and_reset() {
    let mut map = KeyMap::new(defaults());
    let mut input = RawInput::default();
    map.listen(Act::Jump, RebindTarget::Binding(0)).unwrap();
    map.cancel_rebind();
    frame(&mut map, &mut input, |i| i.press(KeyCode::KeyX));
    assert_eq!(map.bindings(), &defaults());

    map.set_bindings(Act::Jump, vec![]).unwrap();
    map.set_bindings(Act::Inhale, vec![]).unwrap();
    map.reset_action(Act::Jump);
    assert_eq!(map.bindings().get(Act::Jump), defaults().get(Act::Jump));
    assert_eq!(map.bindings().get(Act::Inhale), &[]);
    map.reset_to_defaults();
    assert_eq!(map.bindings(), &defaults());
}

#[test]
fn listen_checks_its_target() {
    let mut map = KeyMap::new(defaults());
    assert_eq!(
        map.listen(Act::Jump, RebindTarget::Binding(2)),
        Err(RebindError::NoSuchBinding { index: 2, len: 1 })
    );
    let part = |part| RebindTarget::Part { index: 0, part };
    assert_eq!(
        map.listen(Act::Jump, part(Part::Up)),
        Err(RebindError::NoSuchPart(Part::Up))
    );
    assert_eq!(
        map.listen(Act::Move, part(Part::Negative)),
        Err(RebindError::NoSuchPart(Part::Negative))
    );
    assert!(map.listen(Act::Move, part(Part::Left)).is_ok());
}

#[test]
fn bindings_have_short_names_for_players() {
    use crate::{KeyCode, MouseButton};
    let quad = Binding::ButtonQuad {
        up: KeyCode::KeyW.into(),
        down: KeyCode::KeyS.into(),
        left: KeyCode::KeyA.into(),
        right: KeyCode::KeyD.into(),
    };
    assert_eq!(quad.to_string(), "W/A/S/D");
    let pair = Binding::ButtonPair {
        negative: KeyCode::Comma.into(),
        positive: KeyCode::Period.into(),
    };
    assert_eq!(pair.to_string(), ",/.");
    assert_eq!(Binding::Button(KeyCode::Digit1.into()).to_string(), "1");
    assert_eq!(Binding::Button(KeyCode::F12.into()).to_string(), "F12");
    assert_eq!(
        Binding::Button(KeyCode::ShiftLeft.into()).to_string(),
        "Shift"
    );
    let left = Binding::Button(MouseButton::Left.into());
    assert_eq!(left.to_string(), "Mouse left");
}
