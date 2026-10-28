//! Parallel worlds joined by portals, four deep.
//!
//! Three realms stand at the same coordinates without ever meeting: a meadow, a room and a
//! corridor. A door on the meadow opens into the room (the orange pillar right behind the
//! door is never seen through it), a window beside it looks into the room from the side and
//! lets nothing through, a door at the back of the room opens onto the corridor, and the
//! corridor's far end comes out at its own start, so that it goes on for ever: from the meadow,
//! the corridor is seen through the room through the door four times over (the budget's
//! depth). Each realm has its own sky and light; the room's lamp lights nothing elsewhere.
//!
//! `cargo run --example portals --release` from `engine/`. Click to grab the mouse, WASD
//! (ZQSD on AZERTY) to fly where you look, Space / Ctrl to rise and sink, Shift to go fast,
//! Tab for the depth as distance, F1 for every key. `PORTALS_EYE="x,y,z,yaw,realm"` (yaw in
//! degrees, realm 0 the meadow, 1 the room, 2 the corridor) starts there, for captures.

use std::f64::consts::FRAC_PI_2;

use bevy_ecs::prelude::*;
use fk_app::keymap::{Action, ActionKind, Binding, Bindings, KeyMap, KeyMapPlugin};
use fk_app::{App, ButtonInput, CursorGrab, FixedUpdate, KeyCode, Startup, Update, WindowConfig};
use fk_geometry::{Geometry, GroupElement};
use fk_geometry_euclidean::E3;
use fk_math::Real;
use fk_math::nalgebra::Vector3;
use fk_render::{
    Color, Lighting, MeshInstances, Meshes, PointLight, PortalViews, PostMode, PostSettings,
    RealmLighting, RenderPlugin, Sky, shapes,
};
use fk_scene::{Camera, Portal, Portals, Pose, Realm, Realms, ScenePlugin, SceneSystems, Velocity};
use fk_ui::key_list;

type Isometry = <E3 as Geometry>::Isometry;

const LOOK_SENSITIVITY: Real = 0.0025;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Fly {
    Move,
    Rise,
    Look,
    Fast,
    ShowDistance,
    ReleaseCursor,
    Help,
}

impl Action for Fly {
    const ALL: &'static [Self] = &[
        Self::Move,
        Self::Rise,
        Self::Look,
        Self::Fast,
        Self::ShowDistance,
        Self::ReleaseCursor,
        Self::Help,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::Move => "Move",
            Self::Rise => "Rise",
            Self::Look => "Look",
            Self::Fast => "Fast",
            Self::ShowDistance => "ShowDistance",
            Self::ReleaseCursor => "ReleaseCursor",
            Self::Help => "Help",
        }
    }

    fn kind(self) -> ActionKind {
        match self {
            Self::Move | Self::Look => ActionKind::Axis2,
            Self::Rise => ActionKind::Axis1,
            _ => ActionKind::Button,
        }
    }
}

impl key_list::Listed for Fly {
    const HELP: Self = Self::Help;

    fn describe(self) -> &'static str {
        match self {
            Self::Move => "fly where you look",
            Self::Rise => "sink, rise",
            Self::Look => "look",
            Self::Fast => "go fast (held)",
            Self::ShowDistance => "depth as distance bands, or the image",
            Self::ReleaseCursor => "let go of the cursor",
            Self::Help => "this list",
        }
    }
}

fn bindings() -> Bindings<Fly> {
    let key = ButtonInput::Key;
    Bindings::new()
        .bind(
            Fly::Move,
            Binding::ButtonQuad {
                up: key(KeyCode::KeyW),
                down: key(KeyCode::KeyS),
                left: key(KeyCode::KeyA),
                right: key(KeyCode::KeyD),
            },
        )
        .bind(
            Fly::Rise,
            Binding::ButtonPair {
                negative: key(KeyCode::ControlLeft),
                positive: key(KeyCode::Space),
            },
        )
        .bind(Fly::Look, Binding::MouseMotion)
        .bind(Fly::Fast, Binding::Button(key(KeyCode::ShiftLeft)))
        .bind(Fly::ShowDistance, Binding::Button(key(KeyCode::Tab)))
        .bind(Fly::ReleaseCursor, Binding::Button(key(KeyCode::Escape)))
        .bind(Fly::Help, Binding::Button(key(KeyCode::F1)))
}

/// The rig: turns about its up, carries the velocity, goes through the portals.
#[derive(Component)]
struct Rig {
    eye: Entity,
    /// Its pose before the last step.
    last: Isometry,
}

/// The eye's pitch, in radians.
#[derive(Component, Default)]
struct Pitch(Real);

fn at(x: Real, y: Real, z: Real) -> Isometry {
    Isometry::exp(&E3::transvection(&Vector3::new(x, y, z)))
}

/// A turn of `angle` radians about the vertical, from −z towards +x.
fn yaw(angle: Real) -> Isometry {
    Isometry::exp(&(E3::rotation(&-Vector3::z(), &Vector3::x()) * angle))
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                tracing_subscriber::EnvFilter::new("info,wgpu_hal=error,wgpu_core=warn")
            }),
        )
        .init();
    let mut app = App::new();
    app.set_window(WindowConfig {
        title: "Feuclide · portals".to_owned(),
        ..Default::default()
    })
    .add_plugin(ScenePlugin::<E3>::default())
    .add_plugin(RenderPlugin::<E3>::default())
    .add_plugin(KeyMapPlugin::new(bindings()))
    .insert_resource(Lighting {
        sun: Vector3::new(0.4, 0.8, 0.45),
        fog_density: 0.01,
        ..Lighting::default()
    })
    .insert_resource(PostSettings::default())
    .add_systems(Startup, spawn)
    .add_systems(Update, (look, toggles, report))
    .add_systems(FixedUpdate, fly.before(SceneSystems::Integrate))
    .add_systems(FixedUpdate, through.after(SceneSystems::Integrate));
    key_list::add::<Fly>(&mut app, "portals");
    if let Err(error) = app.run() {
        tracing::error!("{error}");
        std::process::exit(1);
    }
}

fn spawn(mut commands: Commands, mut meshes: ResMut<Meshes>, mut realms: ResMut<Realms>) {
    let room = realms.add("room");
    let corridor = realms.add("corridor");
    let mut boxed = |commands: &mut Commands,
                     realm: Realm,
                     [x0, y0, z0]: [Real; 3],
                     [x1, y1, z1]: [Real; 3],
                     color: u32| {
        let half = Vector3::new(x1 - x0, y1 - y0, z1 - z0) / 2.0;
        let mesh = meshes.add(shapes::cuboid(half).build::<E3>());
        commands.spawn((
            Pose::<E3>(at((x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0)),
            MeshInstances::<E3>::single(mesh, Color::hex(color)),
            realm,
        ));
    };

    // The meadow: its ground, the door's frame, the window's, the pillar right behind the
    // door and a red block in front.
    let main = Realm::MAIN;
    boxed(
        &mut commands,
        main,
        [-40.0, -0.1, -40.0],
        [40.0, 0.0, 40.0],
        0x6f8f5a,
    );
    for x in [-0.6, 0.5] {
        boxed(
            &mut commands,
            main,
            [x, 0.0, -0.1],
            [x + 0.1, 2.3, 0.1],
            0x5a4632,
        );
    }
    boxed(
        &mut commands,
        main,
        [-0.6, 2.2, -0.1],
        [0.6, 2.3, 0.1],
        0x5a4632,
    );
    boxed(
        &mut commands,
        main,
        [-6.1, 0.0, -0.1],
        [-3.9, 0.8, 0.1],
        0x8a8a8a,
    );
    boxed(
        &mut commands,
        main,
        [-6.1, 2.2, -0.1],
        [-3.9, 2.4, 0.1],
        0x8a8a8a,
    );
    for x in [-6.1, -4.0] {
        boxed(
            &mut commands,
            main,
            [x, 0.8, -0.1],
            [x + 0.1, 2.2, 0.1],
            0x8a8a8a,
        );
    }
    boxed(
        &mut commands,
        main,
        [-0.4, 0.0, -3.4],
        [0.4, 3.0, -2.6],
        0xe07b2a,
    );
    boxed(
        &mut commands,
        main,
        [1.5, 0.0, 1.5],
        [2.5, 1.0, 2.5],
        0xb83b3b,
    );

    // The room, at the same coordinates: six by six by three, a doorway in its front wall
    // (z = 3), a window in its left (x = −3), a doorway in its back (z = −3).
    let wall = 0xd9c6a5;
    boxed(
        &mut commands,
        room,
        [-3.2, -0.2, -3.2],
        [3.2, 0.0, 3.2],
        0x8b5a3c,
    );
    boxed(
        &mut commands,
        room,
        [-3.2, 3.0, -3.2],
        [3.2, 3.2, 3.2],
        0xeee6d8,
    );
    for z in [3.0, -3.2] {
        boxed(
            &mut commands,
            room,
            [-3.2, 0.0, z],
            [-0.5, 3.0, z + 0.2],
            wall,
        );
        boxed(
            &mut commands,
            room,
            [0.5, 0.0, z],
            [3.2, 3.0, z + 0.2],
            wall,
        );
        boxed(
            &mut commands,
            room,
            [-0.5, 2.2, z],
            [0.5, 3.0, z + 0.2],
            wall,
        );
    }
    boxed(&mut commands, room, [3.0, 0.0, -3.0], [3.2, 3.0, 3.0], wall);
    boxed(
        &mut commands,
        room,
        [-3.2, 0.0, -3.0],
        [-3.0, 0.8, 3.0],
        wall,
    );
    boxed(
        &mut commands,
        room,
        [-3.2, 2.2, -3.0],
        [-3.0, 3.0, 3.0],
        wall,
    );
    boxed(
        &mut commands,
        room,
        [-3.2, 0.8, -3.0],
        [-3.0, 2.2, -1.0],
        wall,
    );
    boxed(
        &mut commands,
        room,
        [-3.2, 0.8, 1.0],
        [-3.0, 2.2, 3.0],
        wall,
    );
    boxed(
        &mut commands,
        room,
        [1.6, 0.0, -1.0],
        [2.4, 0.8, 1.0],
        0x3b6fb8,
    );
    boxed(
        &mut commands,
        room,
        [-1.0, 0.0, -0.4],
        [-0.2, 1.4, 0.4],
        0x46a36b,
    );
    commands.spawn((
        Pose::<E3>(at(0.0, 2.6, 0.0)),
        PointLight {
            color: Color::hex(0xffc27a),
            intensity: 4.0,
            range: 8.0,
        },
        room,
    ));

    // The corridor, at the same coordinates again: from z = 0 to z = −10, its far end coming
    // out at its start.
    boxed(
        &mut commands,
        corridor,
        [-1.4, -0.2, -10.2],
        [1.4, 0.0, 0.2],
        0x4a4a52,
    );
    boxed(
        &mut commands,
        corridor,
        [-1.4, 2.6, -10.2],
        [1.4, 2.8, 0.2],
        0x2f2f36,
    );
    for x in [-1.4, 1.2] {
        boxed(
            &mut commands,
            corridor,
            [x, 0.0, -10.2],
            [x + 0.2, 2.6, 0.2],
            0x7a6f9b,
        );
    }
    for z in [-2.0, -6.0] {
        commands.spawn((
            Pose::<E3>(at(0.0, 2.3, z)),
            PointLight {
                color: Color::hex(0xb8c8ff),
                intensity: 3.0,
                range: 7.0,
            },
            corridor,
        ));
    }
    for k in 0..5 {
        let z = -1.0 - 2.0 * Real::from(k);
        let color = [0xe0b04f, 0xd9774b, 0x8fae8b, 0x6f8fb5, 0xd98c9a][k as usize];
        boxed(
            &mut commands,
            corridor,
            [1.0, 0.0, z - 0.2],
            [1.2, 1.8, z + 0.2],
            color,
        );
    }

    // The portals: openings 1 wide and 2.2 high from the ground, the window 2 by 1.4.
    let door = [0.5, 1.1];
    commands.spawn(Portal::<E3>::between(
        main,
        at(0.0, 1.1, 0.0),
        room,
        at(0.0, 1.1, 3.0),
        door,
    ));
    // The window faces +z on the meadow and comes out in the room's left wall, its −z turned
    // into the room (+x).
    commands.spawn(
        Portal::<E3>::between(
            main,
            at(-5.0, 1.5, 0.0),
            room,
            at(-3.0, 1.5, 0.0).compose(&yaw(FRAC_PI_2)),
            [1.0, 0.7],
        )
        .window(),
    );
    commands.spawn(Portal::<E3>::between(
        room,
        at(0.0, 1.1, -3.0),
        corridor,
        at(0.0, 1.1, 0.0),
        door,
    ));
    commands.spawn(
        Portal::<E3>::between(
            corridor,
            at(0.0, 1.1, -10.0),
            corridor,
            at(0.0, 1.1, 0.0),
            [1.2, 1.3],
        )
        .one_way(),
    );

    // Each realm its own sky and light: the meadow the frame's, the room warm and dim, the
    // corridor cold.
    let mut lit = RealmLighting::default();
    lit.0.insert(
        room,
        Lighting {
            sun: Vector3::new(-0.3, 0.9, 0.2),
            sun_color: Color::hex(0x8a7a66),
            shadow_tint: Color::hex(0x3a3040),
            fog_density: 0.02,
            sky: Sky {
                zenith: Color::hex(0x20180f),
                horizon: Color::hex(0x2a2015),
                below: Color::hex(0x120c08),
                stars: 0.0,
                ..Sky::default()
            },
            ..Lighting::default()
        },
    );
    lit.0.insert(
        corridor,
        Lighting {
            sun: Vector3::new(0.2, 1.0, 0.1),
            sun_color: Color::hex(0x9fb4d9),
            shadow_tint: Color::hex(0x2a2f45),
            fog_density: 0.06,
            sky: Sky {
                zenith: Color::hex(0x10131f),
                horizon: Color::hex(0x1b2033),
                below: Color::hex(0x0a0c14),
                stars: 0.0,
                ..Sky::default()
            },
            ..Lighting::default()
        },
    );
    commands.insert_resource(lit);

    // The eye: six metres in front of the door, or where PORTALS_EYE says.
    let (start, realm) = start(room, corridor);
    let eye = commands
        .spawn((
            Pose::<E3>::identity(),
            Pitch::default(),
            Camera::<E3>::new(70.0_f64.to_radians(), 0.02, 200.0),
        ))
        .id();
    commands
        .spawn((
            Rig { eye, last: start },
            Pose::<E3>(start),
            Velocity::<E3>::default(),
            realm,
        ))
        .add_child(eye);
}

/// Where the eye starts: `PORTALS_EYE="x,y,z,yaw,realm"`, or 6 m in front of the door.
fn start(room: Realm, corridor: Realm) -> (Isometry, Realm) {
    let given = std::env::var("PORTALS_EYE").ok().and_then(|text| {
        let n: Vec<Real> = text
            .split(',')
            .filter_map(|t| t.trim().parse().ok())
            .collect();
        (n.len() == 5).then_some(n)
    });
    match given {
        Some(n) => {
            let realm = match n[4] as u32 {
                1 => room,
                2 => corridor,
                _ => Realm::MAIN,
            };
            (at(n[0], n[1], n[2]).compose(&yaw(n[3].to_radians())), realm)
        }
        None => (at(0.3, 1.6, 6.0), Realm::MAIN),
    }
}

fn rotation(from: Vector3<Real>, to: Vector3<Real>, angle: Real) -> Isometry {
    Isometry::exp(&(E3::rotation(&from, &to) * angle))
}

fn look(
    map: Res<KeyMap<Fly>>,
    grab: Res<CursorGrab>,
    mut rigs: Query<(&Rig, &mut Pose<E3>), Without<Pitch>>,
    mut eyes: Query<(&mut Pitch, &mut Pose<E3>)>,
) {
    if !grab.is_grabbed() {
        return;
    }
    let delta = map.axis2(Fly::Look) * LOOK_SENSITIVITY;
    for (rig, mut pose) in &mut rigs {
        pose.0 = pose
            .0
            .compose(&rotation(-Vector3::z(), Vector3::x(), delta.x));
        pose.0.renormalize();
        if let Ok((mut pitch, mut eye)) = eyes.get_mut(rig.eye) {
            let limit = FRAC_PI_2 - 0.01;
            pitch.0 = (pitch.0 - delta.y).clamp(-limit, limit);
            eye.0 = rotation(-Vector3::z(), Vector3::y(), pitch.0);
        }
    }
}

fn fly(
    map: Res<KeyMap<Fly>>,
    mut rigs: Query<(&mut Rig, &Pose<E3>, &mut Velocity<E3>)>,
    eyes: Query<&Pose<E3>, With<Pitch>>,
) {
    let speed = if map.held(Fly::Fast) { 6.0 } else { 2.0 };
    let walk = map.axis2(Fly::Move);
    let rise = map.axis1(Fly::Rise);
    for (mut rig, pose, mut velocity) in &mut rigs {
        rig.last = pose.0;
        let pitch = eyes
            .get(rig.eye)
            .map_or_else(|_| Isometry::identity(), |pose| pose.0);
        let looking = E3::apply_tangent(&pitch, &(Vector3::x() * walk.x - Vector3::z() * walk.y));
        velocity.0 = E3::transvection(&((looking + Vector3::y() * rise) * speed));
    }
}

/// After each step, through whatever opening the eye went through: its pose carried, its realm
/// the other side's.
fn through(portals: Portals<E3>, mut rigs: Query<(&Rig, &mut Pose<E3>, &mut Realm)>) {
    let origin = E3::origin();
    for (rig, mut pose, mut realm) in &mut rigs {
        let (from, to) = (E3::apply(&rig.last, &origin), E3::apply(&pose.0, &origin));
        if let Some(crossing) = portals.cross(*realm, &from, &to) {
            pose.0 = crossing.carry.compose(&pose.0);
            pose.0.renormalize();
            *realm = crossing.realm;
        }
    }
}

fn toggles(map: Res<KeyMap<Fly>>, mut post: ResMut<PostSettings>, mut grab: ResMut<CursorGrab>) {
    if map.pressed(Fly::ShowDistance) {
        post.mode = match post.mode {
            PostMode::Image => PostMode::Distance,
            PostMode::Distance => PostMode::Image,
        };
    }
    if map.pressed(Fly::ReleaseCursor) {
        grab.release();
    }
}

/// Logs the views seen through the portals whenever how many there are changes.
fn report(views: Option<Res<PortalViews<E3>>>, realms: Res<Realms>, mut last: Local<usize>) {
    let Some(views) = views else { return };
    if views.0.len() != *last {
        *last = views.0.len();
        let seen: Vec<String> = views
            .0
            .iter()
            .map(|view| {
                format!(
                    "{} ({})",
                    realms.name(view.realm).unwrap_or("?"),
                    view.depth
                )
            })
            .collect();
        tracing::info!("{} views: {}", views.0.len(), seen.join(", "));
    }
}
