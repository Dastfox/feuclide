//! Flying through the regular tilings of a curved space and through closed manifolds: the
//! shared code of `fly-h3`, `fly-s3`, `seifert-weber` and `lens-space`, generic over the
//! geometry.
//!
//! The scene is one tiling's cells, each drawn as the frame of its edges, coloured by how many
//! steps it is from the origin's. Whenever the eye leaves the origin's cell it is carried back
//! by the step that leads to the cell it is in, which is a symmetry of the tiling: the view
//! does not change, and the flight never runs out of cells. All motion is twist integration
//! and every placement goes through the geometry traits.

// Each example includes this module and uses only its tilings or its manifolds.
#![allow(dead_code)]

use std::f64::consts::FRAC_PI_2;

use bevy_ecs::prelude::*;
use fk_app::keymap::{Action, ActionKind, Binding, Bindings, KeyMap, KeyMapPlugin};
use fk_app::{App, ButtonInput, CursorGrab, FixedUpdate, KeyCode, Startup, Update, WindowConfig};
use fk_assets::{Cell, Tiling};
use fk_geometry::{Geometry, GpuGeometry, GroupElement};
use fk_math::Real;
use fk_math::nalgebra::Vector3;
use fk_quotient::{ChartTag, Quotient};
use fk_render::{
    Color, Instance, Lighting, MeshHandle, MeshInstances, Meshes, PostMode, PostSettings,
    QuotientView, RenderPlugin, Sky, shapes, tangent,
};
use fk_scene::{Camera, Pose, ScenePlugin, SceneSystems, Velocity};

use crate::key_list;

const LOOK_SENSITIVITY: Real = 0.0025;

/// What one flight shows.
pub struct Setup {
    /// The window's title and the key list's.
    pub name: &'static str,
    /// The tilings N steps through: a cell and how many round an edge.
    pub tilings: &'static [(Cell, u32)],
    /// How far the camera sees.
    pub far: Real,
    /// Cells drawn: those within this distance of the origin's, at most `most` of them.
    pub within: Real,
    /// The most cells drawn.
    pub most: usize,
    /// Speed in units per second, and with Shift.
    pub speeds: [Real; 2],
    /// Fog per unit of distance.
    pub fog: Real,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Fly {
    Move,
    Rise,
    Look,
    Fast,
    Next,
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
        Self::Next,
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
            Self::Next => "Next",
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
            Self::Next => "the next tiling (in a tiling)",
            Self::ShowDistance => "depth as distance bands, or the image",
            Self::ReleaseCursor => "let go of the cursor",
            Self::Help => "this list",
        }
    }
}

fn bindings() -> Bindings<Fly> {
    let key = ButtonInput::Key;
    let quad = |up, down, left, right| Binding::ButtonQuad {
        up: key(up),
        down: key(down),
        left: key(left),
        right: key(right),
    };
    Bindings::new()
        .bind(
            Fly::Move,
            quad(KeyCode::KeyW, KeyCode::KeyS, KeyCode::KeyA, KeyCode::KeyD),
        )
        .bind(
            Fly::Move,
            quad(
                KeyCode::ArrowUp,
                KeyCode::ArrowDown,
                KeyCode::ArrowLeft,
                KeyCode::ArrowRight,
            ),
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
        .bind(Fly::Next, Binding::Button(key(KeyCode::KeyN)))
        .bind(Fly::ShowDistance, Binding::Button(key(KeyCode::Tab)))
        .bind(Fly::ReleaseCursor, Binding::Button(key(KeyCode::Escape)))
        .bind(Fly::Help, Binding::Button(key(KeyCode::F1)))
}

/// The tilings, built: each one's frame mesh and its cells.
#[derive(Resource)]
struct Tilings<G: Geometry> {
    built: Vec<(Tiling<G>, MeshHandle, Vec<G::Isometry>)>,
    shown: usize,
}

/// The entity drawing the tiling shown.
#[derive(Component)]
struct Shown;

/// The rig: turns about its up, carries the velocity.
#[derive(Component)]
struct Rig {
    eye: Entity,
}

/// The eye's pitch, in radians.
#[derive(Component, Default)]
struct Pitch(Real);

#[derive(Resource, Clone, Copy)]
struct Speeds([Real; 2]);

/// The app every flight shares: the window, the scene and its renderer, the keys, a dusk sky
/// the fog fades into, looking and flying.
fn app<G: GpuGeometry>(name: &str, fog: Real, speeds: [Real; 2]) -> App {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                tracing_subscriber::EnvFilter::new("info,wgpu_hal=error,wgpu_core=warn")
            }),
        )
        .init();
    let lighting = Lighting {
        sun: Vector3::new(0.4, 0.8, 0.45),
        fog_density: fog,
        sky: Sky {
            zenith: Color::hex(0x1d2440),
            horizon: Color::hex(0x2c3558),
            below: Color::hex(0x161b30),
            stars: 0.0,
            ..Sky::default()
        },
        ..Lighting::default()
    };
    let mut app = App::new();
    app.set_window(WindowConfig {
        title: format!("Feuclide · {name}"),
        ..Default::default()
    })
    .add_plugin(ScenePlugin::<G>::default())
    .add_plugin(RenderPlugin::<G>::default())
    .add_plugin(KeyMapPlugin::new(bindings()))
    .insert_resource(lighting)
    .insert_resource(Speeds(speeds))
    .insert_resource(PostSettings::default())
    .add_systems(Update, (look::<G>, toggles))
    .add_systems(FixedUpdate, fly::<G>.before(SceneSystems::Integrate));
    key_list::add::<Fly>(&mut app, "fly");
    app
}

fn launch(app: App) {
    if let Err(error) = app.run() {
        tracing::error!("{error}");
        std::process::exit(1);
    }
}

/// Runs the flight through tilings.
pub fn run<G: GpuGeometry>(setup: Setup) {
    let mut app = app::<G>(setup.name, setup.fog, setup.speeds);
    app.add_systems(
        Startup,
        move |commands: Commands, meshes: ResMut<Meshes>| spawn::<G>(&setup, commands, meshes),
    )
    .add_systems(Update, next_tiling::<G>)
    .add_systems(FixedUpdate, recentre::<G>.after(SceneSystems::Integrate));
    launch(app);
}

/// A thing inside a closed manifold's fundamental domain: a ball, or a cube, of a colour, at
/// a place given by its reference-frame coordinates in the tangent space at the origin.
#[derive(Clone, Copy, Debug)]
pub struct Thing {
    /// Where.
    pub at: [Real; 3],
    /// Its radius, or half its side.
    pub size: Real,
    /// A cube, or a ball.
    pub cube: bool,
    /// Its colour.
    pub color: u32,
}

/// A closed manifold to fly through.
pub struct Manifold<G: Geometry> {
    /// The window's title.
    pub name: &'static str,
    /// The quotient.
    pub quotient: Quotient<G>,
    /// The cell drawn as the fundamental domain's frame, if it is one.
    pub cell: Option<Tiling<G>>,
    /// What is in the domain.
    pub things: Vec<Thing>,
    /// How far the camera sees.
    pub far: Real,
    /// How far the copies of the domain are drawn out to.
    pub horizon: Real,
    /// Speed in units per second, and with Shift.
    pub speeds: [Real; 2],
    /// Fog per unit of distance.
    pub fog: Real,
}

/// The quotient the rig is reduced into.
#[derive(Resource)]
struct Folded<G: Geometry>(Quotient<G>);

/// Runs the flight through a closed manifold: the domain's things drawn once, and again at
/// every deck translate within the horizon (`QuotientView`); the eye carried back across a
/// face whenever it crosses one (`Quotient::reduce`), so that it flies on for ever without
/// ever leaving the domain.
pub fn run_manifold<G: GpuGeometry>(manifold: Manifold<G>) {
    let mut app = app::<G>(manifold.name, manifold.fog, manifold.speeds);
    let view = QuotientView::new(manifold.quotient.clone(), manifold.horizon);
    tracing::info!("{} copies of the domain drawn", view.translates().len() + 1);
    app.insert_resource(view)
        .insert_resource(Folded(manifold.quotient.clone()))
        .add_systems(
            Startup,
            move |mut commands: Commands, mut meshes: ResMut<Meshes>| {
                if let Some(cell) = &manifold.cell {
                    let frame = cell.frame(0.04 * cell.inradius(), 0.0, 0.97).build::<G>();
                    commands.spawn(MeshInstances::<G>::single(
                        meshes.add(frame),
                        Color::hex(0xe0b04f),
                    ));
                }
                for thing in &manifold.things {
                    let mesh = if thing.cube {
                        let half = Vector3::repeat(thing.size);
                        shapes::cuboid(half).build::<G>()
                    } else {
                        shapes::sphere(thing.size, 3).build::<G>()
                    };
                    let place =
                        G::Isometry::exp(&G::transvection(&tangent::<G>(&Vector3::from(thing.at))));
                    commands.spawn((
                        Pose::<G>(place),
                        MeshInstances::<G>::single(meshes.add(mesh), Color::hex(thing.color)),
                    ));
                }
                spawn_rig::<G>(&mut commands, manifold.far);
            },
        )
        .add_systems(FixedUpdate, fold::<G>.after(SceneSystems::Integrate));
    launch(app);
}

/// Across a face of the domain, the rig is carried back across it.
fn fold<G: Geometry>(folded: Res<Folded<G>>, mut rigs: Query<&mut Pose<G>, With<Rig>>) {
    for mut pose in &mut rigs {
        let mut chart = ChartTag::identity();
        if !folded.0.reduce(&mut pose.0, &mut chart).is_empty() {
            pose.0.renormalize();
        }
    }
}

fn spawn_rig<G: Geometry>(commands: &mut Commands, far: Real) {
    let eye = commands
        .spawn((
            Pose::<G>::identity(),
            Pitch::default(),
            Camera::<G>::new(70.0_f64.to_radians(), 0.02, far),
        ))
        .id();
    commands
        .spawn((Rig { eye }, Pose::<G>::identity(), Velocity::<G>::default()))
        .add_child(eye);
}

fn forward() -> Vector3<Real> {
    -Vector3::z()
}

fn rotation<G: Geometry>(from: Vector3<Real>, to: Vector3<Real>, angle: Real) -> G::Isometry {
    G::Isometry::exp(&(G::rotation(&tangent::<G>(&from), &tangent::<G>(&to)) * angle))
}

/// The cells of the tiling at `index`, coloured by their distance from the origin's.
fn instances<G: Geometry>(cells: &[G::Isometry], inradius: Real) -> Vec<Instance<G>> {
    let palette = [0xf4f0e8, 0xe0b04f, 0xd9774b, 0x8fae8b, 0x6f8fb5, 0xd98c9a].map(Color::hex);
    let origin = G::origin();
    cells
        .iter()
        .map(|cell| {
            let d = G::distance(&origin, &G::apply(cell, &origin));
            let ring = (d / (2.0 * inradius)).round() as usize;
            Instance::new(*cell, palette[ring % palette.len()])
        })
        .collect()
}

fn spawn<G: GpuGeometry>(setup: &Setup, mut commands: Commands, mut meshes: ResMut<Meshes>) {
    let built: Vec<_> = setup
        .tilings
        .iter()
        .filter_map(|&(cell, around)| {
            let tiling = Tiling::<G>::regular(cell, around)?;
            let frame = meshes.add(
                tiling
                    .frame(0.04 * tiling.inradius(), 0.05 * tiling.inradius(), 0.97)
                    .build::<G>(),
            );
            let cells = tiling.cells(setup.within, setup.most);
            tracing::info!(
                "{cell:?} {around} round an edge: {} cells, inradius {:.3}",
                cells.len(),
                tiling.inradius()
            );
            Some((tiling, frame, cells))
        })
        .collect();
    let (tiling, frame, cells) = &built[0];
    commands.spawn((
        Shown,
        MeshInstances::<G> {
            mesh: *frame,
            instances: instances::<G>(cells, tiling.inradius()),
        },
    ));
    commands.insert_resource(Tilings { built, shown: 0 });
    spawn_rig::<G>(&mut commands, setup.far);
}

fn look<G: Geometry>(
    map: Res<KeyMap<Fly>>,
    grab: Res<CursorGrab>,
    mut rigs: Query<(&Rig, &mut Pose<G>), Without<Pitch>>,
    mut eyes: Query<(&mut Pitch, &mut Pose<G>)>,
) {
    if !grab.is_grabbed() {
        return;
    }
    let delta = map.axis2(Fly::Look) * LOOK_SENSITIVITY;
    for (rig, mut pose) in &mut rigs {
        pose.0 = pose
            .0
            .compose(&rotation::<G>(forward(), Vector3::x(), delta.x));
        pose.0.renormalize();
        if let Ok((mut pitch, mut eye)) = eyes.get_mut(rig.eye) {
            let limit = FRAC_PI_2 - 0.01;
            pitch.0 = (pitch.0 - delta.y).clamp(-limit, limit);
            eye.0 = rotation::<G>(forward(), Vector3::y(), pitch.0);
        }
    }
}

fn fly<G: Geometry>(
    map: Res<KeyMap<Fly>>,
    speeds: Res<Speeds>,
    mut rigs: Query<(&Rig, &mut Velocity<G>)>,
    eyes: Query<&Pose<G>, With<Pitch>>,
) {
    let speed = speeds.0[usize::from(map.held(Fly::Fast))];
    let walk = map.axis2(Fly::Move);
    let rise = map.axis1(Fly::Rise);
    for (rig, mut velocity) in &mut rigs {
        let pitch = eyes
            .get(rig.eye)
            .map_or_else(|_| G::Isometry::identity(), |pose| pose.0);
        let looking = G::apply_tangent(
            &pitch,
            &tangent::<G>(&(Vector3::x() * walk.x + forward() * walk.y)),
        );
        let up = tangent::<G>(&(Vector3::y() * rise));
        velocity.0 = G::transvection(&((looking + up) * speed));
    }
}

/// Out of the origin's cell, the rig is carried back by the step to the cell it is in.
fn recentre<G: Geometry>(tilings: Res<Tilings<G>>, mut rigs: Query<&mut Pose<G>, With<Rig>>) {
    let (tiling, ..) = &tilings.built[tilings.shown];
    let origin = G::origin();
    for mut pose in &mut rigs {
        let at = G::apply(&pose.0, &origin);
        if G::distance(&origin, &at) <= tiling.inradius() {
            continue;
        }
        let nearest = tiling
            .neighbours()
            .iter()
            .min_by(|a, b| {
                let d = |g: &G::Isometry| G::distance(&G::apply(g, &origin), &at);
                d(a).total_cmp(&d(b))
            })
            .copied();
        if let Some(step) = nearest {
            let d = |g: &G::Isometry| G::distance(&G::apply(g, &origin), &at);
            if d(&step) < G::distance(&origin, &at) {
                pose.0 = step.inverse().compose(&pose.0);
                pose.0.renormalize();
            }
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

fn next_tiling<G: Geometry>(
    map: Res<KeyMap<Fly>>,
    mut tilings: ResMut<Tilings<G>>,
    mut shown: Query<&mut MeshInstances<G>, With<Shown>>,
) {
    if map.pressed(Fly::Next) {
        tilings.shown = (tilings.shown + 1) % tilings.built.len();
        let (tiling, frame, cells) = &tilings.built[tilings.shown];
        tracing::info!("{:?}, {} cells", tiling.cell(), cells.len());
        for mut drawn in &mut shown {
            drawn.mesh = *frame;
            drawn.instances = instances::<G>(cells, tiling.inradius());
        }
    }
}
