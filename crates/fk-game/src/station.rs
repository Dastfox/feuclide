//! One station of the tour: a space, the camera's path through it, and the hand-over to the
//! next station. Generic over the geometry.

use bevy_ecs::prelude::*;
use fk_app::keymap::{Action, ActionKind, Binding, Bindings, KeyMap, KeyMapPlugin};
use fk_app::{
    App, AppExit, ButtonInput, FixedUpdate, Handover, KeyCode, Startup, Time, Update, WindowSize,
};
use fk_assets::Tiling;
use fk_geometry::{Geometry, GpuGeometry, GroupElement};
use fk_math::Real;
use fk_math::nalgebra::{Vector2, Vector3};
use fk_quotient::{ChartTag, Quotient};
use fk_render::{
    CarriedImage, Color, Instance, Lighting, MeshInstances, Meshes, Mosh, MoshKind, PostSettings,
    QuotientView, RenderPlugin, SharedDevice, Sky, TextPanel, shapes, tangent,
};
use fk_scene::{Camera, Pose, ScenePlugin, SceneSystems};

/// How long the camera rides through a station before the next one takes over, in seconds.
const STAY: Real = 30.0;
/// How long the image moshes before the next station takes the window.
const LEAVE: Real = 1.0;
/// How long the next station takes to mosh in through the last image.
const ARRIVE: Real = 3.0;
/// The mosh's blocks, as a share of the image height.
const BLOCK: Real = 0.025;
/// How long the station's name stays in the corner.
const TITLE: Real = 7.0;
/// The longest frame the transition counts, so that a stall does not skip it.
const LONGEST_FRAME: Real = 1.0 / 20.0;

/// What a station shows.
pub struct Station<G: Geometry> {
    /// Its name, in the corner as it opens.
    pub name: &'static str,
    /// A line under the name.
    pub about: &'static str,
    /// The space.
    pub space: Space<G>,
    /// The camera's path.
    pub rails: Rails,
    /// The light, the sky and how far the camera sees.
    pub look: Look,
}

/// A space to ride through.
pub enum Space<G: Geometry> {
    /// A regular tiling, its cells drawn as frames of their edges and coloured by their ring
    /// round the origin's cell. The eye is carried back into the origin's cell by a symmetry
    /// of the tiling whenever it leaves it, so the ride never runs out of cells.
    Tiling {
        /// The tiling.
        tiling: Tiling<G>,
        /// Cells drawn: those within this distance of the origin's...
        within: Real,
        /// ...and at most this many.
        most: usize,
    },
    /// A closed manifold: the domain's things drawn once and again at every deck translate
    /// within the horizon, and the eye folded back across a face whenever it crosses one.
    Manifold {
        /// The manifold, as a quotient of the space.
        quotient: Quotient<G>,
        /// The cell drawn as the domain's frame, if the domain is one.
        cell: Option<Tiling<G>>,
        /// What is in the domain.
        things: Vec<Thing>,
        /// How far the domain's copies are drawn out to.
        horizon: Real,
    },
}

/// A thing in a closed manifold's domain.
#[derive(Clone, Copy, Debug)]
pub struct Thing {
    /// Where, in the reference frame's coordinates in the tangent space at the origin.
    pub at: [Real; 3],
    /// Its radius, or half its side.
    pub size: Real,
    /// A cube, or a ball.
    pub cube: bool,
    /// Its colour.
    pub color: u32,
}

/// The camera's path: on at a steady speed, turning gently from side to side and up and
/// down, as two slow swings of the body-frame twist.
#[derive(Resource, Clone, Copy, Debug)]
pub struct Rails {
    /// Units per second, straight ahead.
    pub speed: Real,
    /// The yaw's swing: radians per second at most, and its period in seconds.
    pub yaw: (Real, Real),
    /// The pitch's swing, the same way.
    pub pitch: (Real, Real),
}

impl Rails {
    /// The body-frame twist at `t` seconds into the ride.
    fn twist<G: Geometry>(&self, t: Real) -> fk_geometry::Algebra<G> {
        use std::f64::consts::TAU;
        let ahead = -Vector3::z();
        let swing =
            |(rate, period): (Real, Real), phase: Real| rate * (TAU * t / period + phase).sin();
        G::transvection(&tangent::<G>(&(ahead * self.speed)))
            + G::rotation(&tangent::<G>(&ahead), &tangent::<G>(&Vector3::x()))
                * swing(self.yaw, 0.0)
            + G::rotation(&tangent::<G>(&ahead), &tangent::<G>(&Vector3::y()))
                * swing(self.pitch, 1.3)
    }
}

/// A station's light and sky.
#[derive(Clone, Copy, Debug)]
pub struct Look {
    /// How far the camera sees.
    pub far: Real,
    /// Fog per unit of distance.
    pub fog: Real,
    /// The sky overhead, at the horizon and below, as `0xrrggbb`.
    pub sky: [u32; 3],
}

/// Where the tour is: this station's number, and how many there are.
#[derive(Resource, Clone, Copy, Debug)]
pub struct Tour {
    /// This station.
    pub index: usize,
    /// Builds the station with a number.
    pub build: fn(usize) -> App,
    /// How many stations there are.
    pub count: usize,
}

/// Where the ride is in the station.
#[derive(Resource, Debug, Default)]
struct Ride {
    /// Seconds of the camera's path.
    t: Real,
}

/// Where the hand-over is.
#[derive(Resource, Debug, Default)]
struct Passage {
    /// Seconds since the station took the window.
    since: Real,
    /// Moshing in through the last station's image, how far along; `None` once in, or on the
    /// first station.
    arriving: Option<Real>,
    /// Moshing out, seconds since it began.
    leaving: Option<Real>,
    /// The window has been handed on.
    gone: bool,
    /// Skip to the next station as soon as it is ready.
    skip: bool,
}

/// Put into the next station's world when it takes the window: it moshes in.
#[derive(Resource)]
struct Arrived;

/// The eye.
#[derive(Component)]
struct Eye;

/// The tiling the eye is carried back into.
#[derive(Resource)]
struct Recentred<G: Geometry>(Tiling<G>);

/// The quotient the eye is folded back into.
#[derive(Resource)]
struct Folded<G: Geometry>(Quotient<G>);

/// The tour's keys: there is no steering.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Keys {
    Next,
    Quit,
}

impl Action for Keys {
    const ALL: &'static [Self] = &[Self::Next, Self::Quit];

    fn name(self) -> &'static str {
        match self {
            Self::Next => "Next",
            Self::Quit => "Quit",
        }
    }

    fn kind(self) -> ActionKind {
        ActionKind::Button
    }
}

fn bindings() -> Bindings<Keys> {
    let key = |code| Binding::Button(ButtonInput::Key(code));
    Bindings::new()
        .bind(Keys::Next, key(KeyCode::KeyN))
        .bind(Keys::Next, key(KeyCode::ArrowRight))
        .bind(Keys::Quit, key(KeyCode::Escape))
}

/// The station's app, its number in the tour `tour`.
pub fn app<G: GpuGeometry>(station: Station<G>, tour: Tour) -> App {
    let [zenith, horizon, below] = station.look.sky.map(Color::hex);
    let lighting = Lighting {
        sun: Vector3::new(0.4, 0.8, 0.45),
        fog_density: station.look.fog,
        sky: Sky {
            zenith,
            horizon,
            below,
            stars: 0.0,
            ..Sky::default()
        },
        ..Lighting::default()
    };
    let mut app = App::new();
    app.add_plugin(ScenePlugin::<G>::default())
        .add_plugin(RenderPlugin::<G>::default())
        .add_plugin(KeyMapPlugin::new(bindings()))
        .insert_resource(lighting)
        .insert_resource(station.rails)
        .insert_resource(tour)
        .init_resource::<Ride>()
        .init_resource::<Passage>()
        .add_systems(Update, (keys, pass))
        .add_systems(FixedUpdate, ride::<G>.before(SceneSystems::Integrate));
    let (name, about, far) = (station.name, station.about, station.look.far);
    match station.space {
        Space::Tiling {
            tiling,
            within,
            most,
        } => {
            let cells = tiling.cells(within, most);
            tracing::info!("{name}: {} cells", cells.len());
            let inradius = tiling.inradius();
            let frame = tiling.frame(0.04 * inradius, 0.05 * inradius, 0.97);
            let instances = rings::<G>(&cells, inradius);
            app.insert_resource(Recentred(tiling))
                .add_systems(
                    Startup,
                    move |mut commands: Commands, mut meshes: ResMut<Meshes>| {
                        commands.spawn(MeshInstances::<G> {
                            mesh: meshes.add(frame.build::<G>()),
                            instances: instances.clone(),
                        });
                        spawn_eye::<G>(&mut commands, far);
                    },
                )
                .add_systems(FixedUpdate, recentre::<G>.after(ride::<G>));
        }
        Space::Manifold {
            quotient,
            cell,
            things,
            horizon,
        } => {
            let view = QuotientView::new(quotient.clone(), horizon);
            tracing::info!(
                "{name}: {} copies of the domain",
                view.translates().len() + 1
            );
            app.insert_resource(view)
                .insert_resource(Folded(quotient))
                .add_systems(
                    Startup,
                    move |mut commands: Commands, mut meshes: ResMut<Meshes>| {
                        if let Some(cell) = &cell {
                            let frame = cell.frame(0.04 * cell.inradius(), 0.0, 0.97).build::<G>();
                            commands.spawn(MeshInstances::<G>::single(
                                meshes.add(frame),
                                Color::hex(0xe0b04f),
                            ));
                        }
                        for thing in &things {
                            let mesh = if thing.cube {
                                shapes::cuboid(Vector3::repeat(thing.size)).build::<G>()
                            } else {
                                shapes::sphere(thing.size, 3).build::<G>()
                            };
                            let at = tangent::<G>(&Vector3::from(thing.at));
                            commands.spawn((
                                Pose::<G>(G::Isometry::exp(&G::transvection(&at))),
                                MeshInstances::<G>::single(
                                    meshes.add(mesh),
                                    Color::hex(thing.color),
                                ),
                            ));
                        }
                        spawn_eye::<G>(&mut commands, far);
                    },
                )
                .add_systems(FixedUpdate, fold::<G>.after(ride::<G>));
        }
    }
    app.add_systems(
        Update,
        move |passage: Res<Passage>, mut post: ResMut<PostSettings>| {
            title(&passage, &mut post, name, about);
        },
    );
    app
}

fn spawn_eye<G: Geometry>(commands: &mut Commands, far: Real) {
    commands.spawn((
        Eye,
        Pose::<G>::identity(),
        Camera::<G>::new(70.0_f64.to_radians(), 0.02, far),
    ));
}

/// The cells, coloured by their ring round the origin's.
fn rings<G: Geometry>(cells: &[G::Isometry], inradius: Real) -> Vec<Instance<G>> {
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

/// The camera along its path: `pose ← pose ∘ exp(dt · ξ(t))`.
fn ride<G: Geometry>(
    time: Res<Time>,
    rails: Res<Rails>,
    mut ride: ResMut<Ride>,
    mut eyes: Query<&mut Pose<G>, With<Eye>>,
) {
    let dt = time.fixed_step();
    let twist = rails.twist::<G>(ride.t);
    ride.t += dt;
    for mut pose in &mut eyes {
        pose.0 = pose.0.compose(&G::Isometry::exp(&(twist * dt)));
        pose.0.renormalize();
    }
}

/// Out of the origin's cell, the eye is carried back by the step to the cell it is in.
fn recentre<G: Geometry>(tiling: Res<Recentred<G>>, mut eyes: Query<&mut Pose<G>, With<Eye>>) {
    let origin = G::origin();
    for mut pose in &mut eyes {
        let at = G::apply(&pose.0, &origin);
        let here = G::distance(&origin, &at);
        if here <= tiling.0.inradius() {
            continue;
        }
        let d = |g: &G::Isometry| G::distance(&G::apply(g, &origin), &at);
        let nearest = tiling
            .0
            .neighbours()
            .iter()
            .min_by(|a, b| d(a).total_cmp(&d(b)))
            .copied();
        if let Some(step) = nearest.filter(|step| d(step) < here) {
            pose.0 = step.inverse().compose(&pose.0);
            pose.0.renormalize();
        }
    }
}

/// Across a face of the domain, the eye is carried back across it.
fn fold<G: Geometry>(folded: Res<Folded<G>>, mut eyes: Query<&mut Pose<G>, With<Eye>>) {
    for mut pose in &mut eyes {
        if !folded
            .0
            .reduce(&mut pose.0, &mut ChartTag::identity())
            .is_empty()
        {
            pose.0.renormalize();
        }
    }
}

fn keys(map: Res<KeyMap<Keys>>, mut passage: ResMut<Passage>, mut exit: ResMut<AppExit>) {
    if map.pressed(Keys::Quit) {
        exit.request();
    }
    if map.pressed(Keys::Next) {
        passage.skip = true;
    }
}

/// The hand-over: the next station prepared once this one is drawing, the image moshed and
/// carried at the end of the stay, and this station moshed in if it came after another.
#[allow(clippy::too_many_arguments)]
fn pass(
    time: Res<Time>,
    tour: Res<Tour>,
    arrived: Option<Res<Arrived>>,
    size: Option<Res<WindowSize>>,
    device: Option<Res<SharedDevice>>,
    mut passage: ResMut<Passage>,
    mut post: ResMut<PostSettings>,
    mut carried: ResMut<CarriedImage>,
    mut handover: ResMut<Handover>,
    mut commands: Commands,
) {
    if passage.gone {
        return;
    }
    let dt = time.delta().min(LONGEST_FRAME);
    passage.since += dt;
    let start_mosh = |post: &mut PostSettings| {
        post.mosh = Mosh {
            kind: MoshKind::Blocks,
            start: post.mosh.start.wrapping_add(1),
            progress: 0.0,
            block: BLOCK,
            ..post.mosh
        };
    };
    // Arriving: mosh in through the last station's image.
    if arrived.is_some() {
        commands.remove_resource::<Arrived>();
        start_mosh(&mut post);
        passage.arriving = Some(0.0);
    } else if let Some(since) = passage.arriving {
        let since = since + dt;
        post.mosh.progress = (since / ARRIVE).min(1.0);
        passage.arriving = (since < ARRIVE).then_some(since);
    }
    // The next station is built while this one plays, on this one's device, once it has one.
    if !handover.is_prepared()
        && let Some(device) = device.as_deref()
    {
        let next = (tour.index + 1) % tour.count;
        let (build, size, device) = (tour.build, size.as_deref().copied(), device.clone());
        tracing::info!("preparing station {next}");
        handover.prepare(move || {
            let mut app = build(next);
            let world = app.world_mut();
            if let Some(size) = size {
                world.insert_resource(size);
            }
            world.insert_resource(device);
            app
        });
    }
    // Leaving: the image moshes for a moment, is read back, and goes with the window.
    match passage.leaving {
        None => {
            let due = passage.since >= STAY - LEAVE || passage.skip;
            if due && handover.is_ready() && passage.arriving.is_none() {
                start_mosh(&mut post);
                passage.leaving = Some(0.0);
            }
        }
        Some(since) => {
            let since = since + dt;
            passage.leaving = Some(since);
            if since >= LEAVE {
                carried.request();
                handover.go(carry);
                passage.gone = true;
            }
        }
    }
}

/// What the station left gives the next: its last image, to mosh in through.
fn carry(old: &mut World, new: &mut World) {
    if let Some(image) = old.resource_mut::<CarriedImage>().take() {
        new.resource_mut::<CarriedImage>().hold(image);
        new.insert_resource(Arrived);
    }
}

/// The station's name in the corner while it opens.
fn title(passage: &Passage, post: &mut PostSettings, name: &str, about: &str) {
    if passage.since < TITLE && passage.leaving.is_none() {
        post.text = TextPanel {
            lines: vec![name.to_owned(), about.to_owned()],
            origin: Vector2::new(24.0, 24.0),
            scale: 2,
            ..TextPanel::default()
        };
    } else {
        post.text.lines.clear();
    }
}
