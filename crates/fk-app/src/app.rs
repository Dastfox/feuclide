use std::any::{TypeId, type_name};
use std::time::Instant;

use bevy_ecs::resource::Resource;
use bevy_ecs::schedule::{IntoScheduleConfigs, Schedule, ScheduleLabel, Schedules};
use bevy_ecs::system::ScheduleSystem;
use bevy_ecs::world::{FromWorld, World};
use fk_geometry::Geometry;
use fk_math::Real;

use crate::input::{CursorGrab, RawInput};
use crate::profile::FrameProfile;
use crate::schedule::{FixedUpdate, PostUpdate, PreUpdate, Render, Startup, Update};
use crate::time::Time;
use crate::window::{AppExit, Handover, WindowConfig, WindowMode};

/// A piece of an app: resources and systems added together.
pub trait Plugin: 'static {
    /// Adds the plugin's resources and systems to `app`.
    fn build(&self, app: &mut App);
}

/// The geometry an app was built for, as a resource.
///
/// Plugins generic over a geometry check it with [`ActiveGeometry::is`] so that an app cannot mix
/// two of them.
#[derive(Resource, Clone, Copy, Debug)]
pub struct ActiveGeometry {
    type_id: TypeId,
    name: &'static str,
}

impl ActiveGeometry {
    fn of<G: Geometry>() -> Self {
        Self {
            type_id: TypeId::of::<G>(),
            name: type_name::<G>(),
        }
    }

    /// Whether this is the geometry `G`.
    pub fn is<G: Geometry>(&self) -> bool {
        self.type_id == TypeId::of::<G>()
    }

    /// The geometry's type name.
    pub fn name(&self) -> &'static str {
        self.name
    }
}

/// An application: a `bevy_ecs` world, its schedules and the window it will open.
///
/// Build it with plugins and systems, then call [`App::run`]. [`App::update`] runs one frame
/// without a window, for tests and tools.
pub struct App {
    world: World,
    window: WindowConfig,
    started: bool,
}

impl App {
    /// An app with empty schedules and the [`Time`], [`RawInput`], [`CursorGrab`], [`AppExit`],
    /// [`Handover`], [`WindowMode`] and [`FrameProfile`] resources.
    pub fn new() -> Self {
        let mut world = World::new();
        let mut schedules = Schedules::new();
        schedules.insert(Schedule::new(Startup));
        schedules.insert(Schedule::new(PreUpdate));
        schedules.insert(Schedule::new(FixedUpdate));
        schedules.insert(Schedule::new(Update));
        schedules.insert(Schedule::new(PostUpdate));
        schedules.insert(Schedule::new(Render));
        world.insert_resource(schedules);
        world.init_resource::<Time>();
        world.init_resource::<RawInput>();
        world.init_resource::<CursorGrab>();
        world.init_resource::<AppExit>();
        world.init_resource::<Handover>();
        world.init_resource::<WindowMode>();
        world.init_resource::<FrameProfile>();
        Self {
            world,
            window: WindowConfig::default(),
            started: false,
        }
    }

    /// Sets how the window opens, and its [`WindowMode`] to match.
    pub fn set_window(&mut self, window: WindowConfig) -> &mut Self {
        self.world.insert_resource(WindowMode {
            fullscreen: window.fullscreen,
        });
        self.window = window;
        self
    }

    /// Builds the app for the geometry `G`, inserting [`ActiveGeometry`].
    ///
    /// # Panics
    ///
    /// If the app was already built for a different geometry.
    pub fn set_geometry<G: Geometry>(&mut self) -> &mut Self {
        if let Some(active) = self.geometry() {
            assert!(
                active.is::<G>(),
                "app is built for {}, cannot switch to {}",
                active.name(),
                type_name::<G>()
            );
        }
        self.world.insert_resource(ActiveGeometry::of::<G>());
        self
    }

    /// The geometry the app is built for, if one was set.
    pub fn geometry(&self) -> Option<ActiveGeometry> {
        self.world.get_resource::<ActiveGeometry>().copied()
    }

    /// Adds a plugin.
    pub fn add_plugin(&mut self, plugin: impl Plugin) -> &mut Self {
        plugin.build(self);
        self
    }

    /// Adds systems to the schedule `label`.
    pub fn add_systems<M>(
        &mut self,
        label: impl ScheduleLabel,
        systems: impl IntoScheduleConfigs<ScheduleSystem, M>,
    ) -> &mut Self {
        self.world
            .resource_mut::<Schedules>()
            .entry(label)
            .add_systems(systems);
        self
    }

    /// Inserts a resource, replacing any previous one of the same type.
    pub fn insert_resource(&mut self, resource: impl Resource) -> &mut Self {
        self.world.insert_resource(resource);
        self
    }

    /// Inserts a resource built from the world, unless one of that type is already there.
    pub fn init_resource<R: Resource + FromWorld>(&mut self) -> &mut Self {
        self.world.init_resource::<R>();
        self
    }

    /// The world.
    pub fn world(&self) -> &World {
        &self.world
    }

    /// The world, mutably.
    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }

    /// Runs one frame that lasted `delta` seconds, running [`Startup`] first if it has not run.
    pub fn update(&mut self, delta: Real) {
        self.start();
        let world = &mut self.world;
        let mut profile = FrameProfile::default();
        let mut clock = Instant::now();
        let mut lap = || {
            let now = Instant::now();
            let took = now.duration_since(clock).as_secs_f64();
            clock = now;
            took
        };
        world.resource_mut::<Time>().advance(delta);
        world.run_schedule(PreUpdate);
        profile.pre_update = lap();
        while world.resource_mut::<Time>().take_fixed_step() {
            world.run_schedule(FixedUpdate);
            profile.fixed_steps += 1;
        }
        profile.fixed_update = lap();
        world.run_schedule(Update);
        profile.update = lap();
        world.run_schedule(PostUpdate);
        profile.post_update = lap();
        world.run_schedule(Render);
        profile.render = lap();
        world.resource_mut::<RawInput>().end_frame();
        world.insert_resource(profile);
        world.clear_trackers();
    }

    /// Opens the window and runs until it is closed.
    pub fn run(self) -> Result<(), crate::RunError> {
        crate::runner::run(self)
    }

    pub(crate) fn window_config(&self) -> &WindowConfig {
        &self.window
    }

    /// Starts the app without its window and runs [`Render`] once, for a renderer to get
    /// ready: what a [`Handover`] does on its own thread.
    pub(crate) fn prepare(&mut self) {
        self.start();
        self.world.run_schedule(Render);
    }

    pub(crate) fn start(&mut self) {
        if !self.started {
            self.started = true;
            self.world.run_schedule(Startup);
        }
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use bevy_ecs::prelude::*;
    use fk_geometry_euclidean::{E2, E3};

    use super::*;

    #[derive(Resource, Default)]
    struct Trace(Vec<&'static str>);

    fn record(name: &'static str) -> impl FnMut(ResMut<Trace>) {
        move |mut trace: ResMut<Trace>| trace.0.push(name)
    }

    #[test]
    fn frame_runs_schedules_in_order() {
        let mut app = App::new();
        app.init_resource::<Trace>()
            .add_systems(Startup, record("startup"))
            .add_systems(PreUpdate, record("pre"))
            .add_systems(FixedUpdate, record("fixed"))
            .add_systems(Update, record("update"))
            .add_systems(PostUpdate, record("post"))
            .add_systems(Render, record("render"));
        app.update(2.5 * Time::DEFAULT_FIXED_STEP);
        app.update(0.0);
        let trace = &app.world().resource::<Trace>().0;
        assert_eq!(
            trace,
            &[
                "startup", "pre", "fixed", "fixed", "update", "post", "render", //
                "pre", "update", "post", "render",
            ]
        );
    }

    #[test]
    fn the_profile_is_of_the_last_frame() {
        let mut app = App::new();
        app.update(2.5 * Time::DEFAULT_FIXED_STEP);
        let profile = *app.world().resource::<FrameProfile>();
        assert_eq!(profile.fixed_steps, 2);
        assert!(profile.total() >= 0.0);
        app.update(0.0);
        assert_eq!(app.world().resource::<FrameProfile>().fixed_steps, 0);
    }

    #[test]
    fn geometry_is_chosen_once() {
        let mut app = App::new();
        assert!(app.geometry().is_none());
        app.set_geometry::<E3>().set_geometry::<E3>();
        assert!(app.geometry().unwrap().is::<E3>());
        assert!(!app.geometry().unwrap().is::<E2>());
    }

    #[test]
    #[should_panic(expected = "cannot switch")]
    fn geometry_cannot_change() {
        App::new().set_geometry::<E3>().set_geometry::<E2>();
    }
}
