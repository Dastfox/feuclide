//! ECS components and schedules: poses as isometries relative to their parent, hierarchy
//! propagation by group composition, twist integration and cameras.
//!
//! Generic over `fk_geometry::Geometry`; never names a concrete geometry. Add a
//! [`ScenePlugin<G>`] for the app's geometry, then spawn entities with a [`Pose`]:
//!
//! - [`Pose`] is relative to the parent ([`ChildOf`](bevy_ecs::hierarchy::ChildOf), the
//!   `bevy_ecs` hierarchy). [`GlobalPose`] is the composition down the hierarchy,
//!   `world(child) = world(parent) ∘ local(child)`, with the [`ChartTag`] word composed alongside.
//! - [`Velocity`] is a body-frame twist; every fixed step moves the pose by
//!   `pose ← pose ∘ exp(dt · ξ)` and renormalizes it.
//! - [`Camera`] marks the eye. Every entity gets its pose relative to it, [`ViewRelative`],
//!   computed in `f64` so that only small numbers reach the GPU.
//! - [`FloatingOrigin`], when inserted, moves the whole scene back to the eye whenever it has
//!   gone too far from the origin, so that the coordinates stay small (open H³ worlds).

mod camera;
mod chart;
mod origin;
mod pose;
mod propagate;

use std::marker::PhantomData;

use bevy_ecs::schedule::{IntoScheduleConfigs, SystemSet};
use fk_app::{App, FixedUpdate, Plugin, PostUpdate};
use fk_geometry::Geometry;

pub use camera::{ActiveView, Camera, ViewRelative, update_view_relative};
pub use chart::{ChartTag, Letter};
pub use origin::{FloatingOrigin, rebase};
pub use pose::{GlobalPose, Pose, Velocity, integrate_velocities};
pub use propagate::propagate_poses;

/// The scene's systems, by what they do.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SceneSystems {
    /// Moves poses by their velocities, in [`FixedUpdate`].
    Integrate,
    /// Moves the roots back to the eye when a [`FloatingOrigin`] says so, in [`PostUpdate`]
    /// before [`Propagate`](Self::Propagate).
    Rebase,
    /// Writes [`GlobalPose`]s, in [`PostUpdate`].
    Propagate,
    /// Finds the [`ActiveView`] and writes [`ViewRelative`] poses, in [`PostUpdate`] after
    /// [`Propagate`](Self::Propagate).
    CameraRelative,
}

/// Poses, velocities, the hierarchy and cameras for the geometry `G`.
///
/// Sets the app's geometry to `G`.
pub struct ScenePlugin<G>(PhantomData<fn() -> G>);

impl<G> Default for ScenePlugin<G> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<G: Geometry> Plugin for ScenePlugin<G> {
    fn build(&self, app: &mut App) {
        app.set_geometry::<G>()
            .add_systems(
                FixedUpdate,
                integrate_velocities::<G>.in_set(SceneSystems::Integrate),
            )
            .add_systems(
                PostUpdate,
                (
                    rebase::<G>.in_set(SceneSystems::Rebase),
                    propagate_poses::<G>.in_set(SceneSystems::Propagate),
                    update_view_relative::<G>.in_set(SceneSystems::CameraRelative),
                )
                    .chain(),
            );
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::FRAC_PI_2;

    use bevy_ecs::prelude::*;
    use fk_app::Time;
    use fk_geometry::{Geometry, GroupElement};
    use fk_geometry_euclidean::{E3, Se3, Se3Twist};
    use fk_math::nalgebra::{Point3, Vector3};

    use super::*;

    fn app() -> App {
        let mut app = App::new();
        app.add_plugin(ScenePlugin::<E3>::default());
        app
    }

    fn translation(x: f64, y: f64, z: f64) -> Se3 {
        Se3::exp(&E3::transvection(&Vector3::new(x, y, z)))
    }

    fn quarter_turn_about_y() -> Se3 {
        Se3::exp(&(E3::rotation(&Vector3::z(), &Vector3::x()) * FRAC_PI_2))
    }

    fn point(app: &App, entity: Entity) -> Point3<f64> {
        app.world().get::<GlobalPose<E3>>(entity).unwrap().point()
    }

    fn close(a: Point3<f64>, b: Point3<f64>) -> bool {
        (a - b).norm() < 1e-9
    }

    #[test]
    fn children_compose_with_their_parents() {
        let mut app = app();
        let world = app.world_mut();
        let root = world
            .spawn(Pose::<E3>(
                translation(1.0, 0.0, 0.0).compose(&quarter_turn_about_y()),
            ))
            .id();
        let child = world
            .spawn((Pose::<E3>(translation(0.0, 0.0, 2.0)), ChildOf(root)))
            .id();
        let grandchild = world
            .spawn((Pose::<E3>(translation(0.0, 3.0, 0.0)), ChildOf(child)))
            .id();
        app.update(0.0);
        // The quarter turn about y carries +z to +x.
        assert!(close(point(&app, child), Point3::new(3.0, 0.0, 0.0)));
        assert!(close(point(&app, grandchild), Point3::new(3.0, 3.0, 0.0)));
    }

    #[test]
    fn charts_compose_down_the_hierarchy() {
        let mut app = app();
        let (a, b) = (Letter::new(0), Letter::new(1));
        let world = app.world_mut();
        let root = world
            .spawn((Pose::<E3>::identity(), ChartTag::from_letters([a])))
            .id();
        let child = world
            .spawn((
                Pose::<E3>::identity(),
                ChartTag::from_letters([a.inverse(), b]),
                ChildOf(root),
            ))
            .id();
        let grandchild = world.spawn((Pose::<E3>::identity(), ChildOf(child))).id();
        app.update(0.0);
        let chart = &app.world().get::<GlobalPose<E3>>(grandchild).unwrap().chart;
        assert_eq!(chart.letters(), &[b]);
    }

    #[test]
    fn velocity_integrates_in_the_body_frame() {
        let mut app = app();
        let body = app
            .world_mut()
            .spawn((
                Pose::<E3>(quarter_turn_about_y()),
                Velocity::<E3>(Se3Twist {
                    angular: Vector3::zeros(),
                    linear: Vector3::new(0.0, 0.0, 1.0),
                }),
            ))
            .id();
        // One second of fixed steps; body +z is root +x.
        for _ in 0..60 {
            app.update(Time::DEFAULT_FIXED_STEP);
        }
        let travelled = point(&app, body);
        assert!(close(travelled, Point3::new(1.0, 0.0, 0.0)), "{travelled}");
    }

    #[test]
    fn poses_are_made_relative_to_the_camera() {
        let mut app = app();
        let world = app.world_mut();
        let eye = translation(0.0, 1.0, 5.0).compose(&quarter_turn_about_y());
        world.spawn((Pose::<E3>(eye), Camera::<E3>::new(1.0, 0.1, 100.0)));
        let thing = world.spawn(Pose::<E3>(translation(0.0, 1.0, 2.0))).id();
        app.update(0.0);
        assert!(app.world().get_resource::<ActiveView<E3>>().is_some());
        let relative = app.world().get::<ViewRelative<E3>>(thing).unwrap().0;
        let seen = E3::apply(&relative, &E3::origin());
        // 3 units along root −z, which the eye's quarter turn makes its +x: to the right.
        assert!(close(seen, Point3::new(3.0, 0.0, 0.0)), "{seen}");
    }

    /// The eye flies twenty units straight on in H³ with a beacon flying the same way one unit
    /// ahead of it: how far from one unit ahead the beacon is seen at the end.
    fn beacon_error(floating: bool) -> f64 {
        use fk_geometry_hyperbolic::{H3, Lorentz3};
        let forward = H3::origin_frame(2) * -1.0;
        let mut app = App::new();
        app.add_plugin(ScenePlugin::<H3>::default());
        if floating {
            app.insert_resource(FloatingOrigin::<H3>::new(4.0));
        }
        let ahead = Lorentz3::exp(&H3::transvection(&forward));
        let world = app.world_mut();
        world.spawn((
            Pose::<H3>::identity(),
            Velocity::<H3>(H3::transvection(&forward)),
            Camera::<H3>::new(1.0, 0.1, 8.0),
        ));
        let beacon = world
            .spawn((
                Pose::<H3>(ahead),
                Velocity::<H3>(H3::transvection(&forward)),
            ))
            .id();
        for _ in 0..1200 {
            app.update(Time::DEFAULT_FIXED_STEP);
        }
        let seen = app.world().get::<ViewRelative<H3>>(beacon).unwrap().0;
        let point = H3::apply(&seen, &H3::origin());
        // A pose gone to NaN is at distance 0 from anything: lost all the same.
        if H3::point_residual(&point).is_nan() {
            return f64::INFINITY;
        }
        H3::distance(&point, &H3::apply(&ahead, &H3::origin()))
    }

    #[test]
    fn far_out_in_h3_the_floating_origin_keeps_what_is_near_the_eye_where_it_is() {
        let floating = beacon_error(true);
        let fixed = beacon_error(false);
        assert!(floating < 1e-6, "with the floating origin: {floating}");
        // Without it, the Lorentz renormalization fails before twenty units: it squares
        // coordinates of the size of cosh d against the −1 it keeps.
        assert!(fixed > 1e-3, "without, it should be lost: {fixed}");
    }
}
