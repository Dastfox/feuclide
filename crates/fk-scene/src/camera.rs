use std::marker::PhantomData;

use bevy_ecs::prelude::*;
use fk_geometry::{Geometry, GroupElement};
use fk_math::Real;

use crate::GlobalPose;

/// A point of view: the entity's frame is the eye.
///
/// The eye looks along `−origin_frame(2)`, with `origin_frame(1)` up and `origin_frame(0)` to
/// the right. Angles are measured at the eye and distances along geodesics, so the same camera
/// means the same thing in every geometry.
#[derive(Component, Debug)]
pub struct Camera<G: Geometry> {
    /// Vertical field of view, in radians: the angle at the eye between the top and bottom
    /// edges of the image. Games animate it (the Fractal's breath strokes narrow it).
    pub fov_y: Real,
    /// Geodesic distance of the near clipping surface.
    pub near: Real,
    /// Geodesic distance of the far clipping surface, beyond which nothing is drawn.
    pub far: Real,
    /// Whether this camera is drawn from. With several active cameras, the first one found is.
    pub active: bool,
    _geometry: PhantomData<fn() -> G>,
}

impl<G: Geometry> Camera<G> {
    /// An active camera.
    ///
    /// # Panics
    ///
    /// If `fov_y` is not in `(0, π)` or `0 < near < far` does not hold.
    pub fn new(fov_y: Real, near: Real, far: Real) -> Self {
        assert!(
            fov_y > 0.0 && fov_y < std::f64::consts::PI,
            "field of view must be in (0, π), got {fov_y}"
        );
        assert!(
            0.0 < near && near < far,
            "need 0 < near < far, got {near} and {far}"
        );
        Self {
            fov_y,
            near,
            far,
            active: true,
            _geometry: PhantomData,
        }
    }

    /// The horizontal field of view of an image `aspect` times as wide as it is high.
    pub fn fov_x(&self, aspect: Real) -> Real {
        2.0 * ((self.fov_y / 2.0).tan() * aspect).atan()
    }
}

impl<G: Geometry> Clone for Camera<G> {
    fn clone(&self) -> Self {
        Self {
            _geometry: PhantomData,
            ..*self
        }
    }
}

/// The camera drawn from this frame, as a resource: which entity, its settings and where its
/// eye is. Written by [`SceneSystems::CameraRelative`](crate::SceneSystems::CameraRelative);
/// absent while no camera is active.
#[derive(Resource, Debug)]
pub struct ActiveView<G: Geometry> {
    /// The camera entity.
    pub entity: Entity,
    /// Its settings.
    pub camera: Camera<G>,
    /// Its global pose.
    pub eye: GlobalPose<G>,
}

impl<G: Geometry> Clone for ActiveView<G> {
    fn clone(&self) -> Self {
        Self {
            entity: self.entity,
            camera: self.camera.clone(),
            eye: self.eye.clone(),
        }
    }
}

/// An entity's pose relative to the active camera's eye, `view⁻¹ ∘ global`, computed in `f64`.
///
/// This is what reaches the GPU: near the eye it is small, so converting it to `f32` loses
/// nothing that matters, however far the scene is from the root. Written for every entity with
/// a [`Pose`](crate::Pose) (which requires it) by
/// [`SceneSystems::CameraRelative`](crate::SceneSystems::CameraRelative).
#[derive(Component, Debug)]
pub struct ViewRelative<G: Geometry>(pub G::Isometry);

impl<G: Geometry> Default for ViewRelative<G> {
    fn default() -> Self {
        Self(G::Isometry::identity())
    }
}

impl<G: Geometry> Clone for ViewRelative<G> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<G: Geometry> Copy for ViewRelative<G> {}

/// Finds the active camera and writes every entity's [`ViewRelative`] pose.
pub fn update_view_relative<G: Geometry>(
    mut commands: Commands,
    cameras: Query<(Entity, &Camera<G>, &GlobalPose<G>)>,
    mut relative: Query<(&GlobalPose<G>, &mut ViewRelative<G>)>,
) {
    let Some((entity, camera, eye)) = cameras.iter().find(|(_, camera, _)| camera.active) else {
        commands.remove_resource::<ActiveView<G>>();
        return;
    };
    let view_inverse = eye.isometry.inverse();
    commands.insert_resource(ActiveView {
        entity,
        camera: camera.clone(),
        eye: eye.clone(),
    });
    for (global, mut relative) in &mut relative {
        relative.0 = view_inverse.compose(&global.isometry);
    }
}
