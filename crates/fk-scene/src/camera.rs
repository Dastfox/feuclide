use std::marker::PhantomData;

use bevy_ecs::prelude::*;
use fk_geometry::{Geometry, GroupElement};
use fk_math::Real;

use crate::{GlobalPose, GlobalRealm, Realm};

/// A point of view: the entity's frame is the eye.
///
/// The eye looks along `−origin_frame(2)`, with `origin_frame(1)` up and `origin_frame(0)` to
/// the right. Angles are measured at the eye and distances along geodesics, so the same camera
/// means the same thing in every geometry.
#[derive(Component, Debug)]
pub struct Camera<G: Geometry> {
    /// Vertical field of view, in radians: the angle at the eye between the top and bottom
    /// edges of the image. Games animate it (a dolly zoom narrows it).
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

/// The active camera this frame, as a resource: which entity, its settings and where its eye
/// is. The frame is drawn from it, unless a [`Spectator`] eye is set ([`DrawnView`]). Written by
/// [`SceneSystems::CameraRelative`](crate::SceneSystems::CameraRelative); absent while no
/// camera is active.
#[derive(Resource, Debug)]
pub struct ActiveView<G: Geometry> {
    /// The camera entity.
    pub entity: Entity,
    /// Its settings.
    pub camera: Camera<G>,
    /// Its global pose.
    pub eye: GlobalPose<G>,
    /// The realm it is in, and sees: the camera entity's [`GlobalRealm`].
    pub realm: Realm,
}

impl<G: Geometry> Clone for ActiveView<G> {
    fn clone(&self) -> Self {
        Self {
            entity: self.entity,
            camera: self.camera.clone(),
            eye: self.eye.clone(),
            realm: self.realm,
        }
    }
}

/// An eye the frame is drawn from in place of the active camera's, as a resource: a camera
/// outside, to watch the scene from, while the [`ActiveView`] stays the active camera's for
/// everything that reads it (what it sees, where it hears, what it may do). With no
/// [`eye`](Self::eye), the frame is drawn from the active camera.
///
/// The engine gives it no meaning. It is drawn with the active camera's settings and realm,
/// but for its own field of view and far, if it has them.
#[derive(Resource, Debug)]
pub struct Spectator<G: Geometry> {
    /// Its global pose, if it is set.
    pub eye: Option<G::Isometry>,
    /// Its vertical field of view, radians, if not the active camera's.
    pub fov_y: Option<Real>,
    /// Its far, if not the active camera's.
    pub far: Option<Real>,
}

impl<G: Geometry> Default for Spectator<G> {
    fn default() -> Self {
        Self {
            eye: None,
            fov_y: None,
            far: None,
        }
    }
}

impl<G: Geometry> Clone for Spectator<G> {
    fn clone(&self) -> Self {
        Self {
            eye: self.eye,
            fov_y: self.fov_y,
            far: self.far,
        }
    }
}

/// The view the frame is drawn from, as a resource: the [`ActiveView`], or while a
/// [`Spectator`] eye is set, the active camera seen from there (that eye, its field of view and
/// far if it has them). [`ViewRelative`] poses are relative to its eye. Written with the
/// [`ActiveView`], absent with it.
#[derive(Resource, Debug)]
pub struct DrawnView<G: Geometry>(pub ActiveView<G>);

impl<G: Geometry> Clone for DrawnView<G> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

/// An entity's pose relative to the eye the frame is drawn from ([`DrawnView`]: the active
/// camera's, or a [`Spectator`]'s), `view⁻¹ ∘ global`, computed in `f64`.
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

/// A camera, where it is and the realm it is in.
type Eye<'a, G> = (
    Entity,
    &'a Camera<G>,
    &'a GlobalPose<G>,
    Option<&'a GlobalRealm>,
);

/// Finds the active camera, and the view drawn from (a [`Spectator`]'s, if one is set), and
/// writes every entity's [`ViewRelative`] pose.
pub fn update_view_relative<G: Geometry>(
    mut commands: Commands,
    cameras: Query<Eye<G>>,
    spectator: Option<Res<Spectator<G>>>,
    mut relative: Query<(&GlobalPose<G>, &mut ViewRelative<G>)>,
) {
    let Some((entity, camera, eye, realm)) = cameras.iter().find(|(_, camera, _, _)| camera.active)
    else {
        commands.remove_resource::<ActiveView<G>>();
        commands.remove_resource::<DrawnView<G>>();
        return;
    };
    let active = ActiveView {
        entity,
        camera: camera.clone(),
        eye: eye.clone(),
        realm: realm.map_or(Realm::MAIN, |realm| realm.0),
    };
    let mut drawn = active.clone();
    if let Some(spectator) = spectator.as_deref()
        && let Some(eye) = spectator.eye
    {
        drawn.eye.isometry = eye;
        drawn.camera.fov_y = spectator.fov_y.unwrap_or(drawn.camera.fov_y);
        drawn.camera.far = spectator
            .far
            .unwrap_or(drawn.camera.far)
            .max(2.0 * drawn.camera.near);
    }
    let view_inverse = drawn.eye.isometry.inverse();
    commands.insert_resource(active);
    commands.insert_resource(DrawnView(drawn));
    for (global, mut relative) in &mut relative {
        relative.0 = view_inverse.compose(&global.isometry);
    }
}
