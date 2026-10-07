use bevy_ecs::prelude::*;
use fk_geometry::{Geometry, GroupElement};
use fk_math::Real;

use crate::{ActiveView, Pose};

/// The floating origin: whenever the active camera's eye is farther than `radius` from the
/// origin, the whole scene is moved so that the eye is at the origin again (every root
/// [`Pose`] composed after the inverse of the transvection carrying the origin to the eye).
/// Nothing changes relative to anything else; the coordinates stay small.
///
/// In hyperbolic space a pose's coordinates grow like `e^d`, and what is computed from two
/// poses far out (their relative pose, a body's spatial momentum) loses digits like `e^{2d}`:
/// an open H³ world needs this, with a radius of a few units. Euclidean space loses digits
/// only linearly, and in S³ every point is within π of the origin. Insert the resource to turn
/// it on; [`SceneSystems::Rebase`](crate::SceneSystems::Rebase) moves the roots in
/// `PostUpdate`, before the poses propagate.
///
/// Anything that keeps world poses outside the roots' `Pose`s (a physics resource, a saved
/// place) compares [`rebases`](Self::rebases) with the count it last saw and moves its poses
/// by [`last`](Self::last) when it changed.
#[derive(Resource, Debug)]
pub struct FloatingOrigin<G: Geometry> {
    /// How far the eye may go from the origin before the scene is moved.
    pub radius: Real,
    /// How many times the scene has been moved.
    pub rebases: u64,
    /// The last move: a world pose `g` became `last ∘ g`.
    pub last: G::Isometry,
    /// Every move so far composed: the first world's pose `g` is now `total ∘ g`.
    pub total: G::Isometry,
}

impl<G: Geometry> FloatingOrigin<G> {
    /// The floating origin, moving the scene whenever the eye is farther than `radius`.
    pub fn new(radius: Real) -> Self {
        Self {
            radius,
            rebases: 0,
            last: G::Isometry::identity(),
            total: G::Isometry::identity(),
        }
    }
}

/// Moves every root pose so that the eye is at the origin, when it has gone past the radius.
pub fn rebase<G: Geometry>(
    origin: Option<ResMut<FloatingOrigin<G>>>,
    view: Option<Res<ActiveView<G>>>,
    mut roots: Query<&mut Pose<G>, Without<ChildOf>>,
) {
    let (Some(mut origin), Some(view)) = (origin, view) else {
        return;
    };
    let eye = view.eye.point();
    if G::distance(&G::origin(), &eye) <= origin.radius {
        return;
    }
    let Some(to_eye) = G::transvection_to(&eye) else {
        return;
    };
    let shift = to_eye.inverse();
    for mut pose in &mut roots {
        pose.0 = shift.compose(&pose.0);
        pose.0.renormalize();
    }
    origin.rebases += 1;
    origin.last = shift;
    origin.total = shift.compose(&origin.total);
    origin.total.renormalize();
}
