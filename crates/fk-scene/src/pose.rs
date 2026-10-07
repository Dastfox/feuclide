use bevy_ecs::prelude::*;
use fk_app::Time;
use fk_geometry::{Algebra, Geometry, GroupElement, VectorSpace};

use crate::ChartTag;

/// Where an entity is: the isometry carrying its parent's frame to its own.
///
/// An entity without a parent ([`ChildOf`]) is relative to the scene root. There is no scale: a
/// pose is an element of the isometry group. Adding a `Pose` adds a [`GlobalPose`] and a
/// [`ViewRelative`](crate::ViewRelative), both filled in every frame.
#[derive(Component, Debug)]
#[require(GlobalPose<G>, crate::ViewRelative<G>)]
pub struct Pose<G: Geometry>(pub G::Isometry);

impl<G: Geometry> Pose<G> {
    /// The parent's frame itself.
    pub fn identity() -> Self {
        Self(G::Isometry::identity())
    }
}

impl<G: Geometry> Default for Pose<G> {
    fn default() -> Self {
        Self::identity()
    }
}

impl<G: Geometry> Clone for Pose<G> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<G: Geometry> Copy for Pose<G> {}

/// Where an entity is relative to the scene root, written by propagation from the [`Pose`]s
/// along its ancestry: `world(child) = world(parent) ∘ local(child)`.
///
/// Read it after [`SceneSystems::Propagate`](crate::SceneSystems::Propagate); writing it has no
/// lasting effect.
#[derive(Component, Debug)]
pub struct GlobalPose<G: Geometry> {
    /// The isometry carrying the root frame to the entity's frame.
    pub isometry: G::Isometry,
    /// The chart the isometry is expressed in.
    pub chart: ChartTag,
}

impl<G: Geometry> GlobalPose<G> {
    /// The global pose of a root entity at `isometry`, in the representative chart.
    pub fn new(isometry: G::Isometry) -> Self {
        Self {
            isometry,
            chart: ChartTag::identity(),
        }
    }

    /// The entity's position: where it carries the origin.
    pub fn point(&self) -> G::Point {
        G::apply(&self.isometry, &G::origin())
    }
}

impl<G: Geometry> Default for GlobalPose<G> {
    fn default() -> Self {
        Self::new(G::Isometry::identity())
    }
}

impl<G: Geometry> Clone for GlobalPose<G> {
    fn clone(&self) -> Self {
        Self {
            isometry: self.isometry,
            chart: self.chart.clone(),
        }
    }
}

/// How an entity moves: a twist in its own (body) frame, in units per second.
///
/// [`SceneSystems::Integrate`](crate::SceneSystems::Integrate) moves the [`Pose`] by
/// `pose ← pose ∘ exp(dt · ξ)` every fixed step and renormalizes it.
#[derive(Component, Debug)]
pub struct Velocity<G: Geometry>(pub Algebra<G>);

impl<G: Geometry> Default for Velocity<G> {
    fn default() -> Self {
        Self(Algebra::<G>::zero())
    }
}

impl<G: Geometry> Clone for Velocity<G> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<G: Geometry> Copy for Velocity<G> {}

/// Advances every [`Pose`] with a [`Velocity`] by one fixed step.
pub fn integrate_velocities<G: Geometry>(
    time: Res<Time>,
    mut bodies: Query<(&mut Pose<G>, &Velocity<G>)>,
) {
    let dt = time.fixed_step();
    for (mut pose, velocity) in &mut bodies {
        let step = G::Isometry::exp(&(velocity.0 * dt));
        let mut moved = pose.0.compose(&step);
        moved.renormalize();
        pose.0 = moved;
    }
}
