//! Curved-space physics: geodesic motion, collision between geodesic balls and capsules, and
//! rigid bodies integrated on the isometry group.
//!
//! Generic over [`Geometry`](fk_geometry::Geometry); never names a concrete geometry. It is
//! staged as PLAN.md's honesty ladder:
//!
//! - **P1, geodesic kinematics** ([`motion`]): every motion is `pose ← pose ∘ exp(dt · ξ)`
//!   with a body-frame twist `ξ`, renormalized after each step; gravity is a force on the
//!   twist ([`Gravity`]), carried to the body by parallel transport; frames a body carries are
//!   parallel-transported ([`transport_frame`]).
//! - **P2, the slice's subset**: closed-form distances between balls and capsules
//!   ([`shapes`]), a uniform grid of buckets to find what is near ([`BucketGrid`]), and a
//!   character controller that walks, runs, jumps and slides along what it bumps into
//!   ([`Character`]). Shapes are measured in the normal coordinates of the body's own frame
//!   (the logarithm at its origin), where the closed forms are those of flat space: exact in
//!   E³, an approximation in curved space until the geodesic-ball closed forms of E1 land.
//!   Any surface given by a distance function collides too ([`field`]: a capsule advanced
//!   through the field and slid along it), with no mesh and no closed form.
//!   Ground detection is against one flat ground ([`Ground`]) and the floors of the field.
//! - **P2, complete (E1)**: the broadphase is a tree of geodesic balls ([`BallTree`]), the
//!   characters' colliders filed in it; in a quotient space each ball is copied across the
//!   faces of the fundamental domain ([`mod@ghosts`], [`GhostTree`]); moving balls that meet are
//!   given an impulse along the geodesic between their centres, their velocities
//!   parallel-transported to the contact point and back ([`contact`]).
//! - **P3, rigid bodies** ([`rigid`], E4): a body's configuration is one isometry and its
//!   momentum lives on the dual of the Lie algebra, integrated by a variational Lie-group
//!   integrator ([`RigidBody`]) with the inertia operator of its mass distribution
//!   ([`Inertia`]); resting contact against a [`DistanceField`].

pub mod bvh;
pub mod character;
pub mod contact;
pub mod field;
pub mod ghosts;
pub mod grid;
pub mod motion;
pub mod rigid;
pub mod shapes;

pub use bvh::BallTree;
pub use character::{Character, CharacterShape, Collider, Colliders, Ground, Surroundings};
pub use contact::{MovingBall, Resolved, resolve};
pub use field::{DistanceField, FieldBody, move_through};
pub use ghosts::{Ghost, GhostTree, ghosts};
pub use grid::BucketGrid;
pub use motion::{Gravity, advance, transport_frame};
pub use rigid::{Inertia, RigidBody};
pub use shapes::{Ball, Capsule, Contact};

use fk_geometry::{Geometry, VectorSpace};
use fk_math::Real;
use fk_math::nalgebra::Vector3;

/// Coordinates in the reference frame of a tangent vector at the origin.
pub fn coords<G: Geometry>(v: &G::Tangent) -> Vector3<Real> {
    let origin = G::origin();
    Vector3::from_fn(|i, _| G::inner(&origin, v, &G::origin_frame(i)))
}

/// The tangent vector at the origin with coordinates `c` in the reference frame.
pub fn tangent<G: Geometry>(c: &Vector3<Real>) -> G::Tangent {
    (0..3).fold(G::Tangent::zero(), |sum, i| sum + G::origin_frame(i) * c[i])
}
