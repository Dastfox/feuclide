//! Geometry as a trait.
//!
//! Feuclide never assumes space is Euclidean. A space is a type implementing [`Geometry`]: it says
//! what its points and tangent vectors are, how far apart two points are, how geodesics run, how
//! vectors are carried along them, and which group of isometries acts on it. Everything above this
//! crate (scene, renderer, physics) is written against these traits and never names a concrete
//! space.
//!
//! # Conventions
//!
//! - **Poses are isometries.** An object's pose is the element `g` of the isometry group that
//!   carries the reference frame at [`Geometry::origin`] to the object's frame. There is no `Mat4`
//!   and no global coordinate system; the origin is only the base point of the identity coset.
//! - **Composition applies the right operand first:** `g.compose(&h)` acts as `x ↦ g(h(x))`.
//! - **Twists are body-frame.** Moving a pose `g` by the Lie algebra element `ξ` for time `dt` is
//!   `g.compose(&exp(ξ · dt))`.
//! - **Tangent vectors are stored in the coordinates of the model.** For embedded models (the
//!   hyperboloid, the unit sphere) a tangent vector at `p` is an ambient vector satisfying the
//!   model's tangency constraint at `p`, and only means something together with `p`.
//!
//! [`GpuGeometry`] is the GPU boundary: a 4-vector embedding of points and a 4×4 matrix per
//! isometry, so the renderer can stay generic.
//!
//! The `battery` feature exposes [`geometry_battery!`], a property-test suite that every geometry
//! implementation runs. It is the executable definition of a correct [`Geometry`].

#[cfg(feature = "battery")]
pub mod battery;
mod geometry;
mod gpu;
mod group;
mod sampled;
mod vector_space;

pub use fk_math::Real;
pub use geometry::{Algebra, Geometry};
pub use gpu::GpuGeometry;
pub use group::GroupElement;
pub use sampled::{SampledMetric, integrate_geodesic};
pub use vector_space::VectorSpace;
