use std::fmt::Debug;

use fk_math::Real;

use crate::VectorSpace;

/// An element of a Lie group, typically the isometry group of a geometry.
///
/// This is what replaces affine `Mat4` transforms: SE(3) for Euclidean space, SO(4) for the
/// 3-sphere, SO⁺(1,3) for hyperbolic space.
///
/// Conventions: [`compose`](Self::compose) is `self ∘ rhs`, applying `rhs` first; twists are in
/// the body frame, so motion is `pose ← pose ∘ exp(dt · ξ)`; integrators
/// [`renormalize`](Self::renormalize) after every step.
pub trait GroupElement: Copy + Debug + Send + Sync + 'static {
    /// The Lie algebra: infinitesimal motions, also called twists.
    type Algebra: VectorSpace;

    /// The neutral element.
    fn identity() -> Self;

    /// `self ∘ rhs`: applies `rhs` first, then `self`.
    fn compose(&self, rhs: &Self) -> Self;

    /// The inverse element.
    fn inverse(&self) -> Self;

    /// The group exponential of a Lie algebra element.
    fn exp(xi: &Self::Algebra) -> Self;

    /// The principal logarithm: some `ξ` with `exp(ξ) = self`.
    ///
    /// Groups with a non-surjective exponential must document what is returned outside its image.
    fn log(&self) -> Self::Algebra;

    /// The adjoint action `Ad_self(ξ)`, defined by `exp(Ad_g ξ) = g ∘ exp(ξ) ∘ g⁻¹`.
    fn adjoint(&self, xi: &Self::Algebra) -> Self::Algebra;

    /// Projects the representation back onto the group after numerical drift.
    ///
    /// Integrators call this after every step.
    fn renormalize(&mut self);

    /// How far the representation is from satisfying the group's constraints, `0` when exact.
    fn residual(&self) -> Real;
}
