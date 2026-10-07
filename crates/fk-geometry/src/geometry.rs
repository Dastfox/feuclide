use std::fmt::Debug;

use fk_math::Real;

use crate::{GroupElement, VectorSpace};

/// The Lie algebra of a geometry's isometry group.
pub type Algebra<G> = <<G as Geometry>::Isometry as GroupElement>::Algebra;

/// A homogeneous Riemannian geometry with closed-form geodesics.
///
/// Implementations are zero-sized marker types (`E3`, `H3`, `S3`, …); all state lives in points,
/// tangent vectors and isometries.
///
/// Geodesic operations ([`exp`](Self::exp), [`log`](Self::log),
/// [`parallel_transport`](Self::parallel_transport)) are exact within the
/// [`injectivity_radius`](Self::injectivity_radius); beyond it, geodesics stop being unique and
/// `log` may return `None`.
///
/// Conventions: the metric [`inner`](Self::inner) is Riemannian, positive definite, at a base
/// point; tangent vectors are stored in model coordinates and only mean something with their
/// base point; the sectional curvature ([`constant_curvature`](Self::constant_curvature)) is
/// positive for spheres, zero for Euclidean space and negative for hyperbolic space.
pub trait Geometry: Sized + Send + Sync + 'static {
    /// Intrinsic dimension.
    const DIM: usize;

    /// Human-readable name, used in diagnostics.
    const NAME: &'static str;

    /// A point, in the coordinates of the model (for example a hyperboloid point in ℝ⁴).
    type Point: Copy + Debug + Send + Sync + 'static;

    /// A tangent vector, in the coordinates of the model. Meaningful only with its base point.
    type Tangent: VectorSpace;

    /// An element of the isometry group.
    type Isometry: GroupElement;

    /// The base point of the identity pose.
    fn origin() -> Self::Point;

    /// Vector `i` of the reference orthonormal frame at [`origin`](Self::origin), for
    /// `i < DIM`.
    ///
    /// # Panics
    ///
    /// If `i >= DIM`.
    fn origin_frame(i: usize) -> Self::Tangent;

    /// The metric: inner product of two tangent vectors at `p`.
    fn inner(p: &Self::Point, u: &Self::Tangent, v: &Self::Tangent) -> Real;

    /// Length of a tangent vector at `p`.
    fn norm(p: &Self::Point, v: &Self::Tangent) -> Real {
        Self::inner(p, v, v).max(0.0).sqrt()
    }

    /// Geodesic distance.
    fn distance(a: &Self::Point, b: &Self::Point) -> Real;

    /// Riemannian exponential: follows the geodesic from `p` with initial velocity `v` for unit
    /// time.
    fn exp(p: &Self::Point, v: &Self::Tangent) -> Self::Point;

    /// Riemannian logarithm: the initial velocity at `p` of the minimizing geodesic reaching `q`
    /// in unit time. `None` when `q` is on the cut locus of `p` (for example its antipode on a
    /// sphere).
    fn log(p: &Self::Point, q: &Self::Point) -> Option<Self::Tangent>;

    /// Carries `v`, tangent at `p`, along the minimizing geodesic from `p` to `q`.
    fn parallel_transport(p: &Self::Point, q: &Self::Point, v: &Self::Tangent) -> Self::Tangent;

    /// The action of an isometry on points.
    fn apply(g: &Self::Isometry, p: &Self::Point) -> Self::Point;

    /// The differential of an isometry: maps a tangent vector at `p` to one at `apply(g, p)`.
    fn apply_tangent(g: &Self::Isometry, v: &Self::Tangent) -> Self::Tangent;

    /// The infinitesimal transvection along the geodesic leaving the origin with velocity `v`.
    ///
    /// `exp(transvection(v))` maps the origin to `exp(origin, v)` and parallel-transports every
    /// tangent vector along the way. This is a translation in Euclidean space and a boost in
    /// hyperbolic space.
    fn transvection(v: &Self::Tangent) -> Algebra<Self>;

    /// The infinitesimal rotation about the origin in the plane of `u` and `w`, turning `u`
    /// towards `w`.
    ///
    /// For orthonormal `u` and `w`, `exp(rotation(u, w) · θ)` rotates `u` by the angle `θ` and
    /// fixes the orthogonal complement of the plane.
    fn rotation(u: &Self::Tangent, w: &Self::Tangent) -> Algebra<Self>;

    /// The transvection carrying the origin to `p`, if `p` is not on the origin's cut locus.
    fn transvection_to(p: &Self::Point) -> Option<Self::Isometry> {
        let v = Self::log(&Self::origin(), p)?;
        Some(Self::Isometry::exp(&Self::transvection(&v)))
    }

    /// The sectional curvature if it is the same everywhere and in every plane, `None` otherwise.
    fn constant_curvature() -> Option<Real>;

    /// Distance within which geodesics from a point are unique and minimizing.
    fn injectivity_radius() -> Real;

    /// Projects an ambient vector onto the tangent space at `p`.
    fn project_tangent(p: &Self::Point, v: &Self::Tangent) -> Self::Tangent;

    /// Projects a point back onto the model after numerical drift.
    fn renormalize_point(p: &mut Self::Point);

    /// How far `p` is from satisfying the model's constraints, `0` when exact.
    fn point_residual(p: &Self::Point) -> Real;

    /// How far `v` is from being tangent at `p`, `0` when exact.
    fn tangent_residual(p: &Self::Point, v: &Self::Tangent) -> Real;
}
