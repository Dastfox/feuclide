use fk_math::Real;
use fk_math::nalgebra::{Matrix4, Vector4};

use crate::Geometry;

/// A geometry the GPU can draw: points and tangent vectors embedded as 4-vectors, isometries
/// acting on them as 4×4 matrices.
///
/// This is the GPU boundary. The renderer composes poses in `f64` on the CPU, makes them
/// camera-relative, and uploads one [`isometry_matrix`](Self::isometry_matrix) per instance;
/// the vertex shader is the same `mat4 · v` in every geometry, followed by the geometry's own
/// projection from the WGSL module named by [`SHADER`](Self::SHADER).
///
/// The embedding is homogeneous coordinates `(x, y, z, 1)` in E³, the hyperboloid in H³ and
/// the unit 3-sphere in ℝ⁴ for S³. Implementations must satisfy, for every isometry `g`, point
/// `p` and tangent vector `v` at `p`:
///
/// - `isometry_matrix(g) · embed_point(p) = embed_point(apply(g, p))`
/// - `isometry_matrix(g) · embed_tangent(p, v) = embed_tangent(apply(g, p), apply_tangent(g, v))`
/// - `isometry_matrix(g ∘ h) = isometry_matrix(g) · isometry_matrix(h)`
///
/// The camera looks along `−origin_frame(2)` with `origin_frame(1)` up and `origin_frame(0)`
/// to the right, so the shader module's projection knows which way is forward.
pub trait GpuGeometry: Geometry {
    /// Name of the shader module, in `fk-shaders`, that implements `fk::geometry` for this
    /// geometry.
    const SHADER: &'static str;

    /// The matrix by which `g` acts on embedded points and tangent vectors.
    fn isometry_matrix(g: &Self::Isometry) -> Matrix4<Real>;

    /// The embedding of a point in ℝ⁴.
    fn embed_point(p: &Self::Point) -> Vector4<Real>;

    /// The embedding of a tangent vector at `p` in ℝ⁴.
    fn embed_tangent(p: &Self::Point, v: &Self::Tangent) -> Vector4<Real>;
}
