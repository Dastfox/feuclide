use fk_geometry::{Geometry, GpuGeometry, GroupElement, SampledMetric, VectorSpace};
use fk_math::Real;
use fk_math::minkowski;
use fk_math::nalgebra::{Matrix4, Vector3, Vector4};
use fk_math::series::sinhc;

use crate::sl2c;

/// Hyperbolic 3-space, curvature −1, in the hyperboloid model.
///
/// Points are `(t, x, y, z)` with `⟨p, p⟩ = −1` and `t > 0` (Minkowski form of
/// [`fk_math::minkowski`], time first); the origin is `(1, 0, 0, 0)`, its reference frame the
/// axes x, y and z. Tangent vectors at `p` are Minkowski-orthogonal to it, and space-like.
/// Geodesics are unique: `log` is always defined.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct H3;

/// An element of SO⁺(1, 3): a 4×4 Lorentz matrix acting on `(t, x, y, z)`, keeping the
/// Minkowski form and the sign of `t`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lorentz3(pub Matrix4<Real>);

/// An element of 𝔰𝔬(1, 3): a boost (rapidity times direction) and a rotation (axis times
/// angle), both in the body frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Lorentz3Twist {
    /// The boost.
    pub boost: Vector3<Real>,
    /// The rotation.
    pub rotation: Vector3<Real>,
}

impl std::ops::Add for Lorentz3Twist {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            boost: self.boost + rhs.boost,
            rotation: self.rotation + rhs.rotation,
        }
    }
}

impl std::ops::Sub for Lorentz3Twist {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            boost: self.boost - rhs.boost,
            rotation: self.rotation - rhs.rotation,
        }
    }
}

impl std::ops::Neg for Lorentz3Twist {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            boost: -self.boost,
            rotation: -self.rotation,
        }
    }
}

impl std::ops::Mul<Real> for Lorentz3Twist {
    type Output = Self;
    fn mul(self, s: Real) -> Self {
        Self {
            boost: self.boost * s,
            rotation: self.rotation * s,
        }
    }
}

impl VectorSpace for Lorentz3Twist {
    fn zero() -> Self {
        Self::default()
    }

    fn coord_norm_squared(&self) -> Real {
        self.boost.norm_squared() + self.rotation.norm_squared()
    }
}

/// The Minkowski form as a matrix, `diag(−1, 1, 1, 1)`.
fn eta() -> Matrix4<Real> {
    Matrix4::from_diagonal(&Vector4::new(-1.0, 1.0, 1.0, 1.0))
}

/// Lorentz Gram–Schmidt on the columns of `m`: the first made a unit future time-like vector,
/// each next one Minkowski-orthogonal to those before and of unit length.
pub(crate) fn lorentz_gram_schmidt<const D: usize>(
    columns: &mut [fk_math::nalgebra::SVector<Real, D>; D],
) {
    for i in 0..D {
        for j in 0..i {
            let e = columns[j];
            let along = minkowski::dot(&columns[i], &e) / minkowski::dot(&e, &e);
            columns[i] -= e * along;
        }
        let n = minkowski::norm_squared(&columns[i]).abs().sqrt();
        columns[i] /= n;
    }
    if columns[0][0] < 0.0 {
        columns[0] = -columns[0];
    }
}

impl GroupElement for Lorentz3 {
    type Algebra = Lorentz3Twist;

    fn identity() -> Self {
        Self(Matrix4::identity())
    }

    fn compose(&self, rhs: &Self) -> Self {
        Self(self.0 * rhs.0)
    }

    /// `η Λᵀ η`.
    fn inverse(&self) -> Self {
        let eta = eta();
        Self(eta * self.0.transpose() * eta)
    }

    fn exp(xi: &Lorentz3Twist) -> Self {
        Self(sl2c::lorentz(&sl2c::exp(&xi.boost, &xi.rotation)))
    }

    /// The principal logarithm in SL(2, ℂ) of the lift whose trace has a non-negative real
    /// part: rotation angles up to π.
    fn log(&self) -> Lorentz3Twist {
        let (boost, rotation) = sl2c::log(&sl2c::from_lorentz(&self.0));
        Lorentz3Twist { boost, rotation }
    }

    fn adjoint(&self, xi: &Lorentz3Twist) -> Lorentz3Twist {
        let (boost, rotation) =
            sl2c::adjoint(&sl2c::from_lorentz(&self.0), &xi.boost, &xi.rotation);
        Lorentz3Twist { boost, rotation }
    }

    fn renormalize(&mut self) {
        let mut columns: [Vector4<Real>; 4] = std::array::from_fn(|i| self.0.column(i).into());
        lorentz_gram_schmidt(&mut columns);
        self.0 = Matrix4::from_columns(&columns);
    }

    /// `|Λᵀ η Λ − η|`, the largest entry.
    fn residual(&self) -> Real {
        let eta = eta();
        (self.0.transpose() * eta * self.0 - eta).amax()
    }
}

impl Geometry for H3 {
    const DIM: usize = 3;
    const NAME: &'static str = "H³";

    type Point = Vector4<Real>;
    type Tangent = Vector4<Real>;
    type Isometry = Lorentz3;

    fn origin() -> Vector4<Real> {
        Vector4::x()
    }

    fn origin_frame(i: usize) -> Vector4<Real> {
        assert!(i < 3, "H³ has three directions, not {}", i + 1);
        Vector4::ith(i + 1, 1.0)
    }

    fn inner(p: &Vector4<Real>, u: &Vector4<Real>, v: &Vector4<Real>) -> Real {
        debug_assert!(
            Self::point_residual(p) < 1e-6,
            "not on the hyperboloid: {p}"
        );
        minkowski::dot(u, v)
    }

    /// From `c = −⟨a, b⟩ = cosh d` and `s = sinh d` (the length of the part of `b` tangent at
    /// `a`): `asinh s` near, exact where `acosh c` loses half its digits, and `ln(c + s)` far.
    fn distance(a: &Vector4<Real>, b: &Vector4<Real>) -> Real {
        let c = -minkowski::dot(a, b);
        let s = minkowski::norm_squared(&(b - a * c)).max(0.0).sqrt();
        if c > 2.0 { (c + s).ln() } else { s.asinh() }
    }

    fn exp(p: &Vector4<Real>, v: &Vector4<Real>) -> Vector4<Real> {
        let theta = minkowski::norm_squared(v).max(0.0).sqrt();
        p * theta.cosh() + v * sinhc(theta)
    }

    fn log(p: &Vector4<Real>, q: &Vector4<Real>) -> Option<Vector4<Real>> {
        let c = -minkowski::dot(p, q);
        let w = q - p * c;
        let s = minkowski::norm_squared(&w).max(0.0).sqrt();
        Some(w * (1.0 / sinhc(s.asinh())))
    }

    /// `v + ⟨q, v⟩ / (1 − ⟨p, q⟩) · (p + q)`: the hyperboloid's twin of the sphere's, defined
    /// everywhere since `−⟨p, q⟩ ≥ 1`.
    fn parallel_transport(
        p: &Vector4<Real>,
        q: &Vector4<Real>,
        v: &Vector4<Real>,
    ) -> Vector4<Real> {
        v + (p + q) * (minkowski::dot(q, v) / (1.0 - minkowski::dot(p, q)))
    }

    fn apply(g: &Lorentz3, p: &Vector4<Real>) -> Vector4<Real> {
        g.0 * p
    }

    fn apply_tangent(g: &Lorentz3, v: &Vector4<Real>) -> Vector4<Real> {
        g.0 * v
    }

    /// A pure boost along `v`, by its length.
    fn transvection(v: &Vector4<Real>) -> Lorentz3Twist {
        Lorentz3Twist {
            boost: Vector3::new(v[1], v[2], v[3]),
            rotation: Vector3::zeros(),
        }
    }

    fn rotation(u: &Vector4<Real>, w: &Vector4<Real>) -> Lorentz3Twist {
        let (u, w) = (
            Vector3::new(u[1], u[2], u[3]),
            Vector3::new(w[1], w[2], w[3]),
        );
        Lorentz3Twist {
            boost: Vector3::zeros(),
            rotation: u.cross(&w),
        }
    }

    fn constant_curvature() -> Option<Real> {
        Some(-1.0)
    }

    fn injectivity_radius() -> Real {
        Real::INFINITY
    }

    fn project_tangent(p: &Vector4<Real>, v: &Vector4<Real>) -> Vector4<Real> {
        v + p * minkowski::dot(p, v)
    }

    fn renormalize_point(p: &mut Vector4<Real>) {
        let n = (-minkowski::norm_squared(p)).max(Real::MIN_POSITIVE).sqrt();
        *p /= n;
        if p[0] < 0.0 {
            *p = -*p;
        }
    }

    fn point_residual(p: &Vector4<Real>) -> Real {
        (minkowski::norm_squared(p) + 1.0).abs() / p[0].abs().max(1.0)
    }

    fn tangent_residual(p: &Vector4<Real>, v: &Vector4<Real>) -> Real {
        minkowski::dot(p, v).abs()
    }
}

/// The geodesic equation in the Poincaré ball, `x ↦ (1 + |x|², 2x) / (1 − |x|²)` (the chart's
/// origin is the origin): the metric is conformal, `4 / (1 − |x|²)² δ`, so
/// `ẍ = (2 |v|² x − 4 (x·v) v) / (1 − |x|²)`.
impl SampledMetric for H3 {
    const CHART_DIM: usize = 3;

    fn geodesic_acceleration(&self, x: &[Real], v: &[Real], out: &mut [Real]) {
        let (x, v) = (Vector3::from_column_slice(x), Vector3::from_column_slice(v));
        let a = (x * (2.0 * v.norm_squared()) - v * (4.0 * x.dot(&v))) / (1.0 - x.norm_squared());
        out.copy_from_slice(a.as_slice());
    }

    fn wgsl_module(&self) -> &'static str {
        "fn geo_geodesic_accel(x: vec3<f32>, v: vec3<f32>) -> vec3<f32> {\n    \
             return (2.0 * dot(v, v) * x - 4.0 * dot(x, v) * v) / (1.0 - dot(x, x));\n\
         }\n"
    }
}

/// The point of the Poincaré ball chart of [`H3`]'s [`SampledMetric`] at `x`.
pub fn from_ball(x: &Vector3<Real>) -> Vector4<Real> {
    let n = x.norm_squared();
    let s = 1.0 / (1.0 - n);
    Vector4::new((1.0 + n) * s, 2.0 * x.x * s, 2.0 * x.y * s, 2.0 * x.z * s)
}

/// The order the GPU reads the hyperboloid in: `(x, y, z, t)`, so that, as in E³'s homogeneous
/// coordinates, `xyz / w` is a point of the Beltrami–Klein model.
fn to_gpu_order() -> Matrix4<Real> {
    Matrix4::new(
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, 1.0, //
        1.0, 0.0, 0.0, 0.0,
    )
}

/// The hyperboloid with the time coordinate last, `(x, y, z, t)`; an isometry is its Lorentz
/// matrix in that order.
impl GpuGeometry for H3 {
    const SHADER: &'static str = "h3";

    fn isometry_matrix(g: &Lorentz3) -> Matrix4<Real> {
        let p = to_gpu_order();
        p * g.0 * p.transpose()
    }

    fn embed_point(p: &Vector4<Real>) -> Vector4<Real> {
        to_gpu_order() * p
    }

    fn embed_tangent(_: &Vector4<Real>, v: &Vector4<Real>) -> Vector4<Real> {
        to_gpu_order() * v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance_is_exact_near_and_far() {
        let o = H3::origin();
        for d in [1e-9, 1e-4, 0.5, 3.0, 20.0] {
            let q = H3::exp(&o, &(H3::origin_frame(0) * d));
            assert!((H3::distance(&o, &q) - d).abs() < 1e-12 * (1.0 + d), "{d}");
        }
    }

    #[test]
    fn gpu_matrix_matches_the_action() {
        let g = Lorentz3::exp(&Lorentz3Twist {
            boost: Vector3::new(0.3, -1.2, 0.7),
            rotation: Vector3::new(-0.4, 0.1, 2.0),
        });
        let h = Lorentz3::exp(&Lorentz3Twist {
            boost: Vector3::new(1.0, 0.2, 0.3),
            rotation: Vector3::new(0.0, 0.5, -0.1),
        });
        let p = H3::exp(&H3::origin(), &Vector4::new(0.0, 0.3, -0.2, 0.9));
        let m = H3::isometry_matrix(&g);
        assert!((m * H3::embed_point(&p) - H3::embed_point(&H3::apply(&g, &p))).norm() < 1e-12);
        let composed = H3::isometry_matrix(&g.compose(&h));
        assert!((composed - m * H3::isometry_matrix(&h)).norm() < 1e-12);
    }

    #[test]
    fn sampled_metric_agrees_with_closed_form() {
        // From the ball's centre with chart velocity v: the hyperboloid's tangent there is 2v.
        let v = Vector3::new(0.3, -0.2, 0.5);
        let (mut x, mut xv) = ([0.0; 3], [v.x, v.y, v.z]);
        fk_geometry::integrate_geodesic(&H3, &mut x, &mut xv, 0.001, 1000);
        let q = H3::exp(
            &H3::origin(),
            &Vector4::new(0.0, 2.0 * v.x, 2.0 * v.y, 2.0 * v.z),
        );
        assert!((from_ball(&Vector3::from(x)) - q).norm() < 1e-9);
    }

    #[test]
    fn renormalizing_keeps_a_lorentz_matrix() {
        let mut g = Lorentz3::exp(&Lorentz3Twist {
            boost: Vector3::new(1.3, -0.2, 0.7),
            rotation: Vector3::new(0.4, 0.1, -2.0),
        });
        g.0[(1, 2)] += 1e-7;
        assert!(g.residual() > 1e-8);
        g.renormalize();
        assert!(g.residual() < 1e-12, "{}", g.residual());
    }
}
