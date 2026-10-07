use fk_geometry::{Geometry, GpuGeometry, GroupElement, SampledMetric, VectorSpace};
use fk_math::Real;
use fk_math::nalgebra::{Matrix4, Quaternion, UnitQuaternion, Vector3, Vector4};
use fk_math::series::sinc;

use crate::{quaternion_exp, quaternion_log};

/// The unit 3-sphere in ℝ⁴, curvature +1.
///
/// Points are unit 4-vectors `(x, y, z, w)`, read as the quaternion `w + xi + yj + zk`; the
/// origin is `(0, 0, 0, 1)`, the quaternion 1, and its reference frame the axes x, y and z.
/// Tangent vectors at `p` are 4-vectors orthogonal to `p`.
///
/// Cut locus: the antipode `−p` of `p` is reached by every geodesic from `p` at distance π, so
/// [`log`](Geometry::log) returns `None` when `q` is within [`ANTIPODE`] of `−p`, and
/// [`parallel_transport`](Geometry::parallel_transport) to the antipode projects instead.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct S3;

/// How near the antipode a point must be for [`S3`]'s `log` to give up (in `1 + p·q`, which
/// goes as half the square of the distance to the antipode).
pub const ANTIPODE: Real = 1e-14;

/// An element of SO(4): the pair of unit quaternions `(l, r)` acting as `x ↦ l x r̄`. The pair
/// `(−l, −r)` is the same rotation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct So4 {
    /// The left factor.
    pub left: UnitQuaternion<Real>,
    /// The right factor.
    pub right: UnitQuaternion<Real>,
}

/// An element of 𝔰𝔬(4) = 𝔰𝔲(2) ⊕ 𝔰𝔲(2): two pure quaternions, the exponents of the factors,
/// `exp(ξ) = (e^left, e^right)`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct So4Twist {
    /// The left exponent, as the imaginary part of a pure quaternion.
    pub left: Vector3<Real>,
    /// The right exponent.
    pub right: Vector3<Real>,
}

impl std::ops::Add for So4Twist {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            left: self.left + rhs.left,
            right: self.right + rhs.right,
        }
    }
}

impl std::ops::Sub for So4Twist {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            left: self.left - rhs.left,
            right: self.right - rhs.right,
        }
    }
}

impl std::ops::Neg for So4Twist {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            left: -self.left,
            right: -self.right,
        }
    }
}

impl std::ops::Mul<Real> for So4Twist {
    type Output = Self;
    fn mul(self, s: Real) -> Self {
        Self {
            left: self.left * s,
            right: self.right * s,
        }
    }
}

impl VectorSpace for So4Twist {
    fn zero() -> Self {
        Self::default()
    }

    fn coord_norm_squared(&self) -> Real {
        self.left.norm_squared() + self.right.norm_squared()
    }
}

/// A 4-vector as a quaternion: `(x, y, z, w) ↦ w + xi + yj + zk`.
fn quaternion(v: &Vector4<Real>) -> Quaternion<Real> {
    Quaternion::from(*v)
}

impl GroupElement for So4 {
    type Algebra = So4Twist;

    fn identity() -> Self {
        Self {
            left: UnitQuaternion::identity(),
            right: UnitQuaternion::identity(),
        }
    }

    fn compose(&self, rhs: &Self) -> Self {
        Self {
            left: self.left * rhs.left,
            right: self.right * rhs.right,
        }
    }

    fn inverse(&self) -> Self {
        Self {
            left: self.left.inverse(),
            right: self.right.inverse(),
        }
    }

    fn exp(xi: &So4Twist) -> Self {
        Self {
            left: quaternion_exp(&xi.left),
            right: quaternion_exp(&xi.right),
        }
    }

    /// Each factor's principal logarithm, its angle in `[0, π]`.
    fn log(&self) -> So4Twist {
        So4Twist {
            left: quaternion_log(&self.left),
            right: quaternion_log(&self.right),
        }
    }

    fn adjoint(&self, xi: &So4Twist) -> So4Twist {
        So4Twist {
            left: self.left * xi.left,
            right: self.right * xi.right,
        }
    }

    fn renormalize(&mut self) {
        self.left.renormalize();
        self.right.renormalize();
    }

    fn residual(&self) -> Real {
        let off = |q: &UnitQuaternion<Real>| (q.quaternion().norm() - 1.0).abs();
        off(&self.left).max(off(&self.right))
    }
}

impl Geometry for S3 {
    const DIM: usize = 3;
    const NAME: &'static str = "S³";

    type Point = Vector4<Real>;
    type Tangent = Vector4<Real>;
    type Isometry = So4;

    fn origin() -> Vector4<Real> {
        Vector4::w()
    }

    fn origin_frame(i: usize) -> Vector4<Real> {
        assert!(i < 3, "S³ has three directions, not {}", i + 1);
        Vector4::ith(i, 1.0)
    }

    fn inner(_: &Vector4<Real>, u: &Vector4<Real>, v: &Vector4<Real>) -> Real {
        u.dot(v)
    }

    fn distance(a: &Vector4<Real>, b: &Vector4<Real>) -> Real {
        let c = a.dot(b);
        (b - a * c).norm().atan2(c)
    }

    fn exp(p: &Vector4<Real>, v: &Vector4<Real>) -> Vector4<Real> {
        let theta = v.norm();
        p * theta.cos() + v * sinc(theta)
    }

    fn log(p: &Vector4<Real>, q: &Vector4<Real>) -> Option<Vector4<Real>> {
        let c = p.dot(q);
        if 1.0 + c < ANTIPODE {
            return None;
        }
        let w = q - p * c;
        let theta = w.norm().atan2(c);
        Some(w * (1.0 / sinc(theta)))
    }

    fn parallel_transport(
        p: &Vector4<Real>,
        q: &Vector4<Real>,
        v: &Vector4<Real>,
    ) -> Vector4<Real> {
        let c = 1.0 + p.dot(q);
        if c < ANTIPODE {
            return Self::project_tangent(q, v);
        }
        v - (p + q) * (q.dot(v) / c)
    }

    fn apply(g: &So4, p: &Vector4<Real>) -> Vector4<Real> {
        (g.left.quaternion() * quaternion(p) * g.right.quaternion().conjugate()).coords
    }

    fn apply_tangent(g: &So4, v: &Vector4<Real>) -> Vector4<Real> {
        Self::apply(g, v)
    }

    /// `(v/2, −v/2)`: `x ↦ e^{v/2} x e^{v/2}` slides the origin along `v` and carries its
    /// frame along.
    fn transvection(v: &Vector4<Real>) -> So4Twist {
        let half = v.xyz() * 0.5;
        So4Twist {
            left: half,
            right: -half,
        }
    }

    /// Conjugation by `e^{(u×w)/2}`: the rotation about `u × w` fixing the origin.
    fn rotation(u: &Vector4<Real>, w: &Vector4<Real>) -> So4Twist {
        let half = u.xyz().cross(&w.xyz()) * 0.5;
        So4Twist {
            left: half,
            right: half,
        }
    }

    fn constant_curvature() -> Option<Real> {
        Some(1.0)
    }

    fn injectivity_radius() -> Real {
        std::f64::consts::PI
    }

    fn project_tangent(p: &Vector4<Real>, v: &Vector4<Real>) -> Vector4<Real> {
        v - p * p.dot(v)
    }

    fn renormalize_point(p: &mut Vector4<Real>) {
        p.normalize_mut();
    }

    fn point_residual(p: &Vector4<Real>) -> Real {
        (p.norm() - 1.0).abs()
    }

    fn tangent_residual(p: &Vector4<Real>, v: &Vector4<Real>) -> Real {
        p.dot(v).abs()
    }
}

/// The geodesic equation in the stereographic chart from the south pole, `x ↦ (2x, 1 − |x|²) /
/// (1 + |x|²)` (the chart's origin is the origin, `(0, 0, 0, 1)`): the metric is conformal,
/// `4 / (1 + |x|²)² δ`, so `ẍ = (4 (x·v) v − 2 |v|² x) / (1 + |x|²)`.
impl SampledMetric for S3 {
    const CHART_DIM: usize = 3;

    fn geodesic_acceleration(&self, x: &[Real], v: &[Real], out: &mut [Real]) {
        let (x, v) = (Vector3::from_column_slice(x), Vector3::from_column_slice(v));
        let a = (v * (4.0 * x.dot(&v)) - x * (2.0 * v.norm_squared())) / (1.0 + x.norm_squared());
        out.copy_from_slice(a.as_slice());
    }

    fn wgsl_module(&self) -> &'static str {
        "fn geo_geodesic_accel(x: vec3<f32>, v: vec3<f32>) -> vec3<f32> {\n    \
             return (4.0 * dot(x, v) * v - 2.0 * dot(v, v) * x) / (1.0 + dot(x, x));\n\
         }\n"
    }
}

/// The point of the stereographic chart of [`S3`]'s [`SampledMetric`] at `x`.
pub fn from_chart(x: &Vector3<Real>) -> Vector4<Real> {
    let n = x.norm_squared();
    let s = 1.0 / (1.0 + n);
    Vector4::new(2.0 * x.x * s, 2.0 * x.y * s, 2.0 * x.z * s, (1.0 - n) * s)
}

/// The unit sphere itself: points and tangent vectors as they are, an isometry as the matrix of
/// `x ↦ l x r̄`.
impl GpuGeometry for S3 {
    const SHADER: &'static str = "s3";

    fn isometry_matrix(g: &So4) -> Matrix4<Real> {
        Matrix4::from_columns(&[0, 1, 2, 3].map(|i| Self::apply(g, &Vector4::ith(i, 1.0))))
    }

    fn embed_point(p: &Vector4<Real>) -> Vector4<Real> {
        *p
    }

    fn embed_tangent(_: &Vector4<Real>, v: &Vector4<Real>) -> Vector4<Real> {
        *v
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use super::*;

    #[test]
    fn flying_straight_comes_back_round_through_the_antipode() {
        let ahead = -S3::origin_frame(2);
        let o = S3::origin();
        let half = S3::exp(&o, &(ahead * PI));
        assert!(
            (half + o).norm() < 1e-12,
            "half way round is the antipode: {half}"
        );
        assert!(
            S3::log(&o, &half).is_none(),
            "the antipode is the cut locus"
        );
        let round = S3::exp(&o, &(ahead * (2.0 * PI)));
        assert!((round - o).norm() < 1e-12, "all the way round is home");
        // The same with the group: a transvection along a great circle, a full turn.
        let g = So4::exp(&(S3::transvection(&ahead) * (2.0 * PI)));
        assert!((S3::apply(&g, &o) - o).norm() < 1e-12);
    }

    #[test]
    fn log_near_the_antipode_is_long_and_points_somewhere() {
        let o = S3::origin();
        let near = S3::exp(&o, &(S3::origin_frame(0) * (PI - 1e-4)));
        let v = S3::log(&o, &near).expect("not on the cut locus yet");
        assert!((S3::norm(&o, &v) - (PI - 1e-4)).abs() < 1e-9);
    }

    #[test]
    fn gpu_matrix_matches_the_action() {
        let g = So4::exp(&So4Twist {
            left: Vector3::new(0.3, -1.2, 0.7),
            right: Vector3::new(-0.4, 0.1, 2.0),
        });
        let h = So4::exp(&So4Twist {
            left: Vector3::new(0.0, 0.5, -0.1),
            right: Vector3::new(1.0, 0.2, 0.3),
        });
        let p = S3::exp(&S3::origin(), &Vector4::new(0.3, -0.2, 0.9, 0.0));
        let m = S3::isometry_matrix(&g);
        assert!((m * S3::embed_point(&p) - S3::apply(&g, &p)).norm() < 1e-12);
        let composed = S3::isometry_matrix(&g.compose(&h));
        assert!((composed - m * S3::isometry_matrix(&h)).norm() < 1e-12);
        assert!(
            (m.transpose() * m - Matrix4::identity()).norm() < 1e-12,
            "orthogonal"
        );
    }

    #[test]
    fn sampled_metric_agrees_with_closed_form() {
        // From the chart's origin with chart velocity v: the sphere's tangent there is 2v.
        let v = Vector3::new(0.3, -0.2, 0.5);
        let (mut x, mut xv) = ([0.0; 3], [v.x, v.y, v.z]);
        fk_geometry::integrate_geodesic(&S3, &mut x, &mut xv, 0.001, 1000);
        let q = S3::exp(&S3::origin(), &(v * 2.0).push(0.0));
        assert!((from_chart(&Vector3::from(x)) - q).norm() < 1e-9);
    }
}
