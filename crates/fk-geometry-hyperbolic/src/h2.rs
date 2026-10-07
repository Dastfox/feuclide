use fk_geometry::{Geometry, GroupElement, SampledMetric};
use fk_math::Real;
use fk_math::minkowski;
use fk_math::nalgebra::{Matrix3, Matrix4, Vector2, Vector3};
use fk_math::series::sinhc;

use crate::h3::lorentz_gram_schmidt;
use crate::sl2c;

/// The hyperbolic plane, curvature −1, in the hyperboloid model: points `(t, x, y)` with
/// `⟨p, p⟩ = −1`, `t > 0`; the origin `(1, 0, 0)`, its frame the axes x and y. It is the slice
/// `z = 0` of [`H3`](crate::H3), and its group SO⁺(1, 2) the transformations of H³ keeping it:
/// exponentials and logarithms go through H³'s.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct H2;

/// An element of SO⁺(1, 2): a 3×3 Lorentz matrix acting on `(t, x, y)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lorentz2(pub Matrix3<Real>);

/// An element of 𝔰𝔬(1, 2): `(boost x, boost y, rotation)`, the rotation turning x towards y.
pub type Lorentz2Twist = Vector3<Real>;

/// `Λ` as a transformation of H³ keeping `z = 0`.
fn lift(m: &Matrix3<Real>) -> Matrix4<Real> {
    let mut big = Matrix4::identity();
    big.fixed_view_mut::<3, 3>(0, 0).copy_from(m);
    big
}

/// The twist of H³ that `ξ` is.
fn lift_twist(xi: &Lorentz2Twist) -> (Vector3<Real>, Vector3<Real>) {
    (Vector3::new(xi.x, xi.y, 0.0), Vector3::new(0.0, 0.0, xi.z))
}

fn drop_twist((boost, rotation): (Vector3<Real>, Vector3<Real>)) -> Lorentz2Twist {
    Vector3::new(boost.x, boost.y, rotation.z)
}

impl GroupElement for Lorentz2 {
    type Algebra = Lorentz2Twist;

    fn identity() -> Self {
        Self(Matrix3::identity())
    }

    fn compose(&self, rhs: &Self) -> Self {
        Self(self.0 * rhs.0)
    }

    fn inverse(&self) -> Self {
        let eta = Matrix3::from_diagonal(&Vector3::new(-1.0, 1.0, 1.0));
        Self(eta * self.0.transpose() * eta)
    }

    fn exp(xi: &Lorentz2Twist) -> Self {
        let (boost, rotation) = lift_twist(xi);
        Self(
            sl2c::lorentz(&sl2c::exp(&boost, &rotation))
                .fixed_view::<3, 3>(0, 0)
                .into(),
        )
    }

    /// Through H³'s: rotation angles up to π.
    fn log(&self) -> Lorentz2Twist {
        drop_twist(sl2c::log(&sl2c::from_lorentz(&lift(&self.0))))
    }

    fn adjoint(&self, xi: &Lorentz2Twist) -> Lorentz2Twist {
        let (boost, rotation) = lift_twist(xi);
        drop_twist(sl2c::adjoint(
            &sl2c::from_lorentz(&lift(&self.0)),
            &boost,
            &rotation,
        ))
    }

    fn renormalize(&mut self) {
        let mut columns: [Vector3<Real>; 3] = std::array::from_fn(|i| self.0.column(i).into());
        lorentz_gram_schmidt(&mut columns);
        self.0 = Matrix3::from_columns(&columns);
    }

    fn residual(&self) -> Real {
        let eta = Matrix3::from_diagonal(&Vector3::new(-1.0, 1.0, 1.0));
        (self.0.transpose() * eta * self.0 - eta).amax()
    }
}

impl Geometry for H2 {
    const DIM: usize = 2;
    const NAME: &'static str = "H²";

    type Point = Vector3<Real>;
    type Tangent = Vector3<Real>;
    type Isometry = Lorentz2;

    fn origin() -> Vector3<Real> {
        Vector3::x()
    }

    fn origin_frame(i: usize) -> Vector3<Real> {
        assert!(i < 2, "H² has two directions, not {}", i + 1);
        Vector3::ith(i + 1, 1.0)
    }

    fn inner(p: &Vector3<Real>, u: &Vector3<Real>, v: &Vector3<Real>) -> Real {
        debug_assert!(
            Self::point_residual(p) < 1e-6,
            "not on the hyperboloid: {p}"
        );
        minkowski::dot(u, v)
    }

    fn distance(a: &Vector3<Real>, b: &Vector3<Real>) -> Real {
        let c = -minkowski::dot(a, b);
        let s = minkowski::norm_squared(&(b - a * c)).max(0.0).sqrt();
        if c > 2.0 { (c + s).ln() } else { s.asinh() }
    }

    fn exp(p: &Vector3<Real>, v: &Vector3<Real>) -> Vector3<Real> {
        let theta = minkowski::norm_squared(v).max(0.0).sqrt();
        p * theta.cosh() + v * sinhc(theta)
    }

    fn log(p: &Vector3<Real>, q: &Vector3<Real>) -> Option<Vector3<Real>> {
        let c = -minkowski::dot(p, q);
        let w = q - p * c;
        let s = minkowski::norm_squared(&w).max(0.0).sqrt();
        Some(w * (1.0 / sinhc(s.asinh())))
    }

    fn parallel_transport(
        p: &Vector3<Real>,
        q: &Vector3<Real>,
        v: &Vector3<Real>,
    ) -> Vector3<Real> {
        v + (p + q) * (minkowski::dot(q, v) / (1.0 - minkowski::dot(p, q)))
    }

    fn apply(g: &Lorentz2, p: &Vector3<Real>) -> Vector3<Real> {
        g.0 * p
    }

    fn apply_tangent(g: &Lorentz2, v: &Vector3<Real>) -> Vector3<Real> {
        g.0 * v
    }

    fn transvection(v: &Vector3<Real>) -> Lorentz2Twist {
        Vector3::new(v[1], v[2], 0.0)
    }

    fn rotation(u: &Vector3<Real>, w: &Vector3<Real>) -> Lorentz2Twist {
        Vector3::new(0.0, 0.0, u[1] * w[2] - u[2] * w[1])
    }

    fn constant_curvature() -> Option<Real> {
        Some(-1.0)
    }

    fn injectivity_radius() -> Real {
        Real::INFINITY
    }

    fn project_tangent(p: &Vector3<Real>, v: &Vector3<Real>) -> Vector3<Real> {
        v + p * minkowski::dot(p, v)
    }

    fn renormalize_point(p: &mut Vector3<Real>) {
        let n = (-minkowski::norm_squared(p)).max(Real::MIN_POSITIVE).sqrt();
        *p /= n;
        if p[0] < 0.0 {
            *p = -*p;
        }
    }

    fn point_residual(p: &Vector3<Real>) -> Real {
        (minkowski::norm_squared(p) + 1.0).abs() / p[0].abs().max(1.0)
    }

    fn tangent_residual(p: &Vector3<Real>, v: &Vector3<Real>) -> Real {
        minkowski::dot(p, v).abs()
    }
}

/// The Poincaré disc, as [`H3`](crate::H3)'s ball: `ẍ = (2 |v|² x − 4 (x·v) v) / (1 − |x|²)`.
impl SampledMetric for H2 {
    const CHART_DIM: usize = 2;

    fn geodesic_acceleration(&self, x: &[Real], v: &[Real], out: &mut [Real]) {
        let (x, v) = (Vector2::from_column_slice(x), Vector2::from_column_slice(v));
        let a = (x * (2.0 * v.norm_squared()) - v * (4.0 * x.dot(&v))) / (1.0 - x.norm_squared());
        out.copy_from_slice(a.as_slice());
    }

    fn wgsl_module(&self) -> &'static str {
        "fn geo_geodesic_accel(x: vec2<f32>, v: vec2<f32>) -> vec2<f32> {\n    \
             return (2.0 * dot(v, v) * x - 4.0 * dot(x, v) * v) / (1.0 - dot(x, x));\n\
         }\n"
    }
}
