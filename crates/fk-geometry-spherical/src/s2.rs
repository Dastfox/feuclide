use fk_geometry::{Geometry, GroupElement, SampledMetric};
use fk_math::Real;
use fk_math::nalgebra::{UnitQuaternion, Vector2, Vector3};
use fk_math::series::sinc;

use crate::{quaternion_exp, quaternion_log};

/// The unit 2-sphere in ℝ³, curvature +1.
///
/// Points are unit 3-vectors; the origin is the north pole `(0, 0, 1)`, its reference frame
/// the axes x and y. Tangent vectors at `p` are 3-vectors orthogonal to `p`. The cut locus is
/// the antipode, as on [`S3`](crate::S3): `log` returns `None` within
/// [`ANTIPODE`](crate::ANTIPODE) of it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct S2;

/// A rotation of the 2-sphere, an element of SO(3), as a unit quaternion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct So3(pub UnitQuaternion<Real>);

impl GroupElement for So3 {
    /// The rotation vector: axis times angle.
    type Algebra = Vector3<Real>;

    fn identity() -> Self {
        Self(UnitQuaternion::identity())
    }

    fn compose(&self, rhs: &Self) -> Self {
        Self(self.0 * rhs.0)
    }

    fn inverse(&self) -> Self {
        Self(self.0.inverse())
    }

    fn exp(xi: &Vector3<Real>) -> Self {
        Self(quaternion_exp(&(xi * 0.5)))
    }

    /// The rotation vector with angle in `[0, 2π]`, of the quaternion as it is (not of its
    /// opposite, which is the same rotation).
    fn log(&self) -> Vector3<Real> {
        quaternion_log(&self.0) * 2.0
    }

    fn adjoint(&self, xi: &Vector3<Real>) -> Vector3<Real> {
        self.0 * xi
    }

    fn renormalize(&mut self) {
        self.0.renormalize();
    }

    fn residual(&self) -> Real {
        (self.0.quaternion().norm() - 1.0).abs()
    }
}

impl Geometry for S2 {
    const DIM: usize = 2;
    const NAME: &'static str = "S²";

    type Point = Vector3<Real>;
    type Tangent = Vector3<Real>;
    type Isometry = So3;

    fn origin() -> Vector3<Real> {
        Vector3::z()
    }

    fn origin_frame(i: usize) -> Vector3<Real> {
        assert!(i < 2, "S² has two directions, not {}", i + 1);
        Vector3::ith(i, 1.0)
    }

    fn inner(_: &Vector3<Real>, u: &Vector3<Real>, v: &Vector3<Real>) -> Real {
        u.dot(v)
    }

    fn distance(a: &Vector3<Real>, b: &Vector3<Real>) -> Real {
        a.cross(b).norm().atan2(a.dot(b))
    }

    fn exp(p: &Vector3<Real>, v: &Vector3<Real>) -> Vector3<Real> {
        let theta = v.norm();
        p * theta.cos() + v * sinc(theta)
    }

    fn log(p: &Vector3<Real>, q: &Vector3<Real>) -> Option<Vector3<Real>> {
        let c = p.dot(q);
        if 1.0 + c < crate::ANTIPODE {
            return None;
        }
        let w = q - p * c;
        let theta = w.norm().atan2(c);
        Some(w * (1.0 / sinc(theta)))
    }

    fn parallel_transport(
        p: &Vector3<Real>,
        q: &Vector3<Real>,
        v: &Vector3<Real>,
    ) -> Vector3<Real> {
        let c = 1.0 + p.dot(q);
        if c < crate::ANTIPODE {
            return Self::project_tangent(q, v);
        }
        v - (p + q) * (q.dot(v) / c)
    }

    fn apply(g: &So3, p: &Vector3<Real>) -> Vector3<Real> {
        g.0 * p
    }

    fn apply_tangent(g: &So3, v: &Vector3<Real>) -> Vector3<Real> {
        g.0 * v
    }

    /// The rotation about `o × v` by `|v|`: it carries the north pole along `v`.
    fn transvection(v: &Vector3<Real>) -> Vector3<Real> {
        Self::origin().cross(v)
    }

    fn rotation(u: &Vector3<Real>, w: &Vector3<Real>) -> Vector3<Real> {
        u.cross(w)
    }

    fn constant_curvature() -> Option<Real> {
        Some(1.0)
    }

    fn injectivity_radius() -> Real {
        std::f64::consts::PI
    }

    fn project_tangent(p: &Vector3<Real>, v: &Vector3<Real>) -> Vector3<Real> {
        v - p * p.dot(v)
    }

    fn renormalize_point(p: &mut Vector3<Real>) {
        p.normalize_mut();
    }

    fn point_residual(p: &Vector3<Real>) -> Real {
        (p.norm() - 1.0).abs()
    }

    fn tangent_residual(p: &Vector3<Real>, v: &Vector3<Real>) -> Real {
        p.dot(v).abs()
    }
}

/// The stereographic chart from the south pole, `x ↦ (2x, 1 − |x|²) / (1 + |x|²)`, as for
/// [`S3`](crate::S3): `ẍ = (4 (x·v) v − 2 |v|² x) / (1 + |x|²)`.
impl SampledMetric for S2 {
    const CHART_DIM: usize = 2;

    fn geodesic_acceleration(&self, x: &[Real], v: &[Real], out: &mut [Real]) {
        let (x, v) = (Vector2::from_column_slice(x), Vector2::from_column_slice(v));
        let a = (v * (4.0 * x.dot(&v)) - x * (2.0 * v.norm_squared())) / (1.0 + x.norm_squared());
        out.copy_from_slice(a.as_slice());
    }

    fn wgsl_module(&self) -> &'static str {
        "fn geo_geodesic_accel(x: vec2<f32>, v: vec2<f32>) -> vec2<f32> {\n    \
             return (4.0 * dot(x, v) * v - 2.0 * dot(v, v) * x) / (1.0 + dot(x, x));\n\
         }\n"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_quarter_turn_from_the_pole_is_the_equator() {
        let q = S2::exp(&S2::origin(), &(Vector3::x() * std::f64::consts::FRAC_PI_2));
        assert!((q - Vector3::x()).norm() < 1e-12);
        let g = So3::exp(&S2::transvection(
            &(Vector3::x() * std::f64::consts::FRAC_PI_2),
        ));
        assert!((S2::apply(&g, &S2::origin()) - Vector3::x()).norm() < 1e-12);
    }
}
