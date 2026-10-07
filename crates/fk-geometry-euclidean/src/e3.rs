use fk_geometry::{Geometry, GpuGeometry, GroupElement, SampledMetric, VectorSpace};
use fk_math::Real;
use fk_math::nalgebra::{Matrix3, Matrix4, Point3, UnitQuaternion, Vector3, Vector4};
use fk_math::series::{cosc, sinc, sinc3};

/// Euclidean 3-space.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct E3;

/// A rigid motion of E³: rotate, then translate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Se3 {
    /// Rotation about the origin.
    pub rotation: UnitQuaternion<Real>,
    /// Translation applied after the rotation.
    pub translation: Vector3<Real>,
}

/// An element of 𝔰𝔢(3): angular velocity and linear velocity, both in the body frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Se3Twist {
    /// Rotation axis scaled by angular speed.
    pub angular: Vector3<Real>,
    /// Linear velocity.
    pub linear: Vector3<Real>,
}

twist_ops!(Se3Twist);

impl VectorSpace for Se3Twist {
    fn zero() -> Self {
        Self::default()
    }

    fn coord_norm_squared(&self) -> Real {
        self.angular.norm_squared() + self.linear.norm_squared()
    }
}

impl GroupElement for Se3 {
    type Algebra = Se3Twist;

    fn identity() -> Self {
        Self {
            rotation: UnitQuaternion::identity(),
            translation: Vector3::zeros(),
        }
    }

    fn compose(&self, rhs: &Self) -> Self {
        Self {
            rotation: self.rotation * rhs.rotation,
            translation: self.rotation * rhs.translation + self.translation,
        }
    }

    fn inverse(&self) -> Self {
        let rotation = self.rotation.inverse();
        Self {
            rotation,
            translation: -(rotation * self.translation),
        }
    }

    fn exp(xi: &Se3Twist) -> Self {
        let theta = xi.angular.norm();
        let w = xi.angular.cross_matrix();
        let v = Matrix3::identity() + w * cosc(theta) + w * w * sinc3(theta);
        Self {
            rotation: UnitQuaternion::from_scaled_axis(xi.angular),
            translation: v * xi.linear,
        }
    }

    fn log(&self) -> Se3Twist {
        let angular = rotation_log(&self.rotation);
        let theta = angular.norm();
        let w = angular.cross_matrix();
        let v_inv = Matrix3::identity() - w * 0.5 + w * w * log_coefficient(theta);
        Se3Twist {
            angular,
            linear: v_inv * self.translation,
        }
    }

    fn adjoint(&self, xi: &Se3Twist) -> Se3Twist {
        let angular = self.rotation * xi.angular;
        Se3Twist {
            angular,
            linear: self.rotation * xi.linear + self.translation.cross(&angular),
        }
    }

    fn renormalize(&mut self) {
        self.rotation.renormalize();
    }

    fn residual(&self) -> Real {
        (self.rotation.quaternion().norm() - 1.0).abs()
    }
}

/// The rotation vector of a unit quaternion, with angle in `[0, π]`.
fn rotation_log(q: &UnitQuaternion<Real>) -> Vector3<Real> {
    let (mut w, mut xyz) = (q.scalar(), q.imag());
    if w < 0.0 {
        w = -w;
        xyz = -xyz;
    }
    let s = xyz.norm();
    if s == 0.0 {
        return Vector3::zeros();
    }
    xyz * (2.0 * s.atan2(w) / s)
}

/// `(1 - θ sin θ / (2 (1 - cos θ))) / θ²`, the `W²` coefficient of the inverse of SE(3)'s left
/// Jacobian.
fn log_coefficient(theta: Real) -> Real {
    if theta < 1e-2 {
        let t2 = theta * theta;
        1.0 / 12.0 + t2 / 720.0 + t2 * t2 / 30_240.0
    } else {
        (1.0 - sinc(theta) / (2.0 * cosc(theta))) / (theta * theta)
    }
}

impl Geometry for E3 {
    const DIM: usize = 3;
    const NAME: &'static str = "E³";

    type Point = Point3<Real>;
    type Tangent = Vector3<Real>;
    type Isometry = Se3;

    fn origin() -> Point3<Real> {
        Point3::origin()
    }

    fn origin_frame(i: usize) -> Vector3<Real> {
        Vector3::ith(i, 1.0)
    }

    fn inner(_: &Point3<Real>, u: &Vector3<Real>, v: &Vector3<Real>) -> Real {
        u.dot(v)
    }

    fn distance(a: &Point3<Real>, b: &Point3<Real>) -> Real {
        (b - a).norm()
    }

    fn exp(p: &Point3<Real>, v: &Vector3<Real>) -> Point3<Real> {
        p + v
    }

    fn log(p: &Point3<Real>, q: &Point3<Real>) -> Option<Vector3<Real>> {
        Some(q - p)
    }

    fn parallel_transport(_: &Point3<Real>, _: &Point3<Real>, v: &Vector3<Real>) -> Vector3<Real> {
        *v
    }

    fn apply(g: &Se3, p: &Point3<Real>) -> Point3<Real> {
        g.rotation * p + g.translation
    }

    fn apply_tangent(g: &Se3, v: &Vector3<Real>) -> Vector3<Real> {
        g.rotation * v
    }

    fn transvection(v: &Vector3<Real>) -> Se3Twist {
        Se3Twist {
            angular: Vector3::zeros(),
            linear: *v,
        }
    }

    fn rotation(u: &Vector3<Real>, w: &Vector3<Real>) -> Se3Twist {
        Se3Twist {
            angular: u.cross(w),
            linear: Vector3::zeros(),
        }
    }

    fn constant_curvature() -> Option<Real> {
        Some(0.0)
    }

    fn injectivity_radius() -> Real {
        Real::INFINITY
    }

    fn project_tangent(_: &Point3<Real>, v: &Vector3<Real>) -> Vector3<Real> {
        *v
    }

    fn renormalize_point(_: &mut Point3<Real>) {}

    fn point_residual(p: &Point3<Real>) -> Real {
        if p.coords.iter().all(|c| c.is_finite()) {
            0.0
        } else {
            Real::INFINITY
        }
    }

    fn tangent_residual(_: &Point3<Real>, v: &Vector3<Real>) -> Real {
        if v.iter().all(|c| c.is_finite()) {
            0.0
        } else {
            Real::INFINITY
        }
    }
}

impl SampledMetric for E3 {
    const CHART_DIM: usize = 3;

    fn geodesic_acceleration(&self, _: &[Real], _: &[Real], out: &mut [Real]) {
        out.fill(0.0);
    }

    fn wgsl_module(&self) -> &'static str {
        "fn geo_geodesic_accel(x: vec3<f32>, v: vec3<f32>) -> vec3<f32> {\n    \
             return vec3<f32>(0.0);\n\
         }\n"
    }
}

/// Homogeneous coordinates: points are `(x, y, z, 1)`, tangent vectors `(x, y, z, 0)`, and a
/// rigid motion is the affine matrix `[R t; 0 1]`.
impl GpuGeometry for E3 {
    const SHADER: &'static str = "e3";

    fn isometry_matrix(g: &Se3) -> Matrix4<Real> {
        let mut m = g.rotation.to_homogeneous();
        m.fixed_view_mut::<3, 1>(0, 3).copy_from(&g.translation);
        m
    }

    fn embed_point(p: &Point3<Real>) -> Vector4<Real> {
        p.to_homogeneous()
    }

    fn embed_tangent(_: &Point3<Real>, v: &Vector3<Real>) -> Vector4<Real> {
        v.to_homogeneous()
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use super::*;

    fn twist(angular: [Real; 3], linear: [Real; 3]) -> Se3Twist {
        Se3Twist {
            angular: Vector3::from(angular),
            linear: Vector3::from(linear),
        }
    }

    fn assert_log_inverts_exp(xi: Se3Twist) {
        let back = Se3::exp(&xi).log();
        let gap = (back - xi).coord_norm_squared().sqrt();
        assert!(gap < 1e-12, "log(exp({xi:?})) = {back:?}");
    }

    #[test]
    fn exp_log_near_zero_rotation() {
        // The rotation angle is √2 · `angle`; 7e-3 and 7.1e-3 straddle the series branch at 1e-2.
        for angle in [0.0, 1e-12, 1e-8, 1e-5, 7e-3, 7.1e-3, 0.1] {
            assert_log_inverts_exp(twist([angle, 0.0, -angle], [1.0, -2.0, 0.5]));
        }
    }

    #[test]
    fn exp_log_near_half_turn() {
        let axis = Vector3::new(1.0, 2.0, -1.0).normalize();
        let angular = axis * (PI - 1e-6);
        assert_log_inverts_exp(Se3Twist {
            angular,
            linear: Vector3::new(0.3, 0.0, 2.0),
        });
    }

    #[test]
    fn screw_motion() {
        let g = Se3::exp(&twist([0.0, 0.0, PI / 2.0], [1.0, 0.0, 1.0]));
        let p = E3::apply(&g, &Point3::origin());
        let expected = Point3::new(2.0 / PI, 2.0 / PI, 1.0);
        assert!((p - expected).norm() < 1e-12, "{p}");
    }

    #[test]
    fn gpu_matrix_matches_the_action() {
        let g = Se3::exp(&twist([0.3, -1.2, 0.7], [2.0, -1.0, 0.5]));
        let h = Se3::exp(&twist([-0.4, 0.1, 2.0], [0.0, 3.0, -1.5]));
        let (p, v) = (Point3::new(1.0, -2.0, 0.5), Vector3::new(0.3, 0.7, -1.1));
        let m = E3::isometry_matrix(&g);
        let point = E3::embed_point(&E3::apply(&g, &p));
        assert!((m * E3::embed_point(&p) - point).norm() < 1e-12);
        let tangent = E3::embed_tangent(&E3::apply(&g, &p), &E3::apply_tangent(&g, &v));
        assert!((m * E3::embed_tangent(&p, &v) - tangent).norm() < 1e-12);
        let composed = E3::isometry_matrix(&g.compose(&h));
        assert!((composed - m * E3::isometry_matrix(&h)).norm() < 1e-12);
    }

    #[test]
    fn sampled_metric_agrees_with_closed_form() {
        let (p, v) = (Point3::new(1.0, -2.0, 0.5), Vector3::new(0.3, 0.7, -1.1));
        let mut x = [p.x, p.y, p.z];
        let mut xv = [v.x, v.y, v.z];
        fk_geometry::integrate_geodesic(&E3, &mut x, &mut xv, 0.01, 100);
        let q = E3::exp(&p, &v);
        assert!((Point3::from(x) - q).norm() < 1e-12);
    }
}
