use fk_geometry::{Geometry, GroupElement, SampledMetric, VectorSpace};
use fk_math::Real;
use fk_math::nalgebra::{Point2, UnitComplex, Vector2};
use fk_math::series::{cosc, sinc};

/// The Euclidean plane.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct E2;

/// A rigid motion of E²: rotate, then translate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Se2 {
    /// Rotation about the origin.
    pub rotation: UnitComplex<Real>,
    /// Translation applied after the rotation.
    pub translation: Vector2<Real>,
}

/// An element of 𝔰𝔢(2): angular velocity and linear velocity, both in the body frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Se2Twist {
    /// Counter-clockwise angular speed.
    pub angular: Real,
    /// Linear velocity.
    pub linear: Vector2<Real>,
}

twist_ops!(Se2Twist);

impl VectorSpace for Se2Twist {
    fn zero() -> Self {
        Self::default()
    }

    fn coord_norm_squared(&self) -> Real {
        self.angular * self.angular + self.linear.norm_squared()
    }
}

/// Quarter turn counter-clockwise, the generator `J` of rotations.
fn perp(v: &Vector2<Real>) -> Vector2<Real> {
    Vector2::new(-v.y, v.x)
}

impl GroupElement for Se2 {
    type Algebra = Se2Twist;

    fn identity() -> Self {
        Self {
            rotation: UnitComplex::identity(),
            translation: Vector2::zeros(),
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

    fn exp(xi: &Se2Twist) -> Self {
        // The left Jacobian is V = a·I + b·J.
        let theta = xi.angular;
        let (a, b) = (sinc(theta), theta * cosc(theta));
        Self {
            rotation: UnitComplex::new(theta),
            translation: xi.linear * a + perp(&xi.linear) * b,
        }
    }

    fn log(&self) -> Se2Twist {
        // V⁻¹ = (a·I - b·J) / (a² + b²), since J² = -I.
        let theta = self.rotation.angle();
        let (a, b) = (sinc(theta), theta * cosc(theta));
        let t = self.translation;
        Se2Twist {
            angular: theta,
            linear: (t * a - perp(&t) * b) / (a * a + b * b),
        }
    }

    fn adjoint(&self, xi: &Se2Twist) -> Se2Twist {
        Se2Twist {
            angular: xi.angular,
            linear: self.rotation * xi.linear - perp(&self.translation) * xi.angular,
        }
    }

    fn renormalize(&mut self) {
        self.rotation.renormalize();
    }

    fn residual(&self) -> Real {
        (self.rotation.complex().norm() - 1.0).abs()
    }
}

impl Geometry for E2 {
    const DIM: usize = 2;
    const NAME: &'static str = "E²";

    type Point = Point2<Real>;
    type Tangent = Vector2<Real>;
    type Isometry = Se2;

    fn origin() -> Point2<Real> {
        Point2::origin()
    }

    fn origin_frame(i: usize) -> Vector2<Real> {
        Vector2::ith(i, 1.0)
    }

    fn inner(_: &Point2<Real>, u: &Vector2<Real>, v: &Vector2<Real>) -> Real {
        u.dot(v)
    }

    fn distance(a: &Point2<Real>, b: &Point2<Real>) -> Real {
        (b - a).norm()
    }

    fn exp(p: &Point2<Real>, v: &Vector2<Real>) -> Point2<Real> {
        p + v
    }

    fn log(p: &Point2<Real>, q: &Point2<Real>) -> Option<Vector2<Real>> {
        Some(q - p)
    }

    fn parallel_transport(_: &Point2<Real>, _: &Point2<Real>, v: &Vector2<Real>) -> Vector2<Real> {
        *v
    }

    fn apply(g: &Se2, p: &Point2<Real>) -> Point2<Real> {
        g.rotation * p + g.translation
    }

    fn apply_tangent(g: &Se2, v: &Vector2<Real>) -> Vector2<Real> {
        g.rotation * v
    }

    fn transvection(v: &Vector2<Real>) -> Se2Twist {
        Se2Twist {
            angular: 0.0,
            linear: *v,
        }
    }

    fn rotation(u: &Vector2<Real>, w: &Vector2<Real>) -> Se2Twist {
        Se2Twist {
            angular: u.x * w.y - u.y * w.x,
            linear: Vector2::zeros(),
        }
    }

    fn constant_curvature() -> Option<Real> {
        Some(0.0)
    }

    fn injectivity_radius() -> Real {
        Real::INFINITY
    }

    fn project_tangent(_: &Point2<Real>, v: &Vector2<Real>) -> Vector2<Real> {
        *v
    }

    fn renormalize_point(_: &mut Point2<Real>) {}

    fn point_residual(p: &Point2<Real>) -> Real {
        if p.coords.iter().all(|c| c.is_finite()) {
            0.0
        } else {
            Real::INFINITY
        }
    }

    fn tangent_residual(_: &Point2<Real>, v: &Vector2<Real>) -> Real {
        if v.iter().all(|c| c.is_finite()) {
            0.0
        } else {
            Real::INFINITY
        }
    }
}

impl SampledMetric for E2 {
    const CHART_DIM: usize = 2;

    fn geodesic_acceleration(&self, _: &[Real], _: &[Real], out: &mut [Real]) {
        out.fill(0.0);
    }

    fn wgsl_module(&self) -> &'static str {
        "fn geo_geodesic_accel(x: vec2<f32>, v: vec2<f32>) -> vec2<f32> {\n    \
             return vec2<f32>(0.0);\n\
         }\n"
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use super::*;

    #[test]
    fn quarter_turn_about_the_origin() {
        let g = Se2::exp(&Se2Twist {
            angular: PI / 2.0,
            linear: Vector2::zeros(),
        });
        let p = E2::apply(&g, &Point2::new(1.0, 0.0));
        assert!((p - Point2::new(0.0, 1.0)).norm() < 1e-15, "{p}");
    }

    #[test]
    fn exp_log_near_zero_and_half_turn() {
        for angle in [0.0, 1e-12, 1e-6, 1.0, PI - 1e-6, -(PI - 1e-6)] {
            let xi = Se2Twist {
                angular: angle,
                linear: Vector2::new(1.5, -0.25),
            };
            let back = Se2::exp(&xi).log();
            let gap = (back - xi).coord_norm_squared().sqrt();
            assert!(gap < 1e-12, "log(exp({xi:?})) = {back:?}");
        }
    }
}
