//! P1, geodesic kinematics: twists, gravity on the twist, carried frames.

use fk_geometry::{Algebra, Geometry, GroupElement};
use fk_math::Real;

/// One step of motion: `pose ∘ exp(dt · twist)`, renormalized. The twist is in the body's
/// frame.
pub fn advance<G: Geometry>(pose: &G::Isometry, twist: &Algebra<G>, dt: Real) -> G::Isometry {
    let mut moved = pose.compose(&G::Isometry::exp(&(*twist * dt)));
    moved.renormalize();
    moved
}

/// Carries every vector of `frame`, tangent at `p`, to `q` along the geodesic joining them:
/// what a body moving from `p` to `q` without turning carries along.
pub fn transport_frame<G: Geometry>(p: &G::Point, q: &G::Point, frame: &mut [G::Tangent]) {
    for v in frame {
        *v = G::parallel_transport(p, q, v);
    }
}

/// A uniform pull: `acceleration` along `down`, a unit tangent vector at the origin, carried
/// to every point by parallel transport along the geodesic from the origin. In flat space that
/// is the same direction everywhere.
#[derive(Debug)]
pub struct Gravity<G: Geometry> {
    /// Unit direction of the pull at the origin.
    pub down: G::Tangent,
    /// Its strength, in units of distance per second squared.
    pub acceleration: Real,
}

impl<G: Geometry> Clone for Gravity<G> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<G: Geometry> Copy for Gravity<G> {}

impl<G: Geometry> Gravity<G> {
    /// Down at the place of a body at `pose`, in the body's own frame: a tangent vector at the
    /// origin.
    pub fn down_in_body(&self, pose: &G::Isometry) -> G::Tangent {
        let origin = G::origin();
        let here = G::apply(pose, &origin);
        let down = G::parallel_transport(&origin, &here, &self.down);
        G::apply_tangent(&pose.inverse(), &down)
    }

    /// The change of a body's velocity over `dt` of free fall, in its own frame.
    pub fn pull(&self, pose: &G::Isometry, dt: Real) -> G::Tangent {
        self.down_in_body(pose) * (self.acceleration * dt)
    }

    /// The body twist after `dt` of free fall: gravity as a force on the twist, added to its
    /// translational part.
    pub fn accelerate(&self, twist: &Algebra<G>, pose: &G::Isometry, dt: Real) -> Algebra<G> {
        *twist + G::transvection(&self.pull(pose, dt))
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::FRAC_PI_2;

    use fk_geometry::VectorSpace;
    use fk_geometry_euclidean::{E3, Se3, Se3Twist};
    use fk_math::nalgebra::{Point3, Vector3};

    use super::*;

    fn gravity() -> Gravity<E3> {
        Gravity {
            down: -Vector3::y(),
            acceleration: 9.81,
        }
    }

    #[test]
    fn a_dropped_body_falls_half_g_t_squared() {
        let mut pose = Se3::exp(&E3::transvection(&Vector3::new(0.0, 100.0, 0.0)));
        let mut twist = Se3Twist::zero();
        let dt = 1e-3;
        for _ in 0..1000 {
            twist = gravity().accelerate(&twist, &pose, dt);
            pose = advance::<E3>(&pose, &twist, dt);
        }
        let height = E3::apply(&pose, &E3::origin()).y;
        assert!((height - (100.0 - 9.81 / 2.0)).abs() < 0.01, "{height}");
        assert!((twist.linear.y + 9.81).abs() < 1e-9);
    }

    #[test]
    fn down_is_seen_in_the_body_frame() {
        // Rolled a quarter turn about −z: the body's +x now points up, so down is its −x.
        let roll = Se3::exp(&(E3::rotation(&Vector3::x(), &Vector3::y()) * FRAC_PI_2));
        let down = gravity().down_in_body(&roll);
        assert!((down - -Vector3::x()).norm() < 1e-12, "{down}");
    }

    #[test]
    fn carried_frames_are_unchanged_in_flat_space() {
        let mut frame = [Vector3::x(), Vector3::y(), Vector3::z()];
        let (p, q) = (Point3::new(1.0, 2.0, 3.0), Point3::new(-4.0, 0.5, 9.0));
        transport_frame::<E3>(&p, &q, &mut frame);
        assert_eq!(frame, [Vector3::x(), Vector3::y(), Vector3::z()]);
    }
}
