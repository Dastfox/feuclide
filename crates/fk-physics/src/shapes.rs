//! Closed-form distances between balls and capsules, in flat coordinates.
//!
//! These are the flat-space (E³) closed forms. The [`Character`](crate::Character) applies them
//! in the normal coordinates of its own frame, where they are exact in flat space.

use fk_math::Real;
use fk_math::nalgebra::Vector3;

/// A ball.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ball {
    /// Its centre.
    pub centre: Vector3<Real>,
    /// Its radius.
    pub radius: Real,
}

/// A capsule: every point within `radius` of the segment from `a` to `b`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Capsule {
    /// One end of the axis.
    pub a: Vector3<Real>,
    /// The other end.
    pub b: Vector3<Real>,
    /// Its radius.
    pub radius: Real,
}

impl Capsule {
    /// The ball about the point at `t ∈ [0, 1]` along the axis.
    pub fn ball_at(&self, t: Real) -> Ball {
        Ball {
            centre: self.a + (self.b - self.a) * t,
            radius: self.radius,
        }
    }
}

/// How two shapes stand: the first's distance from the second and the way out.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Contact {
    /// Distance between the surfaces, negative when they overlap (minus the depth).
    pub distance: Real,
    /// Unit direction from the second shape towards the first, along which moving the first by
    /// `−distance` makes them just touch.
    pub normal: Vector3<Real>,
}

/// Where along the segment `a b` the point nearest `p` is, as `t ∈ [0, 1]`.
pub fn closest_on_segment(p: &Vector3<Real>, a: &Vector3<Real>, b: &Vector3<Real>) -> Real {
    let ab = b - a;
    let length2 = ab.norm_squared();
    if length2 <= Real::EPSILON {
        return 0.0;
    }
    ((p - a).dot(&ab) / length2).clamp(0.0, 1.0)
}

/// The nearest points of the segments `p0 p1` and `q0 q1`, as `(s, t) ∈ [0, 1]²` along each.
/// Parallel segments get one of their nearest pairs.
pub fn closest_between_segments(
    p0: &Vector3<Real>,
    p1: &Vector3<Real>,
    q0: &Vector3<Real>,
    q1: &Vector3<Real>,
) -> (Real, Real) {
    let (d1, d2, r) = (p1 - p0, q1 - q0, p0 - q0);
    let (a, e, f) = (d1.norm_squared(), d2.norm_squared(), d2.dot(&r));
    let tiny = Real::EPSILON;
    if a <= tiny && e <= tiny {
        return (0.0, 0.0);
    }
    if a <= tiny {
        return (0.0, (f / e).clamp(0.0, 1.0));
    }
    let c = d1.dot(&r);
    if e <= tiny {
        return ((-c / a).clamp(0.0, 1.0), 0.0);
    }
    let b = d1.dot(&d2);
    let denominator = a * e - b * b;
    let mut s = if denominator > tiny * a * e {
        ((b * f - c * e) / denominator).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let mut t = (b * s + f) / e;
    if t < 0.0 {
        t = 0.0;
        s = (-c / a).clamp(0.0, 1.0);
    } else if t > 1.0 {
        t = 1.0;
        s = ((b - c) / a).clamp(0.0, 1.0);
    }
    (s, t)
}

/// Ball against ball.
pub fn ball_ball(first: &Ball, second: &Ball) -> Contact {
    let between = first.centre - second.centre;
    let length = between.norm();
    Contact {
        distance: length - first.radius - second.radius,
        normal: direction(&between, length),
    }
}

/// Ball against capsule.
pub fn ball_capsule(ball: &Ball, capsule: &Capsule) -> Contact {
    let t = closest_on_segment(&ball.centre, &capsule.a, &capsule.b);
    ball_ball(ball, &capsule.ball_at(t))
}

/// Capsule against capsule.
pub fn capsule_capsule(first: &Capsule, second: &Capsule) -> Contact {
    let (s, t) = closest_between_segments(&first.a, &first.b, &second.a, &second.b);
    ball_ball(&first.ball_at(s), &second.ball_at(t))
}

/// `v / length`, or some unit vector perpendicular to the vertical when `v` vanishes (two
/// centres on top of each other have no way out of their own).
fn direction(v: &Vector3<Real>, length: Real) -> Vector3<Real> {
    if length > 1e-12 {
        v / length
    } else {
        Vector3::x()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capsule(a: [Real; 3], b: [Real; 3], radius: Real) -> Capsule {
        Capsule {
            a: Vector3::from(a),
            b: Vector3::from(b),
            radius,
        }
    }

    fn close(a: Real, b: Real) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn crossing_capsules_meet_at_their_axes() {
        let x = capsule([-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.25);
        let z = capsule([0.5, 1.0, -1.0], [0.5, 1.0, 1.0], 0.25);
        let contact = capsule_capsule(&z, &x);
        assert!(close(contact.distance, 0.5), "{contact:?}");
        assert!((contact.normal - Vector3::y()).norm() < 1e-12);
    }

    #[test]
    fn parallel_upright_capsules_are_apart_by_their_axes() {
        let tower = capsule([3.0, 0.0, 4.0], [3.0, 48.0, 4.0], 1.8);
        let body = capsule([0.0, 0.3, 0.0], [0.0, 1.5, 0.0], 0.3);
        let contact = capsule_capsule(&body, &tower);
        assert!(close(contact.distance, 5.0 - 2.1), "{contact:?}");
        assert!((contact.normal - Vector3::new(-0.6, 0.0, -0.8)).norm() < 1e-12);
    }

    #[test]
    fn overlap_is_negative_and_the_normal_leads_out() {
        let tower = capsule([1.0, 0.0, 0.0], [1.0, 48.0, 0.0], 1.8);
        let body = capsule([0.0, 0.3, 0.0], [0.0, 1.5, 0.0], 0.3);
        let contact = capsule_capsule(&body, &tower);
        assert!(close(contact.distance, 1.0 - 2.1));
        assert!((contact.normal + Vector3::x()).norm() < 1e-12);
    }

    #[test]
    fn end_caps_count() {
        let low = capsule([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.5);
        let ball = Ball {
            centre: Vector3::new(0.0, 3.0, 0.0),
            radius: 0.5,
        };
        assert!(close(ball_capsule(&ball, &low).distance, 1.0));
        let above = capsule([0.0, 3.0, 0.0], [0.0, 4.0, 0.0], 0.5);
        assert!(close(capsule_capsule(&above, &low).distance, 1.0));
    }

    #[test]
    fn segments_find_their_nearest_points() {
        let (s, t) = closest_between_segments(
            &Vector3::new(0.0, 0.0, 0.0),
            &Vector3::new(2.0, 0.0, 0.0),
            &Vector3::new(1.5, 1.0, -1.0),
            &Vector3::new(1.5, 1.0, 1.0),
        );
        assert!(close(s, 0.75) && close(t, 0.5), "{s} {t}");
        // Degenerate segments are points.
        let (s, t) = closest_between_segments(
            &Vector3::zeros(),
            &Vector3::zeros(),
            &Vector3::new(-1.0, 1.0, 0.0),
            &Vector3::new(1.0, 1.0, 0.0),
        );
        assert!(close(s, 0.0) && close(t, 0.5));
    }
}
