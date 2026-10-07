//! Contacts between moving geodesic balls, resolved by an impulse along the geodesic that
//! joins their centres.
//!
//! Two balls `(c₁, r₁)` and `(c₂, r₂)` at distance `d < r₁ + r₂` overlap by `r₁ + r₂ − d`. Their
//! contact point is on the geodesic from `c₁` to `c₂`, in the middle of the overlap,
//! `(d + r₁ − r₂) / 2` from `c₁`, and the contact normal there is the geodesic's unit tangent.
//! Each ball's velocity lives at its own centre; it is parallel-transported to the contact
//! point, the impulse is applied there along the normal (with restitution `e`: the closing speed
//! `v` along the normal becomes `−e v`, momentum kept), and the change is transported back.
//! The overlap is then taken out by moving both centres apart along the geodesic, each by its
//! share of the inverse masses. In flat space this is the textbook collision of two spheres; in
//! curved space the transport is what makes "along the normal" mean the same thing to both.

use fk_geometry::Geometry;
use fk_math::Real;

/// A ball that moves: where it is, its velocity at its centre, its size and its mass.
#[derive(Clone, Copy, Debug)]
pub struct MovingBall<G: Geometry> {
    /// Its centre.
    pub centre: G::Point,
    /// Its velocity, a tangent vector at its centre.
    pub velocity: G::Tangent,
    /// Its radius.
    pub radius: Real,
    /// Its inverse mass: 0 for a ball nothing moves.
    pub inverse_mass: Real,
}

/// What resolving one contact did.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Resolved {
    /// How deep the balls overlapped before they were separated.
    pub depth: Real,
    /// The impulse given along the normal (0 if they were already parting).
    pub impulse: Real,
}

/// Resolves the contact between `a` and `b`, if they overlap, with restitution `restitution`
/// (0 they stick along the normal, 1 they bounce back as fast). `None` if they do not overlap,
/// or if their centres are too far apart or too close for a normal to be defined.
pub fn resolve<G: Geometry>(
    a: &mut MovingBall<G>,
    b: &mut MovingBall<G>,
    restitution: Real,
) -> Option<Resolved> {
    let towards = G::log(&a.centre, &b.centre)?;
    let d = G::norm(&a.centre, &towards);
    let depth = a.radius + b.radius - d;
    if depth <= 0.0 || d < 1e-12 {
        return None;
    }
    let unit = towards * (1.0 / d);
    let point = G::exp(&a.centre, &(unit * ((d + a.radius - b.radius) / 2.0)));
    let normal = G::parallel_transport(&a.centre, &point, &unit);
    let va = G::parallel_transport(&a.centre, &point, &a.velocity);
    let vb = G::parallel_transport(&b.centre, &point, &b.velocity);
    // The closing speed: how fast `a` comes on `b` along the normal.
    let closing = G::inner(&point, &(va - vb), &normal);
    let mass = a.inverse_mass + b.inverse_mass;
    let impulse = if closing > 0.0 && mass > 0.0 {
        (1.0 + restitution) * closing / mass
    } else {
        0.0
    };
    if impulse > 0.0 {
        let va = va - normal * (impulse * a.inverse_mass);
        let vb = vb + normal * (impulse * b.inverse_mass);
        a.velocity = G::parallel_transport(&point, &a.centre, &va);
        b.velocity = G::parallel_transport(&point, &b.centre, &vb);
    }
    // Apart, each by its share, along the geodesic (the velocities carried along).
    if mass > 0.0 {
        let back = G::log(&b.centre, &a.centre)?;
        let back = back * (1.0 / d);
        let (sa, sb) = (depth * a.inverse_mass / mass, depth * b.inverse_mass / mass);
        let (from_a, from_b) = (a.centre, b.centre);
        a.centre = G::exp(&from_a, &(unit * -sa));
        b.centre = G::exp(&from_b, &(back * -sb));
        a.velocity = G::parallel_transport(&from_a, &a.centre, &a.velocity);
        b.velocity = G::parallel_transport(&from_b, &b.centre, &b.velocity);
    }
    Some(Resolved { depth, impulse })
}

/// The depth by which two balls overlap, 0 if they do not: their radii less the distance
/// between their centres.
pub fn depth<G: Geometry>(a: (&G::Point, Real), b: (&G::Point, Real)) -> Real {
    (a.1 + b.1 - G::distance(a.0, b.0)).max(0.0)
}

#[cfg(test)]
mod tests {
    use fk_geometry_euclidean::E3;
    use fk_math::nalgebra::{Point3, Vector3};

    use super::*;

    fn ball(x: Real, vx: Real, radius: Real, inverse_mass: Real) -> MovingBall<E3> {
        MovingBall {
            centre: Point3::new(x, 0.0, 0.0),
            velocity: Vector3::new(vx, 0.0, 0.0),
            radius,
            inverse_mass,
        }
    }

    #[test]
    fn equal_balls_meeting_head_on_swap_their_velocities() {
        let (mut a, mut b) = (ball(0.0, 1.0, 1.0, 1.0), ball(1.9, -1.0, 1.0, 1.0));
        let resolved = resolve(&mut a, &mut b, 1.0).unwrap();
        assert!((resolved.depth - 0.1).abs() < 1e-12);
        assert!((a.velocity - Vector3::new(-1.0, 0.0, 0.0)).norm() < 1e-12);
        assert!((b.velocity - Vector3::new(1.0, 0.0, 0.0)).norm() < 1e-12);
        // Apart, each by half the depth: just touching.
        assert!((a.centre.x + 0.05).abs() < 1e-12 && (b.centre.x - 1.95).abs() < 1e-12);
        assert!(depth::<E3>((&a.centre, a.radius), (&b.centre, b.radius)) < 1e-12);
    }

    #[test]
    fn momentum_is_kept_and_energy_lost_without_restitution() {
        let (mut a, mut b) = (ball(0.0, 3.0, 0.5, 0.5), ball(0.9, 0.0, 0.5, 1.0));
        let momentum = |a: &MovingBall<E3>, b: &MovingBall<E3>| {
            a.velocity / a.inverse_mass + b.velocity / b.inverse_mass
        };
        let before = momentum(&a, &b);
        resolve(&mut a, &mut b, 0.0).unwrap();
        assert!((momentum(&a, &b) - before).norm() < 1e-12);
        assert!(
            (a.velocity - b.velocity).norm() < 1e-12,
            "stuck along the normal"
        );
    }

    #[test]
    fn a_ball_bounces_off_one_nothing_moves() {
        let mut wall = ball(2.0, 0.0, 1.0, 0.0);
        let mut a = ball(0.0, 2.0, 1.2, 1.0);
        let resolved = resolve(&mut a, &mut wall, 0.5).unwrap();
        assert!((resolved.depth - 0.2).abs() < 1e-12);
        assert!((a.velocity.x + 1.0).abs() < 1e-12, "back at half the speed");
        assert_eq!(wall.velocity, Vector3::zeros());
        assert_eq!(wall.centre, Point3::new(2.0, 0.0, 0.0), "the wall stays");
        assert!((a.centre.x + 0.2).abs() < 1e-12, "a takes all of the depth");
    }

    #[test]
    fn parting_balls_are_separated_without_an_impulse() {
        let (mut a, mut b) = (ball(0.0, -1.0, 1.0, 1.0), ball(1.5, 1.0, 1.0, 1.0));
        let resolved = resolve(&mut a, &mut b, 1.0).unwrap();
        assert_eq!(resolved.impulse, 0.0);
        assert_eq!(a.velocity, Vector3::new(-1.0, 0.0, 0.0));
        assert!(depth::<E3>((&a.centre, a.radius), (&b.centre, b.radius)) < 1e-12);
    }

    #[test]
    fn apart_there_is_no_contact() {
        let (mut a, mut b) = (ball(0.0, 1.0, 1.0, 1.0), ball(2.5, 0.0, 1.0, 1.0));
        assert!(resolve(&mut a, &mut b, 1.0).is_none());
    }

    #[test]
    fn a_glancing_blow_turns_only_the_part_along_the_normal() {
        let mut a = MovingBall::<E3> {
            centre: Point3::origin(),
            velocity: Vector3::new(1.0, 1.0, 0.0),
            radius: 1.0,
            inverse_mass: 1.0,
        };
        let mut wall = ball(1.8, 0.0, 1.0, 0.0);
        resolve(&mut a, &mut wall, 1.0).unwrap();
        assert!((a.velocity - Vector3::new(-1.0, 1.0, 0.0)).norm() < 1e-12);
    }
}
