//! Standing on a flat ground: upright poses by a yaw, and the free spot nearest a point for a
//! body of a [`CharacterShape`] among capsule colliders and a distance field (where to spawn,
//! respawn or land a body without it being inside anything).

use std::f64::consts::PI;

use fk_geometry::{Geometry, GroupElement};
use fk_geometry_euclidean::{E3, Se3};
use fk_math::Real;
use fk_math::nalgebra::{Point3, Vector2, Vector3};
use fk_physics::shapes::{Capsule, capsule_capsule};
use fk_physics::{CharacterShape, Colliders, DistanceField};

/// Feet standing upright at `(x, z)` on the ground (y 0), facing `yaw` (0 along −z, positive to
/// the right).
pub fn standing(at: &Vector2<Real>, yaw: Real) -> Se3 {
    Se3::exp(&E3::transvection(&Vector3::new(at.x, 0.0, at.y))).compose(&Se3::exp(
        &(E3::rotation(&-Vector3::z(), &Vector3::x()) * yaw),
    ))
}

/// The yaw of `pose`'s facing, read off its `−z` axis: see [`standing`].
pub fn yaw_of(pose: &Se3) -> Real {
    let ahead = E3::apply_tangent(pose, &-Vector3::z());
    ahead.x.atan2(-ahead.z)
}

/// Whether a body of `shape` standing at `feet` on the ground overlaps any of `obstacles`.
pub fn blocked(obstacles: &Colliders<E3>, feet: &Vector2<Real>, shape: CharacterShape) -> bool {
    let r = shape.radius;
    let body = Capsule {
        a: Vector3::new(feet.x, r, feet.y),
        b: Vector3::new(feet.x, (shape.height - r).max(r), feet.y),
        radius: r,
    };
    let at = Point3::new(feet.x, 0.0, feet.y);
    obstacles.near(&at, shape.height + r).any(|collider| {
        let obstacle = Capsule {
            a: collider.a.coords,
            b: collider.b.coords,
            radius: collider.radius,
        };
        capsule_capsule(&body, &obstacle).distance < 0.0
    })
}

/// The spots tried round `feet`, nearest first: a spiral out from it in steps of a third of
/// `shape`'s radius.
fn spiral(feet: Vector2<Real>, shape: CharacterShape) -> impl Iterator<Item = Vector2<Real>> {
    let step = shape.radius / 3.0;
    let golden = PI * (3.0 - 5.0_f64.sqrt());
    (0..20_000).map(move |k| {
        let k = Real::from(k);
        let (r, a) = (step * k.sqrt(), golden * k);
        feet + Vector2::new(r * a.cos(), r * a.sin())
    })
}

/// The free spot on the ground nearest `feet`, where a body of `shape` overlaps none of
/// `obstacles`; `feet` itself if none is found.
pub fn free_spot(
    obstacles: &Colliders<E3>,
    feet: &Vector2<Real>,
    shape: CharacterShape,
) -> Vector2<Real> {
    spiral(*feet, shape)
        .find(|spot| !blocked(obstacles, spot, shape))
        .unwrap_or(*feet)
}

/// How far the ground in a field may rise under the feet: a step a body stands on, not a wall
/// it is in.
const STEP: Real = 0.3;

/// Whether a body of `shape` standing at `feet` on the ground is inside `walls` (above a step
/// of 0.3), or, where the flat ground lies below (`ground` under zero, the ground all in the
/// field), has nothing under its feet.
pub fn walled(
    walls: &dyn DistanceField<E3>,
    ground: Real,
    feet: &Vector2<Real>,
    shape: CharacterShape,
) -> bool {
    let r = shape.radius;
    let top = (shape.height - r).max(r);
    let inside = (0..=4).any(|k| {
        let y = STEP + r + (top - r) * Real::from(k) / 4.0;
        walls.distance(&Point3::new(feet.x, y, feet.y)) < r
    });
    let unsupported = ground < -1e-6 && walls.distance(&Point3::new(feet.x, -0.05, feet.y)) > 0.0;
    inside || unsupported
}

/// [`free_spot`] clear of `walls` too, and on ground: see [`walled`].
pub fn free_spot_among(
    obstacles: &Colliders<E3>,
    walls: Option<&dyn DistanceField<E3>>,
    ground: Real,
    feet: &Vector2<Real>,
    shape: CharacterShape,
) -> Vector2<Real> {
    spiral(*feet, shape)
        .find(|spot| {
            !blocked(obstacles, spot, shape)
                && walls.is_none_or(|walls| !walled(walls, ground, spot, shape))
        })
        .unwrap_or(*feet)
}

#[cfg(test)]
mod tests {
    use fk_physics::Collider;

    use super::*;

    const BODY: CharacterShape = CharacterShape {
        radius: 0.3,
        height: 1.8,
    };

    fn towers(at: &[(Real, Real)]) -> Colliders<E3> {
        let mut colliders = Colliders::new(27.0);
        for &(x, z) in at {
            colliders.add(Collider {
                a: Point3::new(x, 0.0, z),
                b: Point3::new(x, 48.0, z),
                radius: 1.8,
            });
        }
        colliders
    }

    #[test]
    fn a_blocked_spot_moves_to_the_nearest_free_one() {
        let obstacles = towers(&[(0.0, 0.0)]);
        let inside = Vector2::new(0.5, 0.0);
        assert!(blocked(&obstacles, &inside, BODY));
        let spot = free_spot(&obstacles, &inside, BODY);
        assert!(!blocked(&obstacles, &spot, BODY));
        assert!(spot.norm() < 1.8 + 0.3 + 0.2, "near: {spot}");
        let open = Vector2::new(10.0, 0.0);
        assert_eq!(free_spot(&obstacles, &open, BODY), open);
        let wall = |p: &Point3<Real>| 5.0 - p.x;
        let among = free_spot_among(&obstacles, Some(&wall), 0.0, &Vector2::new(5.0, 0.0), BODY);
        assert!(among.x < 5.0 - 0.3, "out of the wall: {among}");
    }

    #[test]
    fn standing_faces_its_yaw() {
        let pose = standing(&Vector2::new(1.0, 2.0), 0.7);
        assert!((yaw_of(&pose) - 0.7).abs() < 1e-12);
    }
}
