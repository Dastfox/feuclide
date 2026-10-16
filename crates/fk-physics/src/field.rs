//! Bodies among surfaces given by a distance function: any shape a game can measure, with no
//! mesh and no closed form.

use fk_geometry::{Geometry, GroupElement, VectorSpace};
use fk_math::Real;
use fk_math::nalgebra::Vector3;

use crate::{CharacterShape, tangent};

/// Advancement steps a move takes at most before it stops where it got to.
const STEPS: usize = 48;

/// A signed distance to a surface: positive outside, in units of the reference frame. It only
/// needs to be a lower bound outside (never more than the true distance), as a ray marcher's
/// field is.
///
/// Conventions: distances are in the reference frame's own units, exact in E³ and approximate in
/// curved space (measured in the normal coordinates of that frame), negative inside.
pub trait DistanceField<G: Geometry> {
    /// Distance from `p` to the surface.
    fn distance(&self, p: &G::Point) -> Real;
}

impl<G: Geometry, F: Fn(&G::Point) -> Real> DistanceField<G> for F {
    fn distance(&self, p: &G::Point) -> Real {
        self(p)
    }
}

/// An upright capsule ([`CharacterShape`]) at `pose` (its feet), moved through a
/// [`DistanceField`]: it keeps more than its radius from the surface and slides along it.
///
/// The capsule is measured as a row of balls along its axis, in the normal coordinates of the
/// body's frame (exact in flat space, an approximation in curved space). Scale the shape with
/// any zoom of the field: the field is in the reference frame's units, so a capsule in metres
/// against a field in metres shrinks with the field's zoom by itself.
#[derive(Clone, Copy, Debug)]
pub struct FieldBody<'a, G: Geometry> {
    /// Where its feet are; frame vector 1 is up.
    pub pose: &'a G::Isometry,
    /// Its size.
    pub shape: CharacterShape,
    /// How far beyond its radius it keeps from the surface.
    pub skin: Real,
}

impl<G: Geometry> FieldBody<'_, G> {
    /// Centres of the balls along the axis, in body coordinates, each `shape.radius` across.
    fn balls(&self) -> impl Iterator<Item = Vector3<Real>> + use<G> {
        let r = self.shape.radius;
        let length = (self.shape.height - 2.0 * r).max(0.0);
        let count = (length / r).ceil().max(1.0) as usize + 1;
        (0..count)
            .map(move |k| Vector3::y() * (r + length * k as Real / (count - 1).max(1) as Real))
    }

    /// The body point at body coordinates `v`.
    fn at(&self, v: &Vector3<Real>) -> G::Point {
        G::apply(self.pose, &G::exp(&G::origin(), &tangent::<G>(v)))
    }

    /// How much room the body has with its feet moved by `offset` (body coordinates): the
    /// least distance of its balls from the surface, less their radius.
    pub fn room(&self, field: &(impl DistanceField<G> + ?Sized), offset: &Vector3<Real>) -> Real {
        self.balls()
            .map(|c| field.distance(&self.at(&(c + offset))))
            .fold(Real::INFINITY, Real::min)
            - self.shape.radius
    }

    /// The direction, in body coordinates, in which the room grows fastest at `offset`: six
    /// samples, a step either way along each axis.
    fn away(
        &self,
        field: &(impl DistanceField<G> + ?Sized),
        offset: &Vector3<Real>,
    ) -> Vector3<Real> {
        let h = 0.5 * self.shape.radius;
        let gradient = Vector3::from_fn(|i, _| {
            let mut e = Vector3::zeros();
            e[i] = h;
            self.room(field, &(offset + e)) - self.room(field, &(offset - e))
        });
        gradient.try_normalize(1e-300).unwrap_or_else(Vector3::y)
    }

    /// How far the feet can go along `motion` (body coordinates): the motion advanced through
    /// the field in steps no longer than the room around the body, slid along the surface
    /// where it meets it, and the body pushed back out first if it is already too close.
    pub fn slide(
        &self,
        field: &(impl DistanceField<G> + ?Sized),
        motion: &Vector3<Real>,
    ) -> Vector3<Real> {
        let skin = self.skin.max(1e-9);
        let mut moved = Vector3::zeros();
        let mut left = *motion;
        for _ in 0..STEPS {
            let room = self.room(field, &moved);
            let mut away = None;
            if room < skin {
                // Against the surface: out to the skin, and only along it from here.
                let normal = self.away(field, &moved);
                moved += normal * (skin - room);
                let into = left.dot(&normal);
                if into < 0.0 {
                    left -= normal * into;
                }
                away = Some(normal);
            }
            let length = left.norm();
            if length < 1e-12 {
                break;
            }
            // The room bounds the step towards the surface; near it, a step along the surface
            // only closes in as fast as it heads into it.
            let safe = (room - 0.5 * skin).max(0.25 * skin);
            let limit = if room < 4.0 * skin {
                // The way out found as it was pushed out, else here.
                let normal = away.unwrap_or_else(|| self.away(field, &moved));
                let approach = -(left / length).dot(&normal);
                safe / approach.max(0.05)
            } else {
                safe
            };
            let step = length.min(limit);
            moved += left * (step / length);
            left *= 1.0 - step / length;
        }
        moved
    }
}

/// `pose` moved by `motion` in its own frame, through `field`: see [`FieldBody::slide`].
pub fn move_through<G: Geometry>(
    pose: &G::Isometry,
    shape: CharacterShape,
    skin: Real,
    field: &(impl DistanceField<G> + ?Sized),
    motion: &Vector3<Real>,
) -> G::Isometry {
    let body = FieldBody { pose, shape, skin };
    let moved = body.slide(field, motion);
    let mut next = pose.compose(&G::Isometry::exp(&G::transvection(&tangent::<G>(&moved))));
    next.renormalize();
    next
}

/// How far along the geodesic leaving `from` along `direction` (a unit tangent at `from`) the
/// surface of `field` is, if it is within `reach`: sphere tracing, each step as long as the
/// field's distance, a hit within `hit` of the surface, at most `steps` steps. The raycast of a
/// distance field (where a look falls, what is under a point, whether a light reaches it).
pub fn cast<G: Geometry>(
    field: &(impl DistanceField<G> + ?Sized),
    from: &G::Point,
    direction: &G::Tangent,
    reach: Real,
    (hit, steps): (Real, usize),
) -> Option<Real> {
    let mut t = 0.0;
    for _ in 0..steps {
        let d = field.distance(&G::exp(from, &(*direction * t)));
        if d < hit {
            return Some(t);
        }
        t += d;
        if t > reach {
            return None;
        }
    }
    None
}

/// The reference frame carried to `p` from the origin.
fn frame_at<G: Geometry>(p: &G::Point) -> [G::Tangent; 3] {
    let origin = G::origin();
    std::array::from_fn(|i| G::parallel_transport(&origin, p, &G::origin_frame(i)))
}

/// The unit normal of `field`'s surface at `p` (a tangent at `p`, pointing out), from its
/// gradient by central differences `h` apart along the reference frame carried there; `None`
/// where the field is flat.
pub fn normal<G: Geometry>(
    field: &(impl DistanceField<G> + ?Sized),
    p: &G::Point,
    h: Real,
) -> Option<G::Tangent> {
    let frame = frame_at::<G>(p);
    let slope = frame.iter().fold(G::Tangent::zero(), |sum, &e| {
        let ahead = field.distance(&G::exp(p, &(e * h)));
        let behind = field.distance(&G::exp(p, &(e * -h)));
        sum + e * (ahead - behind)
    });
    let length = G::inner(p, &slope, &slope).sqrt();
    (length > 1e-12).then(|| slope * (1.0 / length))
}

/// How `field`'s surface bends at `p` along the geodesic leaving it along `direction` (a unit
/// tangent at `p`, along the surface): the field's second derivative there, by central
/// differences `h` apart. Near the surface it is its curvature along that direction, in
/// inverse units of distance: positive where it bulges out (a cylinder's side across its axis:
/// one over its radius), 0 flat.
pub fn curvature<G: Geometry>(
    field: &(impl DistanceField<G> + ?Sized),
    p: &G::Point,
    direction: &G::Tangent,
    h: Real,
) -> Real {
    let ahead = field.distance(&G::exp(p, &(*direction * h)));
    let behind = field.distance(&G::exp(p, &(*direction * -h)));
    (ahead + behind - 2.0 * field.distance(p)) / (h * h)
}

#[cfg(test)]
mod tests {
    use fk_geometry_euclidean::{E3, Se3};
    use fk_math::nalgebra::Point3;

    use super::*;

    fn at(x: Real, y: Real, z: Real) -> Se3 {
        Se3::exp(&E3::transvection(&Vector3::new(x, y, z)))
    }

    const WALKER: CharacterShape = CharacterShape {
        radius: 0.3,
        height: 1.8,
    };

    /// A wall across `z = -10`, solid beyond it.
    fn wall(p: &Point3<Real>) -> Real {
        p.z + 10.0
    }

    #[test]
    fn a_ray_finds_the_wall_its_normal_and_a_ball_bends() {
        let origin = Point3::origin();
        let ahead = cast::<E3>(&wall, &origin, &-Vector3::z(), 50.0, (1e-4, 64)).unwrap();
        assert!((ahead - 10.0).abs() < 1e-3, "{ahead}");
        assert!(cast::<E3>(&wall, &origin, &Vector3::z(), 50.0, (1e-4, 64)).is_none());
        let up = normal::<E3>(&wall, &Point3::new(1.0, 2.0, -10.0), 1e-3).unwrap();
        assert!((up - Vector3::z()).norm() < 1e-9, "{up}");
        let ball = |p: &Point3<Real>| p.coords.norm() - 2.0;
        let on = Point3::new(0.0, 0.0, 2.0);
        let bend = curvature::<E3>(&ball, &on, &Vector3::x(), 1e-3);
        assert!((bend - 0.5).abs() < 1e-3, "one over the radius: {bend}");
    }

    #[test]
    fn it_stops_short_of_a_wall() {
        let pose = at(0.0, 0.0, 0.0);
        let next = move_through::<E3>(&pose, WALKER, 0.01, &wall, &Vector3::new(0.0, 0.0, -50.0));
        let z = next.translation.z;
        assert!(z > -10.0 + 0.3 && z < -10.0 + 0.3 + 0.02, "{z}");
    }

    #[test]
    fn it_slides_along_a_wall() {
        let pose = at(0.0, 0.0, -9.0);
        let next = move_through::<E3>(&pose, WALKER, 0.01, &wall, &Vector3::new(4.0, 0.0, -4.0));
        let t = next.translation;
        assert!(
            (t.x - 4.0).abs() < 0.05,
            "kept the part along the wall: {t}"
        );
        assert!(t.z > -10.0 + 0.3, "{t}");
    }

    #[test]
    fn it_is_pushed_out_of_a_ball_it_overlaps() {
        let ball = |p: &Point3<Real>| (p - Point3::new(0.0, 1.0, -0.5)).norm() - 0.5;
        let pose = at(0.0, 0.0, 0.0);
        let next = move_through::<E3>(&pose, WALKER, 0.01, &ball, &Vector3::zeros());
        let body = FieldBody::<E3> {
            pose: &next,
            shape: WALKER,
            skin: 0.01,
        };
        assert!(body.room(&ball, &Vector3::zeros()) > 0.0);
    }

    #[test]
    fn a_free_move_goes_all_the_way() {
        let far = |_: &Point3<Real>| 1e6;
        let next = move_through::<E3>(
            &at(1.0, 2.0, 3.0),
            WALKER,
            0.01,
            &far,
            &Vector3::new(5.0, -1.0, 2.0),
        );
        assert!((next.translation - Vector3::new(6.0, 1.0, 5.0)).norm() < 1e-9);
    }
}
