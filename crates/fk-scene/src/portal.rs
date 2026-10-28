use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemParam;
use fk_geometry::{Geometry, GpuGeometry, GroupElement};
use fk_math::Real;
use fk_math::nalgebra::{Matrix3, Vector4};

use crate::Realm;

/// A way from one realm into another (or into itself elsewhere): an opening, a rectangle, at
/// `here` in `from`, through which `to` is seen and walked into as if it stood behind it, its
/// opening at `there`. Global poses, each in its own realm's coordinates; the entity needs no
/// [`Pose`](crate::Pose).
///
/// The opening is the rectangle of half extents `half` (along the frame's `origin_frame(0)` and
/// `origin_frame(1)`) about the frame's origin in its plane `z = 0`, open from its `+z` side:
/// what is seen through `here` from in front of it is what lies on the `−z` side of `there`,
/// carried by [`carry`](Self::carry) `= there ∘ here⁻¹`, an isometry. A doorway into a room is
/// a portal whose `here` stands in the doorway facing out and whose `there` stands in the
/// room's way in, both with `−z` into the room.
///
/// [`both_ways`](Self::both_ways) (the default) opens it back from `to` too, the same rectangle
/// at `there` facing the other way; [`passable`](Self::passable) (the default) lets bodies
/// through ([`Portals::cross`]). The renderer draws what is seen through it in place, to a depth
/// of portals within portals of its own budget; nothing else is needed for it to show.
#[derive(Component)]
pub struct Portal<G: Geometry> {
    /// The realm its first opening is in.
    pub from: Realm,
    /// Its first opening's frame, in `from`.
    pub here: G::Isometry,
    /// The realm seen and walked into through it.
    pub to: Realm,
    /// Where `here` comes out, in `to`.
    pub there: G::Isometry,
    /// The opening's half extents, along its frame's first two axes.
    pub half: [Real; 2],
    /// Whether bodies go through it. A window only, when not.
    pub passable: bool,
    /// Whether it opens back from `to` too.
    pub both_ways: bool,
    /// Whether what is seen through it casts shadows (a shadow map of its own, drawn from the
    /// eye carried through it). `true` by default.
    pub shadows: bool,
    /// How far is seen through it, from the eye carried through, in place of the camera's far.
    /// `None` (the default) for the camera's.
    pub far: Option<Real>,
}

impl<G: Geometry> Portal<G> {
    /// A portal from `here` in `from` to `there` in `to`, `half` its half extents: passable,
    /// both ways, casting shadows, as far as the camera sees.
    pub fn between(
        from: Realm,
        here: G::Isometry,
        to: Realm,
        there: G::Isometry,
        half: [Real; 2],
    ) -> Self {
        Self {
            from,
            here,
            to,
            there,
            half,
            passable: true,
            both_ways: true,
            shadows: true,
            far: None,
        }
    }

    /// The same, a window only: nothing goes through.
    pub fn window(mut self) -> Self {
        self.passable = false;
        self
    }

    /// The same, open from `from` only.
    pub fn one_way(mut self) -> Self {
        self.both_ways = false;
        self
    }

    /// What it carries `from`'s coordinates to `to`'s by: `there ∘ here⁻¹`.
    pub fn carry(&self) -> G::Isometry {
        self.there.compose(&self.here.inverse())
    }

    /// Its openings: `here` into `to`, and back, `there` turned round into `from`, when it opens
    /// both ways.
    pub fn mouths(&self) -> impl Iterator<Item = Mouth<G>> + '_ {
        let carry = self.carry();
        let front = Mouth {
            realm: self.from,
            frame: self.here,
            half: self.half,
            carry,
            into: self.to,
        };
        let back = self.both_ways.then(|| Mouth {
            realm: self.to,
            frame: self.there.compose(&half_turn::<G>()),
            half: self.half,
            carry: carry.inverse(),
            into: self.from,
        });
        std::iter::once(front).chain(back)
    }
}

impl<G: Geometry> Clone for Portal<G> {
    fn clone(&self) -> Self {
        Self {
            from: self.from,
            here: self.here,
            to: self.to,
            there: self.there,
            half: self.half,
            passable: self.passable,
            both_ways: self.both_ways,
            shadows: self.shadows,
            far: self.far,
        }
    }
}

impl<G: Geometry> std::fmt::Debug for Portal<G> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Portal")
            .field("from", &self.from)
            .field("here", &self.here)
            .field("to", &self.to)
            .field("there", &self.there)
            .field("half", &self.half)
            .field("passable", &self.passable)
            .field("both_ways", &self.both_ways)
            .field("shadows", &self.shadows)
            .field("far", &self.far)
            .finish()
    }
}

/// The half turn about the frame's `origin_frame(1)`: its `+z` to `−z`.
fn half_turn<G: Geometry>() -> G::Isometry {
    G::Isometry::exp(
        &(G::rotation(&G::origin_frame(2), &G::origin_frame(0)) * std::f64::consts::PI),
    )
}

/// One opening of a [`Portal`], facing one way: in `realm`, the rectangle of half extents `half`
/// about `frame`'s origin in its plane `z = 0`, open from its `+z` side onto `into`, whose
/// coordinates `carry` takes `realm`'s to.
pub struct Mouth<G: Geometry> {
    /// The realm it is in.
    pub realm: Realm,
    /// Its frame, global in `realm`.
    pub frame: G::Isometry,
    /// Its half extents.
    pub half: [Real; 2],
    /// `realm`'s coordinates to `into`'s.
    pub carry: G::Isometry,
    /// The realm seen through it.
    pub into: Realm,
}

impl<G: Geometry> Clone for Mouth<G> {
    fn clone(&self) -> Self {
        Self {
            realm: self.realm,
            frame: self.frame,
            half: self.half,
            carry: self.carry,
            into: self.into,
        }
    }
}

impl<G: Geometry> std::fmt::Debug for Mouth<G> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mouth")
            .field("realm", &self.realm)
            .field("frame", &self.frame)
            .field("half", &self.half)
            .field("carry", &self.carry)
            .field("into", &self.into)
            .finish()
    }
}

impl<G: GpuGeometry> Mouth<G> {
    /// Its corners, global points in order round it: `(−x, −y)`, `(x, −y)`, `(x, y)`, `(−x, y)`
    /// of its frame, its `+z` side seeing them anticlockwise.
    pub fn corners(&self) -> [G::Point; 4] {
        let [x, y] = self.half;
        [(-x, -y), (x, -y), (x, y), (-x, y)].map(|(a, b)| {
            let v = G::origin_frame(0) * a + G::origin_frame(1) * b;
            G::apply(&self.frame, &G::exp(&G::origin(), &v))
        })
    }

    /// The linear form on embedded global points positive in front of it, its `+z` side.
    pub fn front(&self) -> Vector4<Real> {
        plane_form::<G>(&self.frame)
    }

    /// Whether the way from `from` to `to` (points of its realm, near each other) goes through it
    /// from its front: on its front side before, not after, and through the rectangle between.
    pub fn crossed(&self, from: &G::Point, to: &G::Point) -> bool {
        let front = self.front();
        let (a, b) = (G::embed_point(from), G::embed_point(to));
        let (before, after) = (front.dot(&a), front.dot(&b));
        if !(before > 0.0 && after <= 0.0) {
            return false;
        }
        // Where the way meets its plane, a positive combination of the two ends: in the
        // projective models every geometry is embedded in, the segment is straight.
        let at = b * before - a * after;
        self.edges().iter().all(|edge| edge.dot(&at) >= 0.0)
    }

    /// Forms positive inside each of its four edges, on its plane: the plane through the edge
    /// and a point off the opening, positive on the side of its middle.
    fn edges(&self) -> [Vector4<Real>; 4] {
        let corners = self.corners().map(|corner| G::embed_point(&corner));
        let off = G::embed_point(&G::apply(
            &self.frame,
            &G::exp(&G::origin(), &G::origin_frame(2)),
        ));
        let middle = G::embed_point(&G::apply(&self.frame, &G::origin()));
        std::array::from_fn(|k| {
            let form = form_through(&corners[k], &corners[(k + 1) % 4], &off);
            if form.dot(&middle) < 0.0 { -form } else { form }
        })
    }
}

/// The plane `z = 0` of `frame` (global) as a linear form on embedded global points, positive
/// on the side its `+z` points to: the form reading the frame's own `z`, carried by the matrix
/// `A` of `frame⁻¹` as `Aᵀ`.
///
/// Every geometry is embedded in ℝ⁴ so that its totally geodesic planes are the zero sets of
/// linear forms (the flat, Klein and gnomonic models), which is what makes a plane one vector
/// here, whatever the curvature.
pub fn plane_form<G: GpuGeometry>(frame: &G::Isometry) -> Vector4<Real> {
    let z = G::embed_tangent(&G::origin(), &G::origin_frame(2));
    G::isometry_matrix(&frame.inverse()).transpose() * z
}

/// The linear form vanishing on the embedded points `a`, `b` and `c`: `x ↦ det(x, a, b, c)`,
/// their generalized cross product. The plane through three points; zero if they are on one
/// geodesic.
pub fn form_through(a: &Vector4<Real>, b: &Vector4<Real>, c: &Vector4<Real>) -> Vector4<Real> {
    let minor = |skip: usize| {
        let rows: Vec<usize> = (0..4).filter(|&i| i != skip).collect();
        Matrix3::from_fn(|i, j| [a, b, c][j][rows[i]]).determinant()
    };
    Vector4::new(minor(0), -minor(1), minor(2), -minor(3))
}

/// Where a body crossing a portal comes out: the realm, and what carries its pose there
/// (`pose ← carry ∘ pose`).
pub struct Crossing<G: Geometry> {
    /// The realm it is in after.
    pub realm: Realm,
    /// Its old realm's coordinates to the new one's.
    pub carry: G::Isometry,
}

impl<G: Geometry> std::fmt::Debug for Crossing<G> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Crossing")
            .field("realm", &self.realm)
            .field("carry", &self.carry)
            .finish()
    }
}

/// The first passable opening among `portals`' in `realm` the way from `from` to `to` goes
/// through, and where it comes out.
pub fn cross<'a, G: GpuGeometry>(
    portals: impl IntoIterator<Item = &'a Portal<G>>,
    realm: Realm,
    from: &G::Point,
    to: &G::Point,
) -> Option<Crossing<G>> {
    portals
        .into_iter()
        .filter(|portal| portal.passable)
        .flat_map(Portal::mouths)
        .find(|mouth| mouth.realm == realm && mouth.crossed(from, to))
        .map(|mouth| Crossing {
            realm: mouth.into,
            carry: mouth.carry,
        })
}

/// Every [`Portal`], for whoever moves a body: ask [`cross`](Self::cross) after each step.
#[derive(SystemParam)]
pub struct Portals<'w, 's, G: Geometry> {
    portals: Query<'w, 's, &'static Portal<G>>,
}

impl<G: GpuGeometry> Portals<'_, '_, G> {
    /// The first passable opening in `realm` the way from `from` to `to` goes through, and
    /// where it comes out: the mover carries the body's pose by it and writes the new
    /// [`Realm`] on the body.
    pub fn cross(&self, realm: Realm, from: &G::Point, to: &G::Point) -> Option<Crossing<G>> {
        cross(self.portals.iter(), realm, from, to)
    }

    /// Every portal.
    pub fn iter(&self) -> impl Iterator<Item = &Portal<G>> {
        self.portals.iter()
    }
}

#[cfg(test)]
mod tests {
    use fk_geometry_euclidean::{E3, Se3};
    use fk_geometry_hyperbolic::H3;
    use fk_math::nalgebra::{Point3, Vector3};

    use super::*;

    fn at<G: Geometry>(x: Real, y: Real, z: Real) -> G::Point {
        let v = G::origin_frame(0) * x + G::origin_frame(1) * y + G::origin_frame(2) * z;
        G::exp(&G::origin(), &v)
    }

    fn shift<G: Geometry>(x: Real, y: Real, z: Real) -> G::Isometry {
        let v = G::origin_frame(0) * x + G::origin_frame(1) * y + G::origin_frame(2) * z;
        G::Isometry::exp(&G::transvection(&v))
    }

    /// A door 2 wide and 2 high at the origin of the main realm, facing +z, into a room realm
    /// whose way in stands 10 along x.
    fn door<G: Geometry>() -> Portal<G> {
        Portal::between(
            Realm::MAIN,
            G::Isometry::identity(),
            Realm(1),
            shift::<G>(10.0, 0.0, 0.0),
            [1.0, 1.0],
        )
    }

    fn walks_through<G: GpuGeometry>() {
        let portal = door::<G>();
        let crossing = cross(
            [&portal],
            Realm::MAIN,
            &at::<G>(0.2, 0.1, 0.3),
            &at::<G>(0.2, 0.1, -0.3),
        )
        .expect("through the opening");
        assert_eq!(crossing.realm, Realm(1));
        // Beside the opening, or from behind it, or in the wrong realm: nothing.
        assert!(
            cross(
                [&portal],
                Realm::MAIN,
                &at::<G>(1.5, 0.0, 0.3),
                &at::<G>(1.5, 0.0, -0.3)
            )
            .is_none()
        );
        assert!(
            cross(
                [&portal],
                Realm::MAIN,
                &at::<G>(0.0, 0.0, -0.3),
                &at::<G>(0.0, 0.0, 0.3)
            )
            .is_none()
        );
        assert!(
            cross(
                [&portal],
                Realm(1),
                &at::<G>(0.0, 0.0, 0.3),
                &at::<G>(0.0, 0.0, -0.3)
            )
            .is_none()
        );
        // Back out of the room: from its −z side through the way in, into the main realm.
        let back = cross(
            [&portal],
            Realm(1),
            &G::apply(&shift::<G>(10.0, 0.0, 0.0), &at::<G>(0.0, 0.0, -0.3)),
            &G::apply(&shift::<G>(10.0, 0.0, 0.0), &at::<G>(0.0, 0.0, 0.3)),
        )
        .expect("back through");
        assert_eq!(back.realm, Realm::MAIN);
        let out = G::apply(
            &back.carry,
            &G::apply(&shift::<G>(10.0, 0.0, 0.0), &at::<G>(0.0, 0.0, 0.3)),
        );
        assert!(G::distance(&out, &at::<G>(0.0, 0.0, 0.3)) < 1e-9);
        // A window lets nothing through.
        let window = door::<G>().window();
        assert!(
            cross(
                [&window],
                Realm::MAIN,
                &at::<G>(0.0, 0.0, 0.3),
                &at::<G>(0.0, 0.0, -0.3)
            )
            .is_none()
        );
    }

    #[test]
    fn bodies_go_through_the_opening_from_its_front_only_in_flat_space() {
        walks_through::<E3>();
    }

    #[test]
    fn bodies_go_through_the_opening_from_its_front_only_in_hyperbolic_space() {
        walks_through::<H3>();
    }

    #[test]
    fn the_carry_takes_the_opening_to_its_way_out() {
        let portal = door::<E3>();
        let through = E3::apply(&portal.carry(), &Point3::new(0.5, 0.0, -1.0));
        assert!((through - Point3::new(10.5, 0.0, -1.0)).norm() < 1e-12);
        let back = portal.mouths().nth(1).expect("both ways");
        // Turned round: its front is the room's side.
        assert!(
            back.front()
                .dot(&E3::embed_point(&Point3::new(10.0, 0.0, -1.0)))
                > 0.0
        );
        let _: Se3 = back.frame;
    }

    #[test]
    fn the_form_through_three_points_vanishes_on_their_plane() {
        let [a, b, c] =
            [Vector3::x(), Vector3::y(), Vector3::z()].map(|v| E3::embed_point(&Point3::from(v)));
        let form = form_through(&a, &b, &c);
        let on = E3::embed_point(&Point3::new(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0));
        assert!(form.dot(&on).abs() < 1e-12);
        assert!(form.dot(&E3::embed_point(&Point3::origin())).abs() > 0.1);
    }
}
