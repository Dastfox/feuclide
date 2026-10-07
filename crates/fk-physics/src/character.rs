//! A walking character: an upright capsule that walks, runs, jumps, falls, lands and slides
//! along what it bumps into.

use fk_geometry::{Geometry, GroupElement, VectorSpace};
use fk_math::Real;
use fk_math::nalgebra::Vector3;

use std::sync::OnceLock;

use fk_quotient::Quotient;

use crate::bvh::BallTree;
use crate::field::{DistanceField, FieldBody};
use crate::shapes::{Capsule, capsule_capsule};
use crate::{Gravity, coords, tangent};

/// Times a step pushes the character out of what it overlaps before it gives up.
const PUSHES: usize = 4;

/// Feet this close above the ground stand on it.
const GROUND_SNAP: Real = 1e-6;

/// The ground: the flat surface through the origin across `up`, raised by `level`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ground {
    /// Unit normal, in the reference frame at the origin.
    pub up: Vector3<Real>,
    /// Height of the surface above the origin along `up`.
    pub level: Real,
}

impl Default for Ground {
    /// The ground plane of the reference frame: frame vector 1 up, through the origin.
    fn default() -> Self {
        Self {
            up: Vector3::y(),
            level: 0.0,
        }
    }
}

impl Ground {
    /// Height of `p` above the ground, negative below it, measured along the normal coordinates
    /// at the origin (exact in flat space). Infinite where `p` is on the origin's cut locus.
    pub fn height<G: Geometry>(&self, p: &G::Point) -> Real {
        G::log(&G::origin(), p).map_or(Real::INFINITY, |v| {
            coords::<G>(&v).dot(&self.up) - self.level
        })
    }
}

/// Something to bump into: every point within `radius` of the geodesic from `a` to `b`.
#[derive(Debug)]
pub struct Collider<G: Geometry> {
    /// One end of the axis, in the reference frame; for things standing on the ground, the
    /// foot.
    pub a: G::Point,
    /// The other end.
    pub b: G::Point,
    /// Its radius.
    pub radius: Real,
}

impl<G: Geometry> Clone for Collider<G> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<G: Geometry> Copy for Collider<G> {}

/// Every [`Collider`], each covered by geodesic balls along its axis, filed in a [`BallTree`]
/// built when they are first looked among after a change.
///
/// In a quotient space ([`in_quotient`](Self::in_quotient)) the colliders are taken to lie in
/// the fundamental domain, and their copies across its faces are filed too (the Γ-ghosts of
/// [`ghosts`](crate::ghosts)): [`near`](Self::near) then gives back the copies within reach as
/// colliders of their own, moved by their deck element, so a body just this side of a face
/// bumps into what stands just across it.
#[derive(Debug)]
pub struct Colliders<G: Geometry> {
    colliders: Vec<Collider<G>>,
    /// The quotient, the longest deck words copied by and how far beyond the domain's faces
    /// copies are kept.
    quotient: Option<(Quotient<G>, usize, Real)>,
    /// The tree over the balls that cover them and their copies, and for each ball the
    /// collider it covers and the deck element that copies it (`None` for the collider itself).
    tree: OnceLock<Filed<G>>,
}

/// The tree of [`Colliders`]: see its `tree` field.
#[derive(Debug)]
struct Filed<G: Geometry> {
    tree: BallTree<G>,
    of: Vec<(u32, Option<u32>)>,
    decks: Vec<G::Isometry>,
}

impl<G: Geometry> Colliders<G> {
    /// None yet. `cell` was the bucket grid's size, before the tree; it is not needed any more.
    pub fn new(cell: Real) -> Self {
        let _ = cell;
        Self {
            colliders: Vec::new(),
            quotient: None,
            tree: OnceLock::new(),
        }
    }

    /// Makes them colliders of the quotient `quotient`: copies of each by the deck elements
    /// with words of at most `length` letters are filed too, those that may come within
    /// `reach` of the fundamental domain (the farthest any body looks beyond a face).
    pub fn in_quotient(&mut self, quotient: Quotient<G>, length: usize, reach: Real) {
        self.quotient = Some((quotient, length, reach));
        self.tree = OnceLock::new();
    }

    /// Adds one.
    pub fn add(&mut self, collider: Collider<G>) {
        u32::try_from(self.colliders.len()).expect("too many colliders");
        self.colliders.push(collider);
        self.tree = OnceLock::new();
    }

    /// Most balls covering one collider.
    const MOST_PIECES: usize = 64;

    /// Balls that cover `collider` together: its axis cut into pieces no longer than its
    /// diameter, each held by the ball about its middle, so a tall tower is not one huge ball.
    fn cover(collider: &Collider<G>, mut ball: impl FnMut(G::Point, Real)) {
        let Some(axis) = G::log(&collider.a, &collider.b) else {
            return ball(collider.a, G::injectivity_radius() + collider.radius);
        };
        let length = G::norm(&collider.a, &axis);
        let pieces = ((length / (2.0 * collider.radius.max(0.25))).ceil() as usize)
            .clamp(1, Self::MOST_PIECES);
        let piece = 1.0 / pieces as Real;
        for i in 0..pieces {
            let middle = G::exp(&collider.a, &(axis * ((i as Real + 0.5) * piece)));
            ball(middle, length * piece / 2.0 + collider.radius);
        }
    }

    fn tree(&self) -> &Filed<G> {
        self.tree.get_or_init(|| {
            let (mut covers, mut of) = (Vec::new(), Vec::new());
            for (id, collider) in (0u32..).zip(&self.colliders) {
                Self::cover(collider, |centre, radius| {
                    covers.push((centre, radius));
                    of.push((id, None));
                });
            }
            let mut decks = Vec::new();
            if let Some((quotient, length, reach)) = &self.quotient {
                let faces: Vec<_> = quotient.faces().collect();
                // The identity is the first translate: the colliders themselves, filed above.
                for translate in quotient.translates(*length).into_iter().skip(1) {
                    let deck = u32::try_from(decks.len()).expect("too many deck elements");
                    for i in 0..of.len().min(covers.len()) {
                        let (centre, radius) = covers[i];
                        if of[i].1.is_some() {
                            break;
                        }
                        let copy = G::apply(&translate.isometry, &centre);
                        if near_domain(quotient, &faces, &copy, radius + reach) {
                            covers.push((copy, radius));
                            of.push((of[i].0, Some(deck)));
                        }
                    }
                    decks.push(translate.isometry);
                }
            }
            Filed {
                tree: BallTree::build(&covers),
                of,
                decks,
            }
        })
    }

    /// How many there are.
    pub fn len(&self) -> usize {
        self.colliders.len()
    }

    /// Whether there are none.
    pub fn is_empty(&self) -> bool {
        self.colliders.is_empty()
    }

    /// Every collider that may come within `radius` of `at`, and in a quotient every copy of
    /// one across the domain's faces, moved by its deck element.
    pub fn near(
        &self,
        at: &G::Point,
        radius: Real,
    ) -> impl Iterator<Item = Collider<G>> + use<'_, G> {
        let Filed { tree, of, decks } = self.tree();
        let mut found = Vec::new();
        tree.near(at, radius, |ball| found.push(of[ball as usize]));
        found.sort_unstable();
        found.dedup();
        found.into_iter().map(|(id, deck)| {
            let collider = self.colliders[id as usize];
            match deck {
                None => collider,
                Some(deck) => {
                    let deck = &decks[deck as usize];
                    Collider {
                        a: G::apply(deck, &collider.a),
                        b: G::apply(deck, &collider.b),
                        radius: collider.radius,
                    }
                }
            }
        })
    }
}

/// Whether `p` may come within `reach` of the fundamental domain of `quotient`: for no face is
/// it surely farther (half its excess `beyond` is at most its distance past the face, by the
/// triangle inequality through the face's nearest point).
fn near_domain<G: Geometry>(
    quotient: &Quotient<G>,
    faces: &[fk_quotient::Letter],
    p: &G::Point,
    reach: Real,
) -> bool {
    faces
        .iter()
        .all(|&face| quotient.beyond(p, face) <= 2.0 * reach)
}

/// What a character moves through.
pub struct Surroundings<'a, G: Geometry> {
    /// What pulls it down.
    pub gravity: &'a Gravity<G>,
    /// What it stands on.
    pub ground: &'a Ground,
    /// What it bumps into.
    pub colliders: &'a Colliders<G>,
    /// Any other shape it bumps into, given by its distance: walls, doors, whatever has no
    /// closed form. `None` for none.
    pub field: Option<&'a dyn DistanceField<G>>,
}

impl<G: Geometry + std::fmt::Debug> std::fmt::Debug for Surroundings<'_, G> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Surroundings")
            .field("gravity", self.gravity)
            .field("ground", self.ground)
            .field("colliders", self.colliders)
            .field("field", &self.field.is_some())
            .finish()
    }
}

/// How far beyond its radius a character keeps from a [`Surroundings::field`].
const FIELD_SKIN: Real = 0.01;

/// A [`Surroundings::field`] that stops the motion from below at least this much (the cosine
/// between what it stopped and straight down) is a floor the character stands on.
const FIELD_FLOOR: Real = 0.7;

/// How far the steepest floor ([`FIELD_FLOOR`]) drops for each unit across: the tangent of
/// its slope.
const STEEPEST: Real = 1.03;

/// The body of a character: an upright capsule standing on its feet.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CharacterShape {
    /// Its radius.
    pub radius: Real,
    /// From the feet to the top of the head.
    pub height: Real,
}

impl CharacterShape {
    /// The capsule of a body whose feet are at `feet`, upright along `up`, in body coordinates.
    fn capsule(&self, feet: &Vector3<Real>, up: &Vector3<Real>) -> Capsule {
        let r = self.radius;
        Capsule {
            a: feet + up * r,
            b: feet + up * (self.height - r).max(r),
            radius: r,
        }
    }
}

/// A character that walks on a [`Ground`] among [`Colliders`].
///
/// Its pose is its feet: frame vector 1 is up (it stands upright as long as only
/// [`turn`](Self::turn) rotates it), it looks along `−origin_frame(2)`. Every step it moves by
/// `pose ← pose ∘ exp(transvection(δ))`, renormalized, with its velocity in its own frame.
/// The walking speed is set directly, with no inertia, on the ground and in the air alike;
/// what goes up is its own: a jump, and gravity on the way down.
#[derive(Debug)]
pub struct Character<G: Geometry> {
    /// Where its feet are and which way it faces.
    pub pose: G::Isometry,
    /// Its velocity, in its own frame: a tangent vector at the origin.
    pub velocity: G::Tangent,
    /// Its body.
    pub shape: CharacterShape,
    /// Whether it stood on the ground, or on a floor of its [`Surroundings::field`], at the end
    /// of the last step.
    pub on_ground: bool,
}

impl<G: Geometry> Character<G> {
    /// A character standing still at `pose`.
    ///
    /// # Panics
    ///
    /// If `G` is not three-dimensional.
    pub fn new(pose: G::Isometry, shape: CharacterShape) -> Self {
        assert_eq!(G::DIM, 3, "characters walk in three dimensions");
        Self {
            pose,
            velocity: G::Tangent::zero(),
            shape,
            on_ground: false,
        }
    }

    /// Turns about the vertical at the feet by `angle` radians, positive to the right (from
    /// `−origin_frame(2)` towards `origin_frame(0)`). The velocity keeps its direction in the
    /// world.
    pub fn turn(&mut self, angle: Real) {
        let rotation =
            G::Isometry::exp(&(G::rotation(&-G::origin_frame(2), &G::origin_frame(0)) * angle));
        self.pose = self.pose.compose(&rotation);
        self.pose.renormalize();
        self.velocity = G::apply_tangent(&rotation.inverse(), &self.velocity);
    }

    /// Advances `dt` seconds, walking at `walk` (in its own frame: x to the right, −z ahead;
    /// any vertical part is ignored) and jumping up at `jump` units per second if it is on the
    /// ground and `jump` is positive.
    pub fn step(
        &mut self,
        walk: &Vector3<Real>,
        jump: Real,
        around: &Surroundings<'_, G>,
        dt: Real,
    ) {
        let origin = G::origin();
        let feet = G::apply(&self.pose, &origin);
        let inverse = self.pose.inverse();
        let in_body = |v: &G::Tangent| {
            let here = G::parallel_transport(&origin, &feet, v);
            coords::<G>(&G::apply_tangent(&inverse, &here))
        };
        let up = in_body(&tangent::<G>(&around.ground.up)).normalize();
        let level = |w: &Vector3<Real>| w - up * w.dot(&up);

        let mut velocity = coords::<G>(&self.velocity);
        let mut climb = velocity.dot(&up);
        let jumping = self.on_ground && jump > 0.0;
        if jumping {
            climb = jump;
        }
        velocity = level(walk) + up * climb + coords::<G>(&around.gravity.pull(&self.pose, dt));
        let rising = velocity.dot(&up).max(0.0);
        let mut motion = velocity * dt;

        // What it bumps into, in its own coordinates.
        let in_reach = motion.norm() + self.shape.radius + self.shape.height;
        let body_point =
            |p: &G::Point| G::log(&origin, &G::apply(&inverse, p)).map(|v| coords::<G>(&v));
        let near: Vec<Capsule> = around
            .colliders
            .near(&feet, in_reach)
            .filter_map(|collider| {
                Some(Capsule {
                    a: body_point(&collider.a)?,
                    b: body_point(&collider.b)?,
                    radius: collider.radius,
                })
            })
            .collect();
        for _ in 0..PUSHES {
            let mut pushed = false;
            for obstacle in &near {
                let contact = capsule_capsule(&self.shape.capsule(&motion, &up), obstacle);
                if contact.distance < 0.0 {
                    // Out along the normal; what was left of the motion slides along it.
                    motion -= contact.normal * contact.distance;
                    let into = velocity.dot(&contact.normal);
                    if into < 0.0 {
                        velocity -= contact.normal * into;
                    }
                    pushed = true;
                }
            }
            if !pushed {
                break;
            }
        }

        // Shapes given by their distance: the motion slides along them. Stopped mostly from
        // below, it stands on them.
        let mut stands_on_field = false;
        if let Some(field) = around.field {
            let body = FieldBody {
                pose: &self.pose,
                shape: self.shape,
                skin: FIELD_SKIN,
            };
            let wanted = motion;
            motion = body.slide(field, &motion);
            // Whatever the field stopped is taken off the velocity.
            let stopped = wanted - motion;
            if let Some(lost) = stopped.try_normalize(1e-12) {
                let into = velocity.dot(&lost);
                if into > 0.0 {
                    velocity -= lost * into;
                }
                stands_on_field = -lost.dot(&up) > FIELD_FLOOR;
            }
            // Walking down a slope, it keeps to the floor rather than run off it into the air
            // and hop down: on a floor last step and not going up, it steps down onto whatever
            // floor is below where it moved, as far as the steepest floor drops over the stride.
            if self.on_ground && climb <= 0.0 && !stands_on_field {
                let moved = self
                    .pose
                    .compose(&G::Isometry::exp(&G::transvection(&tangent::<G>(&motion))));
                let body = FieldBody {
                    pose: &moved,
                    ..body
                };
                let reach = up * -(level(&motion).norm() * STEEPEST + 2.0 * FIELD_SKIN);
                let down = body.slide(field, &reach);
                let stopped = reach - down;
                if let Some(lost) = stopped.try_normalize(1e-12)
                    && -lost.dot(&up) > FIELD_FLOOR
                {
                    motion += down;
                    stands_on_field = true;
                }
            }
        }

        // Whatever it bumped into may slide it up this step, but never throws it up: running
        // into a bump or a lip too steep to stand on would turn the stride into a leap.
        let thrown = velocity.dot(&up) - rising;
        if thrown > 0.0 {
            velocity -= up * thrown;
        }

        let height = around.ground.height::<G>(&feet) + motion.dot(&up);
        if height < 0.0 {
            motion -= up * height;
        }
        let falling = velocity.dot(&up);
        // Going up is a jump, or the slide of a walk up a floor: walking, it keeps its feet, and
        // tops a rise without being thrown off it.
        self.on_ground = (height <= GROUND_SNAP || stands_on_field) && (falling <= 0.0 || !jumping);
        if self.on_ground {
            velocity -= up * falling;
        }

        self.pose = self
            .pose
            .compose(&G::Isometry::exp(&G::transvection(&tangent::<G>(&motion))));
        self.pose.renormalize();
        self.velocity = tangent::<G>(&velocity);
    }

    /// Where its feet are.
    pub fn feet(&self) -> G::Point {
        G::apply(&self.pose, &G::origin())
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::FRAC_PI_2;

    use fk_geometry_euclidean::{E3, Se3};
    use fk_math::nalgebra::Point3;

    use super::*;

    const DT: Real = 1.0 / 60.0;

    fn at(x: Real, y: Real, z: Real) -> Se3 {
        Se3::exp(&E3::transvection(&Vector3::new(x, y, z)))
    }

    fn henri(pose: Se3) -> Character<E3> {
        Character::new(
            pose,
            CharacterShape {
                radius: 0.3,
                height: 1.8,
            },
        )
    }

    fn tower(x: Real, z: Real) -> Collider<E3> {
        Collider {
            a: Point3::new(x, 0.0, z),
            b: Point3::new(x, 48.0, z),
            radius: 1.8,
        }
    }

    struct Field {
        gravity: Gravity<E3>,
        ground: Ground,
        colliders: Colliders<E3>,
    }

    impl Field {
        fn new(towers: &[(Real, Real)]) -> Self {
            let mut colliders = Colliders::new(27.0);
            for &(x, z) in towers {
                colliders.add(tower(x, z));
            }
            Self {
                gravity: Gravity {
                    down: -Vector3::y(),
                    acceleration: 9.81,
                },
                ground: Ground::default(),
                colliders,
            }
        }

        fn around(&self) -> Surroundings<'_, E3> {
            Surroundings {
                gravity: &self.gravity,
                ground: &self.ground,
                colliders: &self.colliders,
                field: None,
            }
        }

        fn run(&self, body: &mut Character<E3>, walk: Vector3<Real>, seconds: Real) {
            for _ in 0..(seconds / DT).round() as usize {
                body.step(&walk, 0.0, &self.around(), DT);
            }
        }
    }

    #[test]
    fn it_stands_on_the_ground_and_lands_when_dropped() {
        let field = Field::new(&[]);
        let mut body = henri(at(0.0, 2.0, 0.0));
        field.run(&mut body, Vector3::zeros(), 0.5);
        assert!(!body.on_ground, "still falling after half a second");
        field.run(&mut body, Vector3::zeros(), 1.0);
        assert!(body.on_ground);
        assert!(body.feet().y.abs() < 1e-9, "{}", body.feet());
        field.run(&mut body, Vector3::zeros(), 1.0);
        assert!(body.on_ground && body.feet().y.abs() < 1e-9, "stays put");
    }

    #[test]
    fn a_jump_goes_up_and_comes_down_in_two_v_over_g() {
        let field = Field::new(&[]);
        let mut body = henri(at(0.0, 0.0, 0.0));
        field.run(&mut body, Vector3::zeros(), 0.1);
        assert!(body.on_ground);
        body.step(&Vector3::zeros(), 5.0, &field.around(), DT);
        let mut airborne = DT;
        let mut top: Real = 0.0;
        while !body.on_ground {
            body.step(&Vector3::zeros(), 5.0, &field.around(), DT);
            airborne += DT;
            top = top.max(body.feet().y);
            assert!(airborne < 5.0, "never lands");
        }
        assert!((airborne - 2.0 * 5.0 / 9.81).abs() < 3.0 * DT, "{airborne}");
        assert!((top - 25.0 / (2.0 * 9.81)).abs() < 0.1, "{top}");
    }

    #[test]
    fn walking_into_a_tower_stops_at_its_side() {
        let field = Field::new(&[(0.0, -10.0)]);
        let mut body = henri(at(0.0, 0.0, 0.0));
        field.run(&mut body, Vector3::new(0.0, 0.0, -4.0), 5.0);
        let feet = body.feet();
        assert!((feet.z - (-10.0 + 1.8 + 0.3)).abs() < 1e-6, "{feet}");
        assert!(feet.x.abs() < 1e-6, "head on, no slide");
    }

    #[test]
    fn walking_at_a_slant_slides_around_it() {
        let field = Field::new(&[(0.5, -10.0)]);
        let mut body = henri(at(0.0, 0.0, 0.0));
        field.run(&mut body, Vector3::new(0.0, 0.0, -4.0), 6.0);
        let feet = body.feet();
        assert!(feet.z < -12.0, "slid past it: {feet}");
        let gap = fk_math::nalgebra::Vector2::new(feet.x - 0.5, feet.z + 10.0).norm();
        assert!(gap >= 2.1 - 1e-6, "{gap}");
    }

    #[test]
    fn turning_turns_the_way_it_walks() {
        let field = Field::new(&[]);
        let mut body = henri(at(0.0, 0.0, 0.0));
        body.turn(FRAC_PI_2);
        field.run(&mut body, Vector3::new(0.0, 0.0, -1.0), 1.0);
        let feet = body.feet();
        assert!((feet.x - 1.0).abs() < 1e-6 && feet.z.abs() < 1e-6, "{feet}");
    }

    #[test]
    fn it_stops_at_a_wall_given_by_its_distance() {
        let field = Field::new(&[]);
        // A wall across z = -6, solid beyond it.
        let wall = |p: &Point3<Real>| p.z + 6.0;
        let around = Surroundings {
            field: Some(&wall),
            ..field.around()
        };
        let mut body = henri(at(0.0, 0.0, 0.0));
        for _ in 0..300 {
            body.step(&Vector3::new(0.0, 0.0, -4.0), 0.0, &around, DT);
        }
        let feet = body.feet();
        assert!(feet.z > -6.0 + 0.3 && feet.z < -6.0 + 0.35, "{feet}");
        assert!(body.on_ground, "still standing on the ground");
    }

    #[test]
    fn it_stands_on_a_floor_given_by_its_distance_and_jumps_off_it() {
        let field = Field::new(&[]);
        // A slab whose top is 1 m above the ground, under x < 5.
        let slab = |p: &Point3<Real>| (p.y - 1.0).max(p.x - 5.0);
        let around = Surroundings {
            field: Some(&slab),
            ..field.around()
        };
        let mut body = henri(at(0.0, 3.0, 0.0));
        for _ in 0..120 {
            body.step(&Vector3::zeros(), 0.0, &around, DT);
        }
        let feet = body.feet();
        assert!(body.on_ground, "stands on it: {feet}");
        assert!(feet.y > 1.0 && feet.y < 1.05, "{feet}");
        body.step(&Vector3::zeros(), 4.0, &around, DT);
        assert!(!body.on_ground && body.feet().y > feet.y, "jumps off it");
        // Walking off its edge, it drops to the ground.
        for _ in 0..240 {
            body.step(&Vector3::new(4.0, 0.0, 0.0), 0.0, &around, DT);
        }
        assert!(
            body.on_ground && body.feet().y.abs() < 1e-9,
            "{}",
            body.feet()
        );
    }

    #[test]
    fn it_walks_and_runs_down_a_slope_on_its_feet_and_still_falls_off_an_edge() {
        let field = Field::new(&[]);
        // A slope dropping 30° along +x from 20 m up, cut off at x = 30 by a sheer drop.
        let (sin, cos) = 30.0_f64.to_radians().sin_cos();
        let slope = |p: &Point3<Real>| (p.y * cos + (p.x * sin) - 20.0 * cos).max(p.x - 30.0);
        let around = Surroundings {
            field: Some(&slope),
            ..field.around()
        };
        for speed in [4.0, 9.0] {
            let mut body = henri(at(0.0, 20.5, 0.0));
            while !body.on_ground {
                body.step(&Vector3::zeros(), 0.0, &around, DT);
            }
            let mut off = 0;
            while body.feet().x < 28.0 {
                body.step(&Vector3::new(speed, 0.0, 0.0), 0.0, &around, DT);
                off += usize::from(!body.on_ground);
            }
            assert_eq!(off, 0, "on its feet all the way down at {speed} m/s");
            for _ in 0..30 {
                body.step(&Vector3::new(speed, 0.0, 0.0), 0.0, &around, DT);
            }
            assert!(!body.on_ground, "off the edge at {speed} m/s");
        }
    }

    #[test]
    fn it_walks_over_rolling_land_meeting_the_ground_on_its_feet() {
        let field = Field::new(&[]);
        // Land rolling gently about the ground's level, now over it and now under it, and rises
        // and crests steep enough to throw it off them at a run.
        let land =
            |p: &Point3<Real>| p.y + 0.02 * (p.x * 0.3).sin() - 0.6 * (p.x * 0.4).sin().max(0.0);
        let around = Surroundings {
            field: Some(&land),
            ..field.around()
        };
        let mut body = henri(at(0.0, 0.5, 0.0));
        for _ in 0..120 {
            body.step(&Vector3::zeros(), 0.0, &around, DT);
        }
        for speed in [4.0, 9.0] {
            let mut off = Vec::new();
            for i in 0..600 {
                body.step(&Vector3::new(speed, 0.0, 0.0), 0.0, &around, DT);
                if !body.on_ground {
                    off.push((i, body.feet().y));
                }
            }
            assert!(
                off.is_empty(),
                "{} frames off at {speed} m/s: {:?}",
                off.len(),
                &off[..off.len().min(10)]
            );
        }
    }

    #[test]
    fn it_never_sinks_into_a_tower_however_it_comes_on() {
        // A regression on penetration depth: a ring of towers close together, run into from
        // every way at a walk, a run and far faster than Henri ever goes; after every step the
        // capsule's axis stays at least the two radii from every tower's.
        let towers: Vec<_> = (0..8)
            .map(|i| {
                let a = Real::from(i) * std::f64::consts::TAU / 8.0;
                (6.0 * a.cos(), 6.0 * a.sin())
            })
            .collect();
        let field = Field::new(&towers);
        for speed in [1.4, 6.0, 40.0] {
            for k in 0..16 {
                let a = Real::from(k) * std::f64::consts::TAU / 16.0 + 0.1;
                let mut body = henri(at(0.0, 0.0, 0.0));
                let walk = Vector3::new(a.cos(), 0.0, a.sin()) * speed;
                for _ in 0..240 {
                    body.step(&walk, 0.0, &field.around(), DT);
                    let feet = body.feet();
                    for &(x, z) in &towers {
                        let gap = fk_math::nalgebra::Vector2::new(feet.x - x, feet.z - z).norm();
                        let depth = 1.8 + 0.3 - gap;
                        assert!(depth < 1e-6, "{depth} deep at {speed} m/s, way {k}");
                    }
                }
            }
        }
    }

    #[test]
    fn across_a_face_of_a_torus_it_bumps_into_what_stands_on_the_other_side() {
        // A flat torus 10 m round along x and z; a tower 0.5 thick at x = 4.8, just inside the
        // +x face. Walking −x from x = −2, Henri meets its copy at x = −5.2 (across the −x
        // face) and stops at its side, x = −5.2 + 0.5 + 0.3, without ever crossing.
        let torus = fk_quotient::Quotient::<E3>::from_translations(
            Point3::origin(),
            &[Vector3::x() * 10.0, Vector3::z() * 10.0],
        );
        let mut field = Field::new(&[]);
        field.colliders.add(Collider {
            a: Point3::new(4.8, 0.0, 0.0),
            b: Point3::new(4.8, 3.0, 0.0),
            radius: 0.5,
        });
        let mut body = henri(at(-2.0, 0.0, 0.0));
        field.run(&mut body, Vector3::new(-2.0, 0.0, 0.0), 3.0);
        assert!(
            body.feet().x < -7.0,
            "without the quotient it walks on: {}",
            body.feet()
        );
        field.colliders.in_quotient(torus, 2, 1.0);
        let mut body = henri(at(-2.0, 0.0, 0.0));
        field.run(&mut body, Vector3::new(-2.0, 0.0, 0.0), 3.0);
        let feet = body.feet();
        assert!((feet.x - (-5.2 + 0.5 + 0.3)).abs() < 1e-6, "{feet}");
        // Found once, as the copy.
        let near: Vec<_> = field.colliders.near(&feet, 0.5).collect();
        assert_eq!(near.len(), 1);
        assert!((near[0].a.x + 5.2).abs() < 1e-9);
    }

    #[test]
    fn a_beam_straddling_a_face_stops_it_from_either_side() {
        // Space closes up along z every 10 m; a low beam from z = 4 to z = 6 crosses the face
        // at z = 5. Walking +z from z = 2 Henri stops at its end, z = 4 − 0.4 − 0.3; walking
        // −z from z = −2 he meets the part that sticks out across the face, ending at
        // z = −4, and stops at z = −4 + 0.4 + 0.3.
        let line = fk_quotient::Quotient::<E3>::from_translations(
            Point3::origin(),
            &[Vector3::z() * 10.0],
        );
        let mut field = Field::new(&[]);
        field.colliders.add(Collider {
            a: Point3::new(0.0, 0.5, 4.0),
            b: Point3::new(0.0, 0.5, 6.0),
            radius: 0.4,
        });
        field.colliders.in_quotient(line, 2, 1.0);
        let mut body = henri(at(0.0, 0.0, 2.0));
        field.run(&mut body, Vector3::new(0.0, 0.0, 2.0), 3.0);
        assert!((body.feet().z - 3.3).abs() < 1e-6, "{}", body.feet());
        let mut body = henri(at(0.0, 0.0, -2.0));
        field.run(&mut body, Vector3::new(0.0, 0.0, -2.0), 3.0);
        assert!((body.feet().z + 3.3).abs() < 1e-6, "{}", body.feet());
    }

    #[test]
    fn colliders_are_found_near_and_not_far() {
        let field = Field::new(&[(3.0, 0.0), (500.0, 0.0)]);
        let near: Vec<_> = field
            .colliders
            .near(&Point3::new(0.0, 0.0, 0.0), 2.0)
            .collect();
        assert_eq!(near.len(), 1);
        assert_eq!(field.colliders.len(), 2);
        // A tall tower is found once, and only near its axis: not from far above its foot.
        let tall = Field::new(&[(0.0, 0.0)]);
        assert_eq!(
            tall.colliders
                .near(&Point3::new(2.5, 40.0, 0.0), 1.0)
                .count(),
            1
        );
        assert_eq!(
            tall.colliders
                .near(&Point3::new(12.0, 3.0, 0.0), 1.0)
                .count(),
            0
        );
    }
}
