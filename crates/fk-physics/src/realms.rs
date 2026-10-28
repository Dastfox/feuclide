//! Bodies in parallel worlds: what each [`Realm`] holds to bump into, and going through the
//! [`Portal`]s between them.

use std::collections::HashMap;

use fk_geometry::{Geometry, GpuGeometry, GroupElement};
use fk_scene::{Portal, Realm, cross};

use crate::character::{Character, Colliders};

/// The [`Colliders`] of each realm: a body bumps into those of the realm it is in, and nothing
/// of another, however near it stands in that realm's coordinates.
#[derive(Debug)]
pub struct RealmColliders<G: Geometry> {
    realms: HashMap<Realm, Colliders<G>>,
    /// What a realm with none holds.
    none: Colliders<G>,
}

impl<G: Geometry> Default for RealmColliders<G> {
    fn default() -> Self {
        Self {
            realms: HashMap::new(),
            none: Colliders::new(1.0),
        }
    }
}

impl<G: Geometry> RealmColliders<G> {
    /// None in any realm.
    pub fn new() -> Self {
        Self::default()
    }

    /// What `realm` holds: none if nothing was added to it.
    pub fn of(&self, realm: Realm) -> &Colliders<G> {
        self.realms.get(&realm).unwrap_or(&self.none)
    }

    /// What `realm` holds, to add to.
    pub fn of_mut(&mut self, realm: Realm) -> &mut Colliders<G> {
        self.realms
            .entry(realm)
            .or_insert_with(|| Colliders::new(1.0))
    }

    /// The realms that hold any.
    pub fn realms(&self) -> impl Iterator<Item = Realm> + '_ {
        self.realms.keys().copied()
    }
}

impl<G: GpuGeometry> Character<G> {
    /// Carries the character through the first passable opening of `portals` in `realm` that
    /// the middle of its body went through since it stood at `before` (its pose before the
    /// last [`step`](Self::step)): its pose by the portal's carry, its velocity as it was in
    /// its own frame. The realm it is in after, `None` when it went through none. Ask after
    /// every step, before anything else moves it.
    pub fn through<'a>(
        &mut self,
        before: &G::Isometry,
        realm: Realm,
        portals: impl IntoIterator<Item = &'a Portal<G>>,
    ) -> Option<Realm> {
        let middle = G::exp(
            &G::origin(),
            &(G::origin_frame(1) * (self.shape.height / 2.0)),
        );
        let (from, to) = (G::apply(before, &middle), G::apply(&self.pose, &middle));
        let crossing = cross(portals, realm, &from, &to)?;
        self.pose = crossing.carry.compose(&self.pose);
        self.pose.renormalize();
        Some(crossing.realm)
    }
}

#[cfg(test)]
mod tests {
    use fk_geometry_euclidean::E3;
    use fk_math::Real;
    use fk_math::nalgebra::{Point3, Vector3};

    use super::*;
    use crate::{CharacterShape, Collider};

    fn shift(x: Real, y: Real, z: Real) -> <E3 as Geometry>::Isometry {
        <E3 as Geometry>::Isometry::exp(&E3::transvection(&Vector3::new(x, y, z)))
    }

    #[test]
    fn a_character_walking_through_a_door_comes_out_in_its_realm() {
        // A door 1 wide, 2.2 high, from the ground up, facing +z at the origin; its way in 100 m
        // along x in the room realm.
        let room = Realm(1);
        let door = Portal::between(
            Realm::MAIN,
            shift(0.0, 1.1, 0.0),
            room,
            shift(100.0, 1.1, 0.0),
            [0.5, 1.1],
        );
        let shape = CharacterShape {
            radius: 0.3,
            height: 1.8,
        };
        let mut body = Character::<E3>::new(shift(0.0, 0.0, 0.2), shape);
        let before = body.pose;
        body.pose = shift(0.0, 0.0, -0.2);
        assert_eq!(body.through(&before, Realm::MAIN, [&door]), Some(room));
        let feet = E3::apply(&body.pose, &E3::origin());
        assert!(
            (feet - Point3::new(100.0, 0.0, -0.2)).norm() < 1e-9,
            "{feet}"
        );
        // Walking on, nothing more is crossed.
        let before = body.pose;
        body.pose = shift(100.0, 0.0, -0.6);
        assert_eq!(body.through(&before, room, [&door]), None);
    }

    #[test]
    fn each_realm_holds_its_own_colliders() {
        let mut realms = RealmColliders::<E3>::new();
        realms.of_mut(Realm(1)).add(Collider {
            a: Point3::origin(),
            b: Point3::new(0.0, 2.0, 0.0),
            radius: 0.5,
        });
        assert_eq!(realms.of(Realm(1)).near(&Point3::origin(), 1.0).count(), 1);
        assert_eq!(
            realms.of(Realm::MAIN).near(&Point3::origin(), 1.0).count(),
            0
        );
    }
}
