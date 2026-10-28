use bevy_ecs::prelude::*;

/// The parallel world an entity is in: a world of its own at its own origin, sharing the app,
/// the clock and the eyes with the others, which no eye in another sees, no light of another
/// lights and no body of another collides with. Realms are joined by
/// [`Portal`](crate::Portal)s.
///
/// An entity without one is in its parent's realm, a root without one in [`Realm::MAIN`]:
/// put it on the root of what lives elsewhere, never on every part. A realm has no geometry,
/// scale or clock of its own: every realm is in the app's geometry, and its poses are
/// isometries as everywhere. Where a body moves through a portal, its mover writes the realm
/// it comes out in ([`Portals::cross`](crate::Portals::cross)).
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Realm(pub u32);

impl Realm {
    /// The realm of everything that says nothing else.
    pub const MAIN: Self = Self(0);
}

/// The realm an entity is in, resolved down the hierarchy from the [`Realm`]s along its
/// ancestry, written with the [`GlobalPose`](crate::GlobalPose)s. Read it after
/// [`SceneSystems::Propagate`](crate::SceneSystems::Propagate); writing it has no lasting
/// effect.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct GlobalRealm(pub Realm);

/// The realms the app has made, as a resource: each new one by name, the name only for logs and
/// readouts. [`Realm::MAIN`] is always there, named `main`.
#[derive(Resource, Debug, Clone)]
pub struct Realms {
    names: Vec<String>,
}

impl Default for Realms {
    fn default() -> Self {
        Self {
            names: vec!["main".to_owned()],
        }
    }
}

impl Realms {
    /// A new realm named `name`. Two realms may share a name; they are still two.
    pub fn add(&mut self, name: impl Into<String>) -> Realm {
        self.names.push(name.into());
        Realm(u32::try_from(self.names.len() - 1).unwrap_or(u32::MAX))
    }

    /// The name `realm` was made with, `None` for one this app never made.
    pub fn name(&self, realm: Realm) -> Option<&str> {
        self.names.get(realm.0 as usize).map(String::as_str)
    }

    /// How many realms there are, [`Realm::MAIN`] among them.
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// Never: [`Realm::MAIN`] is always there.
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use bevy_ecs::prelude::*;
    use fk_app::App;
    use fk_geometry_euclidean::E3;

    use crate::{ActiveView, Camera, GlobalRealm, Pose, Realm, Realms, ScenePlugin};

    #[test]
    fn children_are_in_their_parents_realm_and_the_eye_sees_its_own() {
        let mut app = App::new();
        app.add_plugin(ScenePlugin::<E3>::default());
        let room = app.world_mut().resource_mut::<Realms>().add("room");
        let world = app.world_mut();
        let root = world.spawn((Pose::<E3>::identity(), room)).id();
        let child = world.spawn((Pose::<E3>::identity(), ChildOf(root))).id();
        let eye = world
            .spawn((
                Pose::<E3>::identity(),
                Camera::<E3>::new(1.0, 0.1, 10.0),
                ChildOf(child),
            ))
            .id();
        let elsewhere = world.spawn(Pose::<E3>::identity()).id();
        app.update(0.0);
        let realm = |entity| app.world().get::<GlobalRealm>(entity).unwrap().0;
        assert_eq!(realm(child), room);
        assert_eq!(realm(eye), room);
        assert_eq!(realm(elsewhere), Realm::MAIN);
        assert_eq!(app.world().resource::<ActiveView<E3>>().realm, room);
        assert_eq!(app.world().resource::<Realms>().name(room), Some("room"));
    }
}
