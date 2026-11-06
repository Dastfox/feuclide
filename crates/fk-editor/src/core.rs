//! The editor's world: what is edited, the selection, and how what is edited becomes what the
//! engine draws. After `bevy_editor_core`: the editor keeps its own `bevy_ecs` world, and every
//! pane reads and writes it.
//!
//! What is edited lives on two components, [`Thing`] (a shape, a colour, a light) and the
//! engine's own [`Pose`]; [`sync`] turns them into what the renderer reads (`MeshInstances`,
//! `PointLight`, `GlobalPose`) each frame. Keeping the edited form apart from the engine's is
//! what lets the scene be saved, undone and drawn by an unchanged engine.

use bevy_ecs::hierarchy::{ChildOf, Children};
use bevy_ecs::name::Name;
use bevy_ecs::prelude::*;
use bevy_ecs::schedule::Schedule;
use fk_geometry::GroupElement;
use fk_geometry_euclidean::{E3, Se3};
use fk_math::Real;
use fk_math::nalgebra::{UnitQuaternion, Vector3};
use fk_render::{Color, MeshBuilder, MeshHandle, MeshInstances, Meshes, PointLight, shapes};
use fk_scene::{GlobalPose, Pose};
use serde::{Deserialize, Serialize};

/// A stable name for a thing across saves, undos and redos (entities are spawned again by
/// each).
#[derive(Component, Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct ThingId(pub u64);

/// What the editor makes and the scene file keeps: a shape in a colour, maybe a light.
#[derive(Component, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Thing {
    /// Its mesh.
    pub shape: Shape,
    /// Linear RGB.
    pub color: [f32; 3],
    /// A point light at its origin.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub light: Option<Lamp>,
}

impl Default for Thing {
    fn default() -> Self {
        Self {
            shape: Shape::Cuboid {
                half: [0.5, 0.5, 0.5],
            },
            color: [0.8, 0.8, 0.8],
            light: None,
        }
    }
}

/// A point light's values (`fk_render::PointLight`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Lamp {
    /// Linear RGB.
    pub color: [f32; 3],
    /// Light on a surface facing it a metre away.
    pub intensity: Real,
    /// Past this it lights nothing.
    pub range: Real,
}

impl Default for Lamp {
    fn default() -> Self {
        Self {
            color: [1.0, 0.85, 0.6],
            intensity: 2.0,
            range: 8.0,
        }
    }
}

/// The shapes of `fk_render::shapes` the editor offers.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Shape {
    /// Nothing drawn: a group, or a light alone.
    Empty,
    /// A box by its half sizes.
    Cuboid {
        /// Half sizes.
        half: [Real; 3],
    },
    /// A box with bevelled edges.
    Bevelled {
        /// Half sizes.
        half: [Real; 3],
        /// The bevel's width.
        bevel: Real,
    },
    /// A ball.
    Sphere {
        /// Its radius.
        radius: Real,
    },
    /// An upright cylinder.
    Cylinder {
        /// Its radius.
        radius: Real,
        /// Its height.
        height: Real,
    },
    /// An upright cone.
    Cone {
        /// Its base's radius.
        radius: Real,
        /// Its height.
        height: Real,
    },
    /// A square of ground, flat.
    Ground {
        /// Its side.
        size: Real,
    },
    /// A lumpy stone.
    Stone {
        /// Its sizes.
        scale: [Real; 3],
        /// How lumpy.
        lumps: Real,
        /// Which stone.
        seed: u64,
    },
}

impl Shape {
    /// Every kind once, at a default size, for menus.
    pub const KINDS: [Shape; 8] = [
        Shape::Empty,
        Shape::Cuboid {
            half: [0.5, 0.5, 0.5],
        },
        Shape::Bevelled {
            half: [0.5, 0.5, 0.5],
            bevel: 0.08,
        },
        Shape::Sphere { radius: 0.5 },
        Shape::Cylinder {
            radius: 0.4,
            height: 1.0,
        },
        Shape::Cone {
            radius: 0.5,
            height: 1.0,
        },
        Shape::Ground { size: 10.0 },
        Shape::Stone {
            scale: [0.6, 0.4, 0.5],
            lumps: 0.3,
            seed: 1,
        },
    ];

    /// What the menus call it.
    pub fn label(&self) -> &'static str {
        match self {
            Shape::Empty => "Empty",
            Shape::Cuboid { .. } => "Cuboid",
            Shape::Bevelled { .. } => "Bevelled box",
            Shape::Sphere { .. } => "Sphere",
            Shape::Cylinder { .. } => "Cylinder",
            Shape::Cone { .. } => "Cone",
            Shape::Ground { .. } => "Ground",
            Shape::Stone { .. } => "Stone",
        }
    }

    /// Its mesh, in the tangent space at the origin.
    pub fn builder(&self) -> Option<MeshBuilder> {
        let v = |a: [Real; 3]| Vector3::new(a[0], a[1], a[2]);
        Some(match *self {
            Shape::Empty => return None,
            Shape::Cuboid { half } => shapes::cuboid(v(half)),
            Shape::Bevelled { half, bevel } => shapes::bevelled(v(half), bevel),
            Shape::Sphere { radius } => shapes::sphere(radius, 3),
            Shape::Cylinder { radius, height } => shapes::cylinder(radius, height, 32),
            Shape::Cone { radius, height } => shapes::cone(radius, height, 32),
            Shape::Ground { size } => shapes::ground_tile(size, 8),
            Shape::Stone { scale, lumps, seed } => shapes::stone(v(scale), lumps, seed),
        })
    }
}

/// Marks what the editor draws for itself (the grid): never saved, listed or picked.
#[derive(Component)]
pub struct EditorOnly;

/// The selected things, the primary one last (as `EditorSelection` in Bevy's editor).
#[derive(Resource, Default, Debug, Clone, PartialEq)]
pub struct Selection(pub Vec<ThingId>);

impl Selection {
    /// The primary selection.
    pub fn primary(&self) -> Option<ThingId> {
        self.0.last().copied()
    }

    /// Only `id`.
    pub fn set(&mut self, id: ThingId) {
        self.0 = vec![id];
    }

    /// Adds `id`, or takes it out if it was in.
    pub fn toggle(&mut self, id: ThingId) {
        if let Some(at) = self.0.iter().position(|&s| s == id) {
            self.0.remove(at);
        } else {
            self.0.push(id);
        }
    }

    /// Whether `id` is selected.
    pub fn contains(&self, id: ThingId) -> bool {
        self.0.contains(&id)
    }
}

/// Every shape's mesh built once, with its builder kept for picking and outlines.
#[derive(Resource, Default)]
pub struct MeshCache(Vec<(Shape, MeshHandle, MeshBuilder)>);

impl MeshCache {
    /// The mesh of `shape`, built and added to `meshes` the first time.
    pub fn get(&mut self, shape: Shape, meshes: &mut Meshes) -> Option<(MeshHandle, &MeshBuilder)> {
        let at = match self.0.iter().position(|(s, _, _)| *s == shape) {
            Some(at) => at,
            None => {
                let builder = shape.builder()?;
                let handle = meshes.add(builder.build::<E3>());
                self.0.push((shape, handle, builder));
                self.0.len() - 1
            }
        };
        let (_, handle, builder) = &self.0[at];
        Some((*handle, builder))
    }

    /// The builder of `shape`, if it was built.
    pub fn builder(&self, shape: Shape) -> Option<&MeshBuilder> {
        self.0
            .iter()
            .find(|(s, _, _)| *s == shape)
            .map(|(_, _, b)| b)
    }
}

/// Hands out [`ThingId`]s.
#[derive(Resource, Default)]
pub struct NextId(pub u64);

impl NextId {
    /// A fresh id.
    pub fn take(&mut self) -> ThingId {
        self.0 += 1;
        ThingId(self.0)
    }
}

/// The editor's world, ready to draw: meshes, the mesh cache, the selection, the grid, and a
/// schedule that brings the engine's components up to date.
pub struct EditorWorld {
    /// The world.
    pub world: World,
    sync: Schedule,
}

impl EditorWorld {
    /// An empty scene with a grid.
    pub fn new() -> Self {
        let mut world = World::new();
        world.insert_resource(Meshes::default());
        world.init_resource::<MeshCache>();
        world.init_resource::<Selection>();
        world.init_resource::<NextId>();
        let grid = world
            .resource_mut::<Meshes>()
            .add(grid(12, 1.0).build::<E3>());
        world.spawn((
            EditorOnly,
            Name::new("grid"),
            Pose::<E3>(Se3::identity()),
            MeshInstances::<E3>::single(grid, Color::linear(0.12, 0.12, 0.13)),
        ));
        let mut sync = Schedule::default();
        sync.add_systems((build_things, fk_scene::propagate_poses::<E3>).chain());
        Self { world, sync }
    }

    /// Brings the engine's components up to date with what was edited.
    pub fn sync(&mut self) {
        self.sync.run(&mut self.world);
    }

    /// The entity of `id`.
    pub fn entity(&mut self, id: ThingId) -> Option<Entity> {
        self.world
            .query::<(Entity, &ThingId)>()
            .iter(&self.world)
            .find(|(_, t)| **t == id)
            .map(|(e, _)| e)
    }

    /// The id of `entity`, if it is a thing.
    pub fn id(&self, entity: Entity) -> Option<ThingId> {
        self.world.get::<ThingId>(entity).copied()
    }

    /// Spawns a thing named `name` at `pose`, under `parent`.
    pub fn spawn(
        &mut self,
        name: &str,
        thing: Thing,
        pose: Se3,
        parent: Option<ThingId>,
    ) -> ThingId {
        let id = self.world.resource_mut::<NextId>().take();
        self.spawn_with(id, name, thing, pose, parent);
        id
    }

    /// Spawns a thing with a given id (loading, undoing).
    pub fn spawn_with(
        &mut self,
        id: ThingId,
        name: &str,
        thing: Thing,
        pose: Se3,
        parent: Option<ThingId>,
    ) -> Entity {
        let parent = parent.and_then(|p| self.entity(p));
        let mut next = self.world.resource_mut::<NextId>();
        next.0 = next.0.max(id.0);
        let mut entity =
            self.world
                .spawn((id, Name::new(name.to_owned()), thing, Pose::<E3>(pose)));
        if let Some(parent) = parent {
            entity.insert(ChildOf(parent));
        }
        entity.id()
    }

    /// Despawns `id` and everything under it.
    pub fn despawn(&mut self, id: ThingId) {
        if let Some(entity) = self.entity(id) {
            self.world.entity_mut(entity).despawn();
        }
        let mut selection = self.world.resource_mut::<Selection>();
        selection.0.retain(|&s| s != id);
    }

    /// Every thing's id.
    pub fn ids(&mut self) -> Vec<ThingId> {
        let mut ids: Vec<_> = self
            .world
            .query::<&ThingId>()
            .iter(&self.world)
            .copied()
            .collect();
        ids.sort_by_key(|id| id.0);
        ids
    }

    /// The things at the root, in id order.
    pub fn roots(&mut self) -> Vec<Entity> {
        let mut roots: Vec<_> = self
            .world
            .query_filtered::<(Entity, &ThingId), Without<ChildOf>>()
            .iter(&self.world)
            .map(|(e, id)| (id.0, e))
            .collect();
        roots.sort();
        roots.into_iter().map(|(_, e)| e).collect()
    }

    /// The children of `entity` that are things.
    pub fn children(&self, entity: Entity) -> Vec<Entity> {
        self.world
            .get::<Children>(entity)
            .map(|c| {
                c.iter()
                    .filter(|&e| self.world.get::<ThingId>(e).is_some())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// The parent of `entity`, if any.
    pub fn parent(&self, entity: Entity) -> Option<Entity> {
        self.world.get::<ChildOf>(entity).map(|c| c.parent())
    }

    /// The pose of `entity` relative to the root, as last synced.
    pub fn global(&self, entity: Entity) -> Se3 {
        self.world
            .get::<GlobalPose<E3>>(entity)
            .map(|g| g.isometry)
            .unwrap_or_else(Se3::identity)
    }

    /// Whether `ancestor` is `entity` or above it.
    pub fn is_under(&self, entity: Entity, ancestor: Entity) -> bool {
        let mut at = Some(entity);
        while let Some(e) = at {
            if e == ancestor {
                return true;
            }
            at = self.parent(e);
        }
        false
    }

    /// Moves `entity` under `parent` (or to the root), keeping where it is in the world.
    pub fn reparent(&mut self, entity: Entity, parent: Option<Entity>) {
        if parent.is_some_and(|p| self.is_under(p, entity)) {
            return;
        }
        let global = self.global(entity);
        let parent_global = parent.map(|p| self.global(p)).unwrap_or_else(Se3::identity);
        let local = compose(&inverse(&parent_global), &global);
        let mut e = self.world.entity_mut(entity);
        e.insert(Pose::<E3>(local));
        match parent {
            Some(p) => {
                e.insert(ChildOf(p));
            }
            None => {
                e.remove::<ChildOf>();
            }
        }
    }
}

impl Default for EditorWorld {
    fn default() -> Self {
        Self::new()
    }
}

/// Gives every changed thing its mesh and light.
fn build_things(
    mut commands: Commands,
    mut cache: ResMut<MeshCache>,
    mut meshes: ResMut<Meshes>,
    things: Query<(Entity, &Thing), Changed<Thing>>,
) {
    for (entity, thing) in &things {
        let mut e = commands.entity(entity);
        let [r, g, b] = thing.color;
        match cache.get(thing.shape, &mut meshes) {
            Some((mesh, _)) => {
                e.insert(MeshInstances::<E3>::single(mesh, Color::linear(r, g, b)));
            }
            None => {
                e.remove::<MeshInstances<E3>>();
            }
        }
        match thing.light {
            Some(lamp) => {
                let [r, g, b] = lamp.color;
                e.insert(PointLight {
                    color: Color::linear(r, g, b),
                    intensity: lamp.intensity,
                    range: lamp.range,
                });
            }
            None => {
                e.remove::<PointLight>();
            }
        }
    }
}

/// A grid of `2n + 1` lines each way, `step` apart, a millimetre above the ground.
fn grid(n: i32, step: Real) -> MeshBuilder {
    let mut mesh = MeshBuilder::new();
    let half = Real::from(n) * step;
    let up = Vector3::y();
    for i in -n..=n {
        let at = Real::from(i) * step;
        let width = if i == 0 { 0.03 } else { 0.012 };
        let (y, w) = (0.001, width);
        mesh.flat_quad(
            [
                Vector3::new(at - w, y, -half),
                Vector3::new(at - w, y, half),
                Vector3::new(at + w, y, half),
                Vector3::new(at + w, y, -half),
            ],
            up,
        );
        mesh.flat_quad(
            [
                Vector3::new(-half, y, at - w),
                Vector3::new(half, y, at - w),
                Vector3::new(half, y, at + w),
                Vector3::new(-half, y, at + w),
            ],
            up,
        );
    }
    mesh
}

/// `a ∘ b`: `b` first.
pub fn compose(a: &Se3, b: &Se3) -> Se3 {
    Se3 {
        rotation: a.rotation * b.rotation,
        translation: a.translation + a.rotation * b.translation,
    }
}

/// The inverse motion.
pub fn inverse(a: &Se3) -> Se3 {
    let rotation = a.rotation.inverse();
    Se3 {
        rotation,
        translation: -(rotation * a.translation),
    }
}

/// `g` applied to the point `p`.
pub fn apply(g: &Se3, p: &Vector3<Real>) -> Vector3<Real> {
    g.rotation * p + g.translation
}

/// A pose from a place and Euler angles in degrees (about x, then y, then z, nalgebra's).
pub fn pose_of(translation: [Real; 3], degrees: [Real; 3]) -> Se3 {
    let [x, y, z] = degrees.map(Real::to_radians);
    Se3 {
        rotation: UnitQuaternion::from_euler_angles(x, y, z),
        translation: Vector3::new(translation[0], translation[1], translation[2]),
    }
}

/// The place and Euler angles in degrees of a pose.
pub fn parts_of(pose: &Se3) -> ([Real; 3], [Real; 3]) {
    let (x, y, z) = pose.rotation.euler_angles();
    let t = pose.translation;
    ([t.x, t.y, t.z], [x, y, z].map(Real::to_degrees))
}
