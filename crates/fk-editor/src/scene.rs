//! The scene file: every thing and the light, in TOML a person can read and edit
//! (`scenes/*.scene.toml`). After Bevy's scene files (BSN): the scene is data, the editor
//! spawns it, and the same form serves undo and redo as a whole-scene snapshot.

use bevy_ecs::name::Name;
use fk_geometry_euclidean::E3;
use fk_math::Real;
use fk_math::nalgebra::Vector3;
use fk_render::{Color, Lighting};
use fk_scene::Pose;
use serde::{Deserialize, Serialize};

use crate::core::{EditorWorld, Selection, Thing, ThingId, parts_of, pose_of};

/// A scene: its light and its things, parents before children.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SceneDoc {
    /// How it is lit.
    #[serde(default)]
    pub lighting: LightingDoc,
    /// What is in it.
    #[serde(default, rename = "thing")]
    pub things: Vec<ThingDoc>,
}

/// One thing.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ThingDoc {
    /// Its stable id.
    pub id: u64,
    /// Its name.
    pub name: String,
    /// The thing it hangs from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<u64>,
    /// Where, relative to its parent.
    pub translation: [Real; 3],
    /// Turned how, in degrees about x, y and z.
    pub rotation: [Real; 3],
    /// What it is.
    #[serde(flatten)]
    pub thing: Thing,
}

/// The values of `fk_render::Lighting` the editor shows.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LightingDoc {
    /// Towards the sun.
    pub sun: [Real; 3],
    /// Light on surfaces facing it, linear RGB.
    pub sun_color: [f32; 3],
    /// The painted shadow colour.
    pub shadow_tint: [f32; 3],
    /// The sky straight up.
    pub zenith: [f32; 3],
    /// The sky at the horizon, and the clear colour.
    pub horizon: [f32; 3],
    /// How thick the fog is.
    pub fog_density: Real,
    /// Light bands between shadow and full light.
    pub bands: u32,
    /// 0 painted in bands, 1 smooth.
    pub smooth: Real,
}

fn rgb(c: Color) -> [f32; 3] {
    [c.r, c.g, c.b]
}

fn color([r, g, b]: [f32; 3]) -> Color {
    Color::linear(r, g, b)
}

impl Default for LightingDoc {
    fn default() -> Self {
        let l = Lighting::default();
        Self {
            sun: [l.sun.x, l.sun.y, l.sun.z],
            sun_color: rgb(l.sun_color),
            shadow_tint: rgb(l.shadow_tint),
            zenith: rgb(l.sky.zenith),
            horizon: rgb(l.sky.horizon),
            fog_density: l.fog_density,
            bands: l.bands,
            smooth: l.smooth,
        }
    }
}

impl LightingDoc {
    /// The engine's lighting with these values.
    pub fn lighting(&self) -> Lighting {
        let mut l = Lighting::default();
        let [x, y, z] = self.sun;
        let sun = Vector3::new(x, y, z);
        l.sun = if sun.norm() > 1e-9 {
            sun.normalize()
        } else {
            Vector3::y()
        };
        l.sun_color = color(self.sun_color);
        l.shadow_tint = color(self.shadow_tint);
        l.sky.zenith = color(self.zenith);
        l.sky.horizon = color(self.horizon);
        l.fog_density = self.fog_density;
        l.bands = self.bands.max(1);
        l.smooth = self.smooth;
        l
    }
}

impl SceneDoc {
    /// What `editor` holds now.
    pub fn capture(editor: &mut EditorWorld, lighting: &LightingDoc) -> Self {
        let mut things = Vec::new();
        let mut stack: Vec<_> = editor.roots().into_iter().rev().collect();
        while let Some(entity) = stack.pop() {
            let world = &editor.world;
            let (Some(id), Some(thing), Some(pose)) = (
                world.get::<ThingId>(entity),
                world.get::<Thing>(entity),
                world.get::<Pose<E3>>(entity),
            ) else {
                continue;
            };
            let parent = editor
                .parent(entity)
                .and_then(|p| world.get::<ThingId>(p))
                .map(|p| p.0);
            let name = world
                .get::<Name>(entity)
                .map(|n| n.as_str().to_owned())
                .unwrap_or_default();
            let (translation, rotation) = parts_of(&pose.0);
            things.push(ThingDoc {
                id: id.0,
                name,
                parent,
                translation,
                rotation,
                thing: thing.clone(),
            });
            stack.extend(editor.children(entity).into_iter().rev());
        }
        Self {
            lighting: lighting.clone(),
            things,
        }
    }

    /// Replaces what `editor` holds with this scene, keeping the selection where it still
    /// names a thing.
    pub fn restore(&self, editor: &mut EditorWorld) {
        for id in editor.ids() {
            if let Some(entity) = editor.entity(id)
                && editor.world.get_entity(entity).is_ok()
            {
                editor.world.entity_mut(entity).despawn();
            }
        }
        for doc in &self.things {
            editor.spawn_with(
                ThingId(doc.id),
                &doc.name,
                doc.thing.clone(),
                pose_of(doc.translation, doc.rotation),
                doc.parent.map(ThingId),
            );
        }
        let ids: Vec<_> = self.things.iter().map(|t| ThingId(t.id)).collect();
        editor
            .world
            .resource_mut::<Selection>()
            .0
            .retain(|s| ids.contains(s));
    }

    /// Reads a scene file.
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        toml::from_str(&text).map_err(|e| e.to_string())
    }

    /// Writes it.
    pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
        let text = toml::to_string_pretty(self).map_err(|e| e.to_string())?;
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
        std::fs::write(path, text).map_err(|e| e.to_string())
    }

    /// A small scene to start from.
    pub fn example() -> Self {
        let thing = |id, name: &str, parent, t, r, thing| ThingDoc {
            id,
            name: name.to_owned(),
            parent,
            translation: t,
            rotation: r,
            thing,
        };
        use crate::core::{Lamp, Shape};
        Self {
            lighting: LightingDoc::default(),
            things: vec![
                thing(
                    1,
                    "ground",
                    None,
                    [0.0, -0.01, 0.0],
                    [0.0; 3],
                    Thing {
                        shape: Shape::Ground { size: 16.0 },
                        color: [0.25, 0.3, 0.22],
                        light: None,
                    },
                ),
                thing(
                    2,
                    "table",
                    None,
                    [0.0, 0.4, 0.0],
                    [0.0, 15.0, 0.0],
                    Thing {
                        shape: Shape::Bevelled {
                            half: [1.0, 0.05, 0.6],
                            bevel: 0.02,
                        },
                        color: [0.35, 0.2, 0.1],
                        light: None,
                    },
                ),
                thing(
                    3,
                    "bowl",
                    Some(2),
                    [0.3, 0.17, 0.1],
                    [0.0; 3],
                    Thing {
                        shape: Shape::Sphere { radius: 0.12 },
                        color: [0.7, 0.68, 0.6],
                        light: None,
                    },
                ),
                thing(
                    4,
                    "candle",
                    Some(2),
                    [-0.5, 0.17, -0.2],
                    [0.0; 3],
                    Thing {
                        shape: Shape::Cylinder {
                            radius: 0.04,
                            height: 0.25,
                        },
                        color: [0.9, 0.86, 0.7],
                        light: Some(Lamp::default()),
                    },
                ),
                thing(
                    5,
                    "stone",
                    None,
                    [2.5, 0.2, 1.5],
                    [0.0, 40.0, 0.0],
                    Thing {
                        shape: Shape::Stone {
                            scale: [0.6, 0.4, 0.5],
                            lumps: 0.3,
                            seed: 7,
                        },
                        color: [0.4, 0.4, 0.42],
                        light: None,
                    },
                ),
            ],
        }
    }
}
