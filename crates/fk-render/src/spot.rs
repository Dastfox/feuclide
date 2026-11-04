//! Shadows of spots: a perspective depth map from each light with a [`SpotShadow`], each a layer
//! of one texture, handed out to the lights nearest the eye and kept while a light stays put.
//!
//! Flat space only, like the directional [`Shadows`](crate::Shadows): elsewhere no light is
//! handed a map and every spot lights as before.

use bevy_ecs::prelude::*;
use fk_geometry::{GpuGeometry, GroupElement};
use fk_math::Real;
use fk_math::nalgebra::{Matrix4, Vector3, Vector4};
use fk_scene::{ActiveView, GlobalPose, GlobalRealm, Realm};

use crate::light::{PointLight, Spot};

/// Most maps at once, whatever [`SpotShadows::most`] asks.
pub const MAX_SPOT_SHADOWS: usize = 16;

/// Bytes between two maps' matrices in the buffer the pass reads them from (a uniform's
/// dynamic offsets are multiples of 256).
pub(crate) const TILE_STRIDE: u64 = 256;

/// Beside a [`PointLight`] and its [`Spot`], gives it a shadow map while [`SpotShadows`] is
/// enabled: depth seen from the light along its axis, as wide as its cone (to
/// [`SpotShadows::widest`]) and as deep as its range. What stands between the light and a
/// surface keeps its light off it.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct SpotShadow {
    /// Nearer the light than this nothing casts: room for the lamp's own housing about it.
    pub near: Real,
}

impl Default for SpotShadow {
    fn default() -> Self {
        Self { near: 0.1 }
    }
}

/// The spots' shadow maps, as a resource: off by default.
///
/// Each frame the lights with a [`SpotShadow`] in the eye's realm whose light reaches nearest
/// the eye are handed a map each, [`most`](Self::most) of them. A map is drawn when it is
/// handed out, when its light has moved, and otherwise in turn, [`refresh`](Self::refresh) a
/// frame, so that a still lamp costs nothing most frames and what changes within its reach is
/// seen a few frames late. Casters with [`CastsNoShadow`](crate::CastsNoShadow) cast none;
/// every other side casts, both ways.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct SpotShadows {
    /// Whether the maps are drawn.
    pub enabled: bool,
    /// Texels across a map.
    pub size: u32,
    /// Maps at most, to [`MAX_SPOT_SHADOWS`].
    pub most: u32,
    /// Radius of the soft edge, in texels.
    pub softness: Real,
    /// How far a surface is moved along its normal before it is tested, in texels at its
    /// distance from the light.
    pub normal_offset: Real,
    /// Half-angle past which a map does not reach, radians: a cone wider than this is lit past
    /// it as without a map (a map nearly as wide as a hemisphere spends its texels at its edge).
    pub widest: Real,
    /// Maps of still lights drawn again each frame, in turn.
    pub refresh: u32,
    /// How far a light may move (metres, or as much turning in radians) and keep its map: a
    /// light carried a hair this way and that (by a deform's sway) is drawn again only in turn.
    pub stray: Real,
}

impl Default for SpotShadows {
    fn default() -> Self {
        Self {
            enabled: false,
            size: 512,
            most: 8,
            softness: 1.5,
            normal_offset: 1.5,
            widest: 75f64.to_radians(),
            refresh: 1,
            stray: 0.0,
        }
    }
}

/// A light handed a map this frame.
#[derive(Clone, Debug)]
pub(crate) struct SpotTile {
    pub entity: Entity,
    /// Its layer of the texture.
    pub layer: u32,
    /// Where the light was when its map was drawn: from its frame to the root's coordinates.
    pub light: Matrix4<Real>,
    /// From the light's frame to the map's clip space.
    pub projection: Matrix4<Real>,
    /// How far along its normal a surface is moved before it is tested, per unit of its
    /// distance from the light.
    pub offset: Real,
    /// Whether its map is drawn this frame.
    pub draw: bool,
    /// Its range, its cone's half-angle as the map reaches it.
    pub range: Real,
    pub half: Real,
}

impl SpotTile {
    /// From the coordinates of an eye at `eye` (relative to the root) to the map's clip space.
    pub fn matrix<G: GpuGeometry>(&self, eye: &G::Isometry) -> Matrix4<Real> {
        self.projection
            * self
                .seen_from::<G>(eye)
                .try_inverse()
                .unwrap_or_else(Matrix4::identity)
    }

    /// From the light's frame to the eye's coordinates.
    fn seen_from<G: GpuGeometry>(&self, eye: &G::Isometry) -> Matrix4<Real> {
        G::isometry_matrix(&eye.inverse()) * self.light
    }

    /// What the casters are culled against, seen from an eye at `eye`.
    pub fn cast<G: GpuGeometry>(&self, eye: &G::Isometry) -> SpotCast {
        let m = self.seen_from::<G>(eye);
        SpotCast {
            layer: self.layer,
            matrix: self.matrix::<G>(eye),
            at: m.fixed_view::<3, 1>(0, 3).into_owned(),
            axis: -m.fixed_view::<3, 1>(0, 2).into_owned().normalize(),
            range: self.range,
            half: self.half,
        }
    }
}

/// A map drawn this frame, as the raster path culls its casters: flat space, in the eye's frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpotCast {
    /// Its layer of the texture.
    pub layer: u32,
    /// From the eye's coordinates to the map's clip space.
    pub matrix: Matrix4<Real>,
    /// Where the light is.
    pub at: Vector3<Real>,
    /// Its axis, a unit vector.
    pub axis: Vector3<Real>,
    /// Its range.
    pub range: Real,
    /// Its cone's half-angle as the map reaches it.
    pub half: Real,
}

impl SpotCast {
    /// Whether any of the ball of radius `radius` about `centre` (embedded, in the eye's frame)
    /// lies within the light's range and cone: whether it can cast into the map.
    pub fn sees(&self, centre: &Vector4<Real>, radius: Real) -> bool {
        let v = centre.xyz() / centre.w - self.at;
        let d = v.norm();
        if d > self.range + radius {
            return false;
        }
        if d <= radius {
            return true;
        }
        let spread = (self.half + (radius / d).asin()).min(std::f64::consts::PI);
        v.dot(&self.axis) / d >= spread.cos()
    }
}

/// The maps' lights this frame, as a resource, read where each view's lighting is written.
#[derive(Resource, Clone, Debug)]
pub(crate) struct SpotLit {
    pub tiles: Vec<SpotTile>,
    /// One texel, in uv; the soft edge, in texels.
    pub texel: Real,
    pub softness: Real,
}

impl SpotLit {
    pub fn of(&self, entity: Entity) -> Option<&SpotTile> {
        self.tiles.iter().find(|tile| tile.entity == entity)
    }
}

/// Which light holds which layer, from frame to frame.
#[derive(Default)]
pub(crate) struct SpotTiles {
    held: Vec<Option<(Entity, Matrix4<Real>)>>,
    turn: usize,
}

impl SpotTiles {
    /// Every map handed back.
    pub fn clear(&mut self) {
        self.held.iter_mut().for_each(|held| *held = None);
    }

    /// Hands this frame's maps out for the eye of `view`: kept by the lights that hold one and
    /// are still among the nearest, a free layer each to those newly among them; drawn when
    /// handed out, moved, all of them when `redraw` (the texture made again), and `refresh` of
    /// the rest in turn.
    pub fn assign<G: GpuGeometry>(
        &mut self,
        world: &World,
        view: &ActiveView<G>,
        settings: &SpotShadows,
        redraw: bool,
    ) -> Vec<SpotTile> {
        let most = (settings.most as usize).min(MAX_SPOT_SHADOWS);
        self.held.resize(most, None);
        if !settings.enabled || most == 0 || G::constant_curvature() != Some(0.0) {
            self.clear();
            return Vec::new();
        }
        let Some(mut query) =
            world.try_query::<(Entity, &GlobalPose<G>, &PointLight, &Spot, &SpotShadow)>()
        else {
            self.clear();
            return Vec::new();
        };
        let origin = G::origin();
        let to_eye = view.eye.isometry.inverse();
        let mut lights: Vec<_> = query
            .iter(world)
            .filter(|(_, _, light, ..)| light.range > 0.0 && light.intensity > 0.0)
            .filter(|(entity, ..)| {
                world
                    .get::<GlobalRealm>(*entity)
                    .map_or(Realm::MAIN, |realm| realm.0)
                    == view.realm
            })
            .map(|(entity, pose, light, spot, shadow)| {
                let at = G::apply(&to_eye.compose(&pose.isometry), &origin);
                let short = G::distance(&origin, &at) - light.range;
                (short, entity, pose.isometry, *light, *spot, *shadow)
            })
            .collect();
        lights.sort_by(|a, b| a.0.total_cmp(&b.0));
        lights.truncate(most);
        // Layers of lights no longer among them freed; theirs kept by the rest.
        for held in &mut self.held {
            if held.is_some_and(|(entity, _)| !lights.iter().any(|light| light.1 == entity)) {
                *held = None;
            }
        }
        let mut tiles = Vec::with_capacity(lights.len());
        let mut still = Vec::new();
        let now: Vec<G::Isometry> = lights.iter().map(|light| light.2).collect();
        for (_, entity, pose, light, spot, shadow) in lights {
            let matrix = G::isometry_matrix(&pose);
            let (layer, draw) = match self
                .held
                .iter()
                .position(|held| held.is_some_and(|(e, _)| e == entity))
            {
                Some(layer) => {
                    let (_, was) = self.held[layer].expect("held");
                    let moved = (matrix - was).abs().max() > settings.stray.max(1e-9);
                    (layer, moved || redraw)
                }
                None => {
                    let layer = self
                        .held
                        .iter()
                        .position(Option::is_none)
                        .expect("a free layer for each light");
                    (layer, true)
                }
            };
            // Kept as it was drawn while it strays no further.
            let drawn = match self.held[layer] {
                Some((_, was)) if !draw => was,
                _ => matrix,
            };
            self.held[layer] = Some((entity, drawn));
            if !draw {
                still.push(tiles.len());
            }
            let half = spot.outer.min(settings.widest).clamp(0.01, 1.55);
            let near = shadow.near.clamp(1e-3, light.range * 0.5);
            let texel = 2.0 * half.tan() / Real::from(settings.size.max(1));
            tiles.push(SpotTile {
                entity,
                layer: layer as u32,
                light: drawn,
                projection: projection(half, near, light.range),
                offset: settings.normal_offset.max(0.0) * texel,
                draw,
                range: light.range,
                half,
            });
        }
        // Some of the still ones drawn again, in turn by layer.
        if !still.is_empty() {
            still.sort_by_key(|&k| (tiles[k].layer as usize + most - self.turn % most) % most);
            for &k in still.iter().take(settings.refresh as usize) {
                let tile = &mut tiles[k];
                tile.draw = true;
                tile.light = G::isometry_matrix(&now[k]);
                self.held[tile.layer as usize] = Some((tile.entity, tile.light));
            }
            self.turn = (self.turn + settings.refresh as usize) % most;
        }
        tiles
    }
}

/// Hands the spots' maps out for the frame's eye (`view`), the texture made again if it
/// changed size, and keeps the lights holding them for every view's lighting ([`SpotLit`]): what
/// the casters of those drawn this frame are culled against, their matrices written for the
/// spots' pass.
pub(crate) fn hand_out<G: GpuGeometry>(
    gpu: &crate::Gpu,
    map: &mut crate::shadow::ShadowMap,
    held: &mut SpotTiles,
    world: &mut World,
    view: Option<&ActiveView<G>>,
) -> Vec<SpotCast> {
    let settings = world
        .get_resource::<SpotShadows>()
        .copied()
        .unwrap_or_default();
    let (tiles, casts) = match view {
        Some(view) if settings.enabled => {
            let redraw = map.resize_spots(gpu, settings.size, settings.most.max(1));
            let tiles = held.assign(world, view, &settings, redraw);
            let casts: Vec<_> = tiles
                .iter()
                .filter(|tile| tile.draw)
                .map(|tile| tile.cast::<G>(&view.eye.isometry))
                .collect();
            (tiles, casts)
        }
        _ => {
            held.clear();
            (Vec::new(), Vec::new())
        }
    };
    for cast in &casts {
        map.write_spot(gpu, cast.layer, &cast.matrix);
    }
    world.insert_resource(SpotLit {
        tiles,
        texel: 1.0 / Real::from(map.spot_size),
        softness: settings.softness.max(0.0),
    });
    casts
}

/// A perspective projection along `−z`, `half` either side of it, from `near` to `far`: depth 0
/// at `near` and 1 at `far`.
pub(crate) fn projection(half: Real, near: Real, far: Real) -> Matrix4<Real> {
    let s = 1.0 / half.tan();
    let a = far / (far - near);
    #[rustfmt::skip]
    let m = Matrix4::new(
        s, 0.0, 0.0, 0.0,
        0.0, s, 0.0, 0.0,
        0.0, 0.0, -a, -a * near,
        0.0, 0.0, -1.0, 0.0,
    );
    m
}

#[cfg(test)]
mod tests {
    use fk_geometry::Geometry;
    use fk_geometry_euclidean::E3;
    use fk_math::nalgebra::Point3;
    use fk_scene::Camera;

    use super::*;

    fn translation(x: Real, y: Real, z: Real) -> <E3 as Geometry>::Isometry {
        <E3 as Geometry>::Isometry::exp(&E3::transvection(&Vector3::new(x, y, z)))
    }

    #[test]
    fn depth_runs_from_near_to_far_along_the_axis() {
        let p = projection(0.5, 0.1, 10.0);
        let at = |z: Real| {
            let q = p * Vector4::new(0.0, 0.0, z, 1.0);
            q.z / q.w
        };
        assert!(at(-0.1).abs() < 1e-9);
        assert!((at(-10.0) - 1.0).abs() < 1e-9);
        assert!(at(-2.0) < at(-3.0), "deeper further off");
        let edge = p * Vector4::new(0.5f64.tan() * 4.0, 0.0, -4.0, 1.0);
        assert!(
            (edge.x / edge.w - 1.0).abs() < 1e-9,
            "the cone's edge at the map's"
        );
    }

    #[test]
    fn a_ball_casts_only_within_the_cone_and_the_range() {
        let cast = SpotCast {
            layer: 0,
            matrix: Matrix4::identity(),
            at: Vector3::new(0.0, 4.0, 0.0),
            axis: -Vector3::y(),
            range: 10.0,
            half: 0.6,
        };
        let ball = |x: Real, y: Real| Vector4::new(x, y, 0.0, 1.0);
        assert!(cast.sees(&ball(0.0, 0.0), 0.5), "under it");
        assert!(!cast.sees(&ball(0.0, 8.0), 0.5), "behind it");
        assert!(!cast.sees(&ball(20.0, 0.0), 0.5), "out of range");
        assert!(!cast.sees(&ball(6.0, 0.0), 0.5), "out of the cone");
        assert!(
            cast.sees(&ball(6.0, 0.0), 4.0),
            "a big one reaching into it"
        );
        assert!(cast.sees(&ball(0.0, 4.2), 0.5), "about the light");
    }

    fn lamp(world: &mut World, x: Real) -> Entity {
        world
            .spawn((
                GlobalPose::<E3>::new(translation(x, 3.0, 0.0)),
                PointLight::default(),
                Spot {
                    inner: 0.3,
                    outer: 0.6,
                },
                SpotShadow::default(),
            ))
            .id()
    }

    #[test]
    fn the_nearest_hold_maps_drawn_once_unless_they_move() {
        let mut world = World::new();
        let near = lamp(&mut world, 1.0);
        let far = lamp(&mut world, 30.0);
        let farther = lamp(&mut world, 60.0);
        let view = ActiveView::<E3> {
            entity: Entity::PLACEHOLDER,
            camera: Camera::new(1.0, 0.1, 100.0),
            eye: GlobalPose::new(translation(0.0, 1.7, 0.0)),
            realm: Realm::MAIN,
        };
        let settings = SpotShadows {
            enabled: true,
            most: 2,
            refresh: 0,
            ..SpotShadows::default()
        };
        let mut tiles = SpotTiles::default();
        let first = tiles.assign(&world, &view, &settings, false);
        let entities: Vec<_> = first.iter().map(|tile| tile.entity).collect();
        assert_eq!(entities, [near, far]);
        assert!(first.iter().all(|tile| tile.draw));
        let second = tiles.assign(&world, &view, &settings, false);
        assert!(second.iter().all(|tile| !tile.draw), "still: kept");
        assert_eq!(
            second.iter().map(|t| t.layer).collect::<Vec<_>>(),
            first.iter().map(|t| t.layer).collect::<Vec<_>>()
        );
        // The near one moves; walked over to the farther one, the far one's layer goes to it.
        world
            .entity_mut(near)
            .insert(GlobalPose::<E3>::new(translation(1.5, 3.0, 0.0)));
        let third = tiles.assign(&world, &view, &settings, false);
        assert!(third.iter().find(|t| t.entity == near).unwrap().draw);
        assert!(!third.iter().find(|t| t.entity == far).unwrap().draw);
        let there = ActiveView::<E3> {
            eye: GlobalPose::new(translation(62.0, 1.7, 0.0)),
            ..view.clone()
        };
        let fourth = tiles.assign(&world, &there, &settings, false);
        let handed = fourth.iter().find(|t| t.entity == farther).unwrap();
        assert!(handed.draw);
        assert_eq!(
            handed.layer,
            third.iter().find(|t| t.entity == near).unwrap().layer
        );
        // In turn, a still map is drawn again.
        let refreshing = SpotShadows {
            refresh: 1,
            ..settings
        };
        let fifth = tiles.assign(&world, &there, &refreshing, false);
        assert_eq!(fifth.iter().filter(|t| t.draw).count(), 1);
        // The map's matrix sends the light's own axis to the map's middle.
        let tile = &fifth[0];
        let eye = there.eye.isometry;
        let below = E3::apply(
            &eye.inverse(),
            &E3::apply(&translation(60.0, 3.0, 0.0), &Point3::new(0.0, 0.0, -2.0)),
        );
        let q = tile.matrix::<E3>(&eye) * Vector4::new(below.x, below.y, below.z, 1.0);
        assert!((q.x / q.w).abs() < 1e-9 && (q.y / q.w).abs() < 1e-9);
    }

    #[test]
    fn a_light_straying_a_hair_keeps_its_map_as_it_was_drawn() {
        let mut world = World::new();
        let lamp = lamp(&mut world, 1.0);
        let view = ActiveView::<E3> {
            entity: Entity::PLACEHOLDER,
            camera: Camera::new(1.0, 0.1, 100.0),
            eye: GlobalPose::new(translation(0.0, 1.7, 0.0)),
            realm: Realm::MAIN,
        };
        let settings = SpotShadows {
            enabled: true,
            refresh: 0,
            stray: 0.05,
            ..SpotShadows::default()
        };
        let mut tiles = SpotTiles::default();
        let drawn = tiles.assign(&world, &view, &settings, false)[0].light;
        let mut moved = |x: Real| {
            world
                .entity_mut(lamp)
                .insert(GlobalPose::<E3>::new(translation(x, 3.0, 0.0)));
            tiles.assign(&world, &view, &settings, false)[0].clone()
        };
        let swayed = moved(1.03);
        assert!(!swayed.draw);
        assert_eq!(swayed.light, drawn, "sampled as it was drawn");
        // Strayed further from where it was drawn, a step at a time: drawn again.
        assert!(!moved(1.045).draw);
        assert!(moved(1.06).draw);
    }
}
