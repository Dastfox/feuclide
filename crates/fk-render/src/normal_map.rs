//! Normal maps: tangent-space normals in images, laid on meshes by their uvs.
//!
//! Every map is a square RGBA8 image of the same size, the normal in the surface's tangent frame
//! at each texel, `(x, y, z) ↦ (x, y, z) · 0.5 + 0.5` (z out of the surface, x along the
//! surface's `u`, y along its `v`: the usual "OpenGL" convention). A mesh takes one
//! ([`MeshBuilder::normal_map`](crate::MeshBuilder::normal_map)); its vertices then carry their
//! uv and a tangent along `u`, embedded with the normal (carried along the same geodesic from
//! the origin), and the fragment stage bends the normal by the map in the frame made of the
//! normal, the tangent and their cross product in the geometry (`geo_cross`).
//!
//! The maps are sampled through mipmaps made by averaging the normals of each level below, so a
//! far surface takes their mean and does not shimmer; past [`NormalMaps::fade`] the map flattens
//! to the surface's own normal altogether. [`NormalMaps::set_enabled`] flattens them all at once,
//! to compare.

use bevy_ecs::resource::Resource;
use fk_math::Real;
use fk_math::nalgebra::Vector3;

/// Which normal map, by its index in [`NormalMaps`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NormalMap(pub(crate) u32);

impl NormalMap {
    /// Its layer in the texture array the shader samples: 0 is the flat map every mesh without
    /// one samples.
    pub(crate) fn layer(self) -> u32 {
        self.0 + 1
    }
}

/// Every normal map, as a resource: square images of [`size`](Self::size) texels across.
/// Uploaded again whenever one is added.
#[derive(Resource, Debug)]
pub struct NormalMaps {
    size: u32,
    /// The flat map first, then each added.
    layers: Vec<Vec<[u8; 4]>>,
    /// Where the maps start flattening and where they are flat, geodesic distance from the eye.
    fade: [Real; 2],
    /// Whether the maps bend the light at all.
    enabled: bool,
}

impl Default for NormalMaps {
    fn default() -> Self {
        Self::new(256)
    }
}

impl NormalMaps {
    /// None yet; each will be `size` texels across.
    pub fn new(size: u32) -> Self {
        let size = size.max(1);
        Self {
            size,
            layers: vec![vec![FLAT; (size * size) as usize]],
            fade: [Real::INFINITY; 2],
            enabled: true,
        }
    }

    /// Lays the maps on (the default), or flattens every one to its surface's own normal: the
    /// meshes stay as they are, only the shader stops bending the light.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Whether the maps bend the light ([`set_enabled`](Self::set_enabled)).
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Flattens every map with distance from the eye: whole nearer than `from`, flat from `to`
    /// on, smoothly between. Never, by default.
    pub fn set_fade(&mut self, from: Real, to: Real) {
        self.fade = [from.max(0.0), to.max(from.max(0.0))];
    }

    /// Where the maps start flattening and where they are flat ([`set_fade`](Self::set_fade)).
    pub fn fade(&self) -> [Real; 2] {
        self.fade
    }

    /// How many texels across each map is.
    pub fn size(&self) -> u32 {
        self.size
    }

    /// Adds a map, its texels row by row from `v = 0`.
    ///
    /// # Panics
    ///
    /// If it is not [`size`](Self::size) texels square.
    pub fn add(&mut self, texels: Vec<[u8; 4]>) -> NormalMap {
        assert_eq!(
            texels.len(),
            (self.size * self.size) as usize,
            "a normal map is {0}×{0} texels",
            self.size
        );
        let index = u32::try_from(self.len()).expect("too many normal maps");
        self.layers.push(texels);
        NormalMap(index)
    }

    /// Adds the map of a height field `height(u, v)` (u and v in [0, 1), wrapping), its slopes
    /// times `strength`: the bumps it would make if it were displaced.
    pub fn add_heights(
        &mut self,
        strength: Real,
        height: impl Fn(Real, Real) -> Real,
    ) -> NormalMap {
        let texels = heights_to_normals(self.size, strength, height);
        self.add(texels)
    }

    /// How many maps there are.
    pub fn len(&self) -> usize {
        self.layers.len() - 1
    }

    /// Whether there are none.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Every layer the shader samples: the flat map, then each map in the order added.
    pub(crate) fn layers(&self) -> &[Vec<[u8; 4]>] {
        &self.layers
    }

    /// How many mip levels each layer has, down to one texel.
    pub(crate) fn mip_levels(&self) -> u32 {
        32 - self.size.leading_zeros()
    }

    /// Every layer with its mipmaps, layer by layer and each from its largest level down, as
    /// the texture is uploaded.
    pub(crate) fn texels_with_mips(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for layer in &self.layers {
            let (mut level, mut size) = (layer.clone(), self.size);
            out.extend(level.iter().flatten());
            for _ in 1..self.mip_levels() {
                (level, size) = (half(&level, size), (size / 2).max(1));
                out.extend(level.iter().flatten());
            }
        }
        out
    }
}

/// The level below `texels` (`size` across): each texel the mean normal of the four over it.
fn half(texels: &[[u8; 4]], size: u32) -> Vec<[u8; 4]> {
    let next = (size / 2).max(1);
    let decode = |t: [u8; 4]| {
        fk_math::nalgebra::Vector3::new(t[0], t[1], t[2]).map(|x| Real::from(x) / 127.5 - 1.0)
    };
    let mut out = Vec::with_capacity((next * next) as usize);
    for j in 0..next {
        for i in 0..next {
            let mut sum = fk_math::nalgebra::Vector3::zeros();
            for (di, dj) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let (x, y) = ((2 * i + di).min(size - 1), (2 * j + dj).min(size - 1));
                sum += decode(texels[(y * size + x) as usize]);
            }
            let n = if sum.norm() > 1e-9 {
                sum.normalize()
            } else {
                fk_math::nalgebra::Vector3::z()
            };
            out.push([byte(n.x), byte(n.y), byte(n.z), 255]);
        }
    }
    out
}

/// A unit coordinate as a texel's byte.
fn byte(x: Real) -> u8 {
    ((x * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8
}

/// The flat normal, straight out of the surface.
pub(crate) const FLAT: [u8; 4] = [128, 128, 255, 255];

/// The normal map of a height field over the unit square, `size` texels across, wrapping.
pub fn heights_to_normals(
    size: u32,
    strength: Real,
    height: impl Fn(Real, Real) -> Real,
) -> Vec<[u8; 4]> {
    let n = Real::from(size.max(1));
    let h = 1.0 / n;
    let mut texels = Vec::with_capacity((size * size) as usize);
    for j in 0..size {
        for i in 0..size {
            let (u, v) = ((Real::from(i) + 0.5) * h, (Real::from(j) + 0.5) * h);
            let du = (height((u + h).rem_euclid(1.0), v) - height((u - h).rem_euclid(1.0), v))
                / (2.0 * h);
            let dv = (height(u, (v + h).rem_euclid(1.0)) - height(u, (v - h).rem_euclid(1.0)))
                / (2.0 * h);
            let normal =
                fk_math::nalgebra::Vector3::new(-strength * du, -strength * dv, 1.0).normalize();
            texels.push([byte(normal.x), byte(normal.y), byte(normal.z), 255]);
        }
    }
    texels
}

/// Texture coordinates of `p` (box projection, in the units of `p`), projected along the axis `normal` faces most: the ground
/// and roofs on x and z, walls across and up, so that grain running along v runs up a trunk.
pub fn box_uv(p: &Vector3<Real>, normal: &Vector3<Real>) -> [Real; 2] {
    match axis(normal) {
        0 => [p.z, p.y],
        1 => [p.x, p.z],
        _ => [p.x, p.y],
    }
}

/// The axis `normal` faces most along: 0 x, 1 y, 2 z.
pub fn axis(normal: &Vector3<Real>) -> usize {
    let a = normal.abs();
    if a.x >= a.y && a.x >= a.z {
        0
    } else if a.y >= a.z {
        1
    } else {
        2
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::TAU;

    use super::*;

    #[test]
    fn a_flat_field_is_the_flat_map() {
        let texels = heights_to_normals(8, 1.0, |_, _| 0.3);
        assert!(texels.iter().all(|t| *t == [128, 128, 255, 255]));
    }

    #[test]
    fn a_slope_tilts_the_normal_against_it() {
        // Rising along u: the normal leans back along −u.
        let texels = heights_to_normals(8, 1.0, |u, _| u.min(1.0 - u));
        let [x, y, z, _] = texels[1];
        assert!(x < 128 && y == 128 && z < 255, "{x} {y} {z}");
    }

    #[test]
    fn the_shader_samples_the_flat_map_first() {
        let mut maps = NormalMaps::new(4);
        let bumps = maps.add_heights(1.0, |u, v| (u * 6.0).sin() * (v * 6.0).cos());
        assert_eq!(bumps.layer(), 1);
        let layers = maps.layers();
        assert_eq!(layers.len(), 2);
        assert!(layers[0].iter().all(|t| *t == FLAT));
        assert_eq!(layers[0].len(), 16);
    }

    #[test]
    fn mipmaps_go_down_to_a_texel_and_average_out_to_flat() {
        let mut maps = NormalMaps::new(8);
        maps.add_heights(0.05, |u, _| (u * TAU * 4.0).sin());
        assert_eq!(maps.mip_levels(), 4);
        let texels = maps.texels_with_mips();
        let per_layer = (64 + 16 + 4 + 1) * 4;
        assert_eq!(texels.len(), 2 * per_layer);
        // The ridges' last level is their mean: flat.
        let last = &texels[2 * per_layer - 4..];
        assert_eq!(last, &FLAT);
    }
}
