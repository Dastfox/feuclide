//! Normal maps: tangent-space normals in images, laid on meshes by their uvs.
//!
//! Every map is a square RGBA8 image of the same size, the normal in the surface's tangent frame
//! at each texel, `(x, y, z) ↦ (x, y, z) · 0.5 + 0.5` (z out of the surface, x along the
//! surface's `u`, y along its `v`: the usual "OpenGL" convention). A mesh takes one
//! ([`MeshBuilder::normal_map`](crate::MeshBuilder::normal_map)); its vertices then carry their
//! uv and a tangent along `u`, embedded with the normal (carried along the same geodesic from
//! the origin), and the fragment stage bends the normal by the map in the frame made of the
//! normal, the tangent and their cross product in the geometry (`geo_cross`).

use bevy_ecs::resource::Resource;
use fk_math::Real;

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
        }
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
            let byte = |x: Real| ((x * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8;
            texels.push([byte(normal.x), byte(normal.y), byte(normal.z), 255]);
        }
    }
    texels
}

#[cfg(test)]
mod tests {
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
}
