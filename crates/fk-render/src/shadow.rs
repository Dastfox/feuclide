//! The directional light's shadow map: where it is, and the texture renderers draw into.

use fk_geometry::GpuGeometry;
use fk_math::Real;
use fk_math::nalgebra::{Matrix4, Vector3};

use crate::{Gpu, Shadows};

pub(crate) const SHADOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// The shadow map as seen from the eye, for drawing casters into it and culling them.
///
/// Flat space only. The map is an orthographic box along the light: `x` and `y` in `[−1, 1]`
/// across it, depth in `[0, 1]` from the side nearest the light.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadowView {
    /// From the eye's coordinates (embedded points) to the map's clip space.
    pub matrix: Matrix4<Real>,
    /// Half-width of the map, in units of distance.
    pub range: Real,
    /// Half-depth of the map along the light.
    pub depth: Real,
}

impl ShadowView {
    /// The map of `shadows` around the eye at `eye` (its global pose), for a light shining
    /// from `light` (a direction in the reference frame). Its centre is snapped to whole
    /// texels of a basis fixed in the reference frame, so the map stays put under the scene as
    /// the eye moves. `None` outside flat space.
    pub fn new<G: GpuGeometry>(
        eye: &G::Isometry,
        light: &Vector3<Real>,
        shadows: &Shadows,
    ) -> Option<Self> {
        if G::constant_curvature() != Some(0.0) || light.norm() < 1e-9 {
            return None;
        }
        let range = shadows.range.max(1e-3);
        let depth = shadows.depth.max(1e-3);
        // The light travels along `forward`; `right` and `up` span the map.
        let forward = -light.normalize();
        let hint = if forward.z.abs() < 0.9 {
            Vector3::z()
        } else {
            Vector3::x()
        };
        let right = forward.cross(&hint).normalize();
        let up = right.cross(&forward);
        // Eye coordinates to reference coordinates: in flat space the embedding is affine.
        let to_root = G::isometry_matrix(eye);
        let eye_at = to_root.fixed_view::<3, 1>(0, 3).into_owned();
        let texel = 2.0 * range / Real::from(shadows.size.max(1));
        let snap = |x: Real| (x / texel).round() * texel;
        let centre = Vector3::new(
            snap(right.dot(&eye_at)),
            snap(up.dot(&eye_at)),
            forward.dot(&eye_at) - depth,
        );
        let scale = Vector3::new(1.0 / range, 1.0 / range, 0.5 / depth);
        let mut to_map = Matrix4::identity();
        for (row, (axis, offset)) in [right, up, forward].iter().zip(centre.iter()).enumerate() {
            for column in 0..3 {
                to_map[(row, column)] = axis[column] * scale[row];
            }
            to_map[(row, 3)] = -offset * scale[row];
        }
        Some(Self {
            matrix: to_map * to_root,
            range,
            depth,
        })
    }

    /// Whether any of the ball of radius `radius` about `centre` (relative to the eye) is in the
    /// map: whether it can cast a shadow there.
    pub fn sees<G: GpuGeometry>(&self, centre: &G::Point, radius: Real) -> bool {
        let p = self.matrix * G::embed_point(centre);
        let across = 1.0 + radius / self.range;
        let along = radius / (2.0 * self.depth);
        p.x.abs() <= across && p.y.abs() <= across && p.z >= -along && p.z <= 1.0 + along
    }
}

/// The texture the casters are drawn into and the bind group (`fk::shadow`, group 1) the
/// receivers read it through.
pub(crate) struct ShadowMap {
    pub layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    pub size: u32,
    pub view: wgpu::TextureView,
    pub bind_group: wgpu::BindGroup,
}

impl ShadowMap {
    /// A map of one texel, until shadows are drawn.
    pub fn new(gpu: &Gpu) -> Self {
        let device = &gpu.device;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fk shadow"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("fk shadow"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            // Lit where the surface is no further from the light than the nearest caster.
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        let (view, bind_group) = Self::texture(gpu, &layout, &sampler, 1);
        Self {
            layout,
            sampler,
            size: 1,
            view,
            bind_group,
        }
    }

    /// Makes the texture `size` texels across, if it is not already.
    pub fn resize(&mut self, gpu: &Gpu, size: u32) {
        let size = size.clamp(1, gpu.device.limits().max_texture_dimension_2d);
        if size != self.size {
            (self.view, self.bind_group) = Self::texture(gpu, &self.layout, &self.sampler, size);
            self.size = size;
        }
    }

    fn texture(
        gpu: &Gpu,
        layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        size: u32,
    ) -> (wgpu::TextureView, wgpu::BindGroup) {
        let view = gpu
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("fk shadow map"),
                size: wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: SHADOW_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default());
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fk shadow"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });
        (view, bind_group)
    }
}

#[cfg(test)]
mod tests {
    use fk_geometry::{Geometry, GroupElement};
    use fk_geometry_euclidean::E3;
    use fk_math::nalgebra::{Point3, Vector4};

    use super::*;

    fn translation(x: Real, y: Real, z: Real) -> <E3 as Geometry>::Isometry {
        <E3 as Geometry>::Isometry::exp(&E3::transvection(&Vector3::new(x, y, z)))
    }

    fn shadows() -> Shadows {
        Shadows {
            size: 100,
            range: 50.0,
            depth: 200.0,
            ..Shadows::default()
        }
    }

    #[test]
    fn the_eye_is_mid_depth_and_casters_further_from_the_light_are_deeper() {
        let eye = translation(3.0, 1.7, -8.0);
        let map = ShadowView::new::<E3>(&eye, &Vector3::new(0.3, 1.0, 0.2), &shadows()).unwrap();
        let at_eye = map.matrix * Vector4::new(0.0, 0.0, 0.0, 1.0);
        assert!(
            at_eye.x.abs() <= 1.0 / 100.0 + 1e-9,
            "snapped within a texel"
        );
        assert!((at_eye.z - 0.5).abs() < 1e-9);
        let below = map.matrix * Vector4::new(0.0, -1.0, 0.0, 1.0);
        assert!(
            below.z > at_eye.z,
            "the ground is further from a high light"
        );
        assert!(map.sees::<E3>(&Point3::new(40.0, 0.0, 0.0), 1.0));
        assert!(!map.sees::<E3>(&Point3::new(200.0, 0.0, 0.0), 1.0));
    }

    #[test]
    fn the_map_stays_put_under_the_scene_while_the_eye_moves_within_a_texel() {
        let light = Vector3::new(-0.5, 0.6, 0.1);
        let ground = |eye: &<E3 as Geometry>::Isometry| {
            let map = ShadowView::new::<E3>(eye, &light, &shadows()).unwrap();
            // The same root point, seen from the eye.
            let p = E3::apply(&eye.inverse(), &Point3::new(10.0, 0.0, 10.0));
            map.matrix * Vector4::new(p.x, p.y, p.z, 1.0)
        };
        let (a, b) = (
            ground(&translation(0.0, 1.7, 0.0)),
            ground(&translation(0.2, 1.7, 0.1)),
        );
        // Texels are 1 m: the snapped centre moves by whole texels or not at all.
        let texels = (a - b).xy() * 100.0 / 2.0;
        for t in texels.iter() {
            assert!((t - t.round()).abs() < 1e-6, "{texels}");
        }
    }
}
