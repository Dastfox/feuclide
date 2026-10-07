//! Another eye: the scene drawn from it before the frame, and the surfaces that show it.

use bevy_ecs::component::Component;
use bevy_ecs::resource::Resource;
use fk_geometry::Geometry;

use crate::Gpu;

/// Another eye the scene is drawn from before each frame, as a resource; the surfaces marked
/// [`ShowsElsewhere`] show what it sees. With no [`eye`](Self::eye) nothing more is drawn, and
/// neither are those surfaces.
///
/// The engine gives it no meaning. Through a portal it is the eye carried across by the
/// portal's deck transform. In a mirror it is the eye reflected in the mirror, but a
/// reflection is not an isometry: set the reflected eye with its own up axis reversed as well
/// (two reflections, an isometry: `mirror ∘ eye ∘ flip_up`) and [`flip`](Self::flip), and the
/// image is turned back the right way up where it is shown.
///
/// It is drawn at the window's size with the frame's camera, sky, light and shadows, and
/// without the post effects, which the frame applies once to everything.
#[derive(Resource)]
pub struct Elsewhere<G: Geometry> {
    /// The other eye's global pose, `None` for none.
    pub eye: Option<G::Isometry>,
    /// Whether its image is shown upside down.
    pub flip: bool,
}

impl<G: Geometry> Default for Elsewhere<G> {
    fn default() -> Self {
        Self {
            eye: None,
            flip: false,
        }
    }
}

impl<G: Geometry> Clone for Elsewhere<G> {
    fn clone(&self) -> Self {
        Self {
            eye: self.eye,
            flip: self.flip,
        }
    }
}

impl<G: Geometry> std::fmt::Debug for Elsewhere<G> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Elsewhere")
            .field("eye", &self.eye)
            .field("flip", &self.flip)
            .finish()
    }
}

/// Draws an entity's meshes as windows onto the [`Elsewhere`] view: each pixel of them shows
/// what the other eye sees at that pixel, times the instance's colour (white for all of it),
/// unlit and unfogged, since that image has its own light and fog. A portal's opening, a
/// mirror's surface. Not drawn while there is no other eye, nor in its own image, and they cast
/// no shadow.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShowsElsewhere;

/// The other eye's image: what the opaque pass draws into, with the frame's formats and sample
/// count, and the bind group the windows read it through.
pub(crate) struct ElsewhereTargets {
    /// The resolved image the windows read.
    pub color: wgpu::TextureView,
    /// The samples, when multisampling, resolved into `color`.
    pub multisampled: Option<wgpu::TextureView>,
    pub depth: wgpu::TextureView,
    pub bind_group: wgpu::BindGroup,
}

/// Group 2 of the window pipeline: the image, its sampler and `(flip, 0, 0, 0)`.
pub(crate) fn layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("fk elsewhere"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

/// What the windows read the image with: its sampler and settings.
pub(crate) struct ElsewhereShared {
    pub layout: wgpu::BindGroupLayout,
    pub sampler: wgpu::Sampler,
    pub settings: wgpu::Buffer,
}

impl ElsewhereShared {
    pub fn new(gpu: &Gpu) -> Self {
        let device = &gpu.device;
        Self {
            layout: layout(device),
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("fk elsewhere"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            settings: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fk elsewhere"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        }
    }

    /// Writes whether the image is shown upside down.
    pub fn write(&self, gpu: &Gpu, flip: bool) {
        let settings = [f32::from(u8::from(flip)), 0.0, 0.0, 0.0];
        gpu.queue
            .write_buffer(&self.settings, 0, bytemuck::cast_slice(&settings));
    }

    /// The bind group reading `color`.
    pub fn bind_group(&self, gpu: &Gpu, color: &wgpu::TextureView) -> wgpu::BindGroup {
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fk elsewhere"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(color),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.settings.as_entire_binding(),
                },
            ],
        })
    }
}
