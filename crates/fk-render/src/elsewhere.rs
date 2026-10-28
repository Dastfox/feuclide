//! Another eye: the scene drawn from it before the frame, and the surfaces that show it.

use bevy_ecs::component::Component;
use bevy_ecs::resource::Resource;
use fk_geometry::Geometry;
use fk_math::Real;
use fk_scene::Realm;

use crate::{Gpu, Lighting};

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
/// It is drawn at the window's size with the frame's camera and shadows, the frame's sky and
/// light unless it has a [`lighting`](Self::lighting) of its own, and without the post effects,
/// which the frame applies once to everything.
#[derive(Resource)]
pub struct Elsewhere<G: Geometry> {
    /// The other eye's global pose, `None` for none.
    pub eye: Option<G::Isometry>,
    /// Whether its image is shown upside down.
    pub flip: bool,
    /// The light, the sky and the fog it sees by, in place of the frame's [`Lighting`]: what
    /// lies through a portal under another hour, a mirror showing another sky. `None` (the
    /// default) for the frame's.
    pub lighting: Option<Lighting>,
    /// What it sees is cut off at a plane: a frame whose plane `z = 0` (spanned by the
    /// origin's frame vectors 0 and 1) keeps only what lies on the side its `+z` points to, in
    /// the raster path. A portal's mouth, so that nothing between the eye carried across and
    /// the mouth stands in its way. `None` (the default) for all of it. The ray marcher does
    /// not cut its fields.
    pub clip: Option<G::Isometry>,
    /// The corners of the one window it is seen through, global points in order round it:
    /// only the part of its image they cover in the frame's is drawn, the rest neither drawn
    /// nor culled for, so that a window seen small costs little. `None` (the default) for the
    /// whole image, as a [framed](crate::MeshBuilder::framed) screen needs. Any other window
    /// onto it shows nothing beyond that part.
    pub window: Option<[G::Point; 4]>,
    /// Whether it casts shadows: its own shadow map, drawn again from where it stands, before
    /// its image (`true`, the default). `false` lights what it sees unshadowed and spares the
    /// pass: a window seen small, a night whose shadows cannot be told.
    pub shadows: bool,
    /// How far it sees, in place of the frame's camera's far: what is beyond is neither drawn
    /// nor culled for. `None` (the default) for the frame's. A fog that has swallowed everything
    /// past a distance needs nothing drawn past it.
    pub far: Option<Real>,
    /// The realm it sees, `None` (the default) for the frame eye's.
    pub realm: Option<Realm>,
}

impl<G: Geometry> Default for Elsewhere<G> {
    fn default() -> Self {
        Self {
            eye: None,
            flip: false,
            lighting: None,
            clip: None,
            window: None,
            shadows: true,
            far: None,
            realm: None,
        }
    }
}

impl<G: Geometry> Clone for Elsewhere<G> {
    fn clone(&self) -> Self {
        Self {
            eye: self.eye,
            flip: self.flip,
            lighting: self.lighting.clone(),
            clip: self.clip,
            window: self.window,
            shadows: self.shadows,
            far: self.far,
            realm: self.realm,
        }
    }
}

impl<G: Geometry> std::fmt::Debug for Elsewhere<G> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Elsewhere")
            .field("eye", &self.eye)
            .field("flip", &self.flip)
            .field("lighting", &self.lighting)
            .field("clip", &self.clip)
            .field("window", &self.window)
            .field("shadows", &self.shadows)
            .field("far", &self.far)
            .field("realm", &self.realm)
            .finish()
    }
}

/// How many more eyes [`OtherEyes`] holds, besides the [`Elsewhere`] eye.
pub const MAX_OTHER_EYES: usize = 3;

/// One more eye the scene is drawn from before each frame, as the [`Elsewhere`] eye is (its
/// fields mean the same), shown by the surfaces marked [`ShowsOther`] with its number.
pub struct OtherEye<G: Geometry> {
    /// Its global pose.
    pub eye: G::Isometry,
    /// Whether its image is shown upside down.
    pub flip: bool,
    /// The light, the sky and the fog it sees by; `None` for the frame's.
    pub lighting: Option<Lighting>,
    /// What it sees is cut off at the plane `z = 0` of this frame, keeping its `+z` side, in
    /// the raster path; `None` for all of it.
    pub clip: Option<G::Isometry>,
    /// The corners of the one window it is seen through, as [`Elsewhere::window`]; `None` for
    /// the whole image.
    pub window: Option<[G::Point; 4]>,
    /// Whether it casts shadows, as [`Elsewhere::shadows`].
    pub shadows: bool,
    /// How far it sees, as [`Elsewhere::far`]; `None` for the frame's camera's.
    pub far: Option<Real>,
    /// The realm it sees, `None` for the frame eye's.
    pub realm: Option<Realm>,
}

impl<G: Geometry> OtherEye<G> {
    /// An eye at `eye`, the right way up, under the frame's light, cut off nowhere, its whole
    /// image drawn, its shadows cast, seeing as far as the frame's camera, in the frame eye's
    /// realm.
    pub fn at(eye: G::Isometry) -> Self {
        Self {
            eye,
            flip: false,
            lighting: None,
            clip: None,
            window: None,
            shadows: true,
            far: None,
            realm: None,
        }
    }
}

impl<G: Geometry> Clone for OtherEye<G> {
    fn clone(&self) -> Self {
        Self {
            eye: self.eye,
            flip: self.flip,
            lighting: self.lighting.clone(),
            clip: self.clip,
            window: self.window,
            shadows: self.shadows,
            far: self.far,
            realm: self.realm,
        }
    }
}

impl<G: Geometry> std::fmt::Debug for OtherEye<G> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OtherEye")
            .field("eye", &self.eye)
            .field("flip", &self.flip)
            .field("lighting", &self.lighting)
            .field("clip", &self.clip)
            .field("window", &self.window)
            .field("shadows", &self.shadows)
            .field("far", &self.far)
            .field("realm", &self.realm)
            .finish()
    }
}

/// More eyes the scene is drawn from before each frame, besides the [`Elsewhere`] eye, as a
/// resource: up to [`MAX_OTHER_EYES`], each drawn as that one is (its own shadows, prepare and
/// submit, at the window's size) into an image of its own, which the surfaces marked
/// [`ShowsOther`] with its number show. Several portals seen at once, each its own carried eye,
/// one level deep: no window onto any eye is drawn in any eye's image. An eye's image is only
/// made once the eye is first set; each costs about a frame without its effects.
#[derive(Resource)]
pub struct OtherEyes<G: Geometry>(pub [Option<OtherEye<G>>; MAX_OTHER_EYES]);

impl<G: Geometry> Default for OtherEyes<G> {
    fn default() -> Self {
        Self(std::array::from_fn(|_| None))
    }
}

impl<G: Geometry> Clone for OtherEyes<G> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<G: Geometry> std::fmt::Debug for OtherEyes<G> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("OtherEyes").field(&self.0).finish()
    }
}

/// Draws an entity's meshes as windows onto eye `k` of [`OtherEyes`] (from 0), as
/// [`ShowsElsewhere`] does onto the [`Elsewhere`] eye: not drawn while that eye is not set, nor
/// in any eye's image, casting no shadow. With both, [`ShowsElsewhere`] wins.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShowsOther(pub usize);

/// Draws an entity's meshes as windows onto the [`Elsewhere`] view: each pixel of them shows
/// what the other eye sees at that pixel, times the instance's colour (white for all of it),
/// unlit and unfogged, since that image has its own light and fog. A portal's opening, a
/// mirror's surface. Not drawn while there is no other eye, nor in its own image, and they cast
/// no shadow.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShowsElsewhere;

/// Which eye draws an entity's meshes in the raster path; without it, every eye does. A thing
/// missing from a mirror is seen by the [`Eye`](Self::Eye) only; a thing only a mirror or a
/// portal shows, by the other eyes ([`Elsewhere`](Self::Elsewhere): the [`Elsewhere`] eye and
/// [`OtherEyes`]) only. Seen by the other eyes only, it casts no shadow into the frame's own
/// image.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeenBy {
    /// Only the frame's own eye: left out of the [`Elsewhere`] image.
    Eye,
    /// Only the other eyes, the [`Elsewhere`] eye and [`OtherEyes`]: left out of the frame's
    /// own image.
    Elsewhere,
}

/// An other eye's image: what the opaque pass draws into, with the frame's formats and sample
/// count, and the bind group the windows read it through, with its settings.
pub(crate) struct ElsewhereTargets {
    /// The resolved image the windows read.
    pub color: wgpu::TextureView,
    /// The samples, when multisampling, resolved into `color`.
    pub multisampled: Option<wgpu::TextureView>,
    pub depth: wgpu::TextureView,
    pub bind_group: wgpu::BindGroup,
    /// `(flip, 0, 0, 0)`, read through `bind_group`.
    pub settings: wgpu::Buffer,
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

/// What the windows read an image with: the layout and the sampler.
pub(crate) struct ElsewhereShared {
    pub layout: wgpu::BindGroupLayout,
    pub sampler: wgpu::Sampler,
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
        }
    }

    /// An image's settings, `(flip, 0, 0, 0)`.
    pub fn settings(gpu: &Gpu) -> wgpu::Buffer {
        gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fk elsewhere"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Writes whether the image is shown upside down into its `settings`.
    pub fn write(gpu: &Gpu, settings: &wgpu::Buffer, flip: bool) {
        let values = [f32::from(u8::from(flip)), 0.0, 0.0, 0.0];
        gpu.queue
            .write_buffer(settings, 0, bytemuck::cast_slice(&values));
    }

    /// The bind group reading `color` with `settings`.
    pub fn bind_group(
        &self,
        gpu: &Gpu,
        color: &wgpu::TextureView,
        settings: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
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
                    resource: settings.as_entire_binding(),
                },
            ],
        })
    }
}
