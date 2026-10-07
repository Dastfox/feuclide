//! The mosh pass: a fullscreen triangle that rebuilds the image from the held one, moved by the
//! live image's motion, then takes the opaque image's place. Between moshes it runs the
//! [`MoshPatch`]es the same way, the live image everywhere outside them.
//!
//! The mosh lives in one of two textures. Each frame it is redrawn into the other, from itself,
//! the live image and its depth, then copied over the colour target, so every later pass reads
//! it as the image.
//!
//! A [`CarriedImage`] takes the image out of one app and lets the next one's mosh start on it.

use bevy_ecs::resource::Resource;
use fk_math::Real;
use fk_math::nalgebra::Matrix4;

use crate::frame::{MoshBlock, uniform_bytes};
use crate::gpu::{Vec4f, matrix_columns};
use crate::{Gpu, Mosh, MoshKind, MoshPatch};

/// The moshed image of one frame, read back to the CPU: what the [`Mosh`] holds, linear HDR
/// before the post pass, in the colour target's format.
#[derive(Clone, Debug)]
pub struct HeldImage {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

impl HeldImage {
    /// Its width and height, in pixels.
    pub fn size(&self) -> [u32; 2] {
        [self.width, self.height]
    }

    /// An image drawn on the CPU, as a mosh would hold it: `width × height` sRGB RGBA, 8 bits a
    /// channel, rows top to bottom, premultiplied (what an [`Overlay`](crate::Overlay) takes),
    /// laid over black and decoded to linear. Its alpha is 0, as the sky's.
    ///
    /// # Panics
    ///
    /// If there are not `4 · width · height` bytes.
    pub fn from_srgb(width: u32, height: u32, pixels: &[u8]) -> Self {
        assert_eq!(
            pixels.len(),
            4 * width as usize * height as usize,
            "an image of {width}×{height} takes 4 bytes a pixel"
        );
        // Premultiplied over black is the colour itself; decoded once per level.
        let linear: Vec<[u8; 2]> = (0..=255u8)
            .map(|c| f16_bits(crate::color::decode(f32::from(c) / 255.0)).to_le_bytes())
            .collect();
        let mut out = Vec::with_capacity(8 * pixels.len() / 4);
        for pixel in pixels.as_chunks::<4>().0 {
            for &c in &pixel[..3] {
                out.extend_from_slice(&linear[usize::from(c)]);
            }
            out.extend_from_slice(&[0, 0]);
        }
        Self {
            width,
            height,
            pixels: out,
        }
    }
}

/// `x` (finite, not negative) as a half float's bits, rounded to the nearest; below the
/// smallest normal half, 0.
fn f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    if x <= 0.0 || exponent <= 0 {
        return 0;
    }
    if exponent >= 0x1f {
        return 0x7bff;
    }
    let mantissa = bits & 0x7f_ffff;
    let half = ((exponent as u32) << 10) | (mantissa >> 13);
    // Rounded: the next half up when the dropped bits are over half of one.
    (half + ((mantissa >> 12) & 1)) as u16
}

/// Carries the image across a [`Handover`](fk_app::Handover), as a resource: the old app
/// [`request`](Self::request)s a frame's image and [`take`](Self::take)s it on the next frame;
/// the new app is given it with [`hold`](Self::hold), and the first [`Mosh`] it starts holds
/// that image instead of its own first frame, so its scene moshes in through the old one's.
/// The engine gives it no meaning.
#[derive(Resource, Clone, Debug, Default)]
pub struct CarriedImage {
    requested: bool,
    read: Option<HeldImage>,
    held: Option<HeldImage>,
}

impl CarriedImage {
    /// Reads this frame's image back once it is drawn, after the ink and the mosh (blocking
    /// until the GPU is done).
    pub fn request(&mut self) {
        self.requested = true;
    }

    /// The image read back since the last request, if it was drawn.
    pub fn take(&mut self) -> Option<HeldImage> {
        self.read.take()
    }

    /// Has the next mosh to start hold `image`, if it is the size of the image then.
    pub fn hold(&mut self, image: HeldImage) {
        self.held = Some(image);
    }

    pub(crate) fn take_request(&mut self) -> bool {
        std::mem::take(&mut self.requested)
    }

    pub(crate) fn set_read(&mut self, width: u32, height: u32, pixels: Vec<u8>) {
        self.read = Some(HeldImage {
            width,
            height,
            pixels,
        });
    }

    pub(crate) fn take_held(&mut self) -> Option<HeldImage> {
        self.held.take()
    }
}

pub(crate) struct MoshPass {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    /// The two textures and their views, the same size and format as the colour target.
    buffers: Vec<(wgpu::Texture, wgpu::TextureView)>,
    /// `bind_groups[i]` reads the mosh from `buffers[i]`, and the live image and its depth from
    /// the targets.
    bind_groups: Vec<wgpu::BindGroup>,
    /// Which buffer holds the mosh.
    current: usize,
    /// The last [`Mosh::start`] and [`Mosh::cut`] served.
    started: u32,
    cut: u32,
    /// Whether a mosh runs.
    running: bool,
    /// Whether patches run, outside a mosh.
    patching: bool,
    /// Whether this frame's image is to be held, starting the mosh.
    hold: bool,
}

impl MoshPass {
    pub fn new(
        gpu: &Gpu,
        geometry: &str,
        frame_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
        multisampled_depth: bool,
    ) -> Self {
        let device = &gpu.device;
        let defs: &[&str] = if multisampled_depth {
            &["MULTISAMPLED"]
        } else {
            &[]
        };
        let module = fk_shaders::compose_with(fk_shaders::Shader::Mosh, geometry, defs)
            .unwrap_or_else(|error| panic!("mosh shader for {geometry}: {error}"));
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fk mosh"),
            source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(module)),
        });
        let texture = |binding, sample_type, multisampled| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled,
            },
            count: None,
        };
        let color = wgpu::TextureSampleType::Float { filterable: false };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fk mosh"),
            entries: &[
                texture(0, color, false),
                texture(1, color, false),
                texture(2, wgpu::TextureSampleType::Depth, multisampled_depth),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fk mosh"),
            bind_group_layouts: &[Some(frame_layout), Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("fk mosh"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vertex"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fragment"),
                compilation_options: Default::default(),
                targets: &[Some(color_format.into())],
            }),
            multiview_mask: None,
            cache: None,
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fk mosh"),
            size: uniform_bytes(&block(
                (&Mosh::default(), &[]),
                &Matrix4::identity(),
                [1.0, 1.0],
                0.0,
            ))
            .len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipeline,
            layout,
            uniform,
            buffers: Vec::new(),
            bind_groups: Vec::new(),
            current: 0,
            started: 0,
            cut: 0,
            running: false,
            patching: false,
            hold: false,
        }
    }

    /// Points the pass at new targets, after a resize. A running mosh ends.
    pub fn set_targets(&mut self, gpu: &Gpu, color: &wgpu::Texture, depth: &wgpu::TextureView) {
        let color_view = color.create_view(&Default::default());
        self.buffers = (0..2)
            .map(|_| {
                let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("fk mosh"),
                    size: color.size(),
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: color.format(),
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::COPY_SRC
                        | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                let view = texture.create_view(&Default::default());
                (texture, view)
            })
            .collect();
        self.bind_groups = self
            .buffers
            .iter()
            .map(|(_, held)| {
                let entry = |binding, view| wgpu::BindGroupEntry {
                    binding,
                    resource: wgpu::BindingResource::TextureView(view),
                };
                gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("fk mosh"),
                    layout: &self.layout,
                    entries: &[
                        entry(0, held),
                        entry(1, &color_view),
                        entry(2, depth),
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self.uniform.as_entire_binding(),
                        },
                    ],
                })
            })
            .collect();
        self.running = false;
        self.patching = false;
        self.hold = false;
    }

    /// Whether `mosh` starts a mosh this frame.
    pub fn starts(&self, mosh: &Mosh) -> bool {
        mosh.start != self.started && mosh.progress < 1.0
    }

    /// Decides what this frame does. `motion` takes this frame's eye coordinates to the last
    /// frame's; it is ignored on the frame [`Mosh::cut`] changes. `size` is the target's, in
    /// pixels. A mosh that starts holds `carried` rather than this frame's image, when it is
    /// the target's size.
    pub fn prepare(
        &mut self,
        gpu: &Gpu,
        (mosh, patches): (&Mosh, &[MoshPatch]),
        motion: &Matrix4<Real>,
        size: [f32; 2],
        delta: f32,
        carried: Option<HeldImage>,
    ) {
        let start = mosh.start != self.started;
        self.started = mosh.start;
        let cut = mosh.cut != self.cut;
        self.cut = mosh.cut;
        if mosh.progress >= 1.0 {
            self.running = false;
        } else if start {
            self.running = true;
        }
        let patching = !self.running && patches.iter().any(|patch| patch.radius > 0.0);
        let carried = carried
            .filter(|_| start && self.running)
            .is_some_and(|image| self.hold_carried(gpu, &image));
        // A mosh starts on this frame's image, or the carried one; patches start on it too,
        // and again after a mosh, whose last image is not this one.
        self.hold = (start && self.running && !carried) || (patching && !self.patching);
        self.patching = patching;
        if (self.running || self.patching) && !self.hold {
            let motion = if cut { Matrix4::identity() } else { *motion };
            let mosh = if self.running {
                *mosh
            } else {
                Mosh {
                    progress: 1.0,
                    ..*mosh
                }
            };
            let bytes = uniform_bytes(&block((&mosh, patches), &motion, size, delta));
            gpu.queue.write_buffer(&self.uniform, 0, &bytes);
        }
    }

    /// Writes `image` into the buffer that holds the mosh; returns whether it fit.
    fn hold_carried(&self, gpu: &Gpu, image: &HeldImage) -> bool {
        let Some((texture, _)) = self.buffers.get(self.current) else {
            return false;
        };
        let bytes = texture.format().block_copy_size(None).unwrap_or(0);
        let row = image.width * bytes;
        if image.size() != [texture.width(), texture.height()]
            || image.pixels.len() != (row * image.height) as usize
        {
            tracing::warn!("the carried image is not the target's size, not held");
            return false;
        }
        gpu.queue.write_texture(
            texture.as_image_copy(),
            &image.pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: Some(image.height),
            },
            texture.size(),
        );
        true
    }

    /// Holds or moshes the colour target, after the opaque pass. Does nothing unless a mosh
    /// or patches run.
    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        color: &wgpu::Texture,
        frame: &wgpu::BindGroup,
    ) {
        if !(self.running || self.patching) || self.buffers.len() != 2 {
            return;
        }
        if self.hold {
            // The first frame is the held one, unchanged.
            encoder.copy_texture_to_texture(
                color.as_image_copy(),
                self.buffers[self.current].0.as_image_copy(),
                color.size(),
            );
            return;
        }
        let next = 1 - self.current;
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("fk mosh"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.buffers[next].1,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, frame, &[]);
            pass.set_bind_group(1, &self.bind_groups[self.current], &[]);
            pass.draw(0..3, 0..1);
        }
        encoder.copy_texture_to_texture(
            self.buffers[next].0.as_image_copy(),
            color.as_image_copy(),
            color.size(),
        );
        self.current = next;
    }
}

fn block(
    (mosh, patches): (&Mosh, &[MoshPatch]),
    motion: &Matrix4<Real>,
    [width, height]: [f32; 2],
    delta: f32,
) -> MoshBlock {
    let patches: Vec<_> = patches
        .iter()
        .filter(|patch| patch.radius > 0.0)
        .take(MoshPatch::MAX)
        .collect();
    MoshBlock {
        motion: matrix_columns(motion).map(Vec4f),
        progress: mosh.progress.clamp(0.0, 1.0) as f32,
        block: (mosh.block as f32 * height).max(1.0),
        bleed: mosh.bleed.max(0.0) as f32,
        delta,
        seed: (mosh.start % 1024) as f32,
        patch_count: patches.len() as u32,
        patches: std::array::from_fn(|i| {
            patches.get(i).map_or_else(Vec4f::default, |patch| {
                Vec4f([
                    0.5 * width + patch.centre.x as f32 * height,
                    (0.5 - patch.centre.y as f32) * height,
                    patch.radius as f32 * height,
                    (patch.block as f32 * height).max(1.0),
                ])
            })
        }),
        patch_motion: std::array::from_fn(|i| {
            patches.get(i).map_or_else(Vec4f::default, |patch| {
                Vec4f([
                    patch.flow as f32 * height,
                    patch.refresh.max(0.0) as f32,
                    patch.bleed.max(0.0) as f32,
                    i as f32,
                ])
            })
        }),
        patch_arms: std::array::from_fn(|i| {
            patches.get(i).map_or_else(Vec4f::default, |patch| {
                Vec4f([
                    patch.arms as f32,
                    patch.reach.max(0.0) as f32 * height,
                    patch.arm_width.max(0.0) as f32 * height,
                    patch.curl as f32,
                ])
            })
        }),
        patch_turn: std::array::from_fn(|i| {
            patches.get(i).map_or_else(Vec4f::default, |patch| {
                Vec4f([patch.turn as f32, patch.wave as f32, 0.0, 0.0])
            })
        }),
        kind: match mosh.kind {
            MoshKind::Blocks => 0,
            MoshKind::Pixels => 1,
            MoshKind::Smear => 2,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halves_are_encoded() {
        assert_eq!(f16_bits(0.0), 0);
        assert_eq!(f16_bits(1.0), 0x3c00);
        assert_eq!(f16_bits(0.5), 0x3800);
        assert_eq!(f16_bits(0.333_333), 0x3555);
        let image = HeldImage::from_srgb(1, 1, &[255, 0, 128, 255]);
        assert_eq!(image.pixels[..2], 0x3c00u16.to_le_bytes());
        assert_eq!(image.pixels.len(), 8);
    }
}
