//! The settle pass: a fullscreen triangle blending the opaque image with the last ones, found
//! again where the eye's motion took them, after the opaque pass and before the ink (see [`Settle`]).
//!
//! It draws into one of two pairs of targets of its own (the colour, the distance), reading the
//! other pair, the last frame's, then copies its colour over the colour target, so every later
//! pass reads the settled image. Skipped while the image does not settle.

use fk_math::Real;
use fk_math::nalgebra::Matrix4;

use crate::Settle;
use crate::frame::{SettleBlock, uniform_bytes};
use crate::gpu::{Gpu, Vec4f, matrix_columns};

const DISTANCE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

/// The points of their pixels the rays go through, one a frame over this many frames: the
/// first of the Halton sequence in bases 2 and 3. Prime, so that a scene marched one tile in a
/// square of them every so many frames (`RayMarched::interleave`) goes through each in turn.
const JITTERS: u32 = 17;

/// One of the two pairs of targets: what the pass draws into one frame and reads the next.
struct Pair {
    color: wgpu::Texture,
    color_view: wgpu::TextureView,
    distance_view: wgpu::TextureView,
    /// Reading this pair as the last image, with the colour and depth targets.
    reading: wgpu::BindGroup,
}

pub(crate) struct SettlePass {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    sampler: wgpu::Sampler,
    pairs: Option<[Pair; 2]>,
    /// The pair drawn into this frame; the other is the last image.
    drawn: usize,
    /// Whether the other pair holds the last frame's image.
    last: bool,
    /// Whether the image settles this frame.
    on: bool,
    /// Frames settled, for the jitter.
    frame: u32,
}

impl SettlePass {
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
        let module = fk_shaders::compose_with(fk_shaders::Shader::Settle, geometry, defs)
            .unwrap_or_else(|error| panic!("settle shader for {geometry}: {error}"));
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fk settle"),
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
        let unfiltered = wgpu::TextureSampleType::Float { filterable: false };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fk settle"),
            entries: &[
                texture(0, unfiltered, false),
                texture(1, wgpu::TextureSampleType::Depth, multisampled_depth),
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
                texture(
                    3,
                    wgpu::TextureSampleType::Float { filterable: true },
                    false,
                ),
                texture(4, unfiltered, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fk settle"),
            bind_group_layouts: &[Some(frame_layout), Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("fk settle"),
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
                targets: &[Some(color_format.into()), Some(DISTANCE_FORMAT.into())],
            }),
            multiview_mask: None,
            cache: None,
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fk settle"),
            size: uniform_bytes(&block(&Matrix4::identity(), &Settle::default(), false)).len()
                as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("fk settle"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            pipeline,
            layout,
            uniform,
            sampler,
            pairs: None,
            drawn: 0,
            last: false,
            on: false,
            frame: 0,
        }
    }

    /// Points the pass at new targets, after a resize: the last image is gone.
    pub fn set_targets(&mut self, gpu: &Gpu, color: &wgpu::Texture, depth: &wgpu::TextureView) {
        let make = |label| {
            let texture = |format, usage| {
                gpu.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: color.size(),
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING
                        | usage,
                    view_formats: &[],
                })
            };
            let colour = texture(color.format(), wgpu::TextureUsages::COPY_SRC);
            let distance = texture(DISTANCE_FORMAT, wgpu::TextureUsages::empty());
            (colour, distance)
        };
        let (a, a_distance) = make("fk settle a");
        let (b, b_distance) = make("fk settle b");
        let scene = color.create_view(&Default::default());
        let pair = |colour: wgpu::Texture, distance: wgpu::Texture| {
            let color_view = colour.create_view(&Default::default());
            let distance_view = distance.create_view(&Default::default());
            let reading = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fk settle"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&scene),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(depth),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&color_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&distance_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });
            Pair {
                color: colour,
                color_view,
                distance_view,
                reading,
            }
        };
        self.pairs = Some([pair(a, a_distance), pair(b, b_distance)]);
        self.last = false;
    }

    /// Readies this frame: whether the image settles, and how the eye moved since the last
    /// frame (its coordinates now to its coordinates then). The point of their pixels the rays
    /// go through this frame, in pixels from the middle: 0 while the image does not settle.
    pub fn prepare(&mut self, gpu: &Gpu, settle: &Settle, motion: &Matrix4<Real>) -> [f32; 2] {
        self.on = settle.frames > 1.0;
        if !self.on {
            self.last = false;
            return [0.0; 2];
        }
        gpu.queue.write_buffer(
            &self.uniform,
            0,
            &uniform_bytes(&block(motion, settle, self.last)),
        );
        self.frame = self.frame.wrapping_add(1);
        let i = self.frame % JITTERS + 1;
        [halton(i, 2) - 0.5, halton(i, 3) - 0.5]
    }

    /// Settles the colour target, after the opaque pass and before the ink. Does nothing while the
    /// image does not settle.
    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        color: &wgpu::Texture,
        frame: &wgpu::BindGroup,
    ) {
        let Some(pairs) = &self.pairs else {
            return;
        };
        if !self.on {
            return;
        }
        let (drawn, last) = (&pairs[self.drawn], &pairs[1 - self.drawn]);
        {
            let target = |view| {
                Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("fk settle"),
                color_attachments: &[target(&drawn.color_view), target(&drawn.distance_view)],
                ..Default::default()
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, frame, &[]);
            pass.set_bind_group(1, &last.reading, &[]);
            pass.draw(0..3, 0..1);
        }
        encoder.copy_texture_to_texture(
            drawn.color.as_image_copy(),
            color.as_image_copy(),
            color.size(),
        );
        self.drawn = 1 - self.drawn;
        self.last = true;
    }
}

/// The `i`th number of the Halton sequence in base `base`, in [0, 1).
fn halton(mut i: u32, base: u32) -> f32 {
    let mut f = 1.0;
    let mut r = 0.0;
    while i > 0 {
        f /= base as f32;
        r += f * (i % base) as f32;
        i /= base;
    }
    r
}

fn block(motion: &Matrix4<Real>, settle: &Settle, last: bool) -> SettleBlock {
    SettleBlock {
        motion: matrix_columns(motion).map(Vec4f),
        settings: Vec4f([
            (1.0 / settle.frames.max(1.0)) as f32,
            if last { 1.0 } else { 0.0 },
            settle.still.max(0.0) as f32,
            settle.moving.clamp(0.0, 1.0) as f32,
        ]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_jitter_covers_the_pixel() {
        let points: Vec<_> = (1..=JITTERS)
            .map(|i| (halton(i, 2), halton(i, 3)))
            .collect();
        for &(x, y) in &points {
            assert!((0.0..1.0).contains(&x) && (0.0..1.0).contains(&y));
        }
        // Every quarter of the pixel is visited.
        for q in 0..4 {
            let (qx, qy) = ((q % 2) as f32 * 0.5, (q / 2) as f32 * 0.5);
            assert!(
                points
                    .iter()
                    .any(|&(x, y)| x >= qx && x < qx + 0.5 && y >= qy && y < qy + 0.5)
            );
        }
    }
}
