//! The ink pass: a fullscreen triangle laying the [`Ink`] over the opaque image, before the mosh,
//! so a moshed image carries it like the rest of its colours.
//!
//! It draws into a texture of its own from the colour and depth targets, then copies it over the
//! colour target, so every later pass reads the inked image. Skipped while the ink is off.

use crate::frame::{InkBlock, uniform_bytes};
use crate::gpu::Vec4f;
use crate::{Gpu, Ink};

pub(crate) struct InkPass {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    /// What the pass draws into, the same size and format as the colour target, and the bind
    /// group reading the targets.
    target: Option<(wgpu::Texture, wgpu::TextureView, wgpu::BindGroup)>,
    /// Whether the ink is on this frame.
    on: bool,
}

impl InkPass {
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
        let module = fk_shaders::compose_with(fk_shaders::Shader::Ink, geometry, defs)
            .unwrap_or_else(|error| panic!("ink shader for {geometry}: {error}"));
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fk ink"),
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
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fk ink"),
            entries: &[
                texture(
                    0,
                    wgpu::TextureSampleType::Float { filterable: false },
                    false,
                ),
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
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fk ink"),
            bind_group_layouts: &[Some(frame_layout), Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("fk ink"),
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
            label: Some("fk ink"),
            size: uniform_bytes(&block(&Ink::default())).len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipeline,
            layout,
            uniform,
            target: None,
            on: false,
        }
    }

    /// Points the pass at new targets, after a resize.
    pub fn set_targets(&mut self, gpu: &Gpu, color: &wgpu::Texture, depth: &wgpu::TextureView) {
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("fk ink"),
            size: color.size(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: color.format(),
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let color_view = color.create_view(&Default::default());
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fk ink"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(depth),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.uniform.as_entire_binding(),
                },
            ],
        });
        self.target = Some((texture, view, bind_group));
    }

    pub fn prepare(&mut self, gpu: &Gpu, ink: &Ink) {
        self.on = ink.amount > 0.0;
        if self.on {
            gpu.queue
                .write_buffer(&self.uniform, 0, &uniform_bytes(&block(ink)));
        }
    }

    /// Inks the colour target, after the opaque pass and before the mosh. Does nothing while
    /// the ink is off.
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        color: &wgpu::Texture,
        frame: &wgpu::BindGroup,
    ) {
        let Some((texture, view, bind_group)) = &self.target else {
            return;
        };
        if !self.on {
            return;
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("fk ink"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
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
            pass.set_bind_group(1, bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        encoder.copy_texture_to_texture(
            texture.as_image_copy(),
            color.as_image_copy(),
            color.size(),
        );
    }
}

fn block(ink: &Ink) -> InkBlock {
    let [r, g, b, _] = ink.color.to_array();
    InkBlock {
        ink: Vec4f([
            ink.amount.clamp(0.0, 1.0) as f32,
            ink.width.max(0.5) as f32,
            ink.depth.max(1e-4) as f32,
            ink.edges.max(1e-4) as f32,
        ]),
        color: Vec4f([r, g, b, ink.hatch.clamp(0.0, 1.0) as f32]),
        more: Vec4f([
            ink.fade.max(1.0) as f32,
            ink.spacing.max(2.0) as f32,
            ink.shadow.clamp(0.0, 1.0) as f32,
            ink.paper_amount.clamp(0.0, 1.0) as f32,
        ]),
        paper: Vec4f(ink.paper.to_array()),
    }
}
