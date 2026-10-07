//! The sky pass: a fullscreen triangle drawn first in the opaque pass.

use crate::Gpu;

pub(crate) struct SkyPass {
    pipeline: wgpu::RenderPipeline,
}

impl SkyPass {
    pub fn new(
        gpu: &Gpu,
        geometry: &str,
        frame_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> Self {
        let device = &gpu.device;
        let module = fk_shaders::compose(fk_shaders::Shader::Sky, geometry)
            .unwrap_or_else(|error| panic!("sky shader for {geometry}: {error}"));
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fk sky"),
            source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(module)),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fk sky"),
            bind_group_layouts: &[Some(frame_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("fk sky"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vertex"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            // Behind everything: it neither tests nor writes depth, which stays at the far
            // surface for the renderers drawn after it.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: sample_count,
                ..Default::default()
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fragment"),
                compilation_options: Default::default(),
                targets: &[Some(color_format.into())],
            }),
            multiview_mask: None,
            cache: None,
        });
        Self { pipeline }
    }

    pub fn render(&self, pass: &mut wgpu::RenderPass<'_>, frame: &wgpu::BindGroup) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, frame, &[]);
        pass.draw(0..3, 0..1);
    }
}
