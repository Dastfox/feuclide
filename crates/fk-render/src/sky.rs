//! The sky pass: a fullscreen triangle drawn last in the opaque pass, where nothing was.

use crate::Gpu;
use crate::effect::SkyFigure;

pub(crate) struct SkyPass {
    pipeline: wgpu::RenderPipeline,
}

impl SkyPass {
    pub fn new(
        gpu: &Gpu,
        geometry: &str,
        frame_layout: &wgpu::BindGroupLayout,
        (color_format, depth_format, sample_count): (wgpu::TextureFormat, wgpu::TextureFormat, u32),
        figure: Option<&SkyFigure>,
    ) -> Self {
        let device = &gpu.device;
        let hooks = fk_shaders::Hooks {
            figure: figure.map(SkyFigure::module),
            ..Default::default()
        };
        let module = fk_shaders::compose_hooked(fk_shaders::Shader::Sky, geometry, &[], &hooks)
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
            // After everything, only where nothing was drawn: the far surface (depth 0, as its
            // vertices are) still there. A sample covered never needed it, nothing in the pass
            // blending, and one left uncovered gets the colour at its pixel's middle as before.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Equal),
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
