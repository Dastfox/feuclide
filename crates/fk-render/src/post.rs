//! The post pass: a fullscreen triangle from the opaque pass's targets to the window.

use crate::frame::{GaugeBlock, PostBlock, uniform_bytes};
use crate::gpu::Vec4f;
use crate::{DiscShape, Gauge, Gpu, PostMode, PostSettings, TextPanel, Warp};

/// Words of the text buffer holding the font: two per glyph, four rows of 8 bits each, the
/// lowest bit leftmost.
const FONT_WORDS: usize = 2 * 128;
/// Words of the text buffer holding the characters, four to a word.
const GRID_WORDS: usize = TextPanel::MAX_LINES * TextPanel::MAX_COLUMNS / 4;

pub(crate) struct PostPass {
    pipeline: wgpu::RenderPipeline,
    /// The [`TextPanel`] alone, blended over the window's finished image.
    text_pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    /// The font, then the [`TextPanel`]'s characters.
    text: wgpu::Buffer,
    /// The lines last written to it.
    written: Vec<String>,
    sampler: wgpu::Sampler,
    /// Same size and format as the colour target; frames are copied in on request.
    afterimage: Option<wgpu::Texture>,
    /// The last [`Afterimage::capture`](crate::Afterimage::capture) served.
    captured: u32,
    bind_group: Option<wgpu::BindGroup>,
    encode_srgb: bool,
}

impl PostPass {
    pub fn new(
        gpu: &Gpu,
        geometry: &str,
        frame_layout: &wgpu::BindGroupLayout,
        output: wgpu::TextureFormat,
        multisampled_depth: bool,
    ) -> Self {
        let device = &gpu.device;
        let defs: &[&str] = if multisampled_depth {
            &["MULTISAMPLED"]
        } else {
            &[]
        };
        let module = fk_shaders::compose_with(fk_shaders::Shader::Post, geometry, defs)
            .unwrap_or_else(|error| panic!("post shader for {geometry}: {error}"));
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fk post"),
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
            label: Some("fk post"),
            entries: &[
                texture(
                    0,
                    wgpu::TextureSampleType::Float { filterable: true },
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                texture(
                    4,
                    wgpu::TextureSampleType::Float { filterable: true },
                    false,
                ),
                texture(
                    5,
                    wgpu::TextureSampleType::Float { filterable: true },
                    false,
                ),
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fk post"),
            bind_group_layouts: &[Some(frame_layout), Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("fk post"),
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
                targets: &[Some(output.into())],
            }),
            multiview_mask: None,
            cache: None,
        });
        let text_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("fk post text"),
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
                entry_point: Some("text_fragment"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: output,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fk post"),
            size: uniform_bytes(&block(&PostSettings::default(), false)).len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let text = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fk post text"),
            size: (4 * (FONT_WORDS + GRID_WORDS)) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue
            .write_buffer(&text, 0, bytemuck::cast_slice(&font_words()));
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("fk post"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            pipeline,
            text_pipeline,
            layout,
            uniform,
            text,
            written: Vec::new(),
            sampler,
            afterimage: None,
            captured: 0,
            bind_group: None,
            encode_srgb: !output.is_srgb(),
        }
    }

    /// Points the pass at new targets, after a resize. The afterimage starts black.
    pub fn set_targets(
        &mut self,
        gpu: &Gpu,
        color: &wgpu::Texture,
        depth: &wgpu::TextureView,
        layer: &wgpu::TextureView,
    ) {
        let afterimage = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("fk afterimage"),
            size: color.size(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: color.format(),
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let color_view = color.create_view(&Default::default());
        let afterimage_view = afterimage.create_view(&Default::default());
        self.bind_group = Some(gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fk post"),
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
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&afterimage_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(layer),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.text.as_entire_binding(),
                },
            ],
        }));
        self.afterimage = Some(afterimage);
    }

    /// Copies the colour target into the afterimage if a capture was asked for since the last
    /// one. Call between the opaque pass and the post pass.
    pub fn capture(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        color: &wgpu::Texture,
        settings: &PostSettings,
    ) {
        let Some(afterimage) = &self.afterimage else {
            return;
        };
        if settings.afterimage.capture == self.captured {
            return;
        }
        self.captured = settings.afterimage.capture;
        encoder.copy_texture_to_texture(
            color.as_image_copy(),
            afterimage.as_image_copy(),
            color.size(),
        );
    }

    pub fn prepare(&mut self, gpu: &Gpu, settings: &PostSettings) {
        let bytes = uniform_bytes(&block(settings, self.encode_srgb));
        gpu.queue.write_buffer(&self.uniform, 0, &bytes);
        if settings.text.lines != self.written {
            self.written.clone_from(&settings.text.lines);
            gpu.queue.write_buffer(
                &self.text,
                (4 * FONT_WORDS) as u64,
                bytemuck::cast_slice(&grid_words(&settings.text.lines)),
            );
        }
    }

    pub fn render(&self, pass: &mut wgpu::RenderPass<'_>, frame: &wgpu::BindGroup) {
        let Some(bind_group) = &self.bind_group else {
            return;
        };
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, frame, &[]);
        pass.set_bind_group(1, bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Records the [`TextPanel`] over `output`, when it has lines: after the overlay, so a
    /// readout stays over a menu drawn on the CPU.
    pub fn render_text(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        output: &wgpu::TextureView,
        frame: &wgpu::BindGroup,
    ) {
        let Some(bind_group) = &self.bind_group else {
            return;
        };
        if self.written.is_empty() {
            return;
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("fk post text"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        pass.set_pipeline(&self.text_pipeline);
        pass.set_bind_group(0, frame, &[]);
        pass.set_bind_group(1, bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

fn block(settings: &PostSettings, encode_srgb: bool) -> PostBlock {
    PostBlock {
        mode: match settings.mode {
            PostMode::Image => 0,
            PostMode::Distance => 1,
        },
        band_spacing: settings.band_spacing as f32,
        encode_srgb: encode_srgb.into(),
        blur: settings.blur.max(0.0) as f32,
        lid_color: Vec4f(settings.lids.color.to_array()),
        lid_closure: settings.lids.closure.clamp(0.0, 1.0) as f32,
        lid_softness: settings.lids.softness as f32,
        tunnel: settings.tunnel.amount.clamp(0.0, 1.0) as f32,
        tunnel_softness: settings.tunnel.softness as f32,
        tunnel_color: Vec4f(settings.tunnel.color.to_array()),
        saturation: settings.saturation as f32,
        swap: settings.swap.clamp(0.0, 1.0) as f32,
        vignette: settings.vignette as f32,
        posterize: settings.posterize as f32,
        afterimage: settings.afterimage.strength as f32,
        lid_curve: settings.lids.curve.clamp(0.0, 0.95) as f32,
        lid_occlusion: settings.lids.occlusion.clamp(0.0, 1.0) as f32,
        gauge_count: settings.gauges.len().min(Gauge::MAX) as u32,
        gauges: std::array::from_fn(|i| settings.gauges.get(i).map(gauge).unwrap_or_default()),
        mark_shape: {
            let mark = &settings.mark;
            let (kind, points) = match mark.shape {
                DiscShape::Disc => (0.0, 0.0),
                DiscShape::Ring => (1.0, 0.0),
                DiscShape::Star { points } => (2.0, points.clamp(1, 16) as f32),
            };
            Vec4f([
                kind,
                points,
                mark.stroke.max(0.0) as f32,
                mark.radius.max(0.0) as f32,
            ])
        },
        mark_color: {
            let [r, g, b, _] = settings.mark.color.to_array();
            Vec4f([r, g, b, settings.mark.opacity.clamp(0.0, 1.0) as f32])
        },
        mark_glow: {
            let [r, g, b, _] = settings.mark.glow.to_array();
            Vec4f([r, g, b, settings.mark.glow_size.max(0.0) as f32])
        },
        mark_flash: settings.mark.flash.clamp(0.0, 1.0) as f32,
        lid_layer: settings.lids.layer.clamp(0.0, 1.0) as f32,
        warp_count: settings.warps.len().min(Warp::MAX) as u32,
        warps: std::array::from_fn(|i| {
            settings.warps.get(i).map_or_else(Vec4f::default, |warp| {
                Vec4f([
                    warp.centre.x as f32,
                    warp.centre.y as f32,
                    warp.radius.max(0.0) as f32,
                    warp.strength as f32,
                ])
            })
        }),
        warp_twists: Vec4f(std::array::from_fn(|i| {
            settings.warps.get(i).map_or(0.0, |warp| warp.twist as f32)
        })),
        mark_centre: Vec4f([
            settings.mark.centre.x as f32,
            settings.mark.centre.y as f32,
            settings.mark.turn as f32,
            settings.mark.ring.max(0.0) as f32,
        ]),
        text_layout: Vec4f([
            settings.text.origin.x as f32,
            settings.text.origin.y as f32,
            settings.text.scale.max(1) as f32,
            0.0,
        ]),
        text_size: {
            let lines = &settings.text.lines;
            let rows = lines.len().min(TextPanel::MAX_LINES);
            let columns = lines[..rows]
                .iter()
                .map(|line| line.chars().count())
                .max()
                .unwrap_or(0)
                .min(TextPanel::MAX_COLUMNS);
            Vec4f([columns as f32, rows as f32, 0.0, 0.0])
        },
        text_color: Vec4f(settings.text.color.to_array()),
        text_background: {
            let [r, g, b, _] = settings.text.background.to_array();
            Vec4f([r, g, b, settings.text.opacity.clamp(0.0, 1.0) as f32])
        },
        mark_ring_color: Vec4f(settings.mark.ring_color.to_array()),
        mark_ring_glow: {
            let [r, g, b, _] = settings.mark.ring_glow.to_array();
            Vec4f([r, g, b, settings.mark.ring_glow_size.max(0.0) as f32])
        },
        streak: Vec4f([
            settings.streak_centre.x as f32,
            settings.streak_centre.y as f32,
            settings.streak.clamp(0.0, 0.9) as f32,
            0.0,
        ]),
        mark_as_sky: if settings.mark.as_sky { 1.0 } else { 0.0 },
    }
}

/// The font's ASCII, packed for the shader.
fn font_words() -> Vec<u32> {
    font8x8::legacy::BASIC_LEGACY
        .iter()
        .flat_map(|rows| {
            let word = |four: &[u8]| {
                four.iter()
                    .rev()
                    .fold(0, |w, &row| (w << 8) | u32::from(row))
            };
            [word(&rows[..4]), word(&rows[4..])]
        })
        .collect()
}

/// `lines` as character codes on a grid [`TextPanel::MAX_COLUMNS`] wide, four to a word, the
/// first character lowest; `?` for what the font does not have.
fn grid_words(lines: &[String]) -> Vec<u32> {
    let mut codes = vec![b' '; 4 * GRID_WORDS];
    for (row, line) in lines.iter().take(TextPanel::MAX_LINES).enumerate() {
        for (column, c) in line.chars().take(TextPanel::MAX_COLUMNS).enumerate() {
            codes[row * TextPanel::MAX_COLUMNS + column] = match c {
                ' '..='~' => c as u8,
                _ => b'?',
            };
        }
    }
    codes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&four| u32::from_le_bytes(four))
        .collect()
}

fn gauge(gauge: &Gauge) -> GaugeBlock {
    let [r, g, b, _] = gauge.color.to_array();
    GaugeBlock {
        rect: Vec4f([
            gauge.origin.x as f32,
            gauge.origin.y as f32,
            gauge.size.x as f32,
            gauge.size.y as f32,
        ]),
        color: Vec4f([r, g, b, gauge.opacity.clamp(0.0, 1.0) as f32]),
        value: gauge.value.clamp(0.0, 1.0) as f32,
        segments: gauge.segments.max(1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_font_is_packed_a_row_a_byte_the_top_row_lowest() {
        let words = font_words();
        assert_eq!(words.len(), FONT_WORDS);
        let a = font8x8::legacy::BASIC_LEGACY[usize::from(b'A')];
        let low = words[2 * usize::from(b'A')];
        let high = words[2 * usize::from(b'A') + 1];
        assert_eq!(low & 0xff, u32::from(a[0]));
        assert_eq!(low >> 24, u32::from(a[3]));
        assert_eq!(high & 0xff, u32::from(a[4]));
        assert_eq!(high >> 24, u32::from(a[7]));
    }

    #[test]
    fn text_lands_on_its_grid_cell() {
        let words = grid_words(&["ab".to_owned(), "\u{e9}".to_owned()]);
        assert_eq!(words.len(), GRID_WORDS);
        assert_eq!(words[0], u32::from_le_bytes(*b"ab  "));
        let second = TextPanel::MAX_COLUMNS / 4;
        assert_eq!(words[second] & 0xff, u32::from(b'?'));
    }
}
