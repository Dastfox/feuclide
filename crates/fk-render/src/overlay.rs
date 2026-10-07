//! The overlay: an image drawn on the CPU, laid over the window's image after everything else,
//! and a small one over it that moves every frame without the large one being drawn again.

use std::sync::Arc;

use bevy_ecs::resource::Resource;

use crate::Gpu;

/// An image laid over the window's image after everything else but the
/// [`TextPanel`](crate::TextPanel), the post effects included:
/// what a menu draws on the CPU (lines, letters), stretched over the whole window. The engine
/// gives it no meaning.
///
/// The pixels are RGBA, 8 bits a channel, rows top to bottom, encoded in sRGB and premultiplied
/// by their alpha (what a CPU rasteriser such as tiny-skia writes). It is uploaded again only
/// when [`set`](Self::set) is called; draw it at the window's size
/// ([`WindowSize`](fk_app::WindowSize)) for one pixel to a pixel.
///
/// Over it, a sprite: a small image of the same kind laid one pixel to a pixel about a point of
/// the window ([`set_sprite`](Self::set_sprite), [`place_sprite`](Self::place_sprite)), moved
/// every frame while the large image is drawn at its own pace: a pointer that keeps up with the
/// mouse.
#[derive(Resource, Clone, Debug, Default)]
pub struct Overlay {
    image: Option<Arc<OverlayImage>>,
    sprite: Option<Arc<OverlayImage>>,
    /// The sprite's centre in pixels from the window's top left corner, and its opacity.
    sprite_at: Option<([f32; 2], f32)>,
}

/// An image of the overlay's.
#[derive(Debug)]
struct OverlayImage {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

impl OverlayImage {
    /// `pixels`, `width × height`, if any.
    ///
    /// # Panics
    ///
    /// If there are not `4 · width · height` bytes.
    fn new(width: u32, height: u32, pixels: Vec<u8>) -> Option<Arc<Self>> {
        assert_eq!(
            pixels.len(),
            4 * width as usize * height as usize,
            "an image of {width}×{height} takes 4 bytes a pixel"
        );
        (width > 0 && height > 0).then(|| {
            Arc::new(Self {
                width,
                height,
                pixels,
            })
        })
    }
}

impl Overlay {
    /// Lays `pixels` over the image: `width × height` premultiplied sRGB RGBA, rows top to
    /// bottom.
    ///
    /// # Panics
    ///
    /// If there are not `4 · width · height` bytes.
    pub fn set(&mut self, width: u32, height: u32, pixels: Vec<u8>) {
        self.image = OverlayImage::new(width, height, pixels);
    }

    /// Takes the overlay away, and the sprite with it.
    pub fn clear(&mut self) {
        self.image = None;
        self.sprite_at = None;
    }

    /// Whether an image is laid over.
    pub fn is_shown(&self) -> bool {
        self.image.is_some()
    }

    /// The sprite's image: `width × height` premultiplied sRGB RGBA, rows top to bottom,
    /// uploaded again only when this is called. It shows once placed.
    ///
    /// # Panics
    ///
    /// If there are not `4 · width · height` bytes.
    pub fn set_sprite(&mut self, width: u32, height: u32, pixels: Vec<u8>) {
        self.sprite = OverlayImage::new(width, height, pixels);
    }

    /// Shows the sprite centred on `centre`, in pixels from the window's top left corner,
    /// `opacity` strong (0 to 1), over the overlay.
    pub fn place_sprite(&mut self, centre: [f32; 2], opacity: f32) {
        self.sprite_at = Some((centre, opacity.clamp(0.0, 1.0)));
    }

    /// Hides the sprite.
    pub fn hide_sprite(&mut self) {
        self.sprite_at = None;
    }
}

const SHADER: &str = r"
struct Out {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct Layer {
    // Left, top, right and bottom, in normalised device coordinates.
    rect: vec4<f32>,
    // Whether the target wants linear colour.
    linear_out: u32,
    opacity: f32,
}

@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var image_sampler: sampler;
@group(0) @binding(2) var<uniform> layer: Layer;

@vertex
fn vertex(@builtin(vertex_index) index: u32) -> Out {
    // Two triangles over the rectangle.
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
    );
    let uv = corners[index];
    var out: Out;
    out.position = vec4<f32>(
        mix(layer.rect.x, layer.rect.z, uv.x),
        mix(layer.rect.y, layer.rect.w, uv.y),
        0.0,
        1.0,
    );
    out.uv = uv;
    return out;
}

fn to_linear(c: vec3<f32>) -> vec3<f32> {
    return select(pow((c + 0.055) / 1.055, vec3<f32>(2.4)), c / 12.92, c <= vec3<f32>(0.04045));
}

@fragment
fn fragment(in: Out) -> @location(0) vec4<f32> {
    var c = textureSample(image, image_sampler, in.uv);
    if layer.linear_out != 0u && c.a > 0.0 {
        // Premultiplied in sRGB: back to straight colour, to linear, premultiplied again.
        c = vec4<f32>(to_linear(c.rgb / c.a) * c.a, c.a);
    }
    return c * layer.opacity;
}
";

/// One image the pass draws: its uniform, and the image uploaded with its texture's bind
/// group, while one is shown.
struct Layer {
    uniform: wgpu::Buffer,
    shown: Option<(Arc<OverlayImage>, wgpu::Texture, wgpu::BindGroup)>,
    /// Whether it is drawn this frame.
    drawn: bool,
}

impl Layer {
    fn new(gpu: &Gpu) -> Self {
        Self {
            uniform: gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fk overlay"),
                size: 32,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            shown: None,
            drawn: false,
        }
    }

    /// Uploads `image` if it is not the one shown, and writes where it goes (`rect`, in
    /// normalised device coordinates) and how strongly.
    fn prepare(
        &mut self,
        gpu: &Gpu,
        (layout, sampler): (&wgpu::BindGroupLayout, &wgpu::Sampler),
        image: Option<&Arc<OverlayImage>>,
        (rect, linear, opacity): ([f32; 4], u32, f32),
    ) {
        let Some(image) = image else {
            self.shown = None;
            self.drawn = false;
            return;
        };
        let mut block = [0u8; 32];
        block[..16].copy_from_slice(bytemuck::cast_slice(&rect));
        block[16..20].copy_from_slice(&linear.to_ne_bytes());
        block[20..24].copy_from_slice(&opacity.to_ne_bytes());
        gpu.queue.write_buffer(&self.uniform, 0, &block);
        self.drawn = opacity > 0.0;
        if let Some((shown, ..)) = &self.shown
            && Arc::ptr_eq(shown, image)
        {
            return;
        }
        let size = wgpu::Extent3d {
            width: image.width,
            height: image.height,
            depth_or_array_layers: 1,
        };
        let texture = match self.shown.take() {
            Some((_, texture, bind_group)) if texture.size() == size => {
                self.shown = Some((image.clone(), texture.clone(), bind_group));
                texture
            }
            _ => {
                let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("fk overlay"),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                let view = texture.create_view(&Default::default());
                let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("fk overlay"),
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
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.uniform.as_entire_binding(),
                        },
                    ],
                });
                self.shown = Some((image.clone(), texture.clone(), bind_group));
                texture
            }
        };
        gpu.queue.write_texture(
            texture.as_image_copy(),
            &image.pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * image.width),
                rows_per_image: Some(image.height),
            },
            size,
        );
    }
}

/// Draws the [`Overlay`] over the window's image, then its sprite, premultiplied alpha
/// blending.
pub(crate) struct OverlayPass {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    /// 1 if the target wants linear colour (it encodes what it is given), else 0.
    linear: u32,
    image: Layer,
    sprite: Layer,
}

impl OverlayPass {
    pub fn new(gpu: &Gpu, output: wgpu::TextureFormat) -> Self {
        let device = &gpu.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fk overlay"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fk overlay"),
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
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
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
            label: Some("fk overlay"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("fk overlay"),
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
                targets: &[Some(wgpu::ColorTargetState {
                    format: output,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("fk overlay"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            pipeline,
            layout,
            sampler,
            linear: u32::from(output.is_srgb()),
            image: Layer::new(gpu),
            sprite: Layer::new(gpu),
        }
    }

    /// Uploads the overlay's image and sprite if they changed since the last frame, and places
    /// the sprite on the window, `size` pixels.
    pub fn prepare(&mut self, gpu: &Gpu, overlay: Option<&Overlay>, size: [u32; 2]) {
        let bindings = (&self.layout, &self.sampler);
        let whole = [-1.0, 1.0, 1.0, -1.0];
        let image = overlay.and_then(|overlay| overlay.image.as_ref());
        self.image
            .prepare(gpu, bindings, image, (whole, self.linear, 1.0));
        let placed =
            overlay.and_then(|overlay| Some((overlay.sprite.as_ref()?, overlay.sprite_at?)));
        let (sprite, rect, opacity) = match placed {
            Some((sprite, ([x, y], opacity))) => {
                let (w, h) = (size[0].max(1) as f32, size[1].max(1) as f32);
                let (half_w, half_h) = (0.5 * sprite.width as f32, 0.5 * sprite.height as f32);
                // Whole pixels, so the sprite is laid one pixel to a pixel.
                let (left, top) = ((x - half_w).round(), (y - half_h).round());
                let ndc = |px: f32, py: f32| [2.0 * px / w - 1.0, 1.0 - 2.0 * py / h];
                let [x0, y0] = ndc(left, top);
                let [x1, y1] = ndc(left + sprite.width as f32, top + sprite.height as f32);
                (Some(sprite), [x0, y0, x1, y1], opacity)
            }
            None => (None, whole, 0.0),
        };
        self.sprite
            .prepare(gpu, bindings, sprite, (rect, self.linear, opacity));
    }

    /// Records the pass over `output`, when an image or the sprite is shown.
    pub fn render(&self, encoder: &mut wgpu::CommandEncoder, output: &wgpu::TextureView) {
        let layers = [&self.image, &self.sprite];
        if !layers.iter().any(|layer| layer.drawn) {
            return;
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("fk overlay"),
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
        pass.set_pipeline(&self.pipeline);
        for layer in layers {
            if let (true, Some((_, _, bind_group))) = (layer.drawn, &layer.shown) {
                pass.set_bind_group(0, bind_group, &[]);
                pass.draw(0..6, 0..1);
            }
        }
    }
}
