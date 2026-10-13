//! The sky image: a picture hung on the sky in a direction from the eye, drawn last in the
//! opaque pass, so the ink and the mosh carry it as they carry the rest of the image.

use std::sync::Arc;

use bevy_ecs::resource::Resource;
use fk_math::Real;
use fk_math::nalgebra::Vector3;

use crate::{Color, Gpu};

/// A picture hung on the sky, as a resource: what a CPU rasteriser drew (letters, lines),
/// seen in a fixed direction whichever way the eye turns, at a fixed angular size, like the sun.
/// It is drawn over everything the opaque pass drew, before the ink and the [`Mosh`]
/// (and its [`MoshPatch`]es), so whatever datamoshes the image datamoshes it too. The engine
/// gives it no meaning.
///
/// The pixels are RGBA, 8 bits a channel, rows top to bottom, encoded in sRGB and premultiplied
/// by their alpha, as an [`Overlay`](crate::Overlay) takes them; uploaded again only when
/// [`set`](Self::set) is called. It is projected onto the sky gnomonically about its centre
/// (straight lines stay straight), its width [`width`](Self::width) radians across and its
/// height in proportion. Its rows can tear sideways, a few at a time, as a broken video's do
/// ([`tear`](Self::tear)).
///
/// [`Mosh`]: crate::Mosh
/// [`MoshPatch`]: crate::MoshPatch
#[derive(Resource, Clone, Debug)]
pub struct SkyImage {
    image: Option<Arc<SkyPixels>>,
    /// Where its centre is seen, a direction at the root's origin (like a [`SkyDisc`]'s).
    ///
    /// [`SkyDisc`]: crate::SkyDisc
    pub direction: Vector3<Real>,
    /// Which way is up on it, a direction at the root's origin: its part across
    /// [`direction`](Self::direction) is the image's up.
    pub up: Vector3<Real>,
    /// The angle it spans across, radians.
    pub width: Real,
    /// How much it shows, 0 for not at all.
    pub opacity: Real,
    /// Its colours times this, linear: brighter than 1 to glow.
    pub tint: Color,
    /// How far a torn row is thrown sideways at most, as a share of its width; 0 for none.
    pub tear: Real,
    /// How many times a second the torn rows are drawn again.
    pub tear_rate: Real,
    /// How many rows it tears in, top to bottom.
    pub tear_rows: u32,
}

impl Default for SkyImage {
    fn default() -> Self {
        Self {
            image: None,
            direction: -Vector3::z(),
            up: Vector3::y(),
            width: 0.5,
            opacity: 0.0,
            tint: Color::WHITE,
            tear: 0.0,
            tear_rate: 12.0,
            tear_rows: 24,
        }
    }
}

/// The picture of a [`SkyImage`].
#[derive(Debug)]
struct SkyPixels {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

impl SkyImage {
    /// Hangs `pixels` on the sky: `width × height` premultiplied sRGB RGBA, rows top to
    /// bottom.
    ///
    /// # Panics
    ///
    /// If there are not `4 · width · height` bytes.
    pub fn set(&mut self, width: u32, height: u32, pixels: Vec<u8>) {
        assert_eq!(
            pixels.len(),
            4 * width as usize * height as usize,
            "an image of {width}×{height} takes 4 bytes a pixel"
        );
        self.image = (width > 0 && height > 0).then(|| {
            Arc::new(SkyPixels {
                width,
                height,
                pixels,
            })
        });
    }

    /// Takes the picture down.
    pub fn clear(&mut self) {
        self.image = None;
    }

    /// Whether a picture is hung and shows.
    pub fn is_shown(&self) -> bool {
        self.image.is_some() && self.opacity > 0.0
    }

    /// Its width over its height, 1 without a picture.
    pub fn aspect(&self) -> Real {
        self.image.as_ref().map_or(1.0, |image| {
            Real::from(image.width) / Real::from(image.height)
        })
    }
}

const SHADER: &str = r"
struct Hung {
    // Its centre in the eye's frame, and its opacity.
    centre: vec4<f32>,
    // Its right and up across the centre, each with the tangent of half its span that way.
    right: vec4<f32>,
    up: vec4<f32>,
    // The projection's scale, and the target's size in pixels.
    view: vec4<f32>,
    // How far rows tear, how often, how many rows, and the time.
    tear: vec4<f32>,
    tint: vec4<f32>,
}

@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var image_sampler: sampler;
@group(0) @binding(2) var<uniform> hung: Hung;

@vertex
fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn to_linear(c: vec3<f32>) -> vec3<f32> {
    return select(pow((c + 0.055) / 1.055, vec3<f32>(2.4)), c / 12.92, c <= vec3<f32>(0.04045));
}

fn hash2(p: vec2<f32>) -> vec2<f32> {
    var q = fract(vec3<f32>(p.x, p.y, p.x) * vec3<f32>(0.1031, 0.1030, 0.0973));
    q += dot(q, q.yzx + 33.33);
    return fract((q.xx + q.yz) * q.zy);
}

@fragment
fn fragment(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let ndc = vec2<f32>(frag.x / hung.view.z * 2.0 - 1.0, 1.0 - frag.y / hung.view.w * 2.0);
    let dir = normalize(vec3<f32>(ndc / hung.view.xy, -1.0));
    let c = dot(dir, hung.centre.xyz);
    if c <= 1e-3 {
        discard;
    }
    // Gnomonic: straight lines on the sky stay straight on it.
    let p = vec2<f32>(dot(dir, hung.right.xyz), dot(dir, hung.up.xyz)) / c;
    var uv = vec2<f32>(0.5 + 0.5 * p.x / hung.right.w, 0.5 - 0.5 * p.y / hung.up.w);
    // Torn rows, drawn again a few times a second; now and then a block of a row dropped.
    let rows = max(hung.tear.z, 1.0);
    let row = floor(uv.y * rows);
    let frame = floor(hung.tear.w * hung.tear.y);
    let h = hash2(vec2<f32>(row + 17.0, frame));
    if h.x < 0.22 {
        uv.x += (h.y - 0.5) * 2.0 * hung.tear.x;
    }
    let block = floor(uv.x * rows * hung.right.w / hung.up.w);
    let gone = hash2(vec2<f32>(block * 7.0 + row, frame + 3.0));
    if gone.x < 0.04 * step(1e-4, hung.tear.x) {
        uv.y += 1.0 / rows;
    }
    if any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) {
        discard;
    }
    var color = textureSampleLevel(image, image_sampler, uv, 0.0);
    if color.a <= 0.0 {
        discard;
    }
    let a = color.a * hung.centre.w;
    let rgb = to_linear(color.rgb / color.a) * hung.tint.rgb;
    return vec4<f32>(rgb * a, a);
}
";

/// Draws the [`SkyImage`] over the opaque image, premultiplied, the image's alpha left as the
/// surfaces made it.
pub(crate) struct SkyImagePass {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    shown: Option<(Arc<SkyPixels>, wgpu::Texture, wgpu::BindGroup)>,
    drawn: bool,
}

/// Bytes of the pass's uniform block: six `vec4`s.
const BLOCK: usize = 6 * 16;

impl SkyImagePass {
    pub fn new(
        gpu: &Gpu,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> Self {
        let device = &gpu.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fk sky image"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fk sky image"),
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
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fk sky image"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("fk sky image"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vertex"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            // Over everything drawn: it neither tests nor writes depth.
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
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    // The image's alpha says what was drawn there (the sky is 0): kept.
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::Zero,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("fk sky image"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fk sky image"),
            size: BLOCK as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipeline,
            layout,
            sampler,
            uniform,
            shown: None,
            drawn: false,
        }
    }

    /// Uploads the picture if it is not the one shown, and writes where it hangs: `axes` its
    /// centre, right and up in the eye's frame, `proj_scale` and `size` the frame's projection
    /// and target, `time` the frame's.
    pub fn prepare(
        &mut self,
        gpu: &Gpu,
        hung: Option<&SkyImage>,
        axes: Option<[Vector3<Real>; 3]>,
        (proj_scale, size, time): ([f32; 2], [f32; 2], f32),
    ) {
        let (Some(hung), Some([centre, right, up])) = (hung.filter(|h| h.is_shown()), axes) else {
            self.drawn = false;
            return;
        };
        let Some(image) = &hung.image else {
            self.drawn = false;
            return;
        };
        let half_w = (hung.width.clamp(1e-4, 3.0) / 2.0).tan();
        let half_h = half_w / hung.aspect().max(1e-6);
        let v = |v: &Vector3<Real>, w: Real| [v.x as f32, v.y as f32, v.z as f32, w as f32];
        let tint = hung.tint.to_array();
        let block: [[f32; 4]; 6] = [
            v(&centre, hung.opacity.clamp(0.0, 1.0)),
            v(&right, half_w),
            v(&up, half_h),
            [proj_scale[0], proj_scale[1], size[0], size[1]],
            [
                hung.tear.max(0.0) as f32,
                hung.tear_rate.max(0.0) as f32,
                hung.tear_rows.max(1) as f32,
                time,
            ],
            [tint[0], tint[1], tint[2], 1.0],
        ];
        gpu.queue
            .write_buffer(&self.uniform, 0, bytemuck::cast_slice(block.as_flattened()));
        self.drawn = true;
        if let Some((shown, ..)) = &self.shown
            && Arc::ptr_eq(shown, image)
        {
            return;
        }
        let extent = wgpu::Extent3d {
            width: image.width,
            height: image.height,
            depth_or_array_layers: 1,
        };
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("fk sky image"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fk sky image"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.uniform.as_entire_binding(),
                },
            ],
        });
        gpu.queue.write_texture(
            texture.as_image_copy(),
            &image.pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * image.width),
                rows_per_image: Some(image.height),
            },
            extent,
        );
        self.shown = Some((image.clone(), texture, bind_group));
    }

    /// Draws it into the opaque pass, when it shows.
    pub fn render(&self, pass: &mut wgpu::RenderPass<'_>) {
        if let (true, Some((_, _, bind_group))) = (self.drawn, &self.shown) {
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}
