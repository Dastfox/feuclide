//! The crossfade from one station into the next: an [`ImageEffect`] holding the last station's
//! image ([`CarriedImage`]) and blending the live one in over it.

use bevy_ecs::prelude::*;
use bevy_ecs::world::World;
use fk_math::Real;
use fk_render::{
    CarriedImage, EffectFrame, EffectSetup, EffectTargets, Gpu, ImageEffect, uniform_bytes,
};

/// The crossfade's WGSL: the held image, the live one, and how far along.
const SHADER: &str = r"
#import fk::view::view

@group(1) @binding(0) var held: texture_2d<f32>;
@group(1) @binding(1) var live: texture_2d<f32>;
@group(1) @binding(2) var<uniform> fade: vec4<f32>;

@vertex
fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fragment(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(frag.xy);
    let now = textureLoad(live, pixel, 0);
    let was = textureLoad(held, pixel, 0);
    let t = smoothstep(0.0, 1.0, fade.x);
    return vec4<f32>(mix(was.rgb, now.rgb, t), now.a);
}
";

/// The crossfade, as a resource: [`start`](Self::start) it and move
/// [`progress`](Self::progress) from 0 (the held image) to 1 (the live one).
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct Fade {
    /// Change this number to start a crossfade from the [`CarriedImage`] held, or from this
    /// frame's image when none is.
    pub start: u32,
    /// How far along; at 1 or more it is over and costs nothing.
    pub progress: Real,
}

impl Default for Fade {
    fn default() -> Self {
        Self {
            start: 0,
            progress: 1.0,
        }
    }
}

/// The crossfade's pass.
pub struct FadePass {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    /// The held image, what the pass draws into, and the bind group reading them.
    targets: Option<(
        wgpu::Texture,
        wgpu::Texture,
        wgpu::TextureView,
        wgpu::BindGroup,
    )>,
    started: u32,
    /// Whether a crossfade runs, and whether this frame's image is to be held.
    running: bool,
    hold: bool,
}

impl FadePass {
    /// Builds it on the device.
    pub fn new(setup: &EffectSetup<'_>) -> Self {
        let device = &setup.gpu.device;
        let shader = setup.compose_effect(
            fk_shaders::Module {
                path: "fk-game/fade.wgsl",
                source: SHADER,
            },
            &[],
        );
        let texture = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fk-game fade"),
            entries: &[
                texture(0),
                texture(1),
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
        let pipeline = setup.fullscreen_pipeline(
            "fk-game fade",
            &shader,
            &layout,
            &[Some(setup.color_format.into())],
        );
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fk-game fade"),
            size: uniform_bytes(&fk_render::Vec4f::default()).len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipeline,
            layout,
            uniform,
            targets: None,
            started: 0,
            running: false,
            hold: false,
        }
    }

    /// Writes `image` into the held texture; whether it fit.
    fn hold_carried(&self, gpu: &Gpu, image: &fk_render::HeldImage) -> bool {
        let Some((held, ..)) = &self.targets else {
            return false;
        };
        let bytes = held.format().block_copy_size(None).unwrap_or(0);
        let row = image.size()[0] * bytes;
        if image.size() != [held.width(), held.height()]
            || image.pixels().len() != (row * image.size()[1]) as usize
        {
            tracing::warn!("the carried image is not the target's size, not held");
            return false;
        }
        gpu.queue.write_texture(
            held.as_image_copy(),
            image.pixels(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: Some(image.size()[1]),
            },
            held.size(),
        );
        true
    }
}

impl ImageEffect for FadePass {
    fn label(&self) -> &'static str {
        "fade"
    }

    fn set_targets(&mut self, gpu: &Gpu, targets: &EffectTargets<'_>) {
        let texture = |label, usage| {
            gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: targets.color.size(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: targets.color.format(),
                usage,
                view_formats: &[],
            })
        };
        let held = texture(
            "fk-game fade held",
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );
        let drawn = texture(
            "fk-game fade drawn",
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        );
        let view = |texture: &wgpu::Texture| texture.create_view(&Default::default());
        let live = view(targets.color);
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fk-game fade"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view(&held)),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&live),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.uniform.as_entire_binding(),
                },
            ],
        });
        let drawn_view = view(&drawn);
        self.targets = Some((held, drawn, drawn_view, bind_group));
        self.running = false;
    }

    fn prepare(&mut self, gpu: &Gpu, world: &mut World, _: &EffectFrame) {
        let fade = world.get_resource::<Fade>().copied().unwrap_or_default();
        let start = fade.start != self.started;
        self.started = fade.start;
        self.running = fade.progress < 1.0 && (self.running || start);
        self.hold = false;
        if start && self.running {
            let carried = world
                .get_resource_mut::<CarriedImage>()
                .and_then(|mut carried| carried.take_held());
            self.hold = !carried.is_some_and(|image| self.hold_carried(gpu, &image));
        }
        if self.running {
            let progress = fade.progress.clamp(0.0, 1.0) as f32;
            gpu.queue.write_buffer(
                &self.uniform,
                0,
                &uniform_bytes(&fk_render::Vec4f([progress, 0.0, 0.0, 0.0])),
            );
        }
    }

    fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        targets: &EffectTargets<'_>,
        frame: &wgpu::BindGroup,
    ) {
        let Some((held, drawn, drawn_view, bind_group)) = &self.targets else {
            return;
        };
        if !self.running {
            return;
        }
        if self.hold {
            // The first frame is the held one, as it is.
            encoder.copy_texture_to_texture(
                targets.color.as_image_copy(),
                held.as_image_copy(),
                held.size(),
            );
            return;
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("fk-game fade"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: drawn_view,
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
            drawn.as_image_copy(),
            targets.color.as_image_copy(),
            drawn.size(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_crossfade_composes_in_every_geometry() {
        let skeleton = fk_shaders::Module {
            path: "fk-game/fade.wgsl",
            source: SHADER,
        };
        fk_render::validate_effect::<fk_geometry_euclidean::E3>(skeleton, &[]).unwrap();
        fk_render::validate_effect::<fk_geometry_hyperbolic::H3>(skeleton, &[]).unwrap();
        fk_render::validate_effect::<fk_geometry_spherical::S3>(skeleton, &[]).unwrap();
    }
}
