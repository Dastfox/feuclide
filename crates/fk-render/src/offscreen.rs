//! Drawing a scene into an image, without a window: the raster and ray-march paths into one
//! target, for tests and the golden images (`tools/`).
//!
//! Only the opaque pass is drawn: no sky, no post effects, no shadows, no other eye. The target
//! is cleared to the sky's horizon colour and stored as sRGB, so what is read back is the
//! image as the eye would see it before the post pass. Entities are placed by their
//! [`GlobalPose`]; [`Offscreen::draw`] makes them relative to the eye itself.

use bevy_ecs::entity::Entity;
use bevy_ecs::world::World;
use fk_app::WindowSize;
use fk_geometry::{GpuGeometry, GroupElement};
use fk_math::Real;
use fk_scene::{ActiveView, Camera, GlobalPose, ViewRelative};

use crate::cull::Frustum;
use crate::elsewhere::ElsewhereShared;
use crate::graph::{FrameBuffers, frame_blocks, frame_layout};
use crate::renderer::{FrameBindings, FrameView, GeometryRenderer, RendererSetup};
use crate::shadow::{SHADOW_FORMAT, ShadowMap};
use crate::{Gpu, Lighting, RasterPipeline, RayMarchPipeline};

/// Which ways of drawing surfaces draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Paths {
    /// Meshes only.
    Raster,
    /// Distance fields only.
    RayMarch,
    /// Both, the ray marcher after the raster pipeline, as in a frame.
    Both,
}

/// An image read back: RGBA8, sRGB, row after row from the top.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    /// Pixels across.
    pub width: u32,
    /// Pixels down.
    pub height: u32,
    /// Four bytes a pixel.
    pub rgba: Vec<u8>,
}

impl Image {
    /// The pixel at `(x, y)`.
    pub fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let at = ((y * self.width + x) * 4) as usize;
        [0, 1, 2, 3].map(|k| self.rgba[at + k])
    }

    /// Writes it as a PNG.
    ///
    /// # Errors
    ///
    /// If the file cannot be written.
    pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
        crate::screenshot::write_png(path, self.width, self.height, &self.rgba)
    }
}

const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Draws scenes of geometry `G` into an image of a fixed size: see the module.
pub struct Offscreen<G: GpuGeometry> {
    gpu: Gpu,
    size: (u32, u32),
    frame: FrameBuffers,
    shadow_map: ShadowMap,
    raster: RasterPipeline<G>,
    march: RayMarchPipeline<G>,
    color: wgpu::Texture,
    depth: wgpu::Texture,
}

impl<G: GpuGeometry> Offscreen<G> {
    /// Ready to draw images `width` × `height` on `gpu`.
    pub fn new(gpu: &Gpu, width: u32, height: u32) -> Self {
        let device = &gpu.device;
        let frame_layout = frame_layout(device);
        let frame = FrameBuffers::new(device, &frame_layout);
        let shadow_map = ShadowMap::new(gpu);
        let elsewhere = ElsewhereShared::new(gpu);
        let setup = RendererSetup {
            gpu,
            frame_layout: &frame_layout,
            shadow_layout: &shadow_map.layout,
            color_format: COLOR_FORMAT,
            depth_format: DEPTH_FORMAT,
            shadow_format: SHADOW_FORMAT,
            sample_count: 1,
            elsewhere_layout: &elsewhere.layout,
        };
        let texture = |format, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("fk offscreen"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        Self {
            gpu: gpu.clone(),
            size: (width, height),
            raster: RasterPipeline::new(&setup),
            march: RayMarchPipeline::new(&setup),
            frame,
            shadow_map,
            color: texture(
                COLOR_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            ),
            depth: texture(DEPTH_FORMAT, wgpu::TextureUsages::RENDER_ATTACHMENT),
        }
    }

    /// The scene in `world` seen through `camera` from `eye` (a pose relative to the root), lit
    /// by `lighting`, drawn by `paths`. Every entity with a [`GlobalPose`] is given its
    /// [`ViewRelative`] from it first.
    ///
    /// # Panics
    ///
    /// If the image cannot be read back from the GPU.
    pub fn draw(
        &mut self,
        world: &mut World,
        eye: &G::Isometry,
        camera: &Camera<G>,
        lighting: &Lighting,
        paths: Paths,
    ) -> Image {
        let to_eye = eye.inverse();
        let placed: Vec<_> = world
            .query::<(Entity, &GlobalPose<G>)>()
            .iter(world)
            .map(|(entity, pose)| (entity, to_eye.compose(&pose.isometry)))
            .collect();
        for (entity, relative) in placed {
            world.entity_mut(entity).insert(ViewRelative::<G>(relative));
        }

        let (width, height) = self.size;
        let aspect = Real::from(width) / Real::from(height);
        let active = ActiveView {
            entity: Entity::PLACEHOLDER,
            camera: camera.clone(),
            eye: GlobalPose::new(*eye),
        };
        let view = FrameView {
            view: &active,
            aspect,
            size: [width, height],
            frustum: Frustum::new(camera.fov_y, aspect, camera.far),
            shadow: None,
            elsewhere: None,
        };
        let size = WindowSize {
            width,
            height,
            scale_factor: 1.0,
        };
        let blocks = frame_blocks(world, Some(&active), lighting, None, 1, size);
        self.frame.write(&self.gpu.queue, &blocks);
        let gpu = &self.gpu;
        let mut renderers: Vec<&mut dyn GeometryRenderer<G>> = Vec::new();
        if paths != Paths::RayMarch {
            renderers.push(&mut self.raster);
        }
        if paths != Paths::Raster {
            renderers.push(&mut self.march);
        }
        for renderer in &mut renderers {
            renderer.prepare(gpu, world, &view);
        }
        for renderer in &mut renderers {
            renderer.queue(gpu, &view);
        }

        let bindings = FrameBindings {
            frame: &self.frame.bind_group,
            shadows: &self.shadow_map.bind_group,
            elsewhere: None,
        };
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        for renderer in &renderers {
            renderer.encode(&mut encoder, &bindings);
        }
        {
            let color = self.color.create_view(&Default::default());
            let depth = self.depth.create_view(&Default::default());
            let clear = lighting.sky.horizon.to_array().map(f64::from);
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("fk offscreen"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: clear[0],
                            g: clear[1],
                            b: clear[2],
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            for renderer in &renderers {
                renderer.render(&mut pass, &bindings);
            }
        }
        gpu.queue.submit([encoder.finish()]);
        let rgba = crate::screenshot::read_back(gpu, &self.color, 4, "offscreen image")
            .expect("the offscreen image reads back");
        Image {
            width,
            height,
            rgba,
        }
    }
}

#[cfg(test)]
mod tests {
    use fk_geometry::Geometry;
    use fk_geometry_euclidean::E3;
    use fk_math::nalgebra::Vector3;

    use super::*;
    use crate::{Color, MeshInstances, Meshes, shapes};

    #[test]
    fn a_box_ahead_is_drawn_and_the_rest_is_the_sky() {
        let Some(gpu) = Gpu::headless() else {
            eprintln!("no adapter, skipped");
            return;
        };
        let mut world = World::new();
        let mut meshes = Meshes::default();
        let cube = meshes.add(shapes::cuboid(Vector3::new(1.0, 1.0, 1.0)).build::<E3>());
        world.insert_resource(meshes);
        let ahead =
            <E3 as Geometry>::Isometry::exp(&E3::transvection(&Vector3::new(0.0, 0.0, -5.0)));
        world.spawn((
            GlobalPose::<E3>::new(ahead),
            MeshInstances::<E3>::single(cube, Color::hex(0xff0000)),
        ));
        let mut offscreen = Offscreen::<E3>::new(&gpu, 80, 60);
        let lighting = Lighting::default();
        let image = offscreen.draw(
            &mut world,
            &<E3 as Geometry>::Isometry::identity(),
            &Camera::new(1.0, 0.1, 100.0),
            &lighting,
            Paths::Both,
        );
        let [r, g, b, _] = image.pixel(40, 30);
        assert!(
            r > 100 && g < 60 && b < 60,
            "red in the middle: {r} {g} {b}"
        );
        assert_ne!(image.pixel(0, 0), image.pixel(40, 30), "the sky round it");
    }

    /// A red box and a blue one in a curved space, drawn from its origin looking along −z,
    /// each placed by the tangent vector at the origin that reaches it (reference frame).
    fn boxes<G: fk_geometry::GpuGeometry>(
        gpu: &Gpu,
        far: Real,
        red: Option<Vector3<Real>>,
        blue: Option<Vector3<Real>>,
    ) -> Image {
        let mut world = World::new();
        let mut meshes = Meshes::default();
        let cube = meshes.add(shapes::cuboid(Vector3::new(0.3, 0.3, 0.3)).build::<G>());
        world.insert_resource(meshes);
        for (at, color) in [(red, 0xff0000), (blue, 0x0000ff)] {
            let Some(at) = at else { continue };
            let pose = G::Isometry::exp(&G::transvection(&crate::tangent::<G>(&at)));
            world.spawn((
                GlobalPose::<G>::new(pose),
                MeshInstances::<G>::single(cube, Color::hex(color)),
            ));
        }
        let lighting = Lighting {
            fog_density: 0.0,
            ..Lighting::default()
        };
        Offscreen::<G>::new(gpu, 80, 60).draw(
            &mut world,
            &G::Isometry::identity(),
            &Camera::new(1.0, 0.05, far),
            &lighting,
            Paths::Raster,
        )
    }

    fn reddish([r, g, b, _]: [u8; 4]) -> bool {
        r > 80 && g < 60 && b < 60
    }

    fn bluish([r, g, b, _]: [u8; 4]) -> bool {
        b > 80 && r < 60 && g < 60
    }

    #[test]
    fn in_hyperbolic_space_a_box_ahead_is_drawn_and_one_past_the_fog_radius_is_not() {
        use fk_geometry_hyperbolic::H3;
        let Some(gpu) = Gpu::headless() else {
            eprintln!("no adapter, skipped");
            return;
        };
        let near = boxes::<H3>(&gpu, 6.0, Some(Vector3::new(0.0, 0.0, -2.0)), None);
        assert!(reddish(near.pixel(40, 30)), "{:?}", near.pixel(40, 30));
        let beyond = boxes::<H3>(&gpu, 6.0, Some(Vector3::new(0.0, 0.0, -7.0)), None);
        assert!(!reddish(beyond.pixel(40, 30)), "past the far radius");
    }

    #[test]
    fn on_the_sphere_what_is_behind_is_seen_ahead_the_long_way_round() {
        use fk_geometry_spherical::S3;
        let Some(gpu) = Gpu::headless() else {
            eprintln!("no adapter, skipped");
            return;
        };
        // A red box a metre behind the eye: seen ahead, 2π − 1 away the long way round.
        // The camera sees all the way round: its far distance up to 2π.
        let round = 2.0 * std::f64::consts::PI - 0.05;
        let behind = boxes::<S3>(&gpu, round, Some(Vector3::new(0.0, 0.0, 1.0)), None);
        assert!(reddish(behind.pixel(40, 30)), "{:?}", behind.pixel(40, 30));
        // A blue box 2.5 ahead: directly seen, so in front of the red one, although the red
        // one's antipodal image is only π − 1 ≈ 2.14 away.
        let both = boxes::<S3>(
            &gpu,
            round,
            Some(Vector3::new(0.0, 0.0, 1.0)),
            Some(Vector3::new(0.0, 0.0, -2.5)),
        );
        assert!(bluish(both.pixel(40, 30)), "{:?}", both.pixel(40, 30));
    }
}
