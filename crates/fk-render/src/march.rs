//! The ray-march pipeline: signed distance fields the caller supplies, marched per pixel and
//! composited with the raster path through the shared depth encoding.

use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Instant;

use bevy_ecs::component::Component;
use bevy_ecs::query::{Has, QueryState, Without};
use bevy_ecs::resource::Resource;
use bevy_ecs::world::World;
use encase::ShaderType;
use fk_geometry::{GpuGeometry, GroupElement, SampledMetric};
use fk_math::Real;
use fk_scene::ViewRelative;
use fk_shaders::Shader;

use crate::fields::{
    Builder, COARSE, Field, FieldKey, HALF_COLOR, HALF_DISTANCE, Variant, build_later, build_now,
};
use crate::frame::uniform_bytes;
use crate::gpu::{Vec4f, matrix_columns};
use crate::quotient::{MAX_FOLD_FACES, QuotientView, Unrepeated};
use crate::renderer::{FrameBindings, FrameView, GeometryRenderer, Hidden, OnLayer, RendererSetup};
use crate::{Gpu, RenderStats};

/// How many `vec4`s of its own a [`RayMarched`] scene passes to its distance field.
pub const SDF_PARAMS: usize = 192;

/// Pixels across the tiles of a scene marched one tile in each square of them
/// ([`RayMarched::interleave`]): `INTERLEAVED` in `ray_march.wgsl`.
const INTERLEAVED: u32 = 8;
/// Pixels to a group of the compute pass that lists the pixels traced at full resolution.
const CLASSIFY_GROUP: u32 = 8;

/// A signed distance field: a WGSL module that replaces the engine's `fk::sdf` (a ball of
/// radius 1) for the scenes marked with its [`RayMarched`], plus the modules it imports.
///
/// The module defines, under `#define_import_path fk::sdf`,
/// `fn sdf_distance(q: vec3<f32>) -> f32` (a lower bound of the distance from `q` to the
/// surface, negative inside) and `fn sdf_color(q: vec3<f32>) -> vec3<f32>` (the surface's
/// linear colour), in the scene's coordinates and units. It reads its values from
/// `march.params` (`#import fk::march::march`), the [`RayMarched::params`] of the scene drawn,
/// and may read the world block (`fk::view`), which is how a field follows a game value.
///
/// It may also define `fn sdf_alpha(q: vec3<f32>) -> f32`: the image's alpha on the surface
/// there, from 1 (as without it) to 4, a mark the caller's [`ImageEffect`](crate::ImageEffect)s
/// and post pass read (a surface kept out of an effect, one carrying it).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SdfShader {
    /// A name for messages, and the file path the module's errors point into.
    pub name: String,
    /// The `fk::sdf` module.
    pub source: String,
    /// Modules it imports, as `(path, source)`, in an order where each may import the ones
    /// before it.
    pub libraries: Vec<(String, String)>,
}

impl SdfShader {
    /// Composes the ray-march pipeline's shader and the probe with this field for `G`, both
    /// ray paths, and validates them, as the renderer will; for the game's tests.
    ///
    /// # Errors
    ///
    /// The composition or validation error, pointing into the WGSL.
    pub fn validate<G: GpuGeometry>(&self) -> Result<(), String> {
        use fk_shaders::naga::valid::{Capabilities, ValidationFlags, Validator};
        let flat = RayMarching::default().metric;
        let sampled = Some(metric_module(
            "fn geo_geodesic_accel(x: vec3<f32>, v: vec3<f32>) -> vec3<f32> {\n    \
                 return vec3<f32>(0.0);\n}\n",
        ));
        for (shader, metric) in [
            (Shader::RayMarch, &flat),
            (Shader::RayMarch, &sampled),
            (Shader::SdfProbe, &flat),
        ] {
            let module = self.compose(shader, G::SHADER, metric.as_deref())?;
            Validator::new(ValidationFlags::all(), Capabilities::empty())
                .validate(&module)
                .map_err(|error| format!("{}: {error:?}", self.name))?;
        }
        Ok(())
    }

    /// Whether the module marks the image's alpha on its surface: it defines `sdf_alpha`.
    fn marks(&self) -> bool {
        self.source.contains("fn sdf_alpha")
    }

    pub(crate) fn compose(
        &self,
        shader: Shader,
        geometry: &str,
        metric: Option<&str>,
    ) -> Result<fk_shaders::naga::Module, String> {
        let libraries: Vec<_> = self
            .libraries
            .iter()
            .map(|(path, source)| fk_shaders::Module { path, source })
            .collect();
        let hooks = fk_shaders::Hooks {
            sdf: Some(fk_shaders::Module {
                path: &self.name,
                source: &self.source,
            }),
            metric: metric.map(|source| fk_shaders::Module {
                path: "fk/sampled_metric.wgsl",
                source,
            }),
            libraries: &libraries,
            ..Default::default()
        };
        let mut defs = Vec::new();
        if shader == Shader::RayMarch {
            if metric.is_some() {
                defs.push("SAMPLED_METRIC");
            }
            if self.marks() {
                defs.push("SDF_ALPHA");
            }
        }
        fk_shaders::compose_hooked(shader, geometry, &defs, &hooks)
            .map_err(|error| error.to_string())
    }

    /// Evaluates the field on the GPU at `points` (scene coordinates) with `params` as its
    /// values, for tests that hold it against a CPU twin. The world block reads as zeros.
    ///
    /// # Errors
    ///
    /// If the field does not compose for `G`, or the device fails.
    pub fn probe<G: GpuGeometry>(
        &self,
        gpu: &Gpu,
        params: &[[f32; 4]; SDF_PARAMS],
        points: &[[f32; 3]],
    ) -> Result<Vec<f32>, String> {
        use wgpu::util::DeviceExt;
        if points.is_empty() {
            return Ok(Vec::new());
        }
        let device = &gpu.device;
        let module = self.compose(Shader::SdfProbe, G::SHADER, None)?;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&self.name),
            source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(module)),
        });
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty,
            count: None,
        };
        let uniform = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let storage = |read_only| wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let layout = |entries: &[wgpu::BindGroupLayoutEntry]| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("fk sdf probe"),
                entries,
            })
        };
        let frame_layout = layout(&[entry(0, uniform), entry(1, uniform), entry(2, uniform)]);
        let march_layout = layout(&[entry(0, uniform)]);
        let io_layout = layout(&[entry(0, storage(true)), entry(1, storage(false))]);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fk sdf probe"),
            bind_group_layouts: &[Some(&frame_layout), Some(&march_layout), Some(&io_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fk sdf probe"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("probe"),
            compilation_options: Default::default(),
            cache: None,
        });

        let buffer = |contents: &[u8], usage| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fk sdf probe"),
                contents,
                usage,
            })
        };
        let zeros = |size: u64| buffer(&vec![0; size as usize], wgpu::BufferUsages::UNIFORM);
        let frame_buffers = [
            zeros(crate::frame::ViewBlock::min_size().get()),
            zeros(crate::frame::WorldBlock::min_size().get()),
            zeros(crate::frame::LightingBlock::min_size().get()),
        ];
        let march = buffer(
            &uniform_bytes(&MarchBlock::probe(params)),
            wgpu::BufferUsages::UNIFORM,
        );
        let padded: Vec<[f32; 4]> = points.iter().map(|p| [p[0], p[1], p[2], 0.0]).collect();
        let input = buffer(bytemuck::cast_slice(&padded), wgpu::BufferUsages::STORAGE);
        let size = (points.len() * size_of::<f32>()) as u64;
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fk sdf probe"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fk sdf probe"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group = |layout, buffers: &[&wgpu::Buffer]| {
            let entries: Vec<_> = buffers
                .iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fk sdf probe"),
                layout,
                entries: &entries,
            })
        };
        let groups = [
            group(&frame_layout, &frame_buffers.each_ref()),
            group(&march_layout, &[&march]),
            group(&io_layout, &[&input, &output]),
        ];

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            for (index, group) in groups.iter().enumerate() {
                pass.set_bind_group(index as u32, group, &[]);
            }
            pass.dispatch_workgroups(points.len().div_ceil(64) as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, size);
        gpu.queue.submit([encoder.finish()]);
        let bytes = read_buffer(gpu, &readback)?;
        Ok(bytemuck::cast_slice(&bytes).to_vec())
    }
}

/// Maps `buffer` and copies it out.
pub(crate) fn read_buffer(gpu: &Gpu, buffer: &wgpu::Buffer) -> Result<Vec<u8>, String> {
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer.map_async(wgpu::MapMode::Read, .., move |result| {
        let _ = sender.send(result);
    });
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|error| error.to_string())?;
    receiver
        .recv()
        .map_err(|error| error.to_string())?
        .map_err(|error| error.to_string())?;
    let bytes = buffer
        .get_mapped_range(..)
        .map_err(|error| error.to_string())?
        .to_vec();
    buffer.unmap();
    Ok(bytes)
}

/// A geometry's geodesic equation as the `fk::metric` module of the sampled ray path.
fn metric_module(wgsl: &str) -> String {
    format!("#define_import_path fk::metric\n\n{wgsl}")
}

/// Which distance field, by its index in [`Sdfs`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SdfHandle(pub(crate) u32);

/// Every distance field the ray-march pipeline can draw, as a resource. A field is compiled the
/// first frame after it is added; fields cannot be changed or removed afterwards.
#[derive(Resource, Debug, Default)]
pub struct Sdfs(Vec<SdfShader>);

impl Sdfs {
    /// Adds a field.
    pub fn add(&mut self, shader: SdfShader) -> SdfHandle {
        let index = u32::try_from(self.0.len()).expect("too many distance fields");
        self.0.push(shader);
        SdfHandle(index)
    }

    /// Every field, in the order they were added.
    pub fn all(&self) -> &[SdfShader] {
        &self.0
    }
}

/// A scene drawn by ray marching a distance field, in the frame of the entity's
/// [`Pose`](fk_scene::Pose).
///
/// The field is evaluated in the normal coordinates at the entity's origin, multiplied by
/// [`scale`](Self::scale): the scene's units per unit of the world the eye is in. The scale is
/// a zoom kept apart from the pose, which has none; at scale `s` a point `x` of the entity's
/// frame is the field's point `s · x`, and its distances are divided by `s`. Exact in flat
/// space, a first-order approximation in curved space.
///
/// Scenes draw over every one before them where they are nearer; with none on screen the
/// pipeline costs nothing.
#[derive(Component, Clone, Debug, PartialEq)]
pub struct RayMarched {
    /// Which field.
    pub sdf: SdfHandle,
    /// Scene units per unit of the eye's world.
    pub scale: Real,
    /// How much more of the surface fades into the sky, on top of the fog: 0 none, 1 all of it.
    pub haze: Real,
    /// How much of the painted light the surface takes, and of the darkening by the steps its
    /// rays took: 1 all of it, like every surface, 0 none (it is drawn in its own colour).
    pub light: Real,
    /// How much of the fog the surface takes: 1 all of it, like every surface, 0 none.
    pub fog: Real,
    /// The field's own values, `march.params` in WGSL.
    pub params: [[f32; 4]; SDF_PARAMS],
    /// Most steps along a ray of this scene, `None` for [`RayMarching::steps`].
    pub steps: Option<u32>,
    /// Whether it is marched at half resolution, `None` for
    /// [`RayMarching::half_resolution`].
    pub half_resolution: Option<bool>,
    /// Whether it is drawn over everything drawn before it, whatever the depth says (its own
    /// depth is still written): for a thing that must fill the view, nothing left in front
    /// of it. Off, it is depth-tested like every surface.
    pub over: bool,
    /// Pixels across each block whose rays are started together, 0 for none. Before the
    /// scene is drawn, one cone per block of `coarse` × `coarse` pixels is marched, as wide as
    /// the block at every distance, stepping only as far as keeps the whole cone clear of the
    /// field; every ray of the block then starts where its cone stopped, and only marches the
    /// last stretch. The image is the same; most steps across empty space are taken once per
    /// block; only on an edge, where a ray passes within a pixel of a surface, may a pixel
    /// differ, as the march's samples fall elsewhere. Only for a field whose distance is a
    /// lower bound everywhere: one that leaves out what a ray does not come near must leave
    /// nothing out while `march_cone` (of `fk::march`) says the ray is a cone. Not in a
    /// quotient, nor along a sampled metric (it is then off).
    pub coarse: u32,
    /// Whether it is marched before the other scenes, into a target of its own, and their
    /// rays stop where it is: what lies behind it is never marched. For a scene that can fill
    /// much of the view in front of another one (the first such scene; any other is drawn as
    /// usual). The image is the same.
    pub ahead: bool,
    /// Marched [`ahead`](Self::ahead) at full resolution, only one tile (8 pixels across) of
    /// every square of `interleave` × `interleave` is marched each frame, in turn, the others
    /// carried from its last frame's image where the eye's motion took them (by the distance it
    /// had there: exact while the eye only turns): about `interleave`² times cheaper, what moves
    /// in the field as many frames late, and a pixel coming into view from nowhere is marched.
    /// 0 or 1, every pixel is marched every frame. With [`coarse`](Self::coarse) 8, the cones
    /// of the tiles carried are not marched either. With the image settling over frames
    /// ([`Settle`](crate::Settle)) each pixel comes out as the average over it all the same.
    /// Ignored for any other scene.
    pub interleave: u32,
    /// While the eye moves more than about a pixel a frame, one tile in a square of this many
    /// across instead of [`interleave`](Self::interleave): fewer carried, so sharper as it
    /// turns, and dearer. 0, the same as `interleave`.
    pub interleave_moving: u32,
}

impl RayMarched {
    /// The field `sdf` at scale 1, no haze, lit and fogged, its values zero, marched as
    /// [`RayMarching`] says.
    pub fn new(sdf: SdfHandle) -> Self {
        Self {
            sdf,
            scale: 1.0,
            haze: 0.0,
            light: 1.0,
            fog: 1.0,
            params: [[0.0; 4]; SDF_PARAMS],
            steps: None,
            half_resolution: None,
            over: false,
            coarse: 0,
            ahead: false,
            interleave: 1,
            interleave_moving: 0,
        }
    }
}

/// How the ray-march pipeline marches, as a resource.
///
/// A ray steps by the field's distance, over-relaxed, and hits where the field comes within
/// [`hit_pixels`](Self::hit_pixels) pixels' footprint of it, or where it runs out of
/// [`steps`](Self::steps) short of the camera's far surface (a crease it could not resolve).
/// The surface takes the painted light of [`Lighting`](crate::Lighting), darkened by the share
/// of the steps the ray took, and fades into the sky with the fog.
#[derive(Resource, Clone, Debug, PartialEq)]
pub struct RayMarching {
    /// Most steps along a ray.
    pub steps: u32,
    /// Over-relaxation of the steps, 1 for plain sphere tracing; a step that overshoots is taken
    /// back and the ray goes on unrelaxed.
    pub relaxation: Real,
    /// A ray hits when the field is closer than this many pixels' footprint.
    pub hit_pixels: Real,
    /// How much the steps a ray took darken its surface, 0 for not at all.
    pub occlusion: Real,
    /// The longest step, in units of the eye's world, 0 for no limit. Fields that are not true
    /// lower bounds need one.
    pub longest_step: Real,
    /// Marches at half the resolution in each direction and brings the image up to the
    /// window's, depth included.
    pub half_resolution: bool,
    /// The `fk::metric` module rays are integrated along, with RK4 and no relaxation, instead
    /// of the geometry's closed-form geodesics; see [`sampled`](Self::sampled). `None` for the
    /// closed form.
    pub metric: Option<String>,
}

impl Default for RayMarching {
    fn default() -> Self {
        Self {
            steps: 160,
            relaxation: 1.4,
            hit_pixels: 1.0,
            occlusion: 0.6,
            longest_step: 0.0,
            half_resolution: false,
            metric: None,
        }
    }
}

impl RayMarching {
    /// Rays integrated along `metric`'s geodesic equation
    /// ([`SampledMetric::wgsl_module`]), in the eye's normal coordinates. Closed-form geometries
    /// implement it too, so both paths can be compared.
    pub fn sampled<M: SampledMetric>(metric: &M) -> Option<String> {
        Some(metric_module(metric.wgsl_module()))
    }
}

#[derive(ShaderType)]
struct MarchBlock {
    to_scene: [Vec4f; 4],
    settings: Vec4f,
    shading: Vec4f,
    paint: Vec4f,
    interleave: Vec4f,
    carry: [Vec4f; 4],
    fold: Fold,
    params: [Vec4f; SDF_PARAMS],
}

/// The quotient a scene is folded into, in the eye's frame: `count.x` faces (0 for none), the
/// domain's centre, each face's translate of it, and the deck element carrying a point back
/// across each face.
#[derive(ShaderType, Clone, Copy, Default)]
struct Fold {
    count: Vec4f,
    centre: Vec4f,
    faces: [Vec4f; MAX_FOLD_FACES],
    back: [[Vec4f; 4]; MAX_FOLD_FACES],
}

impl Fold {
    /// `space` seen from `eye`.
    fn new<G: GpuGeometry>(space: &QuotientView<G>, eye: &G::Isometry) -> Self {
        let quotient = space.quotient();
        let to_eye = eye.inverse();
        let centre = quotient.centre();
        let mut fold = Self {
            centre: Vec4f::from_real(&G::embed_point(&G::apply(&to_eye, centre))),
            ..Self::default()
        };
        let mut count = 0;
        for (i, face) in quotient.faces().take(MAX_FOLD_FACES).enumerate() {
            let across = to_eye.compose(&quotient.element(face));
            fold.faces[i] = Vec4f::from_real(&G::embed_point(&G::apply(&across, centre)));
            let back = to_eye
                .compose(&quotient.element(face.inverse()))
                .compose(eye);
            fold.back[i] = matrix_columns(&G::isometry_matrix(&back)).map(Vec4f);
            count += 1;
        }
        fold.count = Vec4f([count as f32, 0.0, 0.0, 0.0]);
        fold
    }
}

impl MarchBlock {
    /// The scene at the eye, scale 1, with `params`: what the probe evaluates the field in.
    fn probe(params: &[[f32; 4]; SDF_PARAMS]) -> Self {
        Self {
            to_scene: matrix_columns(&fk_math::nalgebra::Matrix4::identity()).map(Vec4f),
            settings: Vec4f([1.0; 4]),
            shading: Vec4f([0.0, 0.0, 1.0, 0.0]),
            paint: Vec4f([1.0, 1.0, 0.0, 0.0]),
            interleave: Vec4f([0.0; 4]),
            carry: matrix_columns(&fk_math::nalgebra::Matrix4::identity()).map(Vec4f),
            fold: Fold::default(),
            params: params.map(Vec4f),
        }
    }
}

/// What lists and traces at full resolution the pixels a half-resolution scene's composite
/// needs the rays of: how many, the dispatch over them, the list, and the groups binding them.
struct Edges {
    count: wgpu::Buffer,
    args: wgpu::Buffer,
    /// For `classify` and `finalize`.
    listing: wgpu::BindGroup,
    /// For `refine`.
    tracing: wgpu::BindGroup,
}

/// Where the rays of each block of a scene start ([`RayMarched::coarse`]).
struct CoarseTargets {
    size: [u32; 2],
    view: wgpu::TextureView,
}

/// The offscreen targets of one scene at half resolution (or at full resolution, ahead).
struct HalfTargets {
    size: [u32; 2],
    color: wgpu::TextureView,
    distance: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
    /// At half resolution, the pixels traced at full resolution, `full` across.
    edges: Option<(Edges, [u32; 2])>,
}

/// One scene drawn this frame.
struct Scene {
    uniform: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    sdf: u32,
    half: Option<HalfTargets>,
    /// Marched at half resolution this frame, into `half`.
    halved: bool,
    coarse: Option<CoarseTargets>,
    /// Its rays started from its cones this frame, out of `coarse`.
    coarsened: bool,
    /// Marched before the others this frame, into `half` at its resolution
    /// ([`RayMarched::ahead`]).
    ahead: bool,
    /// Marched one tile in each square of them ([`RayMarched::interleave`]): its last frame's targets, the
    /// others carried from them. `half` and these change places each frame.
    last: Option<HalfTargets>,
    /// Carried from `last` this frame, and how many tiles are marched over it.
    carried: Option<u32>,
    /// Where its rays start and where they stop this frame: its cones and the scene ahead.
    starts: Option<wgpu::BindGroup>,
    /// Drawn into the layer render as well.
    layer: bool,
    /// Drawn over everything, the depth test passed whatever is there.
    over: bool,
}

type Marched<G> = (
    &'static ViewRelative<G>,
    &'static RayMarched,
    Has<OnLayer>,
    Has<Unrepeated>,
);

/// Draws every [`RayMarched`] scene: a fullscreen pass per scene, after the raster pipeline,
/// marching each pixel's ray through the scene's distance field ([`SdfShader`]) as set by
/// [`RayMarching`].
///
/// The depth written is the encoding every render path shares (`geo_distance_to_depth`), so a
/// raster mesh in front of the surface hides it and one behind it is hidden.
pub struct RayMarchPipeline<G: GpuGeometry> {
    device: wgpu::Device,
    march_layout: wgpu::BindGroupLayout,
    half_layout: wgpu::BindGroupLayout,
    coarse_layout: wgpu::BindGroupLayout,
    listing_layout: wgpu::BindGroupLayout,
    tracing_layout: wgpu::BindGroupLayout,
    /// What builds the fields' pipelines.
    builder: Arc<Builder>,
    /// What a scene marched at full resolution binds where the pixels traced at full
    /// resolution would be.
    no_refined: wgpu::TextureView,
    /// What a scene without cones binds where its coarse target would be, and one without a
    /// scene ahead where that scene's distances would be.
    no_coarse: wgpu::TextureView,
    no_distance: wgpu::TextureView,
    /// By field; `None` for one that did not compile, which is not drawn. Each pipeline is
    /// built the first time a scene needs it, unless it was built ahead.
    pipelines: Vec<Option<Arc<Field>>>,
    /// The metric the pipelines were built with.
    metric: Option<String>,
    query: Option<QueryState<Marched<G>, Without<Hidden>>>,
    scenes: Vec<Scene>,
    /// How many of `scenes` are drawn this frame.
    drawn: usize,
    /// Frames prepared, for which tile of four an interleaved scene marches.
    frame: u32,
    /// The eye of the last frame, for what an interleaved scene carries.
    last_eye: Option<G::Isometry>,
    _geometry: PhantomData<fn() -> G>,
}

impl<G: GpuGeometry> RayMarchPipeline<G> {
    /// Builds the pipeline. Fields are compiled when they first appear in [`Sdfs`].
    pub fn new(setup: &RendererSetup<'_>) -> Self {
        let device = &setup.gpu.device;
        let march_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fk march"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT | wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let texture = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let half_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fk march half"),
            entries: &[texture(0), texture(1), texture(8), texture(9)],
        });
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let written = |binding, format| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        };
        let listing_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fk march edges"),
            entries: &[
                texture(0),
                texture(1),
                storage(4, false),
                storage(5, false),
                storage(10, false),
            ],
        });
        let tracing_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fk march edges"),
            entries: &[
                storage(4, false),
                storage(11, true),
                written(6, HALF_COLOR),
                written(7, HALF_DISTANCE),
            ],
        });
        let coarse_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fk march coarse"),
            entries: &[texture(2), texture(3), texture(12), texture(13)],
        });
        let no_coarse = coarse_texture(device, [1, 1]);
        let no_distance = offscreen_texture(device, [1, 1], HALF_DISTANCE);
        let no_refined = offscreen_texture(device, [1, 1], HALF_COLOR);
        let pipeline_layout = |groups: &[Option<&wgpu::BindGroupLayout>]| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("fk march"),
                bind_group_layouts: groups,
                immediate_size: 0,
            })
        };
        let builder = Arc::new(Builder {
            device: device.clone(),
            full_layout: pipeline_layout(&[
                Some(setup.frame_layout),
                Some(&march_layout),
                Some(&coarse_layout),
            ]),
            composite_layout: pipeline_layout(&[
                Some(setup.frame_layout),
                Some(&march_layout),
                Some(&coarse_layout),
                Some(&half_layout),
            ]),
            cone_layout: pipeline_layout(&[Some(setup.frame_layout), Some(&march_layout)]),
            listing_layout: pipeline_layout(&[
                Some(setup.frame_layout),
                Some(&march_layout),
                Some(&coarse_layout),
                Some(&listing_layout),
            ]),
            tracing_layout: pipeline_layout(&[
                Some(setup.frame_layout),
                Some(&march_layout),
                Some(&coarse_layout),
                Some(&tracing_layout),
            ]),
            color_format: setup.color_format,
            depth_format: setup.depth_format,
            sample_count: setup.sample_count,
        });
        Self {
            device: device.clone(),
            builder,
            march_layout,
            half_layout,
            coarse_layout,
            listing_layout,
            tracing_layout,
            no_refined,
            no_coarse,
            no_distance,
            pipelines: Vec::new(),
            metric: None,
            query: None,
            scenes: Vec::new(),
            drawn: 0,
            frame: 0,
            last_eye: None,
            _geometry: PhantomData,
        }
    }

    /// Compiles the fields added to `sdfs` since: their shaders now (or as kept on the
    /// device), the pipelines `world`'s scenes draw them with now too, every other one of
    /// them ahead of need.
    fn compile_new_sdfs(&mut self, gpu: &Gpu, sdfs: &Sdfs, world: &mut World) {
        let first = self.pipelines.len();
        if sdfs.all().len() <= first {
            return;
        }
        let started = Instant::now();
        let keys = sdfs.all()[first..]
            .iter()
            .map(|shader| FieldKey {
                geometry: G::SHADER,
                shader: shader.clone(),
                metric: self.metric.clone(),
                color_format: self.builder.color_format,
                depth_format: self.builder.depth_format,
                sample_count: self.builder.sample_count,
            })
            .collect();
        let fields = gpu.kept.fields(&self.builder, keys, |key| {
            key.shader
                .compose(Shader::RayMarch, key.geometry, key.metric.as_deref())
        });
        self.pipelines.extend(fields);
        let settings = world
            .get_resource::<RayMarching>()
            .cloned()
            .unwrap_or_default();
        let new = |index: usize| {
            self.pipelines
                .get(index)
                .filter(|_| index >= first)
                .cloned()
                .flatten()
        };
        // What the scenes there are draw with: now, before the first frame needs it.
        let mut marched = world.query::<&RayMarched>();
        let now: Vec<_> = marched
            .iter(world)
            .filter_map(|marched| new(marched.sdf.0 as usize).map(|field| (field, marched)))
            .flat_map(|(field, marched)| {
                likely(marched, &settings)
                    .into_iter()
                    .map(move |variant| (field.clone(), variant))
            })
            .collect();
        build_now(&now);
        tracing::info!(
            "{} distance fields compiled in {:.0?}",
            self.pipelines.len() - first,
            started.elapsed()
        );
        // The rest ahead of need: first what a scene of each would draw with as marching goes
        // (a scene not drawn yet, its rays started from cones), then every other pipeline
        // (drawn over everything, halved).
        let fields: Vec<_> = (first..self.pipelines.len()).filter_map(new).collect();
        let usual = likely(&RayMarched::new(SdfHandle(0)), &settings);
        let later = usual
            .iter()
            .chain(&[Variant::Coarse])
            .chain(&Variant::ALL)
            .flat_map(|&variant| fields.iter().map(move |field| (field.clone(), variant)))
            .collect();
        build_later(later);
    }

    fn coarse_targets(&self, size: [u32; 2]) -> CoarseTargets {
        CoarseTargets {
            size,
            view: coarse_texture(&self.device, size),
        }
    }

    /// The targets of a scene marched offscreen at `size`; at half resolution (`full` the
    /// window's size), what lists and traces its edges at full resolution too.
    fn half_targets(&self, size: [u32; 2], full: Option<[u32; 2]>) -> HalfTargets {
        let target = |format| {
            self.device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("fk march half"),
                    size: wgpu::Extent3d {
                        width: size[0],
                        height: size[1],
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        let color = target(HALF_COLOR);
        let distance = target(HALF_DISTANCE);
        fn view(binding: u32, view: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry {
                binding,
                resource: wgpu::BindingResource::TextureView(view),
            }
        }
        fn buffer(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            }
        }
        let mut refined = (self.no_refined.clone(), self.no_distance.clone());
        let mut edges = None;
        if let Some(full) = full {
            let written = |format| {
                self.device
                    .create_texture(&wgpu::TextureDescriptor {
                        label: Some("fk march edges"),
                        size: wgpu::Extent3d {
                            width: full[0],
                            height: full[1],
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format,
                        usage: wgpu::TextureUsages::STORAGE_BINDING
                            | wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    })
                    .create_view(&Default::default())
            };
            refined = (written(HALF_COLOR), written(HALF_DISTANCE));
            let storage = |label, size: u64, usage| {
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage: wgpu::BufferUsages::STORAGE | usage,
                    mapped_at_creation: false,
                })
            };
            let pixels = u64::from(full[0]) * u64::from(full[1]);
            let list = storage(
                "fk march edges",
                4 * pixels.max(1),
                wgpu::BufferUsages::empty(),
            );
            let count = storage("fk march edges", 4, wgpu::BufferUsages::COPY_DST);
            let args = storage("fk march edges", 16, wgpu::BufferUsages::INDIRECT);
            let listing = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fk march edges"),
                layout: &self.listing_layout,
                entries: &[
                    view(0, &color),
                    view(1, &distance),
                    buffer(4, &list),
                    buffer(5, &count),
                    buffer(10, &args),
                ],
            });
            let tracing = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fk march edges"),
                layout: &self.tracing_layout,
                entries: &[
                    buffer(4, &list),
                    buffer(11, &args),
                    view(6, &refined.0),
                    view(7, &refined.1),
                ],
            });
            edges = Some((
                Edges {
                    count,
                    args,
                    listing,
                    tracing,
                },
                full,
            ));
        }
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fk march half"),
            layout: &self.half_layout,
            entries: &[
                view(0, &color),
                view(1, &distance),
                view(8, &refined.0),
                view(9, &refined.1),
            ],
        });
        HalfTargets {
            size,
            color,
            distance,
            bind_group,
            edges,
        }
    }

    fn scene(&mut self, index: usize) -> &mut Scene {
        while self.scenes.len() <= index {
            let uniform = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fk march"),
                size: MarchBlock::min_size().get(),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fk march"),
                layout: &self.march_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                }],
            });
            self.scenes.push(Scene {
                uniform,
                bind_group,
                sdf: 0,
                half: None,
                halved: false,
                coarse: None,
                coarsened: false,
                ahead: false,
                last: None,
                carried: None,
                starts: None,
                layer: false,
                over: false,
            });
        }
        &mut self.scenes[index]
    }
}

impl<G: GpuGeometry> GeometryRenderer<G> for RayMarchPipeline<G> {
    fn warm(&mut self, gpu: &Gpu, world: &mut World) {
        let metric = world
            .get_resource::<RayMarching>()
            .and_then(|settings| settings.metric.clone());
        if metric != self.metric {
            self.metric = metric;
            self.pipelines.clear();
        }
        if world.contains_resource::<Sdfs>() {
            world.resource_scope(|world, sdfs: bevy_ecs::world::Mut<'_, Sdfs>| {
                self.compile_new_sdfs(gpu, &sdfs, world);
            });
        }
    }

    fn prepare(&mut self, gpu: &Gpu, world: &mut World, view: &FrameView<'_, G>) {
        self.warm(gpu, world);
        let settings = world
            .get_resource::<RayMarching>()
            .cloned()
            .unwrap_or_default();
        // Whether the image settles over frames: what a scene marched one tile in a square of
        // them carries is then marked for it.
        let settles = world
            .get_resource::<crate::PostSettings>()
            .is_some_and(|post| post.settle.frames > 1.0);
        let half_size = view.size.map(|n| n.div_ceil(2).max(1));
        // How the eye moved since the last frame, and which tile of four is marched.
        let eye = view.view.eye.isometry;
        let carry = self
            .last_eye
            .map(|last| G::isometry_matrix(&last.inverse().compose(&eye)));
        self.last_eye = Some(eye);
        self.frame = self.frame.wrapping_add(1);
        // Whether the eye moved more than about a pixel since the last frame.
        let turned =
            carry.is_some_and(|m| (m - fk_math::nalgebra::Matrix4::identity()).norm() > 2e-3);

        let fold = world
            .get_resource::<QuotientView<G>>()
            .map(|space| Fold::new(space, &view.view.eye.isometry));
        let query = self.query.get_or_insert_with(|| world.query_filtered());
        let scenes: Vec<_> = query
            .iter(world)
            .filter(|(_, marched, _, _)| marched.scale > 0.0)
            .map(|(relative, marched, layer, unrepeated)| {
                let fold = fold.filter(|_| !unrepeated).unwrap_or_default();
                (view.relative(&relative.0), marched.clone(), layer, fold)
            })
            .collect();
        self.drawn = 0;
        // Only fields that compiled are drawn.
        let scenes: Vec<_> = scenes
            .into_iter()
            .filter(|(_, marched, _, _)| {
                matches!(self.pipelines.get(marched.sdf.0 as usize), Some(Some(_)))
            })
            .collect();
        // The scene ahead of the others, if any: the first drawn that asks.
        let ahead = scenes.iter().position(|(_, marched, _, _)| marched.ahead);
        // How the others find its distances: window pixels per pixel of its target, negative
        // when it is drawn over everything (nothing behind it shows at all), 0 without one.
        let ahead_pixels = ahead.map_or(0.0, |i| {
            let marched = &scenes[i].1;
            let pixels = if marched.half_resolution.unwrap_or(settings.half_resolution) {
                2.0
            } else {
                1.0
            };
            if marched.over { -pixels } else { pixels }
        });
        for (relative, marched, layer, fold) in scenes {
            let index = self.drawn;
            self.drawn += 1;
            let is_ahead = ahead == Some(index);
            let half = marched.half_resolution.unwrap_or(settings.half_resolution);
            let pixels: f32 = if half { 2.0 } else { 1.0 };
            let target_size = if half { half_size } else { view.size };
            let steps = marched.steps.unwrap_or(settings.steps);
            // Cones only along closed-form rays, and never carried across a quotient's faces.
            let cones = if settings.metric.is_none() && fold.count.0[0] < 0.5 {
                marched.coarse
            } else {
                0
            };
            let coarse_size = view.size.map(|n| n.div_ceil(cones.max(1)).max(1));
            let block = MarchBlock {
                to_scene: matrix_columns(&G::isometry_matrix(&relative.inverse())).map(Vec4f),
                settings: Vec4f([
                    marched.scale as f32,
                    steps.max(1) as f32,
                    settings.relaxation.max(1.0) as f32,
                    settings.hit_pixels.max(1e-3) as f32,
                ]),
                shading: Vec4f([
                    settings.occlusion.max(0.0) as f32,
                    marched.haze.clamp(0.0, 1.0) as f32,
                    pixels,
                    settings.longest_step.max(0.0) as f32,
                ]),
                paint: Vec4f([
                    marched.light.clamp(0.0, 1.0) as f32,
                    marched.fog.clamp(0.0, 1.0) as f32,
                    cones as f32,
                    if is_ahead { 0.0 } else { ahead_pixels },
                ]),
                interleave: Vec4f([0.0; 4]),
                carry: [Vec4f([0.0; 4]); 4],
                fold,
                params: marched.params.map(Vec4f),
            };
            // While the eye moves, more tiles are marched: fewer carried, sharper as it turns.
            let tiles_across = if turned && marched.interleave_moving > 0 {
                marched.interleave_moving
            } else {
                marched.interleave
            };
            let interleaved = is_ahead && !half && tiles_across > 1;
            // Which tile of the square is marched: in a turn that leaps across it.
            let tiles = tiles_across.max(1) * tiles_across.max(1);
            let leap = (1..tiles)
                .rev()
                .find(|k| gcd(*k, tiles) == 1 && *k * 2 < tiles + 2);
            let marched_tile = (self.frame.wrapping_mul(leap.unwrap_or(1))) % tiles;
            if interleaved {
                // Last frame's targets become what this frame carries from.
                let scene = self.scene(index);
                std::mem::swap(&mut scene.half, &mut scene.last);
            } else if let Some(scene) = self.scenes.get_mut(index) {
                scene.last = None;
            }
            let carried = interleaved
                && carry.is_some()
                && self.scenes[index]
                    .last
                    .as_ref()
                    .is_some_and(|last| last.size == target_size);
            let block = MarchBlock {
                interleave: Vec4f([
                    if interleaved {
                        tiles_across as f32
                    } else {
                        1.0
                    },
                    marched_tile as f32,
                    if carried { 1.0 } else { 0.0 },
                    if settles { 1.0 } else { 0.0 },
                ]),
                carry: matrix_columns(&carry.unwrap_or_else(fk_math::nalgebra::Matrix4::identity))
                    .map(Vec4f),
                ..block
            };
            let offscreen = half || is_ahead;
            // At half resolution its edges are traced at full resolution.
            let full = half.then_some(view.size);
            let needs_targets = offscreen
                && self.scenes.get(index).is_none_or(|scene| {
                    scene.half.as_ref().is_none_or(|targets| {
                        targets.size != target_size
                            || targets.edges.as_ref().map(|(_, size)| *size) != full
                    })
                });
            let targets = needs_targets.then(|| self.half_targets(target_size, full));
            let needs_coarse = cones > 0
                && self.scenes.get(index).is_none_or(|scene| {
                    scene
                        .coarse
                        .as_ref()
                        .is_none_or(|targets| targets.size != coarse_size)
                });
            let coarse = needs_coarse.then(|| self.coarse_targets(coarse_size));
            let tiles_marched = carried.then(|| {
                let n = tiles_across.max(1);
                let tiles = view.size.map(|s| s.div_ceil(INTERLEAVED));
                let first = [marched_tile % n, marched_tile / n];
                let count = |axis: usize| tiles[axis].saturating_sub(first[axis]).div_ceil(n);
                count(0) * count(1)
            });
            let scene = self.scene(index);
            scene.carried = tiles_marched;
            scene.sdf = marched.sdf.0;
            scene.layer = layer;
            scene.halved = offscreen;
            scene.ahead = is_ahead;
            scene.coarsened = cones > 0;
            scene.over = marched.over;
            if let Some(targets) = targets {
                scene.half = Some(targets);
            }
            if let Some(coarse) = coarse {
                scene.coarse = Some(coarse);
            }
            gpu.queue
                .write_buffer(&scene.uniform, 0, &uniform_bytes(&block));
            // The pipelines it draws with this frame, built now unless they were ahead.
            if let Some(Some(field)) = self.pipelines.get(marched.sdf.0 as usize) {
                let carried = tiles_marched.is_some();
                for variant in drawn_with(cones > 0, offscreen, carried, half, marched.over) {
                    field.get(variant);
                }
            }
        }
        // Where each scene's rays start and stop: its own cones, and the distances of the
        // scene ahead (not its own).
        let ahead_distance = ahead
            .and_then(|i| self.scenes[i].half.as_ref())
            .map(|targets| targets.distance.clone());
        for index in 0..self.drawn {
            let scene = &self.scenes[index];
            let coarse = match (&scene.coarse, scene.coarsened) {
                (Some(targets), true) => &targets.view,
                _ => &self.no_coarse,
            };
            let distance = match &ahead_distance {
                Some(view) if !scene.ahead => view,
                _ => &self.no_distance,
            };
            let last = match &scene.last {
                Some(last) => (&last.color, &last.distance),
                None => (&self.no_refined, &self.no_distance),
            };
            let starts =
                coarse_bind_group(&self.device, &self.coarse_layout, coarse, distance, last);
            self.scenes[index].starts = Some(starts);
        }
    }

    fn queue(&mut self, _: &Gpu, _: &FrameView<'_, G>) {}

    fn encode(&self, encoder: &mut wgpu::CommandEncoder, bindings: &FrameBindings<'_>) {
        // The cones first: the rays drawn below start where they stopped.
        for scene in self.scenes[..self.drawn]
            .iter()
            .filter(|scene| scene.coarsened)
        {
            let (Some(pipeline), Some(targets)) =
                (self.pipeline(scene, Variant::Coarse), &scene.coarse)
            else {
                continue;
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("fk march coarse"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &targets.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bindings.frame, &[]);
            pass.set_bind_group(1, &scene.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        // The scene ahead before the others: their rays stop where it is.
        let ahead = self.scenes[..self.drawn].iter().filter(|scene| scene.ahead);
        let others = self.scenes[..self.drawn]
            .iter()
            .filter(|scene| scene.halved && !scene.ahead);
        for scene in ahead.chain(others) {
            let (Some(Some(field)), Some(targets)) =
                (self.pipelines.get(scene.sdf as usize), &scene.half)
            else {
                continue;
            };
            let attachment = |view| {
                Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("fk march half"),
                color_attachments: &[attachment(&targets.color), attachment(&targets.distance)],
                ..Default::default()
            });
            pass.set_bind_group(0, bindings.frame, &[]);
            pass.set_bind_group(1, &scene.bind_group, &[]);
            pass.set_bind_group(2, self.starts(scene), &[]);
            if let (Some(tiles), Some(carry), Some(marched)) = (
                scene.carried,
                field.render(Variant::Carry),
                field.render(Variant::Tiles),
            ) {
                // Every pixel carried, the tiles of this frame marched over them.
                pass.set_pipeline(carry);
                pass.draw(0..3, 0..1);
                pass.set_pipeline(marched);
                pass.draw(0..6, 0..tiles);
            } else if let Some(half) = field.render(Variant::Half) {
                pass.set_pipeline(half);
                pass.draw(0..3, 0..1);
            }
            drop(pass);
            // The pixels between which the half resolution cannot tell: listed, and their own
            // rays traced together at full resolution.
            let (Some((edges, full)), Some(classify), Some(finalize), Some(refine)) = (
                &targets.edges,
                field.compute(Variant::Classify),
                field.compute(Variant::Finalize),
                field.compute(Variant::Refine),
            ) else {
                continue;
            };
            encoder.clear_buffer(&edges.count, 0, None);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fk march edges"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, bindings.frame, &[]);
            pass.set_bind_group(1, &scene.bind_group, &[]);
            pass.set_bind_group(2, self.starts(scene), &[]);
            pass.set_bind_group(3, &edges.listing, &[]);
            pass.set_pipeline(classify);
            pass.dispatch_workgroups(
                full[0].div_ceil(CLASSIFY_GROUP),
                full[1].div_ceil(CLASSIFY_GROUP),
                1,
            );
            pass.set_pipeline(finalize);
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_bind_group(3, &edges.tracing, &[]);
            pass.set_pipeline(refine);
            pass.dispatch_workgroups_indirect(&edges.args, 0);
        }
    }

    fn render(&self, pass: &mut wgpu::RenderPass<'_>, bindings: &FrameBindings<'_>) {
        self.draw(pass, bindings, false);
    }

    fn render_layer(&self, pass: &mut wgpu::RenderPass<'_>, bindings: &FrameBindings<'_>) {
        self.draw(pass, bindings, true);
    }

    fn stats(&self) -> RenderStats {
        RenderStats {
            drawn: self.drawn,
            culled: 0,
            draws: self.drawn,
        }
    }
}

impl<G: GpuGeometry> RayMarchPipeline<G> {
    /// `scene`'s field's `variant`, if it is built (as [`prepare`](GeometryRenderer::prepare)
    /// builds every one the frame draws with).
    fn pipeline(&self, scene: &Scene, variant: Variant) -> Option<&wgpu::RenderPipeline> {
        self.pipelines
            .get(scene.sdf as usize)?
            .as_ref()?
            .render(variant)
    }

    /// Where `scene`'s rays start and stop, as [`prepare`](GeometryRenderer::prepare) bound
    /// them this frame.
    fn starts<'a>(&'a self, scene: &'a Scene) -> &'a wgpu::BindGroup {
        scene
            .starts
            .as_ref()
            .expect("every scene drawn has where its rays start")
    }

    /// Draws every scene of this frame, or only those on the layer.
    fn draw(&self, pass: &mut wgpu::RenderPass<'_>, bindings: &FrameBindings<'_>, layer: bool) {
        for scene in &self.scenes[..self.drawn] {
            if layer && !scene.layer {
                continue;
            }
            let variant = match (scene.halved, scene.over) {
                (true, false) => Variant::Composite,
                (true, true) => Variant::CompositeOver,
                (false, false) => Variant::Full,
                (false, true) => Variant::FullOver,
            };
            let Some(pipeline) = self.pipeline(scene, variant) else {
                continue;
            };
            match (&scene.half, scene.halved) {
                (Some(targets), true) => {
                    pass.set_pipeline(pipeline);
                    // Edges are traced again at full resolution, from the cones.
                    pass.set_bind_group(2, self.starts(scene), &[]);
                    pass.set_bind_group(3, &targets.bind_group, &[]);
                }
                (None, true) => continue,
                _ => {
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(2, self.starts(scene), &[]);
                }
            }
            pass.set_bind_group(0, bindings.frame, &[]);
            pass.set_bind_group(1, &scene.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}

/// The pipelines a scene is drawn with in a frame: with its rays started from `cones`, marched
/// `offscreen` (at half resolution, `half`, or ahead), its pixels `carried` from the last frame,
/// drawn `over` everything.
fn drawn_with(cones: bool, offscreen: bool, carried: bool, half: bool, over: bool) -> Vec<Variant> {
    let mut variants = Vec::new();
    if cones {
        variants.push(Variant::Coarse);
    }
    if offscreen {
        if carried {
            variants.extend([Variant::Carry, Variant::Tiles]);
        } else {
            variants.push(Variant::Half);
        }
        if half {
            variants.extend([Variant::Classify, Variant::Finalize, Variant::Refine]);
        }
        variants.push(if over {
            Variant::CompositeOver
        } else {
            Variant::Composite
        });
    } else {
        variants.push(if over {
            Variant::FullOver
        } else {
            Variant::Full
        });
    }
    variants
}

/// The pipelines a scene `marched` as `settings` say is likely drawn with in its first frames:
/// all it asks for, cones, half resolution, carrying, as it is now.
fn likely(marched: &RayMarched, settings: &RayMarching) -> Vec<Variant> {
    let half = marched.half_resolution.unwrap_or(settings.half_resolution);
    let cones = marched.coarse > 0 && settings.metric.is_none();
    let offscreen = half || marched.ahead;
    let mut variants = drawn_with(cones, offscreen, false, half, marched.over);
    if marched.ahead && !half && marched.interleave.max(marched.interleave_moving) > 1 {
        variants.extend(drawn_with(cones, offscreen, true, half, marched.over));
    }
    variants.sort_by_key(|variant| *variant as usize);
    variants.dedup();
    variants
}

/// The greatest common divisor of `a` and `b`.
fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// A target of where each block's rays start, `size` blocks across.
fn coarse_texture(device: &wgpu::Device, size: [u32; 2]) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("fk march coarse"),
            size: wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: COARSE,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

/// A target of `format`, `size` pixels across, drawn into and read from.
fn offscreen_texture(
    device: &wgpu::Device,
    size: [u32; 2],
    format: wgpu::TextureFormat,
) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("fk march"),
            size: wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

/// Where a scene's rays start (`coarse`, its cones) and stop (`ahead`, the distances of the
/// scene ahead of it), and, marched one tile in each square of them, its last frame's image (`last`: colour
/// and distance) the others are carried from.
fn coarse_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    coarse: &wgpu::TextureView,
    ahead: &wgpu::TextureView,
    last: (&wgpu::TextureView, &wgpu::TextureView),
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("fk march coarse"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(coarse),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(ahead),
            },
            wgpu::BindGroupEntry {
                binding: 12,
                resource: wgpu::BindingResource::TextureView(last.0),
            },
            wgpu::BindGroupEntry {
                binding: 13,
                resource: wgpu::BindingResource::TextureView(last.1),
            },
        ],
    })
}

#[cfg(test)]
mod tests {
    use fk_app::WindowSize;
    use fk_geometry::Geometry;
    use fk_geometry_euclidean::E3;
    use fk_math::nalgebra::Vector3;
    use fk_scene::{ActiveView, Camera, GlobalPose};

    use super::*;
    use crate::graph::{FrameBuffers, frame_blocks, frame_layout};
    use crate::shadow::{SHADOW_FORMAT, ShadowMap};
    use crate::{Frustum, Lighting};

    type Isometry = <E3 as Geometry>::Isometry;

    /// A box of half-size `params[0].xyz`, coloured `params[1].rgb`.
    fn boxed() -> SdfShader {
        SdfShader {
            name: "test/box.wgsl".to_owned(),
            source: r"#define_import_path fk::sdf
#import fk::march::march
fn sdf_distance(q: vec3<f32>) -> f32 {
    let d = abs(q) - march.params[0].xyz;
    return length(max(d, vec3<f32>(0.0))) + min(max(d.x, max(d.y, d.z)), 0.0);
}
fn sdf_color(q: vec3<f32>) -> vec3<f32> {
    return march.params[1].rgb;
}
"
            .to_owned(),
            libraries: Vec::new(),
        }
    }

    fn params() -> [[f32; 4]; SDF_PARAMS] {
        let mut params = [[0.0; 4]; SDF_PARAMS];
        params[0] = [1.0, 1.0, 1.0, 0.0];
        params[1] = [0.9, 0.3, 0.2, 0.0];
        params
    }

    #[test]
    fn a_field_validates() {
        boxed().validate::<E3>().unwrap();
    }

    #[test]
    fn the_probe_evaluates_the_field() {
        let Some(gpu) = Gpu::headless() else {
            eprintln!("no adapter, skipped");
            return;
        };
        let points = [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, -1.5, 0.0],
            [2.0, 2.0, 1.0],
        ];
        let distances = boxed().probe::<E3>(&gpu, &params(), &points).unwrap();
        let expected = [-1.0, 2.0, 0.5, 2.0_f32.sqrt()];
        for (got, want) in distances.iter().zip(expected) {
            assert!((got - want).abs() < 1e-5, "{distances:?}");
        }
    }

    /// Draws a box 4 units ahead of the eye, `scale` scene units per unit, into a small target
    /// and reads it back as RGBA8.
    fn draw(gpu: &Gpu, half: bool, scale: Real) -> (u32, u32, Vec<u8>) {
        draw_with(gpu, half, scale, 0, RayMarching::default().occlusion)
    }

    /// The box drawn as `draw` does, its rays started in blocks of `coarse` pixels, darkened by
    /// their steps as much as `occlusion` says.
    fn draw_with(
        gpu: &Gpu,
        half: bool,
        scale: Real,
        coarse: u32,
        occlusion: Real,
    ) -> (u32, u32, Vec<u8>) {
        let one = RayMarched {
            scale,
            params: params(),
            coarse,
            ..RayMarched::new(SdfHandle(0))
        };
        draw_scenes(gpu, half, occlusion, &[(-4.0, one)])
    }

    /// The boxes `scenes` drawn together, each `.0` ahead of the eye, as `draw` does.
    fn draw_scenes(
        gpu: &Gpu,
        half: bool,
        occlusion: Real,
        scenes: &[(Real, RayMarched)],
    ) -> (u32, u32, Vec<u8>) {
        draw_frames(gpu, half, occlusion, scenes, 1)
    }

    /// The boxes `scenes` drawn as `draw_scenes` does, `frames` frames running, the eye still:
    /// the last frame's image.
    fn draw_frames(
        gpu: &Gpu,
        half: bool,
        occlusion: Real,
        scenes: &[(Real, RayMarched)],
        frames: usize,
    ) -> (u32, u32, Vec<u8>) {
        let (width, height) = (64, 48);
        let device = &gpu.device;
        let frame_layout = frame_layout(device);
        let frame = FrameBuffers::new(device, &frame_layout);
        let shadow_map = ShadowMap::new(gpu);
        let color_format = wgpu::TextureFormat::Rgba8Unorm;
        let depth_format = wgpu::TextureFormat::Depth32Float;
        let setup = RendererSetup {
            gpu,
            frame_layout: &frame_layout,
            shadow_layout: &shadow_map.layout,
            color_format,
            depth_format,
            shadow_format: SHADOW_FORMAT,
            sample_count: 1,
            elsewhere_layout: &crate::elsewhere::layout(device),
        };
        let mut pipeline = RayMarchPipeline::<E3>::new(&setup);

        let mut world = World::new();
        let mut sdfs = Sdfs::default();
        let sdf = sdfs.add(boxed());
        world.insert_resource(sdfs);
        world.insert_resource(RayMarching {
            half_resolution: half,
            occlusion,
            ..Default::default()
        });
        let mut entity = None;
        for (z, marched) in scenes {
            let at = Isometry::exp(&E3::transvection(&Vector3::new(0.0, 0.0, *z)));
            let spawned = world
                .spawn((
                    ViewRelative::<E3>(at),
                    RayMarched {
                        sdf,
                        ..marched.clone()
                    },
                ))
                .id();
            entity.get_or_insert(spawned);
        }
        let entity = entity.expect("a scene to draw");
        let fov: Real = 1.0;
        let aspect = Real::from(width) / Real::from(height);
        let active = ActiveView {
            entity,
            camera: Camera::<E3>::new(fov, 0.1, 100.0),
            eye: GlobalPose::new(Isometry::identity()),
        };
        let view = FrameView {
            view: &active,
            aspect,
            size: [width, height],
            frustum: Frustum::new(fov, aspect, 100.0),
            shadow: None,
            elsewhere: None,
        };
        let size = WindowSize {
            width,
            height,
            scale_factor: 1.0,
        };
        let blocks = frame_blocks(&world, Some(&active), &Lighting::default(), None, 1, size);
        frame.write(&gpu.queue, &blocks);

        let texture = |format, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | usage,
                view_formats: &[],
            })
        };
        let color = texture(color_format, wgpu::TextureUsages::COPY_SRC);
        let depth = texture(depth_format, wgpu::TextureUsages::empty());
        let bindings = FrameBindings {
            frame: &frame.bind_group,
            shadows: &shadow_map.bind_group,
            elsewhere: None,
        };
        let mut encoder = device.create_command_encoder(&Default::default());
        for frame in 0..frames {
            if frame > 0 {
                gpu.queue.submit([encoder.finish()]);
                encoder = device.create_command_encoder(&Default::default());
            }
            pipeline.prepare(gpu, &mut world, &view);
            pipeline.queue(gpu, &view);
            assert_eq!(GeometryRenderer::<E3>::stats(&pipeline).drawn, scenes.len());
            pipeline.encode(&mut encoder, &bindings);
            let color_view = color.create_view(&Default::default());
            let depth_view = depth.create_view(&Default::default());
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pipeline.render(&mut pass, &bindings);
        }
        let row = width * 4;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(row * height),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            color.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(height),
                },
            },
            color.size(),
        );
        gpu.queue.submit([encoder.finish()]);
        (width, height, read_buffer(gpu, &readback).unwrap())
    }

    fn pixel(image: &(u32, u32, Vec<u8>), x: u32, y: u32) -> [u8; 4] {
        let at = ((y * image.0 + x) * 4) as usize;
        image.2[at..at + 4].try_into().unwrap()
    }

    /// Headless smoke test (lavapipe in CI): the pipeline builds, the box is drawn where it
    /// stands and not around it, at both resolutions.
    #[test]
    fn the_pipeline_draws_a_field() {
        let Some(gpu) = Gpu::headless() else {
            eprintln!("no adapter, skipped");
            return;
        };
        for half in [false, true] {
            let image = draw(&gpu, half, 1.0);
            let centre = pixel(&image, 32, 24);
            assert_eq!(centre[3], 255, "half {half}: a surface at the centre");
            assert!(centre[0] > 20, "half {half}: not black: {centre:?}");
            assert!(
                centre[0] > centre[2],
                "half {half}: in the box's colour: {centre:?}"
            );
            assert_eq!(
                pixel(&image, 1, 1)[3],
                0,
                "half {half}: nothing in the corner"
            );
        }
    }

    /// Rays started from their block's cone find the same surfaces as rays started at the eye:
    /// every pixel covered or not alike, in the same colour (not darkened by the steps, which
    /// are counted differently). Only on an edge may a pixel differ, where a ray passes within
    /// a pixel of the box's edge: whether the march catches that near miss, and on which face,
    /// depends on where its samples fall, plain steps from the cone or relaxed ones from the
    /// eye.
    #[test]
    fn a_renderer_after_another_compiles_nothing_again() {
        let Some(gpu) = Gpu::headless() else {
            eprintln!("no adapter, skipped");
            return;
        };
        let first = draw(&gpu, false, 1.0);
        // What was built ahead of need is done before counting.
        let settled = || {
            let mut counts = gpu.kept.counts();
            for _ in 0..200 {
                std::thread::sleep(std::time::Duration::from_millis(20));
                let now = gpu.kept.counts();
                if now == counts && now.1 == Variant::ALL.len() {
                    break;
                }
                counts = now;
            }
            counts
        };
        let counts = settled();
        assert_eq!(
            counts,
            (1, Variant::ALL.len()),
            "one field, all its pipelines"
        );
        let again = draw(&gpu, false, 1.0);
        assert_eq!(gpu.kept.counts(), counts, "nothing compiled again");
        assert_eq!(first, again);
    }

    #[test]
    fn cones_change_nothing_drawn() {
        let Some(gpu) = Gpu::headless() else {
            eprintln!("no adapter, skipped");
            return;
        };
        // Whether the reference changes within `r` pixels of (x, y): in coverage or colour.
        let edge = |image: &(u32, u32, Vec<u8>), x: u32, y: u32, r: i64| {
            let here = pixel(image, x, y);
            (-r..=r).any(|dy| {
                (-r..=r).any(|dx| {
                    let (nx, ny) = (i64::from(x) + dx, i64::from(y) + dy);
                    if nx < 0 || ny < 0 || nx >= i64::from(image.0) || ny >= i64::from(image.1) {
                        return false;
                    }
                    let there = pixel(image, nx as u32, ny as u32);
                    (0..4).any(|c| here[c].abs_diff(there[c]) > 2)
                })
            })
        };
        for half in [false, true] {
            let from_eye = draw_with(&gpu, half, 1.0, 0, 0.0);
            for coarse in [4, 8] {
                let coned = draw_with(&gpu, half, 1.0, coarse, 0.0);
                let mut differ = 0;
                for y in 0..from_eye.1 {
                    for x in 0..from_eye.0 {
                        let (a, b) = (pixel(&from_eye, x, y), pixel(&coned, x, y));
                        if (0..4).all(|c| a[c].abs_diff(b[c]) <= 2) {
                            continue;
                        }
                        // At half resolution a pixel marched is two across.
                        assert!(
                            edge(&from_eye, x, y, if half { 2 } else { 1 }),
                            "half {half}, coarse {coarse}, ({x}, {y}) off an edge: {a:?} {b:?}"
                        );
                        differ += 1;
                    }
                }
                eprintln!("half {half}, coarse {coarse}: {differ} pixels on edges differ");
            }
        }
    }

    /// A scene marched one tile in nine, the eye still, draws what it does marched whole:
    /// the first frame whole, the other tiles carried just where they were.
    #[test]
    fn an_interleaved_scene_settles_as_marched() {
        let Some(gpu) = Gpu::headless() else {
            eprintln!("no adapter, skipped");
            return;
        };
        let mut front = params();
        front[0] = [0.6, 0.6, 0.6, 0.0];
        front[1] = [0.2, 0.3, 0.9, 0.0];
        let scenes = |interleave: bool| {
            [(
                -4.0,
                RayMarched {
                    params: front,
                    ahead: true,
                    interleave: if interleave { 3 } else { 1 },
                    ..RayMarched::new(SdfHandle(0))
                },
            )]
        };
        let marched = draw_frames(&gpu, false, 0.0, &scenes(false), 1);
        for frames in [1, 2, 5] {
            let interleaved = draw_frames(&gpu, false, 0.0, &scenes(true), frames);
            for y in 0..marched.1 {
                for x in 0..marched.0 {
                    let (a, b) = (pixel(&marched, x, y), pixel(&interleaved, x, y));
                    assert!(
                        (0..4).all(|c| a[c].abs_diff(b[c]) <= 2),
                        "{frames} frames, ({x}, {y}): {a:?} {b:?}"
                    );
                }
            }
        }
    }

    /// A box marched ahead of a bigger one behind it hides it as it did drawn in turn: the
    /// one behind stops its rays where the one ahead is, and the image is the same.
    #[test]
    fn a_scene_ahead_hides_as_before() {
        let Some(gpu) = Gpu::headless() else {
            eprintln!("no adapter, skipped");
            return;
        };
        let mut front = params();
        front[0] = [0.6, 0.6, 0.6, 0.0];
        front[1] = [0.2, 0.3, 0.9, 0.0];
        let mut back = params();
        back[0] = [3.0, 2.0, 0.5, 0.0];
        for half in [false, true] {
            let scenes = |ahead: bool, coarse: u32| {
                [
                    (
                        -9.0,
                        RayMarched {
                            params: back,
                            coarse,
                            ..RayMarched::new(SdfHandle(0))
                        },
                    ),
                    (
                        -4.0,
                        RayMarched {
                            params: front,
                            ahead,
                            ..RayMarched::new(SdfHandle(0))
                        },
                    ),
                ]
            };
            for coarse in [0, 8] {
                let in_turn = draw_scenes(&gpu, half, 0.0, &scenes(false, coarse));
                let ahead = draw_scenes(&gpu, half, 0.0, &scenes(true, coarse));
                assert_eq!(
                    pixel(&ahead, 32, 24)[2],
                    pixel(&in_turn, 32, 24)[2],
                    "half {half}: the box ahead in the middle"
                );
                for y in 0..in_turn.1 {
                    for x in 0..in_turn.0 {
                        let (a, b) = (pixel(&in_turn, x, y), pixel(&ahead, x, y));
                        assert!(
                            (0..4).all(|c| a[c].abs_diff(b[c]) <= 2),
                            "half {half}, coarse {coarse}, ({x}, {y}): {a:?} {b:?}"
                        );
                    }
                }
            }
        }
    }

    /// At half resolution the image is the full resolution's: between pixels marched that
    /// agree, what lies between them; at edges, each pixel's own ray. Only on an edge may a
    /// pixel differ, as in [`cones_change_nothing_drawn`].
    #[test]
    fn half_resolution_draws_as_full() {
        let Some(gpu) = Gpu::headless() else {
            eprintln!("no adapter, skipped");
            return;
        };
        let edge = |image: &(u32, u32, Vec<u8>), x: u32, y: u32| {
            let here = pixel(image, x, y);
            (-2i64..=2).any(|dy| {
                (-2i64..=2).any(|dx| {
                    let (nx, ny) = (i64::from(x) + dx, i64::from(y) + dy);
                    if nx < 0 || ny < 0 || nx >= i64::from(image.0) || ny >= i64::from(image.1) {
                        return false;
                    }
                    let there = pixel(image, nx as u32, ny as u32);
                    (0..4).any(|c| here[c].abs_diff(there[c]) > 2)
                })
            })
        };
        for coarse in [0, 8] {
            let full = draw_with(&gpu, false, 1.0, coarse, 0.0);
            let half = draw_with(&gpu, true, 1.0, coarse, 0.0);
            let mut differ = 0;
            for y in 0..full.1 {
                for x in 0..full.0 {
                    let (a, b) = (pixel(&full, x, y), pixel(&half, x, y));
                    if (0..4).all(|c| a[c].abs_diff(b[c]) <= 3) {
                        continue;
                    }
                    assert!(
                        edge(&full, x, y),
                        "coarse {coarse}, ({x}, {y}) off an edge: {a:?} {b:?}"
                    );
                    differ += 1;
                }
            }
            eprintln!("coarse {coarse}: {differ} pixels on edges differ from full resolution");
        }
    }

    #[test]
    fn a_zoom_shrinks_the_field() {
        let Some(gpu) = Gpu::headless() else {
            eprintln!("no adapter, skipped");
            return;
        };
        // At scale 4, the box of half-size 1 is a quarter of a unit across: it no longer
        // reaches a quarter of the image out from the centre.
        let near = draw(&gpu, false, 1.0);
        let far = draw(&gpu, false, 4.0);
        assert_eq!(pixel(&near, 32 + 12, 24)[3], 255);
        assert_eq!(pixel(&far, 32 + 12, 24)[3], 0);
        assert_eq!(pixel(&far, 32, 24)[3], 255);
    }
}
