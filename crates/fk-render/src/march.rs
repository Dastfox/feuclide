//! The ray-march pipeline: signed distance fields the caller supplies, marched per pixel and
//! composited with the raster path through the shared depth encoding.

use std::marker::PhantomData;

use bevy_ecs::component::Component;
use bevy_ecs::query::{Has, QueryState, Without};
use bevy_ecs::resource::Resource;
use bevy_ecs::world::World;
use encase::ShaderType;
use fk_geometry::{GpuGeometry, GroupElement, SampledMetric};
use fk_math::Real;
use fk_scene::ViewRelative;
use fk_shaders::Shader;

use crate::frame::uniform_bytes;
use crate::gpu::{Vec4f, matrix_columns};
use crate::quotient::{MAX_FOLD_FACES, QuotientView, Unrepeated};
use crate::renderer::{FrameBindings, FrameView, GeometryRenderer, Hidden, OnLayer, RendererSetup};
use crate::{Gpu, RenderStats};

/// How many `vec4`s of its own a [`RayMarched`] scene passes to its distance field.
pub const SDF_PARAMS: usize = 192;

const HALF_COLOR: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Where each block's rays start, and the steps it took to get there.
const COARSE: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Float;
const HALF_DISTANCE: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

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
/// It may also define `fn sdf_kept(q: vec3<f32>) -> bool`: where it is true the surface is
/// kept out of the [`MoshPatch`](crate::MoshPatch)es, neither moshed nor smeared into them;
/// and `fn sdf_moshed(q: vec3<f32>) -> bool`: where it is true (and `sdf_kept` is not) the
/// surface carries the first patch's mosh wherever it is on the image, out of the patch too,
/// block by block round it.
#[derive(Clone, Debug, PartialEq, Eq)]
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

    /// Whether the module keeps some of its surface out of the mosh patches: it defines
    /// `sdf_kept`.
    fn keeps(&self) -> bool {
        self.source.contains("fn sdf_kept")
    }

    /// Whether some of its surface carries the mosh patches' mosh: it defines `sdf_moshed`.
    fn moshes(&self) -> bool {
        self.source.contains("fn sdf_moshed")
    }

    fn compose(
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
            if self.keeps() {
                defs.push("SDF_KEPT");
            }
            if self.moshes() {
                defs.push("SDF_MOSHED");
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
            fold: Fold::default(),
            params: params.map(Vec4f),
        }
    }
}

/// The three pipelines of one field.
struct Pipelines {
    /// Full resolution, into the opaque pass.
    full: wgpu::RenderPipeline,
    /// Half resolution, into the offscreen targets.
    half: wgpu::RenderPipeline,
    /// The offscreen targets into the opaque pass.
    composite: wgpu::RenderPipeline,
    /// `full` and `composite` drawn over everything ([`RayMarched::over`]).
    full_over: wgpu::RenderPipeline,
    composite_over: wgpu::RenderPipeline,
    /// The cones of [`RayMarched::coarse`], into the scene's coarse target.
    coarse: wgpu::RenderPipeline,
}

/// Where the rays of each block of a scene start ([`RayMarched::coarse`]).
struct CoarseTargets {
    size: [u32; 2],
    view: wgpu::TextureView,
}

/// The offscreen targets of one scene at half resolution.
struct HalfTargets {
    size: [u32; 2],
    color: wgpu::TextureView,
    distance: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
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
    full_layout: wgpu::PipelineLayout,
    composite_layout: wgpu::PipelineLayout,
    cone_layout: wgpu::PipelineLayout,
    /// What a scene without cones binds where its coarse target would be, and one without a
    /// scene ahead where that scene's distances would be.
    no_coarse: wgpu::TextureView,
    no_distance: wgpu::TextureView,
    color_format: wgpu::TextureFormat,
    depth_format: wgpu::TextureFormat,
    sample_count: u32,
    /// By field; `None` for one that did not compile, which is not drawn.
    pipelines: Vec<Option<Pipelines>>,
    /// The metric the pipelines were built with.
    metric: Option<String>,
    query: Option<QueryState<Marched<G>, Without<Hidden>>>,
    scenes: Vec<Scene>,
    /// How many of `scenes` are drawn this frame.
    drawn: usize,
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
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
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
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let half_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fk march half"),
            entries: &[texture(0), texture(1)],
        });
        let coarse_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fk march coarse"),
            entries: &[texture(2), texture(3)],
        });
        let no_coarse = coarse_texture(device, [1, 1]);
        let no_distance = offscreen_texture(device, [1, 1], HALF_DISTANCE);
        let pipeline_layout = |groups: &[Option<&wgpu::BindGroupLayout>]| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("fk march"),
                bind_group_layouts: groups,
                immediate_size: 0,
            })
        };
        Self {
            device: device.clone(),
            full_layout: pipeline_layout(&[
                Some(setup.frame_layout),
                Some(&march_layout),
                Some(&coarse_layout),
            ]),
            composite_layout: pipeline_layout(&[
                Some(setup.frame_layout),
                Some(&march_layout),
                Some(&half_layout),
            ]),
            cone_layout: pipeline_layout(&[Some(setup.frame_layout), Some(&march_layout)]),
            march_layout,
            half_layout,
            coarse_layout,
            no_coarse,
            no_distance,
            color_format: setup.color_format,
            depth_format: setup.depth_format,
            sample_count: setup.sample_count,
            pipelines: Vec::new(),
            metric: None,
            query: None,
            scenes: Vec::new(),
            drawn: 0,
            _geometry: PhantomData,
        }
    }

    fn compile_new_sdfs(&mut self, sdfs: &Sdfs) {
        for shader in &sdfs.all()[self.pipelines.len()..] {
            let pipelines =
                match shader.compose(Shader::RayMarch, G::SHADER, self.metric.as_deref()) {
                    Ok(module) => Some(self.build(&shader.name, module)),
                    Err(error) => {
                        let name = &shader.name;
                        tracing::error!(
                            "distance field {name} does not compile, not drawn: {error}"
                        );
                        None
                    }
                };
            self.pipelines.push(pipelines);
        }
    }

    fn build(&self, label: &str, module: fk_shaders::naga::Module) -> Pipelines {
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(module)),
            });
        let pipeline = |layout,
                        entry,
                        targets: &[Option<wgpu::ColorTargetState>],
                        depth: Option<wgpu::CompareFunction>| {
            self.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vertex"),
                        compilation_options: Default::default(),
                        buffers: &[],
                    },
                    primitive: Default::default(),
                    depth_stencil: depth.map(|compare| wgpu::DepthStencilState {
                        format: self.depth_format,
                        depth_write_enabled: Some(true),
                        depth_compare: Some(compare),
                        stencil: Default::default(),
                        bias: Default::default(),
                    }),
                    multisample: wgpu::MultisampleState {
                        count: if depth.is_some() {
                            self.sample_count
                        } else {
                            1
                        },
                        ..Default::default()
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some(entry),
                        compilation_options: Default::default(),
                        targets,
                    }),
                    multiview_mask: None,
                    cache: None,
                })
        };
        let opaque = [Some(self.color_format.into())];
        let (tested, over) = (
            Some(wgpu::CompareFunction::Greater),
            Some(wgpu::CompareFunction::Always),
        );
        Pipelines {
            full: pipeline(&self.full_layout, "fragment", &opaque, tested),
            half: pipeline(
                &self.full_layout,
                "fragment_half",
                &[Some(HALF_COLOR.into()), Some(HALF_DISTANCE.into())],
                None,
            ),
            composite: pipeline(&self.composite_layout, "composite", &opaque, tested),
            full_over: pipeline(&self.full_layout, "fragment", &opaque, over),
            composite_over: pipeline(&self.composite_layout, "composite", &opaque, over),
            coarse: pipeline(
                &self.cone_layout,
                "fragment_coarse",
                &[Some(COARSE.into())],
                None,
            ),
        }
    }

    fn coarse_targets(&self, size: [u32; 2]) -> CoarseTargets {
        CoarseTargets {
            size,
            view: coarse_texture(&self.device, size),
        }
    }

    fn half_targets(&self, size: [u32; 2]) -> HalfTargets {
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
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fk march half"),
            layout: &self.half_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&color),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&distance),
                },
            ],
        });
        HalfTargets {
            size,
            color,
            distance,
            bind_group,
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
                starts: None,
                layer: false,
                over: false,
            });
        }
        &mut self.scenes[index]
    }
}

impl<G: GpuGeometry> GeometryRenderer<G> for RayMarchPipeline<G> {
    fn warm(&mut self, _: &Gpu, world: &mut World) {
        let metric = world
            .get_resource::<RayMarching>()
            .and_then(|settings| settings.metric.clone());
        if metric != self.metric {
            self.metric = metric;
            self.pipelines.clear();
        }
        if let Some(sdfs) = world.get_resource::<Sdfs>() {
            self.compile_new_sdfs(sdfs);
        }
    }

    fn prepare(&mut self, gpu: &Gpu, world: &mut World, view: &FrameView<'_, G>) {
        self.warm(gpu, world);
        let settings = world
            .get_resource::<RayMarching>()
            .cloned()
            .unwrap_or_default();
        let half_size = view.size.map(|n| n.div_ceil(2).max(1));

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
                fold,
                params: marched.params.map(Vec4f),
            };
            let offscreen = half || is_ahead;
            let needs_targets = offscreen
                && self.scenes.get(index).is_none_or(|scene| {
                    scene
                        .half
                        .as_ref()
                        .is_none_or(|targets| targets.size != target_size)
                });
            let targets = needs_targets.then(|| self.half_targets(target_size));
            let needs_coarse = cones > 0
                && self.scenes.get(index).is_none_or(|scene| {
                    scene
                        .coarse
                        .as_ref()
                        .is_none_or(|targets| targets.size != coarse_size)
                });
            let coarse = needs_coarse.then(|| self.coarse_targets(coarse_size));
            let scene = self.scene(index);
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
            let starts = coarse_bind_group(&self.device, &self.coarse_layout, coarse, distance);
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
            let (Some(Some(pipelines)), Some(targets)) =
                (self.pipelines.get(scene.sdf as usize), &scene.coarse)
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
            pass.set_pipeline(&pipelines.coarse);
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
            let (Some(Some(pipelines)), Some(targets)) =
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
            pass.set_pipeline(&pipelines.half);
            pass.set_bind_group(0, bindings.frame, &[]);
            pass.set_bind_group(1, &scene.bind_group, &[]);
            pass.set_bind_group(2, self.starts(scene), &[]);
            pass.draw(0..3, 0..1);
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
            let Some(Some(pipelines)) = self.pipelines.get(scene.sdf as usize) else {
                continue;
            };
            match (&scene.half, scene.halved, scene.over) {
                (Some(targets), true, over) => {
                    pass.set_pipeline(if over {
                        &pipelines.composite_over
                    } else {
                        &pipelines.composite
                    });
                    pass.set_bind_group(2, &targets.bind_group, &[]);
                }
                (_, _, true) => {
                    pass.set_pipeline(&pipelines.full_over);
                    pass.set_bind_group(2, self.starts(scene), &[]);
                }
                _ => {
                    pass.set_pipeline(&pipelines.full);
                    pass.set_bind_group(2, self.starts(scene), &[]);
                }
            }
            pass.set_bind_group(0, bindings.frame, &[]);
            pass.set_bind_group(1, &scene.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }
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
/// scene ahead of it).
fn coarse_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    coarse: &wgpu::TextureView,
    ahead: &wgpu::TextureView,
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
        pipeline.prepare(gpu, &mut world, &view);
        pipeline.queue(gpu, &view);
        assert_eq!(GeometryRenderer::<E3>::stats(&pipeline).drawn, scenes.len());

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
        pipeline.encode(&mut encoder, &bindings);
        {
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
