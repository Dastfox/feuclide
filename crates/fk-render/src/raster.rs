//! The raster pipeline for constant-curvature geometries: instanced meshes, painted material.

use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::ops::Range;
use std::time::Instant;

use bevy_ecs::query::{Has, QueryState, With, Without};
use bevy_ecs::world::World;
use bytemuck::{Pod, Zeroable};
use fk_geometry::{Geometry, GpuGeometry, GroupElement};
use fk_scene::ViewRelative;
use wgpu::util::DeviceExt;

use crate::deform::{CastsNoShadow, Deform, Deforms};
use crate::elsewhere::{MAX_OTHER_EYES, SeenBy, ShowsElsewhere, ShowsOther};
use crate::gpu::matrix_columns;
use crate::mesh::{MeshInstances, Meshes, Vertex};
use crate::normal_map::NormalMaps;
use crate::parting::{Parting, SeeThrough};
use crate::quotient::{QuotientView, Unrepeated};
use crate::renderer::{FrameBindings, FrameView, GeometryRenderer, Hidden, OnLayer, RendererSetup};
use crate::{Gpu, RenderStats};

/// One instance as the GPU reads it.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuInstance {
    /// Columns of the camera-relative isometry matrix.
    iso: [[f32; 4]; 4],
    color: [f32; 4],
    data: [f32; 4],
}

struct GpuMesh {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
    radius: f64,
}

/// A candidate for drawing: an instance made camera-relative, not yet culled.
struct Candidate<G: Geometry> {
    /// Which pipelines: 0 the engine's, `k + 1` the deform `k`.
    slot: u32,
    mesh: u32,
    relative: G::Isometry,
    color: [f32; 4],
    data: [f32; 4],
    casts: bool,
    layer: bool,
    /// A window onto an other eye's view: 0 the [`Elsewhere`](crate::Elsewhere) eye's, `k + 1`
    /// eye `k` of the [`OtherEyes`](crate::OtherEyes).
    window: Option<usize>,
    /// Which eye draws it, if only one does.
    seen_by: Option<SeenBy>,
    /// Cut away in the [`Parting`]'s cones: [`SeeThrough`], while the parting cuts.
    see_through: bool,
}

/// Which pipelines draw a bin's instances, and of what mesh: the slot, whether they are seen
/// through, the mesh.
type BinKey = (u32, bool, u32);

/// Instances of one mesh drawn through one slot's pipelines.
struct Draw {
    slot: u32,
    /// Through the slot's [`SeeThrough`] pipelines.
    see_through: bool,
    mesh: u32,
    instances: Range<u32>,
}

/// The pipelines of one deform: the painted surfaces, the same cut off at the view's clip
/// plane (for the other eye's image) and the shadow casters, by their far sides or by both
/// ([`Shadows::two_sided`](crate::Shadows::two_sided)); and the surfaces to be seen through,
/// built the first time an entity marked [`SeeThrough`] is there to draw through this deform.
struct Pipelines {
    shader: wgpu::ShaderModule,
    label: String,
    opaque: wgpu::RenderPipeline,
    clipped: wgpu::RenderPipeline,
    shadow: wgpu::RenderPipeline,
    shadow_both: wgpu::RenderPipeline,
    see_through: Option<SeeThroughPipelines>,
}

/// The painted pipelines of a deform for the surfaces to be seen through: cut away in the
/// [`Parting`]'s cones, and the same cut off at the view's clip plane as well.
struct SeeThroughPipelines {
    opaque: wgpu::RenderPipeline,
    clipped: wgpu::RenderPipeline,
}

impl Pipelines {
    /// The painted pipeline, cut off at the clip plane or not, seen through or not (drawn
    /// whole while the see-through ones are not built).
    fn painted(&self, clipped: bool, see_through: bool) -> &wgpu::RenderPipeline {
        let seen = self.see_through.as_ref().filter(|_| see_through);
        match (seen, clipped) {
            (Some(seen), false) => &seen.opaque,
            (Some(seen), true) => &seen.clipped,
            (None, false) => &self.opaque,
            (None, true) => &self.clipped,
        }
    }
}

/// The normal maps as the shader samples them: a texture array, the flat map first.
struct GpuNormalMaps {
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    /// How many maps it holds, besides the flat one, and how big they are.
    uploaded: (usize, u32),
    /// Where the maps flatten with distance (x from, y to) and whether they are on at all (z),
    /// as the shader reads it.
    fade: wgpu::Buffer,
    faded: [f32; 3],
}

impl GpuNormalMaps {
    fn new(gpu: &Gpu) -> Self {
        let layout = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("fk normal maps"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
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
        let faded = [f32::MAX, f32::MAX, 1.0];
        let fade = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fk normal maps' fade"),
                contents: bytemuck::cast_slice(&[faded[0], faded[1], faded[2], 0.0]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let bind_group = Self::upload(gpu, &layout, &NormalMaps::new(1), &fade);
        Self {
            layout,
            bind_group,
            uploaded: (0, 1),
            fade,
            faded,
        }
    }

    /// Uploads `maps` again if any were added since, and their fade if it changed; whether
    /// the maps were.
    fn update(&mut self, gpu: &Gpu, maps: &NormalMaps) -> bool {
        let now = (maps.len(), maps.size());
        let uploading = now != self.uploaded;
        if uploading {
            self.bind_group = Self::upload(gpu, &self.layout, maps, &self.fade);
            self.uploaded = now;
        }
        let [from, to] = maps.fade().map(|d| d.min(f32::MAX as f64) as f32);
        let faded = [from, to, if maps.enabled() { 1.0 } else { 0.0 }];
        if faded != self.faded {
            gpu.queue.write_buffer(
                &self.fade,
                0,
                bytemuck::cast_slice(&[faded[0], faded[1], faded[2], 0.0]),
            );
            self.faded = faded;
        }
        uploading
    }

    fn upload(
        gpu: &Gpu,
        layout: &wgpu::BindGroupLayout,
        maps: &NormalMaps,
        fade: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let layers = maps.layers();
        let size = maps.size();
        let texels = maps.texels_with_mips();
        let texture = gpu.device.create_texture_with_data(
            &gpu.queue,
            &wgpu::TextureDescriptor {
                label: Some("fk normal maps"),
                size: wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: layers.len() as u32,
                },
                mip_level_count: maps.mip_levels(),
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &texels,
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("fk normal maps"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            anisotropy_clamp: 4,
            ..Default::default()
        });
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fk normal maps"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: fade.as_entire_binding(),
                },
            ],
        })
    }
}

/// What every pipeline is built with.
struct Builder {
    device: wgpu::Device,
    opaque_layout: wgpu::PipelineLayout,
    shadow_layout: wgpu::PipelineLayout,
    color_format: wgpu::TextureFormat,
    depth_format: wgpu::TextureFormat,
    shadow_format: wgpu::TextureFormat,
    sample_count: u32,
}

type Drawables<G> = (
    &'static ViewRelative<G>,
    &'static MeshInstances<G>,
    Option<&'static Deform>,
    Has<CastsNoShadow>,
    Has<OnLayer>,
    Has<ShowsElsewhere>,
    Option<&'static ShowsOther>,
    Has<Unrepeated>,
    Option<&'static SeenBy>,
    Has<SeeThrough>,
);

/// The deforms of the entities to be seen through.
type SeeThroughQuery = QueryState<Option<&'static Deform>, (With<SeeThrough>, Without<Hidden>)>;

/// Draws every [`MeshInstances`] with classic rasterization and the geometry's projection.
///
/// Instances are composed camera-relative in `f64` (`view⁻¹ ∘ entity ∘ instance`), culled by
/// their mesh's geodesic bounding ball, grouped by deform and mesh and drawn with one instanced
/// call per group. Their vertices go through the entity's [`Deform`], if it has one. The
/// fragment stage paints flat colours: banded light, a violet shadow tint where the light does
/// not reach (turned away, or in the shadow map's shadow) and geodesic-distance fog to the sky.
///
/// Instances within the [`Shadows`](crate::Shadows) map cast into it, unless their entity has
/// [`CastsNoShadow`].
///
/// Entities marked [`ShowsElsewhere`] (or [`ShowsOther`]) are drawn after the rest, by a
/// pipeline of their own, as windows onto the [`Elsewhere`](crate::Elsewhere) view (or one of
/// the [`OtherEyes`](crate::OtherEyes)).
///
/// Entities marked [`SeeThrough`] are drawn, while the [`Parting`] cuts, by pipelines of their
/// own that discard their fragments within its cones (in the other eye's image, cut off at its
/// clip as well), built the first time one is there to draw through its deform. They cast their
/// shadows whole.
pub struct RasterPipeline<G: GpuGeometry> {
    builder: Builder,
    normal_maps: GpuNormalMaps,
    /// Draws the windows onto the other eye's view: the engine's vertex stage, no deform.
    window: wgpu::RenderPipeline,
    /// By slot; `None` for a deform that did not compile, drawn as built instead.
    pipelines: Vec<Option<Pipelines>>,
    meshes: Vec<GpuMesh>,
    query: Option<QueryState<Drawables<G>, Without<Hidden>>>,
    see_through_query: Option<SeeThroughQuery>,
    candidates: Vec<Candidate<G>>,
    bins: BTreeMap<BinKey, Vec<GpuInstance>>,
    shadow_bins: BTreeMap<BinKey, Vec<GpuInstance>>,
    layer_bins: BTreeMap<BinKey, Vec<GpuInstance>>,
    /// By eye, as [`Candidate::window`] numbers them.
    window_bins: [BTreeMap<BinKey, Vec<GpuInstance>>; 1 + MAX_OTHER_EYES],
    instances: Option<wgpu::Buffer>,
    draws: Vec<Draw>,
    shadow_draws: Vec<Draw>,
    layer_draws: Vec<Draw>,
    window_draws: [Vec<Draw>; 1 + MAX_OTHER_EYES],
    /// Whether the view last queued is cut off at its clip plane.
    clipped: bool,
    /// Whether the casters last queued cast by both their sides.
    two_sided: bool,
    stats: RenderStats,
    _geometry: PhantomData<fn() -> G>,
}

impl<G: GpuGeometry> RasterPipeline<G> {
    /// Builds the pipeline.
    ///
    /// # Panics
    ///
    /// If the shader does not compose for `G`, which the shader tests rule out for every
    /// geometry that has a module.
    pub fn new(setup: &RendererSetup<'_>) -> Self {
        let device = &setup.gpu.device;
        let layout = |label, groups: &[Option<&wgpu::BindGroupLayout>]| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: groups,
                immediate_size: 0,
            })
        };
        let normal_maps = GpuNormalMaps::new(setup.gpu);
        let builder = Builder {
            device: device.clone(),
            opaque_layout: layout(
                "fk raster",
                &[
                    Some(setup.frame_layout),
                    Some(setup.shadow_layout),
                    Some(&normal_maps.layout),
                ],
            ),
            shadow_layout: layout("fk raster shadow", &[Some(setup.frame_layout)]),
            color_format: setup.color_format,
            depth_format: setup.depth_format,
            shadow_format: setup.shadow_format,
            sample_count: setup.sample_count,
        };
        let module = fk_shaders::compose(fk_shaders::Shader::Raster, G::SHADER)
            .unwrap_or_else(|error| panic!("raster shader for {}: {error}", G::NAME));
        let window_layout = layout(
            "fk raster window",
            &[
                Some(setup.frame_layout),
                Some(setup.shadow_layout),
                Some(setup.elsewhere_layout),
            ],
        );
        let window = builder.build_window(module.clone(), &window_layout);
        let engine = builder.build("fk raster", module);
        Self {
            builder,
            normal_maps,
            window,
            pipelines: vec![Some(engine)],
            meshes: Vec::new(),
            query: None,
            see_through_query: None,
            candidates: Vec::new(),
            bins: BTreeMap::new(),
            shadow_bins: BTreeMap::new(),
            layer_bins: BTreeMap::new(),
            window_bins: Default::default(),
            instances: None,
            draws: Vec::new(),
            shadow_draws: Vec::new(),
            layer_draws: Vec::new(),
            window_draws: Default::default(),
            clipped: false,
            two_sided: false,
            stats: RenderStats::default(),
            _geometry: PhantomData,
        }
    }

    fn upload_new_meshes(&mut self, gpu: &Gpu, meshes: &Meshes) {
        for mesh in &meshes.all()[self.meshes.len()..] {
            let buffer = |contents: &[u8], usage| {
                gpu.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("fk mesh"),
                        contents,
                        usage,
                    })
            };
            self.meshes.push(GpuMesh {
                vertices: buffer(
                    bytemuck::cast_slice(&mesh.vertices),
                    wgpu::BufferUsages::VERTEX,
                ),
                indices: buffer(
                    bytemuck::cast_slice(&mesh.indices),
                    wgpu::BufferUsages::INDEX,
                ),
                index_count: u32::try_from(mesh.indices.len()).expect("mesh too large"),
                radius: mesh.radius,
            });
        }
    }

    fn compile_new_deforms(&mut self, deforms: &Deforms) {
        for shader in &deforms.all()[self.pipelines.len() - 1..] {
            let pipelines = match shader.compose(G::SHADER) {
                Ok(module) => Some(self.builder.build(&shader.name, module)),
                Err(error) => {
                    let name = &shader.name;
                    tracing::error!("deform {name} does not compile, drawn as built: {error}");
                    None
                }
            };
            self.pipelines.push(pipelines);
        }
    }

    /// The slot an entity with `deform` is drawn through: its deform's, or the engine's when
    /// that did not compile.
    fn slot_of(&self, deform: Option<&Deform>) -> u32 {
        let slot = deform.map_or(0, |deform| deform.0.0 + 1);
        match self.pipelines.get(slot as usize) {
            Some(Some(_)) => slot,
            _ => 0,
        }
    }

    /// Builds the see-through pipelines of every deform an entity marked [`SeeThrough`] is
    /// drawn through and that has none yet.
    fn compile_see_through(&mut self, world: &mut World) {
        let query = self
            .see_through_query
            .get_or_insert_with(|| world.query_filtered());
        let mut deforms: Vec<Option<Deform>> = Vec::new();
        for deform in query.iter(world) {
            if !deforms.contains(&deform.copied()) {
                deforms.push(deform.copied());
            }
        }
        for deform in deforms {
            let slot = self.slot_of(deform.as_ref());
            let Some(Some(pipelines)) = self.pipelines.get_mut(slot as usize) else {
                continue;
            };
            if pipelines.see_through.is_none() {
                let started = Instant::now();
                pipelines.see_through = Some(self.builder.build_see_through(pipelines));
                let label = &pipelines.label;
                tracing::info!(
                    "{label} seen through, compiled in {:.0?}",
                    started.elapsed()
                );
            }
        }
    }

    /// The draws of `bins`, appended to `all`.
    fn collect(
        bins: &mut BTreeMap<BinKey, Vec<GpuInstance>>,
        all: &mut Vec<GpuInstance>,
        draws: &mut Vec<Draw>,
    ) {
        draws.clear();
        for (&(slot, see_through, mesh), bin) in bins.iter_mut() {
            if bin.is_empty() {
                continue;
            }
            let start = all.len() as u32;
            all.append(bin);
            draws.push(Draw {
                slot,
                see_through,
                mesh,
                instances: start..all.len() as u32,
            });
        }
    }

    /// Records `draws`, `bind` setting the pipeline and its bind groups for each slot and
    /// whether its instances are seen through.
    fn draw(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        draws: &[Draw],
        bind: impl Fn(&mut wgpu::RenderPass<'_>, &Pipelines, bool),
    ) {
        let Some(instances) = &self.instances else {
            return;
        };
        let mut bound = None;
        for draw in draws {
            if bound != Some((draw.slot, draw.see_through)) {
                let Some(Some(pipelines)) = self.pipelines.get(draw.slot as usize) else {
                    continue;
                };
                bind(pass, pipelines, draw.see_through);
                pass.set_vertex_buffer(1, instances.slice(..));
                bound = Some((draw.slot, draw.see_through));
            }
            let mesh = &self.meshes[draw.mesh as usize];
            pass.set_vertex_buffer(0, mesh.vertices.slice(..));
            pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..mesh.index_count, 0, draw.instances.clone());
        }
    }
}

/// The vertex buffers every pipeline reads: the mesh's vertices, then the instances.
fn vertex_layouts<'a>(
    vertex_attributes: &'a [wgpu::VertexAttribute],
    instance_attributes: &'a [wgpu::VertexAttribute],
) -> [Option<wgpu::VertexBufferLayout<'a>>; 2] {
    [
        Some(wgpu::VertexBufferLayout {
            array_stride: size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: vertex_attributes,
        }),
        Some(wgpu::VertexBufferLayout {
            array_stride: size_of::<GpuInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: instance_attributes,
        }),
    ]
}

impl Builder {
    /// The windows' pipeline, from the engine's module.
    fn build_window(
        &self,
        module: fk_shaders::naga::Module,
        layout: &wgpu::PipelineLayout,
    ) -> wgpu::RenderPipeline {
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("fk raster window"),
                source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(module)),
            });
        let vertex_attributes = wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4, 8 => Float32x4, 9 => Float32x4, 10 => Uint8x4];
        let instance_attributes = wgpu::vertex_attr_array![
            2 => Float32x4, 3 => Float32x4, 4 => Float32x4, 5 => Float32x4,
            6 => Float32x4, 7 => Float32x4,
        ];
        let buffers = vertex_layouts(&vertex_attributes, &instance_attributes);
        self.device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("fk raster window"),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vertex"),
                    compilation_options: Default::default(),
                    buffers: &buffers,
                },
                primitive: wgpu::PrimitiveState {
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: self.depth_format,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Greater),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: self.sample_count,
                    ..Default::default()
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("window_fragment"),
                    compilation_options: Default::default(),
                    targets: &[Some(self.color_format.into())],
                }),
                multiview_mask: None,
                cache: None,
            })
    }

    /// A painted pipeline of `shader`, its fragment stage `fragment`.
    fn painted(
        &self,
        label: &str,
        shader: &wgpu::ShaderModule,
        fragment: &str,
    ) -> wgpu::RenderPipeline {
        let vertex_attributes = wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4, 8 => Float32x4, 9 => Float32x4, 10 => Uint8x4];
        let instance_attributes = wgpu::vertex_attr_array![
            2 => Float32x4, 3 => Float32x4, 4 => Float32x4, 5 => Float32x4,
            6 => Float32x4, 7 => Float32x4,
        ];
        let buffers = vertex_layouts(&vertex_attributes, &instance_attributes);
        self.device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&self.opaque_layout),
                vertex: wgpu::VertexState {
                    module: shader,
                    entry_point: Some("vertex"),
                    compilation_options: Default::default(),
                    buffers: &buffers,
                },
                primitive: wgpu::PrimitiveState {
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: self.depth_format,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Greater),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: self.sample_count,
                    ..Default::default()
                },
                fragment: Some(wgpu::FragmentState {
                    module: shader,
                    entry_point: Some(fragment),
                    compilation_options: Default::default(),
                    targets: &[Some(self.color_format.into())],
                }),
                multiview_mask: None,
                cache: None,
            })
    }

    /// The see-through pipelines of the deform `pipelines` were built for.
    fn build_see_through(&self, pipelines: &Pipelines) -> SeeThroughPipelines {
        let label = format!("{} seen through", pipelines.label);
        SeeThroughPipelines {
            opaque: self.painted(&label, &pipelines.shader, "see_through_fragment"),
            clipped: self.painted(&label, &pipelines.shader, "clipped_see_through_fragment"),
        }
    }

    fn build(&self, label: &str, module: fk_shaders::naga::Module) -> Pipelines {
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(module)),
            });
        let vertex_attributes = wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4, 8 => Float32x4, 9 => Float32x4, 10 => Uint8x4];
        let instance_attributes = wgpu::vertex_attr_array![
            2 => Float32x4, 3 => Float32x4, 4 => Float32x4, 5 => Float32x4,
            6 => Float32x4, 7 => Float32x4,
        ];
        let buffers = vertex_layouts(&vertex_attributes, &instance_attributes);
        let opaque = self.painted(label, &shader, "fragment");
        let clipped = self.painted(label, &shader, "clipped_fragment");
        let shadow = |cull_mode| {
            self.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&self.shadow_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("shadow_vertex"),
                        compilation_options: Default::default(),
                        buffers: &buffers,
                    },
                    primitive: wgpu::PrimitiveState {
                        cull_mode,
                        ..Default::default()
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: self.shadow_format,
                        depth_write_enabled: Some(true),
                        depth_compare: Some(wgpu::CompareFunction::LessEqual),
                        stencil: Default::default(),
                        bias: wgpu::DepthBiasState {
                            constant: 0,
                            slope_scale: 1.0,
                            clamp: 0.0,
                        },
                    }),
                    multisample: Default::default(),
                    fragment: None,
                    multiview_mask: None,
                    cache: None,
                })
        };
        Pipelines {
            shader: shader.clone(),
            label: label.to_owned(),
            opaque,
            clipped,
            // The sides turned away from the light cast: the lit sides never shadow
            // themselves, and the dark sides are painted in the shadow colour anyway.
            shadow: shadow(Some(wgpu::Face::Front)),
            // Every side: where solids meet, the nearer side shades the corner.
            shadow_both: shadow(None),
            see_through: None,
        }
    }
}

impl<G: GpuGeometry> GeometryRenderer<G> for RasterPipeline<G> {
    fn warm(&mut self, gpu: &Gpu, world: &mut World) {
        if let Some(meshes) = world.get_resource::<Meshes>() {
            let (before, started) = (self.meshes.len(), Instant::now());
            self.upload_new_meshes(gpu, meshes);
            if self.meshes.len() > before {
                let new = self.meshes.len() - before;
                tracing::info!("{new} meshes uploaded in {:.0?}", started.elapsed());
            }
        }
        if let Some(deforms) = world.get_resource::<Deforms>() {
            let (before, started) = (self.pipelines.len(), Instant::now());
            self.compile_new_deforms(deforms);
            if self.pipelines.len() > before {
                let new = self.pipelines.len() - before;
                tracing::info!("{new} deforms compiled in {:.0?}", started.elapsed());
            }
        }
        self.compile_see_through(world);
        if let Some(maps) = world.get_resource::<NormalMaps>() {
            let started = Instant::now();
            if self.normal_maps.update(gpu, maps) {
                tracing::info!("normal maps uploaded in {:.0?}", started.elapsed());
            }
        }
    }

    fn prepare(&mut self, gpu: &Gpu, world: &mut World, view: &FrameView<'_, G>) {
        self.warm(gpu, world);
        // In a quotient, each deck translate `γ` made relative to the eye drawing, `D⁻¹ ∘ γ ∘ D`:
        // what carries a pose relative to that eye to its copy's.
        let eye = view.view.eye.isometry;
        let shifts: Vec<G::Isometry> = world
            .get_resource::<QuotientView<G>>()
            .map(|space| {
                let to_eye = eye.inverse();
                space
                    .translates()
                    .iter()
                    .map(|translate| to_eye.compose(translate).compose(&eye))
                    .collect()
            })
            .unwrap_or_default();
        let cuts = world.get_resource::<Parting>().is_some_and(Parting::cuts);
        // Which slots can draw instances seen through.
        let seen_through: Vec<bool> = self
            .pipelines
            .iter()
            .map(|pipelines| pipelines.as_ref().is_some_and(|p| p.see_through.is_some()))
            .collect();
        let query = self.query.get_or_insert_with(|| world.query_filtered());
        self.candidates.clear();
        for (
            relative,
            drawable,
            deform,
            casts_none,
            layer,
            shows,
            other,
            unrepeated,
            seen_by,
            see_through,
        ) in query.iter(world)
        {
            let window = if shows {
                Some(0)
            } else {
                other
                    .filter(|other| other.0 < MAX_OTHER_EYES)
                    .map(|other| other.0 + 1)
            };
            let relative = view.relative(&relative.0);
            let slot = deform.map_or(0, |deform| deform.0.0 + 1);
            let slot = match self.pipelines.get(slot as usize) {
                Some(Some(_)) => slot,
                _ => 0,
            };
            // Seen through while the parting cuts, its pipelines built; whole otherwise.
            let see_through = see_through
                && cuts
                && window.is_none()
                && seen_through.get(slot as usize).copied().unwrap_or(false);
            let copies = if unrepeated || window.is_some() {
                &[][..]
            } else {
                &shifts[..]
            };
            let at = std::iter::once(relative).chain(copies.iter().map(|s| s.compose(&relative)));
            for relative in at {
                for instance in &drawable.instances {
                    self.candidates.push(Candidate {
                        slot,
                        mesh: drawable.mesh.0,
                        relative: relative.compose(&instance.pose),
                        color: instance.color.to_array(),
                        data: instance.data,
                        casts: !casts_none && window.is_none(),
                        layer,
                        window,
                        seen_by: seen_by.copied(),
                        see_through,
                    });
                }
            }
        }
    }

    fn queue(&mut self, gpu: &Gpu, view: &FrameView<'_, G>) {
        let origin = G::origin();
        let mut stats = RenderStats::default();
        self.clipped = view.clipped;
        self.two_sided = view.shadow.is_some_and(|shadow| shadow.two_sided);
        // On a sphere every instance is seen twice, the second time the long way round.
        let antipode = G::constant_curvature()
            .filter(|&k| k > 0.0)
            .map(|k| std::f64::consts::PI / k.sqrt());
        let along = view.shadow.map(|shadow| shadow.along());
        for candidate in &self.candidates {
            // No window onto any other eye's view in any eye's image.
            if candidate.window.is_some() && view.elsewhere.is_some() {
                continue;
            }
            // Drawn by one eye only.
            match (candidate.seen_by, view.elsewhere.is_some()) {
                (Some(SeenBy::Eye), true) | (Some(SeenBy::Elsewhere), false) => continue,
                _ => {}
            }
            let Some(mesh) = self.meshes.get(candidate.mesh as usize) else {
                continue;
            };
            let centre = G::apply(&candidate.relative, &origin);
            let seen = view.frustum.sees::<G>(&centre, mesh.radius);
            // Cast into the map, and only where its shadow can fall in view.
            let casts = candidate.casts
                && view
                    .shadow
                    .is_some_and(|shadow| shadow.sees::<G>(&centre, mesh.radius))
                && along
                    .is_some_and(|along| view.frustum.sees_cast::<G>(&centre, mesh.radius, &along));
            if !seen {
                stats.culled += 1;
            }
            // Seen the long way round, it stands where its antipode would, `π − d` away the
            // other way: drawn there again, marked by its colour's alpha turned negative (the
            // shader's far side).
            let far_seen = antipode.is_some_and(|round| {
                let Some(towards) = G::log(&origin, &centre) else {
                    return true;
                };
                let d = G::norm(&origin, &towards);
                d <= 1e-12
                    || view.frustum.sees::<G>(
                        &G::exp(&origin, &(towards * (-(round - d) / d))),
                        mesh.radius,
                    )
            });
            if !seen && !casts && !far_seen {
                continue;
            }
            // The colour's alpha is the image's alpha on the surface, 1 at least.
            let mut instance = GpuInstance {
                iso: matrix_columns(&G::isometry_matrix(&candidate.relative)),
                color: candidate.color,
                data: candidate.data,
            };
            instance.color[3] = instance.color[3].max(1.0);
            let key = (candidate.slot, candidate.see_through, candidate.mesh);
            // The shadows are cast whole, alongside the rest's.
            let cast_key = (candidate.slot, false, candidate.mesh);
            if let Some(eye) = candidate.window {
                if seen {
                    self.window_bins[eye]
                        .entry((0, false, candidate.mesh))
                        .or_default()
                        .push(instance);
                }
                continue;
            }
            if seen {
                self.bins.entry(key).or_default().push(instance);
                if candidate.layer {
                    self.layer_bins.entry(key).or_default().push(instance);
                }
            }
            if far_seen {
                let mut far = instance;
                far.color[3] = -instance.color[3];
                self.bins.entry(key).or_default().push(far);
                if candidate.layer {
                    self.layer_bins.entry(key).or_default().push(far);
                }
            }
            if casts {
                self.shadow_bins.entry(cast_key).or_default().push(instance);
            }
        }
        let mut all = Vec::with_capacity(self.candidates.len());
        Self::collect(&mut self.bins, &mut all, &mut self.draws);
        stats.drawn = all.len();
        stats.draws = self.draws.len();
        Self::collect(&mut self.shadow_bins, &mut all, &mut self.shadow_draws);
        Self::collect(&mut self.layer_bins, &mut all, &mut self.layer_draws);
        for (bins, draws) in self.window_bins.iter_mut().zip(&mut self.window_draws) {
            Self::collect(bins, &mut all, draws);
        }
        self.stats = stats;
        if all.is_empty() {
            return;
        }
        let bytes: &[u8] = bytemuck::cast_slice(&all);
        let fits = self
            .instances
            .as_ref()
            .is_some_and(|buffer| buffer.size() >= bytes.len() as u64);
        if !fits {
            self.instances = Some(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fk instances"),
                size: (bytes.len() as u64).next_power_of_two(),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        if let Some(buffer) = &self.instances {
            gpu.queue.write_buffer(buffer, 0, bytes);
        }
    }

    fn render_shadows(&self, pass: &mut wgpu::RenderPass<'_>, frame: &wgpu::BindGroup) {
        self.draw(pass, &self.shadow_draws, |pass, pipelines, _| {
            pass.set_pipeline(if self.two_sided {
                &pipelines.shadow_both
            } else {
                &pipelines.shadow
            });
            pass.set_bind_group(0, frame, &[]);
        });
    }

    fn render(&self, pass: &mut wgpu::RenderPass<'_>, bindings: &FrameBindings<'_>) {
        self.draw(pass, &self.draws, |pass, pipelines, see_through| {
            pass.set_pipeline(pipelines.painted(self.clipped, see_through));
            pass.set_bind_group(0, bindings.frame, &[]);
            pass.set_bind_group(1, bindings.shadows, &[]);
            pass.set_bind_group(2, &self.normal_maps.bind_group, &[]);
        });
        let images = std::iter::once(bindings.elsewhere).chain(bindings.others);
        for (draws, image) in self.window_draws.iter().zip(images) {
            let Some(image) = image else { continue };
            self.draw(pass, draws, |pass, _, _| {
                pass.set_pipeline(&self.window);
                pass.set_bind_group(0, bindings.frame, &[]);
                pass.set_bind_group(1, bindings.shadows, &[]);
                pass.set_bind_group(2, image, &[]);
            });
        }
    }

    fn render_layer(&self, pass: &mut wgpu::RenderPass<'_>, bindings: &FrameBindings<'_>) {
        self.draw(pass, &self.layer_draws, |pass, pipelines, see_through| {
            pass.set_pipeline(pipelines.painted(false, see_through));
            pass.set_bind_group(0, bindings.frame, &[]);
            pass.set_bind_group(1, bindings.shadows, &[]);
            pass.set_bind_group(2, &self.normal_maps.bind_group, &[]);
        });
    }

    fn stats(&self) -> RenderStats {
        self.stats
    }
}

#[cfg(test)]
mod tests {
    use fk_app::WindowSize;
    use fk_geometry_euclidean::E3;
    use fk_math::nalgebra::Vector3;
    use fk_scene::{ActiveView, Camera, GlobalPose};

    use super::*;
    use crate::elsewhere::ElsewhereShared;
    use crate::graph::{FrameBuffers, frame_blocks, frame_layout};
    use crate::march::read_buffer;
    use crate::parting::Part;
    use crate::shadow::{SHADOW_FORMAT, ShadowMap};
    use crate::{Color, Frustum, Lighting, MeshBuilder, NormalMaps, PointLight, Spot, shapes};

    type Isometry = <E3 as Geometry>::Isometry;

    const SIZE: (u32, u32) = (64, 48);

    /// A window 4 units ahead of the eye, covering the middle of a small target, onto an image
    /// red above and blue below; read back as RGBA8.
    fn draw_window(gpu: &Gpu, flip: bool) -> Vec<u8> {
        let mut world = World::new();
        let mut meshes = Meshes::default();
        let quad = meshes.add(shapes::cuboid(Vector3::new(2.0, 2.0, 0.05)).build::<E3>());
        world.insert_resource(meshes);
        let ahead = Isometry::exp(&E3::transvection(&Vector3::new(0.0, 0.0, -4.0)));
        let entity = world
            .spawn((
                ViewRelative::<E3>(ahead),
                MeshInstances::<E3>::single(quad, Color::WHITE),
                ShowsElsewhere,
            ))
            .id();
        draw(gpu, &mut world, entity, &Lighting::default(), Some(flip))
    }

    /// The scene in `world` seen from `eye` (at the root's origin), into a small target, read
    /// back as RGBA8; with `window` set, windows onto the other eye show an image red above and
    /// blue below (turned over if it is true).
    fn draw(
        gpu: &Gpu,
        world: &mut World,
        eye: bevy_ecs::entity::Entity,
        lighting: &Lighting,
        window: Option<bool>,
    ) -> Vec<u8> {
        draw_with(gpu, world, eye, lighting, window, 0)
    }

    /// As [`draw`], the image bound as eye `slot`'s: 0 the other eye's, `k + 1` that of eye `k`
    /// of the more eyes.
    fn draw_with(
        gpu: &Gpu,
        world: &mut World,
        eye: bevy_ecs::entity::Entity,
        lighting: &Lighting,
        window: Option<bool>,
        slot: usize,
    ) -> Vec<u8> {
        let (width, height) = SIZE;
        let device = &gpu.device;
        let frame_layout = frame_layout(device);
        let frame = FrameBuffers::new(device, &frame_layout);
        let shadow_map = ShadowMap::new(gpu);
        let elsewhere = ElsewhereShared::new(gpu);
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
            elsewhere_layout: &elsewhere.layout,
        };
        let mut pipeline = RasterPipeline::<E3>::new(&setup);
        let fov = 1.0;
        let aspect = f64::from(width) / f64::from(height);
        let active = ActiveView {
            entity: eye,
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
            clipped: false,
        };
        let size = WindowSize {
            width,
            height,
            scale_factor: 1.0,
        };
        let blocks = frame_blocks(world, Some(&active), lighting, None, 1, size);
        frame.write(&gpu.queue, &blocks);
        pipeline.prepare(gpu, world, &view);
        pipeline.queue(gpu, &view);

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
                usage,
                view_formats: &[],
            })
        };
        // The other eye's image: red on the top half, blue on the bottom.
        let image = texture(
            color_format,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );
        let texels: Vec<u8> = (0..height)
            .flat_map(|y| {
                let texel = if y < height / 2 {
                    [255, 0, 0, 255]
                } else {
                    [0, 0, 255, 255]
                };
                (0..width).flat_map(move |_| texel)
            })
            .collect();
        gpu.queue.write_texture(
            image.as_image_copy(),
            &texels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            image.size(),
        );
        let settings = ElsewhereShared::settings(gpu);
        ElsewhereShared::write(gpu, &settings, window.unwrap_or(false));
        let bind_group =
            elsewhere.bind_group(gpu, &image.create_view(&Default::default()), &settings);

        let color = texture(
            color_format,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        );
        let depth = texture(depth_format, wgpu::TextureUsages::RENDER_ATTACHMENT);
        let image = window.map(|_| &bind_group);
        let bindings = FrameBindings {
            frame: &frame.bind_group,
            shadows: &shadow_map.bind_group,
            elsewhere: image.filter(|_| slot == 0),
            others: std::array::from_fn(|k| image.filter(|_| slot == k + 1)),
        };
        let mut encoder = device.create_command_encoder(&Default::default());
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
        read_buffer(gpu, &readback).unwrap()
    }

    fn pixel(image: &[u8], x: u32, y: u32) -> [u8; 4] {
        let at = ((y * SIZE.0 + x) * 4) as usize;
        image[at..at + 4].try_into().unwrap()
    }

    /// Headless (lavapipe in CI): a window shows the other eye's image where it stands on
    /// screen, the right way up or turned over, and nothing around it.
    #[test]
    fn a_window_shows_the_other_eyes_image() {
        let Some(gpu) = Gpu::headless() else {
            eprintln!("no adapter, skipped");
            return;
        };
        let (cx, cy) = (SIZE.0 / 2, SIZE.1 / 2);
        let upright = draw_window(&gpu, false);
        assert_eq!(pixel(&upright, cx, cy - 3), [255, 0, 0, 255], "red above");
        assert_eq!(pixel(&upright, cx, cy + 3), [0, 0, 255, 255], "blue below");
        assert_eq!(pixel(&upright, 0, 0)[3], 0, "nothing in the corner");
        let flipped = draw_window(&gpu, true);
        assert_eq!(pixel(&flipped, cx, cy - 3), [0, 0, 255, 255], "turned over");
        assert_eq!(pixel(&flipped, cx, cy + 3), [255, 0, 0, 255]);
    }

    /// Headless: a window onto one of the more eyes shows that eye's image, and neither the
    /// other eye's nor another of the more eyes'.
    #[test]
    fn a_window_onto_one_of_more_eyes_shows_its_image_only() {
        let Some(gpu) = Gpu::headless() else {
            eprintln!("no adapter, skipped");
            return;
        };
        let (cx, cy) = (SIZE.0 / 2, SIZE.1 / 2);
        let seen = |slot: usize| {
            let mut world = World::new();
            let mut meshes = Meshes::default();
            let quad = meshes.add(shapes::cuboid(Vector3::new(2.0, 2.0, 0.05)).build::<E3>());
            world.insert_resource(meshes);
            let ahead = Isometry::exp(&E3::transvection(&Vector3::new(0.0, 0.0, -4.0)));
            let entity = world
                .spawn((
                    ViewRelative::<E3>(ahead),
                    MeshInstances::<E3>::single(quad, Color::WHITE),
                    ShowsOther(1),
                ))
                .id();
            draw_with(
                &gpu,
                &mut world,
                entity,
                &Lighting::default(),
                Some(false),
                slot,
            )
        };
        let own = seen(2);
        assert_eq!(pixel(&own, cx, cy - 3), [255, 0, 0, 255], "its eye's image");
        assert_eq!(pixel(&own, cx, cy + 3), [0, 0, 255, 255]);
        assert_eq!(pixel(&seen(0), cx, cy)[3], 0, "not the other eye's");
        assert_eq!(pixel(&seen(1), cx, cy)[3], 0, "nor another eye's");
    }

    /// Headless: a wall facing the eye, lit from the side, ridged by a normal map along x: its
    /// pixels go light and dark across, where the same wall without the map is even.
    #[test]
    fn a_normal_map_ridges_a_flat_wall() {
        let Some(gpu) = Gpu::headless() else {
            eprintln!("no adapter, skipped");
            return;
        };
        let lighting = Lighting {
            sun: Vector3::new(1.0, 0.0, 0.4),
            ..Lighting::default()
        };
        let wall = |mapped: bool, enabled: bool| {
            let mut world = World::new();
            let mut maps = NormalMaps::new(64);
            maps.set_enabled(enabled);
            let ridges = maps.add_heights(0.15, |u, _| (u * std::f64::consts::TAU * 3.0).sin());
            let mut mesh = MeshBuilder::new();
            let corners = [(-3.0, -3.0), (3.0, -3.0), (3.0, 3.0), (-3.0, 3.0)];
            let [a, b, c, d] = corners.map(|(x, y)| {
                let uv = [(x + 3.0) / 6.0, (y + 3.0) / 6.0];
                mesh.vertex_uv(Vector3::new(x, y, 0.0), Vector3::z(), uv)
            });
            mesh.quad(a, b, c, d);
            if mapped {
                mesh.normal_map(ridges);
            }
            let mut meshes = Meshes::default();
            let wall = meshes.add(mesh.build::<E3>());
            world.insert_resource(meshes);
            world.insert_resource(maps);
            let ahead = Isometry::exp(&E3::transvection(&Vector3::new(0.0, 0.0, -4.0)));
            world.spawn((
                ViewRelative::<E3>(ahead),
                MeshInstances::<E3>::single(wall, Color::WHITE),
            ));
            let eye = world
                .spawn(GlobalPose::<E3>::new(Isometry::identity()))
                .id();
            draw(&gpu, &mut world, eye, &lighting, None)
        };
        let spread = |image: &[u8]| {
            let row: Vec<u32> = (8..SIZE.0 - 8)
                .map(|x| u32::from(pixel(image, x, SIZE.1 / 2)[0]))
                .collect();
            row.iter().max().unwrap() - row.iter().min().unwrap()
        };
        let (flat, ridged, off) = (wall(false, true), wall(true, true), wall(true, false));
        assert!(
            spread(&flat) <= 2,
            "the flat wall is even: {}",
            spread(&flat)
        );
        assert!(spread(&ridged) > 40, "the ridges show: {}", spread(&ridged));
        assert!(spread(&off) <= 2, "maps off, it is flat: {}", spread(&off));
    }

    /// Headless: a wall marked to be seen through is cut away in the parting's cone ahead, the
    /// sky showing there, and whole outside it; a wall not marked, or nearer than the cone's
    /// near distance, or with no parting, is whole.
    #[test]
    fn a_wall_seen_through_is_cut_away_in_the_parting() {
        let Some(gpu) = Gpu::headless() else {
            eprintln!("no adapter, skipped");
            return;
        };
        let cone = Part {
            direction: -Vector3::z(),
            angle: 0.2,
            ragged: 0.0,
            cell: 1.0,
            near: 1.0,
        };
        let wall = |marked: bool, parting: Parting| {
            let mut world = World::new();
            let mut meshes = Meshes::default();
            let wall = meshes.add(shapes::cuboid(Vector3::new(6.0, 6.0, 0.05)).build::<E3>());
            world.insert_resource(meshes);
            world.insert_resource(parting);
            let ahead = Isometry::exp(&E3::transvection(&Vector3::new(0.0, 0.0, -4.0)));
            let entity = world
                .spawn((
                    ViewRelative::<E3>(ahead),
                    MeshInstances::<E3>::single(wall, Color::WHITE),
                ))
                .id();
            if marked {
                world.entity_mut(entity).insert(SeeThrough);
            }
            let eye = world
                .spawn(GlobalPose::<E3>::new(Isometry::identity()))
                .id();
            draw(&gpu, &mut world, eye, &Lighting::default(), None)
        };
        let (cx, cy) = (SIZE.0 / 2, SIZE.1 / 2);
        let cut = wall(true, Parting(vec![cone]));
        assert_eq!(pixel(&cut, cx, cy)[3], 0, "the sky in the cone");
        assert_eq!(pixel(&cut, cx + 20, cy)[3], 255, "the wall outside it");
        let plain = wall(false, Parting(vec![cone]));
        assert_eq!(pixel(&plain, cx, cy)[3], 255, "a wall not marked is whole");
        let near = wall(true, Parting(vec![Part { near: 5.0, ..cone }]));
        assert_eq!(pixel(&near, cx, cy)[3], 255, "nearer than its near, whole");
        let none = wall(true, Parting::default());
        assert_eq!(none, plain, "no parting, exactly as without");
    }

    /// Headless: a point light a metre before a wall the sun is behind lights it, most where it
    /// is nearest, and nothing past its range.
    #[test]
    fn a_point_light_lights_the_wall_before_it() {
        let Some(gpu) = Gpu::headless() else {
            eprintln!("no adapter, skipped");
            return;
        };
        let lighting = Lighting {
            sun: -Vector3::z(),
            ..Lighting::default()
        };
        let shone_by = |light: Option<PointLight>, spot: Option<(Spot, fk_math::Real)>| {
            let mut world = World::new();
            let mut meshes = Meshes::default();
            let wall = meshes.add(shapes::cuboid(Vector3::new(6.0, 6.0, 0.05)).build::<E3>());
            world.insert_resource(meshes);
            let at = |z| Isometry::exp(&E3::transvection(&Vector3::new(0.0, 0.0, z)));
            world.spawn((
                ViewRelative::<E3>(at(-4.0)),
                MeshInstances::<E3>::single(wall, Color::WHITE),
            ));
            if let Some(light) = light {
                let light = world.spawn((GlobalPose::<E3>::new(at(-3.0)), light)).id();
                if let Some((spot, turn)) = spot {
                    let turned = at(-3.0).compose(&Isometry::exp(
                        &(E3::rotation(&Vector3::z(), &Vector3::x()) * turn),
                    ));
                    world
                        .entity_mut(light)
                        .insert((spot, GlobalPose::<E3>::new(turned)));
                }
            }
            let eye = world
                .spawn(GlobalPose::<E3>::new(Isometry::identity()))
                .id();
            draw(&gpu, &mut world, eye, &lighting, None)
        };
        let lit_by = |light: Option<PointLight>| shone_by(light, None);
        let brightness = |image: &[u8], x: u32, y: u32| {
            let [r, g, b, _] = pixel(image, x, y);
            u32::from(r) + u32::from(g) + u32::from(b)
        };
        let (cx, cy) = (SIZE.0 / 2, SIZE.1 / 2);
        let dark = lit_by(None);
        let light = PointLight {
            color: Color::WHITE,
            intensity: 0.8,
            range: 4.0,
        };
        let lit = lit_by(Some(light));
        assert!(
            brightness(&lit, cx, cy) > brightness(&dark, cx, cy) + 60,
            "{} against {}",
            brightness(&lit, cx, cy),
            brightness(&dark, cx, cy)
        );
        assert!(
            brightness(&lit, cx, cy) >= brightness(&lit, cx + 20, cy),
            "brightest nearest"
        );
        let short = lit_by(Some(PointLight {
            range: 0.9,
            ..light
        }));
        assert_eq!(
            pixel(&short, cx, cy),
            pixel(&dark, cx, cy),
            "out of its range"
        );
        // A spot lights the wall it shines at, and nothing turned away from it.
        let spot = Spot {
            inner: 0.3,
            outer: 0.5,
        };
        let at_it = shone_by(Some(light), Some((spot, 0.0)));
        assert_eq!(
            pixel(&at_it, cx, cy),
            pixel(&lit, cx, cy),
            "whole on its axis"
        );
        let away = shone_by(Some(light), Some((spot, std::f64::consts::PI)));
        assert_eq!(pixel(&away, cx, cy), pixel(&dark, cx, cy), "turned away");
    }
}
