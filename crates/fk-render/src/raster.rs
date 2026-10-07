//! The raster pipeline for constant-curvature geometries: instanced meshes, painted material.

use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::ops::Range;

use bevy_ecs::query::{Has, QueryState, Without};
use bevy_ecs::world::World;
use bytemuck::{Pod, Zeroable};
use fk_geometry::{Geometry, GpuGeometry, GroupElement};
use fk_scene::ViewRelative;
use wgpu::util::DeviceExt;

use crate::deform::{CastsNoShadow, Deform, Deforms};
use crate::elsewhere::ShowsElsewhere;
use crate::gpu::matrix_columns;
use crate::mesh::{MeshInstances, Meshes, Vertex};
use crate::normal_map::NormalMaps;
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
    /// A window onto the other eye's view.
    window: bool,
}

/// Instances of one mesh drawn through one slot's pipelines.
struct Draw {
    slot: u32,
    mesh: u32,
    instances: Range<u32>,
}

/// The two pipelines of one deform: the painted surfaces and the shadow casters.
struct Pipelines {
    opaque: wgpu::RenderPipeline,
    shadow: wgpu::RenderPipeline,
}

/// The normal maps as the shader samples them: a texture array, the flat map first.
struct GpuNormalMaps {
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    /// How many maps it holds, besides the flat one, and how big they are.
    uploaded: (usize, u32),
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
                ],
            });
        let bind_group = Self::upload(gpu, &layout, &NormalMaps::new(1));
        Self {
            layout,
            bind_group,
            uploaded: (0, 1),
        }
    }

    /// Uploads `maps` again if any were added since.
    fn update(&mut self, gpu: &Gpu, maps: &NormalMaps) {
        let now = (maps.len(), maps.size());
        if now != self.uploaded {
            self.bind_group = Self::upload(gpu, &self.layout, maps);
            self.uploaded = now;
        }
    }

    fn upload(gpu: &Gpu, layout: &wgpu::BindGroupLayout, maps: &NormalMaps) -> wgpu::BindGroup {
        let layers = maps.layers();
        let size = maps.size();
        let texels: Vec<u8> = layers.iter().flatten().flatten().copied().collect();
        let texture = gpu.device.create_texture_with_data(
            &gpu.queue,
            &wgpu::TextureDescriptor {
                label: Some("fk normal maps"),
                size: wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: layers.len() as u32,
                },
                mip_level_count: 1,
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
    Has<Unrepeated>,
);

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
/// Entities marked [`ShowsElsewhere`] are drawn after the rest, by a pipeline of their own,
/// as windows onto the [`Elsewhere`](crate::Elsewhere) view.
pub struct RasterPipeline<G: GpuGeometry> {
    builder: Builder,
    normal_maps: GpuNormalMaps,
    /// Draws the windows onto the other eye's view: the engine's vertex stage, no deform.
    window: wgpu::RenderPipeline,
    /// By slot; `None` for a deform that did not compile, drawn as built instead.
    pipelines: Vec<Option<Pipelines>>,
    meshes: Vec<GpuMesh>,
    query: Option<QueryState<Drawables<G>, Without<Hidden>>>,
    candidates: Vec<Candidate<G>>,
    bins: BTreeMap<(u32, u32), Vec<GpuInstance>>,
    shadow_bins: BTreeMap<(u32, u32), Vec<GpuInstance>>,
    layer_bins: BTreeMap<(u32, u32), Vec<GpuInstance>>,
    window_bins: BTreeMap<(u32, u32), Vec<GpuInstance>>,
    instances: Option<wgpu::Buffer>,
    draws: Vec<Draw>,
    shadow_draws: Vec<Draw>,
    layer_draws: Vec<Draw>,
    window_draws: Vec<Draw>,
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
            candidates: Vec::new(),
            bins: BTreeMap::new(),
            shadow_bins: BTreeMap::new(),
            layer_bins: BTreeMap::new(),
            window_bins: BTreeMap::new(),
            instances: None,
            draws: Vec::new(),
            shadow_draws: Vec::new(),
            layer_draws: Vec::new(),
            window_draws: Vec::new(),
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

    /// The draws of `bins`, appended to `all`.
    fn collect(
        bins: &mut BTreeMap<(u32, u32), Vec<GpuInstance>>,
        all: &mut Vec<GpuInstance>,
        draws: &mut Vec<Draw>,
    ) {
        draws.clear();
        for (&(slot, mesh), bin) in bins.iter_mut() {
            if bin.is_empty() {
                continue;
            }
            let start = all.len() as u32;
            all.append(bin);
            draws.push(Draw {
                slot,
                mesh,
                instances: start..all.len() as u32,
            });
        }
    }

    fn draw(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        draws: &[Draw],
        bind: impl Fn(&mut wgpu::RenderPass<'_>, &Pipelines),
    ) {
        let Some(instances) = &self.instances else {
            return;
        };
        let mut slot = None;
        for draw in draws {
            if slot != Some(draw.slot) {
                let Some(Some(pipelines)) = self.pipelines.get(draw.slot as usize) else {
                    continue;
                };
                bind(pass, pipelines);
                pass.set_vertex_buffer(1, instances.slice(..));
                slot = Some(draw.slot);
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
        let vertex_attributes = wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4, 8 => Float32x4, 9 => Float32x4];
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

    fn build(&self, label: &str, module: fk_shaders::naga::Module) -> Pipelines {
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(module)),
            });
        let vertex_attributes = wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4, 8 => Float32x4, 9 => Float32x4];
        let instance_attributes = wgpu::vertex_attr_array![
            2 => Float32x4, 3 => Float32x4, 4 => Float32x4, 5 => Float32x4,
            6 => Float32x4, 7 => Float32x4,
        ];
        let buffers = vertex_layouts(&vertex_attributes, &instance_attributes);
        let opaque = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&self.opaque_layout),
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
                    entry_point: Some("fragment"),
                    compilation_options: Default::default(),
                    targets: &[Some(self.color_format.into())],
                }),
                multiview_mask: None,
                cache: None,
            });
        let shadow = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&self.shadow_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("shadow_vertex"),
                    compilation_options: Default::default(),
                    buffers: &buffers,
                },
                // The sides turned away from the light cast: the lit sides never shadow
                // themselves, and the dark sides are painted in the shadow colour anyway.
                primitive: wgpu::PrimitiveState {
                    cull_mode: Some(wgpu::Face::Front),
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
            });
        Pipelines { opaque, shadow }
    }
}

impl<G: GpuGeometry> GeometryRenderer<G> for RasterPipeline<G> {
    fn warm(&mut self, gpu: &Gpu, world: &mut World) {
        if let Some(meshes) = world.get_resource::<Meshes>() {
            self.upload_new_meshes(gpu, meshes);
        }
        if let Some(deforms) = world.get_resource::<Deforms>() {
            self.compile_new_deforms(deforms);
        }
        if let Some(maps) = world.get_resource::<NormalMaps>() {
            self.normal_maps.update(gpu, maps);
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
        let query = self.query.get_or_insert_with(|| world.query_filtered());
        self.candidates.clear();
        for (relative, drawable, deform, casts_none, layer, window, unrepeated) in query.iter(world)
        {
            let relative = view.relative(&relative.0);
            let slot = deform.map_or(0, |deform| deform.0.0 + 1);
            let slot = match self.pipelines.get(slot as usize) {
                Some(Some(_)) => slot,
                _ => 0,
            };
            let copies = if unrepeated || window {
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
                        casts: !casts_none && !window,
                        layer,
                        window,
                    });
                }
            }
        }
    }

    fn queue(&mut self, gpu: &Gpu, view: &FrameView<'_, G>) {
        let origin = G::origin();
        let mut stats = RenderStats::default();
        // On a sphere every instance is seen twice, the second time the long way round.
        let antipode = G::constant_curvature()
            .filter(|&k| k > 0.0)
            .map(|k| std::f64::consts::PI / k.sqrt());
        for candidate in &self.candidates {
            // No window onto the other eye's view in its own image.
            if candidate.window && view.elsewhere.is_some() {
                continue;
            }
            let Some(mesh) = self.meshes.get(candidate.mesh as usize) else {
                continue;
            };
            let centre = G::apply(&candidate.relative, &origin);
            let seen = view.frustum.sees::<G>(&centre, mesh.radius);
            let casts = candidate.casts
                && view
                    .shadow
                    .is_some_and(|shadow| shadow.sees::<G>(&centre, mesh.radius));
            if !seen {
                stats.culled += 1;
            }
            // Seen the long way round, it stands where its antipode would, `π − d` away the
            // other way: drawn there again, marked by its colour's alpha (the shader's far side).
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
            let instance = GpuInstance {
                iso: matrix_columns(&G::isometry_matrix(&candidate.relative)),
                color: candidate.color,
                data: candidate.data,
            };
            let key = (candidate.slot, candidate.mesh);
            if candidate.window {
                if seen {
                    self.window_bins
                        .entry((0, candidate.mesh))
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
                far.color[3] = 2.0;
                self.bins.entry(key).or_default().push(far);
                if candidate.layer {
                    self.layer_bins.entry(key).or_default().push(far);
                }
            }
            if casts {
                self.shadow_bins.entry(key).or_default().push(instance);
            }
        }
        let mut all = Vec::with_capacity(self.candidates.len());
        Self::collect(&mut self.bins, &mut all, &mut self.draws);
        stats.drawn = all.len();
        stats.draws = self.draws.len();
        Self::collect(&mut self.shadow_bins, &mut all, &mut self.shadow_draws);
        Self::collect(&mut self.layer_bins, &mut all, &mut self.layer_draws);
        Self::collect(&mut self.window_bins, &mut all, &mut self.window_draws);
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
        self.draw(pass, &self.shadow_draws, |pass, pipelines| {
            pass.set_pipeline(&pipelines.shadow);
            pass.set_bind_group(0, frame, &[]);
        });
    }

    fn render(&self, pass: &mut wgpu::RenderPass<'_>, bindings: &FrameBindings<'_>) {
        self.draw(pass, &self.draws, |pass, pipelines| {
            pass.set_pipeline(&pipelines.opaque);
            pass.set_bind_group(0, bindings.frame, &[]);
            pass.set_bind_group(1, bindings.shadows, &[]);
            pass.set_bind_group(2, &self.normal_maps.bind_group, &[]);
        });
        if let Some(elsewhere) = bindings.elsewhere {
            self.draw(pass, &self.window_draws, |pass, _| {
                pass.set_pipeline(&self.window);
                pass.set_bind_group(0, bindings.frame, &[]);
                pass.set_bind_group(1, bindings.shadows, &[]);
                pass.set_bind_group(2, elsewhere, &[]);
            });
        }
    }

    fn render_layer(&self, pass: &mut wgpu::RenderPass<'_>, bindings: &FrameBindings<'_>) {
        self.draw(pass, &self.layer_draws, |pass, pipelines| {
            pass.set_pipeline(&pipelines.opaque);
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
    use crate::shadow::{SHADOW_FORMAT, ShadowMap};
    use crate::{Color, Frustum, Lighting, MeshBuilder, NormalMaps, PointLight, shapes};

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
    /// back as RGBA8; with `window` set, windows show an image red above and blue below (turned
    /// over if it is true).
    fn draw(
        gpu: &Gpu,
        world: &mut World,
        eye: bevy_ecs::entity::Entity,
        lighting: &Lighting,
        window: Option<bool>,
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
        elsewhere.write(gpu, window.unwrap_or(false));
        let bind_group = elsewhere.bind_group(gpu, &image.create_view(&Default::default()));

        let color = texture(
            color_format,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        );
        let depth = texture(depth_format, wgpu::TextureUsages::RENDER_ATTACHMENT);
        let bindings = FrameBindings {
            frame: &frame.bind_group,
            shadows: &shadow_map.bind_group,
            elsewhere: window.map(|_| &bind_group),
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
        let wall = |mapped: bool| {
            let mut world = World::new();
            let mut maps = NormalMaps::new(64);
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
        let (flat, ridged) = (wall(false), wall(true));
        assert!(
            spread(&flat) <= 2,
            "the flat wall is even: {}",
            spread(&flat)
        );
        assert!(spread(&ridged) > 40, "the ridges show: {}", spread(&ridged));
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
        let lit_by = |light: Option<PointLight>| {
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
                world.spawn((GlobalPose::<E3>::new(at(-3.0)), light));
            }
            let eye = world
                .spawn(GlobalPose::<E3>::new(Isometry::identity()))
                .id();
            draw(&gpu, &mut world, eye, &lighting, None)
        };
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
    }
}
