//! The ray-march pipeline's compiled fields: each field's shader and its pipelines, built one
//! at a time as they are first needed, in parallel, and kept on the device for every renderer
//! after (a [`SharedDevice`](crate::SharedDevice) handed to the next app keeps them too). A
//! frame never waits for one: a cold driver takes a minute over a big field's, and the window
//! would stop answering meanwhile.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::time::Instant;

use crate::march::SdfShader;

/// Which pipeline of a field: what it draws, into what.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Variant {
    /// Full resolution, into the opaque pass.
    Full,
    /// Half resolution (or ahead), into the offscreen targets.
    Half,
    /// The offscreen targets into the opaque pass.
    Composite,
    /// `Full` and `Composite` drawn over everything ([`RayMarched::over`](crate::RayMarched)).
    FullOver,
    CompositeOver,
    /// The cones of [`RayMarched::coarse`](crate::RayMarched), into the scene's coarse target.
    Coarse,
    /// A scene marched one tile in each square of them: every pixel carried from its last
    /// frame, then the tiles marched this frame drawn over them.
    Carry,
    Tiles,
    /// At half resolution, the pixels whose own rays the composite needs: listed, the dispatch
    /// over them written, and their rays traced at full resolution.
    Classify,
    Finalize,
    Refine,
}

impl Variant {
    pub(crate) const ALL: [Self; 11] = [
        Self::Full,
        Self::Half,
        Self::Composite,
        Self::FullOver,
        Self::CompositeOver,
        Self::Coarse,
        Self::Carry,
        Self::Tiles,
        Self::Classify,
        Self::Finalize,
        Self::Refine,
    ];

    fn index(self) -> usize {
        self as usize
    }
}

/// A pipeline built.
pub(crate) enum Built {
    Render(wgpu::RenderPipeline),
    Compute(wgpu::ComputePipeline),
}

/// What builds the pipelines of a renderer's fields: the device, the layouts and the formats.
/// Layouts made alike on one device are one layout ([`wgpu::Device::create_bind_group_layout`]
/// pools them), so the pipelines a renderer builds draw for the next one on that device too.
pub(crate) struct Builder {
    pub(crate) device: wgpu::Device,
    pub(crate) full_layout: wgpu::PipelineLayout,
    pub(crate) composite_layout: wgpu::PipelineLayout,
    pub(crate) cone_layout: wgpu::PipelineLayout,
    pub(crate) listing_layout: wgpu::PipelineLayout,
    pub(crate) tracing_layout: wgpu::PipelineLayout,
    pub(crate) color_format: wgpu::TextureFormat,
    pub(crate) depth_format: wgpu::TextureFormat,
    pub(crate) sample_count: u32,
}

/// Where the half-resolution passes write.
pub(crate) const HALF_COLOR: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub(crate) const HALF_DISTANCE: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
/// Where each block's rays start, and the steps it took to get there.
pub(crate) const COARSE: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Float;

impl Builder {
    fn build(&self, label: &str, shader: &wgpu::ShaderModule, variant: Variant) -> Built {
        let render = |vertex: &str,
                      layout: &wgpu::PipelineLayout,
                      entry: &str,
                      targets: &[Option<wgpu::ColorTargetState>],
                      depth: Option<wgpu::CompareFunction>| {
            Built::Render(
                self.device
                    .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        label: Some(label),
                        layout: Some(layout),
                        vertex: wgpu::VertexState {
                            module: shader,
                            entry_point: Some(vertex),
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
                            module: shader,
                            entry_point: Some(entry),
                            compilation_options: Default::default(),
                            targets,
                        }),
                        multiview_mask: None,
                        cache: None,
                    }),
            )
        };
        let compute = |layout: &wgpu::PipelineLayout, entry: &str| {
            Built::Compute(
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(label),
                        layout: Some(layout),
                        module: shader,
                        entry_point: Some(entry),
                        compilation_options: Default::default(),
                        cache: None,
                    }),
            )
        };
        let opaque = [Some(self.color_format.into())];
        let offscreen = [Some(HALF_COLOR.into()), Some(HALF_DISTANCE.into())];
        let (tested, over) = (
            Some(wgpu::CompareFunction::Greater),
            Some(wgpu::CompareFunction::Always),
        );
        match variant {
            Variant::Full => render("vertex", &self.full_layout, "fragment", &opaque, tested),
            Variant::Half => render(
                "vertex",
                &self.full_layout,
                "fragment_half",
                &offscreen,
                None,
            ),
            Variant::Composite => render(
                "vertex",
                &self.composite_layout,
                "composite",
                &opaque,
                tested,
            ),
            Variant::FullOver => render("vertex", &self.full_layout, "fragment", &opaque, over),
            Variant::CompositeOver => {
                render("vertex", &self.composite_layout, "composite", &opaque, over)
            }
            Variant::Coarse => render(
                "vertex",
                &self.cone_layout,
                "fragment_coarse",
                &[Some(COARSE.into())],
                None,
            ),
            Variant::Carry => render(
                "vertex",
                &self.full_layout,
                "fragment_carry",
                &offscreen,
                None,
            ),
            Variant::Tiles => render(
                "vertex_tiles",
                &self.full_layout,
                "fragment_half",
                &offscreen,
                None,
            ),
            Variant::Classify => compute(&self.listing_layout, "classify"),
            Variant::Finalize => compute(&self.listing_layout, "finalize"),
            Variant::Refine => compute(&self.tracing_layout, "refine"),
        }
    }
}

/// One field compiled: its shader module, and each of its pipelines once built.
pub(crate) struct Field {
    label: String,
    shader: wgpu::ShaderModule,
    builder: Arc<Builder>,
    built: [OnceLock<Built>; Variant::ALL.len()],
    /// Which pipelines a frame asked for ([`build_soon`]).
    asked: [AtomicBool; Variant::ALL.len()],
}

impl Field {
    /// `variant`'s pipeline, built now if it is not yet (waiting for it if another thread is
    /// building it).
    pub(crate) fn get(&self, variant: Variant) -> &Built {
        self.built[variant.index()]
            .get_or_init(|| self.builder.build(&self.label, &self.shader, variant))
    }

    /// Whether `variant` is built.
    pub(crate) fn has(&self, variant: Variant) -> bool {
        self.built[variant.index()].get().is_some()
    }

    /// `variant`'s render pipeline, if it is built.
    pub(crate) fn render(&self, variant: Variant) -> Option<&wgpu::RenderPipeline> {
        match self.built[variant.index()].get() {
            Some(Built::Render(pipeline)) => Some(pipeline),
            _ => None,
        }
    }

    /// `variant`'s compute pipeline, if it is built.
    pub(crate) fn compute(&self, variant: Variant) -> Option<&wgpu::ComputePipeline> {
        match self.built[variant.index()].get() {
            Some(Built::Compute(pipeline)) => Some(pipeline),
            _ => None,
        }
    }
}

/// What a field's compiled shader depends on: the same key, the same pipelines.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct FieldKey {
    pub(crate) geometry: &'static str,
    pub(crate) shader: SdfShader,
    pub(crate) metric: Option<String>,
    pub(crate) color_format: wgpu::TextureFormat,
    pub(crate) depth_format: wgpu::TextureFormat,
    pub(crate) sample_count: u32,
}

/// A field kept: made once, by whichever renderer asks first, the others waiting for it;
/// `None` if it does not compose.
type Slot = Arc<OnceLock<Option<Arc<Field>>>>;

/// The fields compiled on a device, kept with its [`Gpu`](crate::Gpu) for every renderer on
/// it: two renderers preparing at once (the app running and the next one) make each field once.
#[derive(Clone, Default)]
pub(crate) struct KeptFields(Arc<Mutex<HashMap<FieldKey, Slot>>>);

impl std::fmt::Debug for KeptFields {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kept = self.0.lock().unwrap_or_else(PoisonError::into_inner).len();
        f.debug_struct("KeptFields").field("kept", &kept).finish()
    }
}

impl KeptFields {
    /// How many fields are kept, and how many of their pipelines are built.
    #[cfg(test)]
    pub(crate) fn counts(&self) -> (usize, usize) {
        let slots = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        let fields: Vec<_> = slots
            .values()
            .filter_map(|slot| slot.get().cloned().flatten())
            .collect();
        let built = fields
            .iter()
            .map(|field| Variant::ALL.iter().filter(|v| field.has(**v)).count())
            .sum();
        (fields.len(), built)
    }

    /// The field of every key in `keys`: those kept as they are, the others composed by
    /// `compose` and their shader modules made, in parallel, and kept. `None` for a field that
    /// does not compose.
    pub(crate) fn fields(
        &self,
        builder: &Arc<Builder>,
        keys: Vec<FieldKey>,
        compose: impl Fn(&FieldKey) -> Result<fk_shaders::naga::Module, String> + Sync,
    ) -> Vec<Option<Arc<Field>>> {
        let slots: Vec<Slot> = {
            let mut slots = self.0.lock().unwrap_or_else(PoisonError::into_inner);
            keys.iter()
                .map(|key| slots.entry(key.clone()).or_default().clone())
                .collect()
        };
        let make = |key: &FieldKey| {
            let started = Instant::now();
            let module = match compose(key) {
                Ok(module) => module,
                Err(error) => {
                    let name = &key.shader.name;
                    tracing::error!("distance field {name} does not compile, not drawn: {error}");
                    return None;
                }
            };
            let composing = started.elapsed();
            let shader = builder
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(&key.shader.name),
                    source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(module)),
                });
            tracing::info!(
                "distance field {}: composed in {composing:.0?}, its shader made in {:.0?}",
                key.shader.name,
                started.elapsed() - composing
            );
            Some(Arc::new(Field {
                label: key.shader.name.clone(),
                shader,
                builder: builder.clone(),
                built: Default::default(),
                asked: Default::default(),
            }))
        };
        std::thread::scope(|scope| {
            let threads: Vec<_> = keys
                .iter()
                .zip(&slots)
                .map(|(key, slot)| {
                    let make = &make;
                    scope.spawn(move || slot.get_or_init(|| make(key)).clone())
                })
                .collect();
            threads
                .into_iter()
                .map(|thread| thread.join().ok().flatten())
                .collect()
        })
    }
}

/// Builds every pipeline of `needs` that is not built yet, spread over the cores, and returns
/// once they all are.
pub(crate) fn build_now(needs: &[(Arc<Field>, Variant)]) {
    let todo: Vec<_> = needs
        .iter()
        .filter(|(field, variant)| !field.has(*variant))
        .cloned()
        .collect();
    if todo.is_empty() {
        return;
    }
    let started = Instant::now();
    let count = todo.len();
    run_spread(todo, threads());
    tracing::info!("{count} pipelines built in {:.0?}", started.elapsed());
}

/// Builds every pipeline of `later` that is not built yet, in order, on threads of their own,
/// without waiting: ready by the time they are asked for, or nearer to it.
pub(crate) fn build_later(later: Vec<(Arc<Field>, Variant)>) {
    let todo: Vec<_> = later
        .into_iter()
        .filter(|(field, variant)| !field.has(*variant))
        .collect();
    spawn_builds(todo, "ahead of need");
}

/// Builds every pipeline of `needs` neither built nor asked for yet, on threads of their own
/// beside those building ahead of need, without waiting: what a frame would draw with, drawn
/// once they are.
pub(crate) fn build_soon(needs: &[(Arc<Field>, Variant)]) {
    let todo: Vec<_> = needs
        .iter()
        .filter(|(field, variant)| {
            !field.has(*variant) && !field.asked[variant.index()].swap(true, Ordering::Relaxed)
        })
        .cloned()
        .collect();
    spawn_builds(todo, "as a frame asked");
}

/// Builds `todo` on a thread of its own, spread over half the cores (the game plays on
/// meanwhile), logged as built `why`.
fn spawn_builds(todo: Vec<(Arc<Field>, Variant)>, why: &'static str) {
    if todo.is_empty() {
        return;
    }
    let threads = threads().div_ceil(2);
    let spawned = std::thread::Builder::new()
        .name("fk pipelines".to_owned())
        .spawn(move || {
            let started = Instant::now();
            let count = todo.len();
            run_spread(todo, threads);
            tracing::info!("{count} pipelines built {why} in {:.0?}", started.elapsed());
        });
    if let Err(error) = spawned {
        tracing::warn!("cannot build pipelines on a thread of their own: {error}");
    }
}

/// How many threads build pipelines at once.
fn threads() -> usize {
    std::thread::available_parallelism().map_or(4, |n| n.get())
}

/// Builds `todo` on `threads` threads, each taking the next not yet taken.
fn run_spread(todo: Vec<(Arc<Field>, Variant)>, threads: usize) {
    let next = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..threads.clamp(1, todo.len()) {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some((field, variant)) = todo.get(i) else {
                        break;
                    };
                    field.get(*variant);
                }
            });
        }
    });
}
