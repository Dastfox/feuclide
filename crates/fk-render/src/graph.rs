//! The render graph: surface, targets and the fixed pass order
//! (the other eye's image, when there is one) → visibility → opaque → layer (while it is
//! enabled) → the caller's image effects settled with the image → settle → the other image
//! effects → late (when anything is) → post → UI.

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use bevy_ecs::prelude::*;
use fk_app::{Time, WindowHandle, WindowSize};
use fk_geometry::{GpuGeometry, GroupElement};
use fk_math::Real;
use fk_math::nalgebra::{Matrix4, Rotation3, Unit, Vector3};
use fk_scene::ActiveView;

use crate::carried::CarriedImage;
use crate::effect::{
    EffectFactories, EffectFrame, EffectSetup, EffectTargets, GradeShader, ImageEffect, SkyFigure,
};
use crate::elsewhere::{
    Elsewhere, ElsewhereShared, ElsewhereTargets, MAX_OTHER_EYES, OtherEye, OtherEyes,
};
use crate::frame::{
    DiscBlock, LightingBlock, MAX_SKY_DISCS, PointLightBlock, SkyDiscs, ViewBlock, WorldBlock,
    uniform_bytes,
};
use crate::gpu::{Vec2f, Vec4f, matrix_columns};
use crate::late::DepthResolve;
use crate::light::{MAX_POINT_LIGHTS, PointLight, Spot};
use crate::mesh::tangent;
use crate::overlay::{Overlay, OverlayPass};
use crate::post::PostPass;
use crate::renderer::{FrameBindings, FrameView, GeometryRenderer, RendererFactory, RendererSetup};
use crate::screenshot::{self, Screenshots};
use crate::seen::{DiscProbe, SeenProbe};
use crate::settle::SettlePass;
use crate::shadow::{SHADOW_FORMAT, ShadowMap, ShadowView};
use crate::sky::SkyPass;
use crate::timing::GpuTimer;
use crate::{Color, DiscShape, SkyDisc};
use crate::{
    FrameCapture, Frustum, Gpu, GradeValues, LayerRender, Lighting, Multisampling, PostSettings,
    RenderStats, Shadows, Vsync, WorldUniforms,
};

const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Renderers waiting for the device, in the order they draw.
#[derive(Resource)]
pub(crate) struct RendererFactories<G>(pub Vec<RendererFactory<G>>);

impl<G> Default for RendererFactories<G> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

/// The device a renderer draws with and the window's surface, as a resource the renderer
/// inserts when it opens, for the next app's renderer across a
/// [`Handover`](fk_app::Handover). Put into that app's world before it starts, its renderer
/// opens on this device rather than a new one, without the window: it compiles its pipelines
/// and uploads its meshes there while the old app still draws, and draws from the first frame
/// after the old renderer is dropped, which gives up the surface to it.
#[derive(Resource, Clone)]
pub struct SharedDevice {
    gpu: Gpu,
    adapter: wgpu::Adapter,
    config: wgpu::SurfaceConfiguration,
    /// The surface, while no renderer holds it.
    surface: Arc<Mutex<Option<wgpu::Surface<'static>>>>,
}

impl std::fmt::Debug for SharedDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedDevice")
            .field("adapter", &self.gpu.adapter.name)
            .field("format", &self.config.format)
            .finish_non_exhaustive()
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

struct Targets {
    /// The resolved image the image effects and the post pass read.
    color: wgpu::Texture,
    color_view: wgpu::TextureView,
    /// What the opaque pass draws into when multisampling, resolved into `color`.
    multisampled: Option<wgpu::TextureView>,
    depth: wgpu::TextureView,
    /// The layer render, read by the post pass.
    layer: LayerTargets,
    /// The other eye's image, read by the windows onto it.
    elsewhere: ElsewhereTargets,
    /// The images of the [`OtherEyes`], each made when its eye is first set.
    others: [Option<ElsewhereTargets>; MAX_OTHER_EYES],
    /// The late pass's depth when the opaque pass is multisampled: its depth resolved to one
    /// sample, and what reads it.
    late: Option<LateTargets>,
}

/// The depth the late pass tests against when the opaque pass is multisampled.
struct LateTargets {
    depth: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
}

/// The layer render's own colour and depth, the opaque pass's formats and sample count.
struct LayerTargets {
    color: wgpu::TextureView,
    multisampled: Option<wgpu::TextureView>,
    depth: wgpu::TextureView,
}

pub(crate) struct FrameBuffers {
    view: wgpu::Buffer,
    world: wgpu::Buffer,
    lighting: wgpu::Buffer,
    pub(crate) bind_group: wgpu::BindGroup,
}

#[derive(Resource)]
pub(crate) struct RenderState<G: GpuGeometry> {
    gpu: Gpu,
    shared: SharedDevice,
    /// The window's surface; `None` until the renderer before this one gives it up.
    surface: Option<wgpu::Surface<'static>>,
    config: wgpu::SurfaceConfiguration,
    reconfigure: bool,
    frame: FrameBuffers,
    targets: Option<Targets>,
    sample_count: u32,
    renderers: Vec<Box<dyn GeometryRenderer<G>>>,
    sky: SkyPass,
    shadow_map: ShadowMap,
    settle: SettlePass,
    /// The caller's image effects, in the order they run.
    effects: Vec<Box<dyn ImageEffect>>,
    /// Resolves the depth for the late pass, when the opaque pass is multisampled.
    late: Option<DepthResolve>,
    post: PostPass,
    overlay: OverlayPass,
    elsewhere: ElsewhereShared,
    discs: DiscProbe,
    /// Times the passes for a performance recording.
    timer: GpuTimer,
    /// The eye of the last frame drawn, for the settling's and the effects' motion.
    last_eye: Option<G::Isometry>,
}

/// What the caller adds to the renderer, read when the device opens.
struct Additions<'a, G> {
    renderers: &'a [RendererFactory<G>],
    effects: &'a EffectFactories,
    grade: Option<&'a GradeShader>,
    figure: Option<&'a SkyFigure>,
}

/// Opens the device on the window, or takes the [`SharedDevice`] when there is one. Without
/// either (a headless `App::update`) there is nothing to draw into, and nothing happens until
/// there is a window.
pub(crate) fn start<G: GpuGeometry>(world: &mut World) {
    let shared = world.get_resource::<SharedDevice>().cloned();
    let window = world.get_resource::<WindowHandle>().cloned();
    if shared.is_none() && window.is_none() {
        tracing::debug!("no window, the renderer stays off");
        return;
    }
    let samples = world
        .get_resource::<Multisampling>()
        .copied()
        .unwrap_or_default()
        .samples;
    let factories = world
        .remove_resource::<RendererFactories<G>>()
        .unwrap_or_default();
    let effects = world
        .remove_resource::<EffectFactories>()
        .unwrap_or_default();
    if let Some(progress) = world.get_resource::<fk_app::Progress>() {
        progress.step("compiling the shaders");
    }
    let additions = Additions {
        renderers: &factories.0,
        effects: &effects,
        grade: world.get_resource::<GradeShader>(),
        figure: world.get_resource::<SkyFigure>(),
    };
    let state = match (shared, window) {
        (Some(shared), _) => Ok(RenderState::<G>::on(
            shared,
            world.get_resource::<WindowSize>().copied(),
            samples,
            &additions,
        )),
        (None, Some(window)) => {
            let size = *world.resource::<WindowSize>();
            RenderState::<G>::open(window, size, samples, &additions)
        }
        (None, None) => return,
    };
    match state {
        Ok(state) => {
            tracing::info!(
                "rendering {} on {} ({:?}), output {:?}, {}× multisampling",
                G::NAME,
                state.gpu.adapter.name,
                state.gpu.adapter.backend,
                state.config.format,
                state.sample_count,
            );
            world.insert_resource(state.gpu.clone());
            world.insert_resource(state.shared.clone());
            world.insert_resource(state);
        }
        Err(error) => tracing::error!("cannot render: {error}"),
    }
}

/// Draws one frame, opening the device first if it could not be opened at the start.
pub(crate) fn render<G: GpuGeometry>(world: &mut World) {
    if !world.contains_resource::<RenderState<G>>()
        && world.contains_resource::<RendererFactories<G>>()
    {
        start::<G>(world);
    }
    let Some(mut state) = world.remove_resource::<RenderState<G>>() else {
        return;
    };
    state.frame(world);
    world.insert_resource(state);
}

impl<G: GpuGeometry> Drop for RenderState<G> {
    /// Gives up the surface, for a renderer on the same [`SharedDevice`].
    fn drop(&mut self) {
        if let Some(surface) = self.surface.take() {
            *lock(&self.shared.surface) = Some(surface);
        }
    }
}

impl<G: GpuGeometry> RenderState<G> {
    /// Opens a device of its own on the window.
    fn open(
        window: WindowHandle,
        size: WindowSize,
        samples: u32,
        additions: &Additions<'_, G>,
    ) -> Result<Self, String> {
        let mut descriptor =
            wgpu::InstanceDescriptor::new_with_display_handle_from_env(Box::new(window.clone()));
        // Shaders built with debug information only when asked for (`WGPU_DEBUG=1`): the driver
        // leaves them unoptimised, and over a large distance field a frame then takes seconds
        // and the window stops answering.
        if std::env::var("WGPU_DEBUG").as_deref() != Ok("1") {
            descriptor.flags.remove(wgpu::InstanceFlags::DEBUG);
        }
        let instance = wgpu::Instance::new(descriptor);
        let surface = instance
            .create_surface(window)
            .map_err(|error| format!("no surface: {error}"))?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
            apply_limit_buckets: false,
        }))
        .map_err(|error| format!("no adapter: {error}"))?;
        // Timestamps only for a performance recording, and only where the adapter has them.
        let timed = if crate::perf::enabled() {
            adapter.features() & GpuTimer::FEATURES
        } else {
            wgpu::Features::empty()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("fk"),
            required_features: timed,
            ..Default::default()
        }))
        .map_err(|error| format!("no device: {error}"))?;
        let gpu = Gpu {
            device,
            queue,
            adapter: adapter.get_info(),
            kept: Default::default(),
        };
        let capabilities = surface.get_capabilities(&adapter);
        let format = capabilities
            .formats
            .iter()
            .copied()
            .find(wgpu::TextureFormat::is_srgb)
            .or_else(|| capabilities.formats.first().copied())
            .ok_or("the surface supports no format")?;
        // Copying out of the surface is what screenshots need; not every platform allows it.
        let copy_out = capabilities.usages & wgpu::TextureUsages::COPY_SRC;
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | copy_out,
            format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: Vsync::default().present_mode(),
            desired_maximum_frame_latency: 2,
            alpha_mode: capabilities.alpha_modes[0],
            view_formats: Vec::new(),
        };
        let shared = SharedDevice {
            gpu,
            adapter,
            config,
            surface: Arc::default(),
        };
        Ok(Self::build(shared, Some(surface), size, samples, additions))
    }

    /// Opens on a shared device, without the surface until it is given up.
    fn on(
        shared: SharedDevice,
        size: Option<WindowSize>,
        samples: u32,
        additions: &Additions<'_, G>,
    ) -> Self {
        let size = size.unwrap_or(WindowSize {
            width: shared.config.width,
            height: shared.config.height,
            scale_factor: 1.0,
        });
        Self::build(shared, None, size, samples, additions)
    }

    fn build(
        shared: SharedDevice,
        surface: Option<wgpu::Surface<'static>>,
        size: WindowSize,
        samples: u32,
        additions: &Additions<'_, G>,
    ) -> Self {
        let gpu = shared.gpu.clone();
        let supported = |format| {
            shared
                .adapter
                .get_texture_format_features(format)
                .flags
                .sample_count_supported(samples)
        };
        let sample_count = if samples <= 1 || (supported(COLOR_FORMAT) && supported(DEPTH_FORMAT)) {
            samples.max(1)
        } else {
            tracing::warn!("{samples}× multisampling is not supported here, drawing without");
            1
        };
        let config = wgpu::SurfaceConfiguration {
            width: size.width.max(1),
            height: size.height.max(1),
            ..shared.config.clone()
        };
        let format = config.format;

        let frame_layout = frame_layout(&gpu.device);
        let frame = FrameBuffers::new(&gpu.device, &frame_layout);
        let shadow_map = ShadowMap::new(&gpu);
        let elsewhere = ElsewhereShared::new(&gpu);
        let setup = RendererSetup {
            gpu: &gpu,
            frame_layout: &frame_layout,
            shadow_layout: &shadow_map.layout,
            color_format: COLOR_FORMAT,
            depth_format: DEPTH_FORMAT,
            shadow_format: SHADOW_FORMAT,
            sample_count,
            elsewhere_layout: &elsewhere.layout,
        };
        let renderers = additions
            .renderers
            .iter()
            .map(|factory| factory(&setup))
            .collect();
        let sky = SkyPass::new(
            &gpu,
            G::SHADER,
            &frame_layout,
            (COLOR_FORMAT, DEPTH_FORMAT, sample_count),
            additions.figure,
        );
        let settle = SettlePass::new(
            &gpu,
            G::SHADER,
            &frame_layout,
            COLOR_FORMAT,
            sample_count > 1,
        );
        let effect_setup = EffectSetup {
            gpu: &gpu,
            geometry: G::SHADER,
            frame_layout: &frame_layout,
            color_format: COLOR_FORMAT,
            multisampled_depth: sample_count > 1,
        };
        let effects = additions
            .effects
            .0
            .iter()
            .map(|factory| factory(&effect_setup))
            .collect();
        let post = PostPass::new(
            &gpu,
            G::SHADER,
            &frame_layout,
            format,
            sample_count > 1,
            (additions.grade, additions.figure),
        );
        let late = (sample_count > 1).then(|| DepthResolve::new(&gpu.device, DEPTH_FORMAT));
        let overlay = OverlayPass::new(&gpu, format);
        let discs = DiscProbe::new(&gpu);
        let timer = GpuTimer::new(&gpu, crate::perf::enabled());
        Self {
            gpu,
            shared,
            surface,
            config,
            reconfigure: true,
            frame,
            targets: None,
            sample_count,
            renderers,
            sky,
            shadow_map,
            settle,
            effects,
            late,
            post,
            overlay,
            elsewhere,
            discs,
            timer,
            last_eye: None,
        }
    }

    /// An other eye's image, `width` × `height`, in the frame's formats and sample count.
    fn elsewhere_targets(&self, width: u32, height: u32) -> ElsewhereTargets {
        let target = |label, format, sample_count, usage| {
            self.gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | usage,
                view_formats: &[],
            })
        };
        let samples = self.sample_count;
        let color = target(
            "fk elsewhere",
            COLOR_FORMAT,
            1,
            wgpu::TextureUsages::TEXTURE_BINDING,
        )
        .create_view(&Default::default());
        let settings = ElsewhereShared::settings(&self.gpu);
        ElsewhereTargets {
            bind_group: self.elsewhere.bind_group(&self.gpu, &color, &settings),
            color,
            multisampled: (samples > 1).then(|| {
                target(
                    "fk elsewhere samples",
                    COLOR_FORMAT,
                    samples,
                    wgpu::TextureUsages::empty(),
                )
                .create_view(&Default::default())
            }),
            depth: target(
                "fk elsewhere depth",
                DEPTH_FORMAT,
                samples,
                wgpu::TextureUsages::empty(),
            )
            .create_view(&Default::default()),
            settings,
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width;
        self.config.height = height;
        if let Some(surface) = &self.surface {
            surface.configure(&self.gpu.device, &self.config);
        }
        let target = |label, format, sample_count, usage| {
            self.gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | usage,
                view_formats: &[],
            })
        };
        let samples = self.sample_count;
        // The colour target is read by the post pass, copied into the captured frame on
        // request, and read and overwritten by the settling and the image effects.
        let color = target(
            "fk colour",
            COLOR_FORMAT,
            1,
            wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
        );
        let multisampled = (samples > 1).then(|| {
            target(
                "fk colour samples",
                COLOR_FORMAT,
                samples,
                wgpu::TextureUsages::empty(),
            )
            .create_view(&Default::default())
        });
        let depth = target(
            "fk depth",
            DEPTH_FORMAT,
            samples,
            wgpu::TextureUsages::TEXTURE_BINDING,
        )
        .create_view(&Default::default());
        let layer = LayerTargets {
            color: target(
                "fk layer",
                COLOR_FORMAT,
                1,
                wgpu::TextureUsages::TEXTURE_BINDING,
            )
            .create_view(&Default::default()),
            multisampled: (samples > 1).then(|| {
                target(
                    "fk layer samples",
                    COLOR_FORMAT,
                    samples,
                    wgpu::TextureUsages::empty(),
                )
                .create_view(&Default::default())
            }),
            depth: target(
                "fk layer depth",
                DEPTH_FORMAT,
                samples,
                wgpu::TextureUsages::empty(),
            )
            .create_view(&Default::default()),
        };
        let elsewhere = self.elsewhere_targets(width, height);
        self.settle.set_targets(&self.gpu, &color, &depth);
        let effect_targets = EffectTargets {
            color: &color,
            depth: &depth,
        };
        for effect in &mut self.effects {
            effect.set_targets(&self.gpu, &effect_targets);
        }
        self.post
            .set_targets(&self.gpu, &color, &depth, &layer.color);
        let late = self.late.as_ref().map(|resolve| LateTargets {
            depth: target(
                "fk late depth",
                DEPTH_FORMAT,
                1,
                wgpu::TextureUsages::empty(),
            )
            .create_view(&Default::default()),
            bind_group: resolve.bind(&self.gpu.device, &depth),
        });
        let targets = Targets {
            color_view: color.create_view(&Default::default()),
            color,
            multisampled,
            depth,
            layer,
            elsewhere,
            others: Default::default(),
            late,
        };
        self.targets = Some(targets);
        self.reconfigure = false;
    }

    fn frame(&mut self, world: &mut World) {
        if self.surface.is_none() {
            self.surface = lock(&self.shared.surface).take();
            self.reconfigure = true;
        }
        if self.surface.is_none() {
            // Nothing to draw to yet: the renderers get ready.
            for renderer in &mut self.renderers {
                renderer.warm(&self.gpu, world);
            }
            return;
        }
        let size = *world.resource::<WindowSize>();
        if size.width == 0 || size.height == 0 {
            return;
        }
        let present_mode = world
            .get_resource::<Vsync>()
            .copied()
            .unwrap_or_default()
            .present_mode();
        if present_mode != self.config.present_mode {
            self.config.present_mode = present_mode;
            if let (Some(surface), false) = (&self.surface, self.reconfigure) {
                surface.configure(&self.gpu.device, &self.config);
            }
        }
        if self.reconfigure || (self.config.width, self.config.height) != (size.width, size.height)
        {
            self.resize(size.width, size.height);
        }
        if let Some(timings) = self.timer.collect(&self.gpu) {
            world.insert_resource(timings);
        }
        self.timer.begin();
        let aspect = Real::from(size.width) / Real::from(size.height);
        let view = world.get_resource::<ActiveView<G>>().cloned();
        let lighting = world
            .get_resource::<Lighting>()
            .cloned()
            .unwrap_or_default();
        let shadows = world.get_resource::<Shadows>().copied().unwrap_or_default();
        // The other eye's image first, submitted on its own: the frame's buffers are written
        // again for the frame's own eye after it.
        let other = world
            .get_resource::<Elsewhere<G>>()
            .cloned()
            .unwrap_or_default();
        let elsewhere_drawn = match (&view, other.eye) {
            (Some(view), Some(eye)) => {
                let seen_by = other.lighting.as_ref().unwrap_or(&lighting);
                let eye = OtherEye {
                    eye,
                    flip: other.flip,
                    lighting: None,
                    clip: other.clip,
                    window: other.window,
                };
                self.draw_elsewhere(world, view, &eye, None, seen_by, &shadows, size)
            }
            _ => false,
        };
        // Then the more eyes, each on its own.
        let more = world
            .get_resource::<OtherEyes<G>>()
            .cloned()
            .unwrap_or_default();
        let mut others_drawn = [false; MAX_OTHER_EYES];
        if let Some(view) = &view {
            for (k, other) in more.0.iter().enumerate() {
                let Some(other) = other else { continue };
                let seen_by = other.lighting.as_ref().unwrap_or(&lighting);
                others_drawn[k] =
                    self.draw_elsewhere(world, view, other, Some(k), seen_by, &shadows, size);
            }
        }
        let shadow = view
            .as_ref()
            .filter(|_| shadows.enabled)
            .and_then(|view| ShadowView::new::<G>(&view.eye.isometry, &lighting.sun, &shadows));
        if shadow.is_some() {
            self.shadow_map.resize(&self.gpu, shadows.size);
        }
        let post = world
            .get_resource::<PostSettings>()
            .cloned()
            .unwrap_or_default();
        // How the eye moved since the last frame: its coordinates now to its coordinates then.
        // Picked up by the settling and the image effects.
        let motion = match (&view, &self.last_eye) {
            (Some(view), Some(last)) => {
                G::isometry_matrix(&last.inverse().compose(&view.eye.isometry))
            }
            _ => Matrix4::identity(),
        };
        self.last_eye = view.as_ref().map(|view| view.eye.isometry);
        // The point of their pixels this frame's rays and vertices go through.
        let jitter = self.settle.prepare(&self.gpu, &post.settle, &motion);
        let mut blocks = frame_blocks(
            world,
            view.as_ref(),
            &lighting,
            shadow.as_ref(),
            self.shadow_map.size,
            size,
        );
        blocks.view.jitter = Vec2f(jitter);
        self.frame.write(&self.gpu.queue, &blocks);
        self.post.prepare(
            &self.gpu,
            &post,
            &world
                .get_resource::<GradeValues>()
                .cloned()
                .unwrap_or_default(),
        );
        self.overlay.prepare(
            &self.gpu,
            world.get_resource::<Overlay>(),
            [size.width, size.height],
        );
        let effect_frame = EffectFrame {
            motion,
            size: [size.width, size.height],
            delta: world.get_resource::<Time>().map_or(0.0, Time::delta) as f32,
        };
        for effect in &mut self.effects {
            effect.prepare(&self.gpu, world, &effect_frame);
        }
        let layer_render = world
            .get_resource::<LayerRender>()
            .copied()
            .unwrap_or_default();
        let capture = world
            .get_resource::<FrameCapture>()
            .copied()
            .unwrap_or_default();

        // Visibility.
        let mut stats = RenderStats::default();
        if let Some(view) = &view {
            let frame_view = FrameView {
                view,
                aspect,
                size: [size.width, size.height],
                frustum: Frustum::new(view.camera.fov_y, aspect, view.camera.far),
                shadow,
                elsewhere: None,
                clipped: false,
            };
            for renderer in &mut self.renderers {
                renderer.prepare(&self.gpu, world, &frame_view);
            }
            for renderer in &mut self.renderers {
                renderer.queue(&self.gpu, &frame_view);
                let drawn = renderer.stats();
                stats.drawn += drawn.drawn;
                stats.culled += drawn.culled;
                stats.draws += drawn.draws;
            }
        }
        world.insert_resource(stats);
        world.insert_resource(self.discs.collect(&self.gpu));

        let Some(surface) = &self.surface else { return };
        let output = match surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture) => texture,
            wgpu::CurrentSurfaceTexture::Suboptimal(texture) => {
                self.reconfigure = true;
                texture
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return;
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.reconfigure = true;
                return;
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                tracing::warn!("surface texture failed validation");
                return;
            }
        };
        let Some(targets) = &self.targets else { return };
        let output_view = output.texture.create_view(&Default::default());
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("fk") });

        self.timer.start(&mut encoder);
        if view.is_some() {
            self.shadow_and_offscreen(&mut encoder, shadow.is_some());
        }
        self.timer.mark(&mut encoder, "shadows");

        // Opaque. Alpha 0 is the sky: nothing drawn there.
        let clear = wgpu::Color {
            a: 0.0,
            ..lighting.sky.horizon.into()
        };
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("fk opaque"),
                color_attachments: &[Some(color_attachment(
                    targets.multisampled.as_ref(),
                    &targets.color_view,
                    clear,
                ))],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &targets.depth,
                    // Reverse-Z: 0 is the far surface.
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            if view.is_some() {
                self.sky.render(&mut pass, &self.frame.bind_group);
                let bindings = FrameBindings {
                    frame: &self.frame.bind_group,
                    shadows: &self.shadow_map.bind_group,
                    elsewhere: elsewhere_drawn.then_some(&targets.elsewhere.bind_group),
                    others: others_bindings(targets, &others_drawn),
                };
                for renderer in &self.renderers {
                    renderer.render(&mut pass, &bindings);
                }
            }
        }
        self.timer.mark(&mut encoder, "opaque");
        // How much of the sun, the moon and the probe the surfaces hide, before anything is
        // laid over.
        if let Some(view) = &view {
            let lighting = &blocks.lighting;
            let probe = world.get_resource::<SeenProbe>().and_then(|probe| {
                let direction = in_eye_frame(view, &probe.direction?.try_normalize(1e-12)?);
                let [x, y, z] = [direction.x, direction.y, direction.z].map(|c| c as f32);
                Some(Vec4f([x, y, z, probe.radius.max(0.0) as f32]))
            });
            self.discs.copy(
                &mut encoder,
                &targets.color,
                blocks.view.proj_scale.0,
                [
                    Some(&lighting.sun_disc.direction),
                    Some(&lighting.moon_disc.direction),
                    probe.as_ref(),
                ],
            );
        }

        // The layer, only while it is enabled: what is marked on it, over a flat colour.
        if view.is_some() && layer_render.enabled {
            let layer = &targets.layer;
            let clear = wgpu::Color {
                a: 1.0,
                ..layer_render.clear.into()
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("fk layer"),
                color_attachments: &[Some(color_attachment(
                    layer.multisampled.as_ref(),
                    &layer.color,
                    clear,
                ))],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &layer.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            let bindings = FrameBindings {
                frame: &self.frame.bind_group,
                shadows: &self.shadow_map.bind_group,
                elsewhere: None,
                others: [None; MAX_OTHER_EYES],
            };
            for renderer in &self.renderers {
                renderer.render_layer(&mut pass, &bindings);
            }
        }
        self.timer.mark(&mut encoder, "layer");

        // The caller's image effects, in order: those settled with the image first, then the
        // settling over frames, then the rest, which work on the settled image.
        let effect_targets = EffectTargets {
            color: &targets.color,
            depth: &targets.depth,
        };
        for settled in [true, false] {
            if !settled {
                self.settle
                    .render(&mut encoder, &targets.color, &self.frame.bind_group);
                self.timer.mark(&mut encoder, "settle");
            }
            for effect in &mut self.effects {
                if effect.settled() == settled {
                    effect.render(&mut encoder, &effect_targets, &self.frame.bind_group);
                    self.timer.mark(&mut encoder, effect.label());
                }
            }
        }

        // Late: what is drawn over the image as the effects left it, against the depth as the
        // opaque pass left it, one sample.
        if view.is_some() && self.renderers.iter().any(|renderer| renderer.has_late()) {
            // All they draw late may lie in one rectangle: nothing outside it is touched.
            let scissor = self
                .renderers
                .iter()
                .filter(|renderer| renderer.has_late())
                .map(|renderer| renderer.late_scissor())
                .try_fold(None::<[u32; 4]>, |union, rect| {
                    let [x, y, w, h] = rect?;
                    Some(Some(match union {
                        None => [x, y, w, h],
                        Some([ux, uy, uw, uh]) => {
                            let (x0, y0) = (ux.min(x), uy.min(y));
                            let (x1, y1) = ((ux + uw).max(x + w), (uy + uh).max(y + h));
                            [x0, y0, x1 - x0, y1 - y0]
                        }
                    }))
                })
                .flatten();
            let depth = match (&self.late, &targets.late) {
                (Some(resolve), Some(late)) => {
                    resolve.resolve(&mut encoder, &late.bind_group, &late.depth, scissor);
                    &late.depth
                }
                _ => &targets.depth,
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("fk late"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &targets.color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            let bindings = FrameBindings {
                frame: &self.frame.bind_group,
                shadows: &self.shadow_map.bind_group,
                elsewhere: elsewhere_drawn.then_some(&targets.elsewhere.bind_group),
                others: others_bindings(targets, &others_drawn),
            };
            for renderer in &self.renderers {
                renderer.render_late(&mut pass, &bindings);
            }
        }
        self.timer.mark(&mut encoder, "late");

        // Post.
        self.post.capture(&mut encoder, &targets.color, capture);
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("fk post"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &output_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            self.post.render(&mut pass, &self.frame.bind_group);
        }
        self.timer.mark(&mut encoder, "post");

        // What the CPU drew over everything: a menu.
        self.overlay.render(&mut encoder, &output_view);
        // The text panel, a readout, over that too.
        self.post
            .render_text(&mut encoder, &output_view, &self.frame.bind_group);
        self.timer.mark(&mut encoder, "overlay");
        self.timer.finish(&mut encoder);

        self.gpu.queue.submit([encoder.finish()]);
        self.timer.submitted();
        self.discs.map();
        if let Some(mut carried) = world.get_resource_mut::<CarriedImage>()
            && carried.take_request()
        {
            let color = &targets.color;
            let bytes = COLOR_FORMAT.block_copy_size(None).unwrap_or(8);
            if let Some(pixels) = screenshot::read_back(&self.gpu, color, bytes, "carried image") {
                carried.set_read(color.width(), color.height(), pixels);
            }
        }
        let screenshots = world
            .get_resource_mut::<Screenshots>()
            .map(|mut screenshots| screenshots.take())
            .unwrap_or_default();
        if !screenshots.is_empty() {
            if self.config.usage.contains(wgpu::TextureUsages::COPY_SRC) {
                screenshot::save(&self.gpu, &output.texture, &screenshots);
            } else {
                tracing::warn!("this surface cannot be copied from, no screenshot");
            }
        }
        self.gpu.queue.present(output);
    }
}

impl<G: GpuGeometry> RenderState<G> {
    /// Records the shadow pass, when `shadows`, and the renderers' offscreen passes, for the
    /// view their last `queue` was given.
    fn shadow_and_offscreen(&self, encoder: &mut wgpu::CommandEncoder, shadows: bool) {
        // Shadows: depth from the light, 1 where nothing casts.
        if shadows {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("fk shadow"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow_map.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            for renderer in &self.renderers {
                renderer.render_shadows(&mut pass, &self.frame.bind_group);
            }
        }
        // Offscreen passes of the renderers.
        let bindings = FrameBindings {
            frame: &self.frame.bind_group,
            shadows: &self.shadow_map.bind_group,
            elsewhere: None,
            others: [None; MAX_OTHER_EYES],
        };
        for renderer in &self.renderers {
            renderer.encode(encoder, &bindings);
        }
    }

    /// Draws the scene seen from `other` (cut off at its clip, shown upside down if it is
    /// flipped, only what is seen through its window) into its image and submits it: the
    /// [`Elsewhere`] eye's for `which` `None`, else eye `which` of the [`OtherEyes`] (its image
    /// made the first time). The frame's camera and shadows from there under `lighting` (its
    /// own or the frame's), its poses made relative to that eye. Whether it was drawn.
    #[allow(clippy::too_many_arguments)]
    fn draw_elsewhere(
        &mut self,
        world: &mut World,
        view: &ActiveView<G>,
        other: &OtherEye<G>,
        which: Option<usize>,
        lighting: &Lighting,
        shadows: &Shadows,
        size: WindowSize,
    ) -> bool {
        let Some(targets) = &self.targets else {
            return false;
        };
        let OtherEye {
            eye,
            flip,
            clip,
            window,
            ..
        } = *other;
        let aspect = Real::from(size.width) / Real::from(size.height);
        // The part of its image the window covers, upside down when it is shown so.
        let part = window
            .and_then(|corners| seen_part(view, aspect, &corners))
            .map(|[x0, y0, x1, y1]| {
                if flip {
                    [x0, -y1, x1, -y0]
                } else {
                    [x0, y0, x1, y1]
                }
            });
        if part.is_some_and(|[x0, y0, x1, y1]| x0 >= x1 || y0 >= y1) {
            return false;
        }
        if let Some(k) = which
            && targets.others[k].is_none()
        {
            let made = self.elsewhere_targets(self.config.width, self.config.height);
            if let Some(targets) = &mut self.targets {
                targets.others[k] = Some(made);
            }
        }
        let mut seen = view.clone();
        seen.eye.isometry = eye;
        let shift = eye.inverse().compose(&view.eye.isometry);
        let shadow = shadows
            .enabled
            .then(|| ShadowView::new::<G>(&eye, &lighting.sun, shadows))
            .flatten();
        if shadow.is_some() {
            self.shadow_map.resize(&self.gpu, shadows.size);
        }
        let mut blocks = frame_blocks(
            world,
            Some(&seen),
            lighting,
            shadow.as_ref(),
            self.shadow_map.size,
            size,
        );
        if let Some(clip) = &clip {
            blocks.view.clip = clip_form::<G>(&eye, clip);
        }
        self.frame.write(&self.gpu.queue, &blocks);
        let (fov_y, far) = (seen.camera.fov_y, seen.camera.far);
        let frame_view = FrameView {
            view: &seen,
            aspect,
            size: [size.width, size.height],
            frustum: match part {
                Some(part) => Frustum::part(fov_y, aspect, far, part),
                None => Frustum::new(fov_y, aspect, far),
            },
            shadow,
            elsewhere: Some(shift),
            clipped: clip.is_some(),
        };
        for renderer in &mut self.renderers {
            renderer.prepare(&self.gpu, world, &frame_view);
        }
        for renderer in &mut self.renderers {
            renderer.queue(&self.gpu, &frame_view);
        }
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fk elsewhere"),
            });
        self.timer.start(&mut encoder);
        self.shadow_and_offscreen(&mut encoder, shadow.is_some());
        let Some(targets) = &self.targets else {
            return false;
        };
        let target = match which {
            None => &targets.elsewhere,
            Some(k) => match &targets.others[k] {
                Some(target) => target,
                None => return false,
            },
        };
        ElsewhereShared::write(&self.gpu, &target.settings, flip);
        let clear = wgpu::Color {
            a: 0.0,
            ..lighting.sky.horizon.into()
        };
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("fk elsewhere"),
                color_attachments: &[Some(color_attachment(
                    target.multisampled.as_ref(),
                    &target.color,
                    clear,
                ))],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &target.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            if let Some(part) = part {
                let [x, y, width, height] = scissor(part, [size.width, size.height]);
                pass.set_scissor_rect(x, y, width, height);
            }
            self.sky.render(&mut pass, &self.frame.bind_group);
            let bindings = FrameBindings {
                frame: &self.frame.bind_group,
                shadows: &self.shadow_map.bind_group,
                elsewhere: None,
                others: [None; MAX_OTHER_EYES],
            };
            for renderer in &self.renderers {
                renderer.render(&mut pass, &bindings);
            }
        }
        self.timer.mark(&mut encoder, "elsewhere");
        self.gpu.queue.submit([encoder.finish()]);
        true
    }
}

/// The rectangle `[x0, y0, x1, y1]` of the frame's image (normalized device coordinates, `y`
/// up, clamped to it) that the window with these corners (global points, round it) covers,
/// seen from the frame's eye, cut off at the eye's plane: empty when none of it is in front.
/// `None` outside flat space, for the whole image.
fn seen_part<G: GpuGeometry>(
    view: &ActiveView<G>,
    aspect: Real,
    corners: &[G::Point; 4],
) -> Option<[Real; 4]> {
    // Nearer the eye's plane than this, a window's edge is cut off.
    const NEAR: Real = 1e-4;
    if G::constant_curvature() != Some(0.0) {
        return None;
    }
    let origin = G::origin();
    let to_eye = view.eye.isometry.inverse();
    let mut round = Vec::with_capacity(4);
    for corner in corners {
        let v = G::log(&origin, &G::apply(&to_eye, corner))?;
        round.push(Vector3::from_fn(|i, _| {
            G::inner(&origin, &v, &G::origin_frame(i))
        }));
    }
    // The polygon's part in front of the eye (the eye looks along −z), its edges straight.
    let mut ahead = Vec::with_capacity(8);
    for (k, a) in round.iter().enumerate() {
        let b = round[(k + 1) % round.len()];
        let (in_a, in_b) = (a.z <= -NEAR, b.z <= -NEAR);
        if in_a {
            ahead.push(*a);
        }
        if in_a != in_b {
            let t = (-NEAR - a.z) / (b.z - a.z);
            ahead.push(a + (b - a) * t);
        }
    }
    let tan_y = (view.camera.fov_y / 2.0).tan();
    let tan_x = tan_y * aspect;
    let mut part = [1.0, 1.0, -1.0, -1.0];
    for (k, d) in ahead.iter().enumerate() {
        let (x, y) = (d.x / -d.z / tan_x, d.y / -d.z / tan_y);
        part = if k == 0 {
            [x, y, x, y]
        } else {
            [
                part[0].min(x),
                part[1].min(y),
                part[2].max(x),
                part[3].max(y),
            ]
        };
    }
    Some(part.map(|t| t.clamp(-1.0, 1.0)))
}

/// The pixels `[x, y, width, height]` of an image `size` wide and high that the rectangle
/// `part` of it covers (normalized device coordinates), a little more for the filtering.
fn scissor([x0, y0, x1, y1]: [Real; 4], [width, height]: [u32; 2]) -> [u32; 4] {
    const MARGIN: Real = 2.0;
    let (w, h) = (Real::from(width), Real::from(height));
    let pixel = |t: Real, across: Real| t.clamp(0.0, across) as u32;
    let left = pixel(((x0 + 1.0) / 2.0 * w - MARGIN).floor(), w);
    let right = pixel(((x1 + 1.0) / 2.0 * w + MARGIN).ceil(), w);
    let top = pixel(((1.0 - y1) / 2.0 * h - MARGIN).floor(), h);
    let bottom = pixel(((1.0 - y0) / 2.0 * h + MARGIN).ceil(), h);
    [left, top, right - left, bottom - top]
}

/// The bind groups of the [`OtherEyes`]' images drawn this frame (`drawn`), for the windows onto
/// them.
fn others_bindings<'a>(
    targets: &'a Targets,
    drawn: &[bool; MAX_OTHER_EYES],
) -> [Option<&'a wgpu::BindGroup>; MAX_OTHER_EYES] {
    std::array::from_fn(|k| {
        targets.others[k]
            .as_ref()
            .filter(|_| drawn[k])
            .map(|target| &target.bind_group)
    })
}

/// A colour attachment cleared to `clear`: drawn into `samples` and resolved into `resolved`
/// when multisampling (the samples are only needed until they are resolved), straight into
/// `resolved` otherwise.
fn color_attachment<'a>(
    samples: Option<&'a wgpu::TextureView>,
    resolved: &'a wgpu::TextureView,
    clear: wgpu::Color,
) -> wgpu::RenderPassColorAttachment<'a> {
    match samples {
        Some(samples) => wgpu::RenderPassColorAttachment {
            view: samples,
            depth_slice: None,
            resolve_target: Some(resolved),
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(clear),
                store: wgpu::StoreOp::Discard,
            },
        },
        None => wgpu::RenderPassColorAttachment {
            view: resolved,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(clear),
                store: wgpu::StoreOp::Store,
            },
        },
    }
}

/// The plane `z = 0` of `clip` as a linear form on points embedded in the frame of `eye`,
/// positive on the side its `+z` points to: the form reading the frame's own `z`, carried by
/// `eye⁻¹ ∘ clip`'s matrix `A` as `A⁻ᵀ` (a point `p` of the eye's frame is `A⁻¹ p` in the
/// clip's).
fn clip_form<G: GpuGeometry>(eye: &G::Isometry, clip: &G::Isometry) -> Vec4f {
    let z = G::embed_tangent(&G::origin(), &G::origin_frame(2));
    let to_clip = G::isometry_matrix(&clip.inverse().compose(eye));
    let form = to_clip.transpose() * z;
    Vec4f(form.map(|c| c as f32).into())
}

/// The three blocks of bind group 0.
pub(crate) struct FrameBlocks {
    view: ViewBlock,
    world: WorldBlock,
    lighting: LightingBlock,
}

/// The blocks of bind group 0 for a frame drawn from `view` in a target of `size`, with the
/// shadow map `shadow` (`shadow_texels` across) when shadows are drawn.
pub(crate) fn frame_blocks<G: GpuGeometry>(
    world: &World,
    view: Option<&ActiveView<G>>,
    lighting: &Lighting,
    shadow: Option<&ShadowView>,
    shadow_texels: u32,
    size: WindowSize,
) -> FrameBlocks {
    let (width, height) = (size.width as f32, size.height as f32);
    let (fov_y, near, far) = view.map_or((1.0, 0.1, 100.0), |view| {
        (view.camera.fov_y, view.camera.near, view.camera.far)
    });
    let scale_y = 1.0 / (fov_y / 2.0).tan();
    let scale_x = scale_y * Real::from(height / width);
    let view_block = ViewBlock {
        proj_scale: Vec2f([scale_x as f32, scale_y as f32]),
        near: near as f32,
        far: far as f32,
        size: Vec2f([width, height]),
        jitter: Vec2f([0.0; 2]),
        clip: Vec4f([0.0; 4]),
    };

    let origin = G::origin();
    // Directions live at the root's origin; carry them to the eye.
    let carry = |direction: &Vector3<Real>| {
        let direction = tangent::<G>(&direction.normalize());
        match view {
            Some(view) => {
                let inverse = view.eye.isometry.inverse();
                let at = G::apply(&inverse, &origin);
                let moved = G::apply_tangent(&inverse, &direction);
                G::parallel_transport(&at, &origin, &moved)
            }
            None => direction,
        }
    };
    // A direction at the eye as coordinates in the eye's frame, with `w` in the last slot.
    let in_eye = |direction: &Vector3<Real>, w: Real| {
        let at_eye = carry(direction);
        let c = |i| G::inner(&origin, &at_eye, &G::origin_frame(i)) as f32;
        Vec4f([c(0), c(1), c(2), w as f32])
    };
    let color = |c: Color| Vec4f(c.to_array());
    let disc = |disc: &SkyDisc| {
        let (kind, figure) = match disc.shape {
            DiscShape::Disc => (0.0, [0.0; 4]),
            DiscShape::Ring => (1.0, [0.0; 4]),
            DiscShape::Figure { params } => (2.0, params),
        };
        DiscBlock {
            direction: in_eye(&disc.direction, disc.radius.max(0.0)),
            shape: Vec4f([kind, 0.0, disc.stroke.max(0.0) as f32, 0.0]),
            color: color(disc.color),
            glow: Vec4f([disc.glow.r, disc.glow.g, disc.glow.b, disc.glow_size as f32]),
            figure: Vec4f(figure),
        }
    };
    let sky = &lighting.sky;
    // The stars' frame: the reference axes turned with the sky, carried to the eye, so the
    // star grid stays put in the world as the eye turns.
    let star_turn = match Unit::try_new(sky.star_axis, 1e-9) {
        Some(axis) => Rotation3::from_axis_angle(&axis, sky.star_angle),
        None => Rotation3::identity(),
    };
    let star_frame =
        [Vector3::x(), Vector3::y(), Vector3::z()].map(|axis| in_eye(&(star_turn * axis), 0.0));
    let (point_lights, point_count) = point_lights::<G>(world, view);
    let extra = world
        .get_resource::<SkyDiscs>()
        .map_or(&[][..], |discs| &discs.0[..]);
    let lighting_block = LightingBlock {
        sun_direction: Vec4f::from_real(&G::embed_tangent(&origin, &carry(&lighting.sun))),
        sun_color: color(lighting.sun_color),
        shadow_tint: color(lighting.shadow_tint),
        sky_zenith: color(sky.zenith),
        sky_horizon: color(sky.horizon),
        sky_below: color(sky.below),
        sky_up: in_eye(&sky.up, sky.gradient),
        star_frame,
        sun_disc: disc(&sky.sun),
        moon_disc: disc(&sky.moon),
        fog_density: lighting.fog_density as f32,
        bands: lighting.bands.max(1) as f32,
        stars: sky.stars.max(0.0) as f32,
        star_density: sky.star_density.max(1.0) as f32,
        smoothness: lighting.smooth.clamp(0.0, 1.0) as f32,
        shadow_matrix: shadow.map_or([Vec4f::default(); 4], |shadow| {
            matrix_columns(&shadow.matrix).map(Vec4f)
        }),
        shadow: match shadow {
            Some(shadow) => {
                let texels = Real::from(shadow_texels);
                let texel = 2.0 * shadow.range / texels;
                let shadows = world.get_resource::<Shadows>().copied().unwrap_or_default();
                Vec4f([
                    1.0,
                    (1.0 / texels) as f32,
                    shadows.softness.max(0.0) as f32,
                    (shadows.normal_offset.max(0.0) * texel) as f32,
                ])
            }
            None => Vec4f::default(),
        },
        shadow_fade: Vec4f([
            world
                .get_resource::<Shadows>()
                .map_or(0.8, |shadows| shadows.fade.clamp(0.0, 0.999)) as f32,
            0.0,
            0.0,
            0.0,
        ]),
        point_lights,
        point_count: Vec4f([point_count as f32, 0.0, 0.0, 0.0]),
        discs: std::array::from_fn(|i| disc(extra.get(i).unwrap_or(&SkyDisc::default()))),
        disc_count: Vec4f([extra.len().min(MAX_SKY_DISCS) as f32, 0.0, 0.0, 0.0]),
    };

    let values = world
        .get_resource::<WorldUniforms>()
        .cloned()
        .unwrap_or_default()
        .values;
    let world_block = WorldBlock {
        time: world.get_resource::<Time>().map_or(0.0, Time::elapsed) as f32,
        values: values.map(Vec4f),
    };

    FrameBlocks {
        view: view_block,
        world: world_block,
        lighting: lighting_block,
    }
}

/// A direction at the root's origin as coordinates in the eye's frame, as the sky's discs are
/// carried there.
fn in_eye_frame<G: GpuGeometry>(view: &ActiveView<G>, direction: &Vector3<Real>) -> Vector3<Real> {
    let origin = G::origin();
    let direction = tangent::<G>(&direction.normalize());
    let inverse = view.eye.isometry.inverse();
    let at = G::apply(&inverse, &origin);
    let moved = G::apply_tangent(&inverse, &direction);
    let at_eye = G::parallel_transport(&at, &origin, &moved);
    Vector3::from_fn(|i, _| G::inner(&origin, &at_eye, &G::origin_frame(i)))
}

/// The point lights that reach nearest the eye, made relative to it in f64 (`eye⁻¹ ∘ light`),
/// and how many there are.
fn point_lights<G: GpuGeometry>(
    world: &World,
    view: Option<&ActiveView<G>>,
) -> ([PointLightBlock; MAX_POINT_LIGHTS], usize) {
    let mut blocks = [PointLightBlock::default(); MAX_POINT_LIGHTS];
    let (Some(view), Some(mut query)) = (
        view,
        world.try_query::<(Entity, &fk_scene::GlobalPose<G>, &PointLight)>(),
    ) else {
        return (blocks, 0);
    };
    let origin = G::origin();
    let to_eye = view.eye.isometry.inverse();
    let mut lights: Vec<_> = query
        .iter(world)
        .filter(|(_, _, light)| light.range > 0.0 && light.intensity > 0.0)
        .map(|(entity, pose, light)| {
            let relative = to_eye.compose(&pose.isometry);
            let at = G::apply(&relative, &origin);
            // Looked up apart: a query asking for a component no entity ever had finds nothing.
            let spot = world.get::<Spot>(entity).map(|spot| {
                let axis = G::apply_tangent(&relative, &(-G::origin_frame(2)));
                (G::embed_tangent(&at, &axis), *spot)
            });
            // How far short of the eye its light stops: the nearest reach first.
            (G::distance(&origin, &at) - light.range, at, *light, spot)
        })
        .collect();
    lights.sort_by(|a, b| a.0.total_cmp(&b.0));
    let count = lights.len().min(MAX_POINT_LIGHTS);
    for (block, (_, at, light, spot)) in blocks.iter_mut().zip(lights) {
        let (cones, direction) = match spot {
            Some((axis, spot)) => (
                [spot.inner.cos() as f32, spot.outer.cos() as f32],
                Vec4f::from_real(&axis),
            ),
            None => ([0.0, 0.0], Vec4f::default()),
        };
        *block = PointLightBlock {
            position: Vec4f::from_real(&G::embed_point(&at)),
            color: Vec4f(light.color.to_array()),
            shape: Vec4f([
                light.intensity as f32,
                light.range as f32,
                cones[0],
                cones[1],
            ]),
            direction,
        };
    }
    (blocks, count)
}

pub(crate) fn frame_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    // Compute too: the ray marcher traces the pixels at its edges in a compute pass.
    let uniform = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT | wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("fk frame"),
        entries: &[uniform(0), uniform(1), uniform(2)],
    })
}

impl FrameBuffers {
    pub(crate) fn write(&self, queue: &wgpu::Queue, blocks: &FrameBlocks) {
        queue.write_buffer(&self.view, 0, &uniform_bytes(&blocks.view));
        queue.write_buffer(&self.world, 0, &uniform_bytes(&blocks.world));
        queue.write_buffer(&self.lighting, 0, &uniform_bytes(&blocks.lighting));
    }

    pub(crate) fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> Self {
        let buffer = |label, size: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let view = buffer("fk view", size_of_block::<ViewBlock>());
        let world = buffer("fk world", size_of_block::<WorldBlock>());
        let lighting = buffer("fk lighting", size_of_block::<LightingBlock>());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fk frame"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: view.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: world.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: lighting.as_entire_binding(),
                },
            ],
        });
        Self {
            view,
            world,
            lighting,
            bind_group,
        }
    }
}

fn size_of_block<T: encase::ShaderType>() -> usize {
    T::min_size().get() as usize
}
