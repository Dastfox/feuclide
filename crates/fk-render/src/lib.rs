//! The renderer: a metric-aware rasterization path for constant-curvature spaces (and, from S5,
//! a geodesic ray-marching path for arbitrary metrics) behind one abstraction.
//!
//! Generic over [`GpuGeometry`]; never names a concrete geometry. Add a [`RenderPlugin<G>`]
//! next to a [`ScenePlugin<G>`](fk_scene::ScenePlugin), spawn a [`Camera`](fk_scene::Camera),
//! add meshes to [`Meshes`] (built with [`MeshBuilder`] or the [`shapes`]) and draw them with
//! [`MeshInstances`].
//!
//! # The frame
//!
//! A thin, fixed render graph, run in the [`Render`] schedule:
//!
//! 0. **Elsewhere.** When an [`Elsewhere`] eye is set, the scene is first drawn from it (its
//!    shadows, its opaque pass) into an image of the window's size, and submitted. Surfaces
//!    marked [`ShowsElsewhere`] show that image in the frame's opaque pass: a portal's
//!    opening, a mirror.
//! 1. **Visibility.** Every [`GeometryRenderer`] extracts what it draws and culls it. The
//!    raster pipeline composes each instance camera-relative in `f64`
//!    (`view⁻¹ ∘ entity ∘ instance`) and tests its mesh's geodesic bounding ball against the
//!    frustum ([`Frustum`]).
//! 2. **Shadows.** When [`Shadows`] are on (flat space only), every renderer draws its casters
//!    into the directional light's shadow map, depth only ([`ShadowView`]).
//! 3. **Opaque.** One pass into a linear HDR colour target and a depth target, multisampled
//!    ([`Multisampling`], 4× by default) and resolved at the end: first the
//!    [`Sky`] behind everything (a fullscreen triangle that leaves depth alone), then every
//!    renderer. The image's alpha is 1 where a surface was drawn and 0 on the sky, so what
//!    only applies to surfaces (the posterize) follows the image. While a [`Mosh`] runs, it
//!    then replaces the resolved image, alpha included.
//! 4. **Post.** A fullscreen pass to the window. [`PostSettings`] selects the image or the
//!    depth buffer decoded to distance, and composes the image's effects (blur, saturation,
//!    vignette, a [`Tunnel`] closing in, [`Lids`] closing over an [`Afterimage`], [`Warp`]
//!    lenses). The engine gives them no meaning; the game decides what they are. Debug
//!    [`Gauge`]s go on top, and a [`TextPanel`] over everything, the [`Overlay`] included. While the lids show it, a [`LayerRender`] is drawn first: only the
//!    entities marked [`OnLayer`], over a flat colour, for the inside of the lids.
//! 5. **UI.** Empty: there is no text in play (the [`TextPanel`] is for debugging).
//!
//! # Shaping meshes on the GPU
//!
//! An entity's instances can go through a game-supplied vertex [`Deform`] ([`DeformShader`],
//! registered in [`Deforms`]): a WGSL module replacing the engine's identity `fk::deform`,
//! reading the world block and built from the `fk::displace` helpers. That is how a landscape
//! follows a game value, in the opaque and the shadow passes alike, with no work on the CPU.
//!
//! # Depth
//!
//! Depth is reverse-Z hardware perspective depth (1 at `near`, 0 at `far`), never written from
//! the fragment shader in the raster path so early-Z stays on. Along a pixel's ray it is a
//! strictly monotonic function of geodesic distance, and each geometry's shader module converts
//! both ways (`geo_depth_to_distance`, `geo_distance_to_depth`); that is how the ray marcher
//! composites into the same buffer, and what [`PostMode::Distance`] shows.

mod color;
mod cull;
mod deform;
mod elsewhere;
mod frame;
mod gpu;
mod graph;
mod ink;
pub mod light;
mod march;
mod mesh;
mod mosh;
mod normal_map;
pub mod offscreen;
mod overlay;
pub mod perf;
mod post;
pub mod quotient;
mod raster;
mod renderer;
mod screenshot;
mod seen;
mod shadow;
pub mod shapes;
mod sky;
mod timing;

use std::marker::PhantomData;

use fk_app::{App, Plugin, Render, Startup, Update};
use fk_geometry::GpuGeometry;

pub use color::Color;
pub use cull::Frustum;
pub use deform::{CastsNoShadow, Deform, DeformHandle, DeformShader, Deforms};
pub use elsewhere::{Elsewhere, ShowsElsewhere};
pub use frame::{
    Afterimage, DiscShape, Gauge, Ink, LayerRender, Lids, Lighting, Mark, Mosh, MoshKind,
    MoshPatch, Multisampling, PostMode, PostSettings, RenderStats, Shadows, Sky, SkyDisc,
    TextPanel, Tunnel, Vsync, Warp, WorldUniforms,
};
pub use gpu::{Gpu, Vec2f, Vec4f, matrix_columns};
pub use graph::SharedDevice;
pub use light::{MAX_POINT_LIGHTS, PointLight};
pub use march::{
    RayMarchPipeline, RayMarched, RayMarching, SDF_PARAMS, SdfHandle, SdfShader, Sdfs,
};
pub use mesh::{Instance, Mesh, MeshBuilder, MeshHandle, MeshInstances, Meshes, Vertex, tangent};
pub use mosh::{CarriedImage, HeldImage};
pub use normal_map::{NormalMap, NormalMaps, heights_to_normals};
pub use offscreen::{Image, Offscreen, Paths};
pub use overlay::Overlay;
pub use quotient::{QuotientView, Unrepeated};
pub use raster::RasterPipeline;
pub use renderer::{
    FrameBindings, FrameView, GeometryRenderer, Hidden, OnLayer, RendererFactory, RendererSetup,
};
pub use screenshot::Screenshots;
pub use seen::DiscsSeen;
pub use shadow::ShadowView;
pub use timing::GpuTimings;

/// Draws the scene of geometry `G` into the window, with the [`RasterPipeline`] and then the
/// [`RayMarchPipeline`].
///
/// Inserts [`Meshes`], [`Deforms`], [`Sdfs`], [`RayMarching`], [`Lighting`], [`Shadows`],
/// [`WorldUniforms`], [`PostSettings`], [`RenderStats`], [`Vsync`], [`Screenshots`], [`CarriedImage`], [`Elsewhere`] and [`Overlay`]. The device ([`Gpu`]) is opened in [`Startup`] once the window exists;
/// without a window nothing is drawn. Once it draws, it also writes [`DiscsSeen`].
///
/// For look development, setting `FK_CAPTURE` to a file path saves one screenshot there after
/// `FK_CAPTURE_AFTER` seconds (3 by default) and exits. For performance tests, setting
/// `FK_PERF` to a file path records the frames' times there and exits ([`perf`]).
///
/// Frames wait for the display's refresh unless [`Vsync`] says otherwise; setting `FK_NO_VSYNC`
/// presents them as fast as they are drawn whatever it says, to measure frame rates.
pub struct RenderPlugin<G>(PhantomData<fn() -> G>);

impl<G> Default for RenderPlugin<G> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<G: GpuGeometry> Plugin for RenderPlugin<G> {
    fn build(&self, app: &mut App) {
        app.set_geometry::<G>()
            .init_resource::<Meshes>()
            .init_resource::<NormalMaps>()
            .init_resource::<Deforms>()
            .init_resource::<Sdfs>()
            .init_resource::<RayMarching>()
            .init_resource::<Lighting>()
            .init_resource::<Shadows>()
            .init_resource::<WorldUniforms>()
            .init_resource::<PostSettings>()
            .init_resource::<RenderStats>()
            .init_resource::<Multisampling>()
            .init_resource::<Vsync>()
            .init_resource::<Screenshots>()
            .init_resource::<CarriedImage>()
            .init_resource::<Elsewhere<G>>()
            .init_resource::<Overlay>()
            .add_systems(Startup, graph::start::<G>)
            .add_systems(Update, (screenshot::capture_once, perf::record))
            .add_systems(Render, graph::render::<G>);
        if let Some(capture) = screenshot::CaptureOnce::from_env() {
            app.insert_resource(capture);
        }
        if let Some(recording) = perf::PerfRecording::from_env() {
            app.insert_resource(recording);
        }
        add_renderer::<G>(app, |setup| Box::new(RasterPipeline::<G>::new(setup)));
        add_renderer::<G>(app, |setup| Box::new(RayMarchPipeline::<G>::new(setup)));
    }
}

/// Adds a renderer to the opaque pass of the app's [`RenderPlugin<G>`], after the ones already
/// there. `factory` builds it once the device is open.
///
/// Call it while building the app, before it runs.
pub fn add_renderer<G: GpuGeometry>(
    app: &mut App,
    factory: impl Fn(&RendererSetup<'_>) -> Box<dyn GeometryRenderer<G>> + Send + Sync + 'static,
) {
    app.world_mut()
        .get_resource_or_insert_with(graph::RendererFactories::<G>::default)
        .0
        .push(Box::new(factory));
}
