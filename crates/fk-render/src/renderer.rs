//! The extension point of the render graph: one [`GeometryRenderer`] per way of drawing.

use bevy_ecs::component::Component;
use bevy_ecs::world::World;
use fk_geometry::{GpuGeometry, GroupElement};
use fk_math::Real;
use fk_scene::ActiveView;

use crate::{Frustum, Gpu, ShadowView};

/// Keeps an entity out of every renderer: its meshes and its ray-marched scene are not drawn,
/// nor cast shadows, until the marker is removed.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Hidden;

/// Draws an entity into the [`LayerRender`](crate::LayerRender) as well as into the image: its
/// meshes and its ray-marched scene, alone over the layer's flat colour.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OnLayer;

/// What a renderer is built against: the device, the shared bindings and the targets of the
/// opaque pass.
pub struct RendererSetup<'a> {
    /// The device.
    pub gpu: &'a Gpu,
    /// Layout of bind group 0 (`fk::view`: view, world, lighting), set by every pipeline.
    pub frame_layout: &'a wgpu::BindGroupLayout,
    /// Layout of bind group 1 for pipelines that receive shadows (`fk::shadow`: the shadow map
    /// and its comparison sampler). The shadow pass itself only has group 0.
    pub shadow_layout: &'a wgpu::BindGroupLayout,
    /// Format of the opaque pass's colour target, linear HDR.
    pub color_format: wgpu::TextureFormat,
    /// Format of the depth target. Depth is reverse-Z (1 near, 0 far, compare `Greater`) and
    /// encodes geodesic distance along each pixel's ray, see `geo_depth_to_distance`.
    pub depth_format: wgpu::TextureFormat,
    /// Format of the shadow map: depth from the light, 0 at its near side (compare
    /// `LessEqual`), one sample.
    pub shadow_format: wgpu::TextureFormat,
    /// Samples per pixel of both targets ([`Multisampling`](crate::Multisampling)): every
    /// pipeline drawing in the opaque pass uses this count.
    pub sample_count: u32,
    /// Layout of bind group 2 for pipelines drawing windows onto the
    /// [`Elsewhere`](crate::Elsewhere) view: its image, a sampler and its settings.
    pub elsewhere_layout: &'a wgpu::BindGroupLayout,
}

/// The view a frame is drawn from.
pub struct FrameView<'a, G: GpuGeometry> {
    /// The active camera.
    pub view: &'a ActiveView<G>,
    /// Width over height of the target.
    pub aspect: Real,
    /// The target's width and height, in pixels.
    pub size: [u32; 2],
    /// The camera's frustum, for culling.
    pub frustum: Frustum,
    /// The shadow map this frame, when shadows are drawn: which casters to draw into it.
    pub shadow: Option<ShadowView>,
    /// Set when this is the image of the [`Elsewhere`](crate::Elsewhere) eye, drawn before the
    /// frame: that eye relative to the active camera's, inverted (`elsewhere⁻¹ ∘ eye`).
    /// Poses relative to the active camera ([`ViewRelative`](fk_scene::ViewRelative)) are
    /// composed with it first, and windows onto it ([`ShowsElsewhere`](crate::ShowsElsewhere))
    /// are not drawn. [`view`](Self::view) is then the other eye's.
    pub elsewhere: Option<G::Isometry>,
}

impl<G: GpuGeometry> FrameView<'_, G> {
    /// A pose relative to the active camera's eye, made relative to this view's eye.
    pub fn relative(&self, to_camera: &G::Isometry) -> G::Isometry {
        match &self.elsewhere {
            Some(shift) => shift.compose(to_camera),
            None => *to_camera,
        }
    }
}

/// The bind groups a renderer sets in the opaque pass.
pub struct FrameBindings<'a> {
    /// Group 0, `fk::view`.
    pub frame: &'a wgpu::BindGroup,
    /// Group 1 for pipelines that receive shadows, `fk::shadow`. Bound even when shadows are
    /// off; the lighting block says so.
    pub shadows: &'a wgpu::BindGroup,
    /// Group 2 for pipelines drawing windows onto the [`Elsewhere`](crate::Elsewhere) view,
    /// when it was drawn this frame; `None` otherwise, and then windows are not drawn.
    pub elsewhere: Option<&'a wgpu::BindGroup>,
}

/// One way of drawing a geometry into the opaque pass: the raster pipeline, later the ray
/// marcher.
///
/// When an [`Elsewhere`](crate::Elsewhere) eye is set, the same calls run first for its
/// image, with [`FrameView::elsewhere`] set, and are submitted before the frame's own.
///
/// Every frame the render graph runs, in order: [`prepare`](Self::prepare) for every renderer
/// (extract from the world and upload), [`queue`](Self::queue) for every renderer (visibility),
/// [`render_shadows`](Self::render_shadows) for every renderer inside the shadow pass when
/// [`FrameView::shadow`] is set, [`encode`](Self::encode) for every renderer, then
/// [`render`](Self::render) for every renderer inside one opaque pass, on frames that show the
/// [`LayerRender`](crate::LayerRender) [`render_layer`](Self::render_layer) for every renderer
/// inside the layer pass, then the post pass.
pub trait GeometryRenderer<G: GpuGeometry>: Send + Sync + 'static {
    /// Reads what this frame draws from the world and uploads whatever does not depend on
    /// visibility.
    fn prepare(&mut self, gpu: &Gpu, world: &mut World, view: &FrameView<'_, G>);

    /// Gets ready to draw what is in the world, with nothing to draw to yet: compiles and
    /// uploads what `prepare` would on its first frame (a [`SharedDevice`](crate::SharedDevice)
    /// before its surface is given up). Does nothing by default.
    fn warm(&mut self, gpu: &Gpu, world: &mut World) {
        let _ = (gpu, world);
    }

    /// Decides what is visible, and what casts into the shadow map, and uploads the draw data.
    fn queue(&mut self, gpu: &Gpu, view: &FrameView<'_, G>);

    /// Records the casters' draws into the shadow map: depth only, from the light, through
    /// `lighting.shadow_matrix`. Bind group 0 is `frame`. Draws nothing by default.
    fn render_shadows(&self, pass: &mut wgpu::RenderPass<'_>, frame: &wgpu::BindGroup) {
        let _ = (pass, frame);
    }

    /// Records passes of its own into offscreen targets, before the opaque pass (a ray march
    /// at a lower resolution). Records nothing by default.
    fn encode(&self, encoder: &mut wgpu::CommandEncoder, bindings: &FrameBindings<'_>) {
        let _ = (encoder, bindings);
    }

    /// Records the draws. Set the bind groups of `bindings` again after changing pipelines
    /// whose layouts differ.
    fn render(&self, pass: &mut wgpu::RenderPass<'_>, bindings: &FrameBindings<'_>);

    /// Records the draws of what is marked [`OnLayer`] only, into the layer render: a pass with
    /// the opaque pass's formats and sample count, cleared to the layer's colour, after the
    /// opaque pass. Only called on frames that show the layer. Draws nothing by default.
    fn render_layer(&self, pass: &mut wgpu::RenderPass<'_>, bindings: &FrameBindings<'_>) {
        let _ = (pass, bindings);
    }

    /// Instances drawn and culled by the last [`queue`](Self::queue), and draw calls.
    fn stats(&self) -> crate::RenderStats {
        crate::RenderStats::default()
    }
}

/// Builds a renderer once the device exists.
pub type RendererFactory<G> =
    Box<dyn Fn(&RendererSetup<'_>) -> Box<dyn GeometryRenderer<G>> + Send + Sync>;
