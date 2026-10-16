//! The caller's image effects: fullscreen passes of its own between the opaque pass and the post
//! pass, and its hooks into the post pass ([`GradeShader`]) and the sky ([`SkyFigure`]).

use bevy_ecs::resource::Resource;
use bevy_ecs::world::World;
use fk_geometry::GpuGeometry;
use fk_math::Real;
use fk_math::nalgebra::Matrix4;

use crate::Gpu;

/// What an [`ImageEffect`] is built against.
pub struct EffectSetup<'a> {
    /// The device.
    pub gpu: &'a Gpu,
    /// The geometry's shader module name (`GpuGeometry::SHADER`), for
    /// [`compose_effect`](Self::compose_effect).
    pub geometry: &'static str,
    /// Layout of bind group 0 (`fk::view`: view, world, lighting), which the pass sets.
    pub frame_layout: &'a wgpu::BindGroupLayout,
    /// Format of the colour target, linear HDR. Its alpha is 0 on the sky, 1 on surfaces, or
    /// what a ray-marched field's `sdf_alpha` says.
    pub color_format: wgpu::TextureFormat,
    /// Whether the depth target is multisampled: bind it as `texture_depth_multisampled_2d`.
    pub multisampled_depth: bool,
}

impl EffectSetup<'_> {
    /// Composes `skeleton`, a fullscreen pass of the effect's own, against the engine's modules
    /// (`fk::view`, `fk::geometry`, `fk::sky`…) with `MULTISAMPLED` defined when the depth
    /// target is multisampled, and makes its shader module.
    ///
    /// # Panics
    ///
    /// If it does not compose: an effect's shader is checked in its tests with
    /// [`validate_effect`].
    pub fn compose_effect(
        &self,
        skeleton: fk_shaders::Module<'_>,
        libraries: &[fk_shaders::Module<'_>],
    ) -> wgpu::ShaderModule {
        let module = compose(skeleton, self.geometry, self.multisampled_depth, libraries)
            .unwrap_or_else(|error| panic!("{}: {error}", skeleton.path));
        self.gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(skeleton.path),
                source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(module)),
            })
    }

    /// A pipeline drawing one fullscreen triangle (entry points `vertex` and `fragment` of
    /// `shader`) into `targets`, with bind group 0 the frame's and group 1 `layout`.
    pub fn fullscreen_pipeline(
        &self,
        label: &str,
        shader: &wgpu::ShaderModule,
        layout: &wgpu::BindGroupLayout,
        targets: &[Option<wgpu::ColorTargetState>],
    ) -> wgpu::RenderPipeline {
        let device = &self.gpu.device;
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[Some(self.frame_layout), Some(layout)],
            immediate_size: 0,
        });
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vertex"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fragment"),
                compilation_options: Default::default(),
                targets,
            }),
            multiview_mask: None,
            cache: None,
        })
    }
}

fn compose(
    skeleton: fk_shaders::Module<'_>,
    geometry: &str,
    multisampled: bool,
    libraries: &[fk_shaders::Module<'_>],
) -> Result<fk_shaders::naga::Module, fk_shaders::ShaderError> {
    let defs: &[&str] = if multisampled { &["MULTISAMPLED"] } else { &[] };
    let hooks = fk_shaders::Hooks {
        libraries,
        ..Default::default()
    };
    fk_shaders::compose_module(skeleton, geometry, defs, &hooks)
}

/// Composes an [`ImageEffect`]'s skeleton for `G`, with and without a multisampled depth
/// target, and validates it, as [`EffectSetup::compose_effect`] will; for the caller's tests.
///
/// # Errors
///
/// The composition or validation error, pointing into the WGSL.
pub fn validate_effect<G: GpuGeometry>(
    skeleton: fk_shaders::Module<'_>,
    libraries: &[fk_shaders::Module<'_>],
) -> Result<(), String> {
    use fk_shaders::naga::valid::{Capabilities, ValidationFlags, Validator};
    for multisampled in [false, true] {
        let module = compose(skeleton, G::SHADER, multisampled, libraries)
            .map_err(|error| format!("{}: {error}", skeleton.path))?;
        Validator::new(ValidationFlags::all(), Capabilities::MULTISAMPLED_SHADING)
            .validate(&module)
            .map_err(|error| format!("{}: {error:?}", skeleton.path))?;
    }
    Ok(())
}

/// One frame, as an [`ImageEffect`] sees it.
pub struct EffectFrame {
    /// How the eye moved since the last frame drawn: its coordinates now to its coordinates
    /// then, the identity on the first frame.
    pub motion: Matrix4<Real>,
    /// The targets' width and height, in pixels.
    pub size: [u32; 2],
    /// Seconds since the last frame.
    pub delta: f32,
}

/// The targets an [`ImageEffect`] works on: the image (resolved, one sample, its usages
/// including texture binding and copies both ways) and its depth (reverse-Z, geodesic distance
/// along each pixel's ray).
pub struct EffectTargets<'a> {
    /// The image.
    pub color: &'a wgpu::Texture,
    /// Its depth.
    pub depth: &'a wgpu::TextureView,
}

/// A fullscreen pass of the caller's own over the image, after the opaque pass (and the
/// [`Settle`](crate::Settle), unless it is [`settled`](Self::settled)) and before the post
/// pass: an inked look, a datamosh, a crossfade. It reads the image and its depth and writes
/// the image back (drawing into a target of its own and copying it over the image), so every
/// later effect and the post pass read what it made. Effects run in the order they were added
/// ([`add_image_effect`]), the settled ones first.
pub trait ImageEffect: Send + Sync + 'static {
    /// Its name, for the GPU timings of a performance recording.
    fn label(&self) -> &'static str;

    /// Points it at the targets, when they are made and after every resize.
    fn set_targets(&mut self, gpu: &Gpu, targets: &EffectTargets<'_>);

    /// Whether it is drawn before the [`Settle`](crate::Settle) and settled with the image:
    /// drawn each frame through another point of its pixels, what it draws averages out over
    /// frames as the image does (lines found in the depth stop coming in stairs and crawling).
    /// False by default: drawn on the settled image, crisp.
    fn settled(&self) -> bool {
        false
    }

    /// Reads what this frame does from the world and uploads it, before anything is drawn.
    fn prepare(&mut self, gpu: &Gpu, world: &mut World, frame: &EffectFrame);

    /// Records its passes over `targets`; `frame` is bind group 0. Records nothing when it has
    /// nothing to do, so the image is exactly as without it.
    fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        targets: &EffectTargets<'_>,
        frame: &wgpu::BindGroup,
    );
}

/// Builds an [`ImageEffect`] once the device exists.
pub type EffectFactory = Box<dyn Fn(&EffectSetup<'_>) -> Box<dyn ImageEffect> + Send + Sync>;

/// Image effects waiting for the device, in the order they run.
#[derive(Resource, Default)]
pub(crate) struct EffectFactories(pub Vec<EffectFactory>);

/// Adds an [`ImageEffect`] to the app's [`RenderPlugin`](crate::RenderPlugin), after the ones
/// already there. `factory` builds it once the device is open.
///
/// Call it while building the app, before it runs.
pub fn add_image_effect(
    app: &mut fk_app::App,
    factory: impl Fn(&EffectSetup<'_>) -> Box<dyn ImageEffect> + Send + Sync + 'static,
) {
    app.world_mut()
        .get_resource_or_insert_with(EffectFactories::default)
        .0
        .push(Box::new(factory));
}

/// The caller's `fk::grade`, as a resource: its effects in the post pass, at fixed points of
/// its order (see `fk_shaders::Hooks::grade`), reading [`GradeValues`](crate::GradeValues),
/// the [`FrameCapture`](crate::FrameCapture) and the [`LayerRender`](crate::LayerRender)
/// through `fk::post`. Read when the device opens; without it the post pass leaves the image
/// as it is.
#[derive(Resource, Clone, Debug, PartialEq, Eq)]
pub struct GradeShader {
    /// A name for messages, and the file path the module's errors point into.
    pub name: String,
    /// The `fk::grade` module.
    pub source: String,
    /// Modules it imports, as `(path, source)`, in an order where each may import the ones
    /// before it.
    pub libraries: Vec<(String, String)>,
}

impl GradeShader {
    /// Composes the post pass's shader with this grade for `G`, with and without a
    /// multisampled depth target, and validates it, as the renderer will; for the caller's
    /// tests.
    ///
    /// # Errors
    ///
    /// The composition or validation error, pointing into the WGSL.
    pub fn validate<G: GpuGeometry>(&self) -> Result<(), String> {
        use fk_shaders::naga::valid::{Capabilities, ValidationFlags, Validator};
        for multisampled in [false, true] {
            let module = self.compose(G::SHADER, multisampled, None)?;
            Validator::new(ValidationFlags::all(), Capabilities::MULTISAMPLED_SHADING)
                .validate(&module)
                .map_err(|error| format!("{}: {error:?}", self.name))?;
        }
        Ok(())
    }

    pub(crate) fn compose(
        &self,
        geometry: &str,
        multisampled: bool,
        figure: Option<&SkyFigure>,
    ) -> Result<fk_shaders::naga::Module, String> {
        compose_post(Some(self), geometry, multisampled, figure)
    }
}

/// The post pass's shader with `grade` (or the engine's) and `figure` (or the engine's).
pub(crate) fn compose_post(
    grade: Option<&GradeShader>,
    geometry: &str,
    multisampled: bool,
    figure: Option<&SkyFigure>,
) -> Result<fk_shaders::naga::Module, String> {
    let libraries: Vec<_> = grade
        .map(|grade| &grade.libraries[..])
        .unwrap_or_default()
        .iter()
        .map(|(path, source)| fk_shaders::Module { path, source })
        .collect();
    let hooks = fk_shaders::Hooks {
        grade: grade.map(|grade| fk_shaders::Module {
            path: &grade.name,
            source: &grade.source,
        }),
        figure: figure.map(SkyFigure::module),
        libraries: &libraries,
        ..Default::default()
    };
    let defs: &[&str] = if multisampled { &["MULTISAMPLED"] } else { &[] };
    fk_shaders::compose_hooked(fk_shaders::Shader::Post, geometry, defs, &hooks)
        .map_err(|error| error.to_string())
}

/// The caller's `fk::figure`, as a resource: the lines of the sky's bodies drawn as a
/// [`DiscShape::Figure`](crate::DiscShape::Figure) (see `fk_shaders::Hooks::figure`). Read
/// when the device opens; without it a figure is a circle.
#[derive(Resource, Clone, Debug, PartialEq, Eq)]
pub struct SkyFigure {
    /// A name for messages, and the file path the module's errors point into.
    pub name: String,
    /// The `fk::figure` module.
    pub source: String,
}

impl SkyFigure {
    pub(crate) fn module(&self) -> fk_shaders::Module<'_> {
        fk_shaders::Module {
            path: &self.name,
            source: &self.source,
        }
    }

    /// Composes the sky's shader with this figure for `G` and validates it, as the renderer
    /// will; for the caller's tests.
    ///
    /// # Errors
    ///
    /// The composition or validation error, pointing into the WGSL.
    pub fn validate<G: GpuGeometry>(&self) -> Result<(), String> {
        use fk_shaders::naga::valid::{Capabilities, ValidationFlags, Validator};
        let hooks = fk_shaders::Hooks {
            figure: Some(self.module()),
            ..Default::default()
        };
        let module = fk_shaders::compose_hooked(fk_shaders::Shader::Sky, G::SHADER, &[], &hooks)
            .map_err(|error| error.to_string())?;
        Validator::new(ValidationFlags::all(), Capabilities::empty())
            .validate(&module)
            .map(|_| ())
            .map_err(|error| format!("{}: {error:?}", self.name))
    }
}
