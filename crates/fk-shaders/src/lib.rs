//! WGSL templates and per-geometry shader modules, composed at pipeline creation with
//! `naga_oil`.
//!
//! A pipeline's shader is a geometry-agnostic skeleton ([`Shader`]) that imports two shared
//! modules:
//!
//! - `fk::view`: the bindings of group 0, the same in every pipeline. `View` is the eye and its
//!   projection; `World` is the caller-filled block (time plus eight opaque `vec4`s the caller
//!   gives meaning to); `Lighting` is the light, the sky and the fog.
//! - `fk::sky`: the sky seen along a direction at the eye (`sky_color`), and the haze far
//!   surfaces fade into (`sky_haze`). Directions at the eye mean the same in every geometry.
//!   Bodies drawn as figures take their lines from `fk::figure` (`figure_lines`): the engine's
//!   is a circle; a caller replaces it through [`Hooks::figure`].
//! - `fk::geometry`: one module per geometry, chosen by the geometry's `GpuGeometry::SHADER`
//!   (in `fk-geometry`). It provides
//!   `geo_apply_iso`, `geo_inverse_iso`, `geo_normalize_point`, `geo_project_tangent`, `geo_inner`,
//!   `geo_transport_from_eye`, `geo_distance_from_eye`, `geo_direction_from_eye`,
//!   `geo_point_from_eye`, `geo_fog`,
//!   `geo_project`, and the depth
//!   encoding `geo_depth_to_distance` / `geo_distance_to_depth` shared by every render path;
//!   also `geo_log_origin` / `geo_exp_origin` (the coordinates meshes are authored in) and
//!   `geo_pixel_footprint`.
//!
//! The raster skeleton also imports:
//!
//! - `fk::displace`: helpers for deform modules (displacement by distance from the eye,
//!   stretching, a minimum width on screen, a pull or a lean towards a point) and the `Shaped` vertex they return. They never read
//!   the world block; the deform module passes in what its game put there.
//! - `fk::deform`: one function, `deform(position, normal, iso, data) -> Shaped`, shaping a
//!   mesh's vertices before their instance's isometry. The engine's module is the identity; a
//!   caller replaces it through [`Hooks::deform`], as the ray marcher will take the caller's SDF.
//! - `fk::shadow`: the directional light's shadow map (bind group 1) and `shadow_light`.
//! - `fk::paint`: the painted light every surface takes (`banded`).
//!
//! The ray-march skeleton imports, besides `fk::view`, `fk::geometry`, `fk::sky` and
//! `fk::paint`:
//!
//! - `fk::march`: the block of one ray-marched scene (bind group 1): the eye's coordinates to
//!   the scene's, the zoom, the march's budgets and the caller's own values.
//! - `fk::sdf`: the scene's signed distance field, `sdf_distance` and `sdf_color`. The engine's
//!   module is a ball of radius 1; a caller replaces it through [`Hooks::sdf`].
//! - `fk::metric`, composed with `SAMPLED_METRIC` only: the geodesic equation
//!   `geo_geodesic_accel` the rays are integrated along (RK4) instead of the closed-form
//!   geodesics. The engine's module is flat; a caller passes a geometry's
//!   `SampledMetric::wgsl_module` through [`Hooks::metric`].
//!
//! The post skeleton imports, besides `fk::view` and `fk::geometry`:
//!
//! - `fk::post`: its bindings (group 1): the image, its depth, a frame captured on request, the
//!   layer render, and the caller's values (`grade.values`).
//! - `fk::grade`: the caller's effects, called at fixed points of the post pass's order
//!   (`grade_source`, `grade_blurred`, `grade_saturated`, `grade_over`). The engine's module
//!   leaves the image as it is; a caller replaces it through [`Hooks::grade`].
//!
//! [`compose`] builds and validates one permutation; [`permutations`] lists all of them, and the
//! crate's tests validate each one with naga. A skeleton can have variants switched by shader
//! defs ([`Shader::defs`]): the post and settle passes read a multisampled depth target when
//! composed with `MULTISAMPLED`. [`compose_hooked`] adds the caller's modules, and
//! [`compose_module`] composes a skeleton of the caller's own (a fullscreen pass of its own)
//! against the same modules.

use std::collections::HashMap;
use std::fmt;

pub use naga;
use naga_oil::compose::{
    ComposableModuleDescriptor, Composer, ComposerError, NagaModuleDescriptor, ShaderDefValue,
    ShaderLanguage,
};

const VIEW: (&str, &str) = ("fk/view.wgsl", include_str!("wgsl/view.wgsl"));
const SKY: (&str, &str) = ("fk/sky.wgsl", include_str!("wgsl/sky.wgsl"));
const DISPLACE: (&str, &str) = ("fk/displace.wgsl", include_str!("wgsl/displace.wgsl"));
const SHADOW: (&str, &str) = ("fk/shadow.wgsl", include_str!("wgsl/shadow.wgsl"));
const PAINT: (&str, &str) = ("fk/paint.wgsl", include_str!("wgsl/paint.wgsl"));
const POST: (&str, &str) = ("fk/post.wgsl", include_str!("wgsl/post_bindings.wgsl"));
const MARCH: (&str, &str) = ("fk/march.wgsl", include_str!("wgsl/march.wgsl"));
/// The engine's `fk::deform`, the identity.
const DEFORM: (&str, &str) = ("fk/deform.wgsl", include_str!("wgsl/deform.wgsl"));
/// The engine's `fk::sdf`, a ball of radius 1.
const SDF: (&str, &str) = ("fk/sdf.wgsl", include_str!("wgsl/sdf.wgsl"));
/// The engine's `fk::metric`, flat.
const METRIC: (&str, &str) = ("fk/metric.wgsl", include_str!("wgsl/metric.wgsl"));
/// The engine's `fk::grade`, leaving the image as it is.
const GRADE: (&str, &str) = ("fk/grade.wgsl", include_str!("wgsl/grade.wgsl"));
/// The engine's `fk::figure`, a circle.
const FIGURE: (&str, &str) = ("fk/figure.wgsl", include_str!("wgsl/figure.wgsl"));

/// The geometry modules, by the name a geometry gives in `GpuGeometry::SHADER`.
const GEOMETRIES: &[(&str, &str, &str)] = &[
    (
        "e3",
        "fk/geometry/e3.wgsl",
        include_str!("wgsl/geometry/e3.wgsl"),
    ),
    (
        "h3",
        "fk/geometry/h3.wgsl",
        include_str!("wgsl/geometry/h3.wgsl"),
    ),
    (
        "s3",
        "fk/geometry/s3.wgsl",
        include_str!("wgsl/geometry/s3.wgsl"),
    ),
];

/// A pipeline skeleton.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Shader {
    /// Instanced meshes in flat painted colours. Entry points `vertex` and `fragment`,
    /// `shadow_vertex` for the shadow map (no fragment stage), and `window_fragment` for
    /// windows onto the scene seen from another eye (its group 2).
    Raster,
    /// The fullscreen post pass. Entry points `vertex`, `fragment` and `text_fragment`.
    Post,
    /// The image settled over frames, after the opaque pass and before the caller's image
    /// effects, a fullscreen pass. Entry points `vertex` and `fragment`.
    Settle,
    /// The sky behind everything, a fullscreen pass. Entry points `vertex` and `fragment`.
    Sky,
    /// A signed distance field ray-marched in a fullscreen pass. Entry points `vertex`,
    /// `fragment` (full resolution, with depth), `fragment_half` (into offscreen colour and
    /// distance targets) and `composite` (those targets brought to full resolution).
    RayMarch,
    /// The caller's signed distance field evaluated at given points, a compute pass. Entry
    /// point `probe`.
    SdfProbe,
}

impl Shader {
    /// Every skeleton.
    pub const ALL: &[Self] = &[
        Self::Raster,
        Self::Post,
        Self::Settle,
        Self::Sky,
        Self::RayMarch,
        Self::SdfProbe,
    ];

    /// The shader defs a skeleton reacts to; [`compose_with`] can turn each on.
    pub fn defs(self) -> &'static [&'static str] {
        match self {
            Self::Post | Self::Settle => &["MULTISAMPLED"],
            Self::RayMarch => &["SAMPLED_METRIC", "SDF_ALPHA"],
            Self::Raster | Self::Sky | Self::SdfProbe => &[],
        }
    }

    /// Its entry points.
    pub fn entry_points(self) -> &'static [&'static str] {
        match self {
            Self::Raster => &["vertex", "fragment", "shadow_vertex", "window_fragment"],
            Self::Post => &["vertex", "fragment", "text_fragment"],
            Self::Settle | Self::Sky => &["vertex", "fragment"],
            Self::RayMarch => &[
                "vertex",
                "fragment",
                "fragment_half",
                "composite",
                "fragment_carry",
                "vertex_tiles",
            ],
            Self::SdfProbe => &["probe"],
        }
    }

    fn source(self) -> (&'static str, &'static str) {
        match self {
            Self::Raster => ("fk/raster.wgsl", include_str!("wgsl/raster.wgsl")),
            Self::Post => ("fk/post.wgsl", include_str!("wgsl/post.wgsl")),
            Self::Settle => ("fk/settle.wgsl", include_str!("wgsl/settle.wgsl")),
            Self::Sky => ("fk/sky_pass.wgsl", include_str!("wgsl/sky_pass.wgsl")),
            Self::RayMarch => ("fk/ray_march.wgsl", include_str!("wgsl/ray_march.wgsl")),
            Self::SdfProbe => ("fk/sdf_probe.wgsl", include_str!("wgsl/sdf_probe.wgsl")),
        }
    }
}

/// The geometry modules that exist, by name.
pub fn geometries() -> impl Iterator<Item = &'static str> {
    GEOMETRIES.iter().map(|(name, ..)| *name)
}

/// Every (skeleton, geometry) pair that can be composed.
pub fn permutations() -> impl Iterator<Item = (Shader, &'static str)> {
    Shader::ALL
        .iter()
        .flat_map(|&shader| geometries().map(move |geometry| (shader, geometry)))
}

/// Why a shader could not be composed.
#[derive(Debug)]
pub enum ShaderError {
    /// No module implements `fk::geometry` for this geometry name.
    UnknownGeometry(String),
    /// Composition or validation failed; the message points into the WGSL source.
    Compose(String),
}

impl fmt::Display for ShaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownGeometry(name) => write!(f, "no shader module for geometry `{name}`"),
            Self::Compose(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for ShaderError {}

/// A WGSL module the caller supplies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Module<'a> {
    /// Its file path, for error messages.
    pub path: &'a str,
    /// The source, starting with its `#define_import_path`.
    pub source: &'a str,
}

/// What a caller adds to a skeleton.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Hooks<'a> {
    /// Replaces the engine's `fk::deform` (the identity) in [`Shader::Raster`]. The module must
    /// `#define_import_path fk::deform` and define
    /// `fn deform(position: vec4<f32>, normal: vec4<f32>, iso: mat4x4<f32>, data: vec4<f32>) -> Shaped`,
    /// with `Shaped` from `fk::displace`.
    pub deform: Option<Module<'a>>,
    /// Replaces the engine's `fk::sdf` (a ball of radius 1) in [`Shader::RayMarch`] and
    /// [`Shader::SdfProbe`]. The module must `#define_import_path fk::sdf` and define
    /// `fn sdf_distance(q: vec3<f32>) -> f32`, a lower bound of the distance from `q` to the
    /// surface (negative inside), and `fn sdf_color(q: vec3<f32>) -> vec3<f32>`, the surface's
    /// linear colour, both in the scene's coordinates and units. It reads its values from
    /// `march.params` (`fk::march`). Composed with `SDF_ALPHA`, it also defines
    /// `fn sdf_alpha(q: vec3<f32>) -> f32`: the image's alpha on the surface there, 1 to 4, a
    /// mark for the caller's image effects (1 without it).
    pub sdf: Option<Module<'a>>,
    /// Replaces the engine's flat `fk::metric` in [`Shader::RayMarch`] composed with
    /// `SAMPLED_METRIC`: the module must `#define_import_path fk::metric` and define
    /// `fn geo_geodesic_accel(x: vec3<f32>, v: vec3<f32>) -> vec3<f32>`, in the eye's normal
    /// coordinates.
    pub metric: Option<Module<'a>>,
    /// Replaces the engine's `fk::grade` (the image as it is) in [`Shader::Post`]: the module
    /// must `#define_import_path fk::grade` and define, reading its values from `fk::post`'s
    /// `grade.values` and its textures from `fk::post`,
    /// `fn grade_source(uv: vec2<f32>) -> vec2<f32>` (where the image at `uv` is taken from,
    /// `uv` itself for no resampling),
    /// `fn grade_blurred(color: vec3<f32>, source: vec2<f32>, frag: vec2<f32>) -> vec3<f32>`
    /// (after the blur), `fn grade_saturated(color: vec3<f32>, image: vec4<f32>) -> vec3<f32>`
    /// (after the saturation; `image` is the pixel as drawn, its alpha 0 on the sky) and
    /// `fn grade_over(color: vec3<f32>, uv: vec2<f32>, source: vec2<f32>, r: f32, frag: vec2<f32>) -> vec3<f32>`
    /// (after the vignette, `r` the distance from the centre, 1 at the corners).
    pub grade: Option<Module<'a>>,
    /// Replaces the engine's `fk::figure` (a circle) wherever `fk::sky` is imported: the module
    /// must `#define_import_path fk::figure` and define
    /// `fn figure_lines(p: vec2<f32>, radius: f32, params: vec4<f32>) -> f32`, the distance
    /// from `p` (radians in the plane of the sky round the body, y up) to the lines of a body
    /// of angular radius `radius` drawn as a figure, with the body's own values `params`. It may
    /// import `fk::view` only.
    pub figure: Option<Module<'a>>,
    /// Modules the hooks import, added in order after the engine's and before the hooks, so
    /// each may import the engine's and the ones before it.
    pub libraries: &'a [Module<'a>],
}

/// Composes `shader` for the geometry module `geometry` into a validated naga module.
pub fn compose(shader: Shader, geometry: &str) -> Result<naga::Module, ShaderError> {
    compose_with(shader, geometry, &[])
}

/// [`compose`] with the shader defs in `defs` turned on (see [`Shader::defs`]).
pub fn compose_with(
    shader: Shader,
    geometry: &str,
    defs: &[&str],
) -> Result<naga::Module, ShaderError> {
    compose_hooked(shader, geometry, defs, &Hooks::default())
}

/// [`compose_with`] with the caller's modules in `hooks`.
pub fn compose_hooked<'a>(
    shader: Shader,
    geometry: &str,
    defs: &[&str],
    hooks: &Hooks<'a>,
) -> Result<naga::Module, ShaderError> {
    let (path, source) = shader.source();
    compose_module(Module { path, source }, geometry, defs, hooks)
}

/// Composes a skeleton of the caller's own, `skeleton`, for the geometry module `geometry`,
/// with the shader defs in `defs` and the caller's modules in `hooks`: it may import every
/// module a [`Shader`] can (`fk::view`, `fk::geometry`, `fk::sky`…). Bind group 0 is
/// `fk::view`'s; a fullscreen pass of the caller's binds its own textures in group 1, as the
/// engine's do (and then does not import `fk::post`).
///
/// # Errors
///
/// [`ShaderError::UnknownGeometry`], or the composition error pointing into the WGSL.
pub fn compose_module<'a>(
    skeleton: Module<'_>,
    geometry: &str,
    defs: &[&str],
    hooks: &Hooks<'a>,
) -> Result<naga::Module, ShaderError> {
    let &(_, geometry_path, geometry_source) = GEOMETRIES
        .iter()
        .find(|(name, ..)| *name == geometry)
        .ok_or_else(|| ShaderError::UnknownGeometry(geometry.to_owned()))?;
    let mut composer = Composer::default();
    let describe = |composer: &Composer, error: ComposerError| {
        ShaderError::Compose(error.emit_to_string(composer))
    };
    let hook = |module: Option<Module<'a>>, default: (&'a str, &'a str)| {
        module.map_or(default, |m| (m.path, m.source))
    };
    let shader_defs = defs
        .iter()
        .map(|def| ((*def).to_owned(), ShaderDefValue::Bool(true)))
        .collect::<HashMap<_, _>>();
    let engine = [
        VIEW,
        (geometry_path, geometry_source),
        hook(hooks.figure, FIGURE),
        SKY,
        DISPLACE,
        SHADOW,
        PAINT,
        MARCH,
        POST,
    ];
    let libraries = hooks
        .libraries
        .iter()
        .map(|module| (module.path, module.source));
    let hooked = [
        hook(hooks.deform, DEFORM),
        hook(hooks.sdf, SDF),
        hook(hooks.metric, METRIC),
        hook(hooks.grade, GRADE),
    ];
    for (path, source) in engine.into_iter().chain(libraries).chain(hooked) {
        let added = composer
            .add_composable_module(ComposableModuleDescriptor {
                source,
                file_path: path,
                language: ShaderLanguage::Wgsl,
                shader_defs: shader_defs.clone(),
                ..Default::default()
            })
            .map(|_| ());
        added.map_err(|error| describe(&composer, error))?;
    }
    composer
        .make_naga_module(NagaModuleDescriptor {
            source: skeleton.source,
            file_path: skeleton.path,
            shader_defs,
            ..Default::default()
        })
        .map_err(|error| describe(&composer, error))
}

#[cfg(test)]
mod tests {
    use naga::valid::{Capabilities, ValidationFlags, Validator};

    use super::*;

    #[test]
    fn every_permutation_validates() {
        let mut count = 0;
        for (shader, geometry) in permutations() {
            let module = compose(shader, geometry)
                .unwrap_or_else(|error| panic!("{shader:?} × {geometry}: {error}"));
            Validator::new(ValidationFlags::all(), Capabilities::empty())
                .validate(&module)
                .unwrap_or_else(|error| panic!("{shader:?} × {geometry}: {error:?}"));
            for def in shader.defs() {
                let module = compose_with(shader, geometry, &[def])
                    .unwrap_or_else(|error| panic!("{shader:?} × {geometry} + {def}: {error}"));
                Validator::new(ValidationFlags::all(), Capabilities::MULTISAMPLED_SHADING)
                    .validate(&module)
                    .unwrap_or_else(|error| panic!("{shader:?} × {geometry} + {def}: {error:?}"));
            }
            for &entry in shader.entry_points() {
                assert!(
                    module.entry_points.iter().any(|point| point.name == entry),
                    "{shader:?} × {geometry} has no `{entry}`"
                );
            }
            count += 1;
        }
        assert_eq!(count, Shader::ALL.len() * GEOMETRIES.len());
    }

    #[test]
    fn a_deform_hook_replaces_the_identity() {
        let library = Module {
            path: "test/lift.wgsl",
            source: "#define_import_path test::lift\nfn lift() -> f32 { return 2.0; }\n",
        };
        let deform = Module {
            path: "test/deform.wgsl",
            source: r"#define_import_path fk::deform
#import fk::displace::{Shaped, unshaped, stretch_up}
#import test::lift::lift
fn deform(position: vec4<f32>, normal: vec4<f32>, iso: mat4x4<f32>, data: vec4<f32>) -> Shaped {
    return unshaped(stretch_up(position, lift()), normal);
}
",
        };
        let hooks = Hooks {
            deform: Some(deform),
            libraries: &[library],
            ..Default::default()
        };
        for geometry in geometries() {
            let module = compose_hooked(Shader::Raster, geometry, &[], &hooks)
                .unwrap_or_else(|error| panic!("{geometry}: {error}"));
            Validator::new(ValidationFlags::all(), Capabilities::empty())
                .validate(&module)
                .unwrap_or_else(|error| panic!("{geometry}: {error:?}"));
            let calls_lift = module.functions.iter().any(|(_, function)| {
                function
                    .name
                    .as_deref()
                    .is_some_and(|name| name.starts_with("lift"))
            });
            assert!(calls_lift, "the hook's library is in");
        }
    }

    #[test]
    fn sdf_and_metric_hooks_replace_the_defaults() {
        let sdf = Module {
            path: "test/sdf.wgsl",
            source: r"#define_import_path fk::sdf
#import fk::march::march
fn sdf_distance(q: vec3<f32>) -> f32 {
    return length(max(abs(q) - march.params[0].xyz, vec3<f32>(0.0))) - 0.01;
}
fn sdf_color(q: vec3<f32>) -> vec3<f32> {
    return march.params[1].rgb;
}
",
        };
        let metric = Module {
            path: "test/metric.wgsl",
            source: r"#define_import_path fk::metric
fn geo_geodesic_accel(x: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    return -0.01 * dot(v, v) * x;
}
",
        };
        let hooks = Hooks {
            sdf: Some(sdf),
            metric: Some(metric),
            ..Default::default()
        };
        for geometry in geometries() {
            for (shader, defs) in [
                (Shader::RayMarch, &[][..]),
                (Shader::RayMarch, &["SAMPLED_METRIC"][..]),
                (Shader::SdfProbe, &[][..]),
            ] {
                let module = compose_hooked(shader, geometry, defs, &hooks)
                    .unwrap_or_else(|error| panic!("{shader:?} {defs:?} × {geometry}: {error}"));
                Validator::new(ValidationFlags::all(), Capabilities::empty())
                    .validate(&module)
                    .unwrap_or_else(|error| panic!("{shader:?} × {geometry}: {error:?}"));
                let has = |prefix: &str| {
                    module.functions.iter().any(|(_, function)| {
                        function
                            .name
                            .as_deref()
                            .is_some_and(|name| name.starts_with(prefix))
                    })
                };
                assert!(has("sdf_distance"), "{shader:?}");
                if defs.contains(&"SAMPLED_METRIC") {
                    assert!(has("geo_geodesic_accel"), "the metric is in");
                }
            }
        }
    }

    #[test]
    fn grade_and_figure_hooks_replace_the_defaults() {
        let grade = Module {
            path: "test/grade.wgsl",
            source: r"#define_import_path fk::grade
#import fk::post::{grade, captured, linear}
fn grade_source(uv: vec2<f32>) -> vec2<f32> {
    return uv + grade.values[0].xy;
}
fn grade_blurred(color: vec3<f32>, source: vec2<f32>, frag: vec2<f32>) -> vec3<f32> {
    return color;
}
fn grade_saturated(color: vec3<f32>, image: vec4<f32>) -> vec3<f32> {
    return color * image.a;
}
fn grade_over(color: vec3<f32>, uv: vec2<f32>, source: vec2<f32>, r: f32, frag: vec2<f32>) -> vec3<f32> {
    return mix(color, textureSampleLevel(captured, linear, uv, 0.0).rgb, grade.values[1].x);
}
",
        };
        let figure = Module {
            path: "test/figure.wgsl",
            source: r"#define_import_path fk::figure
fn figure_lines(p: vec2<f32>, radius: f32, params: vec4<f32>) -> f32 {
    return abs(max(abs(p.x), abs(p.y)) - radius * params.x);
}
",
        };
        let hooks = Hooks {
            grade: Some(grade),
            figure: Some(figure),
            ..Default::default()
        };
        for geometry in geometries() {
            for (shader, defs) in [
                (Shader::Post, &[][..]),
                (Shader::Post, &["MULTISAMPLED"][..]),
                (Shader::Sky, &[][..]),
            ] {
                let module = compose_hooked(shader, geometry, defs, &hooks)
                    .unwrap_or_else(|error| panic!("{shader:?} × {geometry}: {error}"));
                Validator::new(ValidationFlags::all(), Capabilities::MULTISAMPLED_SHADING)
                    .validate(&module)
                    .unwrap_or_else(|error| panic!("{shader:?} × {geometry}: {error:?}"));
            }
        }
    }

    #[test]
    fn a_skeleton_of_the_callers_own_composes() {
        let skeleton = Module {
            path: "test/pass.wgsl",
            source: r"#import fk::view::{view, pixel_to_ndc}
#import fk::sky::sky_color
@group(1) @binding(0) var image: texture_2d<f32>;
@vertex
fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}
@fragment
fn fragment(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let dir = normalize(vec3<f32>(pixel_to_ndc(frag.xy) / view.proj_scale, -1.0));
    return textureLoad(image, vec2<i32>(frag.xy), 0) + vec4<f32>(sky_color(dir), 0.0);
}
",
        };
        for geometry in geometries() {
            let module = compose_module(skeleton, geometry, &[], &Hooks::default())
                .unwrap_or_else(|error| panic!("{geometry}: {error}"));
            Validator::new(ValidationFlags::all(), Capabilities::empty())
                .validate(&module)
                .unwrap_or_else(|error| panic!("{geometry}: {error:?}"));
        }
    }

    #[test]
    fn unknown_geometry_is_an_error() {
        assert!(matches!(
            compose(Shader::Raster, "taxicab"),
            Err(ShaderError::UnknownGeometry(_))
        ));
    }
}
