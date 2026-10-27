//! What one frame is drawn with: the settings resources games fill, and their GPU blocks.

use bevy_ecs::resource::Resource;
use encase::ShaderType;
use fk_math::Real;
use fk_math::nalgebra::{Vector2, Vector3};

use crate::Color;
use crate::gpu::{Vec2f, Vec4f};

/// The light, the sky and the fog, as a resource.
///
/// Directions are given in the reference frame at the scene root's origin and carried to the
/// eye every frame, so in flat space each is the same direction everywhere.
#[derive(Resource, Clone, Debug)]
pub struct Lighting {
    /// Direction towards the one directional light (the sun by day, the moon by night: the
    /// game decides).
    pub sun: Vector3<Real>,
    /// Light on surfaces facing it.
    pub sun_color: Color,
    /// Light on surfaces facing away from it: the painted shadow colour.
    pub shadow_tint: Color,
    /// What is drawn behind everything, and what far surfaces fade into.
    pub sky: Sky,
    /// Fog per unit of geodesic distance: a surface at distance `d` keeps `exp(−density · d)`
    /// of its colour in flat space, the rest is the sky's haze in its direction (the gradient
    /// and the glows, without the discs and the stars).
    pub fog_density: Real,
    /// Number of light bands between shadow and full light.
    pub bands: u32,
    /// How far the bands melt into a smooth light, the sun's and the point lights' alike: 0
    /// painted in [`bands`](Self::bands), 1 smooth, a mix between.
    pub smooth: Real,
}

impl Default for Lighting {
    fn default() -> Self {
        Self {
            sun: Vector3::new(0.45, 0.75, 0.35),
            sun_color: Color::hex(0xfff3df),
            shadow_tint: Color::hex(0x8a7fb8),
            sky: Sky::default(),
            fog_density: 0.006,
            bands: 3,
            smooth: 0.0,
        }
    }
}

/// The sky: a gradient, two discs with their glows, and stars.
#[derive(Clone, Debug, PartialEq)]
pub struct Sky {
    /// Straight up.
    pub zenith: Color,
    /// At the horizon; also what the clear colour is.
    pub horizon: Color,
    /// Below the horizon, where nothing was drawn.
    pub below: Color,
    /// Exponent of the gradient from the horizon (0) to the zenith (1) along the sine of the
    /// elevation: below 1 the zenith colour comes down low, above 1 the horizon colour climbs.
    pub gradient: Real,
    /// Up, in the reference frame.
    pub up: Vector3<Real>,
    /// A sun.
    pub sun: SkyDisc,
    /// A moon.
    pub moon: SkyDisc,
    /// Brightness of the stars, 0 for none.
    pub stars: Real,
    /// Cells of the star grid per radian; about one cell in seven holds a star.
    pub star_density: Real,
    /// The axis the stars turn about, in the reference frame.
    pub star_axis: Vector3<Real>,
    /// How far they have turned about it, in radians.
    pub star_angle: Real,
}

impl Default for Sky {
    fn default() -> Self {
        Self {
            zenith: Color::hex(0xdfe2e6),
            horizon: Color::hex(0xf2ebdd),
            below: Color::hex(0xf2ebdd),
            gradient: 0.6,
            up: Vector3::y(),
            sun: SkyDisc::default(),
            moon: SkyDisc::default(),
            stars: 0.0,
            star_density: 220.0,
            star_axis: Vector3::y(),
            star_angle: 0.0,
        }
    }
}

/// A body in the [`Sky`] and the glow around it. Colours may be brighter than 1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkyDisc {
    /// Direction towards it, in the reference frame.
    pub direction: Vector3<Real>,
    /// What it looks like.
    pub shape: DiscShape,
    /// Angular radius in radians; 0 draws nothing.
    pub radius: Real,
    /// Width of the lines of a ring or a figure, in radians.
    pub stroke: Real,
    /// The disc.
    pub color: Color,
    /// Added around it, falling to `1/e` of this over [`glow_size`](Self::glow_size).
    pub glow: Color,
    /// Angle of the glow's falloff, in radians.
    pub glow_size: Real,
}

impl Default for SkyDisc {
    fn default() -> Self {
        Self {
            direction: Vector3::y(),
            shape: DiscShape::Disc,
            radius: 0.0,
            stroke: 0.005,
            color: Color::BLACK,
            glow: Color::BLACK,
            glow_size: 0.1,
        }
    }
}

/// Most bodies [`SkyDiscs`] draws.
pub const MAX_SKY_DISCS: usize = 24;

/// More bodies in the [`Sky`] than its sun and moon, as a resource: drawn as those are, behind
/// every surface, unlit, their glows in the sky and in what far surfaces fade into. The first
/// [`MAX_SKY_DISCS`] are drawn; none by default.
#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct SkyDiscs(pub Vec<SkyDisc>);

/// Shadows cast by the directional light of [`Lighting`], as a resource.
///
/// One square shadow map centred on the eye, its centre snapped to whole texels so shadows do
/// not crawl as the eye moves. It covers [`range`](Self::range) either side of the eye across
/// the light, and everything within [`depth`](Self::depth) of the eye along the light casts
/// into it, so a tall thing far up-sun still shades the ground near the eye. Shadows fade out
/// towards the map's edge. Surfaces in shadow take [`Lighting::shadow_tint`], like the sides
/// turned away from the light.
///
/// Flat space only: in a curved geometry nothing is cast and this is ignored.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct Shadows {
    /// Whether shadows are drawn.
    pub enabled: bool,
    /// Texels across the map; the map is made again when it changes.
    pub size: u32,
    /// Half-width of the map: shadows fall within this distance of the eye.
    pub range: Real,
    /// How far along the light, either side of the eye, casters are drawn.
    pub depth: Real,
    /// Radius of the soft edge, in texels.
    pub softness: Real,
    /// Share of the half-width from which shadows fade out towards the edge.
    pub fade: Real,
    /// How far a surface is moved along its normal before it is tested, in texels, so it does
    /// not shadow itself.
    pub normal_offset: Real,
    /// Whether both sides of every caster are drawn into the map, not only those turned away
    /// from the light. With only the far sides, a lit face meeting another solid (a wall at a
    /// wall's corner, under a ceiling) is tested against that solid's far side, at its own depth
    /// at the corner, and the normal offset and the soft edge carry it past: a strip along the
    /// corner comes out lit. With both, the near side shades it, a solid's thickness nearer the
    /// light. Turn it on where thin solids meet in the sun (rooms under an open sky); lit faces
    /// then cast onto themselves, and the normal offset keeps them clear (at 1.5 texels, with the
    /// casters' slope bias, whatever the light's angle).
    pub two_sided: bool,
}

impl Default for Shadows {
    fn default() -> Self {
        Self {
            enabled: false,
            size: 2048,
            range: 200.0,
            depth: 1000.0,
            softness: 1.5,
            fade: 0.8,
            normal_offset: 1.5,
            two_sided: false,
        }
    }
}

/// Anti-aliasing of the opaque pass, as a resource read once when the device opens.
///
/// Both targets get `samples` samples per pixel and the colour is resolved before the post
/// pass; the post pass reads depth sample 0. Counts the adapter does not support for both
/// targets fall back to 1 (no anti-aliasing) with a warning.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub struct Multisampling {
    /// Samples per pixel: 1, 2, 4 or 8.
    pub samples: u32,
}

impl Default for Multisampling {
    fn default() -> Self {
        Self { samples: 4 }
    }
}

/// Whether frames wait for the display's refresh, as a resource applied to the window's surface
/// whenever it changes. Setting `FK_NO_VSYNC` overrides it: frames are then presented as fast
/// as they are drawn, to measure frame rates.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub struct Vsync(pub bool);

impl Default for Vsync {
    fn default() -> Self {
        Self(true)
    }
}

impl Vsync {
    /// The surface's present mode for this, `FK_NO_VSYNC` taken into account.
    pub(crate) fn present_mode(self) -> wgpu::PresentMode {
        if self.0 && std::env::var_os("FK_NO_VSYNC").is_none() {
            wgpu::PresentMode::AutoVsync
        } else {
            wgpu::PresentMode::AutoNoVsync
        }
    }
}

/// The caller-filled part of the shader's world block, as a resource.
///
/// The engine does not know what the values mean: the game writes them and its shaders read
/// `world.values[i]` through accessors of their own. The block also carries `world.time`,
/// seconds since start, filled in by the engine.
#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct WorldUniforms {
    /// The values, four per slot.
    pub values: [[f32; 4]; WorldUniforms::SLOTS],
}

impl WorldUniforms {
    /// Number of `vec4` slots.
    pub const SLOTS: usize = 8;
}

/// What the post pass shows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum PostMode {
    /// The image, with the effects of [`PostSettings`].
    #[default]
    Image,
    /// The depth buffer decoded back to geodesic distance, as grey bands
    /// [`PostSettings::band_spacing`] apart, fading to black at the far surface. No effects.
    Distance,
}

/// A second image of the scene with only the entities marked [`OnLayer`](crate::OnLayer) in
/// it, over a flat colour: no sky, nothing else; as a resource. Drawn after the opaque pass
/// while [`enabled`](Self::enabled), into a target of the image's size the post pass's
/// `fk::grade` reads as `layer` (`fk::post`). The engine gives it no meaning.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct LayerRender {
    /// Whether it is drawn this frame. Off by default, and then it costs nothing.
    pub enabled: bool,
    /// What is behind the layer's entities, its alpha the image's alpha there: opaque (the
    /// default, white) for a flat colour; 0 lets whoever reads the layer lay what is drawn on it
    /// over something else, by the alpha.
    pub clear: Color,
}

impl Default for LayerRender {
    fn default() -> Self {
        Self {
            enabled: false,
            clear: Color::WHITE,
        }
    }
}

/// A frame kept for later, as a resource: the image as the post pass is about to read it (after
/// the [`ImageEffect`](crate::ImageEffect)s) is copied into a texture of its own whenever
/// [`capture`](Self::capture) changes, and stays there until the next capture; the post pass's
/// `fk::grade` reads it as `captured` (`fk::post`), black until the first. The engine gives it
/// no meaning.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameCapture {
    /// Change this number to capture the next frame's image.
    pub capture: u32,
}

/// The caller's values for the post pass's `fk::grade`, as a resource: `grade.values[i]` in
/// `fk::post`, opaque to the engine.
#[derive(Resource, Clone, Debug, PartialEq)]
pub struct GradeValues {
    /// The values, four per slot.
    pub values: [[f32; 4]; GradeValues::SLOTS],
}

impl GradeValues {
    /// Number of `vec4` slots.
    pub const SLOTS: usize = 32;
}

impl Default for GradeValues {
    fn default() -> Self {
        Self {
            values: [[0.0; 4]; Self::SLOTS],
        }
    }
}

/// The image settled over frames, against the shimmer of what is finer than a pixel: each frame
/// the ray-marched rays and the raster path's vertices go through another point of their pixels,
/// and a pass of its own, after the opaque pass and before the
/// [`ImageEffect`](crate::ImageEffect)s (drawn on the settled image), blends the image with
/// the last ones, found again where the eye's motion took them (by their distance, exactly
/// however the eye turned), so that each pixel comes out as the average over it. The engine
/// gives it no meaning.
///
/// The last images are dropped where something came in front or went, and held to the colours
/// round each pixel inside surfaces, and everywhere while the eye moves more than
/// [`still`](Self::still), so what moves leaves no trail; at edges while the eye is held still,
/// they are not, and even what is finer than a pixel settles.
/// A pixel a scene marched one tile in a square of them carried rather than marched
/// ([`RayMarched::interleave`](crate::RayMarched::interleave)) brings nothing new, and the last
/// images are kept there as they are.
/// Off ([`frames`](Self::frames) 1 or less, the default), that pass is skipped, every ray goes
/// through the middle of its pixel and the image is exactly as without it. On, it costs one
/// fullscreen pass and a copy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Settle {
    /// About how many frames the image settles over: each frame's share of it is one over this.
    pub frames: Real,
    /// How far the eye's motion may carry a pixel in a frame, in pixels, while the eye is still
    /// held still.
    pub still: Real,
    /// Each frame's share of the image while the eye moves, carried pixels too, 0 to 1: the
    /// more, the less the last images (found again between their pixels) soften it as it turns.
    pub moving: Real,
}

impl Default for Settle {
    fn default() -> Self {
        Self {
            frames: 0.0,
            still: 0.1,
            moving: 0.5,
        }
    }
}

/// A segmented bar drawn over the finished image, filling from the bottom: a readout for
/// debugging, not part of the image. It shows its value in whole segments.
///
/// Placed in units of the image height, from the image's bottom-left corner, so it keeps its
/// shape and place at any resolution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gauge {
    /// What it shows, in `[0, 1]`; rounded to the nearest whole segment.
    pub value: Real,
    /// Number of segments, at least 1.
    pub segments: u32,
    /// Its bottom-left corner.
    pub origin: Vector2<Real>,
    /// Its width and height.
    pub size: Vector2<Real>,
    /// The lit segments. Unlit ones are the same colour, dimmer.
    pub color: Color,
    /// How much it covers the image, 0 for not at all.
    pub opacity: Real,
}

impl Gauge {
    /// Number of gauges the post pass draws; [`PostSettings::gauges`] past it are ignored.
    pub const MAX: usize = 4;
}

impl Default for Gauge {
    fn default() -> Self {
        Self {
            value: 0.0,
            segments: 10,
            origin: Vector2::new(0.03, 0.03),
            size: Vector2::new(0.012, 0.2),
            color: Color::WHITE,
            opacity: 0.6,
        }
    }
}

/// Lines of text over the finished image, in a fixed 8×8 pixel font on a flat panel: a readout
/// for debugging, never part of the image. Printable ASCII only; anything else shows as `?`.
///
/// Placed in pixels from the image's top-left corner, and scaled by whole pixels so the font
/// stays crisp. Lines past [`MAX_LINES`](Self::MAX_LINES) and characters past
/// [`MAX_COLUMNS`](Self::MAX_COLUMNS) are cut.
#[derive(Clone, Debug, PartialEq)]
pub struct TextPanel {
    /// What it says, one entry per line. Empty, it is not drawn.
    pub lines: Vec<String>,
    /// Its top-left corner, in pixels from the image's top-left corner.
    pub origin: Vector2<Real>,
    /// Image pixels per font pixel, at least 1.
    pub scale: u32,
    /// The letters.
    pub color: Color,
    /// The panel behind them.
    pub background: Color,
    /// How much the panel covers the image, 0 for not at all.
    pub opacity: Real,
}

impl TextPanel {
    /// The most lines drawn.
    pub const MAX_LINES: usize = 48;
    /// The most characters drawn on a line.
    pub const MAX_COLUMNS: usize = 96;
}

impl Default for TextPanel {
    fn default() -> Self {
        Self {
            lines: Vec::new(),
            origin: Vector2::new(16.0, 16.0),
            scale: 2,
            color: Color::WHITE,
            background: Color::BLACK,
            opacity: 0.7,
        }
    }
}

/// Settings of the post pass, as a resource.
///
/// Its effects are composed in a fixed order: where the image is taken from (the caller's
/// `fk::grade`), blur, the caller's effects after it, saturation, the caller's effects after
/// it, vignette, the caller's effects over the image, over the opaque image (or what the
/// [`ImageEffect`](crate::ImageEffect)s made of it). The [`Gauge`]s are drawn over the result
/// in every [`PostMode`], and the [`TextPanel`] over everything, the
/// [`Overlay`](crate::Overlay) included.
#[derive(Resource, Clone, Debug, PartialEq)]
pub struct PostSettings {
    /// What it shows.
    pub mode: PostMode,
    /// Geodesic distance between two bands in [`PostMode::Distance`].
    pub band_spacing: Real,
    /// Blur radius in pixels, 0 for none.
    pub blur: Real,
    /// Colour saturation, 1 unchanged, 0 grey.
    pub saturation: Real,
    /// How much the corners darken, 0 for none.
    pub vignette: Real,
    /// Readouts over the image, at most [`Gauge::MAX`]. Empty in play.
    pub gauges: Vec<Gauge>,
    /// Text over the image, for debugging. Empty in play.
    pub text: TextPanel,
    /// The image settled over frames, off by default.
    pub settle: Settle,
}

impl Default for PostSettings {
    fn default() -> Self {
        Self {
            mode: PostMode::Image,
            band_spacing: 5.0,
            blur: 0.0,
            saturation: 1.0,
            vignette: 0.0,
            gauges: Vec::new(),
            text: TextPanel::default(),
            settle: Settle::default(),
        }
    }
}

/// What the last frame drew, as a resource.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RenderStats {
    /// Instances drawn.
    pub drawn: usize,
    /// Instances culled.
    pub culled: usize,
    /// Draw calls.
    pub draws: usize,
}

#[derive(ShaderType)]
pub(crate) struct ViewBlock {
    pub proj_scale: Vec2f,
    pub near: f32,
    pub far: f32,
    pub size: Vec2f,
    /// Where in its pixel this frame's rays and vertices go through, pixels from its middle.
    pub jitter: Vec2f,
    /// A linear form on embedded points in the eye's frame: the clipped raster pipelines drop
    /// what it takes below 0 ([`Elsewhere::clip`](crate::Elsewhere::clip)). Zero for none.
    pub clip: Vec4f,
}

#[derive(ShaderType)]
pub(crate) struct WorldBlock {
    pub time: f32,
    pub values: [Vec4f; WorldUniforms::SLOTS],
}

/// What a [`SkyDisc`] is drawn as, oriented on the sky (up is away from the horizon).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum DiscShape {
    /// A filled disc.
    #[default]
    Disc,
    /// A circle line, [`SkyDisc::stroke`] wide.
    Ring,
    /// Lines [`SkyDisc::stroke`] wide drawn by the caller's `fk::figure` ([`SkyFigure`](crate::SkyFigure)), which
    /// is given the body's radius and `params`; without one, a circle.
    Figure {
        /// The figure's own values.
        params: [f32; 4],
    },
}

#[derive(ShaderType)]
pub(crate) struct DiscBlock {
    pub direction: Vec4f,
    pub shape: Vec4f,
    pub color: Vec4f,
    pub glow: Vec4f,
    /// A figure's own values.
    pub figure: Vec4f,
}

#[derive(ShaderType)]
pub(crate) struct LightingBlock {
    pub sun_direction: Vec4f,
    pub sun_color: Vec4f,
    pub shadow_tint: Vec4f,
    pub sky_zenith: Vec4f,
    pub sky_horizon: Vec4f,
    pub sky_below: Vec4f,
    pub sky_up: Vec4f,
    /// The stars' axes in eye coordinates: a direction's star coordinates are its dot products
    /// with them.
    pub star_frame: [Vec4f; 3],
    pub sun_disc: DiscBlock,
    pub moon_disc: DiscBlock,
    pub fog_density: f32,
    pub bands: f32,
    pub stars: f32,
    pub star_density: f32,
    pub smoothness: f32,
    /// Columns of the matrix from eye coordinates to the shadow map's clip space.
    pub shadow_matrix: [Vec4f; 4],
    /// On (1) or off (0), one texel in uv, the soft edge in texels, the normal offset.
    pub shadow: Vec4f,
    /// Where shadows start fading out, as a share of the map's half-width.
    pub shadow_fade: Vec4f,
    /// The point lights nearest the eye; x of `point_count` says how many are used.
    pub point_lights: [PointLightBlock; crate::light::MAX_POINT_LIGHTS],
    pub point_count: Vec4f,
    /// [`SkyDiscs`]; x of `disc_count` says how many are used.
    pub discs: [DiscBlock; MAX_SKY_DISCS],
    pub disc_count: Vec4f,
    /// The [`Parting`](crate::Parting)'s cones; x of `part_count` says how many are used.
    pub parts: [PartBlock; crate::parting::MAX_PARTINGS],
    pub part_count: Vec4f,
}

#[derive(ShaderType, Clone, Copy, Default)]
pub(crate) struct PartBlock {
    /// The cone's axis in the eye's frame, and its half-angle.
    pub direction: Vec4f,
    /// How far its edge strays, the cells it strays by, how near nothing is cut.
    pub shape: Vec4f,
}

#[derive(ShaderType, Clone, Copy, Default)]
pub(crate) struct PointLightBlock {
    /// The embedded point, in the eye's frame.
    pub position: Vec4f,
    pub color: Vec4f,
    /// Intensity, range, and for a [`Spot`](crate::Spot) the cosines of its inner and outer
    /// half-angles.
    pub shape: Vec4f,
    /// A spot's axis, embedded at `position`, in the eye's frame; zero for a light shining every
    /// way.
    pub direction: Vec4f,
}

#[derive(ShaderType)]
pub(crate) struct PostBlock {
    pub mode: u32,
    pub band_spacing: f32,
    pub encode_srgb: u32,
    pub blur: f32,
    pub saturation: f32,
    pub vignette: f32,
    pub gauge_count: u32,
    pub gauges: [GaugeBlock; Gauge::MAX],
    /// The text panel's top-left corner in pixels, and its scale.
    pub text_layout: Vec4f,
    /// Its columns and lines, 0 when there is no text.
    pub text_size: Vec4f,
    pub text_color: Vec4f,
    /// The panel's colour, and its opacity.
    pub text_background: Vec4f,
}

#[derive(ShaderType)]
pub(crate) struct GradeBlock {
    pub values: [Vec4f; GradeValues::SLOTS],
}

#[derive(ShaderType)]
pub(crate) struct SettleBlock {
    /// Columns of the matrix taking this frame's eye coordinates to the last frame's.
    pub motion: [Vec4f; 4],
    /// Each frame's share, whether the last image is there, the motion held still, nothing.
    pub settings: Vec4f,
}

#[derive(ShaderType, Clone, Copy, Default)]
pub(crate) struct GaugeBlock {
    /// Bottom-left corner and size, in image heights.
    pub rect: Vec4f,
    /// Linear RGB, and the opacity.
    pub color: Vec4f,
    pub value: f32,
    pub segments: u32,
}

/// Serializes a block with its WGSL uniform layout (`encase`), for a uniform buffer of an
/// [`ImageEffect`](crate::ImageEffect) of the caller's.
///
/// # Panics
///
/// Never for a block of fixed size.
pub fn uniform_bytes<T: ShaderType + encase::internal::WriteInto>(block: &T) -> Vec<u8> {
    let mut buffer = encase::UniformBuffer::new(Vec::new());
    buffer
        .write(block)
        .expect("uniform blocks have a fixed size");
    buffer.into_inner()
}
