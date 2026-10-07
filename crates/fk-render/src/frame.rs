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
    /// Width of the lines of a ring or a star, in radians.
    pub stroke: Real,
    /// The disc.
    pub color: Color,
    /// Added around it, falling to `1/e` of this over [`glow_size`](Self::glow_size).
    pub glow: Color,
    /// Angle of the glow's falloff, in radians.
    pub glow_size: Real,
    /// How much of it shows over the surfaces in front of it, 0 none (it is behind them, in
    /// the sky), 1 all of it: drawn over the whole scene, under the [`Lids`].
    pub over: Real,
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
            over: 0.0,
        }
    }
}

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
/// The engine does not know what the values mean: the game writes them (breath fill, eyelid
/// closure, sink…) and its shaders read `world.values[i]` through accessors of their own. The
/// block also carries `world.time`, seconds since start, filled in by the engine.
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

/// Two lids closing over the image from the top and bottom edges.
///
/// The opening between them is almond-shaped: its half-height at a column is proportional to
/// `1 − curve · s²`, with `s` from −1 at the left edge to 1 at the right, so the lids reach the
/// corners first and meet in the middle last.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lids {
    /// 0 open, 1 shut.
    pub closure: Real,
    /// What they are.
    pub color: Color,
    /// Width of their soft edge, as a share of the image height.
    pub softness: Real,
    /// How much the edges arc, in `[0, 1)`: 0 is straight.
    pub curve: Real,
    /// How much darker the lids get towards the image's edges and along their own rims, where
    /// they meet, as an ambient-occlusion look: 0 for none, 1 for black there.
    pub occlusion: Real,
    /// How much the inside of the lids shows the [`LayerRender`] instead of their colour and
    /// the afterimage: 0 none, 1 only the layer. As much, the lids themselves are not seen: the
    /// whole image gives way to the layer evenly as they close (by [`closure`](Self::closure)),
    /// without their edges or their darkening, and the layer is seen through the same lenses
    /// as the scene ([`PostSettings::warps`](crate::PostSettings::warps)). The layer is drawn
    /// only while this and [`closure`](Self::closure) are above 0.
    pub layer: Real,
}

impl Default for Lids {
    fn default() -> Self {
        Self {
            closure: 0.0,
            color: Color::BLACK,
            softness: 0.15,
            curve: 0.45,
            occlusion: 0.0,
            layer: 0.0,
        }
    }
}

/// A second image of the scene with only the entities marked [`OnLayer`](crate::OnLayer) in
/// it, over a flat colour: no sky, nothing else. The [`Lids`] show it inside them, as much as
/// [`Lids::layer`] says; it is drawn only then.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerRender {
    /// What is behind the layer's entities.
    pub clear: Color,
}

impl Default for LayerRender {
    fn default() -> Self {
        Self {
            clear: Color::WHITE,
        }
    }
}

/// An inked look over the image, after the comics of Moebius: dark lines where the depth or the
/// colour jumps (silhouettes, the horizon, the edges of the bands of light), cross-hatching in
/// the shadows, a paper tone under it all. The engine gives it no meaning.
///
/// It is laid on in a pass of its own, after the opaque pass and before the [`Mosh`], so a
/// datamosh carries it along with the rest of the image and the post effects (lenses, blur,
/// saturation) bend and drain it too. Off ([`amount`](Self::amount) 0, the default), that pass
/// is skipped and the image is exactly as without it. On, it costs one fullscreen pass and a
/// copy, eight more reads of the colour and depth targets per pixel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ink {
    /// How much of it, 0 none, 1 all.
    pub amount: Real,
    /// The lines' colour.
    pub color: Color,
    /// How far apart, in pixels, the neighbours are that a line is found against: its width.
    pub width: Real,
    /// A line where the distance to the next pixel jumps by this share of the distance.
    pub depth: Real,
    /// A line where the brightness (sRGB) jumps by this much from one pixel to the next.
    pub edges: Real,
    /// Lines are gone by this distance, in the fog, in units of distance.
    pub fade: Real,
    /// How dark the hatching is where it is, 0 none, 1 the lines' colour.
    pub hatch: Real,
    /// Pixels between hatching lines.
    pub spacing: Real,
    /// Below this brightness (sRGB) the shadows are hatched, crossed twice as dark.
    pub shadow: Real,
    /// The paper's tone the image is laid on, and how much of it.
    pub paper: Color,
    /// See [`paper`](Self::paper).
    pub paper_amount: Real,
}

impl Default for Ink {
    fn default() -> Self {
        Self {
            amount: 0.0,
            color: Color::hex(0x1e1a24),
            width: 1.0,
            depth: 0.08,
            edges: 0.12,
            fade: 600.0,
            hatch: 0.5,
            spacing: 5.0,
            shadow: 0.35,
            paper: Color::hex(0xf4ead2),
            paper_amount: 0.15,
        }
    }
}

/// A lens over the image: around a point, the image is drawn from further out (pulled in
/// towards the point) and turned about it, the more so nearer the point, fading to nothing at
/// [`radius`](Self::radius). The engine gives it no meaning.
///
/// Placed in units of the image height from the image's centre, y up, so it keeps its shape at
/// any resolution. Applied to the opaque image (or the [`Mosh`]) before every other effect.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Warp {
    /// The point the image bends around.
    pub centre: Vector2<Real>,
    /// How far from it the image bends; 0 for no lens.
    pub radius: Real,
    /// How much it pulls the image in: a pixel halfway out shows what is `1 + strength / 4`
    /// times as far out. Negative pushes the image out.
    pub strength: Real,
    /// How far the image turns about the centre at its strongest, in radians, positive
    /// anticlockwise.
    pub twist: Real,
}

impl Warp {
    /// Number of lenses the post pass draws; [`PostSettings::warps`] past it are ignored.
    pub const MAX: usize = 4;
}

impl Default for Warp {
    fn default() -> Self {
        Self {
            centre: Vector2::zeros(),
            radius: 0.0,
            strength: 0.0,
            twist: 0.0,
        }
    }
}

/// A dark closing in from the edges towards the centre.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tunnel {
    /// 0 none, 1 the whole image.
    pub amount: Real,
    /// What it is.
    pub color: Color,
    /// Width of its soft edge, as a share of the half-diagonal.
    pub softness: Real,
}

impl Default for Tunnel {
    fn default() -> Self {
        Self {
            amount: 0.0,
            color: Color::BLACK,
            softness: 0.3,
        }
    }
}

/// A figure fixed on the image, at its centre unless placed elsewhere, drawn over everything but
/// the [`Gauge`]s, the [`Lids`] included: lines in the shape of a [`SkyDisc`] (a star's first
/// point up), a glow around them, and a flash of its colour over the whole image. The engine
/// gives it no meaning; a game burns a bright light's print into the eyes with it.
///
/// Sizes and the centre are in image heights (the centre from the image's centre, y up, like a
/// [`Warp`]), so it keeps its shape and place at any resolution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mark {
    /// Where it is.
    pub centre: Vector2<Real>,
    /// What it looks like.
    pub shape: DiscShape,
    /// Its radius; 0 draws no figure.
    pub radius: Real,
    /// Width of the lines of a ring or a star.
    pub stroke: Real,
    /// The figure's colour. May be brighter than 1.
    pub color: Color,
    /// Added around the figure, falling to `1/e` of this over [`glow_size`](Self::glow_size).
    pub glow: Color,
    /// Distance of the glow's falloff.
    pub glow_size: Real,
    /// How much the figure and its glow show, 0 for not at all.
    pub opacity: Real,
    /// How much of the whole image is washed in [`color`](Self::color), 0 none, 1 all of it.
    pub flash: Real,
    /// How far the figure is turned about its centre, radians, counter-clockwise (a star's
    /// points).
    pub turn: Real,
    /// A ring drawn round the figure, its radius this many times the figure's, in the same line
    /// width: 0 for none.
    pub ring: Real,
    /// The ring's colour. May be brighter than 1.
    pub ring_color: Color,
    /// The ring's glow, added on top of the figure's.
    pub ring_glow: Color,
    /// Distance of the ring's glow's falloff.
    pub ring_glow_size: Real,
    /// Drawn as the [`Sky`] draws a [`SkyDisc`]: the glow falls off from the centre rather than
    /// from the lines, and lies under the figure rather than over it, so a disc burnt into the
    /// eyes looks just as it did in the sky.
    pub as_sky: bool,
}

impl Default for Mark {
    fn default() -> Self {
        Self {
            centre: Vector2::zeros(),
            shape: DiscShape::Disc,
            radius: 0.0,
            stroke: 0.004,
            color: Color::WHITE,
            glow: Color::BLACK,
            glow_size: 0.05,
            opacity: 0.0,
            flash: 0.0,
            turn: 0.0,
            ring: 0.0,
            ring_color: Color::WHITE,
            ring_glow: Color::BLACK,
            ring_glow_size: 0.05,
            as_sky: false,
        }
    }
}

/// The last frame seen, captured on request and shown behind the [`Lids`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Afterimage {
    /// Change this number to capture the next frame's image.
    pub capture: u32,
    /// How much of the captured image shows through the lids, 0 for none.
    pub strength: Real,
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

/// A datamosh, like a video whose key frames were lost: one frame is held, and from then on
/// only the live image's motion reaches it. Every frame each block of the held image moves the
/// way the camera's motion moves the live image there (reprojected from the depth buffer, one
/// vector per block), the live image's light and dark bleed into its colours, and blocks of the
/// live image break back in, region by region, until only the live image is left.
///
/// It works on the opaque image, before every other effect of [`PostSettings`].
///
/// [`MoshKind::Pixels`] has no blocks: every pixel of the held image moves by the live image's
/// own motion there, and the live image comes in smoothly, in soft regions, until it has
/// replaced the held one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mosh {
    /// Blocks or pixels.
    pub kind: MoshKind,
    /// Change this number to start a mosh, holding this frame's image.
    pub start: u32,
    /// Change this number on a frame where the camera jumps (a teleport), so the jump is not
    /// read as motion.
    pub cut: u32,
    /// How far along: 0 is the held image, and every block is the live image again before 1.
    /// At 1 or more the mosh is over and costs nothing.
    pub progress: Real,
    /// Size of a block, as a share of the image height; with [`MoshKind::Pixels`], the size of
    /// the soft regions the live image comes in by.
    pub block: Real,
    /// How fast the live image's edges bleed into the held colours, per second at `progress`
    /// 1, in proportion to it before. Only the edges: the held blocks keep their own
    /// brightness.
    pub bleed: Real,
}

/// How a [`Mosh`] moves the held image and lets the live one back in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum MoshKind {
    /// One motion vector per block, and the live image breaking back in block by block.
    #[default]
    Blocks,
    /// Every pixel moved by its own motion, and the live image coming back in smoothly.
    Pixels,
    /// Macroblocks torn apart: each dragged by the motion at its centre, rounded its own way,
    /// a few at a time thrown by a corrupt vector (more of them the more the image moves); the
    /// live image's edges bled in, in full colour, the faster the more it moves, so whatever
    /// moves leaves trails in the held colours; blocks snapping back to the live image one by
    /// one, in ragged regions. `block` is the macroblock.
    Smear,
}

impl Default for Mosh {
    fn default() -> Self {
        Self {
            kind: MoshKind::Blocks,
            start: 0,
            cut: 0,
            progress: 1.0,
            block: 1.0 / 40.0,
            bleed: 1.5,
        }
    }
}

/// A patch of the image that datamoshes without end: inside a disc, every block of the image
/// is the last frame's, moved the way the camera's motion moved the live image there and
/// drifting along a slow swirl outwards from the centre, fading back into the live image
/// smoothly, everywhere and towards its edge. Whatever is drawn in it smears and leaks out of
/// its outline in blocks. The engine gives it no meaning.
///
/// Placed in units of the image height from the image's centre, y up, like a [`Warp`]. Its edge
/// is ragged, block by block, and crawls slowly. It works on the opaque image, like the [`Mosh`], and not while a
/// [`Mosh`] runs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoshPatch {
    /// The disc's centre.
    pub centre: Vector2<Real>,
    /// Its radius; 0 for no patch.
    pub radius: Real,
    /// Size of a block, as a share of the image height.
    pub block: Real,
    /// How fast the blocks drift along the swirl, in image heights per second.
    pub flow: Real,
    /// How fast the held image fades back into the live one, per second: at `r`, a smear
    /// keeps `exp(−r)` of itself after a second.
    pub refresh: Real,
    /// How fast the live image's edges bleed into the held colours, per second.
    pub bleed: Real,
    /// Arms reaching out of it, evenly round it: tentacles of the same mosh. 0 for none, a
    /// disc.
    pub arms: u32,
    /// How far past its radius the arms reach, in image heights.
    pub reach: Real,
    /// How wide an arm is at its root, in image heights; it tapers to nothing at its tip.
    pub arm_width: Real,
    /// How far an arm curls along its length, radians at its tip.
    pub curl: Real,
    /// The arms' turn about the centre, radians; the app turns them.
    pub turn: Real,
    /// How far the arms undulate, radians either way, waves running out along them.
    pub wave: Real,
}

impl MoshPatch {
    /// Number of patches the mosh pass draws; [`PostSettings::mosh_patches`] past it are
    /// ignored.
    pub const MAX: usize = 4;
}

impl Default for MoshPatch {
    fn default() -> Self {
        Self {
            centre: Vector2::zeros(),
            radius: 0.0,
            block: 1.0 / 60.0,
            flow: 0.03,
            refresh: 1.5,
            bleed: 1.0,
            arms: 0,
            reach: 0.0,
            arm_width: 0.0,
            curl: 0.0,
            turn: 0.0,
            wave: 0.0,
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
/// The engine gives the effects no meaning: a game decides that the [`Lids`] are eyelids and
/// the [`Tunnel`] is fainting. They are composed in a fixed order: the [`Warp`]s, blur and streaks,
/// saturation, posterize, vignette, tunnel, lids (over the [`LayerRender`] when they show it),
/// the [`Mark`], over the opaque image or the [`Mosh`] made of it. The [`Gauge`]s are drawn
/// over the result in every [`PostMode`], and the [`TextPanel`] over everything, the
/// [`Overlay`](crate::Overlay) included.
#[derive(Resource, Clone, Debug, PartialEq)]
pub struct PostSettings {
    /// What it shows.
    pub mode: PostMode,
    /// Geodesic distance between two bands in [`PostMode::Distance`].
    pub band_spacing: Real,
    /// Blur radius in pixels, 0 for none.
    pub blur: Real,
    /// The image smeared along lines out of [`streak_centre`](Self::streak_centre), as by
    /// speed: each pixel averages the image from itself in towards the centre over this share
    /// of the way, 0 for none. The centre itself stays sharp.
    pub streak: Real,
    /// Where the streaks come out of, in image heights from the image's centre, y up.
    pub streak_centre: Vector2<Real>,
    /// Colour saturation, 1 unchanged, 0 grey.
    pub saturation: Real,
    /// How far the colours are swapped, 0 unchanged, 1 each turned to its opposite hue at its
    /// own brightness (reflected through its grey).
    pub swap: Real,
    /// How much the corners darken, 0 for none.
    pub vignette: Real,
    /// Steps of brightness, evenly spaced in sRGB, 0 for none: flat bands of paint with every
    /// colour keeping its hue. Only what was drawn is posterized: the sky behind it, the
    /// vignette, the tunnel and the lids stay smooth.
    pub posterize: u32,
    /// The dark closing in from the edges.
    pub tunnel: Tunnel,
    /// The lids.
    pub lids: Lids,
    /// The afterimage behind the lids.
    pub afterimage: Afterimage,
    /// The datamosh, in place of the opaque image while it lasts.
    pub mosh: Mosh,
    /// Patches of the image that datamosh without end, at most [`MoshPatch::MAX`]; none while
    /// the [`mosh`](Self::mosh) runs.
    pub mosh_patches: Vec<MoshPatch>,
    /// Readouts over the image, at most [`Gauge::MAX`]. Empty in play.
    pub gauges: Vec<Gauge>,
    /// Text over the image, for debugging. Empty in play.
    pub text: TextPanel,
    /// The figure at the centre, over everything else in the image.
    pub mark: Mark,
    /// Lenses over the image, at most [`Warp::MAX`].
    pub warps: Vec<Warp>,
    /// What the lids can show inside them.
    pub layer: LayerRender,
    /// The inked look, off by default.
    pub ink: Ink,
}

impl Default for PostSettings {
    fn default() -> Self {
        Self {
            mode: PostMode::Image,
            band_spacing: 5.0,
            blur: 0.0,
            streak: 0.0,
            streak_centre: Vector2::zeros(),
            saturation: 1.0,
            swap: 0.0,
            vignette: 0.0,
            posterize: 0,
            tunnel: Tunnel::default(),
            lids: Lids::default(),
            afterimage: Afterimage::default(),
            mosh: Mosh::default(),
            mosh_patches: Vec::new(),
            gauges: Vec::new(),
            text: TextPanel::default(),
            mark: Mark::default(),
            warps: Vec::new(),
            layer: LayerRender::default(),
            ink: Ink::default(),
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
}

#[derive(ShaderType)]
pub(crate) struct WorldBlock {
    pub time: f32,
    pub values: [Vec4f; WorldUniforms::SLOTS],
}

/// The figure a [`SkyDisc`] is drawn as, oriented on the sky: a star's first point is the one
/// furthest from the horizon.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DiscShape {
    /// A filled disc.
    #[default]
    Disc,
    /// A circle line, [`SkyDisc::stroke`] wide.
    Ring,
    /// A star drawn in lines [`SkyDisc::stroke`] wide: `points` points on a circle of
    /// [`SkyDisc::radius`], joined by concave arcs that leave each point along the radius, so
    /// neighbouring arcs meet in cusps.
    Star {
        /// How many.
        points: u32,
    },
}

#[derive(ShaderType)]
pub(crate) struct DiscBlock {
    pub direction: Vec4f,
    pub shape: Vec4f,
    pub color: Vec4f,
    pub glow: Vec4f,
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
    /// Columns of the matrix from eye coordinates to the shadow map's clip space.
    pub shadow_matrix: [Vec4f; 4],
    /// On (1) or off (0), one texel in uv, the soft edge in texels, the normal offset.
    pub shadow: Vec4f,
    /// Where shadows start fading out, as a share of the map's half-width.
    pub shadow_fade: Vec4f,
    /// The point lights nearest the eye; x of `point_count` says how many are used.
    pub point_lights: [PointLightBlock; crate::light::MAX_POINT_LIGHTS],
    pub point_count: Vec4f,
}

#[derive(ShaderType, Clone, Copy, Default)]
pub(crate) struct PointLightBlock {
    /// The embedded point, in the eye's frame.
    pub position: Vec4f,
    pub color: Vec4f,
    /// Intensity, range.
    pub shape: Vec4f,
}

#[derive(ShaderType)]
pub(crate) struct PostBlock {
    pub mode: u32,
    pub band_spacing: f32,
    pub encode_srgb: u32,
    pub blur: f32,
    pub lid_color: Vec4f,
    pub lid_closure: f32,
    pub lid_softness: f32,
    pub tunnel: f32,
    pub tunnel_softness: f32,
    pub tunnel_color: Vec4f,
    pub saturation: f32,
    pub swap: f32,
    pub vignette: f32,
    pub posterize: f32,
    pub afterimage: f32,
    pub lid_curve: f32,
    pub lid_occlusion: f32,
    pub gauge_count: u32,
    pub gauges: [GaugeBlock; Gauge::MAX],
    pub mark_shape: Vec4f,
    pub mark_color: Vec4f,
    pub mark_glow: Vec4f,
    pub mark_flash: f32,
    pub lid_layer: f32,
    pub warp_count: u32,
    /// Centre, radius and strength of each lens.
    pub warps: [Vec4f; Warp::MAX],
    /// The lenses' twists.
    pub warp_twists: Vec4f,
    /// The mark's centre, in image heights from the image's centre, y up; its turn; its ring's
    /// radius as a share of its own.
    pub mark_centre: Vec4f,
    /// The text panel's top-left corner in pixels, and its scale.
    pub text_layout: Vec4f,
    /// Its columns and lines, 0 when there is no text.
    pub text_size: Vec4f,
    pub text_color: Vec4f,
    /// The panel's colour, and its opacity.
    pub text_background: Vec4f,
    /// The mark's ring's colour.
    pub mark_ring_color: Vec4f,
    /// Its glow's colour, and the distance it falls to 1/e over.
    pub mark_ring_glow: Vec4f,
    /// The streaks' centre (xy, image heights from the centre, y up) and share (z).
    pub streak: Vec4f,
    /// 1 when the mark is drawn as the sky draws its discs, 0 otherwise.
    pub mark_as_sky: f32,
}

#[derive(ShaderType)]
pub(crate) struct InkBlock {
    /// How much, its width in pixels, depth and brightness thresholds.
    pub ink: Vec4f,
    /// The ink's colour, and the hatching's darkness.
    pub color: Vec4f,
    /// The fade distance, the hatching's spacing in pixels, its shadow threshold, the paper's
    /// amount.
    pub more: Vec4f,
    /// The paper's tone.
    pub paper: Vec4f,
}

#[derive(ShaderType)]
pub(crate) struct MoshBlock {
    /// Columns of the matrix taking this frame's eye coordinates to the last frame's.
    pub motion: [Vec4f; 4],
    pub progress: f32,
    /// In pixels.
    pub block: f32,
    pub bleed: f32,
    /// Seconds since the last frame.
    pub delta: f32,
    /// Differs from one mosh to the next.
    pub seed: f32,
    pub patch_count: u32,
    /// Each patch's centre and radius in pixels from the top-left corner, and its block in
    /// pixels.
    pub patches: [Vec4f; MoshPatch::MAX],
    /// Each patch's flow in pixels per second, refresh and bleed per second.
    pub patch_motion: [Vec4f; MoshPatch::MAX],
    /// Each patch's arms: how many, their reach and root width in pixels, their curl.
    pub patch_arms: [Vec4f; MoshPatch::MAX],
    /// Each patch's arms' turn and wave, radians.
    pub patch_turn: [Vec4f; MoshPatch::MAX],
    /// 0 blocks, 1 pixels.
    pub kind: u32,
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

/// Serializes a block with its WGSL uniform layout.
pub(crate) fn uniform_bytes<T: ShaderType + encase::internal::WriteInto>(block: &T) -> Vec<u8> {
    let mut buffer = encase::UniformBuffer::new(Vec::new());
    buffer
        .write(block)
        .expect("uniform blocks have a fixed size");
    buffer.into_inner()
}
