use bevy_ecs::resource::Resource;
use fk_app::tuning::TuningFile;
use fk_math::Real;
use serde::Deserialize;

/// Every constant of the day and the night, read from `daynight.toml` (the crate's own documents
/// each value).
///
/// Hours are in `[0, 24)`, angles in degrees, colours sRGB `0xRRGGBB`. The file documents
/// each value.
#[derive(Resource, Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DayNight {
    /// How time runs and how it is controlled.
    pub clock: ClockTuning,
    /// The sun's arc.
    pub sun: Path,
    /// The moon's arc.
    pub moon: Path,
    /// The stars.
    pub stars: Stars,
    /// The shape of the sky's gradient.
    pub sky: SkyShape,
    /// The colours of the hours.
    pub key: Vec<Key>,
}

impl TuningFile for DayNight {
    const FILE: &'static str = "daynight.toml";
    const SHIPPED: &'static str = include_str!("../daynight.toml");
}

/// See [`DayNight::clock`].
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
#[allow(missing_docs)]
pub struct ClockTuning {
    pub start_hour: Real,
    pub start_paused: bool,
    pub day_seconds: Real,
    pub speed_step: Real,
    pub max_speed: Real,
    pub scrub_hours_per_second: Real,
    pub presets: Vec<Real>,
    pub test_speeds: Vec<Real>,
}

/// The arc of a body across the sky: see [`Path::direction`].
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
#[allow(missing_docs)]
pub struct Path {
    pub shape: BodyShape,
    /// A figure's own values, passed to the sky's `fk::figure` hook.
    #[serde(default)]
    pub figure: [f32; 4],
    pub rise: Real,
    pub set: Real,
    pub peak_elevation: Real,
    pub depth: Real,
    pub rise_azimuth: Real,
    pub radius: Real,
    pub glow_size: Real,
    pub disc_brightness: Real,
}

/// How a body is drawn: see [`Path`] and `daynight.toml`.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum BodyShape {
    /// A filled disc.
    Disc,
    /// A thin circle.
    Ring,
    /// Lines drawn by the app's `fk::figure` hook (`fk_render::SkyFigure`), given
    /// [`Path::figure`].
    Figure,
}

/// See [`DayNight::stars`].
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
#[allow(missing_docs)]
pub struct Stars {
    pub density: Real,
    pub pole: [Real; 3],
    pub turns_per_day: Real,
}

/// See [`DayNight::sky`].
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
#[allow(missing_docs)]
pub struct SkyShape {
    pub gradient: Real,
    pub stroke: Real,
}

/// The colours of one hour. Between two keys every value is mixed linearly.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
#[allow(missing_docs)]
pub struct Key {
    pub hour: Real,
    pub light: u32,
    pub shadow: u32,
    pub zenith: u32,
    pub horizon: u32,
    pub below: u32,
    pub sun_disc: u32,
    pub sun_glow: u32,
    pub moon_disc: u32,
    pub moon_glow: u32,
    pub stars: Real,
    pub fog_density: Real,
}
