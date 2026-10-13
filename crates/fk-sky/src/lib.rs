//! The day and the night: a clock, the sun's and the moon's paths across the sky, and the
//! colours of every hour, written into `fk-render`'s [`Lighting`] every frame (a time of day, as
//! Bevy's atmosphere and sun plugins give).
//!
//! [`DaySkyPlugin`] adds the [`Clock`] and the tuning ([`DayNight`], a TOML
//! [`TuningFile`](fk_app::tuning::TuningFile): `daynight.toml`, the crate's own shipped with it,
//! or the app's through [`DaySkyPlugin::with`]). The clock may stand still
//! (`[clock] start_paused`), moved only by the app; run, it goes at a tuned speed. It takes
//! orders from [`ClockInput`], which the app fills from its key map: pause, scrub, faster,
//! slower, jump to a preset hour, or the next sunrise. Every frame the [`Lighting`] resource
//! gets:
//!
//! - the sun and the moon, each on its own arc ([`Path`]), so the day can be long and the night
//!   short;
//! - the colours of the hour (light, painted shadow, sky gradient, discs, glows, stars, fog),
//!   interpolated between [`Key`]s placed anywhere on the 24-hour circle;
//! - one light direction, blended from the sun by day to the moon by night and to straight up
//!   when neither is up, so the shading never jumps.
//!
//! Light bands are left alone: they belong to the scene, not the hour.

mod tuning;

use std::f64::consts::TAU;
use std::path::PathBuf;

use bevy_ecs::prelude::*;
use fk_app::{App, Plugin, PostUpdate, Time, Update};
use fk_math::Real;
use fk_math::nalgebra::Vector3;
use fk_render::{Color, DiscShape, Lighting, Sky, SkyDisc};

pub use fk_app::tuning::TuningFile;
pub use tuning::{BodyShape, ClockTuning, DayNight, Key, Path, SkyShape, Stars};

/// Hours in a day.
pub const DAY_HOURS: Real = 24.0;

/// The time of day, as a resource.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct Clock {
    /// Hour of the day in `[0, 24)`.
    pub hour: Real,
    /// How many times faster than the tuned day length time runs.
    pub speed: Real,
    /// Whether time stands still. Scrubbing and presets still move it.
    pub paused: bool,
}

impl Clock {
    /// The clock at the tuning's start hour, running at speed 1.
    pub fn new(tuning: &DayNight) -> Self {
        Self {
            hour: wrap_hour(tuning.clock.start_hour),
            speed: 1.0,
            paused: tuning.clock.start_paused,
        }
    }

    /// Advances by `dt` seconds of real time and applies `input`, which it empties.
    pub fn step(&mut self, input: &mut ClockInput, tuning: &ClockTuning, dt: Real) {
        let input = std::mem::take(input);
        if input.toggle_pause {
            self.paused = !self.paused;
        }
        if input.faster {
            self.speed = (self.speed * tuning.speed_step).min(tuning.max_speed);
        }
        if input.slower {
            self.speed = (self.speed / tuning.speed_step).max(1.0 / tuning.max_speed);
        }
        if let Some(speed) = input.test_speed.and_then(|i| tuning.test_speeds.get(i)) {
            self.speed = *speed;
            self.paused = false;
        }
        if let Some(hour) = input.preset.and_then(|i| tuning.presets.get(i)) {
            self.hour = *hour;
        }
        let mut hours = input.scrub * tuning.scrub_hours_per_second * dt;
        if !self.paused && tuning.day_seconds > 0.0 {
            hours += dt * self.speed * DAY_HOURS / tuning.day_seconds;
        }
        self.hour = wrap_hour(self.hour + hours);
    }
}

/// Orders for the [`Clock`], filled by the app and emptied by the clock each frame.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct ClockInput {
    /// Pause if running, run if paused.
    pub toggle_pause: bool,
    /// Scrub through time, `−1` backwards to `1` forwards, at
    /// [`ClockTuning::scrub_hours_per_second`].
    pub scrub: Real,
    /// Multiply the speed by [`ClockTuning::speed_step`].
    pub faster: bool,
    /// Divide the speed by it.
    pub slower: bool,
    /// Jump to [`ClockTuning::presets`]`[i]`.
    pub preset: Option<usize>,
    /// Run at [`ClockTuning::test_speeds`]`[i]`, for trying the cycle out; never bound in
    /// play.
    pub test_speed: Option<usize>,
    /// Jump to the next sunrise, the hour the sun's [`Path::rise`]s.
    pub sunrise: bool,
}

/// The day and the night, with the tuning from [`TuningFile::load`] or another one.
///
/// The sun and the moon may be drawn as figures ([`BodyShape::Figure`]) the app's
/// [`SkyFigure`](fk_render::SkyFigure) draws.
pub struct DaySkyPlugin {
    /// The constants.
    pub tuning: DayNight,
    /// A file to reload the tuning from whenever it is saved, instead of the edited copy in
    /// `FK_TUNING_DIR`.
    pub watch: Option<PathBuf>,
}

impl DaySkyPlugin {
    /// The app's own tuning, `shipped`, or its edited copy in `FK_TUNING_DIR`
    /// ([`fk_app::tuning::load_or`]).
    pub fn with(shipped: DayNight) -> Self {
        Self {
            tuning: fk_app::tuning::load_or(shipped),
            watch: None,
        }
    }

    /// The tuning in the file at `path`, reloaded whenever it is saved.
    pub fn watching(path: PathBuf) -> Self {
        Self {
            tuning: DayNight::load_from(&path),
            watch: Some(path),
        }
    }
}

impl Default for DaySkyPlugin {
    fn default() -> Self {
        Self {
            tuning: DayNight::load(),
            watch: None,
        }
    }
}

/// The day's systems.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DaySkySystems {
    /// Runs the clock, in [`Update`].
    Clock,
    /// Writes the [`Lighting`], in [`PostUpdate`].
    Light,
}

impl Plugin for DaySkyPlugin {
    fn build(&self, app: &mut App) {
        match &self.watch {
            Some(path) => fk_app::tuning::add_watching(app, self.tuning.clone(), path.clone()),
            None => fk_app::tuning::add(app, self.tuning.clone()),
        }
        app.insert_resource(Clock::new(&self.tuning))
            .init_resource::<ClockInput>()
            .add_systems(Update, run_clock.in_set(DaySkySystems::Clock))
            .add_systems(PostUpdate, light_the_hour.in_set(DaySkySystems::Light));
    }
}

fn run_clock(
    time: Res<Time>,
    tuning: Res<DayNight>,
    mut input: ResMut<ClockInput>,
    mut clock: ResMut<Clock>,
) {
    let ordered = *input != ClockInput::default();
    let sunrise = input.sunrise;
    clock.step(&mut input, &tuning.clock, time.delta());
    if sunrise {
        clock.hour = wrap_hour(tuning.sun.rise);
    }
    if ordered {
        let hour = clock.hour;
        tracing::info!(
            "time {:02}:{:02}, ×{:.3}{}",
            hour.floor() as u32,
            ((hour.fract()) * 60.0).floor() as u32,
            clock.speed,
            if clock.paused { ", paused" } else { "" }
        );
    }
}

fn light_the_hour(clock: Res<Clock>, tuning: Res<DayNight>, mut lighting: ResMut<Lighting>) {
    tuning.light(clock.hour, &mut lighting);
}

impl DayNight {
    /// Writes the light, the sky and the fog of `hour` into `lighting`, leaving its bands.
    pub fn light(&self, hour: Real, lighting: &mut Lighting) {
        let key = self.key_at(hour);
        let up = Vector3::y();
        let sun = self.sun.direction(hour);
        let moon = self.moon.direction(hour);
        lighting.sun = light_direction(sun, moon);
        lighting.sun_color = key.light;
        lighting.shadow_tint = key.shadow;
        lighting.fog_density = key.fog_density;
        let disc = |path: &Path, direction, color: Color, glow| SkyDisc {
            direction,
            shape: match path.shape {
                BodyShape::Disc => DiscShape::Disc,
                BodyShape::Ring => DiscShape::Ring,
                BodyShape::Figure => DiscShape::Figure {
                    params: path.figure,
                },
            },
            radius: path.radius.to_radians(),
            stroke: self.sky.stroke.to_radians(),
            color: color.scaled(path.disc_brightness as f32),
            glow,
            glow_size: path.glow_size.to_radians(),
        };
        lighting.sky = Sky {
            zenith: key.zenith,
            horizon: key.horizon,
            below: key.below,
            gradient: self.sky.gradient,
            up,
            sun: disc(&self.sun, sun, key.sun_disc, key.sun_glow),
            moon: disc(&self.moon, moon, key.moon_disc, key.moon_glow),
            stars: key.stars,
            star_density: self.stars.density,
            star_axis: Vector3::from(self.stars.pole),
            star_angle: TAU * self.stars.turns_per_day * hour / DAY_HOURS,
        };
    }

    /// The colours of `hour`, between the keys before and after it on the 24-hour circle.
    ///
    /// # Panics
    ///
    /// If there are no keys; the tuning tests check the shipped file has some.
    pub fn key_at(&self, hour: Real) -> Mixed {
        let hour = wrap_hour(hour);
        let mut keys: Vec<&Key> = self.key.iter().collect();
        keys.sort_by(|a, b| a.hour.total_cmp(&b.hour));
        let after = keys.iter().position(|key| key.hour > hour).unwrap_or(0);
        let before = (after + keys.len() - 1) % keys.len();
        let (a, b) = (keys[before], keys[after]);
        let span = (b.hour - a.hour).rem_euclid(DAY_HOURS);
        let t = if span > 0.0 {
            (hour - a.hour).rem_euclid(DAY_HOURS) / span
        } else {
            0.0
        };
        Mixed::between(a, b, t)
    }
}

/// The colours and amounts of one moment, mixed from two [`Key`]s.
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(missing_docs)]
pub struct Mixed {
    pub light: Color,
    pub shadow: Color,
    pub zenith: Color,
    pub horizon: Color,
    pub below: Color,
    pub sun_disc: Color,
    pub sun_glow: Color,
    pub moon_disc: Color,
    pub moon_glow: Color,
    pub stars: Real,
    pub fog_density: Real,
}

impl Mixed {
    fn between(a: &Key, b: &Key, t: Real) -> Self {
        let color = |a: u32, b: u32| Color::hex(a).mix(Color::hex(b), t as f32);
        let number = |a: Real, b: Real| a + (b - a) * t;
        Self {
            light: color(a.light, b.light),
            shadow: color(a.shadow, b.shadow),
            zenith: color(a.zenith, b.zenith),
            horizon: color(a.horizon, b.horizon),
            below: color(a.below, b.below),
            sun_disc: color(a.sun_disc, b.sun_disc),
            sun_glow: color(a.sun_glow, b.sun_glow),
            moon_disc: color(a.moon_disc, b.moon_disc),
            moon_glow: color(a.moon_glow, b.moon_glow),
            stars: number(a.stars, b.stars),
            fog_density: number(a.fog_density, b.fog_density),
        }
    }
}

impl Path {
    /// Direction towards the body at `hour`, in the reference frame: `y` up, azimuth 0 along
    /// `−z` and 90° along `+x`.
    ///
    /// While up (from `rise` to `set`, across midnight if need be) it climbs to
    /// `peak_elevation` halfway and turns half a circle from `rise_azimuth`; while down it
    /// sinks to `depth` below the horizon and turns the other half.
    pub fn direction(&self, hour: Real) -> Vector3<Real> {
        let up_hours = (self.set - self.rise).rem_euclid(DAY_HOURS);
        let since_rise = (hour - self.rise).rem_euclid(DAY_HOURS);
        let (elevation, azimuth) = if since_rise < up_hours {
            let s = since_rise / up_hours;
            (
                self.peak_elevation * (std::f64::consts::PI * s).sin(),
                self.rise_azimuth + 180.0 * s,
            )
        } else {
            let s = (since_rise - up_hours) / (DAY_HOURS - up_hours).max(1e-9);
            (
                -self.depth * (std::f64::consts::PI * s).sin(),
                self.rise_azimuth + 180.0 + 180.0 * s,
            )
        };
        let (elevation, azimuth) = (elevation.to_radians(), azimuth.to_radians());
        Vector3::new(
            elevation.cos() * azimuth.sin(),
            elevation.sin(),
            -elevation.cos() * azimuth.cos(),
        )
    }
}

/// One light for the whole scene: the sun while it is up or just under the horizon, the moon
/// while it is up and the sun is well down, straight up otherwise. Only one body at a time is
/// blended with the zenith (the sun and the moon can be on opposite sides of the sky), so the
/// light never swings across it.
fn light_direction(sun: Vector3<Real>, moon: Vector3<Real>) -> Vector3<Real> {
    let elevation = |v: Vector3<Real>| v.y.clamp(-1.0, 1.0).asin().to_degrees();
    let by_sun = smoothstep(-4.0, 4.0, elevation(sun));
    let by_moon =
        smoothstep(0.0, 8.0, elevation(moon)) * (1.0 - smoothstep(-12.0, -4.0, elevation(sun)));
    let by_sky = (1.0 - by_sun - by_moon).max(0.0);
    let blend = sun * by_sun + moon * by_moon + Vector3::y() * by_sky;
    if blend.norm() < 1e-6 {
        Vector3::y()
    } else {
        blend.normalize()
    }
}

use fk_math::smoothstep;

fn wrap_hour(hour: Real) -> Real {
    hour.rem_euclid(DAY_HOURS)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tuning() -> DayNight {
        DayNight::shipped()
    }

    fn elevation(v: Vector3<Real>) -> Real {
        v.y.asin().to_degrees()
    }

    #[test]
    fn shipped_tuning_parses_and_has_keys() {
        let tuning = tuning();
        assert!(!tuning.key.is_empty());
        assert!(DayNight::parse("nonsense = 1").is_err());
    }

    #[test]
    fn sun_is_highest_at_midday_and_down_at_night() {
        let sun = &tuning().sun;
        let noon = (sun.rise + (sun.set - sun.rise).rem_euclid(DAY_HOURS) / 2.0) % DAY_HOURS;
        assert!((elevation(sun.direction(noon)) - sun.peak_elevation).abs() < 1e-6);
        assert!(elevation(sun.direction(sun.rise)).abs() < 1e-6);
        assert!(elevation(sun.direction(sun.set)).abs() < 1e-6);
        assert!(elevation(sun.direction(sun.set + 1.0)) < 0.0);
        assert!(elevation(sun.direction(sun.rise - 1.0)) < 0.0);
    }

    #[test]
    fn paths_cross_midnight() {
        let moon = Path {
            shape: BodyShape::Ring,
            figure: [0.0; 4],
            rise: 20.0,
            set: 6.0,
            peak_elevation: 40.0,
            depth: 30.0,
            rise_azimuth: 0.0,
            radius: 1.0,
            glow_size: 5.0,
            disc_brightness: 1.0,
        };
        assert!((elevation(moon.direction(1.0)) - 40.0).abs() < 1e-6);
        assert!(elevation(moon.direction(13.0)) < -29.0);
    }

    #[test]
    fn keys_interpolate_round_the_circle() {
        let mut tuning = tuning();
        let key = |hour, light| Key {
            hour,
            light,
            ..tuning.key[0].clone()
        };
        tuning.key = vec![key(22.0, 0x000000), key(2.0, 0xffffff)];
        // Midnight is halfway from 22:00 to 02:00, across the wrap.
        let mid = tuning.key_at(0.0).light;
        let half = Color::hex(0x000000).mix(Color::hex(0xffffff), 0.5);
        assert!((mid.r - half.r).abs() < 1e-6, "{mid:?}");
        assert_eq!(tuning.key_at(22.0).light, Color::hex(0x000000));
        assert_eq!(tuning.key_at(2.0).light, Color::hex(0xffffff));
    }

    #[test]
    fn light_direction_never_jumps() {
        let tuning = tuning();
        let mut previous: Option<Vector3<Real>> = None;
        // One step is 15 s of the day; at sunrise the light turns a quarter circle in about
        // half an hour, a little over a degree per step. A jump would be tens of degrees.
        let steps = 24 * 60 * 4;
        for i in 0..=steps {
            let hour = DAY_HOURS * Real::from(i) / Real::from(steps);
            let light = light_direction(tuning.sun.direction(hour), tuning.moon.direction(hour));
            assert!((light.norm() - 1.0).abs() < 1e-9, "{hour}");
            if let Some(previous) = previous {
                let turn = previous.dot(&light).clamp(-1.0, 1.0).acos().to_degrees();
                assert!(turn < 2.0, "light turns {turn}° at {hour:.3} h");
            }
            previous = Some(light);
        }
    }

    #[test]
    fn a_sunrise_order_jumps_to_the_rise() {
        let mut app = App::new();
        app.add_plugin(DaySkyPlugin {
            tuning: tuning(),
            watch: None,
        })
        .init_resource::<Lighting>();
        app.world_mut().resource_mut::<ClockInput>().sunrise = true;
        app.update(0.0);
        let hour = app.world().resource::<Clock>().hour;
        assert_eq!(hour, tuning().sun.rise);
        assert_eq!(*app.world().resource::<ClockInput>(), ClockInput::default());
    }

    #[test]
    fn clock_runs_pauses_scrubs_and_jumps() {
        let tuning = tuning();
        let c = &tuning.clock;
        let mut clock = Clock {
            hour: 10.0,
            speed: 1.0,
            paused: false,
        };
        let mut input = ClockInput::default();
        clock.step(&mut input, c, c.day_seconds / DAY_HOURS);
        assert!((clock.hour - 11.0).abs() < 1e-9);

        input.toggle_pause = true;
        clock.step(&mut input, c, 100.0);
        assert!(clock.paused);
        assert!((clock.hour - 11.0).abs() < 1e-9, "a paused clock stays");
        assert_eq!(input, ClockInput::default(), "orders are used up");

        input.scrub = -1.0;
        clock.step(&mut input, c, 1.0);
        assert!((clock.hour - (11.0 - c.scrub_hours_per_second)).abs() < 1e-9);

        input.preset = Some(0);
        clock.step(&mut input, c, 0.0);
        assert_eq!(clock.hour, c.presets[0]);

        input.faster = true;
        clock.step(&mut input, c, 0.0);
        assert_eq!(clock.speed, c.speed_step);

        clock.hour = 23.5;
        clock.paused = false;
        clock.speed = 1.0;
        clock.step(&mut input, c, c.day_seconds / DAY_HOURS);
        assert!((clock.hour - 0.5).abs() < 1e-9, "wraps past midnight");
    }
}
