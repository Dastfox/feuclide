//! Sound: decoded clips played as voices whose gain and rate glide.
//!
//! [`AudioPlugin`] opens the default output device as the [`Audio`] resource. Without one (a
//! server, CI) the resource is silent: every voice plays nothing and reports itself stopped, and
//! the app runs the same. A [`Sound`] is a clip decoded from bytes (MP3 or FLAC, usually
//! `include_bytes!`) or made from samples ([`Sound::from_samples`]), which can be cut
//! ([`Sound::slice`]); [`Audio::play`] starts it, once or
//! looped, as a [`Voice`] that can be faded, sped up and stopped.
//!
//! A sound plays either straight out or in the room ([`Playback::room`]): through a reverb set by
//! the [`Acoustics`] resource, which the app changes to change the room (each place its own) and
//! [`AudioPlugin`] applies, gliding over [`Acoustics::fade`]. The engine gives the room no meaning.
//!
//! Or it plays outside ([`Playback::outside`], and everything an [`Emitter`] plays): through a
//! low-pass filter and a gain set by the [`Outside`] resource, to muffle or hush everything out
//! there at once. An [`Emitter`] places its sounds about the ear: its position is relative to
//! the listener, x to the right, y up, −z ahead, so that the app gives it in whatever frame its
//! eye has (camera-relative, as at the GPU boundary); they pan between the ears and fade with
//! distance.
//!
//! Every sound plays in a [`Channel`], channel 0 unless its [`Playback::channel`] says otherwise
//! (an [`Emitter`]'s always in channel 0), each with a room and an outside of its own; the
//! [`Volumes`] resource sets how loud each channel is, and everything over all. The engine gives
//! the channels no meaning: a game sorts its sounds into them (its world, its player's body)
//! for the player to set apart.
//!
//! Gains are amplitudes: 1 unchanged, 0 silent, above 1 louder ([`decibels`] converts). Rates are
//! playback speeds: 1 unchanged, and the pitch follows. Fades are in seconds. The backend is
//! `kira`; nothing of it shows through this API.

use std::io::Cursor;
use std::sync::Mutex;
use std::time::Duration;

use bevy_ecs::prelude::*;
use fk_app::{App, Plugin, Update};
use fk_math::Real;
use fk_math::nalgebra::Vector3;
use kira::effect::filter::{FilterBuilder, FilterHandle};
use kira::effect::reverb::{ReverbBuilder, ReverbHandle};
use kira::listener::ListenerHandle;
use kira::sound::static_sound::{StaticSoundData, StaticSoundHandle};
use kira::sound::{FromFileError, PlaybackState};
use kira::track::{
    SpatialTrackBuilder, SpatialTrackDistances, SpatialTrackHandle, TrackBuilder, TrackHandle,
};
use kira::{AudioManager, AudioManagerSettings, Decibels, DefaultBackend, Mix, Tween};

/// Opens the default output device as the [`Audio`] resource, unless the app already has one,
/// and applies the [`Acoustics`] to its room whenever they change.
#[derive(Clone, Copy, Debug, Default)]
pub struct AudioPlugin;

impl Plugin for AudioPlugin {
    fn build(&self, app: &mut App) {
        if app.world().contains_resource::<Acoustics>() && app.world().contains_resource::<Audio>()
        {
            return;
        }
        if !app.world().contains_resource::<Audio>() {
            app.insert_resource(Audio::open());
        }
        app.init_resource::<Acoustics>()
            .init_resource::<Outside>()
            .init_resource::<Volumes>()
            .add_systems(Update, (apply_acoustics, apply_outside, apply_volumes));
    }
}

fn apply_volumes(audio: Res<Audio>, volumes: Res<Volumes>) {
    if volumes.is_changed() {
        audio.set_volumes(&volumes);
    }
}

/// One of the channels sounds play in ([`Playback::channel`]), each with its own volume
/// ([`Volumes`]); channel 0 unless said otherwise. The engine gives them no meaning.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Channel(pub u8);

impl Channel {
    /// How many there are: channels 0 to `COUNT - 1`; a higher number plays in the last.
    pub const COUNT: usize = 4;

    fn index(self) -> usize {
        usize::from(self.0).min(Self::COUNT - 1)
    }
}

/// How loud everything is, as a resource: everything at once, and each [`Channel`]; amplitudes,
/// 1 unchanged and 0 silent, gliding into a change over [`fade`](Self::fade) seconds.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct Volumes {
    /// Over everything.
    pub master: Real,
    /// Each channel's, by its number.
    pub channels: [Real; Channel::COUNT],
    /// Seconds to glide into a change.
    pub fade: Real,
}

impl Default for Volumes {
    fn default() -> Self {
        Self {
            master: 1.0,
            channels: [1.0; Channel::COUNT],
            fade: 0.1,
        }
    }
}

fn apply_acoustics(audio: Res<Audio>, acoustics: Res<Acoustics>) {
    if acoustics.is_changed() {
        audio.set_acoustics(&acoustics);
    }
}

fn apply_outside(audio: Res<Audio>, outside: Res<Outside>) {
    if outside.is_changed() {
        audio.set_outside(&outside);
    }
}

/// How everything played outside ([`Playback::outside`], every [`Emitter`]) is heard, as a
/// resource: through a low-pass filter and a gain, gliding into a change over
/// [`fade`](Self::fade) seconds. The default lets it all through.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct Outside {
    /// Frequencies above this many hertz are cut.
    pub cutoff: Real,
    /// Amplitude, 1 unchanged.
    pub gain: Real,
    /// Seconds to glide into a change.
    pub fade: Real,
}

impl Outside {
    /// The cutoff that lets everything through.
    pub const OPEN: Real = 20_000.0;
}

impl Default for Outside {
    fn default() -> Self {
        Self {
            cutoff: Self::OPEN,
            gain: 1.0,
            fade: 0.05,
        }
    }
}

/// The room the sounds played with [`Playback::room`] are heard in, as a resource: a reverb.
/// Change it to change the room; the sounds glide into it over [`fade`](Self::fade) seconds.
/// Reads from a TOML table with these fields.
#[derive(Resource, Clone, Copy, Debug, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Acoustics {
    /// How much of a sound in the room is its reverberation, from 0 (dry: no room) to 1 (only
    /// the reverberation).
    pub reverb: Real,
    /// How big the room sounds, from 0 to just under 1; 1 would ring forever.
    pub size: Real,
    /// How fast the high frequencies die out in the reverberation, from 0 to 1.
    pub damping: Real,
    /// How wide the reverberation spreads, from 0 (mono) to 1 (full stereo).
    pub width: Real,
    /// Seconds to glide into these acoustics from the last ones.
    pub fade: Real,
}

impl Default for Acoustics {
    /// No room: the sounds in it play dry.
    fn default() -> Self {
        Self {
            reverb: 0.0,
            size: 0.5,
            damping: 0.5,
            width: 1.0,
            fade: 1.0,
        }
    }
}

/// The output device, as a resource; silent when there is none.
#[derive(Resource)]
pub struct Audio {
    output: Option<Mutex<Output>>,
}

struct Output {
    manager: AudioManager<DefaultBackend>,
    /// Each channel's tracks, made as it is first played in (channel 0 at once).
    channels: [Option<Tracks>; Channel::COUNT],
    /// The ear the emitters are placed about, at the origin facing −z.
    listener: Option<ListenerHandle>,
    /// What was last set, for the channels made after.
    acoustics: Acoustics,
    outside: Outside,
    volumes: Volumes,
}

/// A channel's tracks: its own, for its volume, and inside it the room and the outside.
struct Tracks {
    track: TrackHandle,
    /// Where the sounds in the room play: a track through the reverb.
    room: Option<(TrackHandle, ReverbHandle)>,
    /// Where the sounds outside play: a track through the low-pass filter.
    outside: Option<(TrackHandle, FilterHandle)>,
}

impl Tracks {
    /// A channel's tracks in `manager`, set to `acoustics`, `outside` and `gain`.
    fn new(
        manager: &mut AudioManager<DefaultBackend>,
        acoustics: &Acoustics,
        outside: &Outside,
        gain: Real,
    ) -> Option<Self> {
        let track = TrackBuilder::new().volume(decibels_of(gain));
        let mut track = manager
            .add_sub_track(track)
            .inspect_err(|error| tracing::warn!("no channel, its sounds are lost: {error}"))
            .ok()?;
        let mut room = TrackBuilder::new();
        let reverb = room.add_effect(
            ReverbBuilder::new()
                .feedback(acoustics.size.clamp(0.0, 0.999))
                .damping(acoustics.damping.clamp(0.0, 1.0))
                .stereo_width(acoustics.width.clamp(0.0, 1.0))
                .mix(Mix(acoustics.reverb.clamp(0.0, 1.0) as f32)),
        );
        let room = match track.add_sub_track(room) {
            Ok(room) => Some((room, reverb)),
            Err(error) => {
                tracing::warn!("no room, sounds in it play dry: {error}");
                None
            }
        };
        let mut out = TrackBuilder::new().volume(decibels_of(outside.gain));
        let filter =
            out.add_effect(FilterBuilder::new().cutoff(outside.cutoff.clamp(20.0, Outside::OPEN)));
        let outside = match track.add_sub_track(out) {
            Ok(out) => Some((out, filter)),
            Err(error) => {
                tracing::warn!("nothing outside, sounds out there play straight: {error}");
                None
            }
        };
        Some(Self {
            track,
            room,
            outside,
        })
    }
}

impl Output {
    /// `channel`'s tracks, made now if they were not.
    fn channel(&mut self, channel: Channel) -> Option<&mut Tracks> {
        let Self {
            manager,
            channels,
            acoustics,
            outside,
            volumes,
            ..
        } = self;
        let index = channel.index();
        let slot = &mut channels[index];
        if slot.is_none() {
            *slot = Tracks::new(manager, acoustics, outside, volumes.channels[index]);
        }
        slot.as_mut()
    }
}

impl Audio {
    /// The default output device, or a [silent](Self::silent) one if it cannot be opened.
    pub fn open() -> Self {
        match AudioManager::<DefaultBackend>::new(AudioManagerSettings::default()) {
            Ok(mut manager) => {
                let origin = mint::Vector3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                };
                let facing = mint::Quaternion { v: origin, s: 1.0 };
                let listener = manager
                    .add_listener(origin, facing)
                    .inspect_err(|error| tracing::warn!("no ear, sounds are not placed: {error}"))
                    .ok();
                let mut output = Output {
                    manager,
                    channels: Default::default(),
                    listener,
                    acoustics: Acoustics::default(),
                    outside: Outside::default(),
                    volumes: Volumes::default(),
                };
                output.channel(Channel::default());
                Self {
                    output: Some(Mutex::new(output)),
                }
            }
            Err(error) => {
                tracing::warn!("no sound: {error}");
                Self::silent()
            }
        }
    }

    /// No device: everything played is silent.
    pub fn silent() -> Self {
        Self { output: None }
    }

    /// Whether there is no device.
    pub fn is_silent(&self) -> bool {
        self.output.is_none()
    }

    /// Glides the room into `acoustics`. [`AudioPlugin`] calls it when the [`Acoustics`]
    /// resource changes.
    pub fn set_acoustics(&self, acoustics: &Acoustics) {
        let Some(output) = &self.output else {
            return;
        };
        let mut output = output
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        output.acoustics = *acoustics;
        let fade = tween(acoustics.fade);
        for tracks in output.channels.iter_mut().flatten() {
            if let Some((_, reverb)) = &mut tracks.room {
                reverb.set_feedback(acoustics.size.clamp(0.0, 0.999), fade);
                reverb.set_damping(acoustics.damping.clamp(0.0, 1.0), fade);
                reverb.set_stereo_width(acoustics.width.clamp(0.0, 1.0), fade);
                reverb.set_mix(Mix(acoustics.reverb.clamp(0.0, 1.0) as f32), fade);
            }
        }
    }

    /// Glides everything outside into `outside`. [`AudioPlugin`] calls it when the [`Outside`]
    /// resource changes.
    pub fn set_outside(&self, outside: &Outside) {
        let Some(output) = &self.output else {
            return;
        };
        let mut output = output
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        output.outside = *outside;
        let fade = tween(outside.fade);
        for tracks in output.channels.iter_mut().flatten() {
            if let Some((track, filter)) = &mut tracks.outside {
                filter.set_cutoff(outside.cutoff.clamp(20.0, Outside::OPEN), fade);
                track.set_volume(decibels_of(outside.gain), fade);
            }
        }
    }

    /// Glides everything, and each channel, to `volumes`. [`AudioPlugin`] calls it when the
    /// [`Volumes`] resource changes.
    pub fn set_volumes(&self, volumes: &Volumes) {
        let Some(output) = &self.output else {
            return;
        };
        let mut output = output
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        output.volumes = *volumes;
        let fade = tween(volumes.fade);
        output
            .manager
            .main_track()
            .set_volume(decibels_of(volumes.master), fade);
        for (tracks, &gain) in output.channels.iter_mut().zip(&volumes.channels) {
            if let Some(tracks) = tracks {
                tracks.track.set_volume(decibels_of(gain), fade);
            }
        }
    }

    /// Starts `sound`. The voice is silent if there is no device or the sound cannot be played.
    pub fn play(&self, sound: &Sound, playback: Playback) -> Voice {
        let Some(output) = &self.output else {
            return Voice::default();
        };
        let data = playback.data(sound);
        let mut output = output
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(Tracks {
            track,
            room,
            outside,
        }) = output.channel(playback.channel)
        else {
            return Voice::default();
        };
        let played = match (room, outside) {
            (Some((room, _)), _) if playback.room => room.play(data),
            (_, Some((outside, _))) if playback.outside => outside.play(data),
            _ => track.play(data),
        };
        Voice::from(played)
    }

    /// A place to play sounds from, outside in channel 0, at `at` about the ear (x to the right, y up, −z
    /// ahead): heard fully within `near` and fading out to nothing at `far`, in the same units.
    /// Silent without a device or an ear.
    pub fn emitter(&self, at: &Vector3<Real>, near: Real, far: Real) -> Emitter {
        let Some(output) = &self.output else {
            return Emitter::default();
        };
        let mut output = output
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(listener) = output.listener.as_ref().map(ListenerHandle::id) else {
            return Emitter::default();
        };
        let Some(Tracks {
            outside: Some((track, _)),
            ..
        }) = output.channel(Channel::default())
        else {
            return Emitter::default();
        };
        let builder = SpatialTrackBuilder::new().distances(SpatialTrackDistances {
            min_distance: near.max(1e-3) as f32,
            max_distance: far.max(near + 1e-3) as f32,
        });
        match track.add_spatial_sub_track(listener, mint_of(at), builder) {
            Ok(track) => Emitter { track: Some(track) },
            Err(error) => {
                tracing::warn!("cannot place a sound: {error}");
                Emitter::default()
            }
        }
    }
}

fn mint_of(at: &Vector3<Real>) -> mint::Vector3<f32> {
    mint::Vector3 {
        x: at.x as f32,
        y: at.y as f32,
        z: at.z as f32,
    }
}

/// A place sounds play from, outside, about the ear: see [`Audio::emitter`]. Silent if made
/// without a device ([`Emitter::default`]). Dropped, its sounds stop.
#[derive(Debug, Default)]
pub struct Emitter {
    track: Option<SpatialTrackHandle>,
}

impl Emitter {
    /// Glides it to `at` about the ear (x to the right, y up, −z ahead) over `fade` seconds.
    pub fn set_position(&mut self, at: &Vector3<Real>, fade: Real) {
        if let Some(track) = &mut self.track {
            track.set_position(mint_of(at), tween(fade));
        }
    }

    /// Starts `sound` from here; [`Playback::room`] and [`Playback::outside`] are ignored.
    pub fn play(&mut self, sound: &Sound, playback: Playback) -> Voice {
        let Some(track) = &mut self.track else {
            return Voice::default();
        };
        Voice::from(track.play(playback.data(sound)))
    }
}

/// A decoded clip. Cheap to clone: the samples are shared.
#[derive(Clone, Debug)]
pub struct Sound {
    data: StaticSoundData,
}

impl Sound {
    /// Decodes a whole file's bytes (MP3 or FLAC).
    pub fn decode(bytes: &'static [u8]) -> Result<Self, DecodeError> {
        StaticSoundData::from_cursor(Cursor::new(bytes))
            .map(|data| Self { data })
            .map_err(DecodeError)
    }

    /// A clip made of samples: `samples` stereo frames `[left, right]` at `rate` frames a
    /// second, amplitudes in −1 to 1. For sounds a game makes rather than records (a chord, a
    /// swell).
    pub fn from_samples(rate: u32, samples: &[[f32; 2]]) -> Self {
        let frames: Vec<kira::Frame> = samples
            .iter()
            .map(|&[left, right]| kira::Frame { left, right })
            .collect();
        Self {
            data: StaticSoundData {
                sample_rate: rate.max(1),
                frames: frames.into(),
                settings: Default::default(),
                slice: None,
            },
        }
    }

    /// The part from `start` to `end` seconds. Durations and loops are then relative to it.
    pub fn slice(&self, start: Real, end: Real) -> Self {
        Self {
            data: self.data.slice(start..end),
        }
    }

    /// Its length in seconds, at rate 1.
    pub fn duration(&self) -> Real {
        self.data.duration().as_secs_f64()
    }
}

/// Why a [`Sound`] could not be decoded.
#[derive(Debug)]
pub struct DecodeError(FromFileError);

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl std::error::Error for DecodeError {}

/// How a [`Sound`] starts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Playback {
    /// Amplitude, 1 unchanged.
    pub gain: Real,
    /// Playback speed, 1 unchanged.
    pub rate: Real,
    /// Loop forever between these two times in seconds, once played up to the second.
    pub looped: Option<(Real, Real)>,
    /// Seconds to fade in over; 0 starts at full gain.
    pub fade_in: Real,
    /// Whether it is heard in the room ([`Acoustics`]) rather than straight out.
    pub room: bool,
    /// Whether it is heard outside ([`Outside`]) rather than straight out; the room wins.
    pub outside: bool,
    /// The channel it plays in ([`Volumes`]); ignored by an [`Emitter`].
    pub channel: Channel,
}

impl Playback {
    /// `sound` set to start this way.
    fn data(&self, sound: &Sound) -> StaticSoundData {
        let data = sound
            .data
            .volume(decibels_of(self.gain))
            .playback_rate(self.rate)
            .fade_in_tween((self.fade_in > 0.0).then(|| tween(self.fade_in)));
        match self.looped {
            Some((start, end)) => data.loop_region(start..end),
            None => data,
        }
    }
}

impl Default for Playback {
    fn default() -> Self {
        Self {
            gain: 1.0,
            rate: 1.0,
            looped: None,
            fade_in: 0.0,
            room: false,
            outside: false,
            channel: Channel::default(),
        }
    }
}

/// A sound playing, or nothing ([`Voice::default`]). Dropping it lets the sound play on to its
/// end (forever, if looped).
#[derive(Debug, Default)]
pub struct Voice {
    handle: Option<StaticSoundHandle>,
}

impl<E: std::fmt::Display> From<Result<StaticSoundHandle, E>> for Voice {
    fn from(played: Result<StaticSoundHandle, E>) -> Self {
        match played {
            Ok(handle) => Self {
                handle: Some(handle),
            },
            Err(error) => {
                tracing::warn!("cannot play a sound: {error}");
                Self::default()
            }
        }
    }
}

impl Voice {
    /// Glides the gain to `gain` over `fade` seconds.
    pub fn set_gain(&mut self, gain: Real, fade: Real) {
        if let Some(handle) = &mut self.handle {
            handle.set_volume(decibels_of(gain), tween(fade));
        }
    }

    /// Glides the playback speed to `rate` over `fade` seconds.
    pub fn set_rate(&mut self, rate: Real, fade: Real) {
        if let Some(handle) = &mut self.handle {
            handle.set_playback_rate(rate, tween(fade));
        }
    }

    /// Fades out over `fade` seconds and stops.
    pub fn stop(&mut self, fade: Real) {
        if let Some(handle) = &mut self.handle {
            handle.stop(tween(fade));
        }
    }

    /// Whether it is still sounding (fading out counts). Always false without a device.
    pub fn is_playing(&self) -> bool {
        self.handle
            .as_ref()
            .is_some_and(|handle| handle.state() != PlaybackState::Stopped)
    }
}

/// The amplitude of a level in decibels: 0 dB is 1, −6 dB about ½, +6 dB about 2.
pub fn decibels(level: Real) -> Real {
    10.0_f64.powf(level / 20.0)
}

/// Amplitudes this small or below are silent.
const SILENT_GAIN: Real = 1e-3;

fn decibels_of(gain: Real) -> Decibels {
    if gain <= SILENT_GAIN {
        Decibels::SILENCE
    } else {
        Decibels((20.0 * gain.log10()) as f32)
    }
}

fn tween(fade: Real) -> Tween {
    Tween {
        duration: Duration::from_secs_f64(fade.max(0.0)),
        ..Tween::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gains_and_decibels_agree() {
        assert_eq!(decibels(0.0), 1.0);
        assert!((decibels(-20.0) - 0.1).abs() < 1e-12);
        assert_eq!(decibels_of(1.0), Decibels(0.0));
        assert!((decibels_of(0.5).0 + 6.0206).abs() < 1e-3);
        assert_eq!(decibels_of(0.0), Decibels::SILENCE);
    }

    #[test]
    fn a_clip_made_of_samples_lasts_as_long_as_they_do() {
        let samples = vec![[0.0, 0.0]; 48_000];
        let sound = Sound::from_samples(24_000, &samples);
        assert!((sound.duration() - 2.0).abs() < 1e-9);
        assert!((sound.slice(0.5, 1.5).duration() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn a_silent_device_plays_nothing() {
        let mut app = App::new();
        app.insert_resource(Audio::silent()).add_plugin(AudioPlugin);
        assert!(app.world().resource::<Audio>().is_silent(), "kept");
        app.world_mut().resource_mut::<Acoustics>().reverb = 0.5;
        app.world_mut().resource_mut::<Outside>().cutoff = 400.0;
        app.world_mut().resource_mut::<Volumes>().channels[1] = 0.5;
        app.update(0.1);
        let audio = app.world().resource::<Audio>();
        let mut emitter = audio.emitter(&Vector3::new(1.0, 0.0, -2.0), 1.0, 50.0);
        emitter.set_position(&Vector3::new(-3.0, 0.0, 0.0), 0.1);
        let click = Sound::from_samples(8_000, &[[0.0; 2]; 80]);
        assert!(!emitter.play(&click, Playback::default()).is_playing());
        let elsewhere = Playback {
            channel: Channel(3),
            ..Playback::default()
        };
        assert!(!audio.play(&click, elsewhere).is_playing());
        let mut voice = Voice::default();
        voice.set_gain(0.5, 1.0);
        voice.stop(0.0);
        assert!(!voice.is_playing());
    }
}
