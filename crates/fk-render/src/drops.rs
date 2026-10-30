//! The drops' recorder: the last seconds of frames kept, and written to a file when the frames
//! drop, to find out why.
//!
//! On with `FK_DROPS=<dir>` in any build, and by default in a debug build (into
//! `target/drops`; `FK_DROPS=off` turns it off); never during a performance recording
//! (`FK_PERF`, [`perf`](crate::perf)), which keeps every frame already. While it is on the
//! renderer times its passes on the GPU ([`GpuTimings`]) where the device can.
//!
//! It keeps `FK_DROPS_BEFORE` seconds of frames (2): each frame's length, the CPU's time per
//! schedule ([`FrameProfile`]), the GPU's per pass, what was drawn ([`RenderStats`]) and what
//! happened ([`FrameEvents`]). When the runner sees a drop ([`FrameDrops`]: a spike or a sag),
//! it goes on `FK_DROPS_AFTER` seconds (0.5: the GPU's timings come back a few frames late),
//! then writes `<dir>/<unix seconds>-<n>-<spike|sag>.toml`, a [`DropReport`] (where the app said
//! it was, [`FrameNotes`]; the frame looked at against the frames before it, what was over the
//! most first), and every frame kept beside it as CSV (the same path, `csv`). Drops within those
//! seconds go in the same file. At most `FK_DROPS_MAX` files (20) a run.
//!
//! A debug build's CPU times are the dev profile's, slower than a release build's and not in the
//! same proportions: its report says which it was ([`DropReport::optimized`]).

use std::collections::{BTreeMap, VecDeque};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use bevy_ecs::prelude::*;
use fk_app::{FrameDrops, FrameEvents, FrameNotes, FrameProfile, Time};
use fk_math::Real;
use serde::{Deserialize, Serialize};

use crate::RenderStats;
use crate::perf::{Drawing, Spread, write_file};
use crate::timing::GpuTimings;

/// Where the drops go and how much is kept, from the environment.
#[derive(Clone, Debug, PartialEq)]
struct Settings {
    dir: PathBuf,
    /// Seconds of frames kept before a drop...
    before: Real,
    /// ...and after it.
    after: Real,
    /// Files written at most.
    max: usize,
}

impl Settings {
    fn from_env() -> Option<Self> {
        if std::env::var_os("FK_PERF").is_some() {
            return None;
        }
        let dir = match std::env::var("FK_DROPS") {
            Ok(dir) if matches!(dir.as_str(), "" | "0" | "off") => return None,
            Ok(dir) => PathBuf::from(dir),
            Err(_) if cfg!(debug_assertions) => PathBuf::from("target/drops"),
            Err(_) => return None,
        };
        let number = |name: &str, default: Real| {
            std::env::var(name)
                .ok()
                .and_then(|text| text.parse::<Real>().ok())
                .filter(|value| *value >= 0.0)
                .unwrap_or(default)
        };
        Some(Self {
            dir,
            before: number("FK_DROPS_BEFORE", 2.0),
            after: number("FK_DROPS_AFTER", 0.5),
            max: number("FK_DROPS_MAX", 20.0) as usize,
        })
    }
}

/// The settings, read once.
fn settings() -> Option<&'static Settings> {
    static SETTINGS: OnceLock<Option<Settings>> = OnceLock::new();
    SETTINGS.get_or_init(Settings::from_env).as_ref()
}

/// Whether the recorder is on.
pub(crate) fn enabled() -> bool {
    settings().is_some()
}

/// Where the recorder writes the drops, if it is on.
pub fn dir() -> Option<&'static Path> {
    settings().map(|settings| settings.dir.as_path())
}

/// What a drop was.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// A frame or a few much longer than the usual.
    Spike,
    /// The usual frame itself slow.
    Sag,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Self::Spike => "spike",
            Self::Sag => "sag",
        }
    }
}

/// One frame kept.
#[derive(Clone, Debug, Default, PartialEq)]
struct Frame {
    /// Its number ([`Time::frame`]).
    number: u64,
    /// Its length, seconds.
    length: Real,
    /// Whether it was a spike.
    spike: bool,
    cpu: FrameProfile,
    /// The GPU's time per pass, seconds, once they came back.
    gpu: Option<Vec<(&'static str, Real)>>,
    stats: RenderStats,
    events: Vec<(&'static str, u32)>,
}

/// A drop seen, the frames after it still coming.
#[derive(Clone, Debug)]
struct Open {
    kind: Kind,
    /// The frame it was seen on.
    at: u64,
    /// The usual frame then, seconds.
    usual: Real,
    /// The app's clock then, seconds.
    seconds: Real,
    /// Seconds of frames still to keep.
    left: Real,
    /// Drops seen since, this one included.
    drops: u32,
    notes: BTreeMap<String, String>,
}

/// The recorder, as a resource, when it is on.
#[derive(Resource, Debug)]
pub(crate) struct DropRecorder {
    settings: Settings,
    frames: VecDeque<Frame>,
    /// The drops' counters as last seen.
    seen: FrameDrops,
    gpu_serial: u64,
    open: Option<Open>,
    written: usize,
}

impl DropRecorder {
    pub(crate) fn from_env() -> Option<Self> {
        settings().cloned().map(Self::new)
    }

    fn new(settings: Settings) -> Self {
        Self {
            settings,
            frames: VecDeque::new(),
            seen: FrameDrops::default(),
            gpu_serial: 0,
            open: None,
            written: 0,
        }
    }

    /// Keeps `frame`, given what the runner saw of it; returns the report to write once a drop's
    /// frames after it are in.
    fn keep(
        &mut self,
        frame: Frame,
        drops: &FrameDrops,
        seconds: Real,
        notes: &FrameNotes,
    ) -> Option<(DropReport, String)> {
        let kind = if drops.sags > self.seen.sags {
            Some(Kind::Sag)
        } else if drops.spikes > self.seen.spikes {
            Some(Kind::Spike)
        } else {
            None
        };
        self.seen = *drops;
        let (number, length) = (frame.number, frame.length);
        self.frames.push_back(frame);
        match (&mut self.open, kind) {
            (Some(open), kind) => {
                open.left -= length;
                open.drops += u32::from(kind.is_some());
            }
            (None, Some(kind)) if self.written < self.settings.max => {
                self.open = Some(Open {
                    kind,
                    at: number,
                    usual: drops.usual,
                    seconds,
                    left: self.settings.after,
                    drops: 1,
                    notes: notes
                        .iter()
                        .map(|(name, value)| (name.to_owned(), value.to_owned()))
                        .collect(),
                });
            }
            (None, Some(_)) if self.written == self.settings.max => {
                tracing::warn!(
                    "{} drops written: no more this run (FK_DROPS_MAX)",
                    self.settings.max
                );
                self.written += 1;
            }
            (None, _) => {}
        }
        let done = self.open.as_ref().is_some_and(|open| open.left <= 0.0);
        let report = done.then(|| {
            let open = self.open.take().expect("open");
            self.written += 1;
            (report(&open, &self.frames), frames_csv(&self.frames))
        });
        if self.open.is_none() {
            let mut kept: Real = self.frames.iter().map(|frame| frame.length).sum();
            while kept > self.settings.before {
                let Some(old) = self.frames.pop_front() else {
                    break;
                };
                kept -= old.length;
            }
        }
        report
    }

    /// Gives `timings` to the frame they are of, if it is still kept.
    fn timed(&mut self, timings: &GpuTimings) {
        self.gpu_serial = timings.serial;
        if let Some(frame) = self
            .frames
            .iter_mut()
            .rev()
            .find(|frame| frame.number == timings.frame)
        {
            frame.gpu = Some(timings.passes.clone());
        }
    }
}

/// A drop, as written to its TOML file.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DropReport {
    /// `spike` (a frame or a few much longer than the usual) or `sag` (the usual frame itself
    /// slow).
    pub kind: String,
    /// Whether the build was optimized: a debug build's CPU times are no measure.
    pub optimized: bool,
    /// The adapter drawn on.
    pub adapter: String,
    /// Its backend.
    pub backend: String,
    /// Its driver and the driver's version.
    pub driver: String,
    /// The image's width, pixels.
    pub width: u32,
    /// The image's height, pixels.
    pub height: u32,
    /// Whether frames waited for the display's refresh.
    pub vsync: bool,
    /// The app's clock at the drop, seconds.
    pub seconds: Real,
    /// Drops seen while the file was kept open, the first included.
    pub drops: u32,
    /// The frame looked at: a spike's longest, or a sag's last.
    pub frame: u64,
    /// The frame looked at, milliseconds: a spike's length, or the mean over a sag's last
    /// second.
    pub ms: Real,
    /// The usual frame when the drop was seen, milliseconds.
    pub usual_ms: Real,
    /// The frame the GPU's times are of: the one looked at, or the nearest after it timed;
    /// `None` when none came back.
    pub gpu_frame: Option<u64>,
    /// Each schedule's and pass's time in the frame looked at against the frames before it,
    /// what grew the most first.
    pub over: Vec<Over>,
    /// What happened in the frame looked at (over a sag's last second, summed).
    pub events: BTreeMap<String, u32>,
    /// Where the app said it was ([`FrameNotes`]) when the drop was seen.
    pub notes: BTreeMap<String, String>,
}

/// A schedule's or a pass's time in the frame looked at against the usual.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Over {
    /// `cpu <schedule>` or `gpu <pass>`.
    pub what: String,
    /// In the frame looked at, milliseconds.
    pub ms: Real,
    /// The median of the frames kept before the drop, milliseconds.
    pub usual_ms: Real,
}

impl Over {
    /// How much longer than usual, milliseconds.
    pub fn grew(&self) -> Real {
        self.ms - self.usual_ms
    }
}

impl DropReport {
    /// Reads a report the recorder wrote.
    pub fn read(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|error| error.to_string())?;
        toml::from_str(&text).map_err(|error| error.to_string())
    }
}

/// The schedules' and the passes' names in `frames`, as [`Over::what`] names them.
fn columns(frames: &VecDeque<Frame>) -> Vec<String> {
    let mut names: Vec<String> = FrameProfile::default()
        .schedules()
        .iter()
        .map(|(name, _)| format!("cpu {name}"))
        .collect();
    for frame in frames {
        for (pass, _) in frame.gpu.iter().flatten() {
            let name = format!("gpu {pass}");
            if !names.contains(&name) {
                names.push(name);
            }
        }
    }
    names
}

/// The time of `column` in `frame`, seconds; `None` for a pass when it was not timed.
fn value(frame: &Frame, column: &str) -> Option<Real> {
    if let Some(schedule) = column.strip_prefix("cpu ") {
        return frame
            .cpu
            .schedules()
            .iter()
            .find(|(name, _)| *name == schedule)
            .map(|(_, seconds)| *seconds);
    }
    let pass = column.strip_prefix("gpu ")?;
    let passes = frame.gpu.as_ref()?;
    Some(
        passes
            .iter()
            .filter(|(name, _)| *name == pass)
            .map(|(_, seconds)| seconds)
            .sum(),
    )
}

/// The report of `open` over the `frames` kept.
fn report(open: &Open, frames: &VecDeque<Frame>) -> DropReport {
    let before: Vec<&Frame> = frames.iter().filter(|f| f.number < open.at).collect();
    // A spike's longest frame; a sag's last second up to where it was seen.
    let looked: Vec<&Frame> = match open.kind {
        Kind::Spike => frames
            .iter()
            .filter(|f| f.number >= open.at)
            .max_by(|a, b| a.length.total_cmp(&b.length))
            .into_iter()
            .collect(),
        Kind::Sag => {
            let mut second = 0.0;
            frames
                .iter()
                .rev()
                .filter(|f| f.number <= open.at)
                .take_while(|f| {
                    second += f.length;
                    second <= 1.0 + f.length
                })
                .collect()
        }
    };
    let before: Vec<&Frame> = match open.kind {
        Kind::Spike => before,
        Kind::Sag => {
            let first = looked.last().map_or(open.at, |f| f.number);
            before.into_iter().filter(|f| f.number < first).collect()
        }
    };
    let last = looked.first().map_or(open.at, |f| f.number);
    // The GPU's times: of the frames looked at, or else of the nearest after them timed.
    let mut gpu_frames: Vec<&Frame> = looked.iter().copied().filter(|f| f.gpu.is_some()).collect();
    if gpu_frames.is_empty()
        && let Some(after) = frames.iter().find(|f| f.number > last && f.gpu.is_some())
    {
        gpu_frames.push(after);
    }
    let gpu_frame = gpu_frames.iter().map(|f| f.number).max();
    let mean = |of: &[&Frame], column: &str| -> Option<Real> {
        let values: Vec<Real> = of.iter().filter_map(|f| value(f, column)).collect();
        (!values.is_empty()).then(|| values.iter().sum::<Real>() / values.len() as Real * 1e3)
    };
    let mut over: Vec<Over> = columns(frames)
        .into_iter()
        .filter_map(|what| {
            let at = if what.starts_with("gpu ") {
                &gpu_frames
            } else {
                &looked
            };
            let ms = mean(at, &what)?;
            let usual: Vec<Real> = before.iter().filter_map(|f| value(f, &what)).collect();
            let usual_ms = if usual.is_empty() {
                0.0
            } else {
                Spread::of(usual).p50
            };
            Some(Over { what, ms, usual_ms })
        })
        .collect();
    over.sort_by(|a, b| b.grew().total_cmp(&a.grew()));
    let mut events = BTreeMap::new();
    for frame in &looked {
        for (name, count) in &frame.events {
            *events.entry((*name).to_owned()).or_default() += count;
        }
    }
    let lengths: Vec<Real> = looked.iter().map(|f| f.length).collect();
    DropReport {
        kind: open.kind.name().to_owned(),
        optimized: !cfg!(debug_assertions),
        seconds: open.seconds,
        drops: open.drops,
        frame: last,
        ms: lengths.iter().sum::<Real>() / lengths.len().max(1) as Real * 1e3,
        usual_ms: open.usual * 1e3,
        gpu_frame,
        over,
        events,
        notes: open.notes.clone(),
        ..DropReport::default()
    }
}

/// Every frame kept, a line each, milliseconds; a frame whose GPU timings did not come back has
/// those columns empty; its events as `name=count` joined by `;`.
fn frames_csv(frames: &VecDeque<Frame>) -> String {
    let columns = columns(frames);
    let mut csv = String::from("frame,ms,spike,fixed_steps,cpu_ms,gpu_ms");
    for column in &columns {
        let _ = write!(csv, ",{}_ms", column.replace(' ', "_"));
    }
    csv.push_str(",drawn,culled,draws,events\n");
    for frame in frames {
        let gpu = frame.gpu.as_ref().map_or(String::new(), |passes| {
            format!("{:.4}", passes.iter().map(|(_, s)| s).sum::<Real>() * 1e3)
        });
        let _ = write!(
            csv,
            "{},{:.4},{},{},{:.4},{gpu}",
            frame.number,
            frame.length * 1e3,
            u8::from(frame.spike),
            frame.cpu.fixed_steps,
            frame.cpu.total() * 1e3,
        );
        for column in &columns {
            csv.push(',');
            if let Some(seconds) = value(frame, column) {
                let _ = write!(csv, "{:.4}", seconds * 1e3);
            }
        }
        let stats = &frame.stats;
        let events: Vec<String> = frame
            .events
            .iter()
            .map(|(name, count)| format!("{name}={count}"))
            .collect();
        let _ = writeln!(
            csv,
            ",{},{},{},{}",
            stats.drawn,
            stats.culled,
            stats.draws,
            events.join(";")
        );
    }
    csv
}

/// The frame's own: what the runner saw of the last frame and what it did.
#[derive(bevy_ecs::system::SystemParam)]
pub(crate) struct LastFrame<'w> {
    drops: Res<'w, FrameDrops>,
    time: Res<'w, Time>,
    profile: Res<'w, FrameProfile>,
    events: Res<'w, FrameEvents>,
    notes: Res<'w, FrameNotes>,
    gpu: Option<Res<'w, GpuTimings>>,
    stats: Option<Res<'w, RenderStats>>,
}

/// Keeps the last frame in the [`DropRecorder`], if there is one, and writes a drop's report once
/// its frames after it are in.
pub(crate) fn record(recorder: Option<ResMut<DropRecorder>>, last: LastFrame, drawing: Drawing) {
    let Some(mut recorder) = recorder else { return };
    // No window watched, no frame to keep (an app updated by hand).
    if last.drops.last <= 0.0 {
        return;
    }
    // The runner's drops, the profile and the events are of the frame before this one.
    let frame = Frame {
        number: last.time.frame().saturating_sub(1),
        length: last.drops.last,
        spike: last.drops.spikes > recorder.seen.spikes,
        cpu: *last.profile,
        gpu: None,
        stats: last.stats.map(|stats| *stats).unwrap_or_default(),
        events: last.events.last().collect(),
    };
    let written = recorder.keep(frame, &last.drops, last.time.elapsed(), &last.notes);
    if let Some(timings) = last
        .gpu
        .filter(|timings| timings.serial != recorder.gpu_serial)
    {
        recorder.timed(&timings);
    }
    let Some((mut report, csv)) = written else {
        return;
    };
    drawing.describe(&mut report);
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |since| since.as_secs());
    let path = recorder
        .settings
        .dir
        .join(format!("{stamp}-{}-{}.toml", recorder.written, report.kind));
    let written = toml::to_string(&report)
        .map_err(|error| error.to_string())
        .and_then(|text| write_file(&path, &text))
        .and_then(|()| write_file(&path.with_extension("csv"), &csv));
    match written {
        Ok(()) => {
            let worst = report.over.first().map_or(String::new(), |over| {
                format!(
                    ", most over: {} {:.1} ms (usually {:.1})",
                    over.what, over.ms, over.usual_ms
                )
            });
            tracing::warn!(
                "frame {} {:.0} ms, usually {:.1}{worst}: written to {}",
                report.kind,
                report.ms,
                report.usual_ms,
                path.display()
            );
        }
        Err(error) => tracing::error!("frame drop {}: {error}", path.display()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recorder() -> DropRecorder {
        DropRecorder::new(Settings {
            dir: PathBuf::new(),
            before: 0.5,
            after: 0.1,
            max: 2,
        })
    }

    /// Keeps a frame of `length` seconds, its `update` taking `update`, a spike if `spike`.
    fn step(
        recorder: &mut DropRecorder,
        drops: &mut FrameDrops,
        number: u64,
        length: Real,
        update: Real,
        spike: bool,
    ) -> Option<(DropReport, String)> {
        drops.spikes += u64::from(spike);
        drops.last = length;
        drops.usual = 0.01;
        let frame = Frame {
            number,
            length,
            spike,
            cpu: FrameProfile {
                update,
                ..FrameProfile::default()
            },
            events: if spike {
                vec![("pipelines built", 3)]
            } else {
                Vec::new()
            },
            ..Frame::default()
        };
        let mut notes = FrameNotes::default();
        notes.set("place", "road");
        recorder.keep(frame, drops, number as Real * 0.01, &notes)
    }

    #[test]
    fn only_the_last_seconds_are_kept() {
        let (mut recorder, mut drops) = (recorder(), FrameDrops::default());
        for number in 0..200 {
            assert!(step(&mut recorder, &mut drops, number, 0.01, 0.002, false).is_none());
        }
        let kept: Real = recorder.frames.iter().map(|f| f.length).sum();
        assert!(kept <= 0.5 + 1e-9 && kept > 0.48, "{kept}");
    }

    #[test]
    fn a_spike_is_written_with_the_frames_on_both_sides() {
        let (mut recorder, mut drops) = (recorder(), FrameDrops::default());
        for number in 0..100 {
            step(&mut recorder, &mut drops, number, 0.01, 0.002, false);
        }
        assert!(step(&mut recorder, &mut drops, 100, 0.2, 0.19, true).is_none());
        // A second spike within the after window: the same file.
        assert!(step(&mut recorder, &mut drops, 101, 0.05, 0.04, true).is_none());
        // GPU timings of the spike come back late, matched to it.
        recorder.timed(&GpuTimings {
            serial: 1,
            frame: 100,
            passes: vec![("opaque", 0.008)],
        });
        let mut written = None;
        for number in 102..120 {
            written = written.or(step(&mut recorder, &mut drops, number, 0.01, 0.002, false));
        }
        let (report, csv) = written.expect("written after `after` seconds");
        assert_eq!(report.kind, "spike");
        assert_eq!(report.drops, 2);
        assert_eq!(report.frame, 100);
        assert!((report.ms - 200.0).abs() < 1e-9);
        assert_eq!(report.over[0].what, "cpu update");
        assert!((report.over[0].ms - 190.0).abs() < 1e-9);
        assert!((report.over[0].usual_ms - 2.0).abs() < 1e-9);
        assert_eq!(report.gpu_frame, Some(100));
        assert!(report.over.iter().any(|o| o.what == "gpu opaque"));
        assert_eq!(report.events["pipelines built"], 3);
        assert_eq!(report.notes["place"], "road");
        // Frames before and after the spike, every line as many columns as the header.
        let lines: Vec<&str> = csv.lines().collect();
        assert!(lines.iter().any(|l| l.starts_with("99,")));
        assert!(lines.iter().any(|l| l.starts_with("105,")));
        let columns = lines[0].split(',').count();
        assert!(lines.iter().all(|l| l.split(',').count() == columns));
        // Read back as written.
        let text = toml::to_string(&report).unwrap();
        let read: DropReport = toml::from_str(&text).unwrap();
        assert_eq!(read, report);
    }

    #[test]
    fn no_more_files_than_the_most() {
        let (mut recorder, mut drops) = (recorder(), FrameDrops::default());
        let mut files = 0;
        for number in 0..2000 {
            let spike = number % 100 == 50;
            if step(&mut recorder, &mut drops, number, 0.01, 0.002, spike).is_some() {
                files += 1;
            }
        }
        assert_eq!(files, 2);
    }

    #[test]
    fn a_sag_looks_at_its_last_second() {
        let (mut recorder, mut drops) = (recorder(), FrameDrops::default());
        recorder.settings.before = 3.0;
        for number in 0..100 {
            step(&mut recorder, &mut drops, number, 0.01, 0.002, false);
        }
        for number in 100..140 {
            step(&mut recorder, &mut drops, number, 0.025, 0.017, false);
        }
        drops.sags += 1;
        let mut written = step(&mut recorder, &mut drops, 140, 0.025, 0.017, false);
        for number in 141..150 {
            written = written.or(step(&mut recorder, &mut drops, number, 0.025, 0.017, false));
        }
        let (report, _) = written.expect("written");
        assert_eq!(report.kind, "sag");
        assert!((report.ms - 25.0).abs() < 1e-9);
        assert_eq!(report.over[0].what, "cpu update");
        assert!((report.over[0].usual_ms - 2.0).abs() < 1e-9);
    }
}
