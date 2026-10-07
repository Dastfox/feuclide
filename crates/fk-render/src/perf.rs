//! Performance recordings: how long frames take, on the CPU and on the GPU, written to a report
//! for the performance tests.
//!
//! Setting `FK_PERF` to a file path makes any app on the [`RenderPlugin`](crate::RenderPlugin)
//! record its frames and exit: it lets `FK_PERF_WARMUP` seconds go by (5 by default; seconds
//! while [`PerfHold`] holds it back do not count), records `FK_PERF_SECONDS` seconds of frames
//! (20 by default), writes a [`PerfReport`] there as TOML and every frame next to it as CSV
//! (the same path with the extension `csv`), and exits. `FK_PERF_LABEL` names the recording in
//! the report (the file's stem by default). While it records, the renderer times its passes
//! on the GPU ([`GpuTimings`]) where the device has timestamp queries.
//!
//! A frame's time is the wall-clock time from one frame to the next: with vsync it is the
//! display's refresh, so a recording to measure meant `FK_NO_VSYNC` too. The CPU's time is the
//! sum of the schedules ([`FrameProfile`]), the renderer's wait for the window's next image
//! included: a frame bound by the GPU shows it in [`PerfReport::schedules`]'s `render`.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemParam;
use fk_app::{AppExit, FrameProfile, WindowSize};
use fk_math::Real;
use serde::{Deserialize, Serialize};

use crate::timing::GpuTimings;
use crate::{Gpu, Multisampling, RenderStats, Vsync};

/// Whether a performance recording was asked for (`FK_PERF`).
pub(crate) fn enabled() -> bool {
    std::env::var_os("FK_PERF").is_some()
}

/// Holds a performance recording back while set, as a resource: neither the warm-up nor the
/// recording goes on. An app sets it every frame until it is where it should be measured (a
/// game until its player has arrived somewhere, say). Without it, nothing is held back.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PerfHold(pub bool);

/// The names of the GPU's passes, in the order the renderer runs them (see [`GpuTimings`]).
pub const GPU_PASSES: [&str; 8] = [
    "elsewhere",
    "shadows",
    "opaque",
    "layer",
    "ink",
    "mosh",
    "post",
    "overlay",
];

/// Durations summed up over a recording, milliseconds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Spread {
    /// The mean.
    pub mean: Real,
    /// The median.
    pub p50: Real,
    /// The 95th percentile: one sample in twenty is longer.
    pub p95: Real,
    /// The 99th percentile: one sample in a hundred is longer.
    pub p99: Real,
    /// The longest.
    pub max: Real,
}

impl Spread {
    /// The spread of `seconds`, in milliseconds; all zeros without any.
    pub fn of(seconds: impl IntoIterator<Item = Real>) -> Self {
        let mut ms: Vec<Real> = seconds.into_iter().map(|s| s * 1e3).collect();
        if ms.is_empty() {
            return Self::default();
        }
        ms.sort_by(Real::total_cmp);
        let at = |p: Real| {
            let rank = (p * ms.len() as Real).ceil() as usize;
            ms[rank.clamp(1, ms.len()) - 1]
        };
        Self {
            mean: ms.iter().sum::<Real>() / ms.len() as Real,
            p50: at(0.5),
            p95: at(0.95),
            p99: at(0.99),
            max: ms[ms.len() - 1],
        }
    }
}

/// What a performance recording measured, as written to its TOML file.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PerfReport {
    /// What was recorded (`FK_PERF_LABEL`).
    pub label: String,
    /// The adapter drawn on.
    pub adapter: String,
    /// Its backend (Vulkan, Dx12, Metal, …).
    pub backend: String,
    /// Its driver and the driver's version.
    pub driver: String,
    /// The image's width, pixels.
    pub width: u32,
    /// The image's height, pixels.
    pub height: u32,
    /// Samples per pixel of the opaque pass.
    pub samples: u32,
    /// Whether frames waited for the display's refresh: the frame times are then no measure.
    pub vsync: bool,
    /// Whether the build was optimized: a debug build's times are no measure either.
    pub optimized: bool,
    /// Whether the app never let the recording start ([`PerfHold`]) before it gave up.
    pub held_out: bool,
    /// Seconds recorded.
    pub seconds: Real,
    /// Frames recorded.
    pub frames: usize,
    /// Frames a second over the recording.
    pub fps: Real,
    /// The time from one frame to the next.
    pub frame: Spread,
    /// The CPU's time in the schedules, together.
    pub cpu: Spread,
    /// The GPU's time over the passes, together, when they were timed.
    pub gpu: Option<Spread>,
    /// Instances drawn a frame, on average.
    pub drawn: Real,
    /// Instances culled a frame, on average.
    pub culled: Real,
    /// Draw calls a frame, on average.
    pub draws: Real,
    /// The CPU's time in each schedule (`pre_update`, `fixed_update`, `update`,
    /// `post_update`, `render`).
    pub schedules: BTreeMap<String, Spread>,
    /// The GPU's time in each pass, when they were timed (see [`GPU_PASSES`]).
    pub passes: BTreeMap<String, Spread>,
}

impl PerfReport {
    /// Reads a report written by a recording.
    pub fn read(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|error| error.to_string())?;
        toml::from_str(&text).map_err(|error| error.to_string())
    }

    /// The GPU's mean time over its passes, if they were timed, else the CPU's, and which it
    /// was: what bounds the frame, as far as one can tell.
    pub fn bound(&self) -> (&'static str, Real) {
        match &self.gpu {
            Some(gpu) if gpu.mean > self.cpu_work() => ("GPU", gpu.mean),
            _ => ("CPU", self.cpu_work()),
        }
    }

    /// The CPU's mean time without the renderer's wait for the window: its own work,
    /// milliseconds. The wait is not told apart from the renderer's own work, so this is the
    /// mean of every schedule but `render`, plus `render` where the GPU was quicker than it.
    pub fn cpu_work(&self) -> Real {
        let render = self.schedules.get("render").map_or(0.0, |s| s.mean);
        let rest = self.cpu.mean - render;
        match &self.gpu {
            Some(gpu) if gpu.mean >= render => rest,
            _ => self.cpu.mean,
        }
    }
}

/// One frame of a recording.
struct Sample {
    frame: Real,
    cpu: FrameProfile,
    gpu: Option<GpuTimings>,
    stats: RenderStats,
}

/// A recording asked for by `FK_PERF`, as a resource.
#[derive(Resource)]
pub(crate) struct PerfRecording {
    path: PathBuf,
    label: String,
    warmup: Real,
    seconds: Real,
    /// Seconds held back after which the recording gives up.
    timeout: Real,
    warmed: Real,
    held: Real,
    recorded: Real,
    last: Option<Instant>,
    gpu_serial: u64,
    samples: Vec<Sample>,
}

impl PerfRecording {
    pub(crate) fn from_env() -> Option<Self> {
        let path = PathBuf::from(std::env::var_os("FK_PERF")?);
        let seconds = |name: &str, default: Real| {
            std::env::var(name)
                .ok()
                .and_then(|text| text.parse().ok())
                .unwrap_or(default)
        };
        let label = std::env::var("FK_PERF_LABEL").unwrap_or_else(|_| {
            path.file_stem().map_or("perf".to_owned(), |stem| {
                stem.to_string_lossy().into_owned()
            })
        });
        Some(Self {
            label,
            warmup: seconds("FK_PERF_WARMUP", 5.0),
            seconds: seconds("FK_PERF_SECONDS", 20.0),
            timeout: seconds("FK_PERF_TIMEOUT", 180.0),
            path,
            warmed: 0.0,
            held: 0.0,
            recorded: 0.0,
            last: None,
            gpu_serial: 0,
            samples: Vec::new(),
        })
    }
}

/// What the renderer draws on and into, for a report.
#[derive(SystemParam)]
pub(crate) struct Drawing<'w> {
    gpu: Option<Res<'w, Gpu>>,
    size: Option<Res<'w, WindowSize>>,
    samples: Option<Res<'w, Multisampling>>,
    vsync: Option<Res<'w, Vsync>>,
}

/// Records a frame of the [`PerfRecording`], if there is one, and once it is over writes its
/// report and asks the app to exit.
pub(crate) fn record(
    mut commands: Commands,
    recording: Option<ResMut<PerfRecording>>,
    hold: Option<Res<PerfHold>>,
    profile: Res<FrameProfile>,
    (gpu_timings, stats): (Option<Res<GpuTimings>>, Option<Res<RenderStats>>),
    drawing: Drawing,
    mut exit: ResMut<AppExit>,
) {
    let Drawing {
        gpu,
        size,
        samples,
        vsync,
    } = drawing;
    let Some(mut recording) = recording else {
        return;
    };
    let now = Instant::now();
    let Some(last) = recording.last.replace(now) else {
        return;
    };
    let frame = now.duration_since(last).as_secs_f64();
    // Nothing is drawn without a device: the recording waits for one.
    let held = gpu.is_none() || hold.is_some_and(|hold| hold.0);
    if held {
        recording.held += frame;
    } else if recording.warmed < recording.warmup {
        recording.warmed += frame;
    } else {
        let gpu_timings = gpu_timings
            .filter(|timings| timings.serial != recording.gpu_serial)
            .map(|timings| timings.clone());
        if let Some(timings) = &gpu_timings {
            recording.gpu_serial = timings.serial;
        }
        recording.recorded += frame;
        recording.samples.push(Sample {
            frame,
            cpu: *profile,
            gpu: gpu_timings,
            stats: stats.map(|stats| *stats).unwrap_or_default(),
        });
    }
    let held_out = recording.samples.is_empty() && recording.held >= recording.timeout;
    if recording.recorded < recording.seconds && !held_out {
        return;
    }
    let mut report = report(&recording);
    report.held_out = held_out;
    if let Some(gpu) = gpu {
        report.adapter.clone_from(&gpu.adapter.name);
        report.backend = format!("{:?}", gpu.adapter.backend);
        report.driver = format!("{} {}", gpu.adapter.driver, gpu.adapter.driver_info)
            .trim()
            .to_owned();
    }
    if let Some(size) = size {
        (report.width, report.height) = (size.width, size.height);
    }
    report.samples = samples.map_or(1, |samples| samples.samples);
    report.vsync = vsync.is_none_or(|vsync| vsync.0) && std::env::var_os("FK_NO_VSYNC").is_none();
    write(&recording, &report);
    commands.remove_resource::<PerfRecording>();
    exit.request();
}

/// The report of `recording`, without what only the world knows (the adapter, the image).
fn report(recording: &PerfRecording) -> PerfReport {
    let samples = &recording.samples;
    let mean = |of: fn(&RenderStats) -> usize| {
        samples.iter().map(|s| of(&s.stats) as Real).sum::<Real>() / samples.len().max(1) as Real
    };
    let timed: Vec<&GpuTimings> = samples.iter().filter_map(|s| s.gpu.as_ref()).collect();
    let pass = |name: &str| -> Vec<Real> {
        timed
            .iter()
            .map(|timings| {
                timings
                    .passes
                    .iter()
                    .find(|(pass, _)| *pass == name)
                    .map_or(0.0, |(_, seconds)| *seconds)
            })
            .collect()
    };
    PerfReport {
        label: recording.label.clone(),
        optimized: !cfg!(debug_assertions),
        seconds: recording.recorded,
        frames: samples.len(),
        fps: samples.len() as Real / recording.recorded.max(1e-9),
        frame: Spread::of(samples.iter().map(|s| s.frame)),
        cpu: Spread::of(samples.iter().map(|s| s.cpu.total())),
        gpu: (!timed.is_empty()).then(|| Spread::of(timed.iter().map(|t| t.total()))),
        drawn: mean(|stats| stats.drawn),
        culled: mean(|stats| stats.culled),
        draws: mean(|stats| stats.draws),
        schedules: FrameProfile::default()
            .schedules()
            .iter()
            .enumerate()
            .map(|(i, (name, _))| {
                let spread = Spread::of(samples.iter().map(|s| s.cpu.schedules()[i].1));
                ((*name).to_owned(), spread)
            })
            .collect(),
        passes: if timed.is_empty() {
            BTreeMap::new()
        } else {
            GPU_PASSES
                .iter()
                .map(|name| ((*name).to_owned(), Spread::of(pass(name))))
                .collect()
        },
        ..PerfReport::default()
    }
}

/// Writes `report` to the recording's path, and every frame next to it as CSV.
fn write(recording: &PerfRecording, report: &PerfReport) {
    let path = &recording.path;
    let written = toml::to_string(report)
        .map_err(|error| error.to_string())
        .and_then(|text| write_file(path, &text));
    match written {
        Ok(()) => tracing::info!(
            "performance of {}: {:.1} fps over {} frames, frame {:.2} ms (p99 {:.2}), written to {}",
            report.label,
            report.fps,
            report.frames,
            report.frame.mean,
            report.frame.p99,
            path.display()
        ),
        Err(error) => tracing::error!("performance report {}: {error}", path.display()),
    }
    let csv = path.with_extension("csv");
    if let Err(error) = write_file(&csv, &frames_csv(&recording.samples)) {
        tracing::error!("performance frames {}: {error}", csv.display());
    }
}

/// Every frame, a line each, milliseconds; a frame whose GPU timings did not come back has
/// those columns empty.
fn frames_csv(samples: &[Sample]) -> String {
    let mut csv = String::from(
        "frame_ms,cpu_ms,pre_update_ms,fixed_update_ms,fixed_steps,update_ms,post_update_ms,render_ms,gpu_ms",
    );
    for pass in GPU_PASSES {
        let _ = write!(csv, ",{pass}_ms");
    }
    csv.push_str(",drawn,culled,draws\n");
    let ms = |seconds: Real| seconds * 1e3;
    for sample in samples {
        let cpu = &sample.cpu;
        let _ = write!(
            csv,
            "{:.4},{:.4},{:.4},{:.4},{},{:.4},{:.4},{:.4},",
            ms(sample.frame),
            ms(cpu.total()),
            ms(cpu.pre_update),
            ms(cpu.fixed_update),
            cpu.fixed_steps,
            ms(cpu.update),
            ms(cpu.post_update),
            ms(cpu.render),
        );
        if let Some(gpu) = &sample.gpu {
            let _ = write!(csv, "{:.4}", ms(gpu.total()));
        }
        for pass in GPU_PASSES {
            csv.push(',');
            if let Some(gpu) = &sample.gpu {
                let seconds = gpu
                    .passes
                    .iter()
                    .find(|(name, _)| *name == pass)
                    .map_or(0.0, |(_, seconds)| *seconds);
                let _ = write!(csv, "{:.4}", ms(seconds));
            }
        }
        let stats = &sample.stats;
        let _ = writeln!(csv, ",{},{},{}", stats.drawn, stats.culled, stats.draws);
    }
    csv
}

fn write_file(path: &Path, text: &str) -> Result<(), String> {
    if let Some(dir) = path.parent().filter(|dir| !dir.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|error| error.to_string())?;
    }
    std::fs::write(path, text).map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_spread_reads_its_percentiles_off_the_sorted_samples() {
        let spread = Spread::of((1..=100).rev().map(|ms| Real::from(ms) * 1e-3));
        assert!((spread.mean - 50.5).abs() < 1e-9);
        assert!((spread.p50 - 50.0).abs() < 1e-9);
        assert!((spread.p95 - 95.0).abs() < 1e-9);
        assert!((spread.p99 - 99.0).abs() < 1e-9);
        assert!((spread.max - 100.0).abs() < 1e-9);
        assert_eq!(Spread::of([]), Spread::default());
    }

    #[test]
    fn a_report_reads_back_as_it_was_written() {
        let mut recording = PerfRecording {
            path: PathBuf::new(),
            label: "field".to_owned(),
            warmup: 0.0,
            seconds: 1.0,
            timeout: 1.0,
            warmed: 0.0,
            held: 0.0,
            recorded: 0.04,
            last: None,
            gpu_serial: 0,
            samples: Vec::new(),
        };
        for i in 0..4 {
            recording.samples.push(Sample {
                frame: 0.01,
                cpu: FrameProfile {
                    update: 0.002,
                    render: 0.006,
                    ..FrameProfile::default()
                },
                gpu: (i % 2 == 0).then(|| GpuTimings {
                    serial: i,
                    passes: vec![("opaque", 0.007), ("post", 0.001)],
                }),
                stats: RenderStats {
                    drawn: 10,
                    culled: 2,
                    draws: 5,
                },
            });
        }
        let report = report(&recording);
        assert_eq!(report.frames, 4);
        assert!((report.fps - 100.0).abs() < 1e-9);
        assert!((report.gpu.unwrap().mean - 8.0).abs() < 1e-9);
        assert!((report.passes["opaque"].mean - 7.0).abs() < 1e-9);
        assert_eq!(report.passes["ink"].max, 0.0);
        // The GPU's 8 ms is longer than the CPU's 2 ms outside the wait: bound by the GPU.
        assert_eq!(report.bound().0, "GPU");
        let text = toml::to_string(&report).unwrap();
        let read: PerfReport = toml::from_str(&text).unwrap();
        assert_eq!(read, report);
        let csv = frames_csv(&recording.samples);
        assert_eq!(csv.lines().count(), 5);
        let columns = csv.lines().next().unwrap().split(',').count();
        assert!(csv.lines().all(|line| line.split(',').count() == columns));
    }
}
