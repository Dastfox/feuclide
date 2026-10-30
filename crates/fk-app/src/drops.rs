//! Frame drops as the app sees them, and what a frame did, for whoever records why.
//!
//! The runner watches the frame rate ([`FrameWatch`]): it logs its drops and tells the app
//! through [`FrameDrops`] before each frame. [`FrameEvents`] counts what happened in a frame
//! (the engine's and the game's own: a pipeline built, a handover), [`FrameNotes`] holds what a
//! game says of where it is; neither is read by the engine but to be recorded with a drop
//! (`fk_render::drops`).

use std::collections::BTreeMap;
use std::time::Instant;

use bevy_ecs::resource::Resource;
use fk_math::Real;

/// The frame drops seen so far, as a resource the runner writes before each frame, of the frame
/// before it (the one [`FrameProfile`](crate::FrameProfile) is of). Counters, as events are:
/// compare them with what was last seen.
///
/// A *spike* is a frame lasting [`FrameWatch::RATIO`] times the usual frame and longer than
/// [`FrameWatch::LEAST`]: something that happened. A *sag* is the usual frame itself over the
/// target's ([`FrameWatch::SAG`] times a frame at `FK_DROPS_TARGET` frames a second, 60 by
/// default) for [`FrameWatch::SAG_HOLD`] seconds: a scene too heavy. An app without a window
/// sees none.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct FrameDrops {
    /// Frames that were spikes, so far.
    pub spikes: u64,
    /// Sags begun, so far.
    pub sags: u64,
    /// Whether the frames are sagging now.
    pub sagging: bool,
    /// The usual frame's length, seconds: a moving average of the frames that were not spikes.
    pub usual: Real,
    /// The last frame's length, seconds.
    pub last: Real,
}

/// What happened in a frame, by name, counted, as a resource: the engine's own (`handover`,
/// `pipelines built`…) and whatever an app adds.
///
/// [`App::update`](crate::App::update) ends the frame after [`Render`](crate::Render): what was
/// added becomes [`last`](Self::last), so that a system reads in a frame what the frame before
/// it did, as with [`FrameProfile`](crate::FrameProfile).
#[derive(Resource, Clone, Debug, Default, PartialEq, Eq)]
pub struct FrameEvents {
    now: BTreeMap<&'static str, u32>,
    last: BTreeMap<&'static str, u32>,
}

impl FrameEvents {
    /// Counts `count` more of `name` in this frame.
    pub fn add(&mut self, name: &'static str, count: u32) {
        if count > 0 {
            *self.now.entry(name).or_default() += count;
        }
    }

    /// How many of `name` the last whole frame counted.
    pub fn count(&self, name: &str) -> u32 {
        self.last.get(name).copied().unwrap_or(0)
    }

    /// Everything the last whole frame counted, by name.
    pub fn last(&self) -> impl Iterator<Item = (&'static str, u32)> + '_ {
        self.last.iter().map(|(name, count)| (*name, *count))
    }

    /// Ends the frame: what was counted becomes the last frame's.
    pub(crate) fn end_frame(&mut self) {
        std::mem::swap(&mut self.now, &mut self.last);
        self.now.clear();
    }
}

/// What an app says of where it is, by name, as text, as a resource: kept until changed, never
/// read by the engine but to be recorded with a drop. Set each only when it changes: it is
/// compared first, and copied only then.
#[derive(Resource, Clone, Debug, Default, PartialEq, Eq)]
pub struct FrameNotes(BTreeMap<&'static str, String>);

impl FrameNotes {
    /// Sets `name` to `value`.
    pub fn set(&mut self, name: &'static str, value: &str) {
        match self.0.get_mut(name) {
            Some(old) if old == value => {}
            Some(old) => value.clone_into(old),
            None => {
                self.0.insert(name, value.to_owned());
            }
        }
    }

    /// Takes `name` out.
    pub fn remove(&mut self, name: &str) {
        self.0.remove(name);
    }

    /// The value of `name`, if set.
    pub fn get(&self, name: &str) -> Option<&str> {
        self.0.get(name).map(String::as_str)
    }

    /// Every note, by name.
    pub fn iter(&self) -> impl Iterator<Item = (&'static str, &str)> + '_ {
        self.0.iter().map(|(name, value)| (*name, value.as_str()))
    }
}

/// Watches the frame rate and logs its drops: a spike is logged at once, then those that follow
/// within [`Self::QUIET`] are counted and logged together, so that a bad stretch is a few lines
/// and not hundreds; a sag is logged as it begins.
#[derive(Debug)]
pub(crate) struct FrameWatch {
    /// The usual frame's length, seconds: a moving average of the frames that were not spikes.
    usual: Option<Real>,
    /// Spikes not logged yet: how many, the longest, and when the first was.
    pending: Option<(u32, Real, Instant)>,
    /// When a spike was last logged.
    logged: Option<Instant>,
    /// The usual frame over which the frames sag, seconds.
    sag_over: Real,
    /// Since when the usual frame has been over `sag_over`, while it is.
    slow_since: Option<Instant>,
    drops: FrameDrops,
}

impl Default for FrameWatch {
    fn default() -> Self {
        let target = std::env::var("FK_DROPS_TARGET")
            .ok()
            .and_then(|text| text.parse::<Real>().ok())
            .filter(|fps| *fps > 0.0)
            .unwrap_or(60.0);
        Self::new(target)
    }
}

impl FrameWatch {
    /// A frame is a spike when it lasts this many times the usual frame...
    pub(crate) const RATIO: Real = 2.0;
    /// ...and at least this long, seconds (below 30 frames a second).
    pub(crate) const LEAST: Real = 1.0 / 30.0;
    /// How much each frame that is not a spike counts in the usual frame.
    const WEIGHT: Real = 0.05;
    /// Seconds after a spike is logged during which spikes are counted rather than logged.
    const QUIET: Real = 1.0;
    /// The usual frame sags over this many times the target's...
    const SAG: Real = 1.15;
    /// ...for this many seconds; it is over when the usual frame is back under the target's.
    const SAG_HOLD: Real = 1.0;

    /// A watch whose sags are under `target` frames a second.
    pub(crate) fn new(target: Real) -> Self {
        Self {
            usual: None,
            pending: None,
            logged: None,
            sag_over: Self::SAG / target,
            slow_since: None,
            drops: FrameDrops::default(),
        }
    }

    /// What was seen, for the app.
    pub(crate) fn drops(&self) -> FrameDrops {
        self.drops
    }

    /// Counts a frame of `delta` seconds, drawn at `now`.
    pub(crate) fn frame(&mut self, delta: Real, now: Instant) {
        let usual = *self.usual.get_or_insert(delta);
        if delta >= Self::LEAST && delta > Self::RATIO * usual {
            let (count, longest, _) = self.pending.get_or_insert((0, 0.0, now));
            *count += 1;
            *longest = longest.max(delta);
            self.drops.spikes += 1;
        } else {
            self.usual = Some(usual + (delta - usual) * Self::WEIGHT);
        }
        self.drops.usual = self.usual.unwrap_or(delta);
        self.drops.last = delta;
        self.sag(now);
        let quiet = self
            .logged
            .is_some_and(|logged| now.duration_since(logged).as_secs_f64() < Self::QUIET);
        if quiet {
            return;
        }
        if let Some((count, longest, first)) = self.pending.take() {
            if count == 1 {
                tracing::warn!(
                    "frame drop: {:.0} ms ({:.0} fps), usually {:.1} ms ({:.0} fps)",
                    longest * 1e3,
                    fps(longest),
                    usual * 1e3,
                    fps(usual),
                );
            } else {
                tracing::warn!(
                    "frame drops: {count} in {:.1}s, the longest {:.0} ms ({:.0} fps), \
                     usually {:.1} ms ({:.0} fps)",
                    now.duration_since(first).as_secs_f64(),
                    longest * 1e3,
                    fps(longest),
                    usual * 1e3,
                    fps(usual),
                );
            }
            self.logged = Some(now);
        }
    }

    /// Follows the usual frame against the target's: a sag begins once it has been over long
    /// enough, and ends once it is back under the target's.
    fn sag(&mut self, now: Instant) {
        let usual = self.drops.usual;
        if self.drops.sagging {
            if usual < self.sag_over / Self::SAG {
                self.drops.sagging = false;
                self.slow_since = None;
            }
            return;
        }
        if usual <= self.sag_over {
            self.slow_since = None;
            return;
        }
        let since = *self.slow_since.get_or_insert(now);
        let slow = now.duration_since(since).as_secs_f64();
        if slow >= Self::SAG_HOLD {
            self.drops.sagging = true;
            self.drops.sags += 1;
            tracing::warn!(
                "frames slow for {slow:.1}s: usually {:.1} ms ({:.0} fps), the target {:.1} ms",
                usual * 1e3,
                fps(usual),
                self.sag_over / Self::SAG * 1e3,
            );
        }
    }
}

/// Frames a second of frames lasting `seconds`.
fn fps(seconds: Real) -> Real {
    1.0 / seconds.max(1e-6)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    /// Frames of the given lengths, one after the other.
    fn stepper() -> impl FnMut(&mut FrameWatch, Real) {
        let mut now = Instant::now();
        move |watch: &mut FrameWatch, delta: Real| {
            now += Duration::from_secs_f64(delta);
            watch.frame(delta, now);
        }
    }

    #[test]
    fn a_drop_does_not_become_the_usual_frame() {
        let mut watch = FrameWatch::new(60.0);
        let mut step = stepper();
        for _ in 0..100 {
            step(&mut watch, 1.0 / 60.0);
        }
        step(&mut watch, 0.5);
        assert!(watch.pending.is_none(), "the first drop is logged at once");
        assert!(watch.logged.is_some());
        step(&mut watch, 0.2);
        step(&mut watch, 0.3);
        assert_eq!(
            watch.pending.map(|(count, longest, _)| (count, longest)),
            Some((2, 0.3))
        );
        assert!((watch.usual.unwrap() - 1.0 / 60.0).abs() < 1e-9);
        assert_eq!(watch.drops().spikes, 3);
        assert_eq!(watch.drops().last, 0.3);
        for _ in 0..60 {
            step(&mut watch, 1.0 / 60.0);
        }
        assert!(
            watch.pending.is_none(),
            "counted drops are logged after the quiet"
        );
        assert_eq!(watch.drops().sags, 0, "spikes are not a sag");
    }

    #[test]
    fn a_sag_is_seen_after_a_second_slow_and_once() {
        let mut watch = FrameWatch::new(60.0);
        let mut step = stepper();
        for _ in 0..100 {
            step(&mut watch, 1.0 / 60.0);
        }
        // 40 fps: never a spike, the usual frame follows it over the target's.
        for _ in 0..30 {
            step(&mut watch, 1.0 / 40.0);
        }
        assert_eq!(watch.drops().sags, 0, "not slow long enough yet");
        for _ in 0..120 {
            step(&mut watch, 1.0 / 40.0);
        }
        assert_eq!(watch.drops().spikes, 0);
        assert_eq!(watch.drops().sags, 1, "logged once however long it lasts");
        assert!(watch.drops().sagging);
        for _ in 0..200 {
            step(&mut watch, 1.0 / 60.0);
        }
        assert!(!watch.drops().sagging, "over once back at the target");
        for _ in 0..200 {
            step(&mut watch, 1.0 / 40.0);
        }
        assert_eq!(watch.drops().sags, 2, "a new sag counted");
    }

    #[test]
    fn events_are_the_last_frames() {
        let mut events = FrameEvents::default();
        events.add("handover", 1);
        events.add("handover", 2);
        events.add("nothing", 0);
        assert_eq!(events.count("handover"), 0, "not over yet");
        events.end_frame();
        assert_eq!(events.count("handover"), 3);
        assert_eq!(events.last().count(), 1);
        events.end_frame();
        assert_eq!(events.count("handover"), 0, "cleared after the frame");
    }

    #[test]
    fn notes_keep_the_last_value() {
        let mut notes = FrameNotes::default();
        notes.set("place", "road");
        notes.set("place", "road");
        notes.set("place", "hut");
        assert_eq!(notes.get("place"), Some("hut"));
        notes.remove("place");
        assert_eq!(notes.iter().count(), 0);
    }
}
