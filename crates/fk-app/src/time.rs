use bevy_ecs::resource::Resource;
use fk_math::Real;

/// Frame and fixed-step clocks, in seconds.
///
/// Every frame adds its duration, clamped to [`Time::MAX_DELTA`], to an accumulator;
/// [`FixedUpdate`](crate::FixedUpdate) runs once per whole [`Time::fixed_step`] in it. Inside
/// `FixedUpdate`, advance the simulation by [`Time::fixed_step`], not [`Time::delta`].
///
/// [Paused](Self::set_paused), frames still come (and are counted) but last nothing: every
/// system sees a zero [`delta`](Self::delta), no fixed step runs, and the world stands still
/// while it is still drawn.
#[derive(Resource, Clone, Debug)]
pub struct Time {
    delta: Real,
    elapsed: Real,
    frame: u64,
    fixed_step: Real,
    accumulator: Real,
    paused: bool,
}

impl Time {
    /// Longest frame the clocks accept. A longer frame (a stall, a breakpoint, a dragged window)
    /// counts as this long, so the simulation does not try to catch up all at once.
    pub const MAX_DELTA: Real = 0.25;

    /// Default fixed step: 60 Hz.
    pub const DEFAULT_FIXED_STEP: Real = 1.0 / 60.0;

    /// Clocks at zero with the given fixed step.
    ///
    /// # Panics
    ///
    /// If `fixed_step` is not positive and finite.
    pub fn new(fixed_step: Real) -> Self {
        assert!(
            fixed_step > 0.0 && fixed_step.is_finite(),
            "fixed step must be positive and finite, got {fixed_step}"
        );
        Self {
            delta: 0.0,
            elapsed: 0.0,
            frame: 0,
            fixed_step,
            accumulator: 0.0,
            paused: false,
        }
    }

    /// Duration of the current frame.
    pub fn delta(&self) -> Real {
        self.delta
    }

    /// Sum of all frame durations so far.
    pub fn elapsed(&self) -> Real {
        self.elapsed
    }

    /// Number of the current frame, starting at 1 for the first frame.
    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// Duration of one [`FixedUpdate`](crate::FixedUpdate) run.
    pub fn fixed_step(&self) -> Real {
        self.fixed_step
    }

    /// How far the frame clock is past the last fixed step, as a fraction of a step in `[0, 1)`.
    /// Use it to interpolate between the last two fixed states when drawing.
    pub fn overstep_fraction(&self) -> Real {
        self.accumulator / self.fixed_step
    }

    /// Stops time, or starts it again: see the type's documentation.
    pub fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
    }

    /// Whether time stands still.
    pub fn is_paused(&self) -> bool {
        self.paused
    }

    /// Starts a frame that lasted `raw_delta` seconds (none, paused).
    pub(crate) fn advance(&mut self, raw_delta: Real) {
        self.delta = if self.paused {
            0.0
        } else {
            raw_delta.clamp(0.0, Self::MAX_DELTA)
        };
        self.elapsed += self.delta;
        self.frame += 1;
        self.accumulator += self.delta;
    }

    /// Consumes one fixed step from the accumulator if a whole one is there.
    pub(crate) fn take_fixed_step(&mut self) -> bool {
        if self.accumulator >= self.fixed_step {
            self.accumulator -= self.fixed_step;
            true
        } else {
            false
        }
    }
}

impl Default for Time {
    fn default() -> Self {
        Self::new(Self::DEFAULT_FIXED_STEP)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn steps(time: &mut Time) -> usize {
        std::iter::from_fn(|| time.take_fixed_step().then_some(())).count()
    }

    #[test]
    fn fixed_steps_follow_accumulated_time() {
        // Powers of two, so the accumulator arithmetic is exact.
        let mut time = Time::new(0.0625);
        time.advance(0.15625);
        assert_eq!(steps(&mut time), 2);
        assert_eq!(time.overstep_fraction(), 0.5);
        time.advance(0.03125);
        assert_eq!(steps(&mut time), 1);
        assert_eq!(time.overstep_fraction(), 0.0);
        assert_eq!(time.frame(), 2);
        assert_eq!(time.elapsed(), 0.1875);
    }

    #[test]
    fn long_frames_are_clamped() {
        let mut time = Time::new(0.0625);
        time.advance(10.0);
        assert_eq!(time.delta(), Time::MAX_DELTA);
        assert_eq!(steps(&mut time), 4);
        time.advance(-1.0);
        assert_eq!(time.delta(), 0.0);
    }

    #[test]
    fn paused_frames_last_nothing_and_run_no_fixed_step() {
        let mut time = Time::new(0.1);
        time.set_paused(true);
        time.advance(0.5);
        assert_eq!((time.delta(), time.elapsed(), time.frame()), (0.0, 0.0, 1));
        assert!(!time.take_fixed_step());
        time.set_paused(false);
        time.advance(0.15);
        assert!(time.take_fixed_step());
        assert!(!time.is_paused());
    }
}
