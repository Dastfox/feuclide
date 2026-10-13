//! A walker's gait: when its feet strike, at a pace of its own whatever its speed, for
//! footsteps and the head's bob.

use std::f64::consts::PI;

use bevy_ecs::resource::Resource;
use fk_math::Real;

/// Where a walker's feet are, each frame: what its [`Gait`] follows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stride {
    /// Distance moved across the ground since the start, on it or just off it (flying does not
    /// count); it only grows. The feet keep their pace while it grows, and stop when it does
    /// not.
    pub travelled: Real,
    /// Whether the feet are on the ground. Landing from a jump is a step at once.
    pub grounded: bool,
    /// Whether it is walking or running of its own accord (a move held): only then do its feet
    /// step. Pushed, slid or nudged out of a surface, it goes along without a step.
    pub walking: bool,
}

impl Default for Stride {
    fn default() -> Self {
        Self {
            travelled: 0.0,
            grounded: true,
            walking: false,
        }
    }
}

/// How a [`Gait`] follows its [`Stride`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaitTuning {
    /// Seconds from setting off to the first strike.
    pub first_step: Real,
    /// Seconds standing after which the next setting off starts a fresh stride.
    pub settle_time: Real,
    /// Seconds off the ground that make a jump; shorter lifts are still on its feet.
    pub airtime: Real,
    /// Speed under which it stands still, whatever creeps it (units a second).
    pub least_speed: Real,
    /// Seconds its speed is smoothed over.
    pub smoothing: Real,
}

/// A walker's gait, as a resource: when its feet strike, for footsteps and a head's bob.
///
/// While its [`Stride`] says it moves across the ground, its feet strike at a pace of its own
/// (a cadence in steps a second, a human one say), whatever its speed in the world; the first
/// soon after setting off, and one on landing from a jump. Lifts shorter than
/// [`GaitTuning::airtime`] (a bump, a slope going down) are still on its feet: the steps go on
/// through them, on time.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct Gait {
    /// Seconds to the next strike.
    to_step: Real,
    /// Seconds between strikes at the cadence now.
    period: Real,
    /// How much it is walking, eased: 0 standing, 1 on its way.
    moving: Real,
    /// Its speed across the ground, smoothed over [`GaitTuning::smoothing`], in units a second.
    speed: Real,
    /// The distance travelled last frame.
    last: Option<Real>,
    /// Seconds without moving.
    still_for: Real,
    /// Seconds off the ground.
    airborne_for: Real,
}

impl Default for Gait {
    fn default() -> Self {
        Self {
            to_step: Real::INFINITY,
            period: 1.0,
            moving: 0.0,
            speed: 0.0,
            last: None,
            // Standing from the start: the first step comes soon.
            still_for: Real::INFINITY,
            airborne_for: 0.0,
        }
    }
}

/// Seconds the bob takes to come in when setting off, and to settle when stopping.
const EASE: Real = 0.25;

impl Gait {
    /// Share of the way from the last strike to the next, 0 at a strike.
    pub fn phase(&self) -> Real {
        if self.to_step.is_finite() {
            (1.0 - self.to_step / self.period).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// Its speed across the ground, smoothed, in units a second.
    pub fn speed(&self) -> Real {
        self.speed
    }

    /// Seconds it has been off the ground.
    pub fn airborne_for(&self) -> Real {
        self.airborne_for
    }

    /// How much it is walking, eased: 0 standing, 1 on its way.
    pub fn moving(&self) -> Real {
        self.moving
    }

    /// The head's bob along the up axis: lowest, `-depth`, as a foot strikes, level between
    /// strikes, and none standing.
    pub fn bob(&self, depth: Real) -> Real {
        -depth * self.moving * (PI * self.phase()).cos().powi(2)
    }

    /// Follows `stride` for `dt` seconds at `cadence` steps a second; whether a foot strikes.
    pub fn step(&mut self, stride: Stride, cadence: Real, tuning: &GaitTuning, dt: Real) -> bool {
        let footing = stride;
        let dt = dt.max(1e-6);
        let last = self
            .last
            .replace(footing.travelled)
            .unwrap_or(footing.travelled);
        // A jump is a lift longer than `airtime`; landing from it is a step.
        let landed = footing.grounded && self.airborne_for >= tuning.airtime;
        self.airborne_for = if footing.grounded {
            0.0
        } else {
            self.airborne_for + dt
        };
        let on_feet = self.airborne_for < tuning.airtime;
        // Smoothed: the distance comes in uneven frames, now and then none in one and twice as
        // much in the next. Standing, the body still creeps a little (pushed out of a slope or a
        // wall it leans on): under `least_speed` the feet stay put.
        let speed = (footing.travelled - last).max(0.0) / dt;
        self.speed += (speed - self.speed) * (1.0 - (-dt / tuning.smoothing.max(1e-3)).exp());
        let on_the_way = on_feet && footing.walking && self.speed > tuning.least_speed;
        let ease = dt / EASE;
        self.moving = if on_the_way {
            (self.moving + ease).min(1.0)
        } else {
            (self.moving - ease).max(0.0)
        };
        // From walking to running and back, the stride keeps its phase.
        let period = 1.0 / cadence.max(1e-3);
        if self.to_step.is_finite() {
            self.to_step *= period / self.period;
        }
        self.period = period;
        self.still_for = if on_the_way { 0.0 } else { self.still_for + dt };
        if self.still_for > tuning.settle_time {
            // Set off again: the first foot comes down soon.
            self.to_step = self.to_step.min(tuning.first_step);
        }
        if landed {
            self.to_step = period;
            return true;
        }
        if !on_the_way {
            return false;
        }
        self.to_step -= dt;
        if self.to_step > 0.0 {
            return false;
        }
        // Late by a long frame, the next keeps the beat rather than the delay.
        self.to_step = (self.to_step + period).max(0.25 * period);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: Real = 1.0 / 60.0;
    const TUNING: GaitTuning = GaitTuning {
        first_step: 0.15,
        settle_time: 0.3,
        airtime: 0.2,
        least_speed: 0.5,
        smoothing: 0.05,
    };

    #[test]
    fn the_feet_keep_their_pace_whatever_the_speed_and_land_once() {
        let cadence = 1.8;
        let mut gait = Gait::default();
        let mut stride = Stride::default();
        let walk = |gait: &mut Gait, stride: &mut Stride, speed: Real, s: Real| {
            let mut count = 0;
            for _ in 0..(s / DT).round() as usize {
                stride.walking = speed > 0.0;
                stride.travelled += speed * DT;
                count += usize::from(gait.step(*stride, cadence, &TUNING, DT));
            }
            count
        };
        assert_eq!(walk(&mut gait, &mut stride, 0.0, 1.0), 0, "standing still");
        assert_eq!(gait.bob(0.02), 0.0);
        assert_eq!(
            walk(&mut gait, &mut stride, 4.0, 0.2),
            1,
            "the first step comes soon"
        );
        for speed in [1.5, 4.0, 9.0] {
            let many = walk(&mut gait, &mut stride, speed, 10.0 / cadence);
            assert!((9..=11).contains(&many), "{many} steps at {speed}");
        }
        stride.grounded = false;
        walk(&mut gait, &mut stride, 4.0, TUNING.airtime);
        assert_eq!(walk(&mut gait, &mut stride, 4.0, 0.5), 0, "in a jump");
        stride.grounded = true;
        assert!(gait.step(stride, cadence, &TUNING, DT), "landing");
    }
}
