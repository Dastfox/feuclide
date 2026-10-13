//! Sounds made rather than recorded: sample a function of time into frames
//! ([`Sound::from_samples`](crate::Sound::from_samples) takes them), with sines that loop
//! without a click, seeded white noise, one-pole low-passes and a soft limiter.

use std::f64::consts::TAU;

use fk_math::Real;
use fk_math::rand::Seeded;

/// `seconds` of `sample(t)` at `rate` frames a second, clamped to `[−1, 1]`, the same in both
/// ears.
pub fn mono(rate: u32, seconds: Real, mut sample: impl FnMut(Real) -> Real) -> Vec<[f32; 2]> {
    let count = (seconds * Real::from(rate)).round() as usize;
    (0..count)
        .map(|i| {
            let s = sample(i as Real / Real::from(rate)).clamp(-1.0, 1.0) as f32;
            [s, s]
        })
        .collect()
}

/// A sum of sines at `t` seconds, `(hertz, amplitude)` each: whole numbers of cycles in a loop
/// make it loop without a click.
pub fn partials(t: Real, partials: &[(Real, Real)]) -> Real {
    partials
        .iter()
        .map(|&(hz, amp)| amp * (TAU * hz * t).sin())
        .sum()
}

/// A one-pole low-pass's coefficient at `cutoff` hertz, for `rate` frames a second: each frame
/// `y += k · (x − y)`.
pub fn one_pole(rate: u32, cutoff: Real) -> Real {
    1.0 - (-TAU * cutoff / Real::from(rate)).exp()
}

/// A gentle limiter: loud stays under 1.
pub fn soft_clip(x: Real) -> Real {
    x.tanh()
}

/// White noise in `[−1, 1)`, the same for the same seed.
#[derive(Clone, Debug)]
pub struct Noise(Seeded);

impl Noise {
    /// Noise from `seed`.
    pub fn new(seed: u64) -> Self {
        Self(Seeded::new(seed))
    }

    /// The next frame's.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Real {
        self.0.next() * 2.0 - 1.0
    }
}

/// `seconds` of white noise from `seed` at `rate` frames a second, smoothed by a one-pole
/// low-pass at `cutoff` hertz: a rumble.
pub fn rumble(rate: u32, seconds: Real, cutoff: Real, seed: u64) -> Vec<Real> {
    let mut noise = Noise::new(seed);
    let k = one_pole(rate, cutoff);
    let mut y = 0.0;
    let count = (seconds * Real::from(rate)).round() as usize;
    (0..count)
        .map(|_| {
            y += k * (noise.next() - y);
            y
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tone_loops_and_noise_stays_in_range() {
        let frames = mono(1000, 1.0, |t| partials(t, &[(5.0, 2.0)]));
        assert_eq!(frames.len(), 1000);
        assert!(frames.iter().all(|[l, r]| l == r && l.abs() <= 1.0));
        assert!(
            partials(1.0, &[(5.0, 1.0)]).abs() < 1e-9,
            "whole cycles end at 0"
        );
        let mut noise = Noise::new(3);
        assert!(
            (0..1000)
                .map(|_| noise.next())
                .all(|x| (-1.0..1.0).contains(&x))
        );
        let low = rumble(1000, 1.0, 20.0, 9);
        assert!(low.iter().all(|x| x.abs() < 1.0));
        assert!(soft_clip(10.0) < 1.0);
    }
}
