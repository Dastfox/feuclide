//! Small, seeded randomness: the same sequence for the same seed, for layouts that must come
//! out the same every run and for what only needs to look random. Not for cryptography.

use nalgebra::Vector3;

use crate::Real;

/// A xorshift sequence of numbers: the same for the same seed.
#[derive(Clone, Debug)]
pub struct Seeded(pub u64);

impl Seeded {
    /// A sequence starting from `seed` (0 is taken as 1: xorshift never leaves 0).
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    /// The next 64 bits.
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.0;
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }

    /// The next number, in `[0, 1)`.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Real {
        (self.next_u64() >> 11) as Real / (1u64 << 53) as Real
    }

    /// The next number, in `[0, 1)`: [`next`](Self::next).
    pub fn unit(&mut self) -> Real {
        self.next()
    }

    /// The next number, between `a` and `b`.
    pub fn between(&mut self, [a, b]: [Real; 2]) -> Real {
        a + (b - a) * self.next()
    }

    /// The next number between `low` and `high` (both above 0), as likely in each doubling:
    /// more small than large.
    pub fn log_between(&mut self, [low, high]: [Real; 2]) -> Real {
        let low = low.max(1e-6);
        low * (high.max(low) / low).powf(self.next())
    }

    /// A whole number from `least` to `most`, both included.
    pub fn count(&mut self, [least, most]: [u32; 2]) -> u32 {
        let most = most.max(least);
        (least + (Real::from(most - least + 1) * self.next()) as u32).min(most)
    }

    /// A seed for something grown from this sequence.
    pub fn seed(&mut self) -> u64 {
        (self.next() * Real::from(u32::MAX)) as u64 + 1
    }

    /// A direction in the plane of x and z, `(cos a, 0, sin a)` for a random `a`.
    pub fn heading(&mut self) -> Vector3<Real> {
        let a = std::f64::consts::TAU * self.next();
        Vector3::new(a.cos(), 0.0, a.sin())
    }

    /// A direction uniform on the unit sphere.
    pub fn direction(&mut self) -> Vector3<Real> {
        let y = 2.0 * self.next() - 1.0;
        let a = std::f64::consts::TAU * self.next();
        let r = (1.0 - y * y).max(0.0).sqrt();
        Vector3::new(r * a.cos(), y, r * a.sin())
    }
}

/// One step of the splitmix64 hash: a well-mixed `u64` from any `u64`, the same each time.
pub fn splitmix64(x: u64) -> u64 {
    let x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    let x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_seed_gives_the_same_numbers_in_range() {
        let (mut a, mut b) = (Seeded::new(7), Seeded::new(7));
        for _ in 0..1000 {
            let x = a.next();
            assert_eq!(x, b.next());
            assert!((0.0..1.0).contains(&x));
        }
        let mut c = Seeded::new(0);
        assert!((0..100).all(|_| c.count([2, 4]) <= 4));
        assert!((a.direction().norm() - 1.0).abs() < 1e-12);
        assert!((1.0..=8.0).contains(&a.log_between([1.0, 8.0])));
        assert_ne!(splitmix64(1), splitmix64(2));
    }
}
