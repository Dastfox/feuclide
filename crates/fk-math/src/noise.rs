//! Value noise over the unit square, wrapping: tileable height fields for textures and
//! normal maps, the same for the same seed.

use crate::Real;

/// A number in [0, 1) for the lattice point `(x, y)`.
pub fn lattice_hash(x: i64, y: i64, seed: u64) -> Real {
    let mut h = (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ seed.wrapping_mul(0x1656_67B1_9E37_79F9);
    h ^= h >> 31;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 29;
    (h >> 11) as Real / (1u64 << 53) as Real
}

/// Value noise over the unit square on a lattice `cells` across, wrapping, in [0, 1).
pub fn value(u: Real, v: Real, cells: u32, seed: u64) -> Real {
    let c = i64::from(cells.max(1));
    let (x, y) = (u * c as Real, v * c as Real);
    let (i, j) = (x.floor() as i64, y.floor() as i64);
    let (fx, fy) = (x - i as Real, y - j as Real);
    let smooth = |t: Real| t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
    let (sx, sy) = (smooth(fx), smooth(fy));
    let at = |di: i64, dj: i64| lattice_hash((i + di).rem_euclid(c), (j + dj).rem_euclid(c), seed);
    let bottom = at(0, 0) + (at(1, 0) - at(0, 0)) * sx;
    let top = at(0, 1) + (at(1, 1) - at(0, 1)) * sx;
    bottom + (top - bottom) * sy
}

/// Octaves of [`value`] noise from `cells` across, each twice as fine and its height halved,
/// about −1 to 1, each octave's height over its cells so every octave slopes alike.
pub fn fbm(u: Real, v: Real, cells: u32, octaves: u32, seed: u64) -> Real {
    let mut sum = 0.0;
    for o in 0..octaves {
        let c = cells << o;
        let weight = 0.5f64.powi(o as i32) * 2.0 / Real::from(cells);
        sum += (2.0 * value(u, v, c, seed + u64::from(o)) - 1.0) * weight;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_wraps_and_stays_in_range() {
        for k in 0..100 {
            let (u, v) = (Real::from(k) * 0.0137, Real::from(k) * 0.031);
            let x = value(u, v, 8, 3);
            assert!((0.0..1.0).contains(&x));
            assert!((value(u + 1.0, v, 8, 3) - x).abs() < 1e-9, "wraps");
            assert!(fbm(u, v, 4, 3, 1).abs() < 1.0);
        }
    }
}
