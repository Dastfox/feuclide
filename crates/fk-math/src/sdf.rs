//! Signed distance primitives and helpers, the CPU twins of `fk-shaders`' `fk::sdf_lib`: a
//! game's field written in WGSL and measured on the CPU (to collide with it) agrees with them.

use nalgebra::{Vector2, Vector3};

use crate::Real;

/// A box's signed distance: half-extents `half` about the origin.
pub fn cuboid(p: &Vector3<Real>, half: &Vector3<Real>) -> Real {
    let q = p.abs() - half;
    q.map(|x| x.max(0.0)).norm() + q.max().min(0.0)
}

/// An upright capped cylinder about the origin's vertical: `radius`, from `bottom` to `top`.
pub fn column(p: &Vector3<Real>, radius: Real, bottom: Real, top: Real) -> Real {
    let across = Vector2::new(p.x, p.z).norm() - radius;
    let along = (p.y - (top + bottom) / 2.0).abs() - (top - bottom).abs() / 2.0;
    Vector2::new(across.max(0.0), along.max(0.0)).norm() + across.max(along).min(0.0)
}

/// A cell's hash: the same bits on the CPU and in the shader (`fk::sdf_lib`'s `cell_hash`).
pub fn cell_hash(i: i32, j: i32, seed: u32) -> u32 {
    let mut h = (i as u32).wrapping_mul(0x8da6_b343)
        ^ (j as u32).wrapping_mul(0xd816_3841)
        ^ seed.wrapping_mul(0xcb1a_b31f);
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1_e995);
    h ^ (h >> 15)
}

/// `q` folded into copies repeated every `period` along each axis, one centred on every
/// multiple of it: which copy, and `q` from its middle (`repeat_index`, `repeat_local`).
pub fn repeated(q: &Vector3<Real>, period: &Vector3<Real>) -> (Vector3<Real>, Vector3<Real>) {
    let index = q.component_div(period).map(Real::round);
    (index, q - index.component_mul(period))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primitives_measure_what_they_say() {
        let half = Vector3::new(1.0, 2.0, 3.0);
        assert!((cuboid(&Vector3::new(2.0, 0.0, 0.0), &half) - 1.0).abs() < 1e-12);
        assert!((cuboid(&Vector3::zeros(), &half) + 1.0).abs() < 1e-12);
        assert!((column(&Vector3::new(3.0, 1.0, 0.0), 1.0, 0.0, 2.0) - 2.0).abs() < 1e-12);
        assert_ne!(cell_hash(1, 2, 3), cell_hash(2, 1, 3));
        let (index, local) = repeated(&Vector3::new(9.0, -1.0, 0.4), &Vector3::new(4.0, 4.0, 1.0));
        assert_eq!(index, Vector3::new(2.0, -0.0, 0.0));
        assert!((local - Vector3::new(1.0, -1.0, 0.4)).norm() < 1e-12);
    }
}
