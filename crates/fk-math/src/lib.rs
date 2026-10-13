//! Raw numerics for Feuclide.
//!
//! Nothing in this crate has geometric meaning: a vector here is a container of numbers, not a
//! point in space. Geometry lives in `fk-geometry` and the concrete geometry crates.
//!
//! All CPU-side geometry is computed in [`Real`] (`f64`). Coordinates in curved models grow
//! exponentially with distance (hyperboloid coordinates grow like `cosh d`), so single precision
//! is reserved for the GPU boundary, after poses have been made camera-relative.

pub mod minkowski;
pub mod rand;
pub mod series;

pub use nalgebra;

/// Scalar type of every CPU-side geometric computation.
pub type Real = f64;

/// Converts a CPU vector to the `f32` array uploaded to the GPU.
///
/// Only call this on camera-relative quantities: far from the camera, curved-space coordinates
/// exceed what `f32` can represent faithfully.
pub fn to_gpu<const D: usize>(v: &nalgebra::SVector<Real, D>) -> [f32; D] {
    std::array::from_fn(|i| v[i] as f32)
}

/// Hermite interpolation from 0 at `edge0` to 1 at `edge1`, flat at both ends (GLSL's
/// `smoothstep`): `t² (3 − 2t)` with `t = (x − edge0) / (edge1 − edge0)` clamped to `[0, 1]`.
/// `edge1` may be below `edge0` (falling from 1 to 0); equal edges are a step at them.
pub fn smoothstep(edge0: Real, edge1: Real, x: Real) -> Real {
    if edge0 == edge1 {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoothstep_is_flat_at_both_ends_and_falls_when_reversed() {
        assert_eq!(smoothstep(1.0, 2.0, 0.5), 0.0);
        assert_eq!(smoothstep(1.0, 2.0, 1.5), 0.5);
        assert_eq!(smoothstep(1.0, 2.0, 3.0), 1.0);
        assert_eq!(smoothstep(2.0, 1.0, 1.25), 1.0 - smoothstep(1.0, 2.0, 1.25));
        assert_eq!(
            (smoothstep(1.0, 1.0, 0.9), smoothstep(1.0, 1.0, 1.0)),
            (0.0, 1.0)
        );
    }
}
