//! Raw numerics for Fucklide.
//!
//! Nothing in this crate has geometric meaning: a vector here is a container of numbers, not a
//! point in space. Geometry lives in `fk-geometry` and the concrete geometry crates.
//!
//! All CPU-side geometry is computed in [`Real`] (`f64`). Coordinates in curved models grow
//! exponentially with distance (hyperboloid coordinates grow like `cosh d`), so single precision
//! is reserved for the GPU boundary, after poses have been made camera-relative.

pub mod minkowski;
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
