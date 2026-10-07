use std::fmt::Debug;
use std::ops::{Add, Mul, Neg, Sub};

use fk_math::Real;
use fk_math::nalgebra::SVector;

/// A real vector space with its coordinates: tangent vectors and Lie algebra elements.
///
/// There is deliberately no inner product here. Lengths of tangent vectors depend on the base
/// point and come from [`Geometry::inner`](crate::Geometry::inner).
pub trait VectorSpace:
    Copy
    + Debug
    + Send
    + Sync
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Neg<Output = Self>
    + Mul<Real, Output = Self>
{
    /// The zero vector.
    fn zero() -> Self;

    /// Sum of squared coordinates.
    ///
    /// This is not a metric quantity; it only exists to compare elements numerically, for
    /// example in tests.
    fn coord_norm_squared(&self) -> Real;
}

impl<const D: usize> VectorSpace for SVector<Real, D> {
    fn zero() -> Self {
        Self::zeros()
    }

    fn coord_norm_squared(&self) -> Real {
        self.norm_squared()
    }
}
