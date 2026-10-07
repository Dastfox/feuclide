//! The Minkowski bilinear form of signature `(-, +, …, +)`, with the time-like coordinate first.
//!
//! This is the ambient form of the hyperboloid model of hyperbolic space and of Lorentzian planes.

use nalgebra::SVector;

use crate::Real;

/// `⟨a, b⟩ = -a₀b₀ + a₁b₁ + … + a_{D-1}b_{D-1}`.
pub fn dot<const D: usize>(a: &SVector<Real, D>, b: &SVector<Real, D>) -> Real {
    let space: Real = (1..D).map(|i| a[i] * b[i]).sum();
    space - a[0] * b[0]
}

/// `⟨v, v⟩`, which is negative for time-like vectors.
pub fn norm_squared<const D: usize>(v: &SVector<Real, D>) -> Real {
    dot(v, v)
}

/// Flips the sign of the time-like coordinate, turning a Minkowski dual into a Euclidean one.
pub fn flip_time<const D: usize>(v: &SVector<Real, D>) -> SVector<Real, D> {
    let mut out = *v;
    out[0] = -out[0];
    out
}

#[cfg(test)]
mod tests {
    use nalgebra::Vector4;

    use super::*;

    #[test]
    fn signature() {
        let t = Vector4::new(1.0, 0.0, 0.0, 0.0);
        let x = Vector4::new(0.0, 1.0, 0.0, 0.0);
        assert_eq!(norm_squared(&t), -1.0);
        assert_eq!(norm_squared(&x), 1.0);
        assert_eq!(dot(&t, &x), 0.0);
    }

    #[test]
    fn flip_time_relates_to_euclidean_dot() {
        let a = Vector4::new(2.0, 1.0, -3.0, 0.5);
        let b = Vector4::new(-1.0, 4.0, 2.0, 7.0);
        assert_eq!(dot(&a, &b), a.dot(&flip_time(&b)));
    }
}
