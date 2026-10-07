//! SL(2, ℂ), the double cover of SO⁺(1, 3), where the exponential and logarithm of a Lorentz
//! transformation have closed forms.
//!
//! A point `(t, x, y, z)` of Minkowski space is the Hermitian matrix `t σ₀ + x σ₁ + y σ₂ + z σ₃`
//! (`σ₀` the identity, `σ₁ σ₂ σ₃` the Pauli matrices), whose determinant is `t² − x² − y² − z²`;
//! `A ∈ SL(2, ℂ)` acts as `X ↦ A X A†`, and `A` and `−A` give the same transformation. A twist
//! of boost `b` and rotation `ω` is the traceless `Ξ = ½ Σ (b_k − i ω_k) σ_k`, whose exponential
//! is `cosh λ + (sinh λ / λ) Ξ` with `λ² = ¼ Σ (b_k − i ω_k)²`.

use fk_math::Real;
use fk_math::nalgebra::{Complex, Matrix4, Vector3};

type C = Complex<Real>;

/// A 2×2 complex matrix, row by row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct M2(pub [[C; 2]; 2]);

const ZERO: C = C::new(0.0, 0.0);
const ONE: C = C::new(1.0, 0.0);
const I: C = C::new(0.0, 1.0);

impl M2 {
    fn identity() -> Self {
        Self([[ONE, ZERO], [ZERO, ONE]])
    }

    /// `σ_k`: the identity for 0, then the Pauli matrices.
    fn pauli(k: usize) -> Self {
        match k {
            0 => Self::identity(),
            1 => Self([[ZERO, ONE], [ONE, ZERO]]),
            2 => Self([[ZERO, -I], [I, ZERO]]),
            3 => Self([[ONE, ZERO], [ZERO, -ONE]]),
            _ => unreachable!("σ₀ to σ₃"),
        }
    }

    fn mul(&self, rhs: &Self) -> Self {
        let (a, b) = (self.0, rhs.0);
        Self(std::array::from_fn(|i| {
            std::array::from_fn(|j| a[i][0] * b[0][j] + a[i][1] * b[1][j])
        }))
    }

    fn add(&self, rhs: &Self) -> Self {
        Self(std::array::from_fn(|i| {
            std::array::from_fn(|j| self.0[i][j] + rhs.0[i][j])
        }))
    }

    fn scale(&self, s: C) -> Self {
        Self(self.0.map(|row| row.map(|x| x * s)))
    }

    /// The conjugate transpose.
    fn dagger(&self) -> Self {
        let a = self.0;
        Self([
            [a[0][0].conj(), a[1][0].conj()],
            [a[0][1].conj(), a[1][1].conj()],
        ])
    }

    fn trace(&self) -> C {
        self.0[0][0] + self.0[1][1]
    }

    fn det(&self) -> C {
        self.0[0][0] * self.0[1][1] - self.0[0][1] * self.0[1][0]
    }

    fn norm_squared(&self) -> Real {
        self.0.iter().flatten().map(|x| x.norm_sqr()).sum()
    }

    /// The inverse of a matrix of determinant one.
    fn inverse_unimodular(&self) -> Self {
        let a = self.0;
        Self([[a[1][1], -a[0][1]], [-a[1][0], a[0][0]]])
    }
}

/// `ξ ↦ Ξ = ½ Σ (b_k − i ω_k) σ_k`.
fn generator(boost: &Vector3<Real>, rotation: &Vector3<Real>) -> M2 {
    (1..4).fold(M2([[ZERO; 2]; 2]), |sum, k| {
        sum.add(&M2::pauli(k).scale(C::new(boost[k - 1], -rotation[k - 1]) * 0.5))
    })
}

/// `Ξ ↦ (b, ω)`, from `tr(Ξ σ_k) = b_k − i ω_k`.
fn twist(xi: &M2) -> (Vector3<Real>, Vector3<Real>) {
    let c: [C; 3] = std::array::from_fn(|k| xi.mul(&M2::pauli(k + 1)).trace());
    (
        Vector3::new(c[0].re, c[1].re, c[2].re),
        Vector3::new(-c[0].im, -c[1].im, -c[2].im),
    )
}

/// `sinh(λ) / λ` for complex `λ`.
fn sinhc(l: C) -> C {
    if l.norm() < 1e-4 {
        let l2 = l * l;
        ONE + l2 / 6.0 + l2 * l2 / 120.0
    } else {
        l.sinh() / l
    }
}

/// `asinh(s) / s` for complex `s`.
fn asinhc(s: C) -> C {
    if s.norm() < 1e-4 {
        let s2 = s * s;
        ONE - s2 / 6.0 + s2 * s2 * (3.0 / 40.0)
    } else {
        s.asinh() / s
    }
}

/// The exponential of the twist (boost `b`, rotation `ω`).
pub(crate) fn exp(boost: &Vector3<Real>, rotation: &Vector3<Real>) -> M2 {
    let xi = generator(boost, rotation);
    // Ξ² = λ² · 1, and λ² = −det Ξ.
    let lambda = (-xi.det()).sqrt();
    M2::identity()
        .scale(lambda.cosh())
        .add(&xi.scale(sinhc(lambda)))
}

/// The principal logarithm of `±a`, the sign taken with the trace's real part non-negative:
/// a twist (boost, rotation) with `exp` giving `±a`.
pub(crate) fn log(a: &M2) -> (Vector3<Real>, Vector3<Real>) {
    let a = if a.trace().re < 0.0 {
        a.scale(-ONE)
    } else {
        *a
    };
    let c = a.trace() * 0.5;
    // The traceless part is (sinh λ / λ) Ξ, and its square sinh² λ: λ from the sine, which
    // keeps every digit near the identity (the cosine, the trace, would lose half).
    let traceless = a.add(&M2::identity().scale(-c));
    let s = (-traceless.det()).sqrt();
    let s = if (ONE + s * s).sqrt().re * c.re < 0.0 {
        -s
    } else {
        s
    };
    twist(&traceless.scale(asinhc(s)))
}

/// The Lorentz matrix of `a` in coordinates `(t, x, y, z)`: `Λ_μν = ½ tr(σ_μ A σ_ν A†)`.
pub(crate) fn lorentz(a: &M2) -> Matrix4<Real> {
    let dagger = a.dagger();
    Matrix4::from_fn(|mu, nu| {
        let y = a.mul(&M2::pauli(nu)).mul(&dagger);
        0.5 * M2::pauli(mu).mul(&y).trace().re
    })
}

/// One of the two matrices `±A` whose Lorentz matrix is `lambda`: with `Y_ν = A σ_ν A†` read
/// from its columns, `Σ_ν Y_ν X σ_ν = 2 tr(A† X) A` for any `X`; `X` is taken among the `σ_κ`
/// to make that largest, and the result scaled to determinant one.
pub(crate) fn from_lorentz(lambda: &Matrix4<Real>) -> M2 {
    let y: [M2; 4] = std::array::from_fn(|nu| {
        (0..4).fold(M2([[ZERO; 2]; 2]), |sum, mu| {
            sum.add(&M2::pauli(mu).scale(C::new(lambda[(mu, nu)], 0.0)))
        })
    });
    let best = (0..4)
        .map(|kappa| {
            let x = M2::pauli(kappa);
            (0..4).fold(M2([[ZERO; 2]; 2]), |sum, nu| {
                sum.add(&y[nu].mul(&x).mul(&M2::pauli(nu)))
            })
        })
        .max_by(|a, b| a.norm_squared().total_cmp(&b.norm_squared()))
        .expect("four candidates");
    best.scale(ONE / best.det().sqrt())
}

/// `Ad_A ξ`: the twist of `A Ξ A⁻¹`.
pub(crate) fn adjoint(
    a: &M2,
    boost: &Vector3<Real>,
    rotation: &Vector3<Real>,
) -> (Vector3<Real>, Vector3<Real>) {
    let xi = generator(boost, rotation);
    twist(&a.mul(&xi).mul(&a.inverse_unimodular()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_lorentz_matrix_comes_back_through_sl2c() {
        let a = exp(&Vector3::new(0.3, -1.2, 0.7), &Vector3::new(2.9, 0.1, -0.4));
        let lambda = lorentz(&a);
        let back = from_lorentz(&lambda);
        assert!((lorentz(&back) - lambda).norm() < 1e-12);
        // Its inverse undoes it, and it keeps the Minkowski form.
        let eta = Matrix4::from_diagonal(&fk_math::nalgebra::Vector4::new(-1.0, 1.0, 1.0, 1.0));
        assert!((lambda.transpose() * eta * lambda - eta).norm() < 1e-12);
    }

    #[test]
    fn log_inverts_exp_near_and_far() {
        for scale in [0.0, 1e-12, 1e-6, 1e-3, 0.5, 1.4] {
            let (b, w) = (
                Vector3::new(0.3, -0.5, 0.8) * scale,
                Vector3::new(-0.6, 0.2, 0.4) * scale,
            );
            let (b2, w2) = log(&exp(&b, &w));
            assert!(
                (b2 - b).norm() + (w2 - w).norm() < 1e-13,
                "{scale}: {b2} {w2}"
            );
        }
        // A half turn: its logarithm turns half way the other way or this way, the same.
        let a = exp(&Vector3::zeros(), &(Vector3::z() * std::f64::consts::PI));
        let (b, w) = log(&a);
        assert!((lorentz(&exp(&b, &w)) - lorentz(&a)).norm() < 1e-12);
    }
}
