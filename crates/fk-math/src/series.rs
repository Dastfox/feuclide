//! Scalar functions that appear in closed-form exponential and logarithm maps, evaluated without
//! cancellation or `0 / 0` near the origin.

use crate::Real;

/// `sin(x) / x`.
pub fn sinc(x: Real) -> Real {
    if x.abs() < 1e-4 {
        let x2 = x * x;
        1.0 - x2 / 6.0 + x2 * x2 / 120.0
    } else {
        x.sin() / x
    }
}

/// `sinh(x) / x`.
pub fn sinhc(x: Real) -> Real {
    if x.abs() < 1e-4 {
        let x2 = x * x;
        1.0 + x2 / 6.0 + x2 * x2 / 120.0
    } else {
        x.sinh() / x
    }
}

/// `(1 - cos(x)) / x²`, computed as `2 sin²(x/2) / x²`.
pub fn cosc(x: Real) -> Real {
    let s = sinc(0.5 * x);
    0.5 * s * s
}

/// `(cosh(x) - 1) / x²`, computed as `2 sinh²(x/2) / x²`.
pub fn coshc(x: Real) -> Real {
    let s = sinhc(0.5 * x);
    0.5 * s * s
}

/// `(x - sin(x)) / x³`.
pub fn sinc3(x: Real) -> Real {
    if x.abs() < 0.5 {
        let x2 = x * x;
        taylor_odd_remainder(x2, -1.0)
    } else {
        (x - x.sin()) / (x * x * x)
    }
}

/// `(sinh(x) - x) / x³`.
pub fn sinhc3(x: Real) -> Real {
    if x.abs() < 0.5 {
        let x2 = x * x;
        taylor_odd_remainder(x2, 1.0)
    } else {
        (x.sinh() - x) / (x * x * x)
    }
}

/// `Σ_{k≥0} sign^k x^{2k} / (2k + 3)!`, truncated after the `x¹⁰` term.
///
/// Accurate to `f64` precision for `x² < 0.25`.
fn taylor_odd_remainder(x2: Real, sign: Real) -> Real {
    const INV_FACT: [Real; 6] = [
        1.0 / 6.0,
        1.0 / 120.0,
        1.0 / 5_040.0,
        1.0 / 362_880.0,
        1.0 / 39_916_800.0,
        1.0 / 6_227_020_800.0,
    ];
    let t = sign * x2;
    INV_FACT.iter().rev().fold(0.0, |acc, c| acc * t + c)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: Real, b: Real) -> bool {
        (a - b).abs() <= 1e-13 * (1.0 + a.abs().max(b.abs()))
    }

    #[test]
    fn limits_at_zero() {
        assert_eq!(sinc(0.0), 1.0);
        assert_eq!(sinhc(0.0), 1.0);
        assert_eq!(cosc(0.0), 0.5);
        assert_eq!(coshc(0.0), 0.5);
        assert!(close(sinc3(0.0), 1.0 / 6.0));
        assert!(close(sinhc3(0.0), 1.0 / 6.0));
    }

    #[test]
    fn match_direct_formulas_away_from_zero() {
        for &x in &[0.3, 0.49, 0.51, 1.0, 2.5, -1.7] {
            let x2 = x * x;
            assert!(close(sinc(x), x.sin() / x), "sinc({x})");
            assert!(close(sinhc(x), x.sinh() / x), "sinhc({x})");
            assert!(close(cosc(x), (1.0 - x.cos()) / x2), "cosc({x})");
            assert!(close(coshc(x), (x.cosh() - 1.0) / x2), "coshc({x})");
            assert!(close(sinc3(x), (x - x.sin()) / (x2 * x)), "sinc3({x})");
            assert!(close(sinhc3(x), (x.sinh() - x) / (x2 * x)), "sinhc3({x})");
        }
    }

    #[test]
    fn continuous_across_branch_points() {
        type Series = fn(Real) -> Real;
        let branch_points: [(Series, Real); 4] =
            [(sinc, 1e-4), (sinhc, 1e-4), (sinc3, 0.5), (sinhc3, 0.5)];
        for (f, at) in branch_points {
            let below = f(at.next_down());
            let above = f(at);
            assert!(close(below, above), "jump at {at}: {below} vs {above}");
        }
    }
}
