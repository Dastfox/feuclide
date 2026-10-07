//! Hyperbolic plane and space (H², H³) in the hyperboloid model, with isometry group SO⁺(1,n).
//!
//! Points satisfy `⟨p, p⟩ = −1`, `t > 0`, in the Minkowski form of [`fk_math::minkowski`]
//! (time first). Isometries are Lorentz matrices ([`Lorentz3`], [`Lorentz2`]), renormalized by
//! Lorentz Gram–Schmidt; their exponential and logarithm go through SL(2, ℂ), the double cover
//! of SO⁺(1, 3), where both have closed forms. Curvature −1, geodesics unique: `log` is always
//! defined. Constraint residuals are asserted in debug builds where the metric is read.

mod h2;
mod h3;
mod sl2c;

pub use h2::{H2, Lorentz2, Lorentz2Twist};
pub use h3::{H3, Lorentz3, Lorentz3Twist, from_ball};
