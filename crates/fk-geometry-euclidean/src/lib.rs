//! Euclidean geometry, as one geometry among others.
//!
//! [`E2`] and [`E3`] get no special treatment anywhere in the engine: they implement the same
//! [`Geometry`](fk_geometry::Geometry) trait as hyperbolic and spherical space, with isometry
//! groups SE(2) and SE(3).

/// Implements the vector-space operators of a twist type with `angular` and `linear` parts.
macro_rules! twist_ops {
    ($twist:ident) => {
        impl ::std::ops::Add for $twist {
            type Output = Self;
            fn add(self, rhs: Self) -> Self {
                Self {
                    angular: self.angular + rhs.angular,
                    linear: self.linear + rhs.linear,
                }
            }
        }

        impl ::std::ops::Sub for $twist {
            type Output = Self;
            fn sub(self, rhs: Self) -> Self {
                Self {
                    angular: self.angular - rhs.angular,
                    linear: self.linear - rhs.linear,
                }
            }
        }

        impl ::std::ops::Neg for $twist {
            type Output = Self;
            fn neg(self) -> Self {
                Self {
                    angular: -self.angular,
                    linear: -self.linear,
                }
            }
        }

        impl ::std::ops::Mul<::fk_math::Real> for $twist {
            type Output = Self;
            fn mul(self, s: ::fk_math::Real) -> Self {
                Self {
                    angular: self.angular * s,
                    linear: self.linear * s,
                }
            }
        }
    };
}

mod e2;
mod e3;

pub use e2::{E2, Se2, Se2Twist};
pub use e3::{E3, Se3, Se3Twist};
