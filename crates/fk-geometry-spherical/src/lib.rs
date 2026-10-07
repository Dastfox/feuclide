//! Spheres (S², S³) embedded in ℝ³ and ℝ⁴, with SO(4) represented as a pair of unit quaternions.
//!
//! [`S3`] is the unit 3-sphere, its isometries [`So4`] the pairs `(l, r)` acting as
//! `x ↦ l x r̄`; [`S2`] the unit 2-sphere, its isometries [`So3`] unit quaternions. Both have
//! curvature +1 and injectivity radius π: the antipode of a point is its cut locus, where
//! `log` returns `None` (see [`ANTIPODE`]) and parallel transport only projects.

mod s2;
mod s3;

pub use s2::{S2, So3};
pub use s3::{ANTIPODE, S3, So4, So4Twist, from_chart};

use fk_math::Real;
use fk_math::nalgebra::{Quaternion, UnitQuaternion, Vector3};
use fk_math::series::sinc;

/// The quaternion exponential of the pure quaternion `u`: `cos |u| + sin |u| û`.
fn quaternion_exp(u: &Vector3<Real>) -> UnitQuaternion<Real> {
    let theta = u.norm();
    let xyz = u * sinc(theta);
    UnitQuaternion::new_unchecked(Quaternion::new(theta.cos(), xyz.x, xyz.y, xyz.z))
}

/// The principal logarithm of a unit quaternion, a pure quaternion with `|u|` in `[0, π]`.
fn quaternion_log(q: &UnitQuaternion<Real>) -> Vector3<Real> {
    let (w, xyz) = (q.scalar(), q.imag());
    let s = xyz.norm();
    let theta = s.atan2(w);
    if s == 0.0 {
        // The identity, or −1 (any axis: x).
        return if w >= 0.0 {
            Vector3::zeros()
        } else {
            Vector3::x() * std::f64::consts::PI
        };
    }
    xyz * (theta / s)
}
