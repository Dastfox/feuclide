//! Geodesic-ball culling, written once for every constant-curvature geometry.
//!
//! A ball of geodesic radius `r` whose centre is at distance `d` from the eye looks, from the
//! eye, like a cone of directions of angular radius `α` with
//!
//! - `sin α = r / d` in flat space,
//! - `sin α = sinh(r√−K) / sinh(d√−K)` at curvature `K < 0`,
//! - `sin α = sin(r√K) / sin(d√K)` at curvature `K > 0`.
//!
//! The four side planes of the frustum go through the eye, so the ball is outside one of them
//! exactly when its centre direction is more than `α` beyond it. Distance culling against the
//! far surface is plain geodesic distance.

use fk_geometry::Geometry;
use fk_math::Real;
use fk_math::nalgebra::Vector3;

/// The view frustum, as seen from the eye at the origin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frustum {
    /// Inward unit normals of the left, right, bottom and top planes, in reference-frame
    /// coordinates (the eye looks along `−z`).
    planes: [Vector3<Real>; 4],
    far: Real,
}

impl Frustum {
    /// The frustum of a camera with vertical field of view `fov_y`, an image `aspect` times as
    /// wide as high, and a far surface at geodesic distance `far`.
    pub fn new(fov_y: Real, aspect: Real, far: Real) -> Self {
        let half_y = fov_y / 2.0;
        let half_x = (half_y.tan() * aspect).atan();
        let (sx, cx) = half_x.sin_cos();
        let (sy, cy) = half_y.sin_cos();
        Self {
            planes: [
                Vector3::new(cx, 0.0, -sx),
                Vector3::new(-cx, 0.0, -sx),
                Vector3::new(0.0, cy, -sy),
                Vector3::new(0.0, -cy, -sy),
            ],
            far,
        }
    }

    /// Whether any of the geodesic ball of radius `radius` about `centre` (relative to the eye)
    /// can be seen. Errs towards visible.
    pub fn sees<G: Geometry>(&self, centre: &G::Point, radius: Real) -> bool {
        let origin = G::origin();
        let d = G::distance(&origin, centre);
        if d <= radius {
            return true;
        }
        if d - radius > self.far {
            return false;
        }
        let Some(sin_alpha) = angular_sin(G::constant_curvature(), d, radius) else {
            return true;
        };
        let Some(v) = G::log(&origin, centre) else {
            return true;
        };
        let direction = Vector3::from_fn(|i, _| G::inner(&origin, &v, &G::origin_frame(i))) / d;
        self.planes
            .iter()
            .all(|normal| normal.dot(&direction) >= -sin_alpha)
    }
}

/// `sin α` of the cone a ball of radius `r` at distance `d > r` fills, at constant curvature
/// `k`; `None` when unknown (variable curvature) or when the ball reaches past the point where
/// geodesics from the eye refocus.
fn angular_sin(k: Option<Real>, d: Real, r: Real) -> Option<Real> {
    let k = k?;
    let s = k.abs().sqrt();
    let ratio = if k == 0.0 {
        r / d
    } else if k < 0.0 {
        (r * s).sinh() / (d * s).sinh()
    } else if (d + r) * s < std::f64::consts::FRAC_PI_2 {
        (r * s).sin() / (d * s).sin()
    } else {
        return None;
    };
    Some(ratio.min(1.0))
}

#[cfg(test)]
mod tests {
    use std::f64::consts::FRAC_PI_2;

    use fk_geometry_euclidean::E3;
    use fk_math::nalgebra::Point3;
    use fk_math::nalgebra::Vector3;

    use super::*;

    fn sees(frustum: &Frustum, x: Real, y: Real, z: Real, r: Real) -> bool {
        frustum.sees::<E3>(&Point3::new(x, y, z), r)
    }

    #[test]
    fn balls_in_front_are_seen() {
        let frustum = Frustum::new(FRAC_PI_2, 1.0, 100.0);
        assert!(sees(&frustum, 0.0, 0.0, -10.0, 1.0));
        assert!(sees(&frustum, 0.0, 0.0, 0.5, 1.0), "the eye is inside");
        assert!(!sees(&frustum, 0.0, 0.0, 10.0, 1.0), "behind");
        assert!(!sees(&frustum, 0.0, 0.0, -102.0, 1.0), "past far");
        assert!(sees(&frustum, 0.0, 0.0, -100.5, 1.0), "straddles far");
    }

    #[test]
    fn side_planes_are_exact_in_flat_space() {
        // A 90° frustum: the right plane is x = −z. A ball of radius r at (x, 0, −10) touches it
        // when its centre is r√2 outside, x = 10 + r√2.
        let frustum = Frustum::new(FRAC_PI_2, 1.0, 100.0);
        let edge = 10.0 + 2.0_f64.sqrt();
        assert!(sees(&frustum, edge - 1e-6, 0.0, -10.0, 1.0));
        assert!(!sees(&frustum, edge + 1e-6, 0.0, -10.0, 1.0));
        assert!(!sees(&frustum, 0.0, edge + 1e-6, -10.0, 1.0), "top");
        assert!(!sees(&frustum, 0.0, -edge - 1e-6, -10.0, 1.0), "bottom");
        assert!(!sees(&frustum, -edge - 1e-6, 0.0, -10.0, 1.0), "left");
    }

    #[test]
    fn aspect_widens_horizontally() {
        let frustum = Frustum::new(FRAC_PI_2, 2.0, 100.0);
        assert!(sees(&frustum, 15.0, 0.0, -10.0, 0.1));
        assert!(!sees(&frustum, 0.0, 15.0, -10.0, 0.1));
    }

    #[test]
    fn curved_cones_match_flat_ones_when_small() {
        let flat = angular_sin(Some(0.0), 2.0, 0.5).unwrap();
        for k in [-1e-6, 1e-6] {
            assert!((angular_sin(Some(k), 2.0, 0.5).unwrap() - flat).abs() < 1e-6);
        }
        // Hyperbolic balls look smaller than flat ones at the same distance.
        assert!(angular_sin(Some(-1.0), 2.0, 0.5).unwrap() < flat);
        assert_eq!(angular_sin(None, 2.0, 0.5), None);
    }

    /// The point at distance `d` from the eye along the reference direction `(x, y, z)`.
    fn along<G: Geometry>(x: Real, y: Real, z: Real, d: Real) -> G::Point {
        let v = crate::tangent::<G>(&Vector3::new(x, y, z).normalize());
        G::exp(&G::origin(), &(v * d))
    }

    #[test]
    fn in_hyperbolic_space_the_fog_radius_is_a_hard_edge_and_the_sides_hold() {
        use fk_geometry_hyperbolic::H3;
        let frustum = Frustum::new(FRAC_PI_2, 1.0, 6.0);
        assert!(frustum.sees::<H3>(&along::<H3>(0.0, 0.0, -1.0, 5.0), 0.5));
        assert!(
            !frustum.sees::<H3>(&along::<H3>(0.0, 0.0, -1.0, 6.6), 0.5),
            "past the fog"
        );
        // 50° off the axis (beyond the 45° side), at distance 3: a ball of radius r reaches
        // back in by asin(sinh r / sinh 3), 2.9° for r = 0.5. Seen at 2°, not at 4° beyond.
        let off = |degrees: Real| {
            let a = degrees.to_radians();
            along::<H3>(a.sin(), 0.0, -a.cos(), 3.0)
        };
        let reach = (0.5_f64.sinh() / 3.0_f64.sinh()).asin().to_degrees();
        assert!((reach - 2.9).abs() < 0.1, "{reach}");
        assert!(frustum.sees::<H3>(&off(45.0 + 2.0), 0.5));
        assert!(!frustum.sees::<H3>(&off(45.0 + 4.0), 0.5));
    }

    #[test]
    fn on_the_sphere_what_is_behind_is_not_seen_directly_but_its_far_image_is() {
        use fk_geometry_spherical::S3;
        let frustum = Frustum::new(FRAC_PI_2, 1.0, 2.0 * std::f64::consts::PI);
        let behind = along::<S3>(0.0, 0.0, 1.0, 1.0);
        assert!(!frustum.sees::<S3>(&behind, 0.3), "behind, directly");
        // Seen the long way round it stands where its antipode does, π − 1 ahead.
        let image = along::<S3>(0.0, 0.0, -1.0, std::f64::consts::PI - 1.0);
        assert!(frustum.sees::<S3>(&image, 0.3));
        // Past the equator a ball's cone of directions is not closed form: errs towards seen.
        assert!(frustum.sees::<S3>(&along::<S3>(1.0, 0.0, 0.2, 2.0), 0.3));
    }
}
