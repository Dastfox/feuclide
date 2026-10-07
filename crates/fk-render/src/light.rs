//! Point lights, with the falloff of their geometry.
//!
//! A point light spreads its light over the geodesic spheres about it, so what reaches a
//! surface at distance `d` goes as one over the area of the sphere of radius `d`: `1/d²` in E³,
//! `1/sinh²(d)` in H³ (spheres grow exponentially, light dies fast), `1/sin²(d)` in S³ (spheres
//! shrink again past the equator, and light gathers at the antipode). With curvature `k`, the
//! radius is scaled by `√|k|` and the whole by `|k|`. Each geometry's `fk::geometry` module has
//! the twin, `geo_light_falloff`; [`falloff`] is the CPU one. Past its range a light is cut
//! off, smoothly, so a handful of lights can be chosen per frame.

use bevy_ecs::component::Component;
use fk_geometry::Geometry;
use fk_math::Real;

use crate::Color;

/// Most point lights drawn in one frame: the nearest to the eye.
pub const MAX_POINT_LIGHTS: usize = 8;

/// A light at the origin of its entity's frame (its `GlobalPose`), shining every way.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct PointLight {
    /// Its colour.
    pub color: Color,
    /// How bright it is: the light on a surface facing it one unit away, in flat space.
    pub intensity: Real,
    /// Geodesic distance past which it lights nothing, faded out towards it.
    pub range: Real,
}

impl Default for PointLight {
    fn default() -> Self {
        Self {
            color: Color::WHITE,
            intensity: 1.0,
            range: 10.0,
        }
    }
}

impl PointLight {
    /// The light on a surface facing it at geodesic distance `d`, in geometry `G`, before its
    /// colour: the intensity, the geometry's falloff and the fade towards the range.
    pub fn at<G: Geometry>(&self, d: Real) -> Real {
        self.intensity * falloff::<G>(d) * window(d, self.range)
    }
}

/// Distance under which the falloff stops growing, so a surface through a light is not lit
/// without bound.
pub const NEAREST: Real = 0.1;

/// One over the area of the geodesic sphere of radius `d`, over that of the unit sphere in flat
/// space: `1/d²`, `|k| / sinh²(√|k| d)` or `k / sin²(√k d)`, for curvature 0, `k < 0` or
/// `k > 0`. Geometries without a constant curvature fall back on flat space's.
pub fn falloff<G: Geometry>(d: Real) -> Real {
    let d = d.max(NEAREST);
    let k = G::constant_curvature().unwrap_or(0.0);
    let s = k.abs().sqrt();
    let area = if k < 0.0 {
        (s * d).sinh() / s
    } else if k > 0.0 {
        // Near the antipode the sphere shrinks to a point; held at the nearest's.
        ((s * d).sin() / s).max(NEAREST)
    } else {
        d
    };
    1.0 / (area * area)
}

/// From 1 at the light to 0 at `range`, smoothly: `(1 − (d/range)⁴)²`.
pub fn window(d: Real, range: Real) -> Real {
    if range <= 0.0 {
        return 0.0;
    }
    let x = (d / range).clamp(0.0, 1.0);
    let f = 1.0 - x.powi(4);
    f * f
}

#[cfg(test)]
mod tests {
    use fk_geometry_euclidean::E3;

    use super::*;

    #[test]
    fn in_flat_space_it_falls_off_as_the_square() {
        assert!((falloff::<E3>(2.0) - 0.25).abs() < 1e-12);
        assert!((falloff::<E3>(10.0) - 0.01).abs() < 1e-12);
        assert_eq!(falloff::<E3>(0.0), falloff::<E3>(NEAREST));
    }

    #[test]
    fn it_fades_to_nothing_at_its_range() {
        let light = PointLight {
            range: 5.0,
            ..PointLight::default()
        };
        assert_eq!(light.at::<E3>(5.0), 0.0);
        assert_eq!(light.at::<E3>(7.0), 0.0);
        assert!(light.at::<E3>(4.9) > 0.0);
        assert!((window(0.0, 5.0) - 1.0).abs() < 1e-12);
        // Near it, nearly the plain falloff.
        assert!((light.at::<E3>(1.0) - 1.0).abs() < 0.01);
    }
}
