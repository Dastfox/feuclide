//! Surfaces cut away round directions of the sky, so that the sky shows through them.

use bevy_ecs::component::Component;
use bevy_ecs::resource::Resource;
use fk_math::Real;
use fk_math::nalgebra::Vector3;

/// Most cones a [`Parting`] cuts.
pub const MAX_PARTINGS: usize = 4;

/// Marks an entity whose surfaces the [`Parting`] cuts away: within its cones they are not
/// drawn, and what is behind them shows, the sky as much as anything not marked.
///
/// Drawn by pipelines of their own whose fragment stage discards, so that no other surface
/// pays for it; while the parting has no cones they are drawn by the ordinary ones. The
/// shadows are left as they are: a surface cut away still casts. In the image of another eye
/// ([`Elsewhere`](crate::Elsewhere), the [`OtherEyes`](crate::OtherEyes)) it is cut away as
/// seen from that eye, and cut off at that eye's clip as well, if it has one. A raster entity
/// only: the ray-marched fields know nothing of it.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SeeThrough;

/// One cone of the [`Parting`]: the directions from the eye within [`angle`](Self::angle) of
/// [`direction`](Self::direction).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Part {
    /// Its axis, in the reference frame at the scene root's origin, carried to the eye every
    /// frame as the [`Lighting`](crate::Lighting)'s directions are: the direction of a body of
    /// the [`Sky`](crate::Sky) is the same here. Of any length but zero (zero cuts nothing).
    pub direction: Vector3<Real>,
    /// Its half-angle, in radians: a surface seen within it of the axis is cut away.
    pub angle: Real,
    /// How far its edge strays either way, in radians, so that it is ragged as foliage is
    /// rather than a clean circle: the half-angle at a point of a surface is moved by up to
    /// this, by a hash of the [`cell`](Self::cell) of the mesh's own space the point is in. The
    /// same point strays the same way in every frame, however the eye moves.
    pub ragged: Real,
    /// The size of those cells, in the units the mesh is built in (metres, in flat space), at
    /// the vertices as built, before any [`Deform`](crate::Deform).
    pub cell: Real,
    /// Nearer the eye than this nothing is cut, in geodesic distance.
    pub near: Real,
}

impl Default for Part {
    fn default() -> Self {
        Self {
            direction: Vector3::y(),
            angle: 0.0,
            ragged: 0.0,
            cell: 1.0,
            near: 0.0,
        }
    }
}

/// The cones the surfaces marked [`SeeThrough`] are cut away in, as a resource: the first
/// [`MAX_PARTINGS`]. None by default, and then nothing is cut, the image exactly as without.
///
/// A game opening its trees round the moon so that the moon shows through them sets one cone
/// about the moon's direction, marks the trees, and leaves the ground and the walls unmarked
/// so that they still hide it.
#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct Parting(pub Vec<Part>);

impl Parting {
    /// Whether it cuts anything: a cone with some angle, or a ragged edge that can open one.
    pub fn cuts(&self) -> bool {
        self.used()
            .iter()
            .any(|part| part.angle + part.ragged.abs() > 0.0)
    }

    /// The cones drawn with: the first [`MAX_PARTINGS`].
    pub(crate) fn used(&self) -> &[Part] {
        &self.0[..self.0.len().min(MAX_PARTINGS)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_by_default_and_none_cuts_nothing() {
        assert!(Parting::default().0.is_empty());
        assert!(!Parting::default().cuts());
        assert!(!Parting(vec![Part::default()]).cuts(), "a cone of no angle");
        let open = Part {
            angle: 0.2,
            ..Part::default()
        };
        assert!(Parting(vec![open]).cuts());
        assert_eq!(Parting(vec![open; 6]).used().len(), MAX_PARTINGS);
    }
}
