//! Drawing a quotient space: the scene as seen when space closes up on itself.
//!
//! With a [`QuotientView`] set, the scene is taken to lie in the fundamental domain of its
//! quotient (in the root's frame), and the eye to stand in it (the game keeps it there with
//! `Quotient::reduce`). The raster pipeline draws every instance again at each deck translate
//! within the view horizon ([`QuotientView::translates`]), culled like any other; the ray
//! marcher carries each ray back across the faces of the domain as it crosses them (the
//! product of the `γ_f⁻¹` crossed, applied to the points it evaluates the field at), so a field
//! drawn once fills every copy of the domain. Entities marked [`Unrepeated`] are drawn once, as
//! they are.

use bevy_ecs::component::Component;
use bevy_ecs::resource::Resource;
use std::collections::HashMap;

use fk_geometry::{Geometry, GpuGeometry, GroupElement};
use fk_math::Real;
use fk_math::nalgebra::Vector4;
use fk_quotient::Quotient;

/// Most faces the ray marcher carries rays across: six for a flat 3-torus, twelve for a
/// dodecahedral domain (the Seifert–Weber space, the Poincaré sphere).
pub const MAX_FOLD_FACES: usize = 12;

/// The quotient the scene is drawn in, as a resource: see the module.
#[derive(Resource, Clone, Debug)]
pub struct QuotientView<G: Geometry> {
    quotient: Quotient<G>,
    translates: Vec<G::Isometry>,
}

impl<G: GpuGeometry> QuotientView<G> {
    /// The scene in `quotient`, its copies drawn out to `horizon` (the camera's far distance,
    /// or the fog's end): every deck element taking the domain's centre within `horizon` and
    /// the domain's reach of it. Elements are told apart by where they take the centre, filed
    /// by its embedded coordinates, so that the thousands of copies within a few units in
    /// hyperbolic space take a moment.
    ///
    /// # Panics
    ///
    /// If the quotient has more faces than [`MAX_FOLD_FACES`].
    pub fn new(quotient: Quotient<G>, horizon: Real) -> Self {
        assert!(
            quotient.faces().count() <= MAX_FOLD_FACES,
            "the ray marcher folds across at most {MAX_FOLD_FACES} faces"
        );
        let centre = *quotient.centre();
        // How far from the centre the domain reaches, at most: no farther than the farthest
        // neighbouring translate of the centre.
        let reach = quotient
            .faces()
            .map(|face| G::distance(&centre, &G::apply(&quotient.element(face), &centre)))
            .fold(0.0, Real::max);
        let within = horizon + reach;
        // Where each found element takes the centre, by its embedding rounded to a grid much
        // finer than the domain; a point is looked for in its own square and those round it.
        let key = |p: &G::Point| G::embed_point(p).map(|x| (x * 1e5).round() as i64);
        let mut found: Vec<(G::Isometry, G::Point)> = vec![(G::Isometry::identity(), centre)];
        let mut seen: HashMap<Vector4<i64>, Vec<usize>> = HashMap::new();
        seen.entry(key(&centre)).or_default().push(0);
        // Breadth first over the generators, through elements no farther than one step past
        // the bound, keeping those within it.
        let mut frontier = vec![0];
        while !frontier.is_empty() {
            let mut next = Vec::new();
            for i in frontier {
                for face in quotient.faces() {
                    let isometry = found[i].0.compose(&quotient.element(face));
                    let point = G::apply(&isometry, &centre);
                    if G::distance(&centre, &point) > within + reach {
                        continue;
                    }
                    let k = key(&point);
                    let known = (0..81).any(|n: i64| {
                        let step =
                            Vector4::new(n % 3 - 1, n / 3 % 3 - 1, n / 9 % 3 - 1, n / 27 - 1);
                        seen.get(&(k + step)).is_some_and(|near| {
                            near.iter()
                                .any(|&j| G::distance(&found[j].1, &point) < 1e-7)
                        })
                    });
                    if known {
                        continue;
                    }
                    found.push((isometry, point));
                    seen.entry(k).or_default().push(found.len() - 1);
                    next.push(found.len() - 1);
                }
            }
            frontier = next;
        }
        let translates = found
            .into_iter()
            .skip(1)
            .filter(|(_, point)| G::distance(&centre, point) <= within)
            .map(|(isometry, _)| isometry)
            .collect();
        Self {
            quotient,
            translates,
        }
    }
}

impl<G: Geometry> QuotientView<G> {
    /// The quotient.
    pub fn quotient(&self) -> &Quotient<G> {
        &self.quotient
    }

    /// The deck elements the scene is drawn again at, the identity left out.
    pub fn translates(&self) -> &[G::Isometry] {
        &self.translates
    }
}

/// Drawn once, as it is, in a [`QuotientView`]: not repeated at the translates nor folded
/// across faces (things carried with the eye, say).
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct Unrepeated;

#[cfg(test)]
mod tests {
    use fk_geometry_euclidean::E3;
    use fk_math::nalgebra::{Point3, Vector3};

    use super::*;

    #[test]
    fn the_translates_are_the_copies_within_the_horizon() {
        // A square torus 10 m round along x and z: copies on the lattice; within 25 m and the
        // domain's reach (10 m) of the centre, every lattice point but the centre.
        let torus = Quotient::<E3>::from_translations(
            Point3::origin(),
            &[Vector3::x() * 10.0, Vector3::z() * 10.0],
        );
        let view = QuotientView::new(torus, 25.0);
        let expected = (-3i32..=3)
            .flat_map(|i| (-3i32..=3).map(move |j| (i, j)))
            .filter(|&(i, j)| (i, j) != (0, 0) && Real::from(i * i + j * j).sqrt() * 10.0 <= 35.0)
            .count();
        assert_eq!(view.translates().len(), expected);
    }
}
