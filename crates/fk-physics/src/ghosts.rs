//! Γ-ghosts: in a quotient space, the copies of each ball across the faces of the fundamental
//! domain, fed into the broadphase with the originals, so that what stands just across a face
//! is found from just this side of it.
//!
//! A copy `γ(c)` of a ball `(c, r)` by a deck element `γ` is kept when it may reach within
//! `margin` of the domain. The test is the Dirichlet domain's own, by the metric alone: for
//! every face, `beyond(p, face) = d(p, c₀) − d(p, γ_face(c₀))` is at most twice the distance
//! from `p` to that face's bisector (the triangle inequality through the nearest point of the
//! bisector), so a copy with `beyond > 2 (r + margin)` for some face is surely too far, and the
//! rest are kept. A ghost found by a query names its ball and the deck element that carries
//! the ball to it.

use fk_geometry::Geometry;
use fk_math::Real;
use fk_quotient::Quotient;

use crate::bvh::BallTree;

/// A copy of one ball by a deck element.
#[derive(Clone, Copy, Debug)]
pub struct Ghost<G: Geometry> {
    /// The ball it copies: its index in the balls it was made from.
    pub of: u32,
    /// The deck element that carries the ball to the copy (the identity for the ball itself).
    pub deck: G::Isometry,
    /// The copy's centre, `deck(c)`.
    pub centre: G::Point,
    /// Its radius.
    pub radius: Real,
}

/// Every ball and every copy of it, by the deck elements with words of at most `length`
/// letters, that may come within `margin` of the fundamental domain: the balls themselves
/// first, in order.
pub fn ghosts<G: Geometry>(
    quotient: &Quotient<G>,
    balls: &[(G::Point, Real)],
    margin: Real,
    length: usize,
) -> Vec<Ghost<G>> {
    let faces: Vec<_> = quotient.faces().collect();
    let translates = quotient.translates(length);
    let mut found = Vec::new();
    for translate in &translates {
        for (i, (centre, radius)) in balls.iter().enumerate() {
            let at = G::apply(&translate.isometry, centre);
            let near = faces
                .iter()
                .all(|&face| quotient.beyond(&at, face) <= 2.0 * (radius + margin));
            if near {
                found.push(Ghost {
                    of: u32::try_from(i).expect("too many balls"),
                    deck: translate.isometry,
                    centre: at,
                    radius: *radius,
                });
            }
        }
    }
    found
}

/// The broadphase of a quotient space: a [`BallTree`] over the balls and their [`ghosts`].
#[derive(Debug)]
pub struct GhostTree<G: Geometry> {
    ghosts: Vec<Ghost<G>>,
    tree: BallTree<G>,
}

impl<G: Geometry> GhostTree<G> {
    /// The tree over `balls` and their copies that may come within `margin` of the domain, by
    /// deck elements of at most `length` letters. Queries within `margin` of the domain find
    /// everything; the balls are taken to lie in the domain.
    pub fn build(
        quotient: &Quotient<G>,
        balls: &[(G::Point, Real)],
        margin: Real,
        length: usize,
    ) -> Self {
        let ghosts = ghosts(quotient, balls, margin, length);
        let tree = BallTree::build(
            &ghosts
                .iter()
                .map(|ghost| (ghost.centre, ghost.radius))
                .collect::<Vec<_>>(),
        );
        Self { ghosts, tree }
    }

    /// How many balls and copies it holds.
    pub fn len(&self) -> usize {
        self.ghosts.len()
    }

    /// Whether it holds none.
    pub fn is_empty(&self) -> bool {
        self.ghosts.is_empty()
    }

    /// Calls `found` with every ball or copy that comes within `radius` of `at`.
    pub fn near(&self, at: &G::Point, radius: Real, mut found: impl FnMut(&Ghost<G>)) {
        self.tree
            .near(at, radius, |i| found(&self.ghosts[i as usize]));
    }
}

#[cfg(test)]
mod tests {
    use fk_geometry_euclidean::E3;
    use fk_math::nalgebra::{Point3, Vector3};

    use super::*;

    /// A flat torus ten metres round every way, about the origin.
    fn torus() -> Quotient<E3> {
        Quotient::from_translations(
            Point3::origin(),
            &[
                Vector3::new(10.0, 0.0, 0.0),
                Vector3::new(0.0, 10.0, 0.0),
                Vector3::new(0.0, 0.0, 10.0),
            ],
        )
    }

    #[test]
    fn a_ball_by_one_face_is_found_across_it() {
        let balls = [(Point3::new(4.6, 0.0, 0.0), 0.3)];
        let tree = GhostTree::build(&torus(), &balls, 1.0, 2);
        // Its copy stands at x = −5.4, 0.6 from a query at −4.8 (reaching 0.5 + 0.3).
        let mut found = Vec::new();
        tree.near(&Point3::new(-4.8, 0.0, 0.0), 0.5, |g| {
            found.push((g.of, g.centre))
        });
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].0, 0);
        assert!((found[0].1 - Point3::new(-5.4, 0.0, 0.0)).norm() < 1e-9);
        // A ball in the middle has no copies.
        let lone = GhostTree::build(&torus(), &[(Point3::origin(), 0.5)], 1.0, 2);
        assert_eq!(lone.len(), 1);
    }

    #[test]
    fn a_ball_in_a_corner_is_copied_into_every_neighbouring_domain() {
        // Near the corner (+, +, +): copies across three faces, three edges and the corner.
        let balls = [(Point3::new(4.8, 4.8, 4.8), 0.1)];
        let tree = GhostTree::build(&torus(), &balls, 0.5, 3);
        assert_eq!(tree.len(), 8);
    }

    #[test]
    fn the_ghost_tree_finds_what_the_shortest_distance_finds() {
        // Balls all over the domain, queries all over it: whatever the quotient's shortest
        // distance puts within reach, the tree finds, and nothing else.
        let torus = torus();
        let balls: Vec<_> = (0..200)
            .map(|i| {
                let t = Real::from(i);
                let c = Point3::new(
                    4.9 * (t * 1.3).sin(),
                    4.9 * (t * 0.7).cos(),
                    4.9 * (t * 2.1).sin(),
                );
                (c, 0.2 + 0.1 * (t * 0.37).cos().abs())
            })
            .collect();
        let margin = 1.0;
        let tree = GhostTree::build(&torus, &balls, margin, 3);
        for k in 0..50 {
            let t = Real::from(k);
            let at = Point3::new(
                5.0 * (t * 0.9).cos(),
                5.0 * (t * 1.7).sin(),
                5.0 * (t * 0.3).cos(),
            );
            let radius = 0.5;
            let mut found: Vec<u32> = Vec::new();
            tree.near(&at, radius, |g| found.push(g.of));
            found.sort_unstable();
            found.dedup();
            let scanned: Vec<u32> = balls
                .iter()
                .enumerate()
                .filter(|(_, (c, r))| torus.shortest(&at, c, 3).0 <= r + radius)
                .map(|(i, _)| i as u32)
                .collect();
            assert_eq!(found, scanned, "query {k}");
        }
    }
}
