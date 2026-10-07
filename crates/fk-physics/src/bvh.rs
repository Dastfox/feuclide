//! A bounding-volume hierarchy of geodesic balls: the broadphase in any geometry.
//!
//! Every leaf is a ball (a centre and a radius) around one thing; every node the smallest ball
//! along the geodesic between its children's centres that holds them both:
//! for balls `(c₁, r₁)` and `(c₂, r₂)` at distance `d`, unless one holds the other, the ball of
//! radius `(d + r₁ + r₂) / 2` centred on the geodesic from `c₁` to `c₂`, `r − r₁` from `c₁`.
//! The triangle inequality makes it hold both in any geometry, as long as the balls stay within
//! the injectivity radius. It is built top-down, splitting the centres at the median of their
//! widest spread in the normal coordinates at their first centre.
//!
//! A query keeps every node its ball touches and goes down: what may come within `radius` of a
//! point. Exact in E³; in curved space, as exact as the distances are.

use fk_geometry::Geometry;
use fk_math::Real;
use fk_math::nalgebra::Vector3;

use crate::coords;

/// One ball of the tree.
#[derive(Debug)]
struct Node<G: Geometry> {
    centre: G::Point,
    radius: Real,
    kind: Kind,
}

#[derive(Clone, Copy, Debug)]
enum Kind {
    /// The thing it holds.
    Leaf(u32),
    /// Its two children.
    Branch(usize, usize),
}

/// A tree of geodesic balls: see the module.
#[derive(Debug)]
pub struct BallTree<G: Geometry> {
    nodes: Vec<Node<G>>,
    root: Option<usize>,
}

impl<G: Geometry> Default for BallTree<G> {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            root: None,
        }
    }
}

/// The smallest ball along the geodesic between their centres that holds both balls.
pub fn merge<G: Geometry>(a: (&G::Point, Real), b: (&G::Point, Real)) -> (G::Point, Real) {
    let ((ca, ra), (cb, rb)) = (a, b);
    let Some(towards) = G::log(ca, cb) else {
        // Past the cut locus: nothing smaller is sure to hold both.
        return (*ca, G::injectivity_radius().max(ra + rb));
    };
    let d = G::norm(ca, &towards);
    if d + rb <= ra {
        return (*ca, ra);
    }
    if d + ra <= rb {
        return (*cb, rb);
    }
    let r = (d + ra + rb) / 2.0;
    let centre = G::exp(ca, &(towards * ((r - ra) / d.max(1e-12))));
    (centre, r)
}

impl<G: Geometry> BallTree<G> {
    /// The tree over `balls`, each `(centre, radius)`; a query gives back their indices.
    pub fn build(balls: &[(G::Point, Real)]) -> Self {
        let mut tree = Self::default();
        if balls.is_empty() {
            return tree;
        }
        let origin = balls[0].0;
        let mut items: Vec<(u32, Vector3<Real>)> = balls
            .iter()
            .enumerate()
            .map(|(i, (c, _))| {
                let at = G::log(&origin, c).map_or_else(Vector3::zeros, |v| coords::<G>(&v));
                (u32::try_from(i).expect("too many balls"), at)
            })
            .collect();
        tree.root = Some(tree.split(balls, &mut items));
        tree
    }

    fn split(&mut self, balls: &[(G::Point, Real)], items: &mut [(u32, Vector3<Real>)]) -> usize {
        if let [(id, _)] = items {
            let (centre, radius) = balls[*id as usize];
            self.nodes.push(Node {
                centre,
                radius,
                kind: Kind::Leaf(*id),
            });
            return self.nodes.len() - 1;
        }
        // The axis of widest spread, split at its median.
        let (lo, hi) = items.iter().fold(
            (
                Vector3::repeat(Real::INFINITY),
                Vector3::repeat(Real::NEG_INFINITY),
            ),
            |(lo, hi), (_, at)| (lo.inf(at), hi.sup(at)),
        );
        let axis = (hi - lo).imax();
        let half = items.len() / 2;
        items.select_nth_unstable_by(half, |a, b| a.1[axis].total_cmp(&b.1[axis]));
        let (left, right) = items.split_at_mut(half);
        let (l, r) = (self.split(balls, left), self.split(balls, right));
        let (centre, radius) = merge::<G>(
            (&self.nodes[l].centre, self.nodes[l].radius),
            (&self.nodes[r].centre, self.nodes[r].radius),
        );
        self.nodes.push(Node {
            centre,
            radius,
            kind: Kind::Branch(l, r),
        });
        self.nodes.len() - 1
    }

    /// How many balls it holds.
    pub fn len(&self) -> usize {
        self.nodes
            .iter()
            .filter(|node| matches!(node.kind, Kind::Leaf(_)))
            .count()
    }

    /// Whether it holds none.
    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    /// Calls `found` with every ball that comes within `radius` of `at` (its index in the
    /// balls it was built from).
    pub fn near(&self, at: &G::Point, radius: Real, mut found: impl FnMut(u32)) {
        let Some(root) = self.root else { return };
        let mut stack = vec![root];
        while let Some(i) = stack.pop() {
            let node = &self.nodes[i];
            if G::distance(&node.centre, at) > node.radius + radius {
                continue;
            }
            match node.kind {
                Kind::Leaf(id) => found(id),
                Kind::Branch(l, r) => stack.extend([l, r]),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use fk_geometry_euclidean::E3;
    use fk_math::nalgebra::Point3;

    use super::*;

    fn ball(x: Real, y: Real, z: Real, r: Real) -> (Point3<Real>, Real) {
        (Point3::new(x, y, z), r)
    }

    #[test]
    fn a_merged_ball_holds_both() {
        let (c, r) = merge::<E3>(
            (&Point3::new(0.0, 0.0, 0.0), 1.0),
            (&Point3::new(4.0, 0.0, 0.0), 2.0),
        );
        assert!((r - 3.5).abs() < 1e-12);
        assert!((c - Point3::new(2.5, 0.0, 0.0)).norm() < 1e-12);
        // One inside the other: the bigger one.
        let (c, r) = merge::<E3>(
            (&Point3::new(0.0, 0.0, 0.0), 5.0),
            (&Point3::new(1.0, 0.0, 0.0), 1.0),
        );
        assert_eq!((c, r), (Point3::new(0.0, 0.0, 0.0), 5.0));
    }

    #[test]
    fn a_query_finds_exactly_what_a_scan_finds() {
        // A spiral of balls, like the Field's towers, and queries all over it.
        let balls: Vec<_> = (0..500)
            .map(|i| {
                let a = Real::from(i) * 2.399_963;
                let r = 3.0 * Real::from(i).sqrt();
                ball(
                    r * a.cos(),
                    0.5 * (a * 3.0).sin(),
                    r * a.sin(),
                    0.5 + 0.3 * (a * 7.0).cos().abs(),
                )
            })
            .collect();
        let tree = BallTree::<E3>::build(&balls);
        assert_eq!(tree.len(), 500);
        for k in 0..60 {
            let a = Real::from(k) * 0.7;
            let at = Point3::new(40.0 * a.cos() * (Real::from(k) / 60.0), 0.0, 40.0 * a.sin());
            let radius = 1.0 + Real::from(k % 5);
            let mut found = Vec::new();
            tree.near(&at, radius, |id| found.push(id));
            found.sort_unstable();
            let scanned: Vec<u32> = balls
                .iter()
                .enumerate()
                .filter(|(_, (c, r))| (c - at).norm() <= r + radius)
                .map(|(i, _)| i as u32)
                .collect();
            assert_eq!(found, scanned, "query {k}");
        }
    }

    #[test]
    fn an_empty_tree_finds_nothing() {
        let tree = BallTree::<E3>::build(&[]);
        assert!(tree.is_empty());
        tree.near(&Point3::origin(), 10.0, |_| panic!("nothing to find"));
    }
}
