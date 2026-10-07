//! A uniform grid of buckets over a plane, to find what is near among thousands of things.

use std::collections::HashMap;

use fk_math::Real;
use fk_math::nalgebra::Vector2;

/// Ids filed by position on a plane, in square buckets `cell` wide.
///
/// The slice's broadphase: the towers of the Field stand on one plane, so a grid over it finds
/// the few near the walker. The geodesic-ball BVH replaces it in E1.
#[derive(Clone, Debug)]
pub struct BucketGrid {
    cell: Real,
    buckets: HashMap<(i64, i64), Vec<u32>>,
}

impl BucketGrid {
    /// An empty grid of buckets `cell` wide.
    ///
    /// # Panics
    ///
    /// If `cell` is not positive.
    pub fn new(cell: Real) -> Self {
        assert!(cell > 0.0, "bucket size must be positive, got {cell}");
        Self {
            cell,
            buckets: HashMap::new(),
        }
    }

    fn bucket(&self, at: &Vector2<Real>) -> (i64, i64) {
        (
            (at.x / self.cell).floor() as i64,
            (at.y / self.cell).floor() as i64,
        )
    }

    /// Files `id` at `at`.
    pub fn insert(&mut self, at: &Vector2<Real>, id: u32) {
        let bucket = self.bucket(at);
        self.buckets.entry(bucket).or_default().push(id);
    }

    /// Every id filed within `radius` of `at`, and maybe some a little further: those in the
    /// buckets the square around the disc touches.
    pub fn near(&self, at: &Vector2<Real>, radius: Real) -> impl Iterator<Item = u32> + use<'_> {
        let reach = Vector2::repeat(radius.max(0.0));
        let (low, high) = (self.bucket(&(at - reach)), self.bucket(&(at + reach)));
        (low.0..=high.0)
            .flat_map(move |i| (low.1..=high.1).map(move |j| (i, j)))
            .filter_map(|bucket| self.buckets.get(&bucket))
            .flatten()
            .copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_what_is_near_and_not_what_is_far() {
        let mut grid = BucketGrid::new(10.0);
        grid.insert(&Vector2::new(1.0, 1.0), 0);
        grid.insert(&Vector2::new(-12.0, 3.0), 1);
        grid.insert(&Vector2::new(100.0, 100.0), 2);
        let mut near: Vec<u32> = grid.near(&Vector2::new(0.0, 0.0), 13.0).collect();
        near.sort_unstable();
        assert_eq!(near, [0, 1]);
        assert_eq!(
            grid.near(&Vector2::new(0.0, 0.0), 1.0).collect::<Vec<_>>(),
            [0]
        );
        assert_eq!(
            grid.near(&Vector2::new(95.0, 99.0), 6.0)
                .collect::<Vec<_>>(),
            [2]
        );
    }
}
