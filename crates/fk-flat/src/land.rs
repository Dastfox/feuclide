//! Ground as a height field: a finite land a body walks on, drawn in bands (meadow, rock,
//! snow), and the round walls that keep it in (Bevy leaves terrain to third-party crates).
//!
//! A ground is a [`Land`]: its height at every point of the ground plane. It is drawn by
//! [`land_meshes`] over a [`Grid`] (a square, or rings round a hole such as a lake), each
//! triangle in the band its middle falls in, and walked on through a [`LandField`]: the height
//! difference, scaled down by the steepest slope nearby so that it stays a lower bound of the
//! distance. A [`Bound`] is a round wall at the world's edge, beyond which the land is only
//! seen.

use std::f64::consts::TAU;
use std::sync::Arc;

use fk_geometry_euclidean::E3;
use fk_math::Real;
use fk_math::nalgebra::{Point3, Vector2, Vector3};
use fk_physics::DistanceField;
use fk_render::MeshBuilder;
use serde::Deserialize;

use crate::kit::{Kit, translation};

/// A place's ground: its height over the ground plane.
pub trait Land: Send + Sync + 'static {
    /// The height at `at` (x, z), metres.
    fn height(&self, at: &Vector2<Real>) -> Real;

    /// The steepest slope anywhere a walker can walk, rise over run.
    fn steepest(&self) -> Real;
}

/// Swells crossing the ground: `[height, length, heading in degrees]` each, every one from 0 to
/// its height and back over its length, along its heading.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(transparent)]
pub struct Waves(pub Vec<[Real; 3]>);

impl Waves {
    /// Their height at `at`.
    pub fn height(&self, at: &Vector2<Real>) -> Real {
        self.0
            .iter()
            .map(|&[height, length, heading]| {
                let h = heading.to_radians();
                let along = at.x * h.cos() + at.y * h.sin();
                height * (0.5 + 0.5 * (TAU * along / length.max(1e-6)).sin())
            })
            .sum()
    }

    /// Their steepest slope.
    pub fn steepest(&self) -> Real {
        self.0
            .iter()
            .map(|&[height, length, _]| height * std::f64::consts::PI / length.max(1e-6))
            .sum()
    }
}

/// A ring of mountains (or of hills, cliffs or rocks) round a centre: the ground rises from
/// `from` metres out to its full `height` at `to`, its crest varying round the ring with
/// `peaks` (how many, and how much as a share of the height).
#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Ring {
    /// Where the rise begins, metres from the centre.
    pub from: Real,
    /// Where it is full.
    pub to: Real,
    /// How high, metres; below zero, a fall.
    pub height: Real,
    /// How many peaks round the ring, and how much they vary its height.
    pub peaks: [Real; 2],
}

impl Ring {
    /// Its height at `at`, from its centre.
    pub fn height(&self, at: &Vector2<Real>) -> Real {
        let r = at.norm();
        let rise = smoothstep(self.from, self.to, r);
        if rise <= 0.0 {
            return 0.0;
        }
        let angle = at.y.atan2(at.x);
        let [count, vary] = self.peaks;
        let crest = 1.0
            + vary
                * (0.6 * (count * angle).sin() + 0.4 * ((count * 1.7 + 1.0) * angle + 1.3).sin());
        self.height * rise * crest
    }

    /// Its steepest slope.
    pub fn steepest(&self) -> Real {
        1.5 * self.height.abs() * (1.0 + self.peaks[1]) / (self.to - self.from).abs().max(1e-6)
    }
}

pub use fk_math::smoothstep;

/// What the ground is drawn over.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Grid {
    /// A square `half` metres either way of `centre`, in cells `cell` wide.
    Square {
        /// Its middle.
        centre: Vector2<Real>,
        /// Half its side.
        half: Real,
        /// A cell's side.
        cell: Real,
    },
    /// A rectangle from `min` to `max` (x, z), in cells about `cell` wide.
    Rect {
        /// One corner.
        min: Vector2<Real>,
        /// The opposite one.
        max: Vector2<Real>,
        /// A cell's side, about.
        cell: Real,
    },
    /// Rings round `centre` from `inner` to `outer` metres: a ground with a round hole.
    Polar {
        /// The hole's middle.
        centre: Vector2<Real>,
        /// The hole's radius.
        inner: Real,
        /// How far out.
        outer: Real,
        /// Rings, closer near the hole.
        rings: u32,
        /// Pieces round each ring.
        segments: u32,
        /// How much closer the rings are near the hole: 1 evenly spaced, more closer.
        stretch: Real,
    },
}

impl Grid {
    /// The points of the grid as rows and columns, and how many of each.
    fn points(&self) -> (Vec<Vector2<Real>>, u32, u32) {
        match *self {
            Self::Square { centre, half, cell } => {
                let n = (2.0 * half / cell.max(0.1)).ceil() as u32;
                let step = 2.0 * half / Real::from(n);
                let points = (0..=n)
                    .flat_map(|i| {
                        (0..=n).map(move |j| {
                            centre
                                + Vector2::new(
                                    -half + Real::from(i) * step,
                                    -half + Real::from(j) * step,
                                )
                        })
                    })
                    .collect();
                (points, n + 1, n + 1)
            }
            Self::Rect { min, max, cell } => {
                let size = max - min;
                let nx = (size.x / cell.max(0.1)).ceil().max(1.0) as u32;
                let nz = (size.y / cell.max(0.1)).ceil().max(1.0) as u32;
                let points = (0..=nx)
                    .flat_map(|i| {
                        (0..=nz).map(move |j| {
                            min + Vector2::new(
                                size.x * Real::from(i) / Real::from(nx),
                                size.y * Real::from(j) / Real::from(nz),
                            )
                        })
                    })
                    .collect();
                (points, nx + 1, nz + 1)
            }
            Self::Polar {
                centre,
                inner,
                outer,
                rings,
                segments,
                stretch,
            } => {
                let rings = rings.max(1);
                let segments = segments.max(3);
                let points = (0..=rings)
                    .flat_map(|i| {
                        let t = (Real::from(i) / Real::from(rings)).powf(stretch.max(1.0));
                        let r = inner + (outer - inner) * t;
                        // The last column closes the ring on the first.
                        (0..=segments).map(move |j| {
                            let a = TAU * Real::from(j) / Real::from(segments);
                            centre + Vector2::new(a.cos(), a.sin()) * r
                        })
                    })
                    .collect();
                (points, rings + 1, segments + 1)
            }
        }
    }
}

/// The ground of `land` over `grid` as one mesh per band: each triangle in the band `band`
/// gives its middle `(x, z)` and height, out of `bands`; a band past the last leaves the
/// triangle out.
pub fn land_meshes(
    land: &dyn Land,
    grid: &Grid,
    bands: usize,
    band: impl Fn(&Vector2<Real>, Real) -> usize,
) -> Vec<MeshBuilder> {
    let (points, rows, columns) = grid.points();
    let e = 0.5;
    let height = |p: &Vector2<Real>| land.height(p);
    let normal = |p: &Vector2<Real>| {
        let dx =
            (height(&(p + Vector2::new(e, 0.0))) - height(&(p - Vector2::new(e, 0.0)))) / (2.0 * e);
        let dz =
            (height(&(p + Vector2::new(0.0, e))) - height(&(p - Vector2::new(0.0, e)))) / (2.0 * e);
        Vector3::new(-dx, 1.0, -dz).normalize()
    };
    let heights: Vec<Real> = points.iter().map(&height).collect();
    let mut meshes: Vec<MeshBuilder> = (0..bands.max(1)).map(|_| MeshBuilder::new()).collect();
    let mut index: Vec<Vec<Option<u32>>> = vec![vec![None; points.len()]; meshes.len()];
    let mut vertex = |b: usize, k: usize, meshes: &mut Vec<MeshBuilder>| -> u32 {
        *index[b][k].get_or_insert_with(|| {
            let p = points[k];
            meshes[b].vertex(Vector3::new(p.x, heights[k], p.y), normal(&p))
        })
    };
    let at = |i: u32, j: u32| (i * columns + j) as usize;
    for i in 0..rows - 1 {
        for j in 0..columns - 1 {
            let corners = [at(i, j), at(i, j + 1), at(i + 1, j + 1), at(i + 1, j)];
            let middle = corners.iter().map(|&k| points[k]).sum::<Vector2<Real>>() / 4.0;
            let h = corners.iter().map(|&k| heights[k]).sum::<Real>() / 4.0;
            let b = band(&middle, h);
            if b >= meshes.len() {
                continue;
            }
            let [a, bb, c, d] = corners.map(|k| vertex(b, k, &mut meshes));
            // Facing up whichever way the grid turns.
            let (pa, pb, pc) = (points[corners[0]], points[corners[1]], points[corners[2]]);
            let up = (pb - pa).perp(&(pc - pa)) < 0.0;
            if up {
                meshes[b].quad(a, bb, c, d);
            } else {
                meshes[b].quad(a, d, c, bb);
            }
        }
    }
    meshes
}

/// The ground as a distance, for walking on it; only within a region, if it is one of several
/// worlds in the same space.
#[derive(Clone)]
pub struct LandField {
    land: Arc<dyn Land>,
    lip: Real,
    region: Option<(Vector2<Real>, Real)>,
}

impl LandField {
    /// The ground `land`.
    pub fn new(land: Arc<dyn Land>) -> Self {
        let lip = 1.0 / (1.0 + land.steepest().powi(2)).sqrt();
        Self {
            land,
            lip,
            region: None,
        }
    }

    /// The same ground, only within `radius` of `centre`: further out it is no ground at all.
    pub fn within(self, centre: Vector2<Real>, radius: Real) -> Self {
        Self {
            region: Some((centre, radius)),
            ..self
        }
    }
}

impl DistanceField<E3> for LandField {
    fn distance(&self, p: &Point3<Real>) -> Real {
        let at = Vector2::new(p.x, p.z);
        if let Some((centre, radius)) = self.region {
            let out = (at - centre).norm() - radius;
            if out > 0.0 {
                // The ground is all inside: at least this far.
                return out;
            }
        }
        height_distance(self.land.as_ref(), self.lip, &at, p.y)
    }
}

/// How far across the slope near a point is sampled, metres: the ground's distance is read from
/// it up to this far, from the steepest slope anywhere beyond.
const NEAR: Real = 1.0;

/// The distance from `y` over `at` to `land`, at least (negative under it): the height
/// difference scaled by the steepest slope within [`NEAR`] of `at` (sampled, with a margin),
/// as far as [`NEAR`], and by `lip` (from [`Land::steepest`]) anywhere, so that cliffs far off
/// don't lift a walker off the flat it stands on.
pub fn height_distance(land: &dyn Land, lip: Real, at: &Vector2<Real>, y: Real) -> Real {
    let gap = y - land.height(at);
    let far = gap * lip;
    if far.abs() >= NEAR {
        return far;
    }
    let slope = |at: Vector2<Real>| {
        let e = 0.05;
        let dx =
            land.height(&(at + Vector2::new(e, 0.0))) - land.height(&(at - Vector2::new(e, 0.0)));
        let dz =
            land.height(&(at + Vector2::new(0.0, e))) - land.height(&(at - Vector2::new(0.0, e)));
        Vector2::new(dx, dz).norm() / (2.0 * e)
    };
    let around = (0..8)
        .flat_map(|i| {
            let a = Real::from(i) * TAU / 8.0;
            let way = Vector2::new(a.cos(), a.sin()) * NEAR;
            [slope(at + way * 0.5), slope(at + way)]
        })
        .fold(slope(*at), Real::max);
    // A margin for what the samples miss, never past the steepest anywhere.
    let steepest = (1.0 / (lip * lip) - 1.0).max(0.0).sqrt();
    let near = (around * 1.25 + 0.05).min(steepest);
    // Within NEAR of `at`, the ground is no nearer than this; or it is further than NEAR.
    let close = (gap.abs() / (1.0 + near * near).sqrt()).min(NEAR);
    gap.signum() * close.max(far.abs())
}

/// A round wall at a world's edge, from `radius` out to [`Bound::THICKNESS`] further: a walker goes
/// no further out than `radius` from `centre`, and the wall leaves the space beyond it free for
/// other worlds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bound {
    /// The middle of the world (x, z).
    pub centre: Vector2<Real>,
    /// How far out a walker can go.
    pub radius: Real,
}

impl Bound {
    /// How thick the wall is, metres.
    pub const THICKNESS: Real = 50.0;
}

impl DistanceField<E3> for Bound {
    fn distance(&self, p: &Point3<Real>) -> Real {
        let r = (Vector2::new(p.x, p.z) - self.centre).norm();
        (self.radius - r).max(r - self.radius - Self::THICKNESS)
    }
}

/// A flat rectangle at `height` from `min` to `max` (x, z), facing up: a sea of mist.
pub fn sheet_between<M: Copy>(min: Vector2<Real>, max: Vector2<Real>, height: Real) -> Kit<M> {
    let mut mesh = MeshBuilder::new();
    let corner = |x: Real, z: Real| Vector3::new(x, height, z);
    mesh.flat_quad(
        [
            corner(min.x, min.y),
            corner(min.x, max.y),
            corner(max.x, max.y),
            corner(max.x, min.y),
        ],
        Vector3::y(),
    );
    Kit::from_mesh(mesh)
}

/// A flat sheet at `height`, `radius` round `centre`: a sea of mist or of cloud.
pub fn sheet<M: Copy>(centre: Vector2<Real>, height: Real, radius: Real) -> Kit<M> {
    let mut kit = Kit::default();
    kit.add(
        &fk_render::shapes::disc(radius, 96),
        &translation(Vector3::new(centre.x, height, centre.y)),
    );
    kit
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Hill;

    impl Land for Hill {
        fn height(&self, at: &Vector2<Real>) -> Real {
            (10.0 - at.norm()).max(0.0) * 0.5
        }
        fn steepest(&self) -> Real {
            0.5
        }
    }

    #[test]
    fn the_land_is_drawn_facing_up_and_walked_on() {
        for grid in [
            Grid::Square {
                centre: Vector2::zeros(),
                half: 12.0,
                cell: 2.0,
            },
            Grid::Polar {
                centre: Vector2::zeros(),
                inner: 2.0,
                outer: 12.0,
                rings: 6,
                segments: 24,
                stretch: 1.4,
            },
        ] {
            let kits = land_meshes(&Hill, &grid, 2, |_, h| usize::from(h > 2.0));
            assert_eq!(kits.len(), 2);
            for kit in &kits {
                let mesh = kit;
                for t in mesh.indices().as_chunks::<3>().0 {
                    let [a, b, c] = t.map(|i| mesh.positions()[i as usize]);
                    assert!((b - a).cross(&(c - a)).y > 0.0, "{grid:?} faces up");
                }
            }
        }
        let field = LandField::new(Arc::new(Hill));
        assert!(field.distance(&Point3::new(0.0, 4.9, 0.0)) < 0.0);
        assert!(field.distance(&Point3::new(0.0, 5.1, 0.0)) > 0.0);
    }

    struct Cliffs(Ring);

    impl Land for Cliffs {
        fn height(&self, at: &Vector2<Real>) -> Real {
            self.0.height(at)
        }
        fn steepest(&self) -> Real {
            self.0.steepest()
        }
    }

    #[test]
    fn far_cliffs_dont_lift_him_off_the_flat() {
        let cliffs = Cliffs(Ring {
            from: 95.0,
            to: 112.0,
            height: -90.0,
            peaks: [7.0, 0.15],
        });
        assert!(cliffs.steepest() > 5.0);
        let field = LandField::new(Arc::new(cliffs));
        for y in [0.1, 0.3, 0.9, -0.5] {
            let d = field.distance(&Point3::new(1.0, y, 2.0));
            assert!((d - y).abs() < 0.02 * y.abs(), "{y}: {d}");
        }
        // Higher, it never falls, and the steepest slope anywhere still bounds it.
        let heights: Vec<Real> = (0..60).map(|i| Real::from(i) * 0.1).collect();
        let d: Vec<Real> = heights
            .iter()
            .map(|&y| field.distance(&Point3::new(1.0, y, 2.0)))
            .collect();
        assert!(d.windows(2).all(|w| w[1] >= w[0]), "{d:?}");
        assert!(field.distance(&Point3::new(1.0, 30.0, 2.0)) < 5.0);
        // Over the cliffs' edge, the nearby slope does.
        let edge = Point3::new(103.0, cliffs_height(103.0) + 0.5, 0.0);
        assert!(field.distance(&edge) < 0.2);
    }

    fn cliffs_height(r: Real) -> Real {
        Ring {
            from: 95.0,
            to: 112.0,
            height: -90.0,
            peaks: [7.0, 0.15],
        }
        .height(&Vector2::new(r, 0.0))
    }

    #[test]
    fn a_ring_rises_to_its_height_and_a_bound_keeps_him_in() {
        let ring = Ring {
            from: 100.0,
            to: 200.0,
            height: 50.0,
            peaks: [5.0, 0.0],
        };
        assert_eq!(ring.height(&Vector2::new(50.0, 0.0)), 0.0);
        assert!((ring.height(&Vector2::new(0.0, 250.0)) - 50.0).abs() < 1e-9);
        let bound = Bound {
            centre: Vector2::zeros(),
            radius: 90.0,
        };
        assert!(bound.distance(&Point3::new(95.0, 0.0, 0.0)) < 0.0);
        assert!(bound.distance(&Point3::new(80.0, 0.0, 0.0)) > 0.0);
    }

    #[test]
    fn waves_rise_from_nothing_to_their_height() {
        let waves = Waves(vec![[2.0, 10.0, 0.0]]);
        let heights: Vec<Real> = (0..20)
            .map(|i| waves.height(&Vector2::new(Real::from(i) * 0.5, 0.0)))
            .collect();
        assert!(heights.iter().all(|h| (0.0..=2.0).contains(h)));
        assert!(heights.iter().cloned().fold(0.0, Real::max) > 1.9);
    }
}
