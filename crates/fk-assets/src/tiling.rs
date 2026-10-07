//! Regular tilings of space by cubes or dodecahedra: `{4,3,r}` and `{5,3,r}`, `r` cells round
//! every edge, in whichever geometry has them. Flat space has `{4,3,4}`; hyperbolic space
//! `{4,3,5}`, `{5,3,4}`, `{5,3,5}` and, with vertices at infinity, `{4,3,6}`; the 3-sphere the
//! 8-cell `{4,3,3}` and the 120-cell `{5,3,3}`.
//!
//! In the projective models of curved space (Beltrami–Klein for H³, gnomonic for S³) geodesics
//! are straight, so a cell is a Euclidean cube or dodecahedron about the origin, its size the
//! one that makes its dihedral angle `θ = 2π/r`. With face planes `n·x = d` (`n` unit), the
//! Minkowski or Euclidean normals `(n, ∓d)` of two adjacent faces (`n₁·n₂ = c`) meet at `π − θ`
//! when `d² = −K (c + cos θ) / (1 + cos θ)` at curvature `K = ±1`.
//!
//! Neighbours are reached without mirrors: the reflection of the cell in its face `f` is, up
//! to a symmetry of the cell, the transvection twice the inradius along the face's normal
//! after a turn of `π/p` about it (none for squares: a cube is symmetric through its centre
//! plane, a dodecahedron's opposite face is turned by a tenth of a turn). So every cell is
//! placed by an orientation-preserving isometry ([`Tiling::cells`]), found breadth first from
//! the origin's cell and each placed once.

use fk_geometry::{Geometry, GroupElement};
use fk_math::Real;
use fk_math::nalgebra::Vector3;
use fk_quotient::Quotient;
use fk_render::{MeshBuilder, tangent};

/// Two unit vectors square to `n` and to each other, `u × w = n`.
fn square_to(n: &Vector3<Real>) -> (Vector3<Real>, Vector3<Real>) {
    let other = if n.x.abs() < 0.9 {
        Vector3::x()
    } else {
        Vector3::y()
    };
    let u = (other - n * n.dot(&other)).normalize();
    (u, n.cross(&u))
}

/// The shape of a cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cell {
    /// Square faces, three at each vertex.
    Cube,
    /// Pentagonal faces, three at each vertex.
    Dodecahedron,
}

impl Cell {
    /// Unit normals of the faces.
    fn normals(self) -> Vec<Vector3<Real>> {
        match self {
            Self::Cube => vec![
                Vector3::x(),
                -Vector3::x(),
                Vector3::y(),
                -Vector3::y(),
                Vector3::z(),
                -Vector3::z(),
            ],
            // The dual icosahedron's vertices, (0, ±φ, ±1) and its cyclic shifts, for the
            // vertices below.
            Self::Dodecahedron => {
                let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
                let mut normals = Vec::new();
                for a in [phi, -phi] {
                    for b in [1.0, -1.0] {
                        normals.push(Vector3::new(0.0, a, b));
                        normals.push(Vector3::new(b, 0.0, a));
                        normals.push(Vector3::new(a, b, 0.0));
                    }
                }
                normals.into_iter().map(|n| n.normalize()).collect()
            }
        }
    }

    /// The vertices of the Euclidean cell whose faces are at distance 1 from its centre.
    fn vertices(self) -> Vec<Vector3<Real>> {
        let raw: Vec<Vector3<Real>> = match self {
            Self::Cube => (0..8)
                .map(|i| {
                    let s = |bit: i32| if i & bit == 0 { 1.0 } else { -1.0 };
                    Vector3::new(s(1), s(2), s(4))
                })
                .collect(),
            Self::Dodecahedron => {
                let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
                let mut v = Vec::new();
                for i in 0..8 {
                    let s = |bit: i32| if i & bit == 0 { 1.0 } else { -1.0 };
                    v.push(Vector3::new(s(1), s(2), s(4)));
                }
                for a in [1.0 / phi, -1.0 / phi] {
                    for b in [phi, -phi] {
                        v.push(Vector3::new(0.0, a, b));
                        v.push(Vector3::new(a, b, 0.0));
                        v.push(Vector3::new(b, 0.0, a));
                    }
                }
                v
            }
        };
        let inradius = self.normals()[0];
        let r = raw.iter().map(|v| v.dot(&inradius)).fold(0.0, Real::max);
        raw.into_iter().map(|v| v / r).collect()
    }

    /// Sides of a face.
    fn sides(self) -> u32 {
        match self {
            Self::Cube => 4,
            Self::Dodecahedron => 5,
        }
    }

    /// `n₁·n₂` for two faces sharing an edge.
    fn adjacent_cos(self) -> Real {
        let normals = self.normals();
        normals[1..]
            .iter()
            .map(|n| n.dot(&normals[0]))
            .filter(|&c| c > -0.99 && c < 0.99)
            .fold(Real::NEG_INFINITY, Real::max)
    }
}

/// A regular tiling of `G` by cells: see the module.
#[derive(Clone, Debug)]
pub struct Tiling<G: Geometry> {
    cell: Cell,
    /// Distance from a cell's centre to its faces in the projective model.
    projective_inradius: Real,
    /// The same, as a geodesic distance.
    inradius: Real,
    /// What carries the origin's cell to its neighbour across each face.
    neighbours: Vec<G::Isometry>,
}

impl<G: Geometry> Tiling<G> {
    /// The tiling of `G` by `cell`s, `around_edge` round every edge, if `G` has one: `None` when
    /// its curvature is not ±1 (or 0 with anything but cubes four round an edge), when the
    /// cells would not close (`d²` not positive), or when their vertices would lie beyond
    /// infinity in hyperbolic space. Vertices at infinity (`{4,3,6}`) are allowed: their
    /// cells' edges are cut short of it ([`frame`](Self::frame)).
    pub fn regular(cell: Cell, around_edge: u32) -> Option<Self> {
        let k = G::constant_curvature()?;
        let theta = std::f64::consts::TAU / Real::from(around_edge);
        let (projective_inradius, inradius) = if k == 0.0 {
            if cell != Cell::Cube || around_edge != 4 {
                return None;
            }
            (1.0, 1.0)
        } else if (k.abs() - 1.0).abs() < 1e-12 {
            let c = cell.adjacent_cos();
            let d2 = -k * (c + theta.cos()) / (1.0 + theta.cos());
            if d2 <= 0.0 {
                return None;
            }
            let d = d2.sqrt();
            if k < 0.0 {
                let circumradius = cell.vertices()[0].norm() * d;
                if circumradius > 1.0 + 1e-9 {
                    return None;
                }
                (d, d.atanh())
            } else {
                (d, d.atan())
            }
        } else {
            return None;
        };
        let turn = if cell.sides().is_multiple_of(2) {
            0.0
        } else {
            -std::f64::consts::PI / Real::from(cell.sides())
        };
        let neighbours = cell
            .normals()
            .iter()
            .map(|n| {
                let (u, w) = square_to(n);
                let rotation =
                    G::Isometry::exp(&(G::rotation(&tangent::<G>(&u), &tangent::<G>(&w)) * turn));
                let across =
                    G::Isometry::exp(&G::transvection(&(tangent::<G>(n) * (2.0 * inradius))));
                across.compose(&rotation)
            })
            .collect();
        Some(Self {
            cell,
            projective_inradius,
            inradius,
            neighbours,
        })
    }

    /// The shape of its cells.
    pub fn cell(&self) -> Cell {
        self.cell
    }

    /// The geodesic distance from a cell's centre to its faces.
    pub fn inradius(&self) -> Real {
        self.inradius
    }

    /// The isometries that carry the origin's cell to each of its neighbours.
    pub fn neighbours(&self) -> &[G::Isometry] {
        &self.neighbours
    }

    /// The space made of one cell, each face glued to the opposite one after a turn of `turn`
    /// about their common normal: the quotient whose generators are the transvections across
    /// the faces after that turn (one per pair of opposite faces; the other face's is its
    /// inverse), its fundamental domain the origin's cell. With dodecahedra, a tenth of a turn
    /// on the 120-cell's is the Poincaré homology sphere, three tenths on `{5,3,5}`'s the
    /// Seifert–Weber space, a closed hyperbolic manifold. Only some turns make a manifold: the
    /// cells round every edge must close up with the identity.
    pub fn glued(&self, turn: Real) -> Quotient<G> {
        let generators = self
            .cell
            .normals()
            .iter()
            .filter(|n| (n.x, n.y, n.z) > (0.0, 0.0, 0.0))
            .map(|n| {
                let (u, w) = square_to(n);
                let twist =
                    G::Isometry::exp(&(G::rotation(&tangent::<G>(&u), &tangent::<G>(&w)) * turn));
                G::Isometry::exp(&G::transvection(&(tangent::<G>(n) * (2.0 * self.inradius))))
                    .compose(&twist)
            })
            .collect();
        Quotient::new(G::origin(), generators)
    }

    /// Every cell whose centre is within `within` of the origin, as the isometry placing the
    /// origin's cell there (the origin's own first), breadth first, at most `most` of them.
    /// On the 3-sphere, `within` π takes them all.
    pub fn cells(&self, within: Real, most: usize) -> Vec<G::Isometry> {
        let origin = G::origin();
        let mut found = vec![G::Isometry::identity()];
        let mut centres = vec![origin];
        let mut frontier = vec![0];
        while !frontier.is_empty() && found.len() < most {
            let mut next = Vec::new();
            for i in frontier {
                for step in &self.neighbours {
                    let mut g = found[i].compose(step);
                    g.renormalize();
                    let centre = G::apply(&g, &origin);
                    if G::distance(&origin, &centre) > within
                        || centres
                            .iter()
                            .any(|seen| G::distance(seen, &centre) < 1e-6 * (1.0 + self.inradius))
                    {
                        continue;
                    }
                    found.push(g);
                    centres.push(centre);
                    next.push(found.len() - 1);
                    if found.len() >= most {
                        return found;
                    }
                }
            }
            frontier = next;
        }
        found
    }

    /// The point of `G` at the projective point `x` (Klein or gnomonic coordinates, about the
    /// origin, in the reference frame).
    fn point(&self, x: &Vector3<Real>) -> G::Point {
        let r = x.norm();
        let origin = G::origin();
        if r < 1e-15 {
            return origin;
        }
        let k = G::constant_curvature().unwrap_or(0.0);
        let d = if k < 0.0 {
            r.min(1.0 - 1e-6).atanh()
        } else if k > 0.0 {
            r.atan()
        } else {
            r
        };
        G::exp(&origin, &(tangent::<G>(&(x / r)) * d))
    }

    /// The origin's cell's edges as square beams `thickness` across, to draw (authored in the
    /// tangent space at the origin, as every mesh), the cell's vertices drawn in by `inset`
    /// towards its centre so that the beams of the cells round an edge lie side by side rather
    /// than one in another. Each beam's corners are placed on the manifold, each pair carried
    /// along the edge by parallel transport, so its long faces are geodesic. Edges running to
    /// infinity (`{4,3,6}`) are cut at `cut` of the way there in the Klein model.
    pub fn frame(&self, thickness: Real, inset: Real, cut: Real) -> MeshBuilder {
        let corners: Vec<Vector3<Real>> = self
            .cell
            .vertices()
            .into_iter()
            .map(|v| {
                let x = v * self.projective_inradius;
                if x.norm() > cut {
                    x * (cut / x.norm())
                } else {
                    x
                }
            })
            .collect();
        let shortest = corners
            .iter()
            .enumerate()
            .flat_map(|(i, a)| corners[i + 1..].iter().map(move |b| (a - b).norm()))
            .fold(Real::INFINITY, Real::min);
        let mut mesh = MeshBuilder::new();
        let origin = G::origin();
        let coords = |p: &G::Point| {
            let v = G::log(&origin, p).expect("a cell lies within its injectivity radius");
            Vector3::from_fn(|i, _| G::inner(&origin, &v, &G::origin_frame(i)))
        };
        let half = thickness / 2.0;
        for (i, a) in corners.iter().enumerate() {
            for b in &corners[i + 1..] {
                if (a - b).norm() > shortest * 1.01 {
                    continue;
                }
                let drawn_in = |x: &Vector3<Real>| {
                    let p = self.point(x);
                    let v = G::log(&origin, &p).expect("within the injectivity radius");
                    let d = G::norm(&origin, &v);
                    G::exp(&origin, &(v * ((d - inset).max(0.0) / d.max(1e-12))))
                };
                let (p, q) = (drawn_in(a), drawn_in(b));
                let Some(along) = G::log(&p, &q) else {
                    continue;
                };
                // A frame about the edge at `p`: two directions square to it and to each other.
                let length = G::norm(&p, &along);
                let e = along * (1.0 / length);
                let towards = G::log(&p, &origin).unwrap_or_else(|| G::origin_frame(0));
                let mut u = towards - e * G::inner(&p, &towards, &e);
                if G::norm(&p, &u) < 1e-9 {
                    u = G::project_tangent(&p, &G::origin_frame(1));
                    u = u - e * G::inner(&p, &u, &e);
                }
                let u = u * (1.0 / G::norm(&p, &u));
                // The third direction: whatever of each reference axis is left, the longest.
                let w = (0..G::DIM)
                    .map(|i| {
                        let t = G::parallel_transport(&origin, &p, &G::origin_frame(i));
                        t - e * G::inner(&p, &t, &e) - u * G::inner(&p, &t, &u)
                    })
                    .max_by(|x, y| G::norm(&p, x).total_cmp(&G::norm(&p, y)))
                    .expect("at least one axis");
                let w = w * (1.0 / G::norm(&p, &w));
                let ring = |at: &G::Point, u: G::Tangent, w: G::Tangent| -> [Vector3<Real>; 4] {
                    [(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)]
                        .map(|(s, t)| coords(&G::exp(at, &(u * (s * half) + w * (t * half)))))
                };
                let start = ring(&p, u, w);
                let end = ring(
                    &q,
                    G::parallel_transport(&p, &q, &u),
                    G::parallel_transport(&p, &q, &w),
                );
                for k in 0..4 {
                    let l = (k + 1) % 4;
                    let normal = (start[k] + start[l] - start[0] - start[2]).normalize();
                    let corners = [start[k], end[k], end[l], start[l]];
                    let [s0, s1, s2, s3] = corners.map(|c| mesh.vertex(c, normal));
                    // Facing out, whichever way the corners turn.
                    let face = (corners[1] - corners[0]).cross(&(corners[3] - corners[0]));
                    if face.dot(&normal) > 0.0 {
                        mesh.quad(s0, s1, s2, s3);
                    } else {
                        mesh.quad(s0, s3, s2, s1);
                    }
                }
            }
        }
        mesh
    }
}

#[cfg(test)]
mod tests {
    use fk_geometry_euclidean::E3;
    use fk_geometry_hyperbolic::H3;
    use fk_geometry_spherical::S3;

    use super::*;

    #[test]
    fn only_the_tilings_that_exist_are_made() {
        assert!(Tiling::<E3>::regular(Cell::Cube, 4).is_some());
        assert!(Tiling::<E3>::regular(Cell::Cube, 5).is_none());
        assert!(Tiling::<H3>::regular(Cell::Dodecahedron, 4).is_some());
        assert!(Tiling::<H3>::regular(Cell::Cube, 5).is_some());
        assert!(
            Tiling::<H3>::regular(Cell::Cube, 6).is_some(),
            "ideal vertices"
        );
        assert!(
            Tiling::<H3>::regular(Cell::Cube, 7).is_none(),
            "beyond infinity"
        );
        assert!(
            Tiling::<H3>::regular(Cell::Cube, 3).is_none(),
            "the 8-cell is spherical"
        );
        assert!(Tiling::<S3>::regular(Cell::Cube, 3).is_some());
        assert!(Tiling::<S3>::regular(Cell::Dodecahedron, 3).is_some());
        assert!(Tiling::<S3>::regular(Cell::Cube, 4).is_none(), "flat");
    }

    #[test]
    fn the_sphere_holds_eight_cubes_and_a_hundred_and_twenty_dodecahedra() {
        let cubes = Tiling::<S3>::regular(Cell::Cube, 3).unwrap();
        assert_eq!(cubes.cells(4.0, 1000).len(), 8);
        let dodecahedra = Tiling::<S3>::regular(Cell::Dodecahedron, 3).unwrap();
        assert_eq!(dodecahedra.cells(4.0, 1000).len(), 120);
    }

    #[test]
    fn neighbours_share_a_face_and_flat_space_is_the_cubic_lattice() {
        let flat = Tiling::<E3>::regular(Cell::Cube, 4).unwrap();
        // Cells within 2.1 of the origin, centres two apart: the 1 + 6 + 12 lattice points
        // within √4 and √8 ≥ 2.1… only the face neighbours and the origin.
        assert_eq!(flat.cells(2.1, 1000).len(), 7);
        assert_eq!(flat.cells(2.9, 1000).len(), 19);
        // Right-angled dodecahedra: each neighbour's centre twice the inradius away.
        let tiling = Tiling::<H3>::regular(Cell::Dodecahedron, 4).unwrap();
        let origin = H3::origin();
        for step in tiling.neighbours() {
            let d = H3::distance(&origin, &H3::apply(step, &origin));
            assert!((d - 2.0 * tiling.inradius()).abs() < 1e-9);
        }
        // Four round each edge: the cells within a few cells' reach grow as space does.
        let near = tiling.cells(2.0 * tiling.inradius() + 1e-6, 10_000).len();
        assert_eq!(near, 13, "the cell and its twelve neighbours");
    }

    /// How many of the cell's vertices each neighbour shares with it.
    fn shared<G: Geometry>(tiling: &Tiling<G>) -> Vec<usize> {
        let vertices: Vec<_> = tiling
            .cell
            .vertices()
            .iter()
            .map(|v| tiling.point(&(v * tiling.projective_inradius)))
            .collect();
        tiling
            .neighbours()
            .iter()
            .map(|step| {
                vertices
                    .iter()
                    .filter(|v| {
                        vertices
                            .iter()
                            .any(|w| G::distance(&G::apply(step, w), v) < 1e-9)
                    })
                    .count()
            })
            .collect()
    }

    #[test]
    fn a_neighbour_is_the_cell_mirrored_in_their_common_face() {
        // Across every face, the neighbour's vertices meet the cell's on that face, all of it.
        let pentagons = |s: Vec<usize>| s.iter().all(|&n| n == 5) && s.len() == 12;
        let squares = |s: Vec<usize>| s.iter().all(|&n| n == 4) && s.len() == 6;
        assert!(pentagons(shared(
            &Tiling::<H3>::regular(Cell::Dodecahedron, 4).unwrap()
        )));
        assert!(pentagons(shared(
            &Tiling::<S3>::regular(Cell::Dodecahedron, 3).unwrap()
        )));
        assert!(squares(shared(
            &Tiling::<H3>::regular(Cell::Cube, 5).unwrap()
        )));
        assert!(squares(shared(
            &Tiling::<S3>::regular(Cell::Cube, 3).unwrap()
        )));
    }

    /// Whether no element of `quotient` with a word of at most `length` letters but the
    /// identity fixes the domain's centre or one of the cell's vertices: the group acts freely
    /// there, as a manifold's does (a bad gluing turns about an edge or a vertex).
    fn free<G: Geometry>(tiling: &Tiling<G>, quotient: &Quotient<G>, length: usize) -> bool {
        let centre = *quotient.centre();
        let mut points: Vec<G::Point> = tiling
            .cell
            .vertices()
            .iter()
            .map(|v| tiling.point(&(v * tiling.projective_inradius)))
            .collect();
        points.push(centre);
        let identity = |g: &G::Isometry| {
            G::distance(&G::apply(g, &centre), &centre) < 1e-6
                && (0..G::DIM).all(|i| {
                    let v = G::origin_frame(i);
                    G::norm(&centre, &(G::apply_tangent(g, &v) - v)) < 1e-6
                })
        };
        let mut words = vec![G::Isometry::identity()];
        for _ in 0..length {
            words = words
                .iter()
                .flat_map(|g| {
                    quotient
                        .faces()
                        .map(move |face| g.compose(&quotient.element(face)))
                        .collect::<Vec<_>>()
                })
                .collect();
            let fixes = |g: &G::Isometry| {
                points
                    .iter()
                    .any(|p| G::distance(&G::apply(g, p), p) < 1e-6)
            };
            if words.iter().any(|g| fixes(g) && !identity(g)) {
                return false;
            }
        }
        true
    }

    #[test]
    fn three_tenths_of_a_turn_glue_the_seifert_weber_space_and_one_or_five_do_not() {
        let tenth = std::f64::consts::TAU / 10.0;
        let tiling = Tiling::<H3>::regular(Cell::Dodecahedron, 5).unwrap();
        let weber = tiling.glued(3.0 * tenth);
        assert_eq!(weber.generators().len(), 6);
        assert!(
            free(&tiling, &weber, 3),
            "the Seifert–Weber space is a manifold"
        );
        // Opposite faces meet turned by an odd number of tenths (an even number would leave the
        // neighbour off the tiling); a tenth and a half turn leave edges or vertices fixed.
        assert!(!free(&tiling, &tiling.glued(tenth), 3));
        assert!(!free(&tiling, &tiling.glued(5.0 * tenth), 3));
        // The Poincaré homology sphere: a tenth of a turn on the 120-cell's dodecahedron. Its
        // group has 120 elements, one for each cell of the 120-cell.
        let cells = Tiling::<S3>::regular(Cell::Dodecahedron, 3).unwrap();
        let poincare = cells.glued(tenth);
        assert!(free(&cells, &poincare, 3));
        assert_eq!(poincare.translates(12).len(), 120);
    }

    #[test]
    fn walking_through_a_face_of_the_seifert_weber_space_changes_nothing_seen() {
        // The eye just past a face, then folded back across it: what it sees of the domain's
        // copies (their centres relative to it, within 3) is the same before and
        // after. That is a seamless walk.
        let tiling = Tiling::<H3>::regular(Cell::Dodecahedron, 5).unwrap();
        let weber = tiling.glued(3.0 * std::f64::consts::TAU / 10.0);
        let view = fk_render::QuotientView::new(weber.clone(), 3.5);
        let normal = Cell::Dodecahedron.normals()[0];
        let past = <H3 as Geometry>::Isometry::exp(&H3::transvection(
            &(tangent::<H3>(&normal) * (tiling.inradius() + 0.05)),
        ));
        let seen = |eye: &<H3 as Geometry>::Isometry| -> Vec<_> {
            let to_eye = eye.inverse();
            std::iter::once(<H3 as Geometry>::Isometry::identity())
                .chain(view.translates().iter().copied())
                .map(|g| H3::apply(&to_eye.compose(&g), &H3::origin()))
                .filter(|p| H3::distance(&H3::origin(), p) < 3.0)
                .collect()
        };
        let before = seen(&past);
        let mut folded = past;
        let crossed = weber.reduce(&mut folded, &mut fk_quotient::ChartTag::identity());
        assert_eq!(crossed.len(), 1, "one face crossed");
        let after = seen(&folded);
        assert_eq!(before.len(), after.len());
        assert!(before.len() > 10);
        for p in &before {
            assert!(
                after.iter().any(|q| H3::distance(p, q) < 1e-6),
                "a copy moved"
            );
        }
    }

    #[test]
    fn a_frame_has_a_beam_per_edge() {
        let tiling = Tiling::<H3>::regular(Cell::Dodecahedron, 4).unwrap();
        let frame = tiling.frame(0.05, 0.05, 0.99);
        // Thirty edges, four sides of two triangles each.
        assert_eq!(frame.indices().len(), 30 * 4 * 2 * 3);
        let cube = Tiling::<S3>::regular(Cell::Cube, 3)
            .unwrap()
            .frame(0.05, 0.05, 0.99);
        assert_eq!(cube.indices().len(), 12 * 4 * 2 * 3);
    }
}
