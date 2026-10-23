//! Static scenery for flat worlds: meshes merged per material into one ([`Kit`]), boxes as
//! meshes and as distance fields ([`Block`], [`Blocks`]), shapes united ([`Union`]), and poses
//! on the ground by a yaw.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use fk_geometry::GpuGeometry;
use fk_geometry::{Geometry, GroupElement};
use fk_geometry_euclidean::{E3, Se3 as Isometry};
use fk_math::Real;
use fk_math::nalgebra::{Matrix3, Point3, UnitQuaternion, Vector3};
use fk_math::noise::lattice_hash;
use fk_math::sdf::cuboid;
use fk_physics::DistanceField;
use fk_render::{Mesh, MeshBuilder, NormalMap, axis, box_uv, shapes};

/// A translation.
pub fn translation(v: Vector3<Real>) -> Isometry {
    Isometry::exp(&E3::transvection(&v))
}

/// The turn about the vertical facing `yaw` radians (0 along −z, positive to the right).
pub fn facing(yaw: Real) -> UnitQuaternion<Real> {
    UnitQuaternion::from_axis_angle(&Vector3::y_axis(), -yaw)
}

/// Standing at `at`, facing `yaw`.
pub fn placed(at: Vector3<Real>, yaw: Real) -> Isometry {
    let mut pose = translation(at);
    pose.rotation = facing(yaw);
    pose
}

/// Static meshes merged into one: parts placed where they stand in the world's frame, so it
/// is drawn in one call (a static batch). Each part keeps the material `M` it was added as, or
/// wears the kit's ([`Kit::lay`]); the materials' normal maps and texture coordinates are laid
/// when it is drawn ([`Kit::surfaced`]).
#[derive(Clone, Debug)]
pub struct Kit<M> {
    mesh: MeshBuilder,
    /// Each vertex's material, where its part was added as one.
    surfaces: Vec<Option<M>>,
    /// Each vertex's part, counted from 0 in the order they were added.
    parts: Vec<u32>,
    /// What the vertices without one wear.
    surface: Option<M>,
    /// Whether they wear nothing, not even the default: water, glass.
    bare: bool,
}

impl<M> Default for Kit<M> {
    fn default() -> Self {
        Self {
            mesh: MeshBuilder::new(),
            surfaces: Vec::new(),
            parts: Vec::new(),
            surface: None,
            bare: false,
        }
    }
}

impl<M: Copy> Kit<M> {
    /// A kit of `mesh`, already where it stands.
    pub fn from_mesh(mesh: MeshBuilder) -> Self {
        Self {
            surfaces: vec![None; mesh.positions().len()],
            parts: vec![0; mesh.positions().len()],
            mesh,
            surface: None,
            bare: false,
        }
    }

    /// Lays the material `surface` on every part added without one of its own.
    pub fn lay(&mut self, surface: M) -> &mut Self {
        self.surface = Some(surface);
        self.bare = false;
        self
    }

    /// The kit with [`lay`](Self::lay)`(surface)`.
    pub fn laid(mut self, surface: M) -> Self {
        self.lay(surface);
        self
    }

    /// Lays nothing on the parts added without a material of their own, not even the default:
    /// what is smooth, water and glass.
    pub fn lay_nothing(&mut self) -> &mut Self {
        self.surface = None;
        self.bare = true;
        self
    }

    /// The kit with [`lay_nothing`](Self::lay_nothing).
    pub fn bare(mut self) -> Self {
        self.lay_nothing();
        self
    }

    /// Adds `part`, authored about its own origin, placed at `at`.
    pub fn add(&mut self, part: &MeshBuilder, at: &Isometry) {
        self.add_with(part, at, None);
    }

    /// Adds `part` placed at `at`, made of the material `surface` whatever the kit wears.
    pub fn add_as(&mut self, part: &MeshBuilder, at: &Isometry, surface: M) {
        self.add_with(part, at, Some(surface));
    }

    fn add_with(&mut self, part: &MeshBuilder, at: &Isometry, surface: Option<M>) {
        let start = self.mesh.positions().len() as u32;
        let number = self.parts.last().map_or(0, |last| last + 1);
        for ((p, n), shade) in part
            .positions()
            .iter()
            .zip(part.normals())
            .zip(part.shades())
        {
            let i = self
                .mesh
                .vertex(at.rotation * p + at.translation, at.rotation * n);
            self.mesh.shade(i, *shade);
            self.surfaces.push(surface);
            self.parts.push(number);
        }
        let indices = part.indices();
        for t in indices.as_chunks::<3>().0 {
            self.mesh.triangle(start + t[0], start + t[1], start + t[2]);
        }
    }

    /// Adds a box.
    pub fn block(&mut self, block: &Block) {
        self.add(&shapes::cuboid(block.half), &block.pose());
    }

    /// Whether nothing was added.
    pub fn is_empty(&self) -> bool {
        self.mesh.positions().is_empty()
    }

    /// The merged mesh, without its surfaces.
    pub fn mesh(&self) -> &MeshBuilder {
        &self.mesh
    }

    /// Each vertex's part, counted from 0 in the order they were added.
    pub fn parts(&self) -> &[u32] {
        &self.parts
    }

    /// The material `vertex` wears, its part's or else the kit's: `None` for nothing, or for
    /// whatever the default is when it is drawn ([`surfaced`](Self::surfaced)) unless the kit is
    /// bare.
    pub fn wears(&self, vertex: usize) -> Option<M> {
        self.surfaces[vertex].or(self.surface)
    }

    /// Whether the parts without a material of their own wear nothing ([`lay_nothing`](Self::lay_nothing)).
    pub fn is_bare(&self) -> bool {
        self.bare
    }

    /// The kit with each vertex's shade ([`MeshBuilder::shade`]) multiplied by `shade(vertex)`.
    pub fn shaded(&self, shade: impl Fn(usize) -> [Real; 3]) -> Self {
        let mut kit = self.clone();
        for (i, old) in self.mesh.shades().iter().enumerate() {
            let by = shade(i);
            kit.mesh.shade(i as u32, [0, 1, 2].map(|k| old[k] * by[k]));
        }
        kit
    }

    /// The kit with each vertex moved by `moved(position, normal)`, keeping its surface.
    pub fn moved(
        &self,
        moved: impl Fn(&Vector3<Real>, &Vector3<Real>) -> [Vector3<Real>; 2],
    ) -> Self {
        let mut mesh = MeshBuilder::new();
        let m = &self.mesh;
        for ((p, n), shade) in m.positions().iter().zip(m.normals()).zip(m.shades()) {
            let [p, n] = moved(p, n);
            let i = mesh.vertex(p, n);
            mesh.shade(i, *shade);
        }
        for t in self.mesh.indices().as_chunks::<3>().0 {
            mesh.triangle(t[0], t[1], t[2]);
        }
        Self {
            mesh,
            surfaces: self.surfaces.clone(),
            parts: self.parts.clone(),
            surface: self.surface,
            bare: self.bare,
        }
    }

    /// The merged mesh with its materials laid: each vertex's map (`map_of` its material: the
    /// map and how many units one tile of it spans, or none), and texture coordinates projected
    /// along the axis its triangle faces most ([`box_uv`]), in tiles of its material. The
    /// vertices with no material wear `default`, unless the kit is bare. A vertex shared by
    /// triangles facing different axes is split.
    pub fn surfaced(
        &self,
        map_of: impl Fn(M) -> Option<(NormalMap, Real)>,
        default: Option<M>,
    ) -> MeshBuilder {
        self.surfaced_grained(map_of, |_| false, default)
    }

    /// [`surfaced`](Self::surfaced), the materials `grained` says run along their piece laid
    /// along it, as wood is: on every face holding the piece's long way (its vertices' widest
    /// spread) v runs along it and u across, and each piece takes its own part of the tile, as
    /// boards cut apart; on its ends, as [`box_uv`] lays them.
    pub fn surfaced_grained(
        &self,
        map_of: impl Fn(M) -> Option<(NormalMap, Real)>,
        grained: impl Fn(M) -> bool,
        default: Option<M>,
    ) -> MeshBuilder {
        let fallback = if self.bare {
            None
        } else {
            self.surface.or(default)
        };
        let laid = |i: usize| self.surfaces[i].or(fallback).and_then(&map_of);
        if (0..self.surfaces.len()).all(|i| laid(i).is_none()) {
            return self.mesh.clone();
        }
        let (positions, normals) = (self.mesh.positions(), self.mesh.normals());
        let along = self.long_ways(|i| self.surfaces[i].or(fallback).is_some_and(&grained));
        let mut out = MeshBuilder::new();
        let mut split: HashMap<(u32, usize), u32> = HashMap::new();
        for t in self.mesh.indices().as_chunks::<3>().0 {
            let [a, b, c] = t.map(|i| positions[i as usize]);
            let face = (b - a).cross(&(c - a));
            let face = if face.norm() > 1e-12 {
                face
            } else {
                normals[t[0] as usize]
            };
            let [i, j, k] = t.map(|i| {
                let map = laid(i as usize);
                // Unmapped vertices need no texture coordinates, so are never split.
                let axis = map.map_or(3, |_| axis(&face));
                *split.entry((i, axis)).or_insert_with(|| {
                    let (p, n) = (positions[i as usize], normals[i as usize]);
                    let at = match map {
                        Some((map, metres)) => {
                            let part = self.parts[i as usize];
                            let uv = match along.get(&part) {
                                Some(long) => grain_uv(&p, &face, long, part, metres),
                                None => box_uv(&p, &face).map(|x| x / metres),
                            };
                            out.vertex_mapped(p, n, uv, Some(map))
                        }
                        None => out.vertex(p, n),
                    };
                    out.shade(at, self.mesh.shades()[i as usize]);
                    at
                })
            });
            out.triangle(i, j, k);
        }
        out
    }

    /// The long way of each part whose vertices are `grained`: the axis its vertices spread
    /// along most.
    fn long_ways(&self, grained: impl Fn(usize) -> bool) -> HashMap<u32, Vector3<Real>> {
        let mut sums: HashMap<u32, (Real, Vector3<Real>, Matrix3<Real>)> = HashMap::new();
        for (i, p) in self.mesh.positions().iter().enumerate() {
            if grained(i) {
                let sum =
                    sums.entry(self.parts[i])
                        .or_insert((0.0, Vector3::zeros(), Matrix3::zeros()));
                sum.0 += 1.0;
                sum.1 += p;
                sum.2 += p * p.transpose();
            }
        }
        sums.into_iter()
            .map(|(part, (count, sum, outer))| {
                let mean = sum / count;
                let spread = (outer / count - mean * mean.transpose()).symmetric_eigen();
                let widest = spread.eigenvalues.imax();
                (part, spread.eigenvectors.column(widest).normalize())
            })
            .collect()
    }
}

/// Texture coordinates of `p` on a face of normal `face`, in tiles `metres` across, laid along a
/// piece's `long` way (v along it, u across) where the face holds it and as [`box_uv`] on its
/// ends; shifted by the piece's own share of a tile, the same for every vertex of `part`.
fn grain_uv(
    p: &Vector3<Real>,
    face: &Vector3<Real>,
    long: &Vector3<Real>,
    part: u32,
    metres: Real,
) -> [Real; 2] {
    // The face's main axis stands in for it, so the vertices a face shares with the bevels
    // round it on that axis take one uv.
    let a = axis(face);
    let main = Vector3::ith(a, 1.0);
    let lying = long - main * long[a];
    let [u, v] = if lying.norm() > 0.5 {
        let v = lying.normalize();
        [p.dot(&main.cross(&v)), p.dot(&v)]
    } else {
        box_uv(p, face)
    };
    let shift = |k: i64| lattice_hash(i64::from(part), k, 0x6a1e);
    [u / metres + shift(0), v / metres + shift(1)]
}

/// A box standing on the ground: its middle, its half-extents and its turn about the vertical
/// (radians, 0 along −z, positive to the right).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Block {
    /// Its middle.
    pub centre: Vector3<Real>,
    /// Its half-extents, in its own frame.
    pub half: Vector3<Real>,
    /// Its turn about the vertical.
    pub yaw: Real,
}

impl Block {
    /// A box about `centre`, not turned.
    pub fn new(centre: Vector3<Real>, half: Vector3<Real>) -> Self {
        Self {
            centre,
            half,
            yaw: 0.0,
        }
    }

    /// The box from `a` to `b`, opposite corners, not turned.
    pub fn spanning(a: Vector3<Real>, b: Vector3<Real>) -> Self {
        Self::new((a + b) / 2.0, (b - a).abs() / 2.0)
    }

    /// The same box turned `yaw` about its middle.
    pub fn turned(self, yaw: Real) -> Self {
        Self { yaw, ..self }
    }

    /// Its pose: its middle, turned.
    pub fn pose(&self) -> Isometry {
        placed(self.centre, self.yaw)
    }

    /// Signed distance from `p`.
    pub fn distance(&self, p: &Vector3<Real>) -> Real {
        let local = facing(self.yaw).inverse() * (p - self.centre);
        cuboid(&local, &self.half)
    }
}

/// Boxes, as one distance field.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Blocks(pub Vec<Block>);

impl DistanceField<E3> for Blocks {
    fn distance(&self, p: &Point3<Real>) -> Real {
        self.0
            .iter()
            .map(|block| block.distance(&p.coords))
            .fold(Real::INFINITY, Real::min)
    }
}

/// A shape given by its distance, shared.
pub type Shape = Arc<dyn DistanceField<E3> + Send + Sync>;

/// Several shapes as one: the least of their distances.
#[derive(Clone, Default)]
pub struct Union(pub Vec<Shape>);

impl DistanceField<E3> for Union {
    fn distance(&self, p: &Point3<Real>) -> Real {
        self.0
            .iter()
            .map(|shape| shape.distance(p))
            .fold(Real::INFINITY, Real::min)
    }
}

impl std::fmt::Debug for Union {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Union").field(&self.0.len()).finish()
    }
}

/// Meshes merged to sway in a wind: grass, flowers. Each vertex is given how far a gust pushes
/// it at its strongest; its root stays put with 0. The push rides to a vertex deform in the
/// fourth coordinate of the vertex's normal, which a tangent's embedding in E³ leaves at 0 (the
/// deform puts it back), and `fk::displace`'s `wind_gust` and `swayed_by` bend it there.
#[derive(Default)]
pub struct SwayKit {
    mesh: MeshBuilder,
    sway: Vec<f32>,
}

impl SwayKit {
    /// A vertex at `position` with `normal`, pushed `sway` metres by a gust.
    pub fn vertex(&mut self, position: Vector3<Real>, normal: Vector3<Real>, sway: Real) -> u32 {
        self.sway.push(sway.max(0.0) as f32);
        self.mesh.vertex(position, normal)
    }

    /// A triangle, counter-clockwise seen from the side it shows.
    pub fn triangle(&mut self, a: u32, b: u32, c: u32) {
        self.mesh.triangle(a, b, c);
    }

    /// Adds `part`, authored about its own origin, placed at `at`, each of its vertices pushed
    /// `sway(p)` metres, `p` where it is in `part`.
    pub fn add(
        &mut self,
        part: &MeshBuilder,
        at: &Isometry,
        sway: impl Fn(&Vector3<Real>) -> Real,
    ) {
        let start = self.mesh.positions().len() as u32;
        for (p, n) in part.positions().iter().zip(part.normals()) {
            self.vertex(at.rotation * p + at.translation, at.rotation * n, sway(p));
        }
        for t in part.indices().as_chunks::<3>().0 {
            self.mesh.triangle(start + t[0], start + t[1], start + t[2]);
        }
    }

    /// Whether nothing was added.
    pub fn is_empty(&self) -> bool {
        self.sway.is_empty()
    }

    /// The mesh for `G`, each vertex's push in its normal's `w`.
    pub fn build<G: GpuGeometry>(&self) -> Mesh {
        let mut mesh = self.mesh.build::<G>();
        for (vertex, &sway) in mesh.vertices.iter_mut().zip(&self.sway) {
            vertex.normal.0[3] = sway;
        }
        mesh
    }
}

/// Kits (any `K`: a [`Kit`], a [`SwayKit`]) bucketed in square chunks of the ground `cell`
/// across, so that a chunk out of sight is culled whole: `kinds` kits a chunk, one per colour
/// say, spawned at once by the app ([`kits`](Self::kits)).
pub struct Chunks<K> {
    cell: Real,
    kinds: usize,
    chunks: BTreeMap<(i64, i64), Vec<K>>,
}

impl<K: Default> Chunks<K> {
    /// No chunks yet: `kinds` kits each, `cell` metres across.
    pub fn new(cell: Real, kinds: usize) -> Self {
        Self {
            cell: cell.max(1.0),
            kinds,
            chunks: BTreeMap::new(),
        }
    }

    /// The kits of the chunk holding `(x, z)`.
    pub fn at(&mut self, x: Real, z: Real) -> &mut [K] {
        let key = (
            (x / self.cell).floor() as i64,
            (z / self.cell).floor() as i64,
        );
        let kinds = self.kinds;
        self.chunks
            .entry(key)
            .or_insert_with(|| (0..kinds).map(|_| K::default()).collect())
    }

    /// The kits of the chunk holding `at`'s x and z.
    pub fn near(&mut self, at: &Vector3<Real>) -> &mut [K] {
        self.at(at.x, at.z)
    }

    /// Every chunk's kits, `kinds` each.
    pub fn kits(&self) -> impl Iterator<Item = &[K]> {
        self.chunks.values().map(Vec::as_slice)
    }

    /// Every chunk's kits, `kinds` each, given up.
    pub fn into_kits(self) -> impl Iterator<Item = Vec<K>> {
        self.chunks.into_values()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shades_and_parts_go_with_the_vertices() {
        let mut part = shapes::cuboid(Vector3::new(1.0, 1.0, 1.0));
        part.shade(0, [0.5, 0.6, 0.7]);
        let mut kit = Kit::<u8>::default();
        kit.add(&part, &translation(Vector3::x()));
        kit.add_as(&part, &translation(Vector3::y()), 1);
        let n = part.positions().len();
        assert_eq!(kit.parts()[0], 0);
        assert_eq!(kit.parts()[n], 1);
        assert_eq!(kit.mesh().shades()[n], [0.5, 0.6, 0.7]);
        assert_eq!(kit.wears(0), None);
        assert_eq!(kit.wears(n), Some(1));

        let kit = kit.shaded(|i| {
            if kit.parts()[i] == 1 {
                [2.0; 3]
            } else {
                [1.0; 3]
            }
        });
        assert_eq!(kit.mesh().shades()[n], [1.0, 1.2, 1.4]);
        assert_eq!(kit.mesh().shades()[1], [1.0; 3]);
        let moved = kit.moved(|p, n| [p + Vector3::z(), *n]);
        assert_eq!(moved.mesh().shades(), kit.mesh().shades());
        // Laid, each vertex keeps its shade, however its corners are split.
        let mut maps = fk_render::NormalMaps::new(4);
        let map = maps.add(vec![[128, 128, 255, 255]; 16]);
        let laid = kit.surfaced(|_| Some((map, 1.0)), Some(0));
        assert!(laid.positions().len() > kit.mesh().positions().len());
        let m = kit.mesh();
        for (p, shade) in laid.positions().iter().zip(laid.shades()) {
            let kept = m
                .positions()
                .iter()
                .zip(m.shades())
                .any(|(q, s)| q == p && s == shade);
            assert!(kept, "{p}: {shade:?}");
        }
    }

    #[test]
    fn the_grain_runs_along_each_piece() {
        // A board lying along x and one standing along y, both grained; a box that is not.
        let mut kit = Kit::<u8>::default();
        kit.add_as(
            &shapes::cuboid(Vector3::new(1.0, 0.05, 0.1)),
            &translation(Vector3::zeros()),
            1,
        );
        kit.add_as(
            &shapes::cuboid(Vector3::new(0.1, 1.0, 0.05)),
            &translation(Vector3::new(5.0, 0.0, 0.0)),
            1,
        );
        kit.add_as(
            &shapes::cuboid(Vector3::new(1.0, 0.05, 0.1)),
            &translation(Vector3::new(9.0, 0.0, 0.0)),
            2,
        );
        let mut maps = fk_render::NormalMaps::new(4);
        let map = maps.add(vec![[128, 128, 255, 255]; 16]);
        let laid = kit.surfaced_grained(|_| Some((map, 1.0)), |m| m == 1, None);
        let (positions, uvs) = (laid.positions(), laid.uvs());
        // On the faces facing z, along each edge: how far v and u go against how far x and y do.
        let mut fronts = [0; 3];
        for t in laid.indices().as_chunks::<3>().0 {
            let [a, b, c] = t.map(|i| positions[i as usize]);
            if (b - a).cross(&(c - a)).normalize().z.abs() < 0.99 {
                continue;
            }
            let piece = (a.x / 4.0).round() as usize;
            fronts[piece] += 1;
            for (i, j) in [(t[0], t[1]), (t[1], t[2])] {
                let (d, w) = (
                    positions[i as usize] - positions[j as usize],
                    [0, 1].map(|k| uvs[i as usize][k] - uvs[j as usize][k]),
                );
                // v along the piece (x, then y), u across; the box as box projection lays it.
                let [along, across] = if piece == 1 { [d.y, d.x] } else { [d.x, d.y] };
                if piece == 2 {
                    assert!((w[0] - d.x).abs() < 1e-9 && (w[1] - d.y).abs() < 1e-9);
                } else {
                    assert!(
                        (w[1].abs() - along.abs()).abs() < 1e-9,
                        "{piece}: {d} {w:?}"
                    );
                    assert!(
                        (w[0].abs() - across.abs()).abs() < 1e-9,
                        "{piece}: {d} {w:?}"
                    );
                }
            }
        }
        assert!(fronts.iter().all(|&n| n >= 2), "{fronts:?}");
    }
}
