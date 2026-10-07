//! Mesh loading, embedding of tangent-space meshes into a geometry, and geodesic refinement.
//!
//! A mesh is authored flat, in the tangent space at the origin, in coordinates of the reference
//! frame (`origin_frame(0..3)`: x, y up, z), metres: a [`TangentMesh`]. It comes from a binary
//! glTF ([`gltf::load`]: positions, normals and the first uv set of every triangle primitive,
//! node transforms and a scale baked in), and is embedded into a geometry by following each
//! vertex's geodesic from the origin, `exp(o, v)` ([`TangentMesh::embed`], through
//! `fk_render::MeshBuilder`).
//!
//! Embedding moves only the vertices: an edge is drawn straight between its embedded ends, so
//! in curved space a long one is a chord, not the image of the flat edge, and the surface sags.
//! [`refine`] splits the flat mesh until no edge is longer than a bound measured between the
//! embedded ends (geodesic distance), keeping it conforming (no T-junctions). In E³ the
//! geodesic is the edge itself, and refining only makes a mesh fine enough to be displaced
//! (a breathing ground). `tools/` refines a file offline: `cargo run -p fk-tools --bin refine`.
//!
//! [`tiling`] makes the regular tilings of space by cubes and dodecahedra, in whichever
//! geometry has them.

pub mod gltf;
pub mod refine;
pub mod tiling;

pub use refine::refine;
pub use tiling::{Cell, Tiling};

use fk_geometry::{Geometry, GpuGeometry};
use fk_math::Real;
use fk_math::nalgebra::Vector3;
use fk_render::{Mesh, MeshBuilder};

/// A triangle mesh in the flat tangent space at the origin, in coordinates of the reference
/// frame, before it is embedded.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TangentMesh {
    /// Each vertex's place.
    pub positions: Vec<Vector3<Real>>,
    /// Each vertex's unit normal.
    pub normals: Vec<Vector3<Real>>,
    /// Each vertex's texture coordinates (zeros if the file had none).
    pub uvs: Vec<[Real; 2]>,
    /// Three indices per triangle, counter-clockwise seen from the side the normals point to.
    pub indices: Vec<u32>,
}

impl TangentMesh {
    /// How many triangles.
    pub fn triangles(&self) -> usize {
        self.indices.len() / 3
    }

    /// Every vertex's place times `scale`: a scale baked into the tangent space, before `exp`.
    pub fn scaled(mut self, scale: Real) -> Self {
        for p in &mut self.positions {
            *p *= scale;
        }
        self
    }

    /// The longest edge, as the geodesic distance in `G` between its embedded ends.
    pub fn longest_edge<G: Geometry>(&self) -> Real {
        let points = self.embedded_points::<G>();
        self.indices
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|t| [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])])
            .map(|(a, b)| G::distance(&points[a as usize], &points[b as usize]))
            .fold(0.0, Real::max)
    }

    /// Each vertex followed along its geodesic from the origin, `exp(o, v)`.
    pub fn embedded_points<G: Geometry>(&self) -> Vec<G::Point> {
        let origin = G::origin();
        self.positions
            .iter()
            .map(|p| G::exp(&origin, &fk_render::tangent::<G>(p)))
            .collect()
    }

    /// The mesh as a `MeshBuilder`, uvs and all (lay a normal map on it with
    /// `MeshBuilder::normal_map`).
    pub fn builder(&self) -> MeshBuilder {
        let mut builder = MeshBuilder::new();
        for ((p, n), uv) in self.positions.iter().zip(&self.normals).zip(&self.uvs) {
            builder.vertex_uv(*p, *n, *uv);
        }
        for t in self.indices.as_chunks::<3>().0 {
            builder.triangle(t[0], t[1], t[2]);
        }
        builder
    }

    /// Embedded into `G`, ready to upload: each vertex at `exp(o, v)`, its normal carried along
    /// the same geodesic.
    pub fn embed<G: GpuGeometry>(&self) -> Mesh {
        self.builder().build::<G>()
    }
}
