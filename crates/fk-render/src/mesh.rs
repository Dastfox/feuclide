//! Meshes: authored in the flat tangent space at the origin, embedded into the geometry.

use bevy_ecs::component::Component;
use bevy_ecs::resource::Resource;
use bytemuck::{Pod, Zeroable};
use fk_geometry::{Geometry, GpuGeometry, VectorSpace};
use fk_math::Real;
use fk_math::nalgebra::Vector3;
use fk_scene::Pose;

use crate::gpu::Vec4f;
use crate::{Color, NormalMap};

/// One vertex as the GPU reads it: the embedded point and its normal, a tangent vector there.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct Vertex {
    /// [`GpuGeometry::embed_point`] of the vertex.
    pub position: Vec4f,
    /// [`GpuGeometry::embed_tangent`] of the unit normal.
    pub normal: Vec4f,
    /// The texture coordinates (x, y), the [`NormalMap`](crate::NormalMap)'s layer (z, 0 for
    /// none) and the handedness of the tangent frame (w, ±1: the sign of `v` along
    /// `normal × tangent`).
    pub uv: Vec4f,
    /// [`GpuGeometry::embed_tangent`] of the unit tangent along `u`, at right angles to the
    /// normal; zero without a normal map.
    pub tangent: Vec4f,
}

/// A triangle mesh embedded in a geometry, ready to upload.
#[derive(Clone, Debug)]
pub struct Mesh {
    /// The vertices.
    pub vertices: Vec<Vertex>,
    /// Three indices per triangle, counter-clockwise seen from the side the normals point to.
    pub indices: Vec<u32>,
    /// Geodesic radius of a ball about the origin that holds every vertex, for culling.
    pub radius: Real,
}

/// Builds a [`Mesh`] in the flat tangent space at the origin, in coordinates of the reference
/// frame (`origin_frame(0..3)`), then embeds it into a geometry.
///
/// Embedding follows each vertex's geodesic from the origin (`exp(o, v)`) and carries its normal
/// along the same geodesic, so in E³ the mesh is unchanged and in curved space it bends the way
/// space does. Edges are drawn straight between their embedded ends, so in curved space long
/// ones sag: refine them first (`fk_assets::refine`).
#[derive(Clone, Debug, Default)]
pub struct MeshBuilder {
    positions: Vec<Vector3<Real>>,
    normals: Vec<Vector3<Real>>,
    uvs: Vec<[Real; 2]>,
    indices: Vec<u32>,
    normal_map: Option<NormalMap>,
}

impl MeshBuilder {
    /// An empty mesh.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a vertex; the normal is normalized. Returns its index.
    pub fn vertex(&mut self, position: Vector3<Real>, normal: Vector3<Real>) -> u32 {
        self.vertex_uv(position, normal, [0.0; 2])
    }

    /// Adds a vertex with texture coordinates; the normal is normalized. Returns its index.
    pub fn vertex_uv(
        &mut self,
        position: Vector3<Real>,
        normal: Vector3<Real>,
        uv: [Real; 2],
    ) -> u32 {
        let index = u32::try_from(self.positions.len()).expect("mesh has too many vertices");
        self.positions.push(position);
        self.normals.push(normal.normalize());
        self.uvs.push(uv);
        index
    }

    /// Lays `map` on the mesh by its texture coordinates. Its tangents are made from them when
    /// it is built.
    pub fn normal_map(&mut self, map: NormalMap) -> &mut Self {
        self.normal_map = Some(map);
        self
    }

    /// Adds a triangle, counter-clockwise seen from outside.
    pub fn triangle(&mut self, a: u32, b: u32, c: u32) {
        self.indices.extend([a, b, c]);
    }

    /// Adds the quad `a b c d`, counter-clockwise seen from outside, as two triangles.
    pub fn quad(&mut self, a: u32, b: u32, c: u32, d: u32) {
        self.triangle(a, b, c);
        self.triangle(a, c, d);
    }

    /// Adds a flat quad with one normal: the four corners counter-clockwise around `normal`.
    pub fn flat_quad(&mut self, corners: [Vector3<Real>; 4], normal: Vector3<Real>) {
        let [a, b, c, d] = corners.map(|corner| self.vertex(corner, normal));
        self.quad(a, b, c, d);
    }

    /// The positions so far, in tangent-space coordinates.
    pub fn positions(&self) -> &[Vector3<Real>] {
        &self.positions
    }

    /// The normals so far, in tangent-space coordinates.
    pub fn normals(&self) -> &[Vector3<Real>] {
        &self.normals
    }

    /// The indices so far.
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// The texture coordinates so far.
    pub fn uvs(&self) -> &[[Real; 2]] {
        &self.uvs
    }

    /// Each vertex's unit tangent along `u`, at right angles to its normal, and the handedness
    /// of its frame (the sign of `v` along `normal × tangent`): from the uvs of its triangles,
    /// weighted by their areas. A vertex whose triangles have no uv gradient takes any tangent.
    pub fn tangents(&self) -> Vec<(Vector3<Real>, Real)> {
        let n = self.positions.len();
        let (mut along_u, mut along_v) = (vec![Vector3::zeros(); n], vec![Vector3::zeros(); n]);
        for t in self.indices.as_chunks::<3>().0 {
            let [a, b, c] = t.map(|i| i as usize);
            let (e1, e2) = (
                self.positions[b] - self.positions[a],
                self.positions[c] - self.positions[a],
            );
            let (du1, dv1) = (
                self.uvs[b][0] - self.uvs[a][0],
                self.uvs[b][1] - self.uvs[a][1],
            );
            let (du2, dv2) = (
                self.uvs[c][0] - self.uvs[a][0],
                self.uvs[c][1] - self.uvs[a][1],
            );
            let det = du1 * dv2 - du2 * dv1;
            if det.abs() < 1e-12 {
                continue;
            }
            // Weighted by the triangle's area: the cross product's length over the uv area's.
            let weight = e1.cross(&e2).norm() / det.abs();
            let u = (e1 * dv2 - e2 * dv1) / det;
            let v = (e2 * du1 - e1 * du2) / det;
            for i in [a, b, c] {
                along_u[i] += u * weight;
                along_v[i] += v * weight;
            }
        }
        (0..n)
            .map(|i| {
                let normal = self.normals[i];
                let mut tangent = along_u[i] - normal * normal.dot(&along_u[i]);
                if tangent.norm() < 1e-12 {
                    // Anything at right angles to the normal.
                    let other = if normal.x.abs() < 0.9 {
                        Vector3::x()
                    } else {
                        Vector3::y()
                    };
                    tangent = other - normal * normal.dot(&other);
                }
                let tangent = tangent.normalize();
                let sign = if normal.cross(&tangent).dot(&along_v[i]) < 0.0 {
                    -1.0
                } else {
                    1.0
                };
                (tangent, sign)
            })
            .collect()
    }

    /// Embeds the mesh into the geometry `G`.
    ///
    /// # Panics
    ///
    /// If `G` is not three-dimensional.
    pub fn build<G: GpuGeometry>(&self) -> Mesh {
        assert_eq!(
            G::DIM,
            3,
            "meshes are three-dimensional, {} is not",
            G::NAME
        );
        let origin = G::origin();
        let mut radius: Real = 0.0;
        let tangents = self.normal_map.map(|_| self.tangents());
        let layer = self.normal_map.map_or(0.0, |map| map.layer() as f32);
        let vertices = self
            .positions
            .iter()
            .zip(&self.normals)
            .zip(&self.uvs)
            .enumerate()
            .map(|(i, ((position, normal), uv))| {
                let point = G::exp(&origin, &tangent::<G>(position));
                let carry = |v: &Vector3<Real>| {
                    let v = G::parallel_transport(&origin, &point, &tangent::<G>(v));
                    Vec4f::from_real(&G::embed_tangent(&point, &v))
                };
                radius = radius.max(G::distance(&origin, &point));
                let (along, sign) = tangents
                    .as_ref()
                    .map_or((None, 1.0), |tangents| (Some(tangents[i].0), tangents[i].1));
                Vertex {
                    position: Vec4f::from_real(&G::embed_point(&point)),
                    normal: carry(normal),
                    uv: Vec4f([uv[0] as f32, uv[1] as f32, layer, sign as f32]),
                    tangent: along.map_or(Vec4f::default(), |along| carry(&along)),
                }
            })
            .collect();
        Mesh {
            vertices,
            indices: self.indices.clone(),
            radius,
        }
    }
}

/// The tangent vector at the origin with coordinates `c` in the reference frame.
pub fn tangent<G: Geometry>(c: &Vector3<Real>) -> G::Tangent {
    (0..3).fold(G::Tangent::zero(), |sum, i| sum + G::origin_frame(i) * c[i])
}

/// Which mesh to draw, by its index in [`Meshes`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MeshHandle(pub(crate) u32);

/// Every mesh the renderer can draw, as a resource.
///
/// Meshes are uploaded the first frame after they are added and cannot be changed afterwards;
/// add a new one instead.
#[derive(Resource, Debug, Default)]
pub struct Meshes(Vec<Mesh>);

impl Meshes {
    /// Adds a mesh.
    pub fn add(&mut self, mesh: Mesh) -> MeshHandle {
        let index = u32::try_from(self.0.len()).expect("too many meshes");
        self.0.push(mesh);
        MeshHandle(index)
    }

    /// A mesh by handle.
    pub fn get(&self, handle: MeshHandle) -> &Mesh {
        &self.0[handle.0 as usize]
    }

    /// Every mesh, in the order they were added.
    pub fn all(&self) -> &[Mesh] {
        &self.0
    }
}

/// One copy of a mesh: where it is relative to its entity, and what the shader gets with it.
#[derive(Debug)]
pub struct Instance<G: Geometry> {
    /// Pose relative to the entity's [`Pose`].
    pub pose: G::Isometry,
    /// Base colour of the painted material.
    pub color: Color,
    /// Free for the game's shaders; the painted material ignores it.
    pub data: [f32; 4],
}

impl<G: Geometry> Instance<G> {
    /// A copy at `pose` in `color`.
    pub fn new(pose: G::Isometry, color: Color) -> Self {
        Self {
            pose,
            color,
            data: [0.0; 4],
        }
    }
}

impl<G: Geometry> Clone for Instance<G> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<G: Geometry> Copy for Instance<G> {}

/// Draws a mesh once per instance, each relative to the entity's [`Pose`].
///
/// One entity with thousands of instances is much cheaper than thousands of entities: the
/// renderer culls each instance and draws all the visible ones in one call.
#[derive(Component, Debug)]
#[require(Pose<G>)]
pub struct MeshInstances<G: Geometry> {
    /// What to draw.
    pub mesh: MeshHandle,
    /// Where, and in which colours.
    pub instances: Vec<Instance<G>>,
}

impl<G: GpuGeometry> MeshInstances<G> {
    /// One copy of `mesh` at the entity's own pose.
    pub fn single(mesh: MeshHandle, color: Color) -> Self {
        Self {
            mesh,
            instances: vec![Instance::new(fk_geometry::GroupElement::identity(), color)],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit square facing +z, `u` along x and `v` along y, or `v` turned over.
    fn square(turned: bool) -> MeshBuilder {
        let mut mesh = MeshBuilder::new();
        let v = |y: Real| if turned { 1.0 - y } else { y };
        let corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let [a, b, c, d] =
            corners.map(|(x, y)| mesh.vertex_uv(Vector3::new(x, y, 0.0), Vector3::z(), [x, v(y)]));
        mesh.quad(a, b, c, d);
        mesh
    }

    #[test]
    fn tangents_follow_u_and_the_frame_its_handedness() {
        for (tangent, sign) in square(false).tangents() {
            assert!((tangent - Vector3::x()).norm() < 1e-12);
            assert_eq!(sign, 1.0);
        }
        for (tangent, sign) in square(true).tangents() {
            assert!((tangent - Vector3::x()).norm() < 1e-12);
            assert_eq!(sign, -1.0, "v turned over");
        }
    }

    #[test]
    fn a_mapped_mesh_carries_its_layer_and_tangents_to_the_gpu() {
        let mut maps = NormalMapsForTest::new();
        let mut mesh = square(false);
        mesh.normal_map(maps.add());
        let built = mesh.build::<fk_geometry_euclidean::E3>();
        for vertex in &built.vertices {
            assert_eq!(vertex.uv.0[2], 1.0, "the first map is layer 1");
            assert_eq!(vertex.tangent.0, [1.0, 0.0, 0.0, 0.0]);
        }
        let plain = square(false).build::<fk_geometry_euclidean::E3>();
        assert!(
            plain
                .vertices
                .iter()
                .all(|v| v.uv.0[2] == 0.0 && v.tangent.0 == [0.0; 4])
        );
    }

    struct NormalMapsForTest(crate::NormalMaps);

    impl NormalMapsForTest {
        fn new() -> Self {
            Self(crate::NormalMaps::new(2))
        }

        fn add(&mut self) -> NormalMap {
            self.0.add_heights(1.0, |u, _| u)
        }
    }
}
