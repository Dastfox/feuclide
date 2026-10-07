//! The glTF subset: binary glTF (`.glb`), every triangle primitive of the default scene's
//! meshes, their positions, normals and first uv set, node transforms baked in. Materials,
//! textures, skins and animations are ignored; buffers outside the file are not read.
//!
//! glTF is right-handed, y up, in metres: the reference frame's own coordinates (x, y up, z),
//! so a file's coordinates are the tangent space's as they are. Normals missing from a
//! primitive are made from its faces, each vertex's the area-weighted mean of its triangles'.

use fk_math::Real;
use fk_math::nalgebra::{Matrix3, Matrix4, Vector3};

use crate::TangentMesh;

/// What went wrong reading a file.
#[derive(Debug)]
pub enum LoadError {
    /// It is not glTF, or not well formed.
    Gltf(::gltf::Error),
    /// A primitive has no positions.
    NoPositions,
    /// A buffer lies outside the file.
    OutsideBuffer,
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gltf(error) => write!(f, "glTF: {error}"),
            Self::NoPositions => write!(f, "a primitive has no positions"),
            Self::OutsideBuffer => write!(f, "a buffer lies outside the file (only .glb is read)"),
        }
    }
}

impl std::error::Error for LoadError {}

impl From<::gltf::Error> for LoadError {
    fn from(error: ::gltf::Error) -> Self {
        Self::Gltf(error)
    }
}

/// The mesh in a binary glTF, every place times `scale` (baked into the tangent space).
pub fn load(bytes: &[u8], scale: Real) -> Result<TangentMesh, LoadError> {
    let file = ::gltf::Gltf::from_slice(bytes)?;
    let blob = file.blob.as_deref();
    let document = &file.document;
    let mut mesh = TangentMesh::default();
    let roots: Vec<_> = match document
        .default_scene()
        .or_else(|| document.scenes().next())
    {
        Some(scene) => scene.nodes().collect(),
        None => document.nodes().collect(),
    };
    let mut stack: Vec<_> = roots
        .into_iter()
        .map(|node| (node, Matrix4::identity()))
        .collect();
    while let Some((node, parent)) = stack.pop() {
        let local = Matrix4::from(node.transform().matrix()).cast::<Real>();
        let transform = parent * local;
        if let Some(gltf_mesh) = node.mesh() {
            for primitive in gltf_mesh.primitives() {
                if primitive.mode() != ::gltf::mesh::Mode::Triangles {
                    continue;
                }
                add_primitive(&mut mesh, &primitive, blob, &transform)?;
            }
        }
        stack.extend(node.children().map(|child| (child, transform)));
    }
    Ok(mesh.scaled(scale))
}

fn add_primitive(
    mesh: &mut TangentMesh,
    primitive: &::gltf::Primitive<'_>,
    blob: Option<&[u8]>,
    transform: &Matrix4<Real>,
) -> Result<(), LoadError> {
    let outside = primitive
        .get(&::gltf::Semantic::Positions)
        .and_then(|accessor| accessor.view())
        .is_some_and(|view| matches!(view.buffer().source(), ::gltf::buffer::Source::Uri(_)));
    let reader = primitive.reader(|buffer| match buffer.source() {
        ::gltf::buffer::Source::Bin => blob,
        ::gltf::buffer::Source::Uri(_) => None,
    });
    let positions: Vec<Vector3<Real>> = reader
        .read_positions()
        .ok_or(if outside {
            LoadError::OutsideBuffer
        } else {
            LoadError::NoPositions
        })?
        .map(|p| Vector3::from(p).cast::<Real>())
        .collect();
    let count = positions.len();
    let first = u32::try_from(mesh.positions.len()).expect("mesh has too many vertices");
    let indices: Vec<u32> = match reader.read_indices() {
        Some(indices) => indices.into_u32().collect(),
        None => (0..u32::try_from(count).expect("too many vertices")).collect(),
    };
    let normals: Option<Vec<Vector3<Real>>> = reader
        .read_normals()
        .map(|n| n.map(|n| Vector3::from(n).cast::<Real>()).collect());
    let uvs: Vec<[Real; 2]> = reader.read_tex_coords(0).map_or_else(
        || vec![[0.0; 2]; count],
        |uv| {
            uv.into_f32()
                .map(|[u, v]| [Real::from(u), Real::from(v)])
                .collect()
        },
    );
    // Places through the whole transform; normals through the inverse transpose of its linear
    // part; a mirroring transform turns the triangles round to keep them counter-clockwise.
    let linear: Matrix3<Real> = transform.fixed_view::<3, 3>(0, 0).into();
    let normal_matrix = linear
        .try_inverse()
        .map_or(Matrix3::identity(), |inverse| inverse.transpose());
    let mirrored = linear.determinant() < 0.0;
    let positions: Vec<_> = positions
        .iter()
        .map(|p| transform.transform_point(&(*p).into()).coords)
        .collect();
    let normals = normals.unwrap_or_else(|| face_normals(&positions, &indices));
    mesh.positions.extend(&positions);
    mesh.normals.extend(normals.iter().map(|n| {
        let n = normal_matrix * n;
        if n.norm() > 1e-12 { n.normalize() } else { n }
    }));
    mesh.uvs.extend(uvs);
    for t in indices.as_chunks::<3>().0 {
        let t = if mirrored {
            [t[0], t[2], t[1]]
        } else {
            [t[0], t[1], t[2]]
        };
        mesh.indices.extend(t.map(|i| first + i));
    }
    Ok(())
}

/// Each vertex's normal: the area-weighted mean of its triangles' (before any transform).
fn face_normals(positions: &[Vector3<Real>], indices: &[u32]) -> Vec<Vector3<Real>> {
    let mut normals = vec![Vector3::zeros(); positions.len()];
    for t in indices.as_chunks::<3>().0 {
        let [a, b, c] = [0, 1, 2].map(|k| positions[t[k] as usize]);
        let face = (b - a).cross(&(c - a));
        for &i in t {
            normals[i as usize] += face;
        }
    }
    normals
        .into_iter()
        .map(|n| {
            if n.norm() > 1e-12 {
                n.normalize()
            } else {
                Vector3::y()
            }
        })
        .collect()
}

/// `mesh` as a binary glTF: one node, one mesh, one primitive with positions, normals, uvs and
/// indices (single-precision, as glTF keeps them).
pub fn write_glb(mesh: &TangentMesh) -> Vec<u8> {
    write_glb_with_node(mesh, "")
}

/// [`write_glb`] with extra properties on its node (`"scale":[2,2,2]`, …), for tests.
fn write_glb_with_node(mesh: &TangentMesh, node: &str) -> Vec<u8> {
    let mut bin = Vec::new();
    let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
    for p in &mesh.positions {
        for k in 0..3 {
            let x = p[k] as f32;
            lo[k] = lo[k].min(x);
            hi[k] = hi[k].max(x);
            bin.extend(x.to_le_bytes());
        }
    }
    let normals_at = bin.len();
    for n in &mesh.normals {
        for k in 0..3 {
            bin.extend((n[k] as f32).to_le_bytes());
        }
    }
    let uvs_at = bin.len();
    for uv in &mesh.uvs {
        for x in uv {
            bin.extend((*x as f32).to_le_bytes());
        }
    }
    let indices_at = bin.len();
    for i in &mesh.indices {
        bin.extend(i.to_le_bytes());
    }
    let end = bin.len();
    let count = mesh.positions.len();
    let (lo, hi) = if count == 0 {
        ([0.0; 3], [0.0; 3])
    } else {
        (lo, hi)
    };
    let comma = if node.is_empty() { "" } else { "," };
    let json = format!(
        concat!(
            r#"{{"asset":{{"version":"2.0","generator":"fk-assets"}},"scene":0,"#,
            r#""scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0{comma}{node}}}],"#,
            r#""meshes":[{{"primitives":[{{"attributes":{{"POSITION":0,"NORMAL":1,"TEXCOORD_0":2}},"indices":3}}]}}],"#,
            r#""buffers":[{{"byteLength":{end}}}],"#,
            r#""bufferViews":["#,
            r#"{{"buffer":0,"byteOffset":0,"byteLength":{normals_at}}},"#,
            r#"{{"buffer":0,"byteOffset":{normals_at},"byteLength":{normals_len}}},"#,
            r#"{{"buffer":0,"byteOffset":{uvs_at},"byteLength":{uvs_len}}},"#,
            r#"{{"buffer":0,"byteOffset":{indices_at},"byteLength":{indices_len}}}],"#,
            r#""accessors":["#,
            r#"{{"bufferView":0,"componentType":5126,"count":{count},"type":"VEC3","min":{lo:?},"max":{hi:?}}},"#,
            r#"{{"bufferView":1,"componentType":5126,"count":{count},"type":"VEC3"}},"#,
            r#"{{"bufferView":2,"componentType":5126,"count":{count},"type":"VEC2"}},"#,
            r#"{{"bufferView":3,"componentType":5125,"count":{index_count},"type":"SCALAR"}}]}}"#,
        ),
        comma = comma,
        node = node,
        end = end,
        normals_at = normals_at,
        normals_len = uvs_at - normals_at,
        uvs_at = uvs_at,
        uvs_len = indices_at - uvs_at,
        indices_at = indices_at,
        indices_len = end - indices_at,
        count = count,
        index_count = mesh.indices.len(),
        lo = lo,
        hi = hi,
    );
    glb(json.into_bytes(), bin)
}

/// The GLB container: the header, the JSON chunk and the binary chunk, each padded to four
/// bytes (JSON with spaces, binary with zeros).
fn glb(mut json: Vec<u8>, mut bin: Vec<u8>) -> Vec<u8> {
    json.resize(json.len().next_multiple_of(4), b' ');
    bin.resize(bin.len().next_multiple_of(4), 0);
    let total = 12 + 8 + json.len() + 8 + bin.len();
    let mut out = Vec::with_capacity(total);
    let word = |n: usize| u32::try_from(n).expect("glTF is under 4 GiB").to_le_bytes();
    out.extend(b"glTF");
    out.extend(2u32.to_le_bytes());
    out.extend(word(total));
    out.extend(word(json.len()));
    out.extend(b"JSON");
    out.extend(json);
    out.extend(word(bin.len()));
    out.extend(b"BIN\0");
    out.extend(bin);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle() -> TangentMesh {
        TangentMesh {
            positions: vec![
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(0.0, 1.0, 0.0),
            ],
            normals: vec![Vector3::z(); 3],
            uvs: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            indices: vec![0, 1, 2],
        }
    }

    #[test]
    fn a_written_mesh_reads_back_the_same() {
        let mesh = triangle();
        assert_eq!(load(&write_glb(&mesh), 1.0).unwrap(), mesh);
    }

    #[test]
    fn the_scale_and_the_node_transform_are_baked_in() {
        // The node doubles it and moves it 3 m up; the load scales it by ten.
        let bytes = write_glb_with_node(&triangle(), r#""scale":[2,2,2],"translation":[0,3,0]"#);
        let mesh = load(&bytes, 10.0).unwrap();
        assert_eq!(mesh.positions[1], Vector3::new(20.0, 30.0, 0.0));
        assert_eq!(mesh.positions[2], Vector3::new(0.0, 50.0, 0.0));
        assert_eq!(mesh.normals[0], Vector3::z());
    }

    #[test]
    fn a_mirroring_node_keeps_the_triangles_facing_out() {
        let bytes = write_glb_with_node(&triangle(), r#""scale":[-1,1,1]"#);
        let mesh = load(&bytes, 1.0).unwrap();
        assert_eq!(mesh.indices, vec![0, 2, 1]);
        let [a, b, c] = [0, 1, 2].map(|k| mesh.positions[mesh.indices[k] as usize]);
        let face = (b - a).cross(&(c - a)).normalize();
        assert!(
            (face - mesh.normals[0]).norm() < 1e-9,
            "{face} against {}",
            mesh.normals[0]
        );
    }

    #[test]
    fn missing_normals_are_made_from_the_faces() {
        let normals = face_normals(&triangle().positions, &[0, 1, 2]);
        assert!(normals.iter().all(|n| (n - Vector3::z()).norm() < 1e-12));
    }

    #[test]
    fn what_is_not_glb_is_refused() {
        assert!(matches!(load(b"not a model", 1.0), Err(LoadError::Gltf(_))));
    }
}
