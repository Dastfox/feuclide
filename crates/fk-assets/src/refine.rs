//! Adaptive geodesic refinement: splits a flat mesh until no edge, measured between its
//! embedded ends, is longer than a bound.
//!
//! Each round measures every edge as the geodesic distance in `G` between `exp(o, a)` and
//! `exp(o, b)`, marks the long ones and splits each at its middle in the flat tangent space (the
//! authored surface is the flat one; the new vertex is embedded like the others). A triangle is
//! split by how many of its edges are marked: one, in two; two, in three; three, in four. Since
//! an edge is marked once for both triangles that share it, the mesh stays conforming: no
//! vertex ever lies on another triangle's edge. Normals and uvs at a middle are the averages of
//! the ends'. Rounds stop when no edge is long, or after [`MOST_ROUNDS`].

use std::collections::HashMap;

use fk_geometry::Geometry;
use fk_math::Real;

use crate::TangentMesh;

/// Most rounds of splitting: each halves every long edge, so this is far finer than any bound
/// a mesh asks for.
pub const MOST_ROUNDS: usize = 24;

/// `mesh` split until no edge is longer than `max_edge` between its embedded ends in `G`.
///
/// # Panics
///
/// If `max_edge` is not positive.
pub fn refine<G: Geometry>(mesh: &TangentMesh, max_edge: Real) -> TangentMesh {
    assert!(max_edge > 0.0, "the longest edge must be positive");
    let mut mesh = mesh.clone();
    for _ in 0..MOST_ROUNDS {
        if !round::<G>(&mut mesh, max_edge) {
            break;
        }
    }
    mesh
}

/// The key of the edge between `a` and `b`, whichever way round.
fn key(a: u32, b: u32) -> (u32, u32) {
    (a.min(b), a.max(b))
}

/// One round: splits every long edge. Whether any was.
fn round<G: Geometry>(mesh: &mut TangentMesh, max_edge: Real) -> bool {
    let points = mesh.embedded_points::<G>();
    let mut middles: HashMap<(u32, u32), u32> = HashMap::new();
    for t in mesh.indices.as_chunks::<3>().0 {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            let long = G::distance(&points[a as usize], &points[b as usize]) > max_edge;
            if long && !middles.contains_key(&key(a, b)) {
                middles.insert(key(a, b), u32::MAX);
            }
        }
    }
    if middles.is_empty() {
        return false;
    }
    // The middles, in a fixed order so the mesh comes out the same each time.
    let mut edges: Vec<_> = middles.keys().copied().collect();
    edges.sort_unstable();
    for (a, b) in edges {
        let (a, b) = (a as usize, b as usize);
        let index = u32::try_from(mesh.positions.len()).expect("mesh has too many vertices");
        mesh.positions
            .push((mesh.positions[a] + mesh.positions[b]) / 2.0);
        let normal = mesh.normals[a] + mesh.normals[b];
        mesh.normals.push(if normal.norm() > 1e-12 {
            normal.normalize()
        } else {
            mesh.normals[a]
        });
        let (ua, ub) = (mesh.uvs[a], mesh.uvs[b]);
        mesh.uvs
            .push([(ua[0] + ub[0]) / 2.0, (ua[1] + ub[1]) / 2.0]);
        middles.insert(key(a as u32, b as u32), index);
    }
    let mut indices = Vec::with_capacity(mesh.indices.len() * 2);
    for t in mesh.indices.as_chunks::<3>().0 {
        let v = [t[0], t[1], t[2]];
        // The middle of edge k, from v[k] to v[k + 1].
        let m: [Option<u32>; 3] =
            std::array::from_fn(|k| middles.get(&key(v[k], v[(k + 1) % 3])).copied());
        match m.iter().filter(|m| m.is_some()).count() {
            0 => indices.extend(v),
            3 => {
                let [m0, m1, m2] = m.map(Option::unwrap);
                indices.extend([v[0], m0, m2, m0, v[1], m1, m2, m1, v[2], m0, m1, m2]);
            }
            1 => {
                // Turned so the split edge is the first.
                let k = m.iter().position(Option::is_some).unwrap();
                let (a, b, c) = (v[k], v[(k + 1) % 3], v[(k + 2) % 3]);
                let mid = m[k].unwrap();
                indices.extend([a, mid, c, mid, b, c]);
            }
            _ => {
                // Turned so the edge left whole is the last, from c back to a.
                let k = (m.iter().position(Option::is_none).unwrap() + 1) % 3;
                let (a, b, c) = (v[k], v[(k + 1) % 3], v[(k + 2) % 3]);
                let (mab, mbc) = (m[k].unwrap(), m[(k + 1) % 3].unwrap());
                indices.extend([mab, b, mbc, a, mab, mbc, a, mbc, c]);
            }
        }
    }
    mesh.indices = indices;
    true
}

#[cfg(test)]
mod tests {
    use fk_geometry_euclidean::E3;
    use fk_math::nalgebra::Vector3;

    use super::*;

    /// A square `side` metres across on the ground, two triangles, facing up.
    fn square(side: Real) -> TangentMesh {
        let h = side / 2.0;
        TangentMesh {
            positions: vec![
                Vector3::new(-h, 0.0, -h),
                Vector3::new(-h, 0.0, h),
                Vector3::new(h, 0.0, h),
                Vector3::new(h, 0.0, -h),
            ],
            normals: vec![Vector3::y(); 4],
            uvs: vec![[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
            indices: vec![0, 1, 2, 0, 2, 3],
        }
    }

    fn area(mesh: &TangentMesh) -> Real {
        mesh.indices
            .as_chunks::<3>()
            .0
            .iter()
            .map(|t| {
                let [a, b, c] = [0, 1, 2].map(|k| mesh.positions[t[k] as usize]);
                (b - a).cross(&(c - a)).norm() / 2.0
            })
            .sum()
    }

    /// How many triangles use each edge, whichever way round.
    fn edge_uses(mesh: &TangentMesh) -> HashMap<(u32, u32), usize> {
        let mut uses = HashMap::new();
        for t in mesh.indices.as_chunks::<3>().0 {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                *uses.entry(key(a, b)).or_insert(0) += 1;
            }
        }
        uses
    }

    #[test]
    fn every_edge_comes_out_short_and_the_surface_the_same() {
        let flat = square(10.0);
        let fine = refine::<E3>(&flat, 1.0);
        assert!(
            fine.longest_edge::<E3>() <= 1.0,
            "{}",
            fine.longest_edge::<E3>()
        );
        assert!((area(&fine) - 100.0).abs() < 1e-9);
        // Still facing up, every one of them.
        for t in fine.indices.as_chunks::<3>().0 {
            let [a, b, c] = [0, 1, 2].map(|k| fine.positions[t[k] as usize]);
            assert!((b - a).cross(&(c - a)).y > 0.0, "a triangle turned over");
        }
    }

    #[test]
    fn it_stays_conforming() {
        // No T-junctions: the edges used once are exactly the square's border, 40 m of it.
        let fine = refine::<E3>(&square(10.0), 0.7);
        let border: Real = edge_uses(&fine)
            .iter()
            .map(|(&(a, b), &uses)| {
                assert!(uses <= 2, "an edge in {uses} triangles");
                if uses == 1 {
                    (fine.positions[a as usize] - fine.positions[b as usize]).norm()
                } else {
                    0.0
                }
            })
            .sum();
        assert!((border - 40.0).abs() < 1e-9, "{border}");
    }

    #[test]
    fn a_short_mesh_is_left_alone_and_uvs_follow_the_splits() {
        let flat = square(1.0);
        assert_eq!(refine::<E3>(&flat, 2.0), flat);
        let fine = refine::<E3>(&square(4.0), 1.5);
        for (p, uv) in fine.positions.iter().zip(&fine.uvs) {
            assert!((uv[0] - (p.x / 4.0 + 0.5)).abs() < 1e-12);
            assert!((uv[1] - (p.z / 4.0 + 0.5)).abs() < 1e-12);
        }
    }
}
