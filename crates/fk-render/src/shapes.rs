//! Procedural meshes, authored in tangent space so they embed into any geometry.
//!
//! Every shape is in the reference frame at the origin: `y` (frame vector 1) is up, `x` and `z`
//! span the ground. Call [`MeshBuilder::build`] on the result for the app's geometry.

use std::collections::HashMap;
use std::f64::consts::TAU;

use fk_math::Real;
use fk_math::nalgebra::Vector3;

use crate::MeshBuilder;

/// A square of side `size` on the ground (`y = 0`), facing up, centred on the origin, cut into
/// `subdivisions × subdivisions` cells so that it can bend (in curved space, or under a
/// displacement shader).
pub fn ground_tile(size: Real, subdivisions: u32) -> MeshBuilder {
    let n = subdivisions.max(1);
    let step = size / Real::from(n);
    let half = size / 2.0;
    let mut mesh = MeshBuilder::new();
    let up = Vector3::y();
    for i in 0..=n {
        for j in 0..=n {
            let position = Vector3::new(
                -half + Real::from(i) * step,
                0.0,
                -half + Real::from(j) * step,
            );
            mesh.vertex(position, up);
        }
    }
    let index = |i: u32, j: u32| i * (n + 1) + j;
    for i in 0..n {
        for j in 0..n {
            mesh.quad(
                index(i, j),
                index(i, j + 1),
                index(i + 1, j + 1),
                index(i + 1, j),
            );
        }
    }
    mesh
}

/// A flat disc of radius `radius` on the ground (`y = 0`), facing up, centred on the origin: a
/// fan of `sides` triangles.
pub fn disc(radius: Real, sides: u32) -> MeshBuilder {
    let sides = sides.max(3);
    let mut mesh = MeshBuilder::new();
    let up = Vector3::y();
    let centre = mesh.vertex(Vector3::zeros(), up);
    let rim: Vec<u32> = (0..sides)
        .map(|k| {
            let angle = TAU * Real::from(k) / Real::from(sides);
            mesh.vertex(Vector3::new(angle.cos(), 0.0, angle.sin()) * radius, up)
        })
        .collect();
    for k in 0..sides as usize {
        mesh.triangle(centre, rim[(k + 1) % rim.len()], rim[k]);
    }
    mesh
}

/// A box centred on the origin with the given half extents, flat faces.
pub fn cuboid(half: Vector3<Real>) -> MeshBuilder {
    let mut mesh = MeshBuilder::new();
    let (x, y, z) = (Vector3::x(), Vector3::y(), Vector3::z());
    // (normal, u, v) with u × v = normal.
    for (normal, u, v) in [
        (x, y, z),
        (-x, z, y),
        (y, z, x),
        (-y, x, z),
        (z, x, y),
        (-z, y, x),
    ] {
        let scale = |w: Vector3<Real>| w.component_mul(&half);
        let (c, u, v) = (scale(normal), scale(u), scale(v));
        mesh.flat_quad([c - u - v, c + u - v, c + u + v, c - u + v], normal);
    }
    mesh
}

/// A vertical prism with `sides` flat sides, standing on the origin: the base circle of radius
/// `radius` on the ground, the top at height `height`.
pub fn prism(radius: Real, height: Real, sides: u32) -> MeshBuilder {
    upright(radius, height, sides, false)
}

/// A vertical cylinder standing on the origin, smooth sides approximated by `sides` facets.
pub fn cylinder(radius: Real, height: Real, sides: u32) -> MeshBuilder {
    upright(radius, height, sides, true)
}

fn upright(radius: Real, height: Real, sides: u32, smooth: bool) -> MeshBuilder {
    let sides = sides.max(3);
    let mut mesh = MeshBuilder::new();
    let rim = |k: u32| {
        let angle = TAU * Real::from(k) / Real::from(sides);
        Vector3::new(angle.cos(), 0.0, angle.sin())
    };
    let up = Vector3::new(0.0, height, 0.0);
    for k in 0..sides {
        let (a, b) = (rim(k), rim(k + 1));
        let (bottom_a, bottom_b) = (a * radius, b * radius);
        let (top_a, top_b) = (bottom_a + up, bottom_b + up);
        if smooth {
            let side = [
                mesh.vertex(bottom_b, b),
                mesh.vertex(bottom_a, a),
                mesh.vertex(top_a, a),
                mesh.vertex(top_b, b),
            ];
            mesh.quad(side[0], side[1], side[2], side[3]);
        } else {
            mesh.flat_quad([bottom_b, bottom_a, top_a, top_b], a + b);
        }
    }
    for (y, normal) in [(height, Vector3::y()), (0.0, -Vector3::y())] {
        let centre = mesh.vertex(Vector3::new(0.0, y, 0.0), normal);
        let first = mesh.vertex(rim(0) * radius + Vector3::new(0.0, y, 0.0), normal);
        let mut previous = first;
        for k in 1..=sides {
            let next = if k == sides {
                first
            } else {
                mesh.vertex(rim(k) * radius + Vector3::new(0.0, y, 0.0), normal)
            };
            if y > 0.0 {
                mesh.triangle(centre, next, previous);
            } else {
                mesh.triangle(centre, previous, next);
            }
            previous = next;
        }
    }
    mesh
}

/// A sphere of radius `radius` about the origin: an icosahedron subdivided `subdivisions`
/// times, with smooth normals. In curved space this is the geodesic sphere.
pub fn sphere(radius: Real, subdivisions: u32) -> MeshBuilder {
    let t = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let mut points: Vec<Vector3<Real>> = [
        (-1.0, t, 0.0),
        (1.0, t, 0.0),
        (-1.0, -t, 0.0),
        (1.0, -t, 0.0),
        (0.0, -1.0, t),
        (0.0, 1.0, t),
        (0.0, -1.0, -t),
        (0.0, 1.0, -t),
        (t, 0.0, -1.0),
        (t, 0.0, 1.0),
        (-t, 0.0, -1.0),
        (-t, 0.0, 1.0),
    ]
    .iter()
    .map(|&(x, y, z)| Vector3::new(x, y, z).normalize())
    .collect();
    let mut faces: Vec<[u32; 3]> = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];
    for _ in 0..subdivisions {
        let mut midpoints = HashMap::new();
        let mut midpoint = |a: u32, b: u32, points: &mut Vec<Vector3<Real>>| {
            *midpoints.entry((a.min(b), a.max(b))).or_insert_with(|| {
                points.push((points[a as usize] + points[b as usize]).normalize());
                u32::try_from(points.len() - 1).expect("sphere has too many vertices")
            })
        };
        faces = faces
            .iter()
            .flat_map(|&[a, b, c]| {
                let ab = midpoint(a, b, &mut points);
                let bc = midpoint(b, c, &mut points);
                let ca = midpoint(c, a, &mut points);
                [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
            })
            .collect();
    }
    let mut mesh = MeshBuilder::new();
    for point in &points {
        mesh.vertex(point * radius, *point);
    }
    for [a, b, c] in faces {
        let (pa, pb, pc) = (points[a as usize], points[b as usize], points[c as usize]);
        if (pb - pa).cross(&(pc - pa)).dot(&(pa + pb + pc)) >= 0.0 {
            mesh.triangle(a, b, c);
        } else {
            mesh.triangle(a, c, b);
        }
    }
    mesh
}

#[cfg(test)]
mod tests {
    use fk_geometry_euclidean::E3;

    use super::*;

    /// Every triangle is counter-clockwise around its vertices' normals, and normals are unit.
    fn assert_outward(name: &str, mesh: &MeshBuilder) {
        let (positions, normals) = (mesh.positions(), mesh.normals());
        assert!(!mesh.indices().is_empty(), "{name} is empty");
        for triangle in mesh.indices().chunks(3) {
            let [a, b, c] = [0, 1, 2].map(|k| triangle[k] as usize);
            let face = (positions[b] - positions[a]).cross(&(positions[c] - positions[a]));
            let normal = normals[a] + normals[b] + normals[c];
            assert!(
                face.dot(&normal) > 0.0,
                "{name}: triangle {triangle:?} faces inwards"
            );
        }
        for normal in normals {
            assert!((normal.norm() - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn shapes_face_outwards() {
        assert_outward("ground tile", &ground_tile(4.0, 3));
        assert_outward("disc", &disc(4.0, 12));
        assert_outward("cuboid", &cuboid(Vector3::new(1.0, 2.0, 0.5)));
        assert_outward("prism", &prism(0.5, 3.0, 6));
        assert_outward("cylinder", &cylinder(0.5, 3.0, 16));
        assert_outward("sphere", &sphere(2.0, 2));
    }

    #[test]
    fn sphere_vertices_are_on_the_sphere() {
        let mesh = sphere(2.0, 3).build::<E3>();
        assert!((mesh.radius - 2.0).abs() < 1e-6);
        assert_eq!(mesh.indices.len(), 20 * 4usize.pow(3) * 3);
        for vertex in &mesh.vertices {
            let [x, y, z, w] = vertex.position.0;
            assert_eq!(w, 1.0);
            assert!(((x * x + y * y + z * z).sqrt() - 2.0).abs() < 1e-5);
        }
    }

    #[test]
    fn bounds_hold_every_vertex() {
        let mesh = prism(0.5, 3.0, 6).build::<E3>();
        let expected = (0.25_f64 + 9.0).sqrt();
        assert!((mesh.radius - expected).abs() < 1e-9, "{}", mesh.radius);
    }
}
