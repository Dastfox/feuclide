//! Procedural meshes, authored in tangent space so they embed into any geometry.
//!
//! Every shape is in the reference frame at the origin: `y` (frame vector 1) is up, `x` and `z`
//! span the ground. Call [`MeshBuilder::build`] on the result for the app's geometry.

use std::collections::{BTreeMap, HashMap};
use std::f64::consts::TAU;

use fk_math::Real;
use fk_math::nalgebra::{Vector2, Vector3};
use fk_math::rand::Seeded;

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

/// How far in from its edges a [`bevelled`] box's faces stop, in the units of its half extents.
pub const BEVEL: Real = 0.015;

/// A box centred on the origin with the given half extents, flat faces.
pub fn cuboid(half: Vector3<Real>) -> MeshBuilder {
    // Trial: every box bevelled, to measure what it costs.
    bevelled(half, BEVEL)
}

/// A box centred on the origin with the given half extents, its edges and corners cut across
/// `bevel` in from each face (at most a fifth of its thinnest half extent). Each face's corners
/// keep the face's normal and the cuts between faces are made of those same corners, so the
/// light rolls round an edge instead of breaking on it: 24 vertices, 44 triangles.
pub fn bevelled(half: Vector3<Real>, bevel: Real) -> MeshBuilder {
    let r = bevel.min(0.2 * half.min());
    if r <= 0.0 {
        return sharp(half);
    }
    let mut mesh = MeshBuilder::new();
    // The corner of face `axis` towards `signs` (its own sign the face's), by `signs` and axis.
    let mut corner = HashMap::new();
    for axis in 0..3 {
        for x in [-1.0, 1.0] {
            for y in [-1.0, 1.0] {
                for z in [-1.0, 1.0] {
                    let signs = Vector3::new(x, y, z);
                    let p = Vector3::from_fn(|k, _| {
                        signs[k] * if k == axis { half[k] } else { half[k] - r }
                    });
                    let n = Vector3::from_fn(|k, _| if k == axis { signs[k] } else { 0.0 });
                    let key = (axis, [x, y, z].map(|s| s > 0.0));
                    corner.insert(key, mesh.vertex(p, n));
                }
            }
        }
    }
    let at = |axis: usize, signs: [bool; 3]| corner[&(axis, signs)];
    // Wound outward: the box is convex about the origin.
    fn facet(mesh: &mut MeshBuilder, [a, b, c]: [u32; 3]) {
        let p = |i: u32| mesh.positions()[i as usize];
        if (p(b) - p(a))
            .cross(&(p(c) - p(a)))
            .dot(&(p(a) + p(b) + p(c)))
            >= 0.0
        {
            mesh.triangle(a, b, c);
        } else {
            mesh.triangle(a, c, b);
        }
    }
    fn quad(mesh: &mut MeshBuilder, [a, b, c, d]: [u32; 4]) {
        facet(mesh, [a, b, c]);
        facet(mesh, [a, c, d]);
    }
    for axis in 0..3 {
        let (j, k) = ((axis + 1) % 3, (axis + 2) % 3);
        for s in [false, true] {
            // The face, its corners in turn round it.
            let q = [(false, false), (true, false), (true, true), (false, true)].map(|(a, b)| {
                let mut signs = [false; 3];
                (signs[axis], signs[j], signs[k]) = (s, a, b);
                at(axis, signs)
            });
            quad(&mut mesh, q);
            // The cut along the edge it shares with face `j`.
            for t in [false, true] {
                let ends = [false, true].map(|b| {
                    let mut signs = [false; 3];
                    (signs[axis], signs[j], signs[k]) = (s, t, b);
                    signs
                });
                quad(
                    &mut mesh,
                    [
                        at(axis, ends[0]),
                        at(axis, ends[1]),
                        at(j, ends[1]),
                        at(j, ends[0]),
                    ],
                );
            }
        }
    }
    // The corners.
    for x in [false, true] {
        for y in [false, true] {
            for z in [false, true] {
                let signs = [x, y, z];
                facet(&mut mesh, [at(0, signs), at(1, signs), at(2, signs)]);
            }
        }
    }
    mesh
}

/// A box with sharp edges: six flat quads.
fn sharp(half: Vector3<Real>) -> MeshBuilder {
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

/// A convex polygon in the xy plane (counter-clockwise) drawn out along z from `z0` to `z1`,
/// with flat faces.
pub fn extruded(polygon: &[Vector2<Real>], z0: Real, z1: Real) -> MeshBuilder {
    let mut mesh = MeshBuilder::new();
    let n = polygon.len();
    for (z, normal) in [(z1, Vector3::z()), (z0, -Vector3::z())] {
        let ids: Vec<u32> = polygon
            .iter()
            .map(|p| mesh.vertex(Vector3::new(p.x, p.y, z), normal))
            .collect();
        for k in 1..n - 1 {
            if normal.z > 0.0 {
                mesh.triangle(ids[0], ids[k], ids[k + 1]);
            } else {
                mesh.triangle(ids[0], ids[k + 1], ids[k]);
            }
        }
    }
    for k in 0..n {
        let (a, b) = (polygon[k], polygon[(k + 1) % n]);
        let normal = Vector3::new(b.y - a.y, a.x - b.x, 0.0).normalize();
        let corners = [
            Vector3::new(a.x, a.y, z1),
            Vector3::new(a.x, a.y, z0),
            Vector3::new(b.x, b.y, z0),
            Vector3::new(b.x, b.y, z1),
        ];
        let [p, q, r, s] = corners.map(|c| mesh.vertex(c, normal));
        mesh.quad(p, q, r, s);
    }
    mesh
}

/// A cone standing on the origin: `radius` at its foot, `height` tall, `sides` facets, shaded
/// round.
pub fn cone(radius: Real, height: Real, sides: u32) -> MeshBuilder {
    let sides = sides.max(3);
    let mut mesh = MeshBuilder::new();
    let rim = |k: u32| {
        let a = TAU * Real::from(k) / Real::from(sides);
        Vector3::new(a.cos(), 0.0, a.sin())
    };
    let slope = radius / height.max(1e-6);
    let normal = |out: Vector3<Real>| Vector3::new(out.x, slope, out.z).normalize();
    for k in 0..sides {
        let (a, b) = (rim(k), rim(k + 1));
        // The apex takes the facet's middle, its feet their own way round.
        let apex = mesh.vertex(Vector3::new(0.0, height, 0.0), normal((a + b).normalize()));
        let foot_a = mesh.vertex(a * radius, normal(a));
        let foot_b = mesh.vertex(b * radius, normal(b));
        mesh.triangle(apex, foot_b, foot_a);
    }
    mesh
}

/// Adds a quad to `mesh` facing up, whichever way its corners turn.
pub fn up_quad(mesh: &mut MeshBuilder, [a, b, c, d]: [u32; 4]) {
    let p = |i: u32| mesh.positions()[i as usize];
    if (p(b) - p(a)).cross(&(p(c) - p(a))).y > 0.0 {
        mesh.quad(a, b, c, d);
    } else {
        mesh.quad(a, d, c, b);
    }
}

/// How far across a boulder's facets are at most, in the units of its `scale`, unless that would
/// take more than three subdivisions: a pebble stays an icosahedron, a rock a man's height is
/// round however near the eye comes.
pub const BOULDER_FACET: Real = 0.15;

/// A boulder: an icosahedron stretched by `scale`, subdivided until its facets are
/// [`BOULDER_FACET`] across, shaded round ([`smooth`]). Exactly an ellipsoid's points, so
/// whatever is laid on it stays on it.
pub fn boulder(scale: Vector3<Real>) -> MeshBuilder {
    stone(scale, 0.0, 0)
}

/// The corners of a boulder of `scale` as the coarsest one lists them: the icosahedron's, once
/// for every facet each is a corner of. Points on its surface to lay things on, the same whatever
/// its detail, so that a layout picking among them stays as it was.
pub fn boulder_corners(scale: Vector3<Real>) -> Vec<Vector3<Real>> {
    let base = sphere(1.0, 0);
    base.indices()
        .iter()
        .map(|&i| base.positions()[i as usize].component_mul(&scale))
        .collect()
}

/// A boulder ([`boulder`]) whose surface swells and dips by up to `lumps` of its size: a few
/// broad waves over its directions, the same for the same `seed`, so it reads as a stone rather
/// than an egg.
pub fn stone(scale: Vector3<Real>, lumps: Real, seed: u64) -> MeshBuilder {
    // An icosahedron's edge on the unit sphere, halved by each subdivision.
    const EDGE: Real = 1.0515;
    let widest = scale.amax();
    let mut subdivisions = 0;
    while subdivisions < 3 && EDGE * widest * 0.5_f64.powi(subdivisions) > BOULDER_FACET {
        subdivisions += 1;
    }
    let base = sphere(1.0, subdivisions as u32);
    // Each wave: its axis, how many radians it turns across the stone, its phase, its weight.
    let mut random = Seeded::new(seed);
    let waves: Vec<(Vector3<Real>, Real, Real, Real)> = [
        ([2.0, 3.0], 1.0),
        ([3.0, 4.5], 0.6),
        ([5.0, 7.0], 0.35),
        ([8.0, 11.0], 0.2),
    ]
    .into_iter()
    .map(|(turns, weight)| {
        let axis = random.direction();
        (axis, random.between(turns), TAU * random.next(), weight)
    })
    .collect();
    let total: Real = waves.iter().map(|w| w.3).sum();
    let swell = |d: &Vector3<Real>| {
        waves
            .iter()
            .map(|(axis, turns, phase, weight)| weight * (turns * axis.dot(d) + phase).sin())
            .sum::<Real>()
            / total
    };
    let mut mesh = MeshBuilder::new();
    let positions: Vec<Vector3<Real>> = base
        .positions()
        .iter()
        .map(|d| (d * (1.0 + lumps * swell(d))).component_mul(&scale))
        .collect();
    for t in base.indices().as_chunks::<3>().0 {
        let [a, b, c] = t.map(|i| positions[i as usize]);
        let mut normal = (b - a).cross(&(c - a)).normalize();
        let outward = (a + b + c).dot(&normal) > 0.0;
        if !outward {
            normal = -normal;
        }
        let [i, j, k] = [a, b, c].map(|p| mesh.vertex(p, normal));
        if outward {
            mesh.triangle(i, j, k);
        } else {
            mesh.triangle(i, k, j);
        }
    }
    smooth(&mesh)
}

/// `mesh` shaded round: each vertex's normal the mean of the faces meeting where it stands,
/// weighted by their areas. For things whose every facet stands for a curve (a boulder); a
/// square-cut thing keeps its own normals.
pub fn smooth(mesh: &MeshBuilder) -> MeshBuilder {
    let positions = mesh.positions();
    let key = |p: &Vector3<Real>| [p.x, p.y, p.z].map(|x| (x * 1e6).round() as i64);
    let mut sums: BTreeMap<[i64; 3], Vector3<Real>> = BTreeMap::new();
    for t in mesh.indices().as_chunks::<3>().0 {
        let [a, b, c] = t.map(|i| positions[i as usize]);
        // The cross product's length is twice the area: weighted by it.
        let face = (b - a).cross(&(c - a));
        for p in [a, b, c] {
            *sums.entry(key(&p)).or_insert_with(Vector3::zeros) += face;
        }
    }
    let mut out = MeshBuilder::new();
    for (p, n) in positions.iter().zip(mesh.normals()) {
        let sum = sums.get(&key(p)).copied().unwrap_or(*n);
        // Faces wound against their normals: keep the side the vertex showed.
        let sum = if sum.dot(n) < 0.0 { -sum } else { sum };
        out.vertex(*p, if sum.norm() > 1e-12 { sum } else { *n });
    }
    for t in mesh.indices().as_chunks::<3>().0 {
        out.triangle(t[0], t[1], t[2]);
    }
    out
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
        assert_outward("bevelled", &bevelled(Vector3::new(1.0, 2.0, 0.5), 0.1));
        assert_outward("boulder", &boulder(Vector3::new(1.2, 0.6, 0.9)));
        assert_outward("stone", &stone(Vector3::new(1.2, 0.6, 0.9), 0.12, 7));
    }

    #[test]
    fn a_bevelled_box_is_closed_and_cut_where_it_says() {
        let half = Vector3::new(1.0, 2.0, 0.5);
        let mesh = bevelled(half, 0.1);
        assert_eq!(mesh.positions().len(), 24);
        assert_eq!(mesh.indices().len(), 44 * 3);
        // Every edge between two of its triangles, once each way.
        let mut edges = BTreeMap::new();
        for t in mesh.indices().as_chunks::<3>().0 {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                *edges.entry((a, b)).or_insert(0) += 1;
            }
        }
        for (&(a, b), &count) in &edges {
            assert_eq!(count, 1, "edge {a}-{b}");
            assert_eq!(edges.get(&(b, a)), Some(&1), "edge {a}-{b} open");
        }
        for p in mesh.positions() {
            let d = half - p.abs();
            let mut d = [d.x, d.y, d.z];
            d.sort_by(Real::total_cmp);
            assert!(d[0].abs() < 1e-12 && (d[1] - 0.1).abs() < 1e-12, "{p}");
        }
        // Thinner than five bevels, the cut shrinks; at nothing, the box is sharp.
        let thin = bevelled(Vector3::new(1.0, 0.01, 1.0), 0.1);
        assert!(thin.positions().iter().all(|p| p.x.abs() > 1.0 - 0.0021));
        assert_eq!(bevelled(half, 0.0).indices().len(), 12 * 3);
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
