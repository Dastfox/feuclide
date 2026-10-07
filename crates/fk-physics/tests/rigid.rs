//! Rigid bodies in E³, H³ and S³: the algebra's coordinates against the group, the inertia
//! operator, a free spinner's invariants, and a box coming to rest on a floor.

use fk_geometry::{Geometry, GroupElement};
use fk_math::Real;
use fk_math::nalgebra::{Vector3, Vector6};
use fk_physics::rigid::{adjoint, algebra, spin, translation, velocity_at};
use fk_physics::{Gravity, Inertia, RigidBody, coords, tangent};

/// Points to compare isometries by.
fn probes<G: Geometry>() -> Vec<G::Point> {
    let o = G::origin();
    let mut points = vec![o];
    points.extend((0..3).map(|i| G::exp(&o, &(G::origin_frame(i) * 0.3))));
    points
}

fn same<G: Geometry>(a: &G::Isometry, b: &G::Isometry) -> Real {
    probes::<G>()
        .iter()
        .map(|p| G::distance(&G::apply(a, p), &G::apply(b, p)))
        .fold(0.0, Real::max)
}

fn element<G: Geometry>(c: &Vector6<Real>) -> G::Isometry {
    G::Isometry::exp(&algebra::<G>(c))
}

/// `g exp(ξ) g⁻¹ = exp(Ad_g ξ)` for the coordinate adjoint: the brackets' signs and the
/// curvature are the group's.
fn the_adjoint_is_the_groups<G: Geometry>() {
    let g = element::<G>(&Vector6::new(0.4, -0.2, 0.3, 0.5, 0.1, -0.7))
        .compose(&element::<G>(&Vector6::new(-0.1, 0.6, 0.2, -0.3, 0.8, 0.2)));
    let xi = Vector6::new(0.3, 0.1, -0.4, 0.2, -0.5, 0.6);
    let conjugated = g.compose(&element::<G>(&xi)).compose(&g.inverse());
    let by_coords = element::<G>(&(adjoint::<G>(&g) * xi));
    let error = same::<G>(&conjugated, &by_coords);
    assert!(error < 1e-10, "{}: Ad off by {error}", G::NAME);
}

/// The velocity a twist gives a body point is the point's derivative along the flow.
fn velocities_are_the_flows<G: Geometry>() {
    let x = Vector3::new(0.2, -0.3, 0.25);
    let c = Vector6::new(0.3, -0.7, 0.2, 0.9, 0.4, -0.5);
    let o = G::origin();
    let frame = G::Isometry::exp(&G::transvection(&tangent::<G>(&x)));
    let point = G::apply(&frame, &o);
    let h = 1e-6;
    let ahead = G::apply(&element::<G>(&(c * h)), &point);
    let behind = G::apply(&element::<G>(&(c * -h)), &point);
    // The flow's velocity, pulled back to the origin's frame along the frame at the point.
    let back = frame.inverse();
    let (a, b) = (G::apply(&back, &ahead), G::apply(&back, &behind));
    let finite = coords::<G>(&G::log(&o, &a).unwrap()) - coords::<G>(&G::log(&o, &b).unwrap());
    let finite = finite / (2.0 * h);
    let exact = velocity_at::<G>(&x) * c;
    assert!(
        (finite - exact).norm() < 1e-6,
        "{}: {finite} against {exact}",
        G::NAME
    );
}

/// Torque-free spin about no principal axis, moving too, for fifty seconds: the energy stays
/// within a bound of where it started and the spatial momentum is kept.
fn a_free_spinner_keeps_its_invariants<G: Geometry>() {
    let inertia = Inertia::<G>::cuboid(2.0, Vector3::new(0.3, 0.15, 0.08));
    // Slow enough to stay within a few units of the origin: far out in H³ the pose's
    // coordinates grow like e^d and `Ad*_{g⁻¹}` loses the momentum to cancellation, which is
    // what the floating origin is for.
    let twist = Vector6::new(0.04, 0.01, -0.02, 3.0, 5.0, 0.4);
    let mut body = RigidBody::moving(
        element::<G>(&translation(&Vector3::new(0.1, 0.0, 0.2))),
        &twist,
        &inertia,
    );
    let (energy, momentum) = (body.energy(&inertia), body.spatial_momentum());
    let dt = 1.0 / 120.0;
    let mut worst: Real = 0.0;
    for _ in 0..6000 {
        body.drift(&inertia, dt);
        worst = worst.max((body.energy(&inertia) - energy).abs() / energy);
    }
    let drift = (body.spatial_momentum() - momentum).norm() / momentum.norm();
    assert!(worst < 2e-4, "{}: energy off by {worst}", G::NAME);
    assert!(
        drift < 1e-11,
        "{}: spatial momentum off by {drift}",
        G::NAME
    );
    assert!(
        body.pose.residual() < 1e-9,
        "{}: pose residual {}",
        G::NAME,
        body.pose.residual()
    );
    // And it went somewhere, turning on the way.
    let moved = G::distance(&G::apply(&body.pose, &G::origin()), &G::origin());
    assert!(moved > 0.05, "{}: moved {moved}", G::NAME);
}

/// The sine of the space's curvature (`sin`, `sinh`, or the identity) and its inverse.
fn sn<G: Geometry>(r: Real) -> Real {
    match G::constant_curvature().unwrap() {
        k if k > 0.0 => r.sin(),
        k if k < 0.0 => r.sinh(),
        _ => r,
    }
}

fn asn<G: Geometry>(s: Real) -> Real {
    match G::constant_curvature().unwrap() {
        k if k > 0.0 => s.asin(),
        k if k < 0.0 => s.asinh(),
        _ => s,
    }
}

/// Signed distance to the totally geodesic plane through the origin normal to frame vector 1
/// (the right triangle's `sn d = sn r · sin θ`).
fn floor<G: Geometry>(p: &G::Point) -> Real {
    let v = coords::<G>(&G::log(&G::origin(), p).unwrap());
    let r = v.norm();
    if r < 1e-12 {
        return v.y;
    }
    asn::<G>(sn::<G>(r) * v.y / r)
}

/// A box dropped tilted and turning onto a floor bounces no higher, comes to rest flat on it
/// and stays there.
fn a_box_settles_on_a_floor<G: Geometry>() {
    let half = Vector3::new(0.1, 0.1, 0.1);
    let inertia = Inertia::<G>::cuboid(1.0, half);
    let corners: Vec<Vector3<Real>> = (0..8)
        .map(|i| {
            let s = |b| if i & b == 0 { -1.0 } else { 1.0 };
            Vector3::new(s(1) * half.x, s(2) * half.y, s(4) * half.z)
        })
        .collect();
    let gravity = Gravity::<G> {
        down: G::origin_frame(1) * -1.0,
        acceleration: 9.81,
    };
    let start = element::<G>(&translation(&Vector3::new(0.05, 0.35, -0.03))).compose(
        &element::<G>(&spin(&(Vector3::new(1.0, 0.2, 1.0).normalize() * 0.5))),
    );
    let mut body = RigidBody::moving(start, &spin(&Vector3::new(0.0, 2.0, 1.0)), &inertia);
    let dt = 1.0 / 120.0;
    let mut deepest: Real = 0.0;
    let mut settled = body.pose;
    for step in 0..600 {
        body.kick(&inertia.gravity(&body.pose, &gravity), dt);
        let depth = body.rest_on(&inertia, &corners, &floor::<G>, 0.6, dt);
        body.drift(&inertia, dt);
        if step > 120 {
            deepest = deepest.max(depth);
        }
        if step == 480 {
            settled = body.pose;
        }
    }
    let up = G::apply_tangent(&body.pose, &G::origin_frame(1));
    let centre = G::apply(&body.pose, &G::origin());
    let level = G::inner(
        &centre,
        &up,
        &G::parallel_transport(&G::origin(), &centre, &G::origin_frame(1)),
    )
    .abs()
    .max(
        G::inner(
            &centre,
            &G::apply_tangent(&body.pose, &G::origin_frame(0)),
            &G::parallel_transport(&G::origin(), &centre, &G::origin_frame(1)),
        )
        .abs(),
    )
    .max(
        G::inner(
            &centre,
            &G::apply_tangent(&body.pose, &G::origin_frame(2)),
            &G::parallel_transport(&G::origin(), &centre, &G::origin_frame(1)),
        )
        .abs(),
    );
    let height = floor::<G>(&centre);
    let still = same::<G>(&settled, &body.pose);
    let speed = body.twist(&inertia).norm();
    eprintln!(
        "{}: height {height}, deepest {deepest}, level {level}, still {still}, speed {speed}",
        G::NAME
    );
    assert!(deepest < 0.01, "{}: sank {deepest} into the floor", G::NAME);
    assert!(
        (height - half.y).abs() < 0.01,
        "{}: rests at {height}",
        G::NAME
    );
    assert!(level > 0.999, "{}: not flat ({level})", G::NAME);
    assert!(
        still < 1e-3,
        "{}: still moving ({still} in a second)",
        G::NAME
    );
    assert!(speed < 1e-2, "{}: twist {speed}", G::NAME);
}

macro_rules! rigid {
    ($name:ident, $g:ty) => {
        mod $name {
            #[test]
            fn the_adjoint_is_the_groups() {
                super::the_adjoint_is_the_groups::<$g>();
            }

            #[test]
            fn velocities_are_the_flows() {
                super::velocities_are_the_flows::<$g>();
            }

            #[test]
            fn a_free_spinner_keeps_its_invariants() {
                super::a_free_spinner_keeps_its_invariants::<$g>();
            }

            #[test]
            fn a_box_settles_on_a_floor() {
                super::a_box_settles_on_a_floor::<$g>();
            }
        }
    };
}

rigid!(e3, fk_geometry_euclidean::E3);
rigid!(h3, fk_geometry_hyperbolic::H3);
rigid!(s3, fk_geometry_spherical::S3);

#[test]
fn a_box_in_flat_space_has_the_textbook_inertia() {
    let (m, a, b, c) = (3.0, 0.5, 0.2, 0.1);
    let inertia = Inertia::<fk_geometry_euclidean::E3>::cuboid(m, Vector3::new(a, b, c));
    let expected = Vector6::new(
        m,
        m,
        m,
        m * (b * b + c * c) / 3.0,
        m * (a * a + c * c) / 3.0,
        m * (a * a + b * b) / 3.0,
    );
    let diagonal = inertia.matrix.diagonal();
    assert!((diagonal - expected).norm() < 1e-12, "{diagonal}");
    let off = inertia.matrix - fk_math::nalgebra::Matrix6::from_diagonal(&diagonal);
    assert!(off.norm() < 1e-12, "{off}");
}

#[test]
fn in_curved_space_the_same_box_is_heavier_to_turn_where_space_opens_out() {
    let half = Vector3::new(0.4, 0.4, 0.4);
    let turn = |m: fk_math::nalgebra::Matrix6<Real>| m[(3, 3)];
    let flat = turn(Inertia::<fk_geometry_euclidean::E3>::cuboid(1.0, half).matrix);
    let open = turn(Inertia::<fk_geometry_hyperbolic::H3>::cuboid(1.0, half).matrix);
    let closed = turn(Inertia::<fk_geometry_spherical::S3>::cuboid(1.0, half).matrix);
    assert!(closed < flat && flat < open, "{closed} {flat} {open}");
}
