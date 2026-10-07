//! Geometry-agnostic property tests.
//!
//! A [`Geometry`] implementation is correct when it passes this battery. Run it from a test file
//! of the implementing crate:
//!
//! ```ignore
//! fk_geometry::geometry_battery!(e3, fk_geometry_euclidean::E3);
//! ```
//!
//! Random samples are generated through the geometry's own API (frames at the origin,
//! transvections, rotations), so the battery needs nothing beyond the [`Geometry`] trait.
//! Isometries are compared by their action on `DIM + 1` points in general position, which
//! determines an isometry uniquely, so different representations of one group element (such as
//! `q` and `-q` for unit quaternions) compare equal.

use std::f64::consts::{PI, TAU};

use fk_math::Real;
use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::{Config, TestCaseError, TestRunner};

use crate::{Algebra, Geometry, GroupElement, VectorSpace};

pub use proptest;

/// Tuning knobs of the battery.
#[derive(Clone, Debug)]
pub struct BatteryConfig {
    /// Random cases per property.
    pub cases: u32,
    /// Sampled points lie within this distance of the origin, and sampled tangent vectors are at
    /// most this long. Must be below the injectivity radius.
    pub radius: Real,
    /// Absolute tolerance on quantities of order one.
    pub eps: Real,
    /// Number of compositions in the renormalization drift test.
    pub drift_steps: usize,
}

impl Default for BatteryConfig {
    fn default() -> Self {
        Self {
            cases: 256,
            radius: 2.0,
            eps: 1e-9,
            drift_steps: 1_000_000,
        }
    }
}

impl BatteryConfig {
    fn tol(&self, scale: Real) -> Real {
        self.eps * (1.0 + scale)
    }
}

/// Generates one `#[test]` per battery property, in a module called `$name`.
#[macro_export]
macro_rules! geometry_battery {
    ($name:ident, $geometry:ty) => {
        $crate::geometry_battery!($name, $geometry, $crate::battery::BatteryConfig::default());
    };
    ($name:ident, $geometry:ty, $config:expr) => {
        #[allow(unused_imports)]
        mod $name {
            use super::*;

            fn config() -> $crate::battery::BatteryConfig {
                $config
            }

            #[test]
            fn metric_axioms() {
                $crate::battery::metric_axioms::<$geometry>(&config());
            }

            #[test]
            fn exp_log() {
                $crate::battery::exp_log::<$geometry>(&config());
            }

            #[test]
            fn geodesic_speed() {
                $crate::battery::geodesic_speed::<$geometry>(&config());
            }

            #[test]
            fn isometry_invariance() {
                $crate::battery::isometry_invariance::<$geometry>(&config());
            }

            #[test]
            fn parallel_transport() {
                $crate::battery::parallel_transport::<$geometry>(&config());
            }

            #[test]
            fn group_axioms() {
                $crate::battery::group_axioms::<$geometry>(&config());
            }

            #[test]
            fn group_exp_log() {
                $crate::battery::group_exp_log::<$geometry>(&config());
            }

            #[test]
            fn adjoint() {
                $crate::battery::adjoint::<$geometry>(&config());
            }

            #[test]
            fn generators() {
                $crate::battery::generators::<$geometry>(&config());
            }

            #[test]
            fn holonomy() {
                $crate::battery::holonomy::<$geometry>(&config());
            }

            #[test]
            fn renormalization_drift() {
                $crate::battery::renormalization_drift::<$geometry>(&config());
            }
        }
    };
}

// ---------------------------------------------------------------------------------------------
// Properties
// ---------------------------------------------------------------------------------------------

/// `d(a, a) = 0`, `d(a, b) = d(b, a) ≥ 0`, `d(a, c) ≤ d(a, b) + d(b, c)`.
pub fn metric_axioms<G: Geometry>(cfg: &BatteryConfig) {
    let r = validate::<G>(cfg);
    let points = (point::<G>(r), point::<G>(r), point::<G>(r));
    check::<G, _>(cfg, "metric axioms", points, |(a, b, c)| {
        let aa = G::distance(&a, &a);
        let ab = G::distance(&a, &b);
        let ba = G::distance(&b, &a);
        let bc = G::distance(&b, &c);
        let ac = G::distance(&a, &c);
        prop_assert!(aa.abs() <= cfg.eps, "d(a, a) = {aa}");
        prop_assert!(ab >= 0.0, "d(a, b) = {ab} is negative");
        prop_assert!(
            (ab - ba).abs() <= cfg.tol(ab),
            "d(a, b) = {ab}, d(b, a) = {ba}"
        );
        prop_assert!(
            ac <= ab + bc + cfg.tol(ab + bc),
            "triangle inequality: d(a, c) = {ac} > d(a, b) + d(b, c) = {}",
            ab + bc
        );
        Ok(())
    });
}

/// `log(p, exp(p, v)) = v` and `d(p, exp(p, v)) = |v|` within the injectivity radius, and
/// `exp(p, log(p, q)) = q` for points close enough together.
pub fn exp_log<G: Geometry>(cfg: &BatteryConfig) {
    let r = validate::<G>(cfg);
    check::<G, _>(cfg, "log ∘ exp", pointed::<G>(r), |(p, v)| {
        let q = G::exp(&p, &v);
        let len = G::norm(&p, &v);
        prop_assert!(G::point_residual(&q) <= cfg.tol(len), "exp left the model");
        let Some(back) = G::log(&p, &q) else {
            return Err(TestCaseError::fail(
                "log is None inside the injectivity radius",
            ));
        };
        let gap = tangent_gap::<G>(&p, &back, &v);
        prop_assert!(
            gap <= cfg.tol(len),
            "log(p, exp(p, v)) is {gap} away from v"
        );
        prop_assert!(
            G::tangent_residual(&p, &back) <= cfg.tol(len),
            "log is not tangent"
        );
        let d = G::distance(&p, &q);
        prop_assert!(
            (d - len).abs() <= cfg.tol(len),
            "d(p, exp(p, v)) = {d}, |v| = {len}"
        );
        Ok(())
    });

    if 2.0 * r < G::injectivity_radius() {
        check::<G, _>(
            cfg,
            "exp ∘ log",
            (point::<G>(r), point::<G>(r)),
            |(p, q)| {
                let Some(v) = G::log(&p, &q) else {
                    return Err(TestCaseError::fail(
                        "log is None inside the injectivity radius",
                    ));
                };
                let d = G::distance(&p, &q);
                let gap = G::distance(&G::exp(&p, &v), &q);
                prop_assert!(gap <= cfg.tol(d), "exp(p, log(p, q)) is {gap} away from q");
                Ok(())
            },
        );
    }
}

/// Geodesics have constant speed and minimize: `d(exp(p, s·v), exp(p, t·v)) = |s - t| |v|`.
pub fn geodesic_speed<G: Geometry>(cfg: &BatteryConfig) {
    let r = validate::<G>(cfg);
    let input = (pointed::<G>(r), 0.0..1.0, 0.0..1.0);
    check::<G, _>(cfg, "geodesic speed", input, |((p, v), s, t)| {
        let a = G::exp(&p, &(v * s));
        let b = G::exp(&p, &(v * t));
        let expected = (s - t).abs() * G::norm(&p, &v);
        let d = G::distance(&a, &b);
        prop_assert!(
            (d - expected).abs() <= cfg.tol(expected),
            "d = {d}, expected {expected}"
        );
        Ok(())
    });
}

/// Isometries preserve distances and inner products, and commute with `exp`.
pub fn isometry_invariance<G: Geometry>(cfg: &BatteryConfig) {
    let r = validate::<G>(cfg);
    let input = (isometry::<G>(r), pointed2::<G>(r));
    check::<G, _>(cfg, "isometry invariance", input, |(g, (p, u, v))| {
        let q = G::exp(&p, &v);
        let (gp, gq) = (G::apply(&g, &p), G::apply(&g, &q));
        let (gu, gv) = (G::apply_tangent(&g, &u), G::apply_tangent(&g, &v));

        prop_assert!(G::point_residual(&gp) <= cfg.tol(r), "g·p left the model");
        prop_assert!(
            G::tangent_residual(&gp, &gv) <= cfg.tol(r),
            "dg·v is not tangent"
        );

        let (d, gd) = (G::distance(&p, &q), G::distance(&gp, &gq));
        prop_assert!(
            (d - gd).abs() <= cfg.tol(d),
            "d(p, q) = {d}, d(gp, gq) = {gd}"
        );

        let (i, gi) = (G::inner(&p, &u, &v), G::inner(&gp, &gu, &gv));
        prop_assert!(
            (i - gi).abs() <= cfg.tol(i.abs()),
            "⟨u, v⟩ = {i}, ⟨gu, gv⟩ = {gi}"
        );

        let gap = G::distance(&gq, &G::exp(&gp, &gv));
        prop_assert!(
            gap <= cfg.tol(d),
            "g·exp(p, v) is {gap} away from exp(gp, dg·v)"
        );
        Ok(())
    });
}

/// Parallel transport is an isometry between tangent spaces, is undone by transporting back, and
/// carries the geodesic's own velocity to the geodesic's velocity.
pub fn parallel_transport<G: Geometry>(cfg: &BatteryConfig) {
    let r = validate::<G>(cfg);
    check::<G, _>(
        cfg,
        "parallel transport",
        pointed3::<G>(r),
        |(p, u, v, w)| {
            let q = G::exp(&p, &w);
            let (tu, tv) = (
                G::parallel_transport(&p, &q, &u),
                G::parallel_transport(&p, &q, &v),
            );
            prop_assert!(
                G::tangent_residual(&q, &tu) <= cfg.tol(r),
                "transport is not tangent"
            );

            let (i, ti) = (G::inner(&p, &u, &v), G::inner(&q, &tu, &tv));
            prop_assert!(
                (i - ti).abs() <= cfg.tol(i.abs()),
                "⟨u, v⟩ = {i}, ⟨Pu, Pv⟩ = {ti}"
            );

            let back = G::parallel_transport(&q, &p, &tu);
            let gap = tangent_gap::<G>(&p, &back, &u);
            prop_assert!(
                gap <= cfg.tol(r),
                "transporting there and back is off by {gap}"
            );

            let Some(w_back) = G::log(&q, &p) else {
                return Err(TestCaseError::fail(
                    "log is None inside the injectivity radius",
                ));
            };
            let tw = G::parallel_transport(&p, &q, &w);
            let gap = tangent_gap::<G>(&q, &tw, &-w_back);
            prop_assert!(
                gap <= cfg.tol(r),
                "the geodesic velocity is not transported to itself"
            );
            Ok(())
        },
    );
}

/// Composition is associative, has an identity and inverses, and matches the action on points
/// and tangent vectors.
pub fn group_axioms<G: Geometry>(cfg: &BatteryConfig) {
    let r = validate::<G>(cfg);
    let input = (
        isometry::<G>(r),
        isometry::<G>(r),
        isometry::<G>(r),
        pointed::<G>(r),
    );
    check::<G, _>(cfg, "group axioms", input, |(g, h, k, (x, v))| {
        let id = G::Isometry::identity();
        let gh = g.compose(&h);
        let tol = cfg.tol(r);

        let gap = G::distance(&G::apply(&gh, &x), &G::apply(&g, &G::apply(&h, &x)));
        prop_assert!(gap <= tol, "(g∘h)·x is {gap} away from g·(h·x)");

        let ghx = G::apply(&gh, &x);
        let gap = tangent_gap::<G>(
            &ghx,
            &G::apply_tangent(&gh, &v),
            &G::apply_tangent(&g, &G::apply_tangent(&h, &v)),
        );
        prop_assert!(gap <= tol, "d(g∘h)·v is {gap} away from dg·(dh·v)");

        prop_assert!(iso_gap::<G>(&g.compose(&id), &g) <= tol, "g∘id ≠ g");
        prop_assert!(iso_gap::<G>(&id.compose(&g), &g) <= tol, "id∘g ≠ g");
        prop_assert!(
            iso_gap::<G>(&g.compose(&g.inverse()), &id) <= tol,
            "g∘g⁻¹ ≠ id"
        );
        prop_assert!(
            iso_gap::<G>(&g.inverse().compose(&g), &id) <= tol,
            "g⁻¹∘g ≠ id"
        );

        let gap = iso_gap::<G>(&gh.compose(&k), &g.compose(&h.compose(&k)));
        prop_assert!(gap <= tol, "composition is not associative (off by {gap})");
        prop_assert!(gh.residual() <= cfg.eps, "composition left the group");
        Ok(())
    });
}

/// The group exponential is a one-parameter subgroup and the logarithm inverts it.
pub fn group_exp_log<G: Geometry>(cfg: &BatteryConfig) {
    let r = validate::<G>(cfg);
    let input = (isometry::<G>(r), algebra::<G>(r), -1.0..1.0, -1.0..1.0);
    check::<G, _>(cfg, "group exp/log", input, |(g, xi, s, t)| {
        let tol = cfg.tol(r);
        let exp = |a: Algebra<G>| G::Isometry::exp(&a);

        let gap = iso_gap::<G>(&exp(g.log()), &g);
        prop_assert!(gap <= tol, "exp(log g) is {gap} away from g");

        let gap = iso_gap::<G>(&exp(xi).compose(&exp(-xi)), &G::Isometry::identity());
        prop_assert!(gap <= tol, "exp(ξ)∘exp(-ξ) is {gap} away from id");

        let gap = iso_gap::<G>(&exp(xi * (s + t)), &exp(xi * s).compose(&exp(xi * t)));
        prop_assert!(gap <= tol, "exp((s+t)ξ) is {gap} away from exp(sξ)∘exp(tξ)");

        let gap = (exp(xi).log() - xi).coord_norm_squared().sqrt();
        prop_assert!(gap <= tol, "log(exp ξ) is {gap} away from ξ");
        Ok(())
    });
}

/// `exp(Ad_g ξ) = g ∘ exp(ξ) ∘ g⁻¹`.
pub fn adjoint<G: Geometry>(cfg: &BatteryConfig) {
    let r = validate::<G>(cfg);
    let input = (isometry::<G>(r), algebra::<G>(r));
    check::<G, _>(cfg, "adjoint", input, |(g, xi)| {
        let lhs = G::Isometry::exp(&g.adjoint(&xi));
        let rhs = g.compose(&G::Isometry::exp(&xi)).compose(&g.inverse());
        let gap = iso_gap::<G>(&lhs, &rhs);
        prop_assert!(
            gap <= cfg.tol(r),
            "exp(Ad_g ξ) is {gap} away from g∘exp(ξ)∘g⁻¹"
        );
        Ok(())
    });
}

/// Transvections and rotations do what [`Geometry::transvection`] and [`Geometry::rotation`]
/// promise.
pub fn generators<G: Geometry>(cfg: &BatteryConfig) {
    let r = validate::<G>(cfg);
    let o = G::origin();
    let input = (origin_tangent::<G>(r), origin_tangent::<G>(r));
    check::<G, _>(cfg, "transvections", input, |(v, u)| {
        let tol = cfg.tol(r);
        let t = G::Isometry::exp(&G::transvection(&v));
        let q = G::exp(&o, &v);

        let gap = G::distance(&G::apply(&t, &o), &q);
        prop_assert!(
            gap <= tol,
            "exp(transvection(v))·o is {gap} away from exp(o, v)"
        );

        let gap = tangent_gap::<G>(
            &q,
            &G::apply_tangent(&t, &u),
            &G::parallel_transport(&o, &q, &u),
        );
        prop_assert!(
            gap <= tol,
            "a transvection does not parallel-transport (off by {gap})"
        );

        let Some(to_q) = G::transvection_to(&q) else {
            return Err(TestCaseError::fail(
                "transvection_to is None inside the radius",
            ));
        };
        prop_assert!(
            iso_gap::<G>(&to_q, &t) <= tol,
            "transvection_to(exp(o, v)) ≠ exp(v)"
        );
        Ok(())
    });

    if G::DIM < 2 {
        return;
    }
    let planes: Vec<_> = planes::<G>().collect();
    let input = (0..planes.len(), -PI..PI);
    check::<G, _>(cfg, "rotations", input, |(k, theta)| {
        let (i, j) = planes[k];
        let (ei, ej) = (G::origin_frame(i), G::origin_frame(j));
        let rot = G::Isometry::exp(&(G::rotation(&ei, &ej) * theta));

        let gap = G::distance(&G::apply(&rot, &o), &o);
        prop_assert!(
            gap <= cfg.eps,
            "rotation about the origin moves it by {gap}"
        );

        let (c, s) = (theta.cos(), theta.sin());
        let gap = tangent_gap::<G>(&o, &G::apply_tangent(&rot, &ei), &(ei * c + ej * s));
        prop_assert!(
            gap <= cfg.eps,
            "e{i} is not rotated by θ towards e{j} (off by {gap})"
        );
        let gap = tangent_gap::<G>(&o, &G::apply_tangent(&rot, &ej), &(ej * c - ei * s));
        prop_assert!(
            gap <= cfg.eps,
            "e{j} is not rotated by θ away from e{i} (off by {gap})"
        );

        for l in (0..G::DIM).filter(|&l| l != i && l != j) {
            let el = G::origin_frame(l);
            let gap = tangent_gap::<G>(&o, &G::apply_tangent(&rot, &el), &el);
            prop_assert!(
                gap <= cfg.eps,
                "e{l} is outside the plane but moved by {gap}"
            );
        }
        Ok(())
    });
}

/// Gauss–Bonnet for constant curvature `K`: transporting a vector around a geodesic triangle
/// rotates it by the angle excess `α + β + γ - π = K · area`, whose sign is the sign of `K`.
///
/// This single test catches most sign errors in curvature conventions. Skipped in dimension 1
/// and for geometries without constant curvature.
pub fn holonomy<G: Geometry>(cfg: &BatteryConfig) {
    let r = validate::<G>(cfg);
    let Some(k) = G::constant_curvature() else {
        return;
    };
    if G::DIM < 2 {
        return;
    }
    let input = (point::<G>(r), point::<G>(r), point::<G>(r));
    check::<G, _>(cfg, "holonomy", input, |(p, q, s)| {
        let logs = (G::log(&p, &q), G::log(&p, &s), G::log(&q, &s));
        let (Some(pq), Some(ps), Some(qs)) = logs else {
            return Err(TestCaseError::reject("vertex on a cut locus"));
        };
        let (Some(qp), Some(sp), Some(sq)) = (G::log(&q, &p), G::log(&s, &p), G::log(&s, &q))
        else {
            return Err(TestCaseError::reject("vertex on a cut locus"));
        };
        let sides = [G::norm(&p, &pq), G::norm(&p, &ps), G::norm(&q, &qs)];
        prop_assume!(sides.iter().all(|&l| l > 0.1));
        let angles = [
            angle::<G>(&p, &pq, &ps),
            angle::<G>(&q, &qp, &qs),
            angle::<G>(&s, &sp, &sq),
        ];
        prop_assume!(angles.iter().all(|&a| a > 0.05 && a < PI - 0.05));

        let excess: Real = angles.iter().sum::<Real>() - PI;
        let tol = 10.0 * cfg.tol(r);
        if k == 0.0 {
            prop_assert!(
                excess.abs() <= tol,
                "flat geometry with angle excess {excess}"
            );
        } else {
            prop_assert!(
                excess * k.signum() >= -tol,
                "curvature {k} but angle excess {excess} has the opposite sign"
            );
        }

        let start = pq * (1.0 / sides[0]);
        let around = G::parallel_transport(
            &s,
            &p,
            &G::parallel_transport(&q, &s, &G::parallel_transport(&p, &q, &start)),
        );
        let turned = angle::<G>(&p, &start, &around);
        let expected = {
            let e = excess.abs() % TAU;
            e.min(TAU - e)
        };
        prop_assert!(
            (turned - expected).abs() <= tol,
            "holonomy {turned}, angle excess {expected}"
        );
        Ok(())
    });
}

/// Composing many small steps and renormalizing after each one, as an integrator does, stays on
/// the group. The steps come in cycles that multiply to the identity, so any accumulated error
/// is visible.
pub fn renormalization_drift<G: Geometry>(cfg: &BatteryConfig) {
    let r = validate::<G>(cfg);
    let mut runner = TestRunner::deterministic();
    let strategy = algebra::<G>(r);
    let steps: Vec<G::Isometry> = (0..16)
        .map(|_| {
            let xi = strategy
                .new_tree(&mut runner)
                .expect("sampling cannot be rejected")
                .current();
            G::Isometry::exp(&(xi * 0.05))
        })
        .collect();
    let cycle: Vec<_> = steps
        .iter()
        .copied()
        .chain(steps.iter().rev().map(|step| step.inverse()))
        .collect();

    let total = cfg.drift_steps.div_ceil(cycle.len()) * cycle.len();
    let mut g = G::Isometry::identity();
    let mut worst: Real = 0.0;
    for step in cycle.iter().cycle().take(total) {
        g = g.compose(step);
        g.renormalize();
        worst = worst.max(g.residual());
    }
    assert!(
        worst <= cfg.eps,
        "{}: group residual reached {worst} over {total} renormalized steps",
        G::NAME
    );
    let gap = iso_gap::<G>(&g, &G::Isometry::identity());
    assert!(
        gap <= 1e3 * cfg.eps,
        "{}: {total} steps that multiply to the identity drifted by {gap}",
        G::NAME
    );
}

// ---------------------------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------------------------

fn validate<G: Geometry>(cfg: &BatteryConfig) -> Real {
    assert!(
        cfg.radius < G::injectivity_radius(),
        "{}: battery radius {} must be below the injectivity radius {}",
        G::NAME,
        cfg.radius,
        G::injectivity_radius()
    );
    cfg.radius
}

fn check<G: Geometry, S: Strategy>(
    cfg: &BatteryConfig,
    property: &str,
    strategy: S,
    test: impl Fn(S::Value) -> Result<(), TestCaseError>,
) {
    let mut runner = TestRunner::new(Config {
        cases: cfg.cases,
        failure_persistence: None,
        ..Config::default()
    });
    if let Err(err) = runner.run(&strategy, test) {
        panic!("{}: {property}: {err}", G::NAME);
    }
}

fn planes<G: Geometry>() -> impl Iterator<Item = (usize, usize)> {
    (0..G::DIM).flat_map(|i| (i + 1..G::DIM).map(move |j| (i, j)))
}

/// A tangent vector at the origin of length at most `radius`.
fn origin_tangent<G: Geometry>(radius: Real) -> impl Strategy<Value = G::Tangent> {
    let scale = radius / (G::DIM as Real).sqrt();
    prop::collection::vec(-1.0..1.0, G::DIM).prop_map(move |coeffs| {
        coeffs
            .iter()
            .enumerate()
            .fold(G::Tangent::zero(), |acc, (i, c)| {
                acc + G::origin_frame(i) * (c * scale)
            })
    })
}

/// A rotation generator about the origin with every plane angle below `max_angle`.
fn rotation_algebra<G: Geometry>(max_angle: Real) -> impl Strategy<Value = Algebra<G>> {
    let count = planes::<G>().count();
    prop::collection::vec(-max_angle..max_angle, count).prop_map(|angles| {
        planes::<G>()
            .zip(angles)
            .fold(Algebra::<G>::zero(), |acc, ((i, j), a)| {
                acc + G::rotation(&G::origin_frame(i), &G::origin_frame(j)) * a
            })
    })
}

/// A Lie algebra element small enough for the principal logarithm to invert its exponential.
fn algebra<G: Geometry>(radius: Real) -> impl Strategy<Value = Algebra<G>> {
    (origin_tangent::<G>(radius), rotation_algebra::<G>(0.5))
        .prop_map(|(v, rot)| G::transvection(&v) + rot)
}

/// An isometry moving the origin by at most `radius`, with an arbitrary rotation.
fn isometry<G: Geometry>(radius: Real) -> impl Strategy<Value = G::Isometry> {
    let count = planes::<G>().count();
    (
        origin_tangent::<G>(radius),
        prop::collection::vec(-PI..PI, count),
    )
        .prop_map(|(v, angles)| {
            planes::<G>().zip(angles).fold(
                G::Isometry::exp(&G::transvection(&v)),
                |g, ((i, j), a)| {
                    let generator = G::rotation(&G::origin_frame(i), &G::origin_frame(j));
                    g.compose(&G::Isometry::exp(&(generator * a)))
                },
            )
        })
}

/// A point within `radius` of the origin.
fn point<G: Geometry>(radius: Real) -> impl Strategy<Value = G::Point> {
    origin_tangent::<G>(radius).prop_map(|v| G::exp(&G::origin(), &v))
}

/// A point within `radius` of the origin with a tangent vector there of length at most `radius`.
fn pointed<G: Geometry>(radius: Real) -> impl Strategy<Value = (G::Point, G::Tangent)> {
    (isometry::<G>(radius), origin_tangent::<G>(radius))
        .prop_map(|(g, v)| (G::apply(&g, &G::origin()), G::apply_tangent(&g, &v)))
}

fn pointed2<G: Geometry>(
    radius: Real,
) -> impl Strategy<Value = (G::Point, G::Tangent, G::Tangent)> {
    (
        isometry::<G>(radius),
        origin_tangent::<G>(radius),
        origin_tangent::<G>(radius),
    )
        .prop_map(|(g, u, v)| {
            (
                G::apply(&g, &G::origin()),
                G::apply_tangent(&g, &u),
                G::apply_tangent(&g, &v),
            )
        })
}

fn pointed3<G: Geometry>(
    radius: Real,
) -> impl Strategy<Value = (G::Point, G::Tangent, G::Tangent, G::Tangent)> {
    (
        isometry::<G>(radius),
        origin_tangent::<G>(radius),
        origin_tangent::<G>(radius),
        origin_tangent::<G>(radius),
    )
        .prop_map(|(g, u, v, w)| {
            (
                G::apply(&g, &G::origin()),
                G::apply_tangent(&g, &u),
                G::apply_tangent(&g, &v),
                G::apply_tangent(&g, &w),
            )
        })
}

// ---------------------------------------------------------------------------------------------
// Comparisons
// ---------------------------------------------------------------------------------------------

fn tangent_gap<G: Geometry>(p: &G::Point, u: &G::Tangent, v: &G::Tangent) -> Real {
    G::norm(p, &(*u - *v))
}

/// The largest displacement between `g` and `h` on `DIM + 1` points in general position.
fn iso_gap<G: Geometry>(g: &G::Isometry, h: &G::Isometry) -> Real {
    let o = G::origin();
    (0..G::DIM)
        .map(|i| G::exp(&o, &(G::origin_frame(i) * 0.5)))
        .chain(std::iter::once(o))
        .map(|x| G::distance(&G::apply(g, &x), &G::apply(h, &x)))
        .fold(0.0, Real::max)
}

/// The angle between two tangent vectors at `p`, accurate near `0` and `π`.
fn angle<G: Geometry>(p: &G::Point, u: &G::Tangent, v: &G::Tangent) -> Real {
    let a = *u * (1.0 / G::norm(p, u));
    let b = *v * (1.0 / G::norm(p, v));
    2.0 * G::norm(p, &(a - b)).atan2(G::norm(p, &(a + b)))
}
