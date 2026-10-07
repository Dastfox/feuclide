//! P3, rigid bodies: a body's configuration is one isometry, its motion a body-frame twist, its
//! momentum an element of the dual of the isometry group's Lie algebra.
//!
//! In a three-dimensional space of constant curvature `K` the isometry algebra has six
//! dimensions, spanned by the transvections `P_i` (along the reference frame's vectors) and the
//! rotations `J_i` about them, with the brackets
//!
//! ```text
//! [J_i, J_j] = ε_ijk J_k,   [J_i, P_j] = ε_ijk P_k,   [P_i, P_j] = K ε_ijk J_k
//! ```
//!
//! (`se(3)`, `so(4)` and `so(1, 3)` for `K` = 0, 1 and −1). Everything here works in those
//! coordinates, first the three `P`s then the three `J`s ([`algebra`] builds the element);
//! momenta and wrenches are their duals, paired by the dot product. This is exact in every
//! geometry the trait knows a constant curvature for: nothing is linearised.
//!
//! - The **inertia operator** `𝔤 → 𝔤*` is the kinetic energy of the mass distribution,
//!   `⟨I ξ, η⟩ = Σ m ⟨ξ(x), η(x)⟩_x`, where `ξ(x)` is the velocity the twist gives the body
//!   point `x` ([`velocity_at`]). A body is a set of weighted points in the normal coordinates
//!   of its frame ([`Inertia::cuboid`] samples a uniform box, weighted by the volume element of
//!   curved space). In curved space translation and rotation do not commute (`[P_i, P_j]`), so
//!   a body that moves and spins at once drifts sideways: physical, not a bug.
//! - The **Euler–Poincaré** equations `μ̇ = ad*_ξ μ`, `ξ = I⁻¹ μ`, `ġ = g ξ`, are integrated by
//!   the discrete Euler–Poincaré scheme on the exponential map ([`RigidBody::drift`]): find
//!   `ξ` with `(dexp⁻¹_{hξ})ᵀ I ξ = μ_k`, then `g ← g exp(hξ)` and
//!   `μ ← (dexp⁻¹_{−hξ})ᵀ I ξ`. It is variational, so symplectic: the energy oscillates
//!   without drifting, and the spatial momentum `Ad*_{g⁻¹} μ` ([`RigidBody::spatial_momentum`])
//!   is conserved to the solver's tolerance. The pose is renormalized after every step.
//! - Forces are kicks on the momentum ([`RigidBody::kick`], [`Inertia::gravity`]); resting
//!   contact is sequential impulses at points of the body against a [`DistanceField`]
//!   ([`RigidBody::rest_on`]), with Coulomb friction. What is exact, what is approximated and
//!   what is research-grade is in `doc/RIGID_BODIES.md`.

use std::marker::PhantomData;

use fk_geometry::{Algebra, Geometry, GroupElement};
use fk_math::Real;
use fk_math::nalgebra::{Matrix3, Matrix3x6, Matrix6, Vector3, Vector6};
use fk_math::series::{sinc, sinhc};

use crate::{DistanceField, Gravity, coords, tangent};

/// Newton iterations the implicit step takes at most.
const NEWTON: usize = 20;

/// The space's curvature.
///
/// # Panics
///
/// If the geometry has no constant curvature, or is not three-dimensional.
fn curvature<G: Geometry>() -> Real {
    assert_eq!(
        G::DIM,
        3,
        "rigid bodies are for three-dimensional geometries"
    );
    G::constant_curvature().expect("rigid bodies need a space of constant curvature")
}

/// The algebra element with coordinates `c`: `Σ c_i P_i + Σ c_{3+i} J_i`, where `P_i` is the
/// transvection along the reference frame's vector `i` and `J_i` the rotation about it.
pub fn algebra<G: Geometry>(c: &Vector6<Real>) -> Algebra<G> {
    let f = G::origin_frame;
    G::transvection(&tangent::<G>(&c.fixed_rows::<3>(0).into_owned()))
        + G::rotation(&f(1), &f(2)) * c[3]
        + G::rotation(&f(2), &f(0)) * c[4]
        + G::rotation(&f(0), &f(1)) * c[5]
}

/// The coordinates of the transvection by `v` (reference-frame coordinates).
pub fn translation(v: &Vector3<Real>) -> Vector6<Real> {
    Vector6::new(v.x, v.y, v.z, 0.0, 0.0, 0.0)
}

/// The coordinates of the rotation with axis and angle `w`.
pub fn spin(w: &Vector3<Real>) -> Vector6<Real> {
    Vector6::new(0.0, 0.0, 0.0, w.x, w.y, w.z)
}

/// The matrix of `ad_x = [x, ·]` in algebra coordinates.
pub fn ad<G: Geometry>(x: &Vector6<Real>) -> Matrix6<Real> {
    let k = curvature::<G>();
    let v = x.fixed_rows::<3>(0).into_owned().cross_matrix();
    let w = x.fixed_rows::<3>(3).into_owned().cross_matrix();
    let mut m = Matrix6::zeros();
    m.fixed_view_mut::<3, 3>(0, 0).copy_from(&w);
    m.fixed_view_mut::<3, 3>(0, 3).copy_from(&v);
    m.fixed_view_mut::<3, 3>(3, 0).copy_from(&(v * k));
    m.fixed_view_mut::<3, 3>(3, 3).copy_from(&w);
    m
}

/// The exponential of a matrix, by scaling and squaring a Taylor series.
fn expm(a: &Matrix6<Real>) -> Matrix6<Real> {
    let norm = a.norm();
    let squarings = if norm > 0.5 {
        (norm / 0.5).log2().ceil() as i32
    } else {
        0
    };
    let b = a / Real::powi(2.0, squarings);
    let (mut sum, mut term) = (Matrix6::identity(), Matrix6::identity());
    for n in 1..=18 {
        term = term * b / n as Real;
        sum += term;
    }
    for _ in 0..squarings {
        sum = sum * sum;
    }
    sum
}

/// The left-trivialised derivative of the exponential, `dexp_A = Σ Aⁿ / (n + 1)!`, of the
/// matrix `A` of some `ad_x`: `exp(x + ε y) = exp(x) exp(ε dexp_{−x} y) + O(ε²)`.
fn dexp(a: &Matrix6<Real>) -> Matrix6<Real> {
    let (mut sum, mut term) = (Matrix6::identity(), Matrix6::identity());
    for n in 1..64 {
        term = term * a / (n + 1) as Real;
        sum += term;
        if term.norm() < 1e-17 * sum.norm() {
            break;
        }
    }
    sum
}

/// The matrix of `Ad_g` in algebra coordinates: `g exp(ξ) g⁻¹ = exp(Ad_g ξ)`.
///
/// # Panics
///
/// If `g` takes the origin onto its cut locus (the antipode, in S³).
pub fn adjoint<G: Geometry>(g: &G::Isometry) -> Matrix6<Real> {
    let origin = G::origin();
    let r = G::log(&origin, &G::apply(g, &origin)).expect("the antipode has no adjoint here");
    let turn = G::Isometry::exp(&G::transvection(&r)).inverse().compose(g);
    let columns: Vec<Vector3<Real>> = (0..3)
        .map(|j| coords::<G>(&G::apply_tangent(&turn, &G::origin_frame(j))))
        .collect();
    let m = Matrix3::from_columns(&columns);
    let mut rotation = Matrix6::zeros();
    rotation.fixed_view_mut::<3, 3>(0, 0).copy_from(&m);
    rotation.fixed_view_mut::<3, 3>(3, 3).copy_from(&m);
    expm(&ad::<G>(&translation(&coords::<G>(&r)))) * rotation
}

/// The velocity a twist with coordinates `c` gives the body point at normal coordinates `x`
/// is `velocity_at(x) · c`, in the reference frame carried to that point from the origin along
/// the geodesic (`exp(transvection(x))`, which is orthonormal there).
pub fn velocity_at<G: Geometry>(x: &Vector3<Real>) -> Matrix3x6<Real> {
    // exp(tξ) T o = T exp(t Ad_{T⁻¹} ξ) o: the translational part of Ad_{T⁻¹} ξ, carried by T.
    expm(&-ad::<G>(&translation(x)))
        .fixed_rows::<3>(0)
        .into_owned()
}

/// The body point at normal coordinates `x`, and the isometry carrying the origin's frame there.
fn frame_at<G: Geometry>(x: &Vector3<Real>) -> G::Isometry {
    G::Isometry::exp(&G::transvection(&tangent::<G>(x)))
}

/// A body's mass distribution and the inertia operator it makes: `𝔤 → 𝔤*`, a symmetric 6×6
/// matrix in algebra coordinates, and its inverse.
#[derive(Debug)]
pub struct Inertia<G: Geometry> {
    /// Masses at normal coordinates of the body's frame.
    pub points: Vec<(Real, Vector3<Real>)>,
    /// The operator: momentum `I · ξ` of the twist `ξ`.
    pub matrix: Matrix6<Real>,
    /// Its inverse: the twist of a momentum.
    pub inverse: Matrix6<Real>,
    geometry: PhantomData<G>,
}

impl<G: Geometry> Clone for Inertia<G> {
    fn clone(&self) -> Self {
        Self {
            points: self.points.clone(),
            matrix: self.matrix,
            inverse: self.inverse,
            geometry: PhantomData,
        }
    }
}

impl<G: Geometry> Inertia<G> {
    /// The inertia of point masses at normal coordinates of the body's frame.
    ///
    /// # Panics
    ///
    /// If the masses do not span every direction of motion (fewer than three points off a
    /// line), so that the operator has no inverse.
    pub fn of_points(points: Vec<(Real, Vector3<Real>)>) -> Self {
        let matrix = points.iter().fold(Matrix6::zeros(), |sum, (m, x)| {
            let b = velocity_at::<G>(x);
            sum + b.transpose() * b * *m
        });
        let inverse = matrix
            .try_inverse()
            .expect("the masses span every direction of motion");
        Self {
            points,
            matrix,
            inverse,
            geometry: PhantomData,
        }
    }

    /// A uniform box of `mass` with half-sides `half`, the box in the normal coordinates of the
    /// body's frame: sampled at 4×4×4 Gauss–Legendre points, each weighted by the volume
    /// element of the space there (`(sn_K r / r)²`), so that the density is uniform in the
    /// space's own volume. Exact for a box in E³.
    pub fn cuboid(mass: Real, half: Vector3<Real>) -> Self {
        const NODES: [(Real, Real); 4] = [
            (-0.861_136_311_594_052_6, 0.347_854_845_137_453_9),
            (-0.339_981_043_584_856_3, 0.652_145_154_862_546_1),
            (0.339_981_043_584_856_3, 0.652_145_154_862_546_1),
            (0.861_136_311_594_052_6, 0.347_854_845_137_453_9),
        ];
        let k = curvature::<G>();
        let volume = |r: Real| match k {
            k if k > 0.0 => sinc(k.sqrt() * r).powi(2),
            k if k < 0.0 => sinhc((-k).sqrt() * r).powi(2),
            _ => 1.0,
        };
        let mut points = Vec::with_capacity(64);
        for &(a, wa) in &NODES {
            for &(b, wb) in &NODES {
                for &(c, wc) in &NODES {
                    let x = Vector3::new(a * half.x, b * half.y, c * half.z);
                    points.push((wa * wb * wc * volume(x.norm()), x));
                }
            }
        }
        let total: Real = points.iter().map(|(m, _)| m).sum();
        for (m, _) in &mut points {
            *m *= mass / total;
        }
        Self::of_points(points)
    }

    /// The body's mass.
    pub fn mass(&self) -> Real {
        self.points.iter().map(|(m, _)| m).sum()
    }

    /// The wrench (body-frame algebra dual coordinates) of `gravity` on the body at `pose`:
    /// each mass pulled along down where it is.
    pub fn gravity(&self, pose: &G::Isometry, gravity: &Gravity<G>) -> Vector6<Real> {
        let origin = G::origin();
        let back = pose.inverse();
        self.points.iter().fold(Vector6::zeros(), |sum, (m, x)| {
            let frame = frame_at::<G>(x);
            let at = pose.compose(&frame);
            let here = G::apply(&at, &origin);
            let down = G::parallel_transport(&origin, &here, &gravity.down);
            // Down at the mass, in the frame carried to it from the body's origin.
            let local = coords::<G>(&G::apply_tangent(
                &frame.inverse(),
                &G::apply_tangent(&back, &down),
            ));
            sum + velocity_at::<G>(x).transpose() * local * (m * gravity.acceleration)
        })
    }
}

/// A rigid body's state: where it is and its momentum, both in its own frame.
#[derive(Debug)]
pub struct RigidBody<G: Geometry> {
    /// Its pose.
    pub pose: G::Isometry,
    /// Its momentum in the body frame (the dual of the algebra): `I ξ` for the twist `ξ`, to
    /// first order in the step.
    pub momentum: Vector6<Real>,
}

impl<G: Geometry> Clone for RigidBody<G> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<G: Geometry> Copy for RigidBody<G> {}

/// The tolerance of the solve that keeps contact: what the bodies may sink into a surface
/// without being pushed out, and how much of the rest they are pushed out of per step.
const SLOP: Real = 1e-3;
const PUSH: Real = 0.2;

impl<G: Geometry> RigidBody<G> {
    /// A body at `pose` moving with the body-frame twist `twist` (algebra coordinates).
    pub fn moving(pose: G::Isometry, twist: &Vector6<Real>, inertia: &Inertia<G>) -> Self {
        Self {
            pose,
            momentum: inertia.matrix * twist,
        }
    }

    /// Its twist, `I⁻¹ μ` (to first order in the step).
    pub fn twist(&self, inertia: &Inertia<G>) -> Vector6<Real> {
        inertia.inverse * self.momentum
    }

    /// Its kinetic energy, `½ ⟨μ, I⁻¹ μ⟩`.
    pub fn energy(&self, inertia: &Inertia<G>) -> Real {
        0.5 * self.momentum.dot(&(inertia.inverse * self.momentum))
    }

    /// Its momentum in the space's frame, `Ad*_{g⁻¹} μ`: conserved in free motion.
    pub fn spatial_momentum(&self) -> Vector6<Real> {
        adjoint::<G>(&self.pose.inverse()).transpose() * self.momentum
    }

    /// A wrench (body-frame algebra dual coordinates) acting for `dt`.
    pub fn kick(&mut self, wrench: &Vector6<Real>, dt: Real) {
        self.momentum += wrench * dt;
    }

    /// Free motion for `dt`: one step of the discrete Euler–Poincaré equations on the
    /// exponential map, the pose renormalized.
    pub fn drift(&mut self, inertia: &Inertia<G>, dt: Real) {
        let p = self.momentum;
        let residual =
            |xi: &Vector6<Real>| inertia.matrix * xi - dexp(&(ad::<G>(xi) * dt)).transpose() * p;
        let scale = p.norm().max(Real::MIN_POSITIVE);
        let mut xi = inertia.inverse * p;
        for _ in 0..NEWTON {
            let r = residual(&xi);
            if r.norm() <= 1e-14 * scale {
                break;
            }
            let h = 1e-7 * (1.0 + xi.norm());
            let jacobian = Matrix6::from_fn(|i, j| {
                let mut e = xi;
                e[j] += h;
                (residual(&e)[i] - r[i]) / h
            });
            match jacobian.lu().solve(&r) {
                Some(step) => xi -= step,
                None => break,
            }
        }
        self.pose = self
            .pose
            .compose(&G::Isometry::exp(&algebra::<G>(&(xi * dt))));
        self.pose.renormalize();
        self.momentum = dexp(&(ad::<G>(&xi) * -dt))
            .transpose()
            .lu()
            .solve(&(inertia.matrix * xi))
            .unwrap_or(self.momentum);
    }

    /// One step under `gravity` alone: half its kick, the drift, the other half at the new
    /// pose.
    pub fn fall(&mut self, inertia: &Inertia<G>, gravity: &Gravity<G>, dt: Real) {
        self.kick(&inertia.gravity(&self.pose, gravity), 0.5 * dt);
        self.drift(inertia, dt);
        self.kick(&inertia.gravity(&self.pose, gravity), 0.5 * dt);
    }

    /// Resting contact: impulses at the body points `touching` (normal coordinates of its
    /// frame) that come within reach of `surface` this step, so that none of them moves into it
    /// and those inside are pushed out over a few steps, with Coulomb friction `friction`.
    /// Call it after the forces' kicks and before [`drift`](Self::drift). Returns how deep the
    /// deepest point is.
    pub fn rest_on(
        &mut self,
        inertia: &Inertia<G>,
        touching: &[Vector3<Real>],
        surface: &(impl DistanceField<G> + ?Sized),
        friction: Real,
        dt: Real,
    ) -> Real {
        struct Touch {
            /// The wrench of a unit impulse along the normal, then along two tangents.
            w: [Vector6<Real>; 3],
            /// The twist each of them gives.
            give: [Vector6<Real>; 3],
            /// Their effective inverse masses.
            k: [Real; 3],
            /// The least normal velocity allowed.
            least: Real,
            /// The impulses so far.
            j: [Real; 3],
        }
        let origin = G::origin();
        let mut deepest: Real = 0.0;
        let mut touches = Vec::new();
        for x in touching {
            let at = self.pose.compose(&frame_at::<G>(x));
            let probe = |v: Vector3<Real>| {
                surface.distance(&G::apply(&at, &G::exp(&origin, &tangent::<G>(&v))))
            };
            let d = probe(Vector3::zeros());
            deepest = deepest.max(-d);
            // Points farther than the body could go this step are left alone.
            let speed = (velocity_at::<G>(x) * self.twist(inertia)).norm();
            if d > speed * dt + SLOP {
                continue;
            }
            let e = 1e-5;
            let n = Vector3::from_fn(|i, _| {
                let mut v = Vector3::zeros();
                v[i] = e;
                (probe(v) - probe(-v)) / (2.0 * e)
            })
            .try_normalize(1e-12);
            let Some(n) = n else { continue };
            let t1 = (if n.x.abs() < 0.9 {
                Vector3::x()
            } else {
                Vector3::y()
            })
            .cross(&n)
            .normalize();
            let t2 = n.cross(&t1);
            let b = velocity_at::<G>(x).transpose();
            let w = [b * n, b * t1, b * t2];
            let give = w.map(|w| inertia.inverse * w);
            let k = [0, 1, 2].map(|i| w[i].dot(&give[i]));
            let least = if d < -SLOP {
                PUSH * (-d - SLOP) / dt
            } else {
                -d.max(0.0) / dt
            };
            touches.push(Touch {
                w,
                give,
                k,
                least,
                j: [0.0; 3],
            });
        }
        let mut xi = self.twist(inertia);
        for _ in 0..24 {
            for t in &mut touches {
                let vn = t.w[0].dot(&xi);
                let jn = (t.j[0] + (t.least - vn) / t.k[0]).max(0.0);
                xi += t.give[0] * (jn - t.j[0]);
                t.j[0] = jn;
                for i in 1..3 {
                    let vt = t.w[i].dot(&xi);
                    let limit = friction * t.j[0];
                    let jt = (t.j[i] - vt / t.k[i]).clamp(-limit, limit);
                    xi += t.give[i] * (jt - t.j[i]);
                    t.j[i] = jt;
                }
            }
        }
        for t in &touches {
            self.momentum += t.w[0] * t.j[0] + t.w[1] * t.j[1] + t.w[2] * t.j[2];
        }
        deepest
    }
}
