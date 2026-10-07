# Rigid bodies in curved space: what is exact, what is approximated

[PLAN.md](PLAN.md)'s P3. The code is `fk_physics::rigid` and
`fk_scene::FloatingOrigin`, and the tests are `crates/fk-physics/tests/rigid.rs` and
`fk-scene`'s floating-origin test. This page records how far each piece can be trusted, and
which parts are research-grade.

## The model

A body's configuration is one isometry `g`, its pose. Its velocity is a body-frame twist
`ξ ∈ 𝔤`, and its momentum is `μ ∈ 𝔤*`. In a three-dimensional space of constant curvature `K`
the algebra has six dimensions: three transvections `P_i` and three rotations `J_i`, with

```text
[J_i, J_j] = ε_ijk J_k,   [J_i, P_j] = ε_ijk P_k,   [P_i, P_j] = K ε_ijk J_k
```

So E³, H³ and S³ share one implementation, written in these coordinates. Only the curvature
changes between them.

## Exact (to rounding)

- **The algebra.** The brackets, `ad`, `Ad_g` (a transvection's `exp(ad)` composed with the
  rotation that is left over) and the velocity a twist gives each body point. Two tests check
  them against the group itself in all three geometries: the coordinate `Ad_g` must reproduce
  `g exp(ξ) g⁻¹` to 1e-10, and velocities must match finite differences of the flow.
- **The inertia operator** of a set of point masses: `⟨Iξ, η⟩ = Σ m ⟨ξ(x), η(x)⟩`, using the
  space's own metric at each point.
- **Spatial momentum.** `Ad*_{g⁻¹} μ` is conserved by the integrator's construction, up to the
  Newton tolerance: within 3e-14 over fifty seconds in every geometry.

## Approximated, with known error

- **The box's mass distribution.** `Inertia::cuboid` is a box in the normal coordinates of the
  body's frame, not a polyhedron with geodesic faces. Its uniform density is sampled at 4×4×4
  Gauss–Legendre points, weighted by the volume element `(sn_K r / r)²`. In E³ this is the
  textbook box exactly. In H³ and S³ (half-sides 0.4 × 0.25 × 0.1) it agrees with a 60³
  midpoint grid to 2e-5, about the grid's own error.
- **The integrator.** It is the discrete Euler–Poincaré scheme on the exponential map:
  `(dexp⁻¹_{hξ})ᵀ Iξ = μ_k`, then `g ← g exp(hξ)` and `μ ← (dexp⁻¹_{−hξ})ᵀ Iξ`. It is
  variational and therefore symplectic, so its energy error is an `O(h²)` oscillation that
  does not drift: within 5e-5 over fifty seconds at 120 Hz for a box spinning about no
  principal axis. The implicit equation is solved by Newton with a finite-difference Jacobian,
  and `dexp` is summed as a series, which suits steps where `h|ξ|` is small. A body turning a
  radian per step would need a look at both.
- **Forces.** They are kicks on the momentum, with gravity split symmetrically about the free
  step (`RigidBody::fall`). Gravity is `fk_physics::Gravity`: "down" at the origin, carried to
  each mass by parallel transport. In flat space that is a uniform field. In curved space it
  has not been shown to derive from a potential, so total energy under gravity is not claimed
  to be conserved, and no test relies on it.
- **The velocity seen by contact.** Contact reads the twist as `I⁻¹ μ`, which is first order
  in the step. The integrator's own `ξ` differs by `O(h)`.

## Research-grade

- **Resting contact** (`RigidBody::rest_on`) uses sequential impulses at chosen body points
  against a `DistanceField`. Each point gets a normal from the field's gradient, taken in the
  frame carried to that point. Friction is applied along two tangents per point, a pyramid
  standing in for the Coulomb cone. There is no restitution. A point that has sunk in is pushed
  out a fifth of the depth per step, past a millimetre of slop, and a point is considered only
  if the body could reach the surface within the step. There is no contact manifold: a box
  touches only at its corners. There is no body-against-body contact, and nothing is promised
  for stacks. Shown to work: a box dropped tilted and turning onto a floor comes to rest flat
  within three seconds in E³, H³ and S³, level to 3e-5, never sunk by more than 3e-7 after
  landing, and moving 1e-9 in its last second. The floor in that test is a totally geodesic
  plane, and gravity carried from the origin is normal to it everywhere.
- **Bodies are plain structs.** A `RigidBody` and its `Inertia` are not yet ECS components, and
  no plugin steps them. A game steps them itself.

## Far from the origin in H³

A Lorentz matrix's entries grow like `e^d`. Lorentz Gram–Schmidt renormalization squares them
against the `−1` it keeps, so a pose loses digits like `e^{2d}`. Measured by flying straight
out with a renormalization after every step:

| Distance | Residual |
| --- | --- |
| 5 | 1e-12 |
| 10 | 1e-8 |
| 15 | 5e-4 |
| before 20 | NaN |

The same holds in every direction. Anything computed from two far poses, such as a relative
pose or a body's spatial momentum, loses digits faster still.

- **The fix is the floating origin** (`fk_scene::FloatingOrigin`, opt-in). Once the eye is
  farther than its radius from the origin, every root pose is moved by the inverse of the
  transvection to the eye, so coordinates stay within a few units. With a radius of 4, an eye
  and a beacon one unit ahead of it fly twenty units and the beacon is still seen one unit
  ahead to 1e-6. Without it, both poses are lost. Anything that keeps world poses outside the
  roots' `Pose`s (a `RigidBody` in a resource, a saved position) must follow the
  `rebases` counter and apply `last`.
- **SL(2, ℂ) was not adopted.** Its entries grow like `e^{d/2}`, so storing H³'s isometries
  that way would double the reach before the same failure, not remove it. H³'s exp and log
  already go through SL(2, ℂ) in closed form. Since the floating origin keeps every pose near
  the origin, the drift that remains is 1e-12, and changing the isometry type would cost every
  user of H³ for no visible gain. Revisit only for a world that cannot rebase, one whose
  content has to be far from every eye at once.
