# Fucklide — A Non-Euclidean-First Game Engine in Rust

> This is the plan the engine was built from. Where the code differs from the sketches below,
> the code's rustdoc is the authority (see "As built in M0").

## Context

Every mainstream engine that "does" non-Euclidean spaces cheats: Euclidean core + portal cameras + shader tricks. Fucklide inverts this — the engine core never assumes Euclidean geometry. Geometry is a first-class trait; hyperbolic (H³/H²), spherical/elliptic (S³/S²), Euclidean (E³/E²), quotient/portal spaces, and later Riemannian, Finsler, Taxicab, Galilean/Lorentzian planes are all implementations of the same abstraction. Bevy-like, code-first, 3D-first (2D allowed).

**Decisions made with the user:**
1. **Rendering: hybrid from day one** — metric-aware rasterization fast path for constant-curvature spaces + GPU geodesic ray-marching as the fully general path, behind one abstraction.
2. **Foundation: standalone workspace** reusing `bevy_ecs`, `wgpu`, `winit`. Not a Bevy plugin — no bevy transform/render crates (Mat4/GlobalTransform are exactly what we forbid).
3. **MVP scope: H³/S³/E³ + portals/quotients first**; exotic metrics slot in later via the same traits.
4. **Physics: full rigid-body dynamics as target**, staged honestly (geodesic motion → character controller + collision → Euler–Poincaré rigid bodies; contact between extended rigid bodies in curved space is research-grade and staged accordingly).

## 1. Workspace Layout

Crates prefixed `fk-`; umbrella crate `fucklide` re-exports everything with a prelude (Bevy-style).

```
Cargo.toml                  # workspace, resolver = "2"
crates/
  fk-math                 # raw numerics: nalgebra wrappers, f32/f64 dual-precision scalar trait
  fk-geometry             # THE core: Geometry / GroupElement / SampledMetric traits + property-test battery
  fk-geometry-euclidean   # E²/E³ (SE(2)/SE(3))
  fk-geometry-hyperbolic  # H²/H³ (hyperboloid model; SO⁺(1,3), SL(2,C) upgrade path)
  fk-geometry-spherical   # S²/S³ (R³/R⁴ embedding; SO(4) as unit-quaternion pair)
  fk-quotient             # portal/quotient spaces: deck groups, fundamental domains, holonomy
  fk-scene                # ECS components (Pose<G>, hierarchy, camera, visibility) + schedules (bevy_ecs)
  fk-render               # wgpu render graph, GeometryRenderer abstraction, both pipelines
  fk-shaders              # WGSL templates + per-geometry modules (naga_oil), CI-validated
  fk-physics              # staged curved-space physics
  fk-assets               # glTF subset loading, embedding + geodesic refinement
  fk-app                  # App builder, winit loop, input, time
  fucklide                # umbrella + prelude
examples/                 # one demo crate per milestone
tools/                    # golden-image diff harness, mesh refinement CLI
```

Layering rule enforced by the dep graph: **`fk-scene` / `fk-render` / `fk-physics` never name a concrete geometry** — they are generic over `fk-geometry` traits. Concrete geometry crates are leaf plugins registered at app build. One geometry per `World` for MVP (quotients of one geometry in scope; portals between *different* geometries out of scope).

## 2. Core Geometry Traits (`fk-geometry`)

```rust
pub trait Geometry: 'static + Send + Sync {
    const DIM: usize;
    type Scalar: RealField;              // f32 or f64
    type Point: Copy + Pod;              // embedding coords (hyperboloid x∈R⁴, unit Vec4 for S³, Vec3 for E³)
    type Tangent: Copy + Pod;
    type Isometry: GroupElement;         // SE(3) | SO⁺(1,3) | SO(4)

    fn origin() -> Self::Point;          // distinguished basepoint o (identity coset, not a global origin)
    fn distance(a, b) -> Scalar;
    fn exp(p, v) -> Point;               // geodesic exponential
    fn log(p, q) -> Tangent;             // Option-like near cut locus (S³ antipode)
    fn parallel_transport(p, q, v) -> Tangent;
    fn apply(iso, p) -> Point;  fn apply_tangent(iso, v) -> Tangent;
    fn renormalize_point(&mut Point);  fn renormalize_iso(&mut Isometry);
}

pub trait GroupElement: Copy {           // this is what replaces Mat4
    type Algebra: Copy;                  // Lie algebra 𝔤 (twist)
    fn identity() -> Self;  fn compose(&self, &Self) -> Self;  fn inverse(&self) -> Self;
    fn exp(xi: Algebra) -> Self;  fn log(&self) -> Algebra;  fn adjoint(&self, Algebra) -> Algebra;
}
```

> **As built in M0** (`crates/fk-geometry`): the scalar is fixed to `Real = f64` (not generic, since the f64 CPU core is mandatory anyway), `Point`/`Tangent` drop the `Pod` bound (GPU upload goes through `fk_math::to_gpu` after camera-relative conversion), `renormalize`/`residual` live on `GroupElement`, `log` returns `Option`, and `Geometry` gains `origin_frame`, `inner`, `transvection(v)` and `rotation(u, w)` (the Lie algebra generators the battery and the camera controllers build motions from), `constant_curvature`, `injectivity_radius` and residual checks. The rustdoc is the authoritative signature.

Concrete representations:
- **E³**: Isometry = `(UnitQuaternion, Vec3)`; Point/Tangent = Vec3.
- **S³**: Point = unit Vec4; Isometry = SO(4) as **pair of unit quaternions (l, r)**, x ↦ l·x·r⁻¹ (cheap compose/renormalize on the CPU; converted to a `mat4` for GPU upload, see §4).
- **H³**: Point = Vec4 on hyperboloid (⟨x,x⟩₋ = −1, x₀>0); Isometry = SO⁺(1,3) as 4×4 Lorentz matrix first (fits a mat4x4 uniform), SL(2,C) later for compose stability.
- 2D variants are the same code one dimension down, shared internally per geometry crate.

**"Transform" replacement:** `Pose<G>(G::Isometry)` — the isometry carrying the basepoint frame to the object's frame, **relative to parent**. Hierarchy propagation = group composition `world(child) = world(parent) ∘ local(child)`; holonomy-aware in quotients (representative pose + `ChartTag` word in the deck group). `Velocity<G>(G::Algebra)` is a body-frame twist; all motion everywhere is `pose ← pose ∘ exp(dt·ξ)`. **No scale in Pose** (scale isn't an isometry and doesn't exist meaningfully in H³/S³); scale is baked at mesh-embedding time.

**Exotic-metric hook:** a weaker `SampledMetric` trait (christoffels / geodesic ODE RHS + a WGSL module) is all the ray-marcher needs. Constant-curvature geometries implement both (used for pipeline-parity testing). Taxicab/Finsler later implement only `SampledMetric` + subsets; planned trait split (`UniqueGeodesics`) at M6 since Taxicab geodesics aren't unique.

## 3. Portals / Quotient Spaces (`fk-quotient`)

**Universal cover + holonomy representation** — not shader portals.
- A quotient = `(G, Γ)`: deck group Γ as a generator set of isometries + a fundamental-domain face test `which_face(p) -> Option<FaceId>`. Covers flat torus (E³/translations), hyperbolic manifolds (dodecahedral H³ tilings), lens spaces (S³ quotients) with one struct.
- Objects live in the fundamental domain; exiting through face f left-multiplies by γ_f and updates `ChartTag`. Shortest-path queries enumerate Γ-translates up to a bounded word length (cached BFS).
- **Seams solved by multiplicity, not stencils:** visible objects are rendered at all Γ-translates within the view horizon (H³ fog radius bounds this tightly; S³ is finite). The same translate-set feeds collision ghosts, so "object straddling a portal" needs zero special-casing.
- Aperture game-portals (à la Portal, in E³) = partial deck transform through a bounded aperture; thin layer on top, with stencil + near-plane replacement (M4).
- Ray-march path: marcher applies γ_f on crossing a face and continues — textbook quotient ray tracing.

## 4. Renderer (`fk-render`, `fk-shaders`)

- Hand-rolled thin wgpu render graph (no bevy_render): visibility/deck-enumeration → opaque pass → fog/post → UI.
- `trait GeometryRenderer { prepare / queue / render }`, two impls:
  1. **RasterPipeline** (constant curvature). Vertices in model space around the basepoint. The CPU composes `view⁻¹ ∘ pose` in f64 and uploads **one `mat4` per instance for every geometry** (SE(3) as homogeneous mat4, SO⁺(1,3) as Lorentz mat4, SO(4) converted from the quaternion pair), so the vertex shader is the same `mat4 · v` everywhere, followed by a per-geometry projection:
     - E³: classic perspective.
     - H³: Lorentz mat4 → view-space **Beltrami–Klein** coords (x/x₀): geodesics map to straight lines so rasterized edges are exact geodesics; fog by hyperbolic distance (the exp falloff *is* the H³ horizon).
     - S³: SO(4) mat4 → view-space gnomonic, clip w = ⟨x, eye⟩ = cos d (the hardware clips w < 0). **Full wraparound, distances [0, 2π)**: every object is drawn twice, as q and −q, across four slices — (q_t, w) → [0, π/2), (q_t, −w) → [π/2, π), (−q_t, −w) → [π, 3π/2), (−q_t, w) → [3π/2, 2π) — so you see your own back. ~2× vertex work; objects straddling the equator land in more slices.
     - **Depth = hardware perspective depth, reverse-Z f32, never `frag_depth` in the raster path** (it disables early-Z/Hi-Z in every geometry, E³ included). Per pixel this ordering is exact: all fragments at a pixel lie on one ray from the eye, and in Klein/gnomonic coords rays are straight lines along which z is monotonic in geodesic distance. S³ slices composite through per-pass viewport depth ranges, with the z mapping flipped in the mirrored slices (2 and 4). Geodesic distance is reconstructed from depth + pixel ray where it is actually needed (fog/post, compositing with the ray-marcher, which writes `frag_depth` in the same encoding — fine for a fullscreen pass).
     - Fragment: interpolate the embedding 4-vector (perspective-correct), put it back on the manifold with one `rsqrt` (a geodesic triangle is the projectivized span of its vertices, so this is exact), and project the interpolated normal onto the tangent space. No inverse-transpose normal matrix: poses carry no scale.
  2. **RayMarchPipeline** (general): fullscreen pass, per-pixel RK4 geodesic integration with the metric's WGSL module inlined; SDF objects first, BVH later. Serves as **ground truth** for golden tests of the raster path. Constant-curvature scenes step with the closed-form `exp` instead of RK4. Exotic RK4 metrics render at half resolution + upscale with adaptive step size.
- Camera: `Camera<G> { pose, projection }`; fov = angle at eye (well-defined in any geometry); near/far as geodesic distances.
- Culling: geodesic bounding balls; frustum tests in view frame (linear in Klein coords for H³). H³: mandatory hard fog radius (exponential volume growth) plus **distance LOD** (see GPU cost model). S³: distance culling does nothing (everything is within 2π), frustum culling per image and slice membership only.
- WGSL: `naga_oil` composition — shared skeleton + per-geometry modules (`geo_apply_iso`, `geo_project`, `geo_distance`, `geo_fog`, light attenuation). All permutations validated by naga in CI. GPU layout via `encase` + `bytemuck`.
- Closed forms keep transcendentals out of the hot path: with m = −⟨p, light⟩ (H³) or ⟨p, light⟩ (S³), attenuation 1/sinh²(r) = 1/(m² − 1) and 1/sin²(r) = 1/(1 − m²); H³ fog e^(−r/λ) = (x₀ + √(x₀² − 1))^(−1/λ), no `acosh`.

### GPU cost model

Shader ALU cost per vertex/fragment is ~E³ in H³ and S³ (same vertex `mat4`, +1 `rsqrt` and a few FMAs per fragment, closed-form lighting). The cost is in **how much gets drawn**:

- **H³ — exponential content, micro-triangles.** An object of radius 0.5 at distance r subtends ≈ 1/sinh(r) rad: ~0.7 px at r = 8 (1080p, 60° fov). Set the fog radius from that (~7–8), and LOD by geodesic distance from M2 on: near the horizon exponentially many sub-pixel triangles waste 2×2-quad shading and choke triangle setup.
- **H³ depth precision.** Klein depth gap 1 − k_z ≈ 2e^(−2r) reaches one f32 ulp around r ≈ 8 (~0.1–0.2 geodesic units of depth resolution there). Matches the angular-size cutoff, so mostly harmless; large floors near the fog edge may z-fight — checked in M2.
- **H³ quotients — instance count.** Translates in range ≈ π(sinh 2r − 2r) / vol(F). Seifert–Weber (vol ≈ 11.2): ~23k at r = 6 (~3k after frustum culling), ~1.2M at r = 8 (~180k). Instanced draws (64 B `mat4` per instance); practical fog radius for H³ manifolds ≈ 5–6.
- **S³ — whole world always in range**, drawn twice for wraparound: ~2× E³ vertex work for the same scene; fragment cost tracks screen coverage as usual.
- **Ray-march (1080p, ~2M px).** Closed-form constant curvature: ~64 steps × ~100 ops ≈ 1.3·10¹⁰ ops/frame (~0.8 TFLOP/s at 60 fps, easy). Exotic RK4: ~300 steps × (4 metric evals + SDF ≈ 500 ops) ≈ 3·10¹¹ ops/frame (~18 TFLOP/s at 60 fps before divergence) — hence half-res + upscale.

## 5. Scene & Content (`fk-assets`)

- Meshes authored in flat tangent space at a basepoint (glTF via `gltf` crate; positions/normals/uv only), embedded at load via `exp(o, v)`. For large geometry: **geodesic refinement pass** (subdivide until max geodesic edge length < ε — long "straight" edges visibly sag in H³). Adaptive, offline or at load (`tools/`).
- Procedural primitives per geometry (geodesic sphere, box-analogue, {p,q}/{5,3,4}-style tiled floors) as demo content and test fixtures.
- No material system beyond base-color + normal; synchronous loading; no hot reload until post-MVP.

## 6. Physics (`fk-physics`) — honesty ladder

**Rapier/parry not reusable:** Isometry3 positions, AABB broadphase, GJK on linear support functions, R³-linearized solver — all meaningless off flat space. Reuse ideas (solver structure, islands), zero code.

- **P1 — Geodesic kinematics (textbook, ships M2):** twist integration, gravity as force on the twist, parallel transport of carried frames. Closed forms in constant curvature.
- **P2 — Character controller + sphere/capsule collision (solid engineering, ships M3):** geodesic balls/capsules with closed-form distance queries (cosh/cos embedding identities); geodesic-ball BVH broadphase (no AABBs) + Γ-ghosts from fk-quotient; impulses along the connecting geodesic with velocities **parallel-transported to the contact point**.
- **P3 — Rigid bodies (research-grade, ships M5):** configuration = one group element; dynamics = **Euler–Poincaré / Lie–Poisson on 𝔤*** with a generalized inertia operator from the mass distribution (translation/rotation couple in H³/S³ — a spinning body drifts; physical, not a bug). Symplectic Lie-group integrator (RK-MK / RATTLE-style) + renormalization. Well-understood: free motion, the equations, energy behavior. **Research:** extended-body contact/friction, stacking. Acceptance scoped accordingly (free spinner conserves invariants; one box rests on a floor; no stack promises).

## 7. Numerical Stability

- Hyperboloid coords grow like cosh(d); f32 dies at d ≈ 20–40. Mitigations baked in from M0:
  1. **Camera-relative everything**: recompute poses as `view⁻¹ ∘ pose` in f64 on CPU each frame; GPU only ever sees small-magnitude f32 (camera sits at the basepoint).
  2. `fk-math` scalar-generic: **f64 CPU core** for pose accumulation/physics, f32 at the GPU boundary.
  3. **Mandatory renormalization** post-step in twist integration (manifold re-projection + Gram–Schmidt w.r.t. the relevant form / quaternion normalize), debug assertions on constraint residuals.
  4. "Floating origin in G" (rebase scene root by camera pose) as the M5 escape hatch for open H³ worlds; quotient spaces already keep coordinates bounded.
- S³ log ill-conditioned at the antipode — explicit cut-locus convention, tested. SO⁺(1,3) drift > SO(4); SL(2,C) is the upgrade path.

## 8. Testing

- **Geometry-agnostic property battery** (`proptest`) in `fk-geometry`, run by every impl crate: metric axioms; exp/log roundtrip; unit-speed geodesics `d(p, exp(p,tv)) = |t||v|`; isometry invariance; parallel transport preserves inner products and inverts; group axioms; **holonomy triangle test** (transport around a geodesic triangle rotates by area·K — catches most curvature-sign bugs); constraint residual < ε over 10⁶ random compositions.
- **Quotient tests:** face-pairing edge-cycle relations = identity; exit-and-reenter returns an equivalent pose.
- **Cross-pipeline golden tests (the killer test):** same scene via raster and ray-march, perceptual diff < threshold. Headless wgpu (lavapipe fallback in CI), blessed PNGs per scene × geometry × pipeline (`tools/`).
- **Physics:** conserved-quantity regression (P3), penetration bounds (P2).

## 9. Milestones

| M | Scope | Acceptance |
|---|---|---|
| **M0** (~1 wk) | git init, workspace, CI (fmt/clippy/test), fk-math, fk-geometry traits + battery, E³ impl | battery green on E³ |
| **M1** (2–3 wk) | H³/S³ (+2D), fk-quotient with flat-torus instance | battery + holonomy test green on all; torus wraparound test |
| **M2** (2–3 wk) | fk-app loop, raster pipeline, fly camera (twist integration), procedural tiled floors | `examples/fly-{e3,h3,s3}`: fly a fogged {5,3,4}-ish H³ room; in S³ fly forward and return, and see your own back (wraparound images) |
| **M3** (3–4 wk) | P1+P2 physics, glTF + refinement, per-geometry lighting, ray-march MVP + first golden test | walk/jump/collide demo per geometry; raster-vs-raymarch diff < threshold on 3 scenes |
| **M4** (2–3 wk) | Full quotient rendering + ghost collisions, H³ manifold demo, lens-space demo, E³ aperture portals | seamless walk through a compact H³ manifold; straddling objects render & collide correctly |
| **M5** (4–6 wk) | Euler–Poincaré rigid bodies, inertia operators, resting contact | energy-conserving free spinner; box settles on floor in E³/H³/S³ |
| **M6** (open) | SampledMetric-only exotics via ray-march: Nil/Sol or Finsler, Taxicab 2D, Galilean plane; `UniqueGeodesics` trait split | one exotic-metric scene renders with a sane camera |

## 10. Reuse vs Write

| Dep | Verdict |
|---|---|
| `bevy_ecs` (+schedule) | Reuse — ECS only, no other bevy crates |
| `wgpu`, `winit`, `raw-window-handle` | Reuse |
| `nalgebra` | Reuse as number container only (f64 generics); its `Isometry3` never in public API; skip glam to avoid dual-math pain |
| `encase`, `bytemuck`, `naga_oil`/`naga` | Reuse |
| `gltf`, `image`, `proptest`, `approx` | Reuse |
| `rapier`/`parry` | **No** — E³ baked into every layer |
| bevy transform/render/scene | **No** — Mat4 assumptions are what we're eliminating |

## Risks (pre-identified)

- H³ tessellation cost → fog radius from angular size + distance LOD (M2) + adaptive refinement (M3).
- H³ f32 Klein depth precision near the fog edge (r ≈ 8) → reverse-Z f32 depth, verified in M2.
- Raster/ray-march visual parity → shared WGSL geometry modules + golden tests.
- P3 contact model → staged acceptance; M6 does not block on it.
- f32 blow-up → camera-relative f64 core from day one (retrofit is painful; it's in M0's fk-math).

## First files to create (M0 order)

1. `Cargo.toml` — workspace graph (encodes the layering rules)
2. `crates/fk-geometry/src/lib.rs` — `Geometry`, `GroupElement`, `SampledMetric`
3. `crates/fk-geometry/src/battery.rs` — reusable property-test suite
4. `crates/fk-geometry-euclidean/src/lib.rs` — E³ impl proving the traits
5. `crates/fk-scene/src/pose.rs` — `Pose<G>`, `Velocity<G>`, holonomy-aware propagation
6. `crates/fk-render/src/renderer.rs` — `GeometryRenderer` split

## Verification

- Every milestone: `cargo fmt --check && cargo clippy -- -D warnings && cargo test` across the workspace.
- M1+: geometry battery is the executable spec — a new geometry is "done" when the battery passes.
- M2+: run the `examples/fly-*` binaries and visually verify (H³ fog horizon, S³ wraparound / own back visible).
- M3+: golden-image harness in CI (headless wgpu); ray-march path is the mathematical ground truth for the raster path.
