# Feuclide

A game engine in Rust whose core never assumes Euclidean geometry.

Every engine that "does" non-Euclidean space usually cheats: a Euclidean core, portal cameras and
shader tricks. Here geometry is a trait. Hyperbolic space H³, the 3-sphere S³, flat E³ and their
quotients (closed manifolds, portals) are all implementations of the same abstraction, and
everything above it (scene, renderer, physics) is written against that trait alone. Poses are
isometries, not matrices; depth is geodesic distance; rasterization and geodesic ray marching sit
behind one renderer and share one depth buffer.

It is built on `bevy_ecs`, `wgpu` and `winit`, but it is not a Bevy plugin: it uses no Bevy
transform or render crates, since `Mat4` and `GlobalTransform` are exactly what it avoids.

## Run the examples

```sh
cargo run --example fly-h3 --release         # hyperbolic space, tiled by right-angled dodecahedra
cargo run --example fly-s3 --release         # the 3-sphere, tiled by the 120-cell
cargo run --example seifert-weber --release  # a closed hyperbolic 3-manifold
cargo run --example lens-space --release     # the lens space L(7, 2)
cargo run -p fk-game --release               # a guided tour through every space, on rails
```

Click to grab the mouse, WASD (ZQSD on AZERTY) to fly where you look, Space / Ctrl to rise and
sink, Shift to go fast, F1 for every key. In `fly-h3` and `fly-s3`, N cycles through the tilings
and Tab shows depth as distance.

Any app can save a screenshot and exit with `FK_CAPTURE=<file.png>` (after `FK_CAPTURE_AFTER`
seconds, default 3). `FK_NO_VSYNC=1` lifts the refresh cap.

## Crates

| Crate | What |
|---|---|
| `fk-math` | f64 numerics, numerically stable series (`sinc`, `sinhc`, …), Minkowski helpers |
| `fk-geometry` | The core traits: `Geometry`, `GroupElement`, `VectorSpace`, `SampledMetric`, `GpuGeometry`, and a property-test battery |
| `fk-geometry-euclidean` | E², E³ |
| `fk-geometry-hyperbolic` | H², H³ in the hyperboloid model |
| `fk-geometry-spherical` | S², S³ |
| `fk-quotient` | Quotient and portal spaces: deck groups, Dirichlet domains, holonomy |
| `fk-app` | App, plugins, schedules, window, input and key mapping |
| `fk-scene` | Poses as isometries, hierarchy, cameras, floating origin |
| `fk-shaders` | WGSL modules, one per geometry |
| `fk-render` | wgpu renderer: metric-aware rasterization and geodesic ray marching in one depth buffer, post effects |
| `fk-physics` | Geodesic motion, a character controller, collision, rigid bodies on Lie groups |
| `fk-assets` | Meshes embedded into curved space, glTF, geodesic refinement, regular tilings |
| `fk-audio` | Sound (kira), silent without a device |
| `feuclide` | Umbrella crate |
| `fk-game` | The showcase tour |
| `tools` (`fk-tools`) | Golden images (raster against ray march), mesh refinement |

The layering is enforced by the dependency graph: `fk-app`, `fk-scene`, `fk-render` and
`fk-physics` depend on `fk-geometry` and never name a concrete geometry; the concrete geometries
are leaves the application picks.

## Conventions

- Poses are isometries relative to the parent. No scale in a pose, no global coordinate system.
- Twists are body-frame; all motion is `pose ← pose ∘ exp(dt · ξ)`. `compose(rhs)` applies `rhs`
  first. Integrators renormalize after every step.
- CPU math is f64. f32 exists only at the GPU boundary, on camera-relative quantities.
- Depth is reverse-Z, monotonic in geodesic distance, so the raster and ray-march paths compose.

The design is in [doc/PLAN.md](doc/PLAN.md) (where code and plan differ, the rustdoc of
`fk-geometry` is the authority), and what is exact and approximated in the rigid bodies in
[doc/RIGID_BODIES.md](doc/RIGID_BODIES.md).

## Adding a geometry

A geometry is done when it passes the property battery. Implement `Geometry`, `GroupElement` and
`SampledMetric`, then add `tests/battery.rs`:

```rust
fk_geometry::geometry_battery!(e3, fk_geometry_euclidean::E3);
```

with `fk-geometry = { workspace = true, features = ["battery"] }` under `[dev-dependencies]`. The
`holonomy` property checks the sign of the curvature; look there first when something fails.

## Build and test

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
```

On Linux, sound needs ALSA (`libasound2-dev`); the headless render tests draw with lavapipe
(`mesa-vulkan-drivers`) and are skipped without an adapter.

## License

Dual-licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option. This is the convention across the Rust ecosystem, including `bevy_ecs`, `wgpu` and
`winit`, the crates Feuclide builds on.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in
this work by you, as defined in the Apache-2.0 license, shall be dual-licensed as above, without
any additional terms or conditions.
