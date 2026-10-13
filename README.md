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

## Showcase:

### h3

<img width="480" height="270" alt="fly-h3" src="https://github.com/user-attachments/assets/0fcde3a2-460e-449e-8723-69b73edbfbec" />

### Lens space

<img width="480" height="270" alt="lens-space" src="https://github.com/user-attachments/assets/f0d0fc9e-166e-43b7-ab5a-772f015f10f1" />

### seifert weber

<img width="480" height="270" alt="seifert-weber" src="https://github.com/user-attachments/assets/6b28839b-861b-4bb1-9381-5f86fa781ab5" />

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

Each crate under `crates/` has a README of its own, which is also its crates.io page.

| Crate | What | Bevy counterpart | What's different |
| --- | --- | --- | --- |
| [`fk-math`](https://crates.io/crates/fk-math) | f64 numerics, numerically stable series (`sinc`, `sinhc`, …), Minkowski helpers | `bevy_math` (on glam) | nalgebra in `f64`; `f32` only at the GPU boundary. Adds stable series for the exp/log maps. No shapes or geometry here: it's numbers only. |
| [`fk-geometry`](https://crates.io/crates/fk-geometry) | The core traits: `Geometry`, `GroupElement`, `VectorSpace`, `SampledMetric`, `GpuGeometry`, and a property-test battery | **None.** The closest is `bevy_transform`'s `Transform` with `bevy_math`'s `Isometry3d` | The heart of Feuclide. Bevy hard-codes flat space (`Vec3` + `Quat` + scale). Here a space is a trait: points, tangents, isometry group, exp/log, parallel transport. E³, H³ and S³ are interchangeable implementations, checked by one property test suite. |
| [`fk-geometry-euclidean`](https://crates.io/crates/fk-geometry-euclidean) | E², E³ | **None as a crate:** flat space is what all of Bevy assumes | Flat space is a leaf the application picks, not a foundation: swap it for H³ or S³ and everything above runs unchanged. |
| [`fk-geometry-hyperbolic`](https://crates.io/crates/fk-geometry-hyperbolic) | H², H³ in the hyperboloid model | **None.** | Bevy has no notion of curved space. |
| [`fk-geometry-spherical`](https://crates.io/crates/fk-geometry-spherical) | S², S³ | **None.** | Bevy has no notion of curved space. |
| [`fk-quotient`](https://crates.io/crates/fk-quotient) | Quotient and portal spaces: deck groups, Dirichlet domains, holonomy | **None.** | Quotient spaces (flat tori, closed hyperbolic and spherical manifolds), where leaving one side brings you back in another; also portals. Bevy has no notion of this. |
| [`fk-app`](https://crates.io/crates/fk-app) | App, plugins, schedules, window, input and key mapping, tuning and kept files | `bevy_app` + `bevy_window` + `bevy_winit` + `bevy_input` + `bevy_time` + asset hot reload | Same App/Plugin/schedules model, slimmed down. It adds a key mapper with player rebinding (Bevy leaves that to third-party crates like leafwing), a fixed step, handing the window over to a second app built on a thread, and TOML tuning files reloaded on save. |
| [`fk-scene`](https://crates.io/crates/fk-scene) | Poses as isometries, hierarchy, cameras, floating origin | `bevy_transform` + `bevy_camera` | `Pose` / `GlobalPose` replace `Transform` / `GlobalTransform`. They are isometries composed down the hierarchy, with no scale. Everything is made camera-relative before reaching the GPU. The hierarchy itself is `bevy_ecs`'s own `ChildOf`, shared rather than replaced. |
| [`fk-shaders`](https://crates.io/crates/fk-shaders) | WGSL modules, one per geometry | `bevy_shader` | Both compose WGSL with **naga_oil**, which is itself a Bevy-org crate. The difference: one geometry module per space (`fk::geometry`), and the game supplies its own world uniforms, SDF, deform, post-pass (`fk::grade`) and sky-figure hooks, or skeletons of its own. |
| [`fk-render`](https://crates.io/crates/fk-render) | wgpu renderer: metric-aware rasterization and geodesic ray marching in one depth buffer, post effects | `bevy_render` + `bevy_core_pipeline` + `bevy_pbr` + `bevy_light` + `bevy_post_process` + `bevy_anti_alias` | Raw wgpu with a small fixed render graph instead of Bevy's extensible one. Rasterization works in any geometry. It adds a **geodesic ray marcher** that shares the depth buffer with the raster path; Bevy has nothing like it. Its frame accumulation (`Settle`) is roughly Bevy's TAA. Games add fullscreen passes of their own (`ImageEffect`, like a post-processing node) and effects inside the post pass (`GradeShader`). |
| `fk-sky` | The day and the night: a clock, the sun's and the moon's paths, the colours of every hour | **None in Bevy itself**; atmosphere and sun plugins | A time-of-day plugin writing `fk-render`'s `Lighting`, tuned in a TOML file. Unreleased yet. |
| `fk-ui` | What an app draws over the image: the frame rate, key lists, a pointer, a console's core, menu pages, CPU text | `bevy_dev_tools`' FPS overlay, `bevy_console`, `bevy_ui` | A model of menus the app draws, and readouts in `fk-render`'s text panel. Unreleased yet. |
| `fk-flat` | Flat (E³) worlds: standing on a ground, free spots, merged scenery, swaying kits, height-field land, a first-person body | Terrain and first-person controller crates | The one engine crate that names its geometry: a leaf an application picks, like the geometry crates. Unreleased yet. |
| [`fk-physics`](https://crates.io/crates/fk-physics) | Geodesic motion, a character controller, collision, rigid bodies on Lie groups | **None in Bevy.** The equivalents are third-party: Avian, Rapier | Rigid bodies integrated on the Lie group itself, so the same code runs in E³, H³ and S³. Also a character controller and collision against distance fields. |
| [`fk-assets`](https://crates.io/crates/fk-assets) | Meshes embedded into curved space, glTF, geodesic refinement, regular tilings | `bevy_gltf` + a little of `bevy_asset` | Synchronous glTF loading only, with no asset server or handles. It adds embedding a flat mesh into curved space and geodesic refinement, so long edges don't sag. |
| [`fk-audio`](https://crates.io/crates/fk-audio) | Sound (kira), silent without a device | `bevy_audio` (on rodio) | Built on kira, like the third-party `bevy_kira_audio`. Adds reverb rooms, directional emitters and gliding voices, and stays silent without an audio device. |
| [`feuclide`](https://crates.io/crates/feuclide) | Umbrella crate | `bevy` / `bevy_internal` | Same idea: an umbrella crate that re-exports the others. |
| `fk-game` | The showcase tour | Bevy's `examples/` and showcase | A guided tour on rails through every space the engine draws. |
| `tools` (`fk-tools`) | Golden images (raster against ray march), mesh refinement | **None.** | Offline checks and asset preparation for this engine's own paths. |

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
