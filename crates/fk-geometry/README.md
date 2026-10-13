# fk-geometry

Geometry as a trait. Part of [Feuclide](https://crates.io/crates/feuclide), a game engine in Rust whose core never assumes Euclidean geometry. Source, examples and design notes: [github.com/Dastfox/feuclide](https://github.com/Dastfox/feuclide).

The heart of Feuclide. A space is a type implementing `Geometry`: it says what its points and
tangent vectors are, how far apart two points are, how geodesics run (`exp` / `log`), how vectors
are carried along them (parallel transport), and which group of isometries acts on it
(`GroupElement`). Poses are isometries, never matrices, and there is no global coordinate system.
`GpuGeometry` is the GPU boundary: a 4-vector per point and a 4×4 matrix per isometry, so the
renderer stays generic. `SampledMetric` describes spaces given by a metric alone, for the ray
marcher.

Everything above this crate (scene, renderer, physics) is written against these traits and never
names a concrete space. The `battery` feature exposes `geometry_battery!`, the property-test suite
every geometry runs: it is the executable definition of a correct `Geometry`.

## Bevy counterpart

**None.** The closest is `bevy_transform`'s `Transform` with `bevy_math`'s `Isometry3d`.

Bevy hard-codes flat space (`Vec3` + `Quat` + scale). Here a space is a trait: points, tangents,
isometry group, exp/log, parallel transport. E³, H³ and S³ are interchangeable implementations,
checked by one property test suite.

## Links

- Engine and umbrella crate: [`feuclide`](https://crates.io/crates/feuclide)
- Repository: [github.com/Dastfox/feuclide](https://github.com/Dastfox/feuclide)
- API docs: [docs.rs/fk-geometry](https://docs.rs/fk-geometry)

## License

Licensed under either of [Apache License, Version 2.0](https://github.com/Dastfox/feuclide/blob/main/LICENSE-APACHE) or
[MIT license](https://github.com/Dastfox/feuclide/blob/main/LICENSE-MIT) at your option.
