# fk-geometry-euclidean

Flat plane and space (E², E³). Part of [Feuclide](https://crates.io/crates/feuclide), a game engine in Rust whose core never assumes Euclidean geometry. Source, examples and design notes: [github.com/Dastfox/feuclide](https://github.com/Dastfox/feuclide).

Euclidean geometry as one geometry among others: `E2` and `E3` implement the same
[`fk-geometry`](https://crates.io/crates/fk-geometry) traits as hyperbolic and spherical space,
with the isometry groups SE(2) and SE(3), and get no special treatment anywhere in the engine.

## Bevy counterpart

**None as a crate:** flat space is what all of Bevy assumes (`Vec3` + `Quat` + scale, `Isometry3d`).

Here flat space is a leaf the application picks, not a foundation: swap `E3` for `H3` or `S3`
and the scene, renderer and physics run unchanged. Its isometries carry no scale.

## Links

- Engine and umbrella crate: [`feuclide`](https://crates.io/crates/feuclide)
- Repository: [github.com/Dastfox/feuclide](https://github.com/Dastfox/feuclide)
- API docs: [docs.rs/fk-geometry-euclidean](https://docs.rs/fk-geometry-euclidean)

## License

Licensed under either of [Apache License, Version 2.0](https://github.com/Dastfox/feuclide/blob/main/LICENSE-APACHE) or
[MIT license](https://github.com/Dastfox/feuclide/blob/main/LICENSE-MIT) at your option.
