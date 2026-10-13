# fk-shaders

WGSL modules. Part of [Feuclide](https://crates.io/crates/feuclide), a game engine in Rust whose core never assumes Euclidean geometry. Source, examples and design notes: [github.com/Dastfox/feuclide](https://github.com/Dastfox/feuclide).

WGSL templates and per-geometry shader modules, composed at pipeline creation with `naga_oil`.
A pipeline's shader is a geometry-agnostic skeleton importing shared modules: `fk::view` (the
eye, the game's world block, the lighting), `fk::sky`, and `fk::geometry`, one module per space,
chosen by the geometry's `GpuGeometry::SHADER`, providing isometries, distances, fog, projection
and the depth encoding every render path shares. Games replace `fk::deform` (vertex shaping) and
`fk::sdf` (what the ray marcher draws) with their own.

## Bevy counterpart

`bevy_shader`.

Both compose WGSL with **naga_oil**, which is itself a Bevy-org crate. The difference: one geometry
module per space (`fk::geometry`), and the game supplies its own world uniforms, SDF and deform
hooks.

## Links

- Engine and umbrella crate: [`feuclide`](https://crates.io/crates/feuclide)
- Repository: [github.com/Dastfox/feuclide](https://github.com/Dastfox/feuclide)
- API docs: [docs.rs/fk-shaders](https://docs.rs/fk-shaders)

## License

Licensed under either of [Apache License, Version 2.0](https://github.com/Dastfox/feuclide/blob/main/LICENSE-APACHE) or
[MIT license](https://github.com/Dastfox/feuclide/blob/main/LICENSE-MIT) at your option.
