# fk-assets

Meshes in curved space. Part of [Feuclide](https://crates.io/crates/feuclide), a game engine in Rust whose core never assumes Euclidean geometry. Source, examples and design notes: [github.com/Dastfox/feuclide](https://github.com/Dastfox/feuclide).

A mesh is authored flat, in the tangent space at the origin, and comes from a binary glTF
(positions, normals and the first uv set). It is embedded into a geometry by following each
vertex's geodesic from the origin, `exp(o, v)`. Embedding moves only the vertices, so a long edge
becomes a chord and the surface sags; `refine` splits the flat mesh until no edge is longer than a
bound measured in geodesic distance, with no T-junctions. `tiling` builds the regular tilings of
space by cubes and dodecahedra, in whichever geometry has them.

## Bevy counterpart

`bevy_gltf` + a little of `bevy_asset`.

Synchronous glTF loading only, with no asset server or handles. It adds embedding a flat mesh into
curved space and geodesic refinement, so long edges don't sag.

## Links

- Engine and umbrella crate: [`feuclide`](https://crates.io/crates/feuclide)
- Repository: [github.com/Dastfox/feuclide](https://github.com/Dastfox/feuclide)
- API docs: [docs.rs/fk-assets](https://docs.rs/fk-assets)

## License

Licensed under either of [Apache License, Version 2.0](https://github.com/Dastfox/feuclide/blob/main/LICENSE-APACHE) or
[MIT license](https://github.com/Dastfox/feuclide/blob/main/LICENSE-MIT) at your option.
