# fk-render

The renderer. Part of [Feuclide](https://crates.io/crates/feuclide), a game engine in Rust whose core never assumes Euclidean geometry. Source, examples and design notes: [github.com/Dastfox/feuclide](https://github.com/Dastfox/feuclide).

A wgpu renderer generic over `GpuGeometry`, with a thin, fixed render graph: visibility and
culling against geodesic bounding balls, shadows (flat space), one multisampled HDR opaque pass,
then a post pass to the window. Two paths draw into the same depth buffer: metric-aware
rasterization for constant-curvature spaces, and geodesic ray marching against a signed distance
field, for any metric. Depth is reverse-Z and monotonic in geodesic distance, so the two compose.
Games shape meshes on the GPU with their own vertex deform and supply their own SDF.

## Bevy counterpart

`bevy_render` + `bevy_core_pipeline` + `bevy_pbr` + `bevy_light` + `bevy_post_process` + `bevy_anti_alias`.

Raw wgpu with a small fixed render graph instead of Bevy's extensible one. Rasterization works in
any geometry. It adds a **geodesic ray marcher** that shares the depth buffer with the raster
path; Bevy has nothing like it. Its frame accumulation (`Settle`) is roughly Bevy's TAA. Games add
fullscreen passes of their own (`ImageEffect`, like a Bevy post-processing node) and their own
effects inside the post pass (`GradeShader`, a WGSL hook).

## Links

- Engine and umbrella crate: [`feuclide`](https://crates.io/crates/feuclide)
- Repository: [github.com/Dastfox/feuclide](https://github.com/Dastfox/feuclide)
- API docs: [docs.rs/fk-render](https://docs.rs/fk-render)

## License

Licensed under either of [Apache License, Version 2.0](https://github.com/Dastfox/feuclide/blob/main/LICENSE-APACHE) or
[MIT license](https://github.com/Dastfox/feuclide/blob/main/LICENSE-MIT) at your option.
