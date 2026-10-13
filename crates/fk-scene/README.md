# fk-scene

Poses, hierarchy and cameras. Part of [Feuclide](https://crates.io/crates/feuclide), a game engine in Rust whose core never assumes Euclidean geometry. Source, examples and design notes: [github.com/Dastfox/feuclide](https://github.com/Dastfox/feuclide).

ECS components and schedules, generic over the geometry. `Pose` is an isometry relative to the
parent; `GlobalPose` is the composition down the hierarchy, `world(child) = world(parent) ∘
local(child)`. `Velocity` is a body-frame twist integrated every fixed step as
`pose ← pose ∘ exp(dt · ξ)`. `Camera` marks the eye, and every entity gets its pose relative to it
(`ViewRelative`), computed in `f64` so only small numbers reach the GPU. `FloatingOrigin` moves the
whole scene back to the eye when it strays too far, for open H³ worlds.

## Bevy counterpart

`bevy_transform` + `bevy_camera`.

`Pose` / `GlobalPose` replace `Transform` / `GlobalTransform`. They are isometries composed down
the hierarchy, with no scale. Everything is made camera-relative before reaching the GPU. The
hierarchy itself is `bevy_ecs`'s own `ChildOf`, shared rather than replaced.

## Links

- Engine and umbrella crate: [`feuclide`](https://crates.io/crates/feuclide)
- Repository: [github.com/Dastfox/feuclide](https://github.com/Dastfox/feuclide)
- API docs: [docs.rs/fk-scene](https://docs.rs/fk-scene)

## License

Licensed under either of [Apache License, Version 2.0](https://github.com/Dastfox/feuclide/blob/main/LICENSE-APACHE) or
[MIT license](https://github.com/Dastfox/feuclide/blob/main/LICENSE-MIT) at your option.
