# fk-physics

Curved-space physics. Part of [Feuclide](https://crates.io/crates/feuclide), a game engine in Rust whose core never assumes Euclidean geometry. Source, examples and design notes: [github.com/Dastfox/feuclide](https://github.com/Dastfox/feuclide).

Physics generic over the geometry. Every motion is `pose ← pose ∘ exp(dt · ξ)` with a body-frame
twist, gravity is carried by parallel transport, and frames are parallel-transported. A character
controller walks, runs, jumps and slides along balls, capsules and any surface given by a
distance function. The broadphase is a tree of geodesic balls, copied across the faces of the
fundamental domain in a quotient space. Rigid bodies keep their configuration as one isometry and
their momentum on the dual of the Lie algebra, integrated by a variational Lie-group integrator.

## Bevy counterpart

**None in Bevy.** The equivalents are third-party: Avian, Rapier.

Rigid bodies integrated on the Lie group itself, so the same code runs in E³, H³ and S³. Also a
character controller and collision against distance fields.

## Links

- Engine and umbrella crate: [`feuclide`](https://crates.io/crates/feuclide)
- Repository: [github.com/Dastfox/feuclide](https://github.com/Dastfox/feuclide)
- API docs: [docs.rs/fk-physics](https://docs.rs/fk-physics)

## License

Licensed under either of [Apache License, Version 2.0](https://github.com/Dastfox/feuclide/blob/main/LICENSE-APACHE) or
[MIT license](https://github.com/Dastfox/feuclide/blob/main/LICENSE-MIT) at your option.
