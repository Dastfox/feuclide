# fk-geometry-spherical

Spheres (S², S³). Part of [Feuclide](https://crates.io/crates/feuclide), a game engine in Rust whose core never assumes Euclidean geometry. Source, examples and design notes: [github.com/Dastfox/feuclide](https://github.com/Dastfox/feuclide).

The unit 2-sphere and 3-sphere embedded in ℝ³ and ℝ⁴. `S3`'s isometries `So4` are pairs of unit
quaternions `(l, r)` acting as `x ↦ l x r̄`; `S2`'s isometries `So3` are unit quaternions. Both
have curvature +1 and injectivity radius π: the antipode of a point is its cut locus, where `log`
returns `None` and parallel transport only projects.

## Bevy counterpart

**None.** Bevy has no notion of curved space.

A full implementation of the `fk-geometry` traits, so everything Feuclide draws, moves and
collides in E³ works in S³ too, passing the same property battery.

## Links

- Engine and umbrella crate: [`feuclide`](https://crates.io/crates/feuclide)
- Repository: [github.com/Dastfox/feuclide](https://github.com/Dastfox/feuclide)
- API docs: [docs.rs/fk-geometry-spherical](https://docs.rs/fk-geometry-spherical)

## License

Licensed under either of [Apache License, Version 2.0](https://github.com/Dastfox/feuclide/blob/main/LICENSE-APACHE) or
[MIT license](https://github.com/Dastfox/feuclide/blob/main/LICENSE-MIT) at your option.
