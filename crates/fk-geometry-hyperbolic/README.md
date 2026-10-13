# fk-geometry-hyperbolic

Hyperbolic plane and space (H², H³). Part of [Feuclide](https://crates.io/crates/feuclide), a game engine in Rust whose core never assumes Euclidean geometry. Source, examples and design notes: [github.com/Dastfox/feuclide](https://github.com/Dastfox/feuclide).

Hyperbolic geometry in the hyperboloid model, with isometry group SO⁺(1,n). Points satisfy
`⟨p, p⟩ = −1`, `t > 0` in the Minkowski form (time first); isometries are Lorentz matrices
(`Lorentz3`, `Lorentz2`), renormalized by Lorentz Gram–Schmidt. Their exponential and logarithm go
through SL(2, ℂ), the double cover of SO⁺(1, 3), where both have closed forms. Curvature −1,
geodesics unique: `log` is always defined.

## Bevy counterpart

**None.** Bevy has no notion of curved space.

A full implementation of the `fk-geometry` traits, so everything Feuclide draws, moves and
collides in E³ works in H³ too, passing the same property battery.

## Links

- Engine and umbrella crate: [`feuclide`](https://crates.io/crates/feuclide)
- Repository: [github.com/Dastfox/feuclide](https://github.com/Dastfox/feuclide)
- API docs: [docs.rs/fk-geometry-hyperbolic](https://docs.rs/fk-geometry-hyperbolic)

## License

Licensed under either of [Apache License, Version 2.0](https://github.com/Dastfox/feuclide/blob/main/LICENSE-APACHE) or
[MIT license](https://github.com/Dastfox/feuclide/blob/main/LICENSE-MIT) at your option.
