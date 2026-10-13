# fk-math

Raw numerics. Part of [Feuclide](https://crates.io/crates/feuclide), a game engine in Rust whose core never assumes Euclidean geometry. Source, examples and design notes: [github.com/Dastfox/feuclide](https://github.com/Dastfox/feuclide).

The numbers underneath every space: nothing in this crate has geometric meaning, a vector here
is a container of numbers, not a point. All CPU-side math is `f64` (on nalgebra), because
coordinates in curved models grow exponentially with distance (hyperboloid coordinates grow like
`cosh d`); single precision is kept for the GPU boundary, after poses have been made
camera-relative. It also holds the numerically stable series (`sinc`, `sinhc`, …) that the
exponential and logarithm maps need near zero, and helpers for the Minkowski form of the
hyperboloid model.

## Bevy counterpart

`bevy_math` (on glam).

nalgebra in `f64`; `f32` only at the GPU boundary. Adds stable series (`sinc`, `sinhc`, …) for
the exp/log maps. No shapes or geometry here: it's numbers only.

## Links

- Engine and umbrella crate: [`feuclide`](https://crates.io/crates/feuclide)
- Repository: [github.com/Dastfox/feuclide](https://github.com/Dastfox/feuclide)
- API docs: [docs.rs/fk-math](https://docs.rs/fk-math)

## License

Licensed under either of [Apache License, Version 2.0](https://github.com/Dastfox/feuclide/blob/main/LICENSE-APACHE) or
[MIT license](https://github.com/Dastfox/feuclide/blob/main/LICENSE-MIT) at your option.
