# fk-game

The showcase tour. Part of [Feuclide](https://crates.io/crates/feuclide), a game engine in Rust whose core never assumes Euclidean geometry. Source, examples and design notes: [github.com/Dastfox/feuclide](https://github.com/Dastfox/feuclide).

A guided tour through every space the engine draws, with the camera on rails: flat space,
hyperbolic space, the 3-sphere, then the closed manifolds made from each (the flat 3-torus, the
Seifert–Weber space, the Poincaré homology sphere and a lens space). Each station lasts about
thirty seconds; the next one is built on a thread while this one plays, then given the window,
and the last image datamoshes into the next.

```sh
cargo run -p fk-game --release [-- <station>]
```

N goes on to the next station, Esc quits. This crate is not published to crates.io.

## Bevy counterpart

Bevy's `examples/` and showcase.

A guided tour on rails through every space the engine draws.

## Links

- Engine and umbrella crate: [`feuclide`](https://crates.io/crates/feuclide)
- Repository: [github.com/Dastfox/feuclide](https://github.com/Dastfox/feuclide)

## License

Licensed under either of [Apache License, Version 2.0](https://github.com/Dastfox/feuclide/blob/main/LICENSE-APACHE) or
[MIT license](https://github.com/Dastfox/feuclide/blob/main/LICENSE-MIT) at your option.
