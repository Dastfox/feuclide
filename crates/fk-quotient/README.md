# fk-quotient

Quotient and portal spaces. Part of [Feuclide](https://crates.io/crates/feuclide), a game engine in Rust whose core never assumes Euclidean geometry. Source, examples and design notes: [github.com/Dastfox/feuclide](https://github.com/Dastfox/feuclide).

A geometry divided by a discrete group of isometries (its deck group): flat tori, closed
hyperbolic and spherical manifolds. The fundamental domain is the Dirichlet domain about a
centre, defined by the metric alone, so it is the same code in every geometry. Crossing a face
brings a pose back into the domain and records the crossing in its `ChartTag`; `translates` lists
the copies of the domain a renderer draws and the ghosts a collider tests; `shortest` is the
distance in the quotient. Also portals.

## Bevy counterpart

**None.**

Quotient spaces (flat tori, closed hyperbolic and spherical manifolds), where leaving one side
brings you back in another; also portals. Bevy has no notion of this.

## Links

- Engine and umbrella crate: [`feuclide`](https://crates.io/crates/feuclide)
- Repository: [github.com/Dastfox/feuclide](https://github.com/Dastfox/feuclide)
- API docs: [docs.rs/fk-quotient](https://docs.rs/fk-quotient)

## License

Licensed under either of [Apache License, Version 2.0](https://github.com/Dastfox/feuclide/blob/main/LICENSE-APACHE) or
[MIT license](https://github.com/Dastfox/feuclide/blob/main/LICENSE-MIT) at your option.
