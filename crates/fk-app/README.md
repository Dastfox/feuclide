# fk-app

App, window, input and main loop. Part of [Feuclide](https://crates.io/crates/feuclide), a game engine in Rust whose core never assumes Euclidean geometry. Source, examples and design notes: [github.com/Dastfox/feuclide](https://github.com/Dastfox/feuclide).

An `App` owns one `bevy_ecs` `World` and drives it from the winit event loop, one frame per
redraw: `Startup`, then `PreUpdate` (input), `FixedUpdate` zero or more times on a fixed step,
`Update`, `PostUpdate` and `Render`. Plugins add systems and resources. The key mapper turns
physical keys into the actions a game declares, with bindings the player can change. A `Handover`
gives the window to a second app built on a thread, on the same GPU device. The app is built for
one geometry, chosen at startup; this crate never names a concrete one.

## Bevy counterpart

`bevy_app` + `bevy_window` + `bevy_winit` + `bevy_input` + `bevy_time`.

Same App/Plugin/schedules model, slimmed down. It adds a key mapper with player rebinding (Bevy
leaves that to third-party crates like leafwing), a fixed step, and handing the window over to a
second app built on a thread.

## Links

- Engine and umbrella crate: [`feuclide`](https://crates.io/crates/feuclide)
- Repository: [github.com/Dastfox/feuclide](https://github.com/Dastfox/feuclide)
- API docs: [docs.rs/fk-app](https://docs.rs/fk-app)

## License

Licensed under either of [Apache License, Version 2.0](https://github.com/Dastfox/feuclide/blob/main/LICENSE-APACHE) or
[MIT license](https://github.com/Dastfox/feuclide/blob/main/LICENSE-MIT) at your option.
