# fk-sky

The day and the night. Part of [Feuclide](https://github.com/Dastfox/feuclide), a game engine in
Rust whose core never assumes Euclidean geometry.

A `Clock` (stopped, scrubbed, sped up through `ClockInput`), the sun's and the moon's arcs across
the sky, and the colours of every hour keyed round the day, written into `fk-render`'s `Lighting`
each frame by `DaySkyPlugin`. Tuned in a TOML file (`daynight.toml`, the crate's own or the app's),
reloaded on save under `FK_TUNING_DIR`.

## Bevy counterpart

A time-of-day plugin over `bevy_light`'s sun and the atmosphere: Bevy leaves the clock and the
keyed colours to the app.
