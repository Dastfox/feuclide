# fk-flat

Tools for flat (E³) worlds, with y up and a ground. Part of
[Feuclide](https://github.com/Dastfox/feuclide), a game engine in Rust whose core never assumes
Euclidean geometry: this crate does, and is a leaf an application picks, like the geometry
crates.

Upright poses by a yaw, the free spot nearest a point for a body among colliders and a distance
field.

## Bevy counterpart

What a flat-world game writes over Bevy's `Transform` and a physics crate's spatial queries.
