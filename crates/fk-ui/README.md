# fk-ui

What an app draws over the image for its player and its developer. Part of
[Feuclide](https://github.com/Dastfox/feuclide), a game engine in Rust whose core never assumes
Euclidean geometry.

The frame rate in a corner (`fps`), the list of every key the app reads (`key_list`), written into
`fk-render`'s `TextPanel`.

## Bevy counterpart

`bevy_dev_tools`' `FpsOverlayPlugin`, and the key lists games write themselves.
