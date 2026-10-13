# fk-audio

Sound. Part of [Feuclide](https://crates.io/crates/feuclide), a game engine in Rust whose core never assumes Euclidean geometry. Source, examples and design notes: [github.com/Dastfox/feuclide](https://github.com/Dastfox/feuclide).

Clips decoded from MP3 or FLAC (or made from samples) play as voices whose gain and rate glide.
A sound plays straight out, in a room (through a reverb the app sets per place, gliding between
rooms), or outside (low-passed and hushed together). An `Emitter` places its sounds about the
ear, camera-relative, panning between the ears and fading with distance. Sounds sort into
channels with their own volumes. Without an output device (a server, CI) everything stays silent
and the app runs the same. The backend is kira; nothing of it shows through the API.

## Bevy counterpart

`bevy_audio` (on rodio).

Built on kira, like the third-party `bevy_kira_audio`. Adds reverb rooms, directional emitters and
gliding voices, and stays silent without an audio device.

## Links

- Engine and umbrella crate: [`feuclide`](https://crates.io/crates/feuclide)
- Repository: [github.com/Dastfox/feuclide](https://github.com/Dastfox/feuclide)
- API docs: [docs.rs/fk-audio](https://docs.rs/fk-audio)

## License

Licensed under either of [Apache License, Version 2.0](https://github.com/Dastfox/feuclide/blob/main/LICENSE-APACHE) or
[MIT license](https://github.com/Dastfox/feuclide/blob/main/LICENSE-MIT) at your option.
