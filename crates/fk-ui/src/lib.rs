//! What an app draws for its player and its developer over the image: the frame rate
//! ([`fps`]), the list of its keys ([`key_list`]), a pointer of its own ([`pointer`]), a debug
//! console ([`console`]), menu pages as a model ([`sheet`]), and drawing on the CPU for an
//! overlay ([`canvas`]).
//!
//! Readouts are written into `fk-render`'s [`TextPanel`](fk_render::TextPanel), drawn over
//! everything. Generic over nothing: no geometry, no game.

pub mod canvas;
pub mod console;
pub mod fps;
pub mod key_list;
pub mod pointer;
pub mod sheet;
