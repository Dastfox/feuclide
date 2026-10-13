//! What an app draws for its player and its developer over the image: the frame rate
//! ([`fps`]), the list of its keys ([`key_list`]).
//!
//! Readouts are written into `fk-render`'s [`TextPanel`](fk_render::TextPanel), drawn over
//! everything. Generic over nothing: no geometry, no game.

pub mod fps;
pub mod key_list;
