//! Fly through hyperbolic space, tiled by right-angled dodecahedra `{5,3,4}` (N for cubes five
//! round an edge `{4,3,5}`, dodecahedra five round `{5,3,5}`, and cubes six round `{4,3,6}`,
//! their vertices at infinity). Space grows so fast that a few cells away the fog takes all:
//! the horizon is the fog's.
//!
//! The demo of E3: `cargo run --example fly-h3 --release`. Click to grab the mouse, WASD (ZQSD
//! on AZERTY) to fly where you look, Space / Ctrl to rise and sink, Shift to go fast, Tab for
//! the depth as distance, F1 for every key.

#[path = "../fly_curved.rs"]
mod fly_curved;
#[path = "../key_list.rs"]
mod key_list;

use fk_assets::Cell;
use fk_geometry_hyperbolic::H3;

fn main() {
    fly_curved::run::<H3>(fly_curved::Setup {
        name: "fly-h3",
        tilings: &[
            (Cell::Dodecahedron, 4),
            (Cell::Cube, 5),
            (Cell::Dodecahedron, 5),
            (Cell::Cube, 6),
        ],
        far: 7.0,
        within: 6.0,
        most: 4000,
        speeds: [0.8, 2.5],
        fog: 0.35,
    });
}
