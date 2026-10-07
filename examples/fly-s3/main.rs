//! Fly through the 3-sphere, tiled by the 120-cell `{5,3,3}` (N for the 8-cell `{4,3,3}`).
//! Space closes up: fly straight on and you come back to where you started, and everything is
//! seen twice, directly and the long way round, smaller as it nears the antipode and growing
//! again past it.
//!
//! The demo of E3: `cargo run --example fly-s3 --release`. Click to grab the mouse, WASD (ZQSD
//! on AZERTY) to fly where you look, Space / Ctrl to rise and sink, Shift to go fast, Tab for
//! the depth as distance, F1 for every key.

#[path = "../fly_curved.rs"]
mod fly_curved;
#[path = "../key_list.rs"]
mod key_list;

use fk_assets::Cell;
use fk_geometry_spherical::S3;

fn main() {
    fly_curved::run::<S3>(fly_curved::Setup {
        name: "fly-s3",
        tilings: &[(Cell::Dodecahedron, 3), (Cell::Cube, 3)],
        far: 2.0 * std::f64::consts::PI - 0.05,
        within: 4.0,
        most: 200,
        speeds: [0.4, 1.2],
        fog: 0.08,
    });
}
