//! Fly through the Seifert–Weber space, a closed hyperbolic 3-manifold: one dodecahedron of
//! `{5,3,5}`, five round every edge, each face glued to the opposite one after three tenths of
//! a turn. Fly through a face and you come back in through the opposite one, turned; the
//! things in the room are seen again in every copy of it, each turned by the gluings, out to
//! the fog.
//!
//! The demo of E3's compact manifold: `cargo run --example seifert-weber --release`. Click to
//! grab the mouse, WASD (ZQSD on AZERTY) to fly where you look, Space / Ctrl to rise and sink,
//! Shift to go fast, F1 for every key.

#[path = "../fly_curved.rs"]
mod fly_curved;
#[path = "../key_list.rs"]
mod key_list;

use fk_assets::{Cell, Tiling};
use fk_geometry_hyperbolic::H3;
use fly_curved::{Manifold, Thing};

fn main() {
    let cell = Tiling::<H3>::regular(Cell::Dodecahedron, 5).expect("{5,3,5} is hyperbolic");
    let quotient = cell.glued(3.0 * std::f64::consts::TAU / 10.0);
    let thing = |at: [f64; 3], size, cube, color| Thing {
        at,
        size,
        cube,
        color,
    };
    fly_curved::run_manifold(Manifold {
        name: "seifert-weber",
        quotient,
        cell: Some(cell),
        things: vec![
            thing([0.0, 0.0, -0.5], 0.12, false, 0xd9774b),
            thing([0.45, 0.2, 0.0], 0.08, true, 0x8fae8b),
            thing([-0.2, -0.45, 0.25], 0.1, true, 0x6f8fb5),
            thing([0.0, 0.5, 0.35], 0.07, false, 0xf4f0e8),
        ],
        far: 6.0,
        horizon: 3.0,
        speeds: [0.6, 2.0],
        fog: 0.4,
    });
}
