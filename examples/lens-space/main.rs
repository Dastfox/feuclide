//! Fly through the lens space L(7, 2): the 3-sphere divided by the screw motion that goes a
//! seventh of the way round a great circle while turning two sevenths about it. The things in
//! the room are seen seven times down the circle ahead, each copy turned two sevenths more,
//! and on round behind you; fly straight on along the circle and you are back in the room after
//! a seventh of the way.
//!
//! The demo of E3's lens space: `cargo run --example lens-space --release`. Click to grab the
//! mouse, WASD (ZQSD on AZERTY) to fly where you look, Space / Ctrl to rise and sink, Shift to
//! go fast, F1 for every key.

#[path = "../fly_curved.rs"]
mod fly_curved;
#[path = "../key_list.rs"]
mod key_list;

use fk_geometry_spherical::S3;
use fk_quotient::Quotient;
use fly_curved::{Manifold, Thing};

fn main() {
    let thing = |at: [f64; 3], size, cube, color| Thing {
        at,
        size,
        cube,
        color,
    };
    // A ring of cubes round the circle's axis, ahead, and a ball off it: the turn shows.
    let palette = [0xd9774b, 0xe0b04f, 0x8fae8b, 0x6f8fb5, 0xd98c9a, 0xf4f0e8];
    let mut things: Vec<Thing> = palette
        .iter()
        .enumerate()
        .map(|(i, &color)| {
            let a = i as f64 * std::f64::consts::TAU / 6.0;
            thing([0.3 * a.cos(), 0.3 * a.sin(), -0.25], 0.05, true, color)
        })
        .collect();
    things.push(thing([0.25, 0.0, -0.1], 0.06, false, 0xf4f0e8));
    fly_curved::run_manifold(Manifold {
        name: "lens-space",
        quotient: Quotient::<S3>::lens(7, 2),
        cell: None,
        things,
        far: 2.0 * std::f64::consts::PI - 0.05,
        horizon: 2.0 * std::f64::consts::PI,
        speeds: [0.3, 1.0],
        fog: 0.1,
    });
}
