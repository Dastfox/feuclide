//! The tour's stations, in order: flat space, hyperbolic space, the 3-sphere, and a closed
//! manifold made from each.

use std::f64::consts::{PI, TAU};

use fk_app::App;
use fk_assets::{Cell, Tiling};
use fk_geometry::Geometry;
use fk_geometry_euclidean::E3;
use fk_geometry_hyperbolic::H3;
use fk_geometry_spherical::S3;
use fk_quotient::Quotient;

use crate::station::{Look, Rails, Space, Station, Thing, Tour, app};

/// How many stations there are.
pub const COUNT: usize = 10;

/// Builds station `index` (modulo [`COUNT`]).
pub fn build(index: usize) -> App {
    let index = index % COUNT;
    let tour = Tour {
        index,
        build,
        count: COUNT,
    };
    match index {
        0 => app(cubic_lattice(), tour),
        1 => app(three_torus(), tour),
        2 => app(
            hyperbolic(Cell::Cube, 5, "{4,3,5}", "five cubes round every edge"),
            tour,
        ),
        3 => app(
            hyperbolic(Cell::Dodecahedron, 4, "{5,3,4}", "right-angled dodecahedra"),
            tour,
        ),
        4 => app(
            hyperbolic(
                Cell::Cube,
                6,
                "{4,3,6}",
                "cubes with their corners at infinity",
            ),
            tour,
        ),
        5 => app(seifert_weber(), tour),
        6 => app(
            spherical(Cell::Cube, "The 8-cell", "eight cubes fill the 3-sphere"),
            tour,
        ),
        7 => app(
            spherical(
                Cell::Dodecahedron,
                "The 120-cell",
                "120 dodecahedra fill the 3-sphere",
            ),
            tour,
        ),
        8 => app(poincare_sphere(), tour),
        _ => app(lens_space(), tour),
    }
}

fn thing(at: [f64; 3], size: f64, cube: bool, color: u32) -> Thing {
    Thing {
        at,
        size,
        cube,
        color,
    }
}

/// Things in a domain of inradius `r`, off the line the eye sets out along.
fn room(r: f64) -> Vec<Thing> {
    vec![
        thing([0.45 * r, 0.25 * r, -0.4 * r], 0.14 * r, false, 0xd9774b),
        thing([-0.5 * r, -0.2 * r, -0.1 * r], 0.12 * r, true, 0x8fae8b),
        thing([-0.15 * r, 0.5 * r, 0.3 * r], 0.1 * r, true, 0x6f8fb5),
        thing([0.25 * r, -0.5 * r, 0.35 * r], 0.09 * r, false, 0xf4f0e8),
    ]
}

const FLAT_SKY: [u32; 3] = [0x3a5170, 0x9fb3c8, 0x2a3242];
const HYPERBOLIC_SKY: [u32; 3] = [0x1d2440, 0x2c3558, 0x161b30];
const SPHERICAL_SKY: [u32; 3] = [0x40263a, 0x6e4258, 0x24161f];

fn cubic_lattice() -> Station<E3> {
    Station {
        name: "Flat space, E3",
        about: "{4,3,4}: the cubic lattice, four cubes round every edge",
        space: Space::Tiling {
            tiling: Tiling::regular(Cell::Cube, 4).expect("{4,3,4} is flat"),
            within: 9.0,
            most: 3000,
        },
        rails: Rails {
            speed: 0.9,
            yaw: (0.18, 23.0),
            pitch: (0.1, 17.0),
        },
        look: Look {
            far: 10.0,
            fog: 0.16,
            sky: FLAT_SKY,
        },
    }
}

fn three_torus() -> Station<E3> {
    let cell = Tiling::<E3>::regular(Cell::Cube, 4).expect("{4,3,4} is flat");
    let side = 2.0 * cell.inradius();
    let translations: Vec<_> = (0..3).map(|i| E3::origin_frame(i) * side).collect();
    let r = cell.inradius();
    Station {
        name: "The flat 3-torus",
        about: "one cube, each face glued to the opposite one",
        space: Space::Manifold {
            quotient: Quotient::from_translations(E3::origin(), &translations),
            things: room(r),
            cell: Some(cell),
            horizon: 9.0,
        },
        rails: Rails {
            speed: 0.7 * r,
            yaw: (0.25, 19.0),
            pitch: (0.15, 13.0),
        },
        look: Look {
            far: 10.0,
            fog: 0.16,
            sky: FLAT_SKY,
        },
    }
}

fn hyperbolic(cell: Cell, around: u32, symbol: &'static str, about: &'static str) -> Station<H3> {
    let name = match symbol {
        "{4,3,5}" => "Hyperbolic space, H3: {4,3,5}",
        "{5,3,4}" => "Hyperbolic space, H3: {5,3,4}",
        _ => "Hyperbolic space, H3: {4,3,6}",
    };
    Station {
        name,
        about,
        space: Space::Tiling {
            tiling: Tiling::regular(cell, around).expect("the tiling is hyperbolic"),
            within: 6.0,
            most: 4000,
        },
        rails: Rails {
            speed: 0.7,
            yaw: (0.2, 21.0),
            pitch: (0.12, 15.0),
        },
        look: Look {
            far: 7.0,
            fog: 0.35,
            sky: HYPERBOLIC_SKY,
        },
    }
}

fn seifert_weber() -> Station<H3> {
    let cell = Tiling::<H3>::regular(Cell::Dodecahedron, 5).expect("{5,3,5} is hyperbolic");
    let r = cell.inradius();
    Station {
        name: "The Seifert-Weber space",
        about: "a dodecahedron, opposite faces glued after 3/10 of a turn",
        space: Space::Manifold {
            quotient: cell.glued(3.0 * TAU / 10.0),
            things: room(r),
            cell: Some(cell),
            horizon: 3.0,
        },
        rails: Rails {
            speed: 0.5,
            yaw: (0.22, 19.0),
            pitch: (0.14, 14.0),
        },
        look: Look {
            far: 6.0,
            fog: 0.4,
            sky: HYPERBOLIC_SKY,
        },
    }
}

fn spherical(cell: Cell, name: &'static str, about: &'static str) -> Station<S3> {
    Station {
        name,
        about,
        space: Space::Tiling {
            tiling: Tiling::regular(cell, 3).expect("the tiling is spherical"),
            within: 4.0,
            most: 200,
        },
        rails: Rails {
            speed: 0.35,
            yaw: (0.15, 25.0),
            pitch: (0.1, 18.0),
        },
        look: Look {
            far: 2.0 * PI - 0.05,
            fog: 0.08,
            sky: SPHERICAL_SKY,
        },
    }
}

fn poincare_sphere() -> Station<S3> {
    let cell = Tiling::<S3>::regular(Cell::Dodecahedron, 3).expect("the 120-cell is spherical");
    let r = cell.inradius();
    Station {
        name: "The Poincare homology sphere",
        about: "a dodecahedron, opposite faces glued after 1/10 of a turn",
        space: Space::Manifold {
            quotient: cell.glued(TAU / 10.0),
            things: room(r),
            cell: Some(cell),
            horizon: 2.0 * PI,
        },
        rails: Rails {
            speed: 0.3,
            yaw: (0.2, 21.0),
            pitch: (0.12, 16.0),
        },
        look: Look {
            far: 2.0 * PI - 0.05,
            fog: 0.08,
            sky: SPHERICAL_SKY,
        },
    }
}

fn lens_space() -> Station<S3> {
    // A ring of cubes round the circle the eye flies along, and a ball off it: each copy down
    // the circle is turned two sevenths further.
    let palette = [0xd9774b, 0xe0b04f, 0x8fae8b, 0x6f8fb5, 0xd98c9a, 0xf4f0e8];
    let mut things: Vec<Thing> = palette
        .iter()
        .enumerate()
        .map(|(i, &color)| {
            let a = i as f64 * TAU / 6.0;
            thing([0.3 * a.cos(), 0.3 * a.sin(), -0.25], 0.05, true, color)
        })
        .collect();
    things.push(thing([0.25, 0.0, -0.1], 0.06, false, 0xf4f0e8));
    Station {
        name: "The lens space L(7,2)",
        about: "the 3-sphere divided by a screw: 1/7 along, 2/7 of a turn",
        space: Space::Manifold {
            quotient: Quotient::lens(7, 2),
            cell: None,
            things,
            horizon: 2.0 * PI,
        },
        rails: Rails {
            speed: 0.3,
            yaw: (0.06, 27.0),
            pitch: (0.04, 19.0),
        },
        look: Look {
            far: 2.0 * PI - 0.05,
            fog: 0.1,
            sky: SPHERICAL_SKY,
        },
    }
}
