//! Tools for flat worlds: E³ with y up and a ground at height 0: standing on it ([`ground`]),
//! static scenery merged and boxes measured ([`kit`]), ground as a height field ([`land`]), a first-person body ([`first_person`]).
//!
//! Unlike the rest of the engine this crate names its geometry, [`E3`](fk_geometry_euclidean::E3):
//! it is a leaf an application picks, as it picks a geometry crate.

pub mod first_person;
pub mod ground;
pub mod kit;
pub mod land;
