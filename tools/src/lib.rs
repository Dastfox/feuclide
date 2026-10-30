//! Offline tools for Feuclide, each a binary of this package:
//!
//! - `refine`: splits a binary glTF until no edge, embedded into a geometry, is longer than a
//!   bound (`fk_assets::refine`), and writes it back as binary glTF.
//! - `golden`: draws the golden scenes both ways, says how they differ and writes the images
//!   (`--bless` into `tools/golden/`, the blessed ones); see [`golden`].
//! - `drops`: tells the frame drops an app wrote (`FK_DROPS`, `fk_render::drops`) as a table,
//!   or how often each cause came back; see [`drops`].
//!
//! `cargo run -p fk-tools --bin <tool> -- --help` says how each is used.

pub mod drops;
pub mod golden;
