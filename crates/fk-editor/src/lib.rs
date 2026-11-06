//! An editor for Feuclide scenes and for an app's tuning files, laid out after Bevy's editor
//! prototypes: a scene tree, properties and environment panes, a viewport drawn by the
//! engine's own renderer, and a tuning pane.
//!
//! - **Viewport.** The scene is drawn by [`fk_render::Offscreen`] (the raster path) on a
//!   headless device and shown in the pane; it is drawn again only when the scene, the light,
//!   the camera or the pane's size change. An orbiting camera, picking by the meshes' triangles,
//!   the selection outlined, a gizmo to move it along the world's axes.
//! - **Scene tree and properties.** Things are shapes of [`fk_render::shapes`] in a colour,
//!   maybe with a point light, posed by [`fk_scene::Pose`] under `bevy_ecs`'s `ChildOf`. Their
//!   name, place, turn, sizes, colour and light are edited in the properties pane; the scene's
//!   [`fk_render::Lighting`] in the environment pane.
//! - **Scene files.** TOML a person can read: the lighting and every thing with its parent.
//! - **Undo.** By whole-scene snapshots: every edit, from any pane or the gizmo, is undone alike.
//! - **Tuning.** A folder of [`fk_app::tuning`] files, every value a field, edited with the
//!   files' comments and layout kept. With "Live", each edit is saved as it ends, and an app run
//!   with `FK_TUNING_DIR` set to that folder reloads it at once. The folder is picked in the
//!   editor ("Open folder…") and opened again next time.
//!
//! Only E³ scenes are edited for now.
//!
//! ```no_run
//! fk_editor::run(fk_editor::Options::default()).expect("the editor runs");
//! ```
//!
//! The `fk-editor` binary does the same from the command line:
//! `fk-editor [scene.toml] [--tuning <folder>]`.

mod app;
mod core;
mod panes;
mod scene;
mod undo;
mod viewport;

use std::path::PathBuf;

/// What the editor opens.
#[derive(Clone, Debug, Default)]
pub struct Options {
    /// The scene file: read if it exists, and where the scene is saved. `scene.toml` in the
    /// working directory when none is given; an example scene is shown until one is read.
    pub scene: Option<PathBuf>,
    /// The folder of tuning files. When none is given, the last one used (kept in the editor's
    /// storage when it closes or a folder is picked), then `FK_TUNING_DIR`, then
    /// `assets/tuning` in the working directory. Another is picked in the editor with the
    /// system's folder dialog.
    pub tuning: Option<PathBuf>,
}

/// Opens the editor's window and runs until it is closed.
///
/// # Errors
///
/// When the window or its renderer cannot be made.
pub fn run(options: Options) -> eframe::Result {
    app::run(options)
}
