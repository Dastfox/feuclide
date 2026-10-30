//! Application builder and main loop: window, input, time, schedules and the key mapper.
//!
//! An [`App`] owns one `bevy_ecs` [`World`](bevy_ecs::world::World). [`App::run`] opens a window
//! and drives the world from the winit event loop, one frame per redraw:
//!
//! 1. [`Startup`], once, after the window exists.
//! 2. [`PreUpdate`]: input is read here (the [`keymap`] updates its actions).
//! 3. [`FixedUpdate`], zero or more times: one run per [`Time::fixed_step`] of accumulated time.
//! 4. [`Update`], [`PostUpdate`], [`Render`].
//!
//! [`RawInput`] holds what happened since the previous frame, on physical keys, the mouse and
//! the gamepads (read through gilrs before each frame, every pad as one). The [`keymap`]
//! turns it into the actions a game declares, with bindings the player can change.
//!
//! The app is built for one geometry, chosen at startup with [`App::set_geometry`]. This crate
//! never names a concrete geometry.
//!
//! Constants live in [`tuning`] files: TOML compiled in, overridden from `FK_TUNING_DIR` and
//! reloaded on save.

mod app;
mod drops;
pub mod input;
pub mod keymap;
mod pad;
pub mod persist;
mod profile;
mod progress;
mod runner;
mod schedule;
mod time;
pub mod tuning;
mod window;

pub use app::{ActiveGeometry, App, Plugin};
pub use drops::{FrameDrops, FrameEvents, FrameNotes};
pub use input::{
    ButtonInput, CursorGrab, Direction, KeyCode, MouseButton, PadButton, RawInput, Stick,
};
pub use profile::FrameProfile;
pub use progress::Progress;
pub use runner::RunError;
pub use schedule::{FixedUpdate, PostUpdate, PreUpdate, Render, Startup, Update};
pub use time::Time;
pub use window::{
    AppExit, Handover, Icon, Splash, WindowConfig, WindowHandle, WindowMode, WindowSize,
};

/// The platform's per-user configuration directory for the application `app_name`, for example
/// `~/.config/<name>` on Linux or `%APPDATA%\<name>\config` on Windows.
///
/// `None` when the platform has no home directory to put it in.
pub fn config_dir(app_name: &str) -> Option<std::path::PathBuf> {
    directories::ProjectDirs::from("", "", app_name).map(|dirs| dirs.config_dir().to_path_buf())
}
