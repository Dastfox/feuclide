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
//! [`RawInput`] holds what happened since the previous frame, on physical keys. The [`keymap`]
//! turns it into the actions a game declares, with bindings the player can change.
//!
//! The app is built for one geometry, chosen at startup with [`App::set_geometry`]. This crate
//! never names a concrete geometry.

mod app;
pub mod input;
pub mod keymap;
mod profile;
mod runner;
mod schedule;
mod time;
mod window;

pub use app::{ActiveGeometry, App, Plugin};
pub use input::{ButtonInput, CursorGrab, KeyCode, MouseButton, RawInput};
pub use profile::FrameProfile;
pub use runner::RunError;
pub use schedule::{FixedUpdate, PostUpdate, PreUpdate, Render, Startup, Update};
pub use time::Time;
pub use window::{AppExit, Handover, Splash, WindowConfig, WindowHandle, WindowMode, WindowSize};

/// The platform's per-user configuration directory for the application `app_name`, for example
/// `~/.config/<name>` on Linux or `%APPDATA%\<name>\config` on Windows.
///
/// `None` when the platform has no home directory to put it in.
pub fn config_dir(app_name: &str) -> Option<std::path::PathBuf> {
    directories::ProjectDirs::from("", "", app_name).map(|dirs| dirs.config_dir().to_path_buf())
}
