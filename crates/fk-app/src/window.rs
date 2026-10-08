use std::sync::Arc;

use bevy_ecs::resource::Resource;
use bevy_ecs::world::World;
use raw_window_handle::{
    DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle, WindowHandle as RawHandle,
};

use crate::App;

/// How the window opens.
#[derive(Clone, Debug)]
pub struct WindowConfig {
    /// Title shown by the window manager.
    pub title: String,
    /// Inner size in logical pixels.
    pub size: (u32, u32),
    /// Frames per second the loop waits down to, or `None` to run as fast as frames are drawn.
    /// Only needed while nothing draws with vsync.
    pub frame_rate_cap: Option<f64>,
    /// Whether the window opens filling the screen ([`WindowMode`]).
    pub fullscreen: bool,
    /// What shows while the app starts, if anything: see [`Splash`].
    pub splash: Option<Splash>,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            title: "Feuclide".to_owned(),
            size: (1280, 720),
            frame_rate_cap: None,
            fullscreen: false,
            splash: None,
        }
    }
}

/// An image shown at once in a small borderless window of its own, in the middle of the
/// screen, while the app starts: its window opens hidden and only shows, the splash closing,
/// once its first frame is drawn. Without one, the window shows as it opens, empty until then.
#[derive(Clone, Debug)]
pub struct Splash {
    /// Width in physical pixels.
    pub width: u32,
    /// Height in physical pixels.
    pub height: u32,
    /// Opaque sRGB RGBA, row by row from the top, `width × height × 4` bytes.
    pub pixels: Arc<[u8]>,
}

/// Whether the window fills the screen (borderless, on the monitor it is on), as a resource.
/// The loop applies it after every frame, and to every app handed the window
/// ([`Handover`]): an app that does not set it takes the window out of fullscreen.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WindowMode {
    /// Whether it fills the screen.
    pub fullscreen: bool,
}

/// The window, as a source of the platform handles a renderer draws into.
///
/// Inserted as a resource before [`Startup`](crate::Startup) runs.
#[derive(Resource, Clone, Debug)]
pub struct WindowHandle(pub(crate) Arc<winit::window::Window>);

impl HasWindowHandle for WindowHandle {
    fn window_handle(&self) -> Result<RawHandle<'_>, HandleError> {
        self.0.window_handle()
    }
}

impl HasDisplayHandle for WindowHandle {
    fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
        self.0.display_handle()
    }
}

/// Size of the window's drawable area, kept current as the window is resized.
///
/// Inserted as a resource before [`Startup`](crate::Startup) runs.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct WindowSize {
    /// Width in physical pixels.
    pub width: u32,
    /// Height in physical pixels.
    pub height: u32,
    /// Physical pixels per logical pixel.
    pub scale_factor: f64,
}

/// Asks the loop to close the window and return from [`App::run`](crate::App::run) once the
/// current frame is done, even with a next app prepared ([`Handover`]).
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AppExit {
    requested: bool,
}

impl AppExit {
    /// Stops the app after this frame.
    pub fn request(&mut self) {
        self.requested = true;
    }

    /// Whether [`request`](Self::request) was called.
    pub fn is_requested(&self) -> bool {
        self.requested
    }
}

/// What moves from the old app's world into the new one's.
type Carry = Box<dyn FnOnce(&mut World, &mut World) + Send + Sync>;

/// The app that takes over the window from this one.
///
/// [`prepare`](Self::prepare) builds it on a thread of its own while this app runs on: its
/// [`Startup`](crate::Startup) runs there without the window, then its
/// [`Render`](crate::Render) schedule once, so that a renderer can get ready (compile, upload)
/// before it has anything to draw to. Once [`go`](Self::go) is called, at the end of the
/// frame, the loop moves what its `carry` says from the old world into the new one, drops
/// the old app on a thread of its own, and gives the new one the window, with the old one's held keys and cursor grab.
/// The new app's [`WindowConfig`](crate::WindowConfig) is ignored. If it is not ready then,
/// the loop waits for it. Without a window ([`App::update`]) nothing happens. Exiting through
/// [`AppExit`] never hands over: a prepared app is dropped with this one.
#[derive(Resource, Default)]
pub struct Handover {
    next: Option<std::thread::JoinHandle<App>>,
    carry: Option<Carry>,
    going: bool,
}

impl Handover {
    /// Builds the next app with `build` on a thread of its own, and starts it there.
    pub fn prepare(&mut self, build: impl FnOnce() -> App + Send + 'static) {
        let thread = std::thread::Builder::new()
            .name("next app".to_owned())
            .spawn(move || {
                let mut app = build();
                app.prepare();
                app
            });
        match thread {
            Ok(thread) => self.next = Some(thread),
            Err(error) => tracing::error!("cannot prepare the next app: {error}"),
        }
    }

    /// Hands the window to the prepared app once this frame is done, moving what `carry`
    /// says from the old app's world (first) into the new one's (second).
    pub fn go(&mut self, carry: impl FnOnce(&mut World, &mut World) + Send + Sync + 'static) {
        self.carry = Some(Box::new(carry));
        self.going = true;
    }

    /// Whether [`go`](Self::go) was called with an app prepared.
    pub fn is_going(&self) -> bool {
        self.going && self.next.is_some()
    }

    /// Gives up the app being prepared, if any, and what it was to carry: it is dropped once
    /// its thread is done.
    pub fn cancel(&mut self) {
        self.next = None;
        self.carry = None;
        self.going = false;
    }

    /// Whether an app is being prepared, or is ready.
    pub fn is_prepared(&self) -> bool {
        self.next.is_some()
    }

    /// Whether the app being prepared is ready to take over.
    pub fn is_ready(&self) -> bool {
        self.next
            .as_ref()
            .is_some_and(std::thread::JoinHandle::is_finished)
    }

    pub(crate) fn take(&mut self) -> Option<(std::thread::JoinHandle<App>, Option<Carry>)> {
        if !std::mem::take(&mut self.going) {
            return None;
        }
        Some((self.next.take()?, self.carry.take()))
    }
}
