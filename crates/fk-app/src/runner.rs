//! The winit event loop that drives an [`App`].

use std::collections::HashSet;
use std::fmt;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::{Duration, Instant};

use fk_math::Real;
use fk_math::nalgebra::Vector2;
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalPosition, PhysicalSize};
use winit::error::{EventLoopError, OsError};
use winit::event::{
    DeviceEvent, DeviceId, ElementState, MouseButton, MouseScrollDelta, WindowEvent,
};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::PhysicalKey;
use winit::window::{CursorGrabMode, Fullscreen, Window, WindowId};

use crate::App;
use crate::input::{CursorGrab, RawInput};
use crate::window::{AppExit, Handover, Splash, WindowHandle, WindowMode, WindowSize};

/// Why [`App::run`] stopped early.
#[derive(Debug)]
pub enum RunError {
    /// The event loop could not be created or failed while running.
    EventLoop(EventLoopError),
    /// The window could not be created.
    Window(OsError),
}

impl fmt::Display for RunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EventLoop(error) => write!(f, "event loop failed: {error}"),
            Self::Window(error) => write!(f, "could not create the window: {error}"),
        }
    }
}

impl std::error::Error for RunError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::EventLoop(error) => Some(error),
            Self::Window(error) => Some(error),
        }
    }
}

pub(crate) fn run(app: App) -> Result<(), RunError> {
    let event_loop = EventLoop::new().map_err(RunError::EventLoop)?;
    let frame_interval = app
        .window_config()
        .frame_rate_cap
        .filter(|fps| *fps > 0.0)
        .map(|fps| Duration::from_secs_f64(1.0 / fps));
    let mut runner = Runner {
        app,
        window: None,
        splash: None,
        fullscreen: false,
        focused: false,
        last_frame: None,
        frame_interval,
        swallowed: HashSet::new(),
        frames: FrameWatch::default(),
        error: None,
    };
    event_loop
        .run_app(&mut runner)
        .map_err(RunError::EventLoop)?;
    runner.error.map_or(Ok(()), Err)
}

struct Runner {
    app: App,
    window: Option<Arc<Window>>,
    /// The splash while the app starts behind it, its window hidden; `None` once it shows.
    splash: Option<SplashWindow>,
    /// Whether the window was last made to fill the screen.
    fullscreen: bool,
    focused: bool,
    last_frame: Option<Instant>,
    frame_interval: Option<Duration>,
    /// Mouse buttons whose press grabbed the cursor; their release is not passed on either.
    swallowed: HashSet<MouseButton>,
    frames: FrameWatch,
    error: Option<RunError>,
}

/// Watches the frame rate and logs its drops: a frame lasting much longer than the frames
/// before it. A drop is logged at once, then those that follow within [`Self::QUIET`] are
/// counted and logged together, so that a bad stretch is a few lines and not hundreds.
#[derive(Debug, Default)]
struct FrameWatch {
    /// The usual frame's length, seconds: a moving average of the frames that were not drops.
    usual: Option<Real>,
    /// Drops not logged yet: how many, the longest, and when the first was.
    pending: Option<(u32, Real, Instant)>,
    /// When a drop was last logged.
    logged: Option<Instant>,
}

impl FrameWatch {
    /// A frame is a drop when it lasts this many times the usual frame...
    const RATIO: Real = 2.0;
    /// ...and at least this long, seconds (below 30 frames a second).
    const LEAST: Real = 1.0 / 30.0;
    /// How much each frame that is not a drop counts in the usual frame.
    const WEIGHT: Real = 0.05;
    /// Seconds after a drop is logged during which drops are counted rather than logged.
    const QUIET: Real = 1.0;

    /// Counts a frame of `delta` seconds, drawn at `now`.
    fn frame(&mut self, delta: Real, now: Instant) {
        let usual = *self.usual.get_or_insert(delta);
        if delta >= Self::LEAST && delta > Self::RATIO * usual {
            let (count, longest, _) = self.pending.get_or_insert((0, 0.0, now));
            *count += 1;
            *longest = longest.max(delta);
        } else {
            self.usual = Some(usual + (delta - usual) * Self::WEIGHT);
        }
        let quiet = self
            .logged
            .is_some_and(|logged| now.duration_since(logged).as_secs_f64() < Self::QUIET);
        if quiet {
            return;
        }
        if let Some((count, longest, first)) = self.pending.take() {
            let fps = |seconds: Real| 1.0 / seconds.max(1e-6);
            if count == 1 {
                tracing::warn!(
                    "frame drop: {:.0} ms ({:.0} fps), usually {:.1} ms ({:.0} fps)",
                    longest * 1e3,
                    fps(longest),
                    usual * 1e3,
                    fps(usual),
                );
            } else {
                tracing::warn!(
                    "frame drops: {count} in {:.1}s, the longest {:.0} ms ({:.0} fps), usually                      {:.1} ms ({:.0} fps)",
                    now.duration_since(first).as_secs_f64(),
                    longest * 1e3,
                    fps(longest),
                    usual * 1e3,
                    fps(usual),
                );
            }
            self.logged = Some(now);
        }
    }
}

impl Runner {
    fn input(&mut self) -> bevy_ecs::change_detection::Mut<'_, RawInput> {
        self.app.world_mut().resource_mut::<RawInput>()
    }

    fn frame(&mut self) {
        let now = Instant::now();
        let delta = self
            .last_frame
            .map_or(0.0, |last| now.duration_since(last).as_secs_f64());
        if self.last_frame.is_some() {
            self.frames.frame(delta, now);
        }
        self.last_frame = Some(now);
        self.app.update(delta);
        self.apply_cursor_grab();
        self.apply_window_mode();
    }

    /// Fills the screen with the window, or stops, as the app's [`WindowMode`] says.
    fn apply_window_mode(&mut self) {
        let Some(window) = &self.window else { return };
        let want = self
            .app
            .world()
            .get_resource::<WindowMode>()
            .is_some_and(|mode| mode.fullscreen);
        if want != self.fullscreen {
            window.set_fullscreen(want.then_some(Fullscreen::Borderless(None)));
            self.fullscreen = want;
        }
    }

    /// Gives the app the window and starts it.
    fn attach(&mut self) {
        let Some(window) = &self.window else { return };
        let started = Instant::now();
        let size = window.inner_size();
        let world = self.app.world_mut();
        world.insert_resource(WindowHandle(window.clone()));
        world.insert_resource(WindowSize {
            width: size.width,
            height: size.height,
            scale_factor: window.scale_factor(),
        });
        self.apply_window_mode();
        self.app.start();
        let took = started.elapsed().as_secs_f64();
        if took >= FrameWatch::LEAST {
            tracing::warn!("the app took {took:.2}s to start, the window waiting meanwhile");
        }
        // A new app's frames are not the old one's.
        self.frames = FrameWatch::default();
    }

    /// Replaces the app with the one its [`Handover`] prepared, if any, in the same window;
    /// returns whether it did.
    fn hand_over(&mut self) -> bool {
        let Some((next, carry)) = self.app.world_mut().resource_mut::<Handover>().take() else {
            return false;
        };
        let waited = Instant::now();
        let next = next.join();
        let waited = waited.elapsed().as_secs_f64();
        if waited >= FrameWatch::LEAST {
            tracing::warn!("waited {waited:.2}s for the next app, the window waiting meanwhile");
        }
        let mut next = match next {
            Ok(next) => next,
            Err(_) => {
                tracing::error!("the next app failed while it was prepared");
                return false;
            }
        };
        let input = self.app.world().resource::<RawInput>().clone();
        let grabbed = self.app.world().resource::<CursorGrab>().is_grabbed();
        if let Some(carry) = carry {
            carry(self.app.world_mut(), next.world_mut());
        }
        // The old app is dropped out of the way, a large world takes a while to free; its
        // renderer gives up the window's surface as it goes, and the new one waits for it.
        let old = std::mem::replace(&mut self.app, next);
        let dropped = std::thread::Builder::new()
            .name("old app".to_owned())
            .spawn(move || drop(old));
        if let Err(error) = dropped {
            tracing::warn!("the old app is dropped in place: {error}");
        }
        self.app.world_mut().insert_resource(input);
        self.last_frame = None;
        self.attach();
        self.app
            .world_mut()
            .resource_mut::<CursorGrab>()
            .set_grabbed(grabbed);
        self.apply_cursor_grab();
        self.follow_focus();
        true
    }

    /// Grabs the cursor if the app asks for it whenever the window has the focus, and hides it
    /// if the app asks for that.
    fn follow_focus(&mut self) {
        let Some(window) = &self.window else { return };
        let grab = self.app.world().resource::<CursorGrab>();
        if grab.hidden {
            window.set_cursor_visible(false);
        }
        if self.focused && grab.grab_on_focus && !grab.is_grabbed() {
            self.app.world_mut().resource_mut::<CursorGrab>().grab();
            self.apply_cursor_grab();
        }
    }

    fn apply_cursor_grab(&mut self) {
        let Some(window) = &self.window else { return };
        let world = self.app.world_mut();
        let Some(want) = world.resource_mut::<CursorGrab>().take_request() else {
            return;
        };
        // Never locked in a window that is not in front: a grab asked for meanwhile is dropped
        // (the window grabs it again on gaining the focus, if asked to).
        if want && !self.focused {
            return;
        }
        let hidden = world.resource::<CursorGrab>().hidden;
        let grabbed = set_grab(window, want, hidden) == want;
        world.resource_mut::<CursorGrab>().set_grabbed(grabbed);
    }

    fn release_cursor(&mut self) {
        if let Some(window) = &self.window {
            let hidden = self.app.world().resource::<CursorGrab>().hidden;
            set_grab(window, false, hidden);
        }
        let mut grab = self.app.world_mut().resource_mut::<CursorGrab>();
        grab.set_grabbed(false);
        grab.take_request();
    }

    fn mouse_button(&mut self, state: ElementState, button: MouseButton) {
        match state {
            ElementState::Pressed => {
                let grab = self.app.world().resource::<CursorGrab>();
                if grab.grab_on_click && !grab.is_grabbed() {
                    self.swallowed.insert(button);
                    self.app.world_mut().resource_mut::<CursorGrab>().grab();
                    self.apply_cursor_grab();
                } else {
                    self.input().press(button);
                }
            }
            ElementState::Released => {
                if !self.swallowed.remove(&button) {
                    self.input().release(button);
                }
            }
        }
    }
}

/// Grabs or releases the cursor, hidden over the window when released if `hidden`; returns
/// whether it is grabbed afterwards.
fn set_grab(window: &Window, grab: bool, hidden: bool) -> bool {
    if !grab {
        if let Err(error) = window.set_cursor_grab(CursorGrabMode::None) {
            tracing::warn!("could not release the cursor: {error}");
        }
        window.set_cursor_visible(!hidden);
        return false;
    }
    // Locked keeps the cursor still (Wayland, macOS); Confined keeps it inside (X11, Windows).
    let result = window
        .set_cursor_grab(CursorGrabMode::Locked)
        .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
    match result {
        Ok(()) => {
            window.set_cursor_visible(false);
            true
        }
        Err(error) => {
            tracing::warn!("could not grab the cursor: {error}");
            false
        }
    }
}

/// The [`Splash`] in its window, put there by the CPU.
struct SplashWindow {
    window: Arc<Window>,
    // Kept for as long as the image shows.
    _surface: softbuffer::Surface<Arc<Window>, Arc<Window>>,
}

impl SplashWindow {
    /// Opens `splash` in a borderless window in the middle of the primary monitor and shows
    /// it at once; `None` (and a warning) if the platform will not.
    fn open(event_loop: &ActiveEventLoop, title: &str, splash: &Splash) -> Option<Self> {
        let (Some(width), Some(height)) = (
            NonZeroU32::new(splash.width),
            NonZeroU32::new(splash.height),
        ) else {
            return None;
        };
        if splash.pixels.len() != (splash.width as usize) * (splash.height as usize) * 4 {
            tracing::warn!(
                "the splash is not {}×{} RGBA: not shown",
                splash.width,
                splash.height
            );
            return None;
        }
        let mut attributes = Window::default_attributes()
            .with_title(title)
            .with_decorations(false)
            .with_resizable(false)
            .with_inner_size(PhysicalSize::new(splash.width, splash.height));
        if let Some(monitor) = event_loop
            .primary_monitor()
            .or_else(|| event_loop.available_monitors().next())
        {
            let (at, size) = (monitor.position(), monitor.size());
            attributes = attributes.with_position(PhysicalPosition::new(
                at.x + (i64::from(size.width) - i64::from(splash.width)).max(0) as i32 / 2,
                at.y + (i64::from(size.height) - i64::from(splash.height)).max(0) as i32 / 2,
            ));
        }
        let shown = (|| -> Result<Self, String> {
            let window = Arc::new(
                event_loop
                    .create_window(attributes)
                    .map_err(|error| error.to_string())?,
            );
            let context = softbuffer::Context::new(window.clone()).map_err(|e| e.to_string())?;
            let mut surface =
                softbuffer::Surface::new(&context, window.clone()).map_err(|e| e.to_string())?;
            surface
                .resize(width, height)
                .map_err(|error| error.to_string())?;
            let mut buffer = surface.buffer_mut().map_err(|error| error.to_string())?;
            for (pixel, rgba) in buffer.iter_mut().zip(splash.pixels.as_chunks::<4>().0) {
                *pixel = u32::from(rgba[0]) << 16 | u32::from(rgba[1]) << 8 | u32::from(rgba[2]);
            }
            buffer.present().map_err(|error| error.to_string())?;
            Ok(Self {
                window,
                _surface: surface,
            })
        })();
        shown
            .inspect_err(|error| tracing::warn!("the splash does not show: {error}"))
            .ok()
    }
}

impl Runner {
    /// Shows the window, its first frame drawn, and closes the splash.
    fn reveal(&mut self) {
        let Some(splash) = self.splash.take() else {
            return;
        };
        if let Some(window) = &self.window {
            window.set_visible(true);
            window.focus_window();
        }
        drop(splash);
    }
}

impl ApplicationHandler for Runner {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let config = self.app.window_config();
        self.splash = config
            .splash
            .as_ref()
            .and_then(|splash| SplashWindow::open(event_loop, &config.title, splash));
        // Behind a splash, hidden until its first frame is drawn: never an empty window.
        let attributes = Window::default_attributes()
            .with_title(config.title.clone())
            .with_inner_size(LogicalSize::new(config.size.0, config.size.1))
            .with_visible(self.splash.is_none());
        let window = match event_loop.create_window(attributes) {
            Ok(window) => Arc::new(window),
            Err(error) => {
                self.error = Some(RunError::Window(error));
                event_loop.exit();
                return;
            }
        };
        self.focused = window.has_focus();
        self.window = Some(window);
        self.attach();
        if self.splash.is_some() {
            // The first frame, which compiles and uploads what the start left, behind it too.
            self.frame();
            self.reveal();
        }
        self.follow_focus();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        if self
            .splash
            .as_ref()
            .is_some_and(|splash| splash.window.id() == id)
        {
            return;
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                self.frame();
                let going = self.app.world().resource::<Handover>().is_going();
                let exit = self.app.world().resource::<AppExit>().is_requested();
                if (going && !self.hand_over()) || (!going && exit) {
                    event_loop.exit();
                }
            }
            WindowEvent::Resized(size) => {
                let mut window_size = self.app.world_mut().resource_mut::<WindowSize>();
                window_size.width = size.width;
                window_size.height = size.height;
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                self.app
                    .world_mut()
                    .resource_mut::<WindowSize>()
                    .scale_factor = scale_factor;
            }
            WindowEvent::Focused(focused) => {
                tracing::debug!("window focused: {focused}");
                self.focused = focused;
                if focused {
                    self.follow_focus();
                } else {
                    self.input().release_all();
                    self.swallowed.clear();
                    self.release_cursor();
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let PhysicalKey::Code(key) = event.physical_key else {
                    tracing::debug!("ignoring unidentified key {:?}", event.physical_key);
                    return;
                };
                if event.state == ElementState::Pressed {
                    self.input().add_typed(key, event.text.as_deref());
                }
                if event.repeat {
                    return;
                }
                match event.state {
                    ElementState::Pressed => self.input().press(key),
                    ElementState::Released => self.input().release(key),
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                tracing::trace!("mouse {button:?} {state:?}");
                self.mouse_button(state, button);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let lines = match delta {
                    MouseScrollDelta::LineDelta(x, y) => Vector2::new(x as Real, y as Real),
                    MouseScrollDelta::PixelDelta(pixels) => {
                        Vector2::new(pixels.x, pixels.y) / RawInput::PIXELS_PER_LINE
                    }
                };
                self.input().add_wheel(lines);
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: DeviceId, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta: (x, y) } = event
            && self.focused
        {
            self.input().add_mouse_motion(Vector2::new(x, y));
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let Some(window) = &self.window else { return };
        let next = self
            .frame_interval
            .zip(self.last_frame)
            .map(|(interval, last)| last + interval);
        match next {
            Some(next) if Instant::now() < next => {
                event_loop.set_control_flow(ControlFlow::WaitUntil(next));
            }
            _ => {
                event_loop.set_control_flow(ControlFlow::Wait);
                window.request_redraw();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn a_drop_does_not_become_the_usual_frame() {
        let mut watch = FrameWatch::default();
        let mut now = Instant::now();
        let mut step = |watch: &mut FrameWatch, delta: Real| {
            now += Duration::from_secs_f64(delta);
            watch.frame(delta, now);
        };
        for _ in 0..100 {
            step(&mut watch, 1.0 / 60.0);
        }
        step(&mut watch, 0.5);
        assert!(watch.pending.is_none(), "the first drop is logged at once");
        assert!(watch.logged.is_some());
        step(&mut watch, 0.2);
        step(&mut watch, 0.3);
        assert_eq!(
            watch.pending.map(|(count, longest, _)| (count, longest)),
            Some((2, 0.3))
        );
        assert!((watch.usual.unwrap() - 1.0 / 60.0).abs() < 1e-9);
        for _ in 0..60 {
            step(&mut watch, 1.0 / 60.0);
        }
        assert!(
            watch.pending.is_none(),
            "counted drops are logged after the quiet"
        );
    }
}
