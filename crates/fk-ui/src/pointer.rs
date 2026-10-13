//! A pointer drawn by the app rather than the system's cursor, for menus while the cursor stays
//! grabbed and hidden ([`fk_app::CursorGrab`]): it moves with the mouse's raw motion, kept
//! inside the window, and fades in where a menu wants it; elsewhere it fades out, standing
//! where it was left.

use fk_math::Real;
use fk_math::nalgebra::Vector2;

/// Seconds the pointer takes to fade in or out.
const FADE: Real = 0.2;

/// The pointer, in pixels from the window's top left corner.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pointer {
    /// Where it points.
    pub at: (f32, f32),
    /// How much it shows, 0 to 1.
    pub shown: f32,
    /// Whether it moved this frame.
    pub moved: bool,
    /// Whether it has been put in the window yet.
    placed: bool,
}

impl Pointer {
    /// Moves it by the mouse's `motion` (pixels) inside a window `size` big, if `wanted`, and
    /// fades it in or out over `dt` seconds as `wanted` says. It starts a little below the
    /// window's middle.
    pub fn step(
        &mut self,
        motion: Vector2<Real>,
        size: Option<(f32, f32)>,
        dt: Real,
        wanted: bool,
    ) {
        let Some((w, h)) = size else { return };
        if !self.placed {
            self.at = (0.5 * w, 0.55 * h);
            self.placed = true;
        }
        self.moved = wanted && motion != Vector2::zeros();
        if self.moved {
            self.at = (
                (self.at.0 + motion.x as f32).clamp(0.0, w - 1.0),
                (self.at.1 + motion.y as f32).clamp(0.0, h - 1.0),
            );
        }
        let fade = (dt / FADE) as f32;
        self.shown = if wanted {
            (self.shown + fade).min(1.0)
        } else {
            (self.shown - fade).max(0.0)
        };
    }

    /// Whether it shows at all.
    pub fn shows(&self) -> bool {
        self.shown > 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_stays_in_the_window_and_fades() {
        let mut pointer = Pointer::default();
        let size = Some((100.0, 50.0));
        pointer.step(Vector2::new(1000.0, -1000.0), size, 0.1, true);
        assert_eq!(pointer.at, (99.0, 0.0));
        assert!(pointer.moved && pointer.shows());
        pointer.step(Vector2::new(-10.0, 0.0), size, 1.0, false);
        assert_eq!(pointer.at, (99.0, 0.0), "unwanted, it stands still");
        assert!(!pointer.shows());
    }
}
