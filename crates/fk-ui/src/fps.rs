//! The frame rate, live and small in the top-right corner: [`show`], a system for
//! [`PostUpdate`](fk_app::PostUpdate). Measured on the wall clock, so a paused clock or a
//! clamped frame still counts as it ran; averaged over half a second so it can be read. It is
//! written into the [`TextPanel`], drawn over everything, menus drawn on the CPU included.

use std::time::Instant;

use bevy_ecs::prelude::*;
use fk_app::WindowSize;
use fk_math::nalgebra::Vector2;
use fk_render::{Color, PostSettings, TextPanel};

/// Seconds of frames averaged into one reading.
const PERIOD: f64 = 0.5;

/// Frames counted since the last reading, and the reading.
#[derive(Default)]
pub struct Meter {
    since: Option<Instant>,
    frames: u32,
    reading: String,
}

/// Writes the frame rate into the text panel, unless something else (a console) has written
/// it: the panel is taken only empty or as this left it.
pub fn show(window: Res<WindowSize>, mut post: ResMut<PostSettings>, mut meter: Local<Meter>) {
    let free = post.text.lines.is_empty() || post.text.lines == [meter.reading.as_str()];
    let now = Instant::now();
    let since = *meter.since.get_or_insert(now);
    meter.frames += 1;
    let seconds = now.duration_since(since).as_secs_f64();
    if seconds >= PERIOD {
        meter.reading = format!("{:>4.0} fps", f64::from(meter.frames) / seconds);
        meter.since = Some(now);
        meter.frames = 0;
    }
    if !free || meter.reading.is_empty() {
        return;
    }
    // One font pixel a screen pixel up to 1080 lines, then whole steps up; a character's
    // margin from the corner. The panel is the text and 4 font pixels round it.
    let scale = (window.height / 1080).max(1);
    let wide = (8 * meter.reading.len() as u32 + 8) * scale;
    let margin = 8 * scale;
    post.text = TextPanel {
        lines: vec![meter.reading.clone()],
        origin: Vector2::new(
            f64::from(window.width.saturating_sub(wide + margin)),
            f64::from(margin),
        ),
        scale,
        color: Color::srgb(0.75, 0.75, 0.75),
        background: Color::BLACK,
        opacity: 0.3,
    };
}
