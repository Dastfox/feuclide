//! Drawing on the CPU for what an app lays over the image (`fk_render::Overlay`): text in a
//! font, measured and written onto a tiny-skia pixmap, a blur and a dither.

use ab_glyph::{Font, FontRef, PxScale, ScaleFont, point};
use fk_math::Real;
use tiny_skia::Pixmap;

/// Writes `text` in `font` at `size` pixels from `at` (its left end on the baseline) onto
/// `pixmap`, `rgb` at `alpha` over what is there; returns where it ends.
pub fn write(
    pixmap: &mut Pixmap,
    font: &FontRef<'_>,
    text: &str,
    size: f32,
    at: (f32, f32),
    rgb: [u8; 3],
    alpha: f32,
) -> f32 {
    let scaled = font.as_scaled(PxScale::from(size));
    let (width, height) = (pixmap.width() as i32, pixmap.height() as i32);
    let data = pixmap.data_mut();
    let mut x = at.0;
    let mut last = None;
    for ch in text.chars() {
        let id = scaled.glyph_id(ch);
        if let Some(last) = last {
            x += scaled.kern(last, id);
        }
        last = Some(id);
        let glyph = id.with_scale_and_position(size, point(x, at.1));
        x += scaled.h_advance(id);
        if alpha <= 0.0 {
            continue;
        }
        let Some(outlined) = font.outline_glyph(glyph) else {
            continue;
        };
        let bounds = outlined.px_bounds();
        outlined.draw(|gx, gy, coverage| {
            let px = bounds.min.x as i32 + gx as i32;
            let py = bounds.min.y as i32 + gy as i32;
            if px < 0 || py < 0 || px >= width || py >= height {
                return;
            }
            let a = (coverage * alpha).clamp(0.0, 1.0);
            let i = 4 * (py * width + px) as usize;
            for (c, &value) in rgb.iter().enumerate() {
                let dst = Real::from(data[i + c]);
                data[i + c] =
                    (Real::from(value) * Real::from(a) + dst * Real::from(1.0 - a)).round() as u8;
            }
            let dst = f32::from(data[i + 3]) / 255.0;
            data[i + 3] = ((a + dst * (1.0 - a)) * 255.0).round() as u8;
        });
    }
    x
}

/// How wide `text` is at `size` pixels.
pub fn measure(font: &FontRef<'_>, text: &str, size: f32) -> f32 {
    let scaled = font.as_scaled(PxScale::from(size));
    let mut width = 0.0;
    let mut last = None;
    for ch in text.chars() {
        let id = scaled.glyph_id(ch);
        if let Some(last) = last {
            width += scaled.kern(last, id);
        }
        last = Some(id);
        width += scaled.h_advance(id);
    }
    width
}

/// Side of the dither's tile, pixels.
const DITHER: usize = 4;

/// Breaks the dark gradients' bands: a 4×4 ordered dither of up to one step either way, on
/// colour that is still a little bright (where a step shows).
pub fn dither(data: &mut [u8], width: usize) {
    const BAYER: [[i8; DITHER]; DITHER] =
        [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]];
    for (row, line) in data.chunks_exact_mut(4 * width).enumerate() {
        let bayer = &BAYER[row % DITHER];
        for (column, pixel) in line.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            // -1, 0 or +1, half the time 0.
            let step = (bayer[column % DITHER] / 6) - 1;
            for value in &mut pixel[..3] {
                *value = value.saturating_add_signed(step);
            }
        }
    }
}

/// Blurs `pixmap` (premultiplied, transparent beyond its edges) by about `radius` pixels: a box
/// blur each way, twice, near enough a gaussian; running sums, so its cost does not grow with
/// the radius.
pub fn box_blur(pixmap: &mut Pixmap, radius: f32) {
    let r = (0.7 * radius).round() as usize;
    if r < 1 {
        return;
    }
    let (w, h) = (pixmap.width() as usize, pixmap.height() as usize);
    let taps = (2 * r + 1) as u32;
    let row = 4 * w;
    let data = pixmap.data_mut();
    let mut from = vec![0u8; data.len()];
    let add = |sum: &mut [u32], bytes: &[u8]| {
        for (total, &value) in sum.iter_mut().zip(bytes) {
            *total += u32::from(value);
        }
    };
    let take = |sum: &mut [u32], bytes: &[u8]| {
        for (total, &value) in sum.iter_mut().zip(bytes) {
            *total -= u32::from(value);
        }
    };
    let mut sums = vec![0u32; row];
    for _ in 0..2 {
        // Along the rows: a running sum over each row's pixels.
        from.copy_from_slice(data);
        for (out, from) in data.chunks_exact_mut(row).zip(from.chunks_exact(row)) {
            let px = |i: usize| &from[4 * i..4 * i + 4];
            let sum = &mut sums[..4];
            sum.fill(0);
            for i in 0..r.min(w) {
                add(sum, px(i));
            }
            for (i, out) in out.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                if i + r < w {
                    add(sum, px(i + r));
                }
                for (out, &total) in out.iter_mut().zip(sum.iter()) {
                    *out = (total / taps) as u8;
                }
                if i >= r {
                    take(sum, px(i - r));
                }
            }
        }
        // Down the columns: one running sum over every column, row by row.
        from.copy_from_slice(data);
        let rows = |y: usize| &from[y * row..(y + 1) * row];
        sums.fill(0);
        for y in 0..r.min(h) {
            add(&mut sums, rows(y));
        }
        for (y, out) in data.chunks_exact_mut(row).enumerate() {
            if y + r < h {
                add(&mut sums, rows(y + r));
            }
            for (out, &total) in out.iter_mut().zip(sums.iter()) {
                *out = (total / taps) as u8;
            }
            if y >= r {
                take(&mut sums, rows(y - r));
            }
        }
    }
}
