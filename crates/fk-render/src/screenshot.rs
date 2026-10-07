//! Saving what the window shows to PNG files.

use std::path::{Path, PathBuf};

use bevy_ecs::prelude::*;
use fk_app::{AppExit, Time};
use fk_math::Real;

use crate::Gpu;

/// Screenshots waiting for the next frame, as a resource.
///
/// Each request saves the next frame the window shows, after the post pass, as an 8-bit sRGB
/// PNG. Surfaces that cannot be copied from, or are not 8-bit RGBA / BGRA, are skipped with a
/// warning.
#[derive(Resource, Clone, Debug, Default)]
pub struct Screenshots {
    pending: Vec<PathBuf>,
}

impl Screenshots {
    /// Saves the next frame to `path`, creating its directory if needed.
    pub fn request(&mut self, path: impl Into<PathBuf>) {
        self.pending.push(path.into());
    }

    pub(crate) fn take(&mut self) -> Vec<PathBuf> {
        std::mem::take(&mut self.pending)
    }
}

/// An unattended capture read from the environment: `FK_CAPTURE` names the PNG to write,
/// `FK_CAPTURE_AFTER` how many seconds to run first (3 by default; while time is paused, each
/// frame counts a sixtieth of a second). The app exits once the file is written, so a script
/// can render a scene and look at it.
#[derive(Resource, Clone, Debug)]
pub(crate) struct CaptureOnce {
    path: PathBuf,
    after: Real,
}

impl CaptureOnce {
    pub(crate) fn from_env() -> Option<Self> {
        let path = PathBuf::from(std::env::var_os("FK_CAPTURE")?);
        let after = std::env::var("FK_CAPTURE_AFTER")
            .ok()
            .and_then(|text| text.parse().ok())
            .unwrap_or(3.0);
        Some(Self { path, after })
    }
}

/// What a frame counts towards a capture while time is paused, seconds.
const PAUSED_FRAME: Real = 1.0 / 60.0;

/// Requests the [`CaptureOnce`] screenshot when its time comes, and the exit with it: the
/// frame is drawn and saved before the loop stops.
pub(crate) fn capture_once(
    mut commands: Commands,
    capture: Option<Res<CaptureOnce>>,
    time: Res<Time>,
    mut screenshots: ResMut<Screenshots>,
    mut exit: ResMut<AppExit>,
    mut ran: Local<Real>,
) {
    let Some(capture) = capture else { return };
    *ran += if time.is_paused() {
        PAUSED_FRAME
    } else {
        time.delta()
    };
    if *ran < capture.after {
        return;
    }
    screenshots.request(capture.path.clone());
    exit.request();
    commands.remove_resource::<CaptureOnce>();
}

/// Copies `texture`, the frame just drawn, into each of `paths`. Blocks until the GPU is done.
pub(crate) fn save(gpu: &Gpu, texture: &wgpu::Texture, paths: &[PathBuf]) {
    let swap_red_blue = match texture.format() {
        wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => false,
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => true,
        format => {
            tracing::warn!("cannot save a screenshot of a {format:?} surface");
            return;
        }
    };
    let Some(mut pixels) = read_back(gpu, texture, 4, "screenshot") else {
        return;
    };
    for pixel in pixels.as_chunks_mut::<4>().0 {
        if swap_red_blue {
            pixel.swap(0, 2);
        }
        // The post pass writes an opaque image; keep it opaque whatever the alpha mode.
        pixel[3] = 255;
    }
    let (width, height) = (texture.width(), texture.height());
    for path in paths {
        match write_png(path, width, height, &pixels) {
            Ok(()) => tracing::info!("screenshot saved to {}", path.display()),
            Err(error) => tracing::warn!("screenshot {}: {error}", path.display()),
        }
    }
}

/// The pixels of `texture`, `bytes` per pixel, row after row without padding. Blocks until the
/// GPU is done; `None` (with a warning naming `what`) if the copy failed.
pub(crate) fn read_back(
    gpu: &Gpu,
    texture: &wgpu::Texture,
    bytes: u32,
    what: &str,
) -> Option<Vec<u8>> {
    let (width, height) = (texture.width(), texture.height());
    let row = width * bytes;
    let padded_row =
        row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fk read back"),
        size: u64::from(padded_row) * u64::from(height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fk read back"),
        });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row),
                rows_per_image: Some(height),
            },
        },
        texture.size(),
    );
    gpu.queue.submit([encoder.finish()]);

    let (sender, receiver) = std::sync::mpsc::channel();
    buffer.map_async(wgpu::MapMode::Read, .., move |result| {
        // The receiver only goes away if this function already returned.
        let _ = sender.send(result);
    });
    if let Err(error) = gpu.device.poll(wgpu::PollType::wait_indefinitely()) {
        tracing::warn!("{what}: {error}");
        return None;
    }
    match receiver.recv() {
        Ok(Ok(())) => {}
        Ok(Err(error)) => {
            tracing::warn!("{what}: {error}");
            return None;
        }
        Err(_) => return None,
    }

    let mut pixels = Vec::with_capacity((row * height) as usize);
    match buffer.get_mapped_range(..) {
        Ok(mapped) => {
            for line in mapped.chunks_exact(padded_row as usize) {
                pixels.extend_from_slice(&line[..row as usize]);
            }
        }
        Err(error) => {
            tracing::warn!("{what}: {error}");
            return None;
        }
    }
    buffer.unmap();
    Some(pixels)
}

pub(crate) fn write_png(path: &Path, width: u32, height: u32, rgba: &[u8]) -> Result<(), String> {
    if let Some(dir) = path.parent().filter(|dir| !dir.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|error| error.to_string())?;
    }
    let file = std::fs::File::create(path).map_err(|error| error.to_string())?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
    let mut writer = encoder.write_header().map_err(|error| error.to_string())?;
    writer
        .write_image_data(rgba)
        .map_err(|error| error.to_string())
}
