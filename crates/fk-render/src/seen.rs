//! How much of the sky's sun and moon the scene leaves in view, read back from the image.

use std::sync::{Arc, OnceLock};

use bevy_ecs::prelude::Resource;
use fk_math::Real;

use crate::Gpu;
use crate::frame::DiscBlock;

/// How much of the [`Sky`](crate::Sky)'s sun and of its moon the scene leaves in view, as a
/// resource the renderer writes: the share of the pixels inside each one's circle where no
/// surface is drawn, 0 hidden behind surfaces (or off the image, or not drawn), 1 in full view.
/// The horizon is not counted, nor [`SkyDisc::over`](crate::SkyDisc::over).
///
/// It is read back from the image a frame or two late, without waiting for the GPU. A large
/// disc is measured over the square at its centre, [`DiscProbe::SIDE`] pixels across.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct DiscsSeen {
    /// The sun.
    pub sun: Real,
    /// The moon.
    pub moon: Real,
}

/// Where a disc is in the read-back image: the copied square's offset in the buffer and size,
/// and the disc's centre and radius in pixels from the square's corner.
#[derive(Clone, Copy, Debug)]
struct Region {
    offset: u64,
    width: u32,
    height: u32,
    centre: [f32; 2],
    radius: f32,
}

/// What the copy in flight holds.
struct Pending {
    regions: [Option<Region>; 2],
    /// Set by the GPU once the buffer is mapped, or failed to be.
    mapped: Option<Arc<OnceLock<Result<(), wgpu::BufferAsyncError>>>>,
}

/// Copies the pixels under the sun and the moon out of the image and reads them back.
pub(crate) struct DiscProbe {
    buffer: wgpu::Buffer,
    pending: Option<Pending>,
    seen: DiscsSeen,
}

/// Bytes per pixel of the image ([`wgpu::TextureFormat::Rgba16Float`]).
const PIXEL: u32 = 8;

impl DiscProbe {
    /// Pixels across the square copied around each disc.
    pub(crate) const SIDE: u32 = 64;
    /// Bytes of one disc's square, rows padded.
    const REGION: u64 = (Self::SIDE * Self::ROW) as u64;
    /// Bytes per copied row, a multiple of [`wgpu::COPY_BYTES_PER_ROW_ALIGNMENT`].
    const ROW: u32 = (Self::SIDE * PIXEL).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);

    pub(crate) fn new(gpu: &Gpu) -> Self {
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fk discs seen"),
            size: 2 * Self::REGION,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self {
            buffer,
            pending: None,
            seen: DiscsSeen::default(),
        }
    }

    /// The last reading: takes the copy in flight if the GPU is done with it.
    pub(crate) fn collect(&mut self, gpu: &Gpu) -> DiscsSeen {
        let Some(pending) = &self.pending else {
            return self.seen;
        };
        let Some(mapped) = &pending.mapped else {
            return self.seen;
        };
        let _ = gpu.device.poll(wgpu::PollType::Poll);
        match mapped.get() {
            None => return self.seen,
            Some(Ok(())) => {
                if let Ok(bytes) = self.buffer.get_mapped_range(..) {
                    let [sun, moon] = pending
                        .regions
                        .map(|region| region.map_or(0.0, |region| share_of_sky(&bytes, &region)));
                    self.seen = DiscsSeen { sun, moon };
                }
                self.buffer.unmap();
            }
            Some(Err(error)) => tracing::warn!("discs seen: {error}"),
        }
        self.pending = None;
        self.seen
    }

    /// Records copying the pixels under `sun` and `moon` (eye-frame discs, as the sky shader
    /// gets them) out of `color`, an image `proj_scale` projects onto, unless a copy is still in
    /// flight.
    pub(crate) fn copy(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        color: &wgpu::Texture,
        proj_scale: [f32; 2],
        [sun, moon]: [&DiscBlock; 2],
    ) {
        if self.pending.is_some() {
            return;
        }
        let size = [color.width(), color.height()];
        let mut regions = [None; 2];
        for (k, disc) in [sun, moon].into_iter().enumerate() {
            let offset = k as u64 * Self::REGION;
            let Some(region) = region(disc, proj_scale, size, offset) else {
                continue;
            };
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: color,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: region.0[0],
                        y: region.0[1],
                        z: 0,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &self.buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset,
                        bytes_per_row: Some(Self::ROW),
                        rows_per_image: Some(region.1.height),
                    },
                },
                wgpu::Extent3d {
                    width: region.1.width,
                    height: region.1.height,
                    depth_or_array_layers: 1,
                },
            );
            regions[k] = Some(region.1);
        }
        self.pending = Some(Pending {
            regions,
            mapped: None,
        });
    }

    /// Once the copy is submitted: asks for it back. Nothing copied, the reading is zero.
    pub(crate) fn map(&mut self) {
        let Some(pending) = &mut self.pending else {
            return;
        };
        if pending.mapped.is_some() {
            return;
        }
        if pending.regions.iter().all(Option::is_none) {
            self.seen = DiscsSeen::default();
            self.pending = None;
            return;
        }
        let mapped = Arc::new(OnceLock::new());
        let done = Arc::clone(&mapped);
        self.buffer
            .map_async(wgpu::MapMode::Read, .., move |result| {
                let _ = done.set(result);
            });
        pending.mapped = Some(mapped);
    }
}

/// The square of the image around `disc`, clipped to the image: its corner in pixels, and the
/// region it fills in the buffer at `offset`. `None` when the disc is not drawn, behind the
/// eye or off the image.
fn region(
    disc: &DiscBlock,
    [scale_x, scale_y]: [f32; 2],
    [width, height]: [u32; 2],
    offset: u64,
) -> Option<([u32; 2], Region)> {
    let [x, y, z, radius] = disc.direction.0;
    if radius <= 0.0 || z >= -1e-4 {
        return None;
    }
    // As the sky shader draws it: the disc's radius is the sine of its angle, about the
    // direction, which is close to its tangent this near the centre.
    let ndc = [x / -z * scale_x, y / -z * scale_y];
    let centre = [
        (ndc[0] * 0.5 + 0.5) * width as f32,
        (0.5 - ndc[1] * 0.5) * height as f32,
    ];
    let pixels = (radius.min(0.99).asin().tan() * scale_y * 0.5 * height as f32).max(1.0);
    let half = pixels.min(DiscProbe::SIDE as f32 / 2.0);
    let lo = |c: f32, size: u32| (c - half).floor().clamp(0.0, size as f32) as u32;
    let hi = |c: f32, size: u32| (c + half).ceil().clamp(0.0, size as f32) as u32;
    let (x0, x1) = (lo(centre[0], width), hi(centre[0], width));
    let (y0, y1) = (lo(centre[1], height), hi(centre[1], height));
    let (w, h) = (x1.saturating_sub(x0), y1.saturating_sub(y0));
    if w == 0 || h == 0 {
        return None;
    }
    let region = Region {
        offset,
        width: w.min(DiscProbe::SIDE),
        height: h.min(DiscProbe::SIDE),
        centre: [centre[0] - x0 as f32, centre[1] - y0 as f32],
        radius: pixels,
    };
    Some(([x0, y0], region))
}

/// The share of `region`'s pixels inside the disc's circle where the image's alpha says sky
/// (below a half), out of the whole circle: what lies off the image is not in view.
fn share_of_sky(bytes: &[u8], region: &Region) -> Real {
    let mut inside = 0u32;
    let mut sky = 0u32;
    for row in 0..region.height {
        for column in 0..region.width {
            let dx = column as f32 + 0.5 - region.centre[0];
            let dy = row as f32 + 0.5 - region.centre[1];
            if dx * dx + dy * dy > region.radius * region.radius {
                continue;
            }
            inside += 1;
            let at = region.offset as usize + (row * DiscProbe::ROW + column * PIXEL) as usize + 6;
            let alpha = u16::from_le_bytes([bytes[at], bytes[at + 1]]);
            // A positive half orders as its bits: 0x3800 is a half.
            if alpha & 0x7fff < 0x3800 {
                sky += 1;
            }
        }
    }
    // The whole circle, clipped or not, at most the square measured.
    let side = (2.0 * region.radius).min(DiscProbe::SIDE as f32);
    let whole = (std::f32::consts::PI * region.radius * region.radius)
        .min(side * side)
        .max(inside as f32)
        .max(1.0);
    Real::from(sky as f32 / whole)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vec4f;

    fn disc(direction: [f32; 3], radius: f32) -> DiscBlock {
        let [x, y, z] = direction;
        let n = (x * x + y * y + z * z).sqrt();
        DiscBlock {
            direction: Vec4f([x / n, y / n, z / n, radius]),
            shape: Vec4f([0.0; 4]),
            color: Vec4f([0.0; 4]),
            glow: Vec4f([0.0; 4]),
        }
    }

    /// An image of `region`'s square, sky where `sky` says so.
    fn image(region: &Region, sky: impl Fn(u32, u32) -> bool) -> Vec<u8> {
        let mut bytes = vec![0; (region.offset + DiscProbe::REGION) as usize];
        for row in 0..region.height {
            for column in 0..region.width {
                let at =
                    region.offset as usize + (row * DiscProbe::ROW + column * PIXEL) as usize + 6;
                let alpha: u16 = if sky(column, row) { 0 } else { 0x3c00 };
                bytes[at..at + 2].copy_from_slice(&alpha.to_le_bytes());
            }
        }
        bytes
    }

    #[test]
    fn a_disc_in_the_open_is_seen_whole_and_behind_a_wall_not_at_all() {
        let (corner, region) =
            region(&disc([0.0, 0.0, -1.0], 0.02), [1.0, 1.0], [800, 600], 0).unwrap();
        assert!(region.radius > 5.0);
        assert!(corner[0] < 400 && corner[1] < 300);
        let open = share_of_sky(&image(&region, |_, _| true), &region);
        assert!((open - 1.0).abs() < 0.1, "{open}");
        let hidden = share_of_sky(&image(&region, |_, _| false), &region);
        assert_eq!(hidden, 0.0);
        let half = share_of_sky(
            &image(&region, |column, _| (column as f32) < region.centre[0]),
            &region,
        );
        assert!((half - 0.5).abs() < 0.1, "{half}");
    }

    #[test]
    fn a_disc_off_the_image_or_behind_is_not_seen() {
        assert!(region(&disc([0.0, 0.0, 1.0], 0.02), [1.0, 1.0], [800, 600], 0).is_none());
        assert!(region(&disc([5.0, 0.0, -1.0], 0.02), [1.0, 1.0], [800, 600], 0).is_none());
        assert!(region(&disc([0.0, 0.0, -1.0], 0.0), [1.0, 1.0], [800, 600], 0).is_none());
        // Half off the edge: half of it in view at most.
        let edge = (0.5_f32).atan();
        let (_, region) = region(
            &disc([edge.sin(), 0.0, -edge.cos()], 0.02),
            [2.0, 2.0],
            [800, 600],
            0,
        )
        .unwrap();
        let share = share_of_sky(&image(&region, |_, _| true), &region);
        assert!(share < 0.6, "{share}");
    }
}
