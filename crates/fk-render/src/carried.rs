//! A frame's image carried across a [`Handover`](fk_app::Handover): read back from one app,
//! held by the next, for an [`ImageEffect`](crate::ImageEffect) of its to start from (a
//! crossfade, a datamosh through the old image).

use bevy_ecs::resource::Resource;

/// One frame's image, read back to the CPU or drawn on it: linear HDR as the image effects and
/// the post pass read it, in the colour target's format (four half floats a pixel).
#[derive(Clone, Debug)]
pub struct HeldImage {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

impl HeldImage {
    /// Its width and height, in pixels.
    pub fn size(&self) -> [u32; 2] {
        [self.width, self.height]
    }

    /// Its pixels, rows top to bottom, in the colour target's format: what
    /// `wgpu::Queue::write_texture` takes for a texture of that format and its size.
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    /// An image drawn on the CPU, as it would be read back: `width × height` sRGB RGBA, 8 bits a
    /// channel, rows top to bottom, premultiplied (what an [`Overlay`](crate::Overlay) takes),
    /// laid over black and decoded to linear. Its alpha is 0, as the sky's.
    ///
    /// # Panics
    ///
    /// If there are not `4 · width · height` bytes.
    pub fn from_srgb(width: u32, height: u32, pixels: &[u8]) -> Self {
        assert_eq!(
            pixels.len(),
            4 * width as usize * height as usize,
            "an image of {width}×{height} takes 4 bytes a pixel"
        );
        // Premultiplied over black is the colour itself; decoded once per level.
        let linear: Vec<[u8; 2]> = (0..=255u8)
            .map(|c| f16_bits(crate::color::decode(f32::from(c) / 255.0)).to_le_bytes())
            .collect();
        let mut out = Vec::with_capacity(8 * pixels.len() / 4);
        for pixel in pixels.as_chunks::<4>().0 {
            for &c in &pixel[..3] {
                out.extend_from_slice(&linear[usize::from(c)]);
            }
            out.extend_from_slice(&[0, 0]);
        }
        Self {
            width,
            height,
            pixels: out,
        }
    }
}

/// `x` (finite, not negative) as a half float's bits, rounded to the nearest; below the
/// smallest normal half, 0.
fn f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    if x <= 0.0 || exponent <= 0 {
        return 0;
    }
    if exponent >= 0x1f {
        return 0x7bff;
    }
    let mantissa = bits & 0x7f_ffff;
    let half = ((exponent as u32) << 10) | (mantissa >> 13);
    // Rounded: the next half up when the dropped bits are over half of one.
    (half + ((mantissa >> 12) & 1)) as u16
}

/// Carries the image across a [`Handover`](fk_app::Handover), as a resource: the old app
/// [`request`](Self::request)s a frame's image and [`take`](Self::take)s it on the next frame;
/// the new app is given it with [`hold`](Self::hold), and an
/// [`ImageEffect`](crate::ImageEffect) of the new app [`take_held`](Self::take_held)s it to
/// start from. The engine gives it no meaning.
#[derive(Resource, Clone, Debug, Default)]
pub struct CarriedImage {
    requested: bool,
    read: Option<HeldImage>,
    held: Option<HeldImage>,
}

impl CarriedImage {
    /// Reads this frame's image back once it is drawn, after the image effects (blocking until
    /// the GPU is done).
    pub fn request(&mut self) {
        self.requested = true;
    }

    /// The image read back since the last request, if it was drawn.
    pub fn take(&mut self) -> Option<HeldImage> {
        self.read.take()
    }

    /// Holds `image` for an image effect to take.
    pub fn hold(&mut self, image: HeldImage) {
        self.held = Some(image);
    }

    pub(crate) fn take_request(&mut self) -> bool {
        std::mem::take(&mut self.requested)
    }

    pub(crate) fn set_read(&mut self, width: u32, height: u32, pixels: Vec<u8>) {
        self.read = Some(HeldImage {
            width,
            height,
            pixels,
        });
    }

    /// The image held, if one was and nothing took it yet.
    pub fn take_held(&mut self) -> Option<HeldImage> {
        self.held.take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halves_are_encoded() {
        assert_eq!(f16_bits(0.0), 0);
        assert_eq!(f16_bits(1.0), 0x3c00);
        assert_eq!(f16_bits(0.5), 0x3800);
        assert_eq!(f16_bits(0.333_333), 0x3555);
        let image = HeldImage::from_srgb(1, 1, &[255, 0, 128, 255]);
        assert_eq!(image.pixels[..2], 0x3c00u16.to_le_bytes());
        assert_eq!(image.pixels.len(), 8);
    }
}
