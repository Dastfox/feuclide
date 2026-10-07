/// A colour, stored as linear RGB with alpha.
///
/// Write colours as they look, in sRGB with [`Color::srgb`] or [`Color::hex`]; shading happens
/// in linear light.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Color {
    /// Linear red.
    pub r: f32,
    /// Linear green.
    pub g: f32,
    /// Linear blue.
    pub b: f32,
    /// Opacity.
    pub a: f32,
}

impl Color {
    /// Opaque white.
    pub const WHITE: Self = Self::linear(1.0, 1.0, 1.0);

    /// Opaque black.
    pub const BLACK: Self = Self::linear(0.0, 0.0, 0.0);

    /// An opaque colour from linear components.
    pub const fn linear(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b, a: 1.0 }
    }

    /// An opaque colour from sRGB components in `[0, 1]`.
    pub fn srgb(r: f32, g: f32, b: f32) -> Self {
        Self::linear(decode(r), decode(g), decode(b))
    }

    /// An opaque colour from sRGB bytes written as `0xRRGGBB`.
    pub fn hex(rgb: u32) -> Self {
        let channel = |shift: u32| ((rgb >> shift) & 0xff) as f32 / 255.0;
        Self::srgb(channel(16), channel(8), channel(0))
    }

    /// Linear `[r, g, b, a]`.
    pub fn to_array(self) -> [f32; 4] {
        [self.r, self.g, self.b, self.a]
    }

    /// Multiplies the RGB components by `factor`.
    pub fn scaled(self, factor: f32) -> Self {
        Self {
            r: self.r * factor,
            g: self.g * factor,
            b: self.b * factor,
            a: self.a,
        }
    }

    /// The linear interpolation from `self` (at 0) to `other` (at 1).
    pub fn mix(self, other: Self, t: f32) -> Self {
        let lerp = |a: f32, b: f32| a + (b - a) * t;
        Self {
            r: lerp(self.r, other.r),
            g: lerp(self.g, other.g),
            b: lerp(self.b, other.b),
            a: lerp(self.a, other.a),
        }
    }
}

pub(crate) fn decode(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

impl From<Color> for wgpu::Color {
    fn from(c: Color) -> Self {
        Self {
            r: c.r.into(),
            g: c.g.into(),
            b: c.b.into(),
            a: c.a.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_decodes_to_linear() {
        assert_eq!(Color::hex(0xffffff), Color::WHITE);
        assert_eq!(Color::hex(0x000000), Color::BLACK);
        let mid = Color::hex(0x808080);
        assert!((mid.r - 0.2158).abs() < 1e-3, "{mid:?}");
    }
}
