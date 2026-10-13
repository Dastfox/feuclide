// The post pass: one fullscreen triangle reading the opaque pass's colour and depth.
//
// Mode 0 is the image with its effects, composed in a fixed order: where the image is taken
// from (`grade_source`), blur, the caller's `grade_blurred`, saturation, `grade_saturated`,
// vignette, then `grade_over`. The engine's `fk::grade` leaves the image as it is; a caller
// replaces it (`Hooks::grade`) and gives its effects their meaning, reading its own values in
// `grade.values`.
// Mode 1 shows the depth buffer decoded back to geodesic distance, as grey bands one
// `band_spacing` apart that darken towards `far`.
// In both modes the gauges and the text panel, debug readouts, are drawn on top.

#import fk::view::{view, pixel_to_ndc}
#import fk::geometry::geo_depth_to_distance
#import fk::post::{Gauge, MAX_GAUGES, scene_color, scene_depth, post, linear, text}
#import fk::grade::{grade_source, grade_blurred, grade_saturated, grade_over}

@vertex
fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn srgb_encode(linear_color: vec3<f32>) -> vec3<f32> {
    let c = max(linear_color, vec3<f32>(0.0));
    let low = c * 12.92;
    let high = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(high, low, c <= vec3<f32>(0.0031308));
}

fn srgb_decode(encoded: vec3<f32>) -> vec3<f32> {
    let c = max(encoded, vec3<f32>(0.0));
    let low = c / 12.92;
    let high = pow((c + 0.055) / 1.055, vec3<f32>(2.4));
    return select(high, low, c <= vec3<f32>(0.04045));
}

// Interleaved gradient noise in [0, 1), a different value on every pixel.
fn dither_noise(frag: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(frag, vec2<f32>(0.06711056, 0.00583715))));
}

const BLUR_TAPS: u32 = 16u;
const GOLDEN_ANGLE: f32 = 2.39996323;

// A disc of taps spread by the golden angle, uniform in area.
fn blurred(uv: vec2<f32>, radius: f32) -> vec3<f32> {
    var sum = vec3<f32>(0.0);
    for (var i = 0u; i < BLUR_TAPS; i++) {
        let r = sqrt((f32(i) + 0.5) / f32(BLUR_TAPS)) * radius;
        let a = f32(i) * GOLDEN_ANGLE;
        let offset = vec2<f32>(cos(a), sin(a)) * r / view.size;
        sum += textureSampleLevel(scene_color, linear, uv + offset, 0.0).rgb;
    }
    return sum / f32(BLUR_TAPS);
}

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn distance_view(pixel: vec2<i32>, frag: vec2<f32>) -> vec3<f32> {
    let depth = textureLoad(scene_depth, pixel, 0);
    let d = geo_depth_to_distance(depth, pixel_to_ndc(frag));
    let band = 0.55 + 0.45 * step(0.5, fract(d / post.band_spacing));
    return vec3<f32>(band * (1.0 - clamp(d / view.far, 0.0, 1.0)));
}

// Share of a gauge segment left empty between it and the next.
const GAUGE_GAP: f32 = 0.2;
// Opacity of the unlit segments, relative to the lit ones.
const GAUGE_UNLIT: f32 = 0.3;

// `gauge` over `color` at `at`, in image heights from the bottom-left corner.
fn gauge_over(color: vec3<f32>, gauge: Gauge, at: vec2<f32>) -> vec3<f32> {
    let local = (at - gauge.rect.xy) / max(gauge.rect.zw, vec2<f32>(1e-6));
    if any(local < vec2<f32>(0.0)) || any(local >= vec2<f32>(1.0)) {
        return color;
    }
    let segments = f32(gauge.segments);
    let cell = local.y * segments;
    let within = fract(cell);
    if within < 0.5 * GAUGE_GAP || within > 1.0 - 0.5 * GAUGE_GAP {
        return color;
    }
    let lit = floor(cell) < round(gauge.value * segments);
    return mix(color, gauge.color.rgb, gauge.color.a * select(GAUGE_UNLIT, 1.0, lit));
}

// Words of `text` holding the font, and the width of its grid.
const TEXT_FONT_WORDS: u32 = 256u;
const TEXT_COLUMNS: u32 = 96u;
// Font pixels around the text, and from one line to the next.
const TEXT_PAD: f32 = 4.0;
const TEXT_LINE: f32 = 10.0;

// The text panel at the pixel `frag`: linear colour premultiplied by how much it covers, 0
// off the panel.
fn text_panel(frag: vec2<f32>) -> vec4<f32> {
    let columns = u32(post.text_size.x);
    let rows = u32(post.text_size.y);
    if columns == 0u || rows == 0u {
        return vec4<f32>(0.0);
    }
    // In font pixels from the panel's corner, then from the text's.
    let local = (frag - post.text_layout.xy) / max(post.text_layout.z, 1.0);
    let size = vec2<f32>(f32(columns) * 8.0, f32(rows) * TEXT_LINE);
    if any(local < vec2<f32>(0.0)) || any(local >= size + 2.0 * TEXT_PAD) {
        return vec4<f32>(0.0);
    }
    let panel = vec4<f32>(post.text_background.rgb * post.text_background.a, post.text_background.a);
    let inner = local - TEXT_PAD;
    if any(inner < vec2<f32>(0.0)) || any(inner >= size) {
        return panel;
    }
    let cell = vec2<u32>(u32(inner.x / 8.0), u32(inner.y / TEXT_LINE));
    let pixel = vec2<u32>(u32(inner.x) % 8u, u32(inner.y - f32(cell.y) * TEXT_LINE));
    if pixel.y >= 8u {
        return panel;
    }
    let index = cell.y * TEXT_COLUMNS + cell.x;
    let code = (text[TEXT_FONT_WORDS + index / 4u] >> (8u * (index % 4u))) & 0x7fu;
    let row = (text[2u * code + pixel.y / 4u] >> (8u * (pixel.y % 4u))) & 0xffu;
    let lit = ((row >> pixel.x) & 1u) == 1u;
    return select(panel, vec4<f32>(post.text_color.rgb, 1.0), lit);
}

// The text panel alone, blended over the window's finished image (the overlay included) in a
// pass of its own.
@fragment
fn text_fragment(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let panel = text_panel(frag.xy);
    if panel.a <= 0.0 {
        discard;
    }
    if post.encode_srgb == 0u {
        // The target encodes for us.
        return panel;
    }
    return vec4<f32>(srgb_encode(panel.rgb / panel.a) * panel.a, panel.a);
}

@fragment
fn fragment(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(frag.xy);
    let uv = frag.xy / view.size;
    var color: vec3<f32>;
    if post.mode == 1u {
        color = distance_view(pixel, frag.xy);
    } else {
        // Taken from elsewhere, the image is sampled between pixels.
        let source = grade_source(uv);
        var image = textureLoad(scene_color, pixel, 0);
        if any(source != uv) {
            image = textureSampleLevel(scene_color, linear, source, 0.0);
        }
        if post.blur > 0.5 {
            color = blurred(source, post.blur);
        } else {
            color = image.rgb;
        }
        color = grade_blurred(color, source, frag.xy);
        color = mix(vec3<f32>(luma(color)), color, post.saturation);
        color = grade_saturated(color, image);

        // Distance from the centre, 1 at the corners.
        let aspect = view.size.x / view.size.y;
        let centred = (uv - 0.5) * vec2<f32>(aspect, 1.0);
        let r = length(centred) / length(vec2<f32>(aspect, 1.0) * 0.5);
        color *= 1.0 - post.vignette * smoothstep(0.35, 1.1, r);

        color = grade_over(color, uv, source, r, frag.xy);
    }
    let at = vec2<f32>(frag.x, view.size.y - frag.y) / view.size.y;
    for (var i = 0u; i < min(post.gauge_count, MAX_GAUGES); i++) {
        color = gauge_over(color, post.gauges[i], at);
    }
    // Dither by a fraction of one 8-bit sRGB step, so dark gradients do not band.
    var encoded = srgb_encode(color) + (dither_noise(frag.xy) - 0.5) / 255.0;
    if post.encode_srgb == 0u {
        // The target encodes for us.
        return vec4<f32>(srgb_decode(encoded), 1.0);
    }
    return vec4<f32>(encoded, 1.0);
}
