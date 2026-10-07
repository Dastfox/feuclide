// The post pass: one fullscreen triangle reading the opaque pass's colour and depth.
//
// Mode 0 is the image with its effects, composed in a fixed order: the lenses (warps), blur and
// streaks,
// saturation, posterize, vignette, tunnel, the sky's bodies asked to be over everything, lids
// (their inside the layer render as much as asked), then the mark over everything. The ink is
// laid on before, in its own pass ahead of the mosh. The engine gives them no meaning; the game decides what they are.
// Mode 1 shows the depth buffer decoded back to geodesic distance, as grey bands one
// `band_spacing` apart that darken towards `far`.
// In both modes the gauges and the text panel, debug readouts, are drawn on top.

#import fk::view::{view, lighting, pixel_to_ndc, Disc}
#import fk::geometry::geo_depth_to_distance
#import fk::sky::{star_lines, disc_cover, disc_glow}

// A segmented bar filling from the bottom.
struct Gauge {
    // Bottom-left corner and size, in image heights from the bottom-left corner.
    rect: vec4<f32>,
    // Linear RGB, and the opacity.
    color: vec4<f32>,
    // In [0, 1], shown rounded to whole segments.
    value: f32,
    segments: u32,
}

const MAX_GAUGES: u32 = 4u;
const MAX_WARPS: u32 = 4u;

struct Post {
    mode: u32,
    band_spacing: f32,
    // Whether to apply the sRGB transfer function here, for targets that are not sRGB.
    encode_srgb: u32,
    // Blur radius in pixels.
    blur: f32,
    lid_color: vec4<f32>,
    // 0 open, 1 shut.
    lid_closure: f32,
    lid_softness: f32,
    // 0 none, 1 the whole image.
    tunnel: f32,
    tunnel_softness: f32,
    tunnel_color: vec4<f32>,
    saturation: f32,
    // 0 unchanged, 1 every colour turned to its opposite hue at its own brightness.
    swap: f32,
    vignette: f32,
    // Steps of brightness in sRGB, 0 for none.
    posterize: f32,
    // How much of the afterimage shows through the lids.
    afterimage: f32,
    // How much the lid edges arc, 0 straight.
    lid_curve: f32,
    // How much the lids darken towards the image edges and along their rims.
    lid_occlusion: f32,
    gauge_count: u32,
    gauges: array<Gauge, MAX_GAUGES>,
    // The mark at the centre. x: 0 a disc, 1 a ring, 2 a star; y: the star's points; z: the
    // lines' width; w: the radius, both in image heights.
    mark_shape: vec4<f32>,
    // Linear RGB, and the opacity.
    mark_color: vec4<f32>,
    // The glow's colour, and the distance it falls to 1/e over, in image heights.
    mark_glow: vec4<f32>,
    // How much of the whole image the mark's colour washes over.
    mark_flash: f32,
    // How much the inside of the lids shows the layer render instead of their colour.
    lid_layer: f32,
    warp_count: u32,
    // Each lens: its centre, in image heights from the image's centre (y up), its radius in
    // image heights, and its strength.
    warps: array<vec4<f32>, MAX_WARPS>,
    // Each lens's twist at its strongest, in radians.
    warp_twists: vec4<f32>,
    // The mark's centre, xy in image heights from the image's centre, y up; z its turn, in
    // radians; w its ring's radius as a share of its own, 0 for no ring.
    mark_centre: vec4<f32>,
    // The text panel's top-left corner, in pixels from the image's top-left, and its scale.
    text_layout: vec4<f32>,
    // Its columns and lines, 0 when there is no text.
    text_size: vec4<f32>,
    text_color: vec4<f32>,
    // The panel's colour, and its opacity.
    text_background: vec4<f32>,
    // The mark's ring's colour (rgb).
    mark_ring_color: vec4<f32>,
    // Its glow's colour, and the distance it falls to 1/e over.
    mark_ring_glow: vec4<f32>,
    // The streaks: their centre, in image heights from the image's centre (y up), and the share
    // of the way in towards it each pixel averages over (z), 0 for none.
    streak: vec4<f32>,
    // 1 when the mark is drawn as the sky draws its discs: its glow falling off from its centre
    // and under its lines.
    mark_as_sky: f32,
}

@group(1) @binding(0) var scene_color: texture_2d<f32>;
// With multisampling the depth target keeps its samples; sample 0 is read.
#ifdef MULTISAMPLED
@group(1) @binding(1) var scene_depth: texture_depth_multisampled_2d;
#else
@group(1) @binding(1) var scene_depth: texture_depth_2d;
#endif
@group(1) @binding(2) var<uniform> post: Post;
@group(1) @binding(3) var linear: sampler;
@group(1) @binding(4) var afterimage: texture_2d<f32>;
// Only the entities on the layer, over a flat colour.
@group(1) @binding(5) var layer: texture_2d<f32>;
// The font (two words a glyph, a row a byte, the top row lowest, the lowest bit leftmost),
// then the text's characters on a TEXT_COLUMNS-wide grid, four to a word.
@group(1) @binding(6) var<storage, read> text: array<u32>;

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

const STREAK_TAPS: u32 = 12u;

// The image smeared along the line from `uv` in towards the streaks' centre, over its share of
// the way, the taps jittered per pixel so they show as grain rather than as copies.
fn streaked(uv: vec2<f32>, pixel: vec2<f32>) -> vec3<f32> {
    let aspect = view.size.x / view.size.y;
    let centre = vec2<f32>(0.5 + post.streak.x / aspect, 0.5 - post.streak.y);
    let jitter = fract(52.9829189 * fract(dot(pixel, vec2<f32>(0.06711056, 0.00583715))));
    var sum = vec3<f32>(0.0);
    for (var i = 0u; i < STREAK_TAPS; i++) {
        let t = (f32(i) + jitter) / f32(STREAK_TAPS) * post.streak.z;
        sum += textureSampleLevel(scene_color, linear, mix(uv, centre, t), 0.0).rgb;
    }
    return sum / f32(STREAK_TAPS);
}

// Where the lenses take the image at `uv` from: each pulls it in towards its centre and turns
// it about it, most strongly halfway out and not at all from its radius on.
fn warped(uv: vec2<f32>) -> vec2<f32> {
    let aspect = view.size.x / view.size.y;
    var at = vec2<f32>((uv.x - 0.5) * aspect, 0.5 - uv.y);
    for (var i = 0u; i < min(post.warp_count, MAX_WARPS); i++) {
        let warp = post.warps[i];
        let o = at - warp.xy;
        let r = length(o);
        if warp.z <= 0.0 || r >= warp.z {
            continue;
        }
        let k = (1.0 - r / warp.z) * (1.0 - r / warp.z);
        let a = post.warp_twists[i] * k;
        let turned = vec2<f32>(cos(a) * o.x - sin(a) * o.y, sin(a) * o.x + cos(a) * o.y);
        at = warp.xy + turned * max(1.0 + warp.w * k, 0.0);
    }
    return vec2<f32>(at.x / aspect + 0.5, 0.5 - at.y);
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

// Half-height of the opening between the lids at the column of `uv`, origin top left: an
// almond, widest in the middle, so the lids reach the corners first and meet in the middle
// last. Fully open it clears the corners; fully shut it is below zero everywhere.
fn lid_aperture(uv: vec2<f32>) -> f32 {
    let soft = max(post.lid_softness, 1e-3);
    let side = 2.0 * uv.x - 1.0;
    let shape = 1.0 - post.lid_curve * side * side;
    let open = (0.5 + soft) / max(1.0 - post.lid_curve, 1e-3);
    return mix(open, -soft, post.lid_closure) * shape;
}

// The lids over `color`: how much they cover the pixel, and what they look like there. Inside
// them, the layer render shows as much as asked, sampled at `source` (through the lenses, as
// the scene is); as much, the lids themselves are not seen: the whole image gives way to the
// layer evenly as they close, without their edges or their darkening.
fn lids_over(
    color: vec3<f32>,
    uv: vec2<f32>,
    source: vec2<f32>,
    r: f32,
    behind: vec3<f32>,
) -> vec3<f32> {
    if post.lid_closure <= 1e-4 {
        return color;
    }
    let soft = max(post.lid_softness, 1e-3);
    let aperture = lid_aperture(uv);
    let height = abs(uv.y - 0.5);
    let shown = clamp(post.lid_layer, 0.0, 1.0);
    let cover = mix(smoothstep(aperture - soft, aperture, height), post.lid_closure, shown);
    // Ambient occlusion: darker towards the corners, and along the rims where the lids meet.
    let occlusion = post.lid_occlusion * (1.0 - shown);
    let corners = 1.0 - occlusion * smoothstep(0.2, 1.1, r);
    let rims = 1.0 - occlusion * exp(-max(height - aperture, 0.0) / soft);
    let dark = post.lid_color.rgb + behind * post.afterimage;
    var inside = dark;
    if shown > 0.0 {
        inside = mix(dark, textureSampleLevel(layer, linear, source, 0.0).rgb, shown);
    }
    return mix(color, inside * corners * rims, cover);
}

// A sky disc over the scene, as much as it asks to be, seen along `dir` (eye frame).
fn disc_over(color: vec3<f32>, disc: Disc, dir: vec3<f32>) -> vec3<f32> {
    let over = disc.shape.w;
    if over <= 0.0 {
        return color;
    }
    let glowed = color + disc_glow(disc, dir) * over;
    return mix(glowed, disc.color.rgb, disc_cover(disc, dir) * over);
}

// The mark over `color`, at `at` in image heights from the centre, y up.
fn mark_over(color: vec3<f32>, at: vec2<f32>) -> vec3<f32> {
    let opacity = post.mark_color.a;
    let radius = post.mark_shape.w;
    if (opacity <= 0.0 || radius <= 0.0) && post.mark_flash <= 0.0 {
        return color;
    }
    let px = 1.0 / view.size.y;
    let stroke = max(post.mark_shape.z, px);
    // Lines thinner than a pixel are drawn a pixel wide and that much fainter, so that they
    // weigh what their width says: the same share of the figure at every size.
    let thin = min(post.mark_shape.z / px, 1.0);
    let lines = select(thin, 1.0, u32(post.mark_shape.x) == 0u);
    // Turned: the figure's own frame.
    let c = cos(post.mark_centre.z);
    let s = sin(post.mark_centre.z);
    let turned = vec2<f32>(c * at.x + s * at.y, -s * at.x + c * at.y);
    var d: f32;
    switch u32(post.mark_shape.x) {
        case 1u: {
            d = abs(length(at) - radius) - 0.5 * stroke;
        }
        case 2u: {
            d = star_lines(turned, radius, post.mark_shape.y) - 0.5 * stroke;
        }
        default: {
            d = length(at) - radius;
        }
    }
    let shown = opacity * step(1e-6, radius);
    let cover = (1.0 - smoothstep(-0.5 * px, 0.5 * px, d)) * shown * lines;
    let as_sky = post.mark_as_sky > 0.5;
    // As in the sky, from the centre and under the lines; otherwise from the lines, over them.
    let off = select(max(d, 0.0), length(at), as_sky);
    let glow = post.mark_glow.rgb * exp(-off / max(post.mark_glow.w, 1e-4)) * shown;
    var marked = select(mix(color, post.mark_color.rgb, cover) + glow,
        mix(color + glow, post.mark_color.rgb, cover), as_sky);
    // The ring round it, in its own colours.
    let ring = post.mark_centre.w;
    if ring > 0.0 {
        let r = abs(length(at) - radius * ring) - 0.5 * stroke;
        let ring_cover = (1.0 - smoothstep(-0.5 * px, 0.5 * px, r)) * shown * thin;
        let ring_glow = post.mark_ring_glow.rgb
            * exp(-max(r, 0.0) / max(post.mark_ring_glow.w, 1e-4)) * shown;
        marked = mix(marked, post.mark_ring_color.rgb, ring_cover) + ring_glow;
    }
    return mix(marked, post.mark_color.rgb, clamp(post.mark_flash, 0.0, 1.0));
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
        // Through the lenses, the image is sampled between pixels.
        var source = uv;
        var image = textureLoad(scene_color, pixel, 0);
        if post.warp_count > 0u {
            source = warped(uv);
            image = textureSampleLevel(scene_color, linear, source, 0.0);
        }
        if post.blur > 0.5 {
            color = blurred(source, post.blur);
        } else {
            color = image.rgb;
        }
        if post.streak.z > 1e-3 {
            color = mix(color, streaked(source, frag.xy), 0.85);
        }
        color = mix(vec3<f32>(luma(color)), color, post.saturation);
        if post.swap > 0.0 {
            // Reflected through its grey: the opposite hue, the same brightness.
            let grey = vec3<f32>(luma(color));
            color = mix(color, max(2.0 * grey - color, vec3<f32>(0.0)), post.swap);
        }
        // The image's alpha is 1 on surfaces and 0 on the sky, which stays smooth. It is not
        // read from depth: a moshed image carries its own alpha, not the live depth's.
        if post.posterize >= 2.0 && image.a > 0.5 {
            // Flat steps of paint: the brightness is rounded, counted in sRGB so the steps look
            // even, and the hue kept (rounding each channel apart tints grey gradients).
            let steps = post.posterize - 1.0;
            let encoded = srgb_encode(color);
            let bright = luma(encoded);
            let stepped = round(bright * steps) / steps;
            color = srgb_decode(encoded * (stepped / max(bright, 1e-4)));
        }

        // Distance from the centre, 1 at the corners.
        let aspect = view.size.x / view.size.y;
        let centred = (uv - 0.5) * vec2<f32>(aspect, 1.0);
        let r = length(centred) / length(vec2<f32>(aspect, 1.0) * 0.5);
        color *= 1.0 - post.vignette * smoothstep(0.35, 1.1, r);

        let soft = max(post.tunnel_softness, 1e-3);
        let edge = (1.0 - post.tunnel) * (1.0 + soft);
        let tunnel = smoothstep(edge - soft, edge, r) * step(1e-4, post.tunnel);
        color = mix(color, post.tunnel_color.rgb, tunnel);

        // The sky's bodies that ask to be over everything, under the lids.
        let dir = normalize(vec3<f32>(pixel_to_ndc(frag.xy) / view.proj_scale, -1.0));
        color = disc_over(color, lighting.moon_disc, dir);
        color = disc_over(color, lighting.sun_disc, dir);

        let behind = textureSampleLevel(afterimage, linear, uv, 0.0).rgb;
        color = lids_over(color, uv, source, r, behind);
        let centre = vec2<f32>(frag.x - 0.5 * view.size.x, 0.5 * view.size.y - frag.y) / view.size.y;
        color = mark_over(color, centre - post.mark_centre.xy);
    }
    let at = vec2<f32>(frag.x, view.size.y - frag.y) / view.size.y;
    for (var i = 0u; i < min(post.gauge_count, MAX_GAUGES); i++) {
        color = gauge_over(color, post.gauges[i], at);
    }
    // Dither by a fraction of one 8-bit sRGB step, so dark gradients (the lids, the tunnel)
    // do not band.
    var encoded = srgb_encode(color) + (dither_noise(frag.xy) - 0.5) / 255.0;
    if post.encode_srgb == 0u {
        // The target encodes for us.
        return vec4<f32>(srgb_decode(encoded), 1.0);
    }
    return vec4<f32>(encoded, 1.0);
}
