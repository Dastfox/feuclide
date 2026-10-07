// The ink pass: one fullscreen triangle laying the inked look over the opaque image, before the
// mosh, so that a moshed image carries its lines and hatching along with the rest of its
// colours, and every later effect (lenses, blur, saturation) bends and drains them too.
//
// Lines where the distance or the brightness jumps against the neighbours `ink.y` pixels away,
// fading into the fog; the shadows hatched; laid on paper. The alpha (surface, sky, kept out of
// the mosh) is passed on as it is.

#import fk::view::{view, pixel_to_ndc}
#import fk::geometry::geo_depth_to_distance

struct Ink {
    // How much (0 off), its width in pixels, the depth and brightness jumps that draw a line.
    ink: vec4<f32>,
    // The ink's colour, and the hatching's darkness.
    color: vec4<f32>,
    // The distance lines fade by, the hatching's spacing in pixels, the brightness the shadows
    // are hatched below, the paper's amount.
    more: vec4<f32>,
    // The paper's tone.
    paper: vec4<f32>,
}

@group(1) @binding(0) var scene_color: texture_2d<f32>;
// With multisampling the depth target keeps its samples; sample 0 is read.
#ifdef MULTISAMPLED
@group(1) @binding(1) var scene_depth: texture_depth_multisampled_2d;
#else
@group(1) @binding(1) var scene_depth: texture_depth_2d;
#endif
@group(1) @binding(2) var<uniform> ink: Ink;

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

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// The distance along the pixel `at`'s ray, clamped into the image; the far surface on the sky.
fn ink_distance(at: vec2<i32>) -> f32 {
    let size = vec2<i32>(view.size);
    let p = clamp(at, vec2<i32>(0), size - 1);
    let depth = textureLoad(scene_depth, p, 0);
    if depth <= 0.0 {
        return view.far;
    }
    return geo_depth_to_distance(depth, pixel_to_ndc(vec2<f32>(p) + 0.5));
}

fn ink_bright(at: vec2<i32>) -> f32 {
    let size = vec2<i32>(view.size);
    return luma(srgb_encode(textureLoad(scene_color, clamp(at, vec2<i32>(0), size - 1), 0).rgb));
}

@fragment
fn fragment(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(frag.xy);
    let image = textureLoad(scene_color, pixel, 0);
    let color = image.rgb;
    let w = i32(max(round(ink.ink.y), 1.0));
    let here = ink_distance(pixel);
    var jump = 0.0;
    var edge = 0.0;
    let bright = luma(srgb_encode(color));
    let steps = array<vec2<i32>, 4>(vec2<i32>(w, 0), vec2<i32>(-w, 0), vec2<i32>(0, w), vec2<i32>(0, -w));
    for (var i = 0; i < 4; i++) {
        let there = ink_distance(pixel + steps[i]);
        // Only the nearer side draws the line, so a silhouette is one line, not two.
        jump = max(jump, (there - here) / max(here, 1e-3));
        edge = max(edge, abs(ink_bright(pixel + steps[i]) - bright));
    }
    let line = max(
        smoothstep(ink.ink.z, ink.ink.z * 2.0, jump),
        smoothstep(ink.ink.w, ink.ink.w * 2.0, edge),
    );
    let near = 1.0 - smoothstep(0.5 * ink.more.x, ink.more.x, here);
    // Hatching: lines one way in the shadows, crossed the other way where they are darker.
    let spacing = ink.more.y;
    let dark = smoothstep(ink.more.z, ink.more.z * 0.5, bright);
    let darker = smoothstep(ink.more.z * 0.6, ink.more.z * 0.25, bright);
    let one = step(fract((frag.x + frag.y) / spacing), 0.22);
    let other = step(fract((frag.x - frag.y) / spacing), 0.22);
    let hatch = max(one * dark, other * darker) * ink.color.a * near;
    var out = mix(color, color * ink.paper.rgb, ink.more.w);
    out = mix(out, ink.color.rgb, max(line * near, hatch));
    return vec4<f32>(mix(color, out, ink.ink.x), image.a);
}
