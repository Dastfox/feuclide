// The sky pass: one fullscreen triangle drawn first in the opaque pass, behind everything.

#import fk::view::{view, pixel_to_ndc}
#import fk::sky::sky_color

@vertex
fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fragment(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let ndc = pixel_to_ndc(frag.xy);
    // Every geometry's projection maps directions at the eye the same way.
    let dir = normalize(vec3<f32>(ndc / view.proj_scale, -1.0));
    // Alpha 0: no surface here (see the post pass's posterize).
    return vec4<f32>(sky_color(dir), 0.0);
}
