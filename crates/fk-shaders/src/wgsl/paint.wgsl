// The painted light shared by every way of drawing surfaces.
#define_import_path fk::paint

#import fk::view::lighting
#import fk::geometry::{geo_distance, geo_towards, geo_inner, geo_light_falloff}

// Light in a few soft-edged bands, so surfaces read as painted rather than shaded; melted
// `lighting.smoothness` of the way into the light as it is.
fn banded(light: f32) -> f32 {
    let x = clamp(light, 0.0, 1.0) * lighting.bands;
    let band = floor(x);
    let painted = (band + smoothstep(0.42, 0.58, x - band)) / lighting.bands;
    return mix(painted, clamp(light, 0.0, 1.0), lighting.smoothness);
}

// The light of the point lights on a surface at `p` with normal `n` (a tangent vector there),
// painted in bands like the sun's: each light's colour times the banded share of it that
// reaches, after the geometry's falloff, the fade at its range and, for a spot, its cone.
fn point_lights(p: vec4<f32>, n: vec4<f32>) -> vec3<f32> {
    return point_lights_in(p, n, 0xffffffffu);
}

// The same, of the lights whose bit is set in `mask` only (bit `i` light `i`), in the same
// order: those an instance's ball lies wholly out of the range of give nothing there anyway.
fn point_lights_in(p: vec4<f32>, n: vec4<f32>, mask: u32) -> vec3<f32> {
    var sum = vec3<f32>(0.0);
    let nn = inverseSqrt(max(geo_inner(p, n, n), 1e-12));
    var bits = lights_masked(mask);
    while bits != 0u {
        let i = firstTrailingBit(bits);
        bits &= bits - 1u;
        let reach = point_light_reach(i, p, n, nn);
        if reach > 0.0 {
            sum += lighting.point_lights[i].color.rgb * banded(reach);
        }
    }
    return sum;
}

// The lights of `mask` there are this frame.
fn lights_masked(mask: u32) -> u32 {
    let count = u32(lighting.point_count.x);
    return mask & select((1u << count) - 1u, 0xffffffffu, count >= 32u);
}

// How much of point light `i` reaches a surface at `p` with normal `n` (`nn` one over its
// length), before its colour and the bands: its intensity after the geometry's falloff, the fade
// at its range, how the surface faces it and, for a spot, its cone. 0 out of its range.
fn point_light_reach(i: u32, p: vec4<f32>, n: vec4<f32>, nn: f32) -> f32 {
    let light = lighting.point_lights[i];
    let d = geo_distance(p, light.position);
    let range = light.shape.y;
    if d >= range {
        return 0.0;
    }
    let facing = max(geo_inner(p, n, geo_towards(p, light.position)) * nn, 0.0);
    let x4 = pow(d / range, 4.0);
    let window = (1.0 - x4) * (1.0 - x4);
    var cone = 1.0;
    if any(light.direction != vec4<f32>(0.0)) {
        let along = geo_inner(light.position, light.direction, geo_towards(light.position, p));
        cone = smoothstep(light.shape.w, light.shape.z, along);
    }
    return light.shape.x * geo_light_falloff(d) * window * facing * cone;
}
