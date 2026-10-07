// The painted light shared by every way of drawing surfaces.
#define_import_path fk::paint

#import fk::view::lighting
#import fk::geometry::{geo_distance, geo_towards, geo_inner, geo_light_falloff}

// Light in a few soft-edged bands, so surfaces read as painted rather than shaded.
fn banded(light: f32) -> f32 {
    let x = clamp(light, 0.0, 1.0) * lighting.bands;
    let band = floor(x);
    return (band + smoothstep(0.42, 0.58, x - band)) / lighting.bands;
}

// The light of the point lights on a surface at `p` with normal `n` (a tangent vector there),
// painted in bands like the sun's: each light's colour times the banded share of it that
// reaches, after the geometry's falloff and the fade at its range.
fn point_lights(p: vec4<f32>, n: vec4<f32>) -> vec3<f32> {
    var sum = vec3<f32>(0.0);
    let count = u32(lighting.point_count.x);
    let nn = inverseSqrt(max(geo_inner(p, n, n), 1e-12));
    for (var i = 0u; i < count; i++) {
        let light = lighting.point_lights[i];
        let d = geo_distance(p, light.position);
        let range = light.shape.y;
        if d >= range {
            continue;
        }
        let facing = max(geo_inner(p, n, geo_towards(p, light.position)) * nn, 0.0);
        let x4 = pow(d / range, 4.0);
        let window = (1.0 - x4) * (1.0 - x4);
        let reach = light.shape.x * geo_light_falloff(d) * window * facing;
        sum += light.color.rgb * banded(reach);
    }
    return sum;
}
