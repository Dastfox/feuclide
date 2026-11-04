// The directional light's shadow map and the spots' (`fk_render::SpotShadow`), bound in group 1
// by the pipelines that receive shadows. Flat space only: the renderer turns shadows off
// (`lighting.shadow.x = 0`, `lighting.spot_shadow.x = 0`) elsewhere.
#define_import_path fk::shadow

#import fk::view::lighting
#import fk::paint::{banded, point_lights_in, lights_masked, point_light_reach}
#import fk::geometry::geo_inner

@group(1) @binding(0) var shadow_map: texture_depth_2d;
@group(1) @binding(1) var shadow_sampler: sampler_comparison;
// The spots' maps, a layer each.
@group(1) @binding(2) var spot_maps: texture_depth_2d_array;

// The light's clip coordinates of a point in the eye's frame.
fn shadow_coords(p: vec4<f32>) -> vec4<f32> {
    return lighting.shadow_matrix * vec4<f32>(p.xyz / p.w, 1.0);
}

// How much of the directional light reaches the surface at `p` (a point in the eye's frame)
// with unit normal `n`: 1 lit, 0 in shadow, with a soft edge `lighting.shadow.z` texels wide.
// Outside the map everything is lit, and shadows fade out towards its edge.
fn shadow_light(p: vec4<f32>, n: vec3<f32>) -> f32 {
    if lighting.shadow.x < 0.5 {
        return 1.0;
    }
    let q = shadow_coords(vec4<f32>(p.xyz / p.w + n * lighting.shadow.w, 1.0));
    let edge = max(abs(q.x), abs(q.y));
    if edge >= 1.0 || q.z <= 0.0 || q.z >= 1.0 {
        return 1.0;
    }
    let uv = vec2<f32>(q.x * 0.5 + 0.5, 0.5 - q.y * 0.5);
    // Nine bilinear comparisons, so sixteen texels, spread over the soft edge.
    let spread = lighting.shadow.y * lighting.shadow.z * 0.5;
    var lit = 0.0;
    for (var i = -1; i <= 1; i++) {
        for (var j = -1; j <= 1; j++) {
            let at = uv + vec2<f32>(f32(i), f32(j)) * spread;
            lit += textureSampleCompareLevel(shadow_map, shadow_sampler, at, q.z);
        }
    }
    return mix(lit / 9.0, 1.0, smoothstep(lighting.shadow_fade.x, 1.0, edge));
}

// How much of point light `i`'s light reaches the surface at `p` (a point in the eye's frame)
// with unit normal `n` past what stands between them, by its spot's map: 1 lit, 0 in shadow,
// with a soft edge `lighting.spot_shadow.z` texels wide. 1 for a light without a map, and out of
// its map.
fn spot_light(i: u32, p: vec3<f32>, n: vec3<f32>) -> f32 {
    let light = lighting.point_lights[i];
    if light.shadow.x < 0.5 {
        return 1.0;
    }
    let d = distance(p, light.position.xyz / light.position.w);
    let q = light.shadow_matrix * vec4<f32>(p + n * light.shadow.y * d, 1.0);
    if q.w <= 0.0 {
        return 1.0;
    }
    let c = q.xyz / q.w;
    if max(abs(c.x), abs(c.y)) >= 1.0 || c.z <= 0.0 || c.z >= 1.0 {
        return 1.0;
    }
    let uv = vec2<f32>(c.x * 0.5 + 0.5, 0.5 - c.y * 0.5);
    let layer = i32(light.shadow.x - 0.5);
    let spread = lighting.spot_shadow.y * lighting.spot_shadow.z * 0.5;
    var lit = 0.0;
    for (var a = -1; a <= 1; a++) {
        for (var b = -1; b <= 1; b++) {
            let at = uv + vec2<f32>(f32(a), f32(b)) * spread;
            lit += textureSampleCompareLevel(spot_maps, shadow_sampler, at, layer, c.z);
        }
    }
    return lit / 9.0;
}

// `fk::paint`'s `point_lights_in`, each spot's light kept off by its map.
fn shadowed_point_lights_in(p: vec4<f32>, n: vec4<f32>, mask: u32) -> vec3<f32> {
    if lighting.spot_shadow.x < 0.5 {
        return point_lights_in(p, n, mask);
    }
    var sum = vec3<f32>(0.0);
    let nn = inverseSqrt(max(geo_inner(p, n, n), 1e-12));
    let unit = n.xyz * nn;
    let at = p.xyz / p.w;
    var bits = lights_masked(mask);
    while bits != 0u {
        let i = firstTrailingBit(bits);
        bits &= bits - 1u;
        let reach = point_light_reach(i, p, n, nn);
        if reach > 0.0 {
            sum += lighting.point_lights[i].color.rgb * banded(reach) * spot_light(i, at, unit);
        }
    }
    return sum;
}
