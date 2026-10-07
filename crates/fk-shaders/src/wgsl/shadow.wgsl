// The directional light's shadow map, bound in group 1 by the pipelines that receive shadows.
// Flat space only: the renderer turns shadows off (`lighting.shadow.x = 0`) elsewhere.
#define_import_path fk::shadow

#import fk::view::lighting

@group(1) @binding(0) var shadow_map: texture_depth_2d;
@group(1) @binding(1) var shadow_sampler: sampler_comparison;

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
