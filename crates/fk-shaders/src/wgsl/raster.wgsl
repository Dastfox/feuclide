// The raster pipeline: instanced meshes in flat painted colours, and windows onto the scene
// seen from another eye (`window_fragment`).
//
// Every instance carries its camera-relative isometry as a matrix, so the vertex stage is the
// same `mat4 · v` in every geometry, followed by the geometry's projection. Before that, the
// vertex goes through `fk::deform`: the identity unless the game supplied its own module for
// these instances.

#import fk::view::{view, world, lighting}
#import fk::geometry::{
    geo_apply_iso, geo_normalize_point, geo_project_tangent, geo_inner, geo_cross,
    geo_transport_from_eye,
    geo_fog, geo_project_side, geo_seen_distance, geo_seen_direction,
}
#import fk::sky::sky_haze
#import fk::paint::{banded, point_lights}
#import fk::displace::Shaped
#import fk::deform::deform
#import fk::shadow::{shadow_coords, shadow_light}

// The window pipeline's own group: the scene seen from another eye (`fk_render::Elsewhere`),
// its sampler, and (x) 1 when that image is shown upside down. The painted pipeline does not
// have it.
@group(2) @binding(0) var elsewhere_image: texture_2d<f32>;
@group(2) @binding(1) var elsewhere_sampler: sampler;
@group(2) @binding(2) var<uniform> elsewhere: vec4<f32>;

// The painted pipeline's own group: the normal maps (`fk_render::NormalMaps`), layer 0 the flat
// one every mesh without a map samples. Bindings apart from the window's, which shares group 2.
@group(2) @binding(3) var normal_maps: texture_2d_array<f32>;
@group(2) @binding(4) var normal_sampler: sampler;

struct Vertex {
    @location(0) position: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) iso_0: vec4<f32>,
    @location(3) iso_1: vec4<f32>,
    @location(4) iso_2: vec4<f32>,
    @location(5) iso_3: vec4<f32>,
    @location(6) color: vec4<f32>,
    // Free for the game: the painted material ignores it, deform modules read it.
    @location(7) data: vec4<f32>,
    // Texture coordinates; z the normal map's layer (0 none), w the tangent frame's handedness.
    @location(8) uv: vec4<f32>,
    // The tangent along u, zero without a normal map.
    @location(9) tangent: vec4<f32>,
}

struct Varyings {
    @builtin(position) clip: vec4<f32>,
    @location(0) point: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) color: vec4<f32>,
    @location(3) haze: f32,
    @location(4) uv: vec4<f32>,
    @location(5) tangent: vec4<f32>,
    // 1 where the instance is seen the long way round a closed space (on a sphere), else 0.
    @location(6) side: f32,
}

fn instance_iso(in: Vertex) -> mat4x4<f32> {
    return mat4x4<f32>(in.iso_0, in.iso_1, in.iso_2, in.iso_3);
}

@vertex
fn vertex(in: Vertex) -> Varyings {
    let iso = instance_iso(in);
    let shaped = deform(in.position, in.normal, iso, in.data);
    let point = geo_apply_iso(iso, shaped.position);
    // The pipeline marks the copy of an instance seen the long way round by its colour's alpha.
    let far_side = in.color.w > 1.5;
    var out: Varyings;
    out.clip = geo_project_side(point, far_side);
    out.side = select(0.0, 1.0, far_side);
    out.point = point;
    out.normal = geo_apply_iso(iso, shaped.normal);
    out.uv = in.uv;
    out.tangent = geo_apply_iso(iso, in.tangent);
    out.color = in.color;
    out.haze = shaped.haze;
    return out;
}

// The shadow map pass: the same shapes, seen from the directional light. No fragment stage.
@vertex
fn shadow_vertex(in: Vertex) -> @builtin(position) vec4<f32> {
    let iso = instance_iso(in);
    let shaped = deform(in.position, in.normal, iso, in.data);
    let q = shadow_coords(geo_apply_iso(iso, shaped.position));
    // Casters nearer the light than the map's near side still cast, from its near side.
    return vec4<f32>(q.xy, max(q.z, 0.0), 1.0);
}

// The normal at `p` bent by the normal map: the map's normal in the frame of the unit normal,
// the tangent along u (made square to it) and their cross product, turned by the handedness.
fn mapped_normal(p: vec4<f32>, normal: vec4<f32>, uv: vec4<f32>, tangent: vec4<f32>) -> vec4<f32> {
    let m = textureSample(normal_maps, normal_sampler, uv.xy, i32(uv.z)).xyz * 2.0 - 1.0;
    let n = normal * inverseSqrt(max(geo_inner(p, normal, normal), 1e-12));
    let along = geo_project_tangent(p, tangent);
    let t = along - n * geo_inner(p, n, along);
    let length2 = geo_inner(p, t, t);
    if length2 < 1e-12 {
        return n;
    }
    let u = t * inverseSqrt(length2);
    let v = geo_cross(p, n, u) * uv.w;
    return u * m.x + v * m.y + n * m.z;
}

@fragment
fn fragment(in: Varyings) -> @location(0) vec4<f32> {
    let p = geo_normalize_point(in.point);
    let n = mapped_normal(p, geo_project_tangent(p, in.normal), in.uv, in.tangent);
    let sun = geo_transport_from_eye(p, lighting.sun_direction);
    let facing = geo_inner(p, n, sun) * inverseSqrt(max(geo_inner(p, n, n) * geo_inner(p, sun, sun), 1e-12));
    // Cast shadows take the painted shadow colour, like the sides turned away from the light.
    let shadow = shadow_light(p, n.xyz * inverseSqrt(max(dot(n.xyz, n.xyz), 1e-12)));
    let light = mix(lighting.shadow_tint.rgb, lighting.sun_color.rgb, banded(facing) * shadow);
    let lit = in.color.rgb * (light + point_lights(p, n));

    let far_side = in.side > 0.5;
    let d = geo_seen_distance(p, far_side);
    // Fade fully into the sky before the far surface, so culling there is invisible. The fog
    // takes the colour of the sky behind the surface, glows included; a deformed vertex can
    // ask for more of it.
    let fog = max(geo_fog(d, lighting.fog_density), smoothstep(0.85 * view.far, view.far, d));
    let haze = 1.0 - (1.0 - fog) * (1.0 - clamp(in.haze, 0.0, 1.0));
    return vec4<f32>(mix(lit, sky_haze(geo_seen_direction(p, far_side)), haze), 1.0);
}

// A window onto the scene seen from another eye: what that eye sees at this pixel, times the
// instance's colour. Unlit and unfogged, since that image has its own light and fog; its
// alpha is the image's, 0 where it shows the sky.
@fragment
fn window_fragment(in: Varyings) -> @location(0) vec4<f32> {
    var uv = in.clip.xy / view.size;
    if elsewhere.x > 0.5 {
        uv.y = 1.0 - uv.y;
    }
    let seen = textureSampleLevel(elsewhere_image, elsewhere_sampler, uv, 0.0);
    return vec4<f32>(seen.rgb * in.color.rgb, seen.a);
}
