// The eye and its projection, plus the caller-filled world block. Bound in group 0 by every
// pipeline.
#define_import_path fk::view

struct View {
    // 1 / tan(half field of view), horizontally and vertically.
    proj_scale: vec2<f32>,
    // Geodesic distances of the clipping surfaces.
    near: f32,
    far: f32,
    // Size of the target in pixels.
    size: vec2<f32>,
}

// Opaque to the engine: `time` is seconds since start, `values` mean whatever the game says.
struct World {
    time: f32,
    values: array<vec4<f32>, 8>,
}

// A disc in the sky (a sun, a moon) and the glow around it. Directions are in the eye's
// frame: x right, y up, -z ahead.
struct Disc {
    // Unit direction towards it; w is its angular radius in radians, 0 for nothing.
    direction: vec4<f32>,
    // x: 0 a filled disc, 1 a ring, 2 a star; y: the star's points; z: the width of
    // the lines of a ring or a star, in radians; w: how much of it shows over the surfaces in
    // front of it (drawn by the post pass), 0 none.
    shape: vec4<f32>,
    color: vec4<f32>,
    // Colour of the glow; w is the angle in radians over which it falls to 1/e.
    glow: vec4<f32>,
}

// A point light. Its light falls off as the geometry's spheres grow (`geo_light_falloff`) and
// fades out at its range.
struct PointLight {
    // Where it is: the embedded point, in the eye's frame.
    position: vec4<f32>,
    color: vec4<f32>,
    // x: intensity, the light on a surface facing it one unit away in flat space; y: range.
    shape: vec4<f32>,
}

struct Lighting {
    // Direction towards the light, a tangent vector at the eye.
    sun_direction: vec4<f32>,
    sun_color: vec4<f32>,
    // What the side facing away from the light is multiplied by.
    shadow_tint: vec4<f32>,
    // The sky's gradient: overhead, at the horizon, and below it.
    sky_zenith: vec4<f32>,
    sky_horizon: vec4<f32>,
    sky_below: vec4<f32>,
    // Up in the eye's frame; w is the gradient's exponent (small: the zenith colour comes
    // down low).
    sky_up: vec4<f32>,
    // The axes of the turning sky the stars are fixed to, in the eye's frame.
    star_frame: array<vec4<f32>, 3>,
    sun_disc: Disc,
    moon_disc: Disc,
    // Fog per unit of geodesic distance.
    fog_density: f32,
    // Number of light bands from shadow to full light.
    bands: f32,
    // Brightness of the stars, 0 for none, and how many cells of the star grid fit in one
    // radian.
    stars: f32,
    star_density: f32,
    // The shadow map of the directional light: eye coordinates to the light's clip space (x
    // and y in [-1, 1] across the map, depth in [0, 1] away from the light). Flat space only.
    shadow_matrix: mat4x4<f32>,
    // x: 1 when shadows are drawn, 0 for none; y: one texel, in uv; z: radius of the soft edge,
    // in texels; w: how far a surface is moved along its normal before the test.
    shadow: vec4<f32>,
    // x: share of the map's half-width where shadows start fading out towards its edge.
    shadow_fade: vec4<f32>,
    // The point lights nearest the eye; x of `point_count` says how many are used.
    point_lights: array<PointLight, 8>,
    point_count: vec4<f32>,
}

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> world: World;
@group(0) @binding(2) var<uniform> lighting: Lighting;

// Pixel coordinates (origin top left) to normalized device coordinates (y up).
fn pixel_to_ndc(pixel: vec2<f32>) -> vec2<f32> {
    let uv = pixel / view.size;
    return vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
}
