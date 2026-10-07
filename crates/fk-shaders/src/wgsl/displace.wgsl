// Helpers for deform modules (`fk::deform`), which shape a mesh's vertices on the GPU before
// their instance's isometry places them: displacement by distance from the eye, stretching,
// a minimum width on screen, a pull towards a point, a lean towards one.
//
// Nothing here reads the world block. A deform module reads the values its game put there and
// passes them in, so every place can move its own way with the same few rules.
//
// Positions are embedded points in the mesh's own frame, as the vertex buffer holds them. The
// helpers work in the coordinates of the tangent space at the mesh's origin (`geo_log_origin`),
// the coordinates meshes are authored in, with frame vector 1 up.
#define_import_path fk::displace

#import fk::geometry::{
    geo_apply_iso, geo_inverse_iso, geo_distance_from_eye, geo_log_origin, geo_exp_origin,
    geo_pixel_footprint,
}

// A vertex as a deform module returns it, in its mesh's frame.
struct Shaped {
    position: vec4<f32>,
    // A tangent vector at `position`.
    normal: vec4<f32>,
    // Share of the final colour replaced by the sky's haze, on top of the fog: 0 keeps it.
    haze: f32,
}

// The vertex unchanged.
fn unshaped(position: vec4<f32>, normal: vec4<f32>) -> Shaped {
    return Shaped(position, normal, 0.0);
}

// Geodesic distance from the eye to the origin of the instance placed by `iso` (the point its
// mesh stands on).
fn instance_distance(iso: mat4x4<f32>) -> f32 {
    return geo_distance_from_eye(geo_apply_iso(iso, geo_exp_origin(vec3<f32>(0.0))));
}

// Share of a displacement that reaches something at distance `d` from the eye: none up to
// `start`, all of it from `end` on, smooth in between.
fn displace_reach(d: f32, start: f32, end: f32) -> f32 {
    return smoothstep(start, end, d);
}

// Displacement by distance: `rest` near the eye, `far` from `end` on, smooth in between. Driven
// by a uniform that moves `far`, the near world stays still while the distance moves.
fn by_distance(rest: f32, far: f32, d: f32, start: f32, end: f32) -> f32 {
    return mix(rest, far, displace_reach(d, start, end));
}

// Scales a vertex along its mesh's up axis by `factor`, from the ground plane through the
// mesh's origin. Exact in flat space; along the normal coordinates in curved space.
fn stretch_up(position: vec4<f32>, factor: f32) -> vec4<f32> {
    let v = geo_log_origin(position);
    return geo_exp_origin(vec3<f32>(v.x, v.y * factor, v.z));
}

// Scales a vertex away from its mesh's vertical axis by `factor`.
fn widen(position: vec4<f32>, factor: f32) -> vec4<f32> {
    let v = geo_log_origin(position);
    return geo_exp_origin(vec3<f32>(v.x * factor, v.y, v.z * factor));
}

// How many times wider than its true `width` something at distance `d` must be drawn to cover
// at least `pixels` pixels across; 1 when it already does.
fn min_width_factor(width: f32, d: f32, pixels: f32) -> f32 {
    return max(1.0, pixels * geo_pixel_footprint(d) / max(width, 1e-6));
}

// The haze that keeps the contrast of something drawn `factor` times wider than it is: covering
// `factor` times its share of a pixel, it should show `1 / factor` of its colour.
fn widened_haze(factor: f32) -> f32 {
    return 1.0 - 1.0 / max(factor, 1.0);
}

// Moves a vertex towards `point`, an embedded point in the eye's frame (camera-relative, like
// `iso`): by `strength` of the way there for a vertex on it, less the further it is, nothing
// from `radius` on. `strength` is kept below 1 so nothing lands on the point. Exact in flat
// space; along the normal coordinates at the mesh's origin in curved space.
fn pull_towards(
    position: vec4<f32>,
    iso: mat4x4<f32>,
    point: vec4<f32>,
    radius: f32,
    strength: f32,
) -> vec4<f32> {
    if radius <= 0.0 || strength <= 0.0 {
        return position;
    }
    let v = geo_log_origin(position);
    let t = geo_log_origin(geo_apply_iso(geo_inverse_iso(iso), point));
    let pull = min(strength, 0.95) * (1.0 - smoothstep(0.0, radius, length(t - v)));
    return geo_exp_origin(mix(v, t, pull));
}

// Leans a standing mesh towards `point`, as `pull_towards` pulls it, but its foot stays where
// it stands: the pull grows from nothing on its ground plane to all of it at `height` up its
// axis (and above). For towers that lean rather than lift off.
fn lean_towards(
    position: vec4<f32>,
    iso: mat4x4<f32>,
    point: vec4<f32>,
    radius: f32,
    strength: f32,
    height: f32,
) -> vec4<f32> {
    let v = geo_log_origin(position);
    let pulled = geo_log_origin(pull_towards(position, iso, point, radius, strength));
    let up = clamp(v.y / max(height, 1e-6), 0.0, 1.0);
    return geo_exp_origin(mix(v, pulled, up));
}
