// fk::geometry for E³. Points are homogeneous (x, y, z, 1), tangent vectors (x, y, z, 0), in the
// eye's frame: x right, y up, looking along −z.
//
// Depth is reverse-Z perspective depth, 1 at `near` and 0 at `far`. Along one pixel's ray it is
// a strictly decreasing function of geodesic distance, so `geo_depth_to_distance` and
// `geo_distance_to_depth` convert exactly, and the ray marcher can write the same encoding.
#define_import_path fk::geometry

#import fk::view::view

fn geo_apply_iso(m: mat4x4<f32>, x: vec4<f32>) -> vec4<f32> {
    return m * x;
}

// The inverse of an isometry: a rotation and a translation, undone.
fn geo_inverse_iso(m: mat4x4<f32>) -> mat4x4<f32> {
    let r = transpose(mat3x3<f32>(m[0].xyz, m[1].xyz, m[2].xyz));
    let t = -(r * m[3].xyz);
    return mat4x4<f32>(
        vec4<f32>(r[0], 0.0),
        vec4<f32>(r[1], 0.0),
        vec4<f32>(r[2], 0.0),
        vec4<f32>(t, 1.0),
    );
}

// Puts an interpolated embedding back on the model.
fn geo_normalize_point(x: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(x.xyz / x.w, 1.0);
}

// Makes an interpolated vector tangent at `p`.
fn geo_project_tangent(p: vec4<f32>, v: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(v.xyz, 0.0);
}

fn geo_inner(p: vec4<f32>, a: vec4<f32>, b: vec4<f32>) -> f32 {
    return dot(a.xyz, b.xyz);
}

// Carries a tangent vector at the eye to `p` along the geodesic joining them.
fn geo_transport_from_eye(p: vec4<f32>, v: vec4<f32>) -> vec4<f32> {
    return v;
}

// The cross product of two tangent vectors at `p`: at right angles to both, `a`, `b` and it
// turning the way x, y and z do.
fn geo_cross(p: vec4<f32>, a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(cross(a.xyz, b.xyz), 0.0);
}

// Distance from `q` to the bisector of `a` and `b` (the points as far from both), positive
// on `a`'s side: the face between two copies of a Dirichlet domain.
fn geo_bisector_distance(q: vec4<f32>, a: vec4<f32>, b: vec4<f32>) -> f32 {
    let x = q.xyz / q.w;
    let u = a.xyz / a.w;
    let v = b.xyz / b.w;
    return (dot(x - v, x - v) - dot(x - u, x - u)) / (2.0 * max(length(v - u), 1e-12));
}

// Geodesic distance between two points.
fn geo_distance(a: vec4<f32>, b: vec4<f32>) -> f32 {
    return length(b.xyz / b.w - a.xyz / a.w);
}

// Unit tangent at `p` of the geodesic towards `q` (zero if they are the same point).
fn geo_towards(p: vec4<f32>, q: vec4<f32>) -> vec4<f32> {
    let v = q.xyz / q.w - p.xyz / p.w;
    return vec4<f32>(v * inverseSqrt(max(dot(v, v), 1e-24)), 0.0);
}

// One over the area of the geodesic sphere of radius `d`, over the unit sphere's: 1/d² in
// flat space (1/sinh² in H³, 1/sin² in S³), held below 0.1 so a surface through a light is
// lit finitely. The twin of `fk_render::light::falloff`.
fn geo_light_falloff(d: f32) -> f32 {
    let r = max(d, 0.1);
    return 1.0 / (r * r);
}

fn geo_distance_from_eye(p: vec4<f32>) -> f32 {
    return length(p.xyz / p.w);
}

// Unit direction, in the eye's frame, of the geodesic leaving the eye towards `p`.
fn geo_direction_from_eye(p: vec4<f32>) -> vec3<f32> {
    return normalize(p.xyz / p.w);
}

// The point at geodesic distance `d` from the eye along the unit direction `dir`, in the eye's
// frame: the exponential map at the eye.
fn geo_point_from_eye(dir: vec3<f32>, d: f32) -> vec4<f32> {
    return vec4<f32>(dir * d, 1.0);
}

// The logarithm at the origin: coordinates, in the reference frame, of the tangent vector at
// the origin whose geodesic reaches `p` in unit time. Meshes are authored in these coordinates
// (in the frame of their own origin) and embedded with `geo_exp_origin`.
fn geo_log_origin(p: vec4<f32>) -> vec3<f32> {
    return p.xyz / p.w;
}

// The exponential at the origin: the point reached in unit time by the geodesic leaving the
// origin with velocity `v`, given in reference-frame coordinates.
fn geo_exp_origin(v: vec3<f32>) -> vec4<f32> {
    return vec4<f32>(v, 1.0);
}

// Width covered by one pixel at geodesic distance `d` from the eye, at the centre of the image.
fn geo_pixel_footprint(d: f32) -> f32 {
    return 2.0 * d / (view.proj_scale.y * view.size.y);
}

// Fraction of the light from a point at geodesic distance `d` lost to fog.
fn geo_fog(d: f32, density: f32) -> f32 {
    return 1.0 - exp(-density * d);
}

fn geo_project(p: vec4<f32>) -> vec4<f32> {
    let z = -p.z;
    let depth_scale = view.near / (view.far - view.near);
    return vec4<f32>(
        p.x * view.proj_scale.x,
        p.y * view.proj_scale.y,
        depth_scale * (view.far * p.w - z),
        z,
    );
}

// Length of the eye-space ray through `ndc` per unit of depth along the view axis.
fn ray_stretch(ndc: vec2<f32>) -> f32 {
    return length(vec3<f32>(ndc / view.proj_scale, 1.0));
}

fn geo_depth_to_distance(depth: f32, ndc: vec2<f32>) -> f32 {
    let z = view.near * view.far / ((view.far - view.near) * depth + view.near);
    return z * ray_stretch(ndc);
}

fn geo_distance_to_depth(d: f32, ndc: vec2<f32>) -> f32 {
    let z = d / ray_stretch(ndc);
    return view.near * (view.far - z) / (z * (view.far - view.near));
}

// A space that does not close up has no far side: `far_side` is never set, and these are the
// plain projection, distance and direction. (On a sphere every point is also seen the long way
// round; see the S³ module.)
fn geo_project_side(p: vec4<f32>, far_side: bool) -> vec4<f32> {
    return geo_project(p);
}

fn geo_seen_distance(p: vec4<f32>, far_side: bool) -> f32 {
    return geo_distance_from_eye(p);
}

fn geo_seen_direction(p: vec4<f32>, far_side: bool) -> vec3<f32> {
    return geo_direction_from_eye(p);
}
