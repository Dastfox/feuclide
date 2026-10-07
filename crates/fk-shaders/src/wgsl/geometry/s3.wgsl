// fk::geometry for S³, the unit sphere in ℝ⁴. Points are unit 4-vectors (x, y, z, w), tangent
// vectors orthogonal to their point; all in the eye's frame (x right, y up, looking along −z),
// the eye at (0, 0, 0, 1) and its antipode at (0, 0, 0, −1).
//
// The projection is perspective on these coordinates: every point of the great circle leaving
// the eye along a direction (up to the antipode, at π) lands on that direction's pixel. Depth
// is reverse-Z on w / (−z) = cot(d) / cos α (α the ray's angle off the view axis), which
// falls monotonically all the way to the antipode, each pass's far held short of π. The
// camera's far distance is how far round it sees, up to 2π: the long way round is fogged and
// faded by its own distance, 2π − d.
//
// Every point is seen twice: directly, at distance d along its direction, and the long way
// round, at 2π − d along the opposite one, where its antipode −p projects. The raster
// pipeline draws everything twice (`geo_project_side`): seen directly into the front half of
// the depth range, [½, 1], the long way round into the back half, [0, ½), behind all of it.
// `geo_depth_to_distance` and `geo_distance_to_depth` convert exactly, both halves.
#define_import_path fk::geometry

#import fk::view::view

const PI: f32 = 3.14159265;

fn geo_apply_iso(m: mat4x4<f32>, x: vec4<f32>) -> vec4<f32> {
    return m * x;
}

// The inverse of a rotation of ℝ⁴: its transpose.
fn geo_inverse_iso(m: mat4x4<f32>) -> mat4x4<f32> {
    return transpose(m);
}

// Puts an interpolated embedding back on the sphere.
fn geo_normalize_point(x: vec4<f32>) -> vec4<f32> {
    return x * inverseSqrt(max(dot(x, x), 1e-30));
}

// Makes an interpolated vector tangent at `p`.
fn geo_project_tangent(p: vec4<f32>, v: vec4<f32>) -> vec4<f32> {
    return v - p * dot(p, v);
}

fn geo_inner(p: vec4<f32>, a: vec4<f32>, b: vec4<f32>) -> f32 {
    return dot(a, b);
}

// The cross product of two tangent vectors at `p`: orthogonal to `p`, `a` and `b`, turning the
// way x, y and z do (at the eye, cross(a, b)). n_i = (−1)^i times the minor of the columns p,
// a, b without row i.
fn geo_cross(p: vec4<f32>, a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    let m0 = determinant(mat3x3<f32>(p.yzw, a.yzw, b.yzw));
    let m1 = determinant(mat3x3<f32>(p.xzw, a.xzw, b.xzw));
    let m2 = determinant(mat3x3<f32>(p.xyw, a.xyw, b.xyw));
    let m3 = determinant(mat3x3<f32>(p.xyz, a.xyz, b.xyz));
    return vec4<f32>(m0, -m1, m2, -m3);
}

// Distance from `q` to the bisector of `a` and `b` (the great sphere x · (a − b) = 0),
// positive on `a`'s side.
fn geo_bisector_distance(q: vec4<f32>, a: vec4<f32>, b: vec4<f32>) -> f32 {
    let n = a - b;
    return asin(clamp(dot(q, n) * inverseSqrt(max(dot(n, n), 1e-30)), -1.0, 1.0));
}

// Geodesic distance between two points.
fn geo_distance(a: vec4<f32>, b: vec4<f32>) -> f32 {
    let c = dot(a, b);
    return atan2(length(b - a * c), c);
}

// Unit tangent at `p` of the geodesic towards `q` (zero if they are the same point).
fn geo_towards(p: vec4<f32>, q: vec4<f32>) -> vec4<f32> {
    let w = q - p * dot(p, q);
    return w * inverseSqrt(max(dot(w, w), 1e-30));
}

// One over the area of the geodesic sphere of radius `d` over the unit sphere's in flat space:
// 1 / sin²(d), held below 0.1 near the light and near its antipode, where the light gathers
// again. The twin of `fk_render::light::falloff`.
fn geo_light_falloff(d: f32) -> f32 {
    let s = max(sin(max(d, 0.1)), 0.1);
    return 1.0 / (s * s);
}

// Carries a tangent vector at the eye to `p` along the geodesic joining them.
fn geo_transport_from_eye(p: vec4<f32>, v: vec4<f32>) -> vec4<f32> {
    let o = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    return v - (o + p) * (dot(p, v) / max(1.0 + p.w, 1e-6));
}

fn geo_distance_from_eye(p: vec4<f32>) -> f32 {
    return atan2(length(p.xyz), p.w);
}

// Unit direction, in the eye's frame, of the geodesic leaving the eye towards `p`.
fn geo_direction_from_eye(p: vec4<f32>) -> vec3<f32> {
    return normalize(p.xyz);
}

// The point at geodesic distance `d` from the eye along the unit direction `dir`.
fn geo_point_from_eye(dir: vec3<f32>, d: f32) -> vec4<f32> {
    return vec4<f32>(dir * sin(d), cos(d));
}

// The logarithm at the origin: reference-frame coordinates of the tangent vector reaching `p`.
fn geo_log_origin(p: vec4<f32>) -> vec3<f32> {
    let s = length(p.xyz);
    return p.xyz * select(atan2(s, p.w) / s, 1.0 / p.w, s < 1e-6);
}

// The exponential at the origin.
fn geo_exp_origin(v: vec3<f32>) -> vec4<f32> {
    let d = length(v);
    let sinc = select(sin(d) / d, 1.0 - d * d / 6.0, d < 1e-3);
    return vec4<f32>(v * sinc, cos(d));
}

// Width covered by one pixel at geodesic distance `d` from the eye, at the centre of the image:
// the angle of a pixel times the radius of the sphere of radius `d`, which closes again at the
// antipode.
fn geo_pixel_footprint(d: f32) -> f32 {
    return 2.0 * max(abs(sin(d)), 1e-4) / (view.proj_scale.y * view.size.y);
}

// Fraction of the light from a point at geodesic distance `d` lost to fog.
fn geo_fog(d: f32, density: f32) -> f32 {
    return 1.0 - exp(-density * d);
}

// The far surface, held short of the antipode.
fn far() -> f32 {
    return min(view.far, PI - 0.01);
}

fn cot(x: f32) -> f32 {
    return cos(x) / sin(x);
}

// 1 / (cot near − cot far): depth per unit of w / (−z).
fn depth_scale() -> f32 {
    return 1.0 / (cot(view.near) - cot(far()));
}

// The projection of one pass, depth over its whole range: see `geo_project_side`.
fn geo_project(p: vec4<f32>) -> vec4<f32> {
    let z = -p.z;
    return vec4<f32>(
        p.x * view.proj_scale.x,
        p.y * view.proj_scale.y,
        depth_scale() * (p.w - cot(far()) * z),
        z,
    );
}

// Length of the eye-space ray through `ndc` per unit of depth along the view axis: 1 / cos α.
fn ray_stretch(ndc: vec2<f32>) -> f32 {
    return length(vec3<f32>(ndc / view.proj_scale, 1.0));
}

// The depth of one pass, before it is put in its half.
fn pass_depth_to_distance(depth: f32, ndc: vec2<f32>) -> f32 {
    let u = depth / depth_scale() + cot(far());
    return atan2(ray_stretch(ndc), u);
}

fn pass_distance_to_depth(d: f32, ndc: vec2<f32>) -> f32 {
    return (cot(d) * ray_stretch(ndc) - cot(far())) * depth_scale();
}

fn geo_depth_to_distance(depth: f32, ndc: vec2<f32>) -> f32 {
    if depth >= 0.5 {
        return pass_depth_to_distance(2.0 * depth - 1.0, ndc);
    }
    return PI + pass_depth_to_distance(2.0 * depth, ndc);
}

fn geo_distance_to_depth(d: f32, ndc: vec2<f32>) -> f32 {
    if d < PI {
        return 0.5 + 0.5 * pass_distance_to_depth(d, ndc);
    }
    return 0.5 * pass_distance_to_depth(d - PI, ndc);
}

// `p` projected as seen directly, or (`far_side`) the long way round: its antipode projected,
// in the back half of the depth range.
fn geo_project_side(p: vec4<f32>, far_side: bool) -> vec4<f32> {
    let c = geo_project(select(p, -p, far_side));
    return vec4<f32>(c.xy, select(0.5 * (c.z + c.w), 0.5 * c.z, far_side), c.w);
}

// How far away `p` is seen, directly or the long way round.
fn geo_seen_distance(p: vec4<f32>, far_side: bool) -> f32 {
    let d = geo_distance_from_eye(p);
    return select(d, 2.0 * PI - d, far_side);
}

// The direction `p` is seen in, directly or the long way round.
fn geo_seen_direction(p: vec4<f32>, far_side: bool) -> vec3<f32> {
    let dir = normalize(p.xyz);
    return select(dir, -dir, far_side);
}
