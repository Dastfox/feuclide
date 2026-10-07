// fk::geometry for H³, the hyperboloid. Points are (x, y, z, t) with t² − x² − y² − z² = 1, t
// last as E³'s homogeneous w; tangent vectors are Minkowski-orthogonal to their point; all in
// the eye's frame (x right, y up, looking along −z), the eye at (0, 0, 0, 1).
//
// The projection is perspective on these coordinates, which is the gnomonic projection of the
// Beltrami–Klein model, xyz / t: geodesics are straight on screen. Depth is reverse-Z on the
// Klein depth along the view axis, from tanh(near) to tanh(far): monotonic in geodesic distance
// along each pixel's ray, so `geo_depth_to_distance` and `geo_distance_to_depth` convert
// exactly. The Klein coordinates crowd towards 1 as `tanh d`, so the far distance must stay
// within a fog radius of a few units (beyond 8, f32 cannot tell distances apart).
#define_import_path fk::geometry

#import fk::view::view

// The Minkowski form, (+, +, +, −) in this order.
fn mink(a: vec4<f32>, b: vec4<f32>) -> f32 {
    return dot(a.xyz, b.xyz) - a.w * b.w;
}

fn geo_apply_iso(m: mat4x4<f32>, x: vec4<f32>) -> vec4<f32> {
    return m * x;
}

// The inverse of a Lorentz matrix: η Mᵀ η.
fn geo_inverse_iso(m: mat4x4<f32>) -> mat4x4<f32> {
    let t = transpose(m);
    let eta = vec4<f32>(1.0, 1.0, 1.0, -1.0);
    return mat4x4<f32>(t[0] * eta * eta.x, t[1] * eta * eta.y, t[2] * eta * eta.z, t[3] * eta * eta.w);
}

// Puts an interpolated embedding back on the hyperboloid.
fn geo_normalize_point(x: vec4<f32>) -> vec4<f32> {
    let p = x * inverseSqrt(max(-mink(x, x), 1e-30));
    return select(p, -p, p.w < 0.0);
}

// Makes an interpolated vector tangent at `p`.
fn geo_project_tangent(p: vec4<f32>, v: vec4<f32>) -> vec4<f32> {
    return v + p * mink(p, v);
}

fn geo_inner(p: vec4<f32>, a: vec4<f32>, b: vec4<f32>) -> f32 {
    return mink(a, b);
}

// The vector Euclidean-orthogonal to `p`, `a` and `b`: n_i = (−1)^i times the minor of the
// columns p, a, b without row i; at the eye, (a × b, 0).
fn triple(p: vec4<f32>, a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    let m0 = determinant(mat3x3<f32>(p.yzw, a.yzw, b.yzw));
    let m1 = determinant(mat3x3<f32>(p.xzw, a.xzw, b.xzw));
    let m2 = determinant(mat3x3<f32>(p.xyw, a.xyw, b.xyw));
    let m3 = determinant(mat3x3<f32>(p.xyz, a.xyz, b.xyz));
    return vec4<f32>(m0, -m1, m2, -m3);
}

// The cross product of two tangent vectors at `p`: Minkowski-orthogonal to `p`, `a` and `b`,
// turning the way x, y and z do (at the eye, cross(a, b)).
fn geo_cross(p: vec4<f32>, a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    // Made Minkowski-orthogonal by η.
    let n = triple(p, a, b);
    return vec4<f32>(n.xyz, -n.w);
}

// Distance from `q` to the bisector of `a` and `b`, positive on `a`'s side: the plane
// ⟨x, a − b⟩ = 0, at signed distance asinh(⟨q, n⟩ / |n|).
fn geo_bisector_distance(q: vec4<f32>, a: vec4<f32>, b: vec4<f32>) -> f32 {
    let n = a - b;
    return asinh(mink(q, n) * inverseSqrt(max(mink(n, n), 1e-30)));
}

// Geodesic distance between two points: asinh of the tangent part near, ln(c + s) far.
fn geo_distance(a: vec4<f32>, b: vec4<f32>) -> f32 {
    let c = -mink(a, b);
    let w = b - a * c;
    let s = sqrt(max(mink(w, w), 0.0));
    return select(asinh(s), log(c + s), c > 2.0);
}

// Unit tangent at `p` of the geodesic towards `q` (zero if they are the same point).
fn geo_towards(p: vec4<f32>, q: vec4<f32>) -> vec4<f32> {
    let w = q + p * mink(p, q);
    return w * inverseSqrt(max(mink(w, w), 1e-30));
}

// One over the area of the geodesic sphere of radius `d` over the unit sphere's in flat space:
// 1 / sinh²(d), held below 0.1. The twin of `fk_render::light::falloff`.
fn geo_light_falloff(d: f32) -> f32 {
    let s = sinh(max(d, 0.1));
    return 1.0 / (s * s);
}

// Carries a tangent vector at the eye to `p` along the geodesic joining them.
fn geo_transport_from_eye(p: vec4<f32>, v: vec4<f32>) -> vec4<f32> {
    let o = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    return v + (o + p) * (mink(p, v) / (1.0 + p.w));
}

fn geo_distance_from_eye(p: vec4<f32>) -> f32 {
    return asinh(length(p.xyz));
}

// Unit direction, in the eye's frame, of the geodesic leaving the eye towards `p`.
fn geo_direction_from_eye(p: vec4<f32>) -> vec3<f32> {
    return normalize(p.xyz);
}

// The point at geodesic distance `d` from the eye along the unit direction `dir`.
fn geo_point_from_eye(dir: vec3<f32>, d: f32) -> vec4<f32> {
    return vec4<f32>(dir * sinh(d), cosh(d));
}

// asinh(s) / s, without 0 / 0.
fn asinhc(s: f32) -> f32 {
    return select(asinh(s) / s, 1.0 - s * s / 6.0, s < 1e-3);
}

// sinh(x) / x, without 0 / 0.
fn sinhc(x: f32) -> f32 {
    return select(sinh(x) / x, 1.0 + x * x / 6.0, x < 1e-3);
}

// The logarithm at the origin: reference-frame coordinates of the tangent vector reaching `p`.
fn geo_log_origin(p: vec4<f32>) -> vec3<f32> {
    return p.xyz * asinhc(length(p.xyz));
}

// The exponential at the origin.
fn geo_exp_origin(v: vec3<f32>) -> vec4<f32> {
    let d = length(v);
    return vec4<f32>(v * sinhc(d), cosh(d));
}

// Width covered by one pixel at geodesic distance `d` from the eye, at the centre of the image:
// the angle of a pixel times the radius of the sphere of radius `d`.
fn geo_pixel_footprint(d: f32) -> f32 {
    return 2.0 * sinh(d) / (view.proj_scale.y * view.size.y);
}

// Fraction of the light from a point at geodesic distance `d` lost to fog.
fn geo_fog(d: f32, density: f32) -> f32 {
    return 1.0 - exp(-density * d);
}

// The Klein depths of the near and far surfaces.
fn klein_near() -> f32 {
    return tanh(view.near);
}

fn klein_far() -> f32 {
    return tanh(view.far);
}

fn geo_project(p: vec4<f32>) -> vec4<f32> {
    let z = -p.z;
    let n = klein_near();
    let f = klein_far();
    return vec4<f32>(
        p.x * view.proj_scale.x,
        p.y * view.proj_scale.y,
        n / (f - n) * (f * p.w - z),
        z,
    );
}

// Length of the eye-space ray through `ndc` per unit of depth along the view axis.
fn ray_stretch(ndc: vec2<f32>) -> f32 {
    return length(vec3<f32>(ndc / view.proj_scale, 1.0));
}

fn geo_depth_to_distance(depth: f32, ndc: vec2<f32>) -> f32 {
    let n = klein_near();
    let f = klein_far();
    let z = n * f / ((f - n) * depth + n);
    return atanh(min(z * ray_stretch(ndc), 0.9999999));
}

fn geo_distance_to_depth(d: f32, ndc: vec2<f32>) -> f32 {
    let n = klein_near();
    let f = klein_far();
    let z = tanh(d) / ray_stretch(ndc);
    return n * (f - z) / (z * (f - n));
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
