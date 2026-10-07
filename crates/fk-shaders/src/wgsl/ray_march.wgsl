// The ray-march pipeline: one fullscreen pass marching each pixel's ray through the caller's
// signed distance field (`fk::sdf`), in the scene block of `fk::march`.
//
// The ray leaves the eye along the pixel's direction. On the fast path it is the geometry's
// geodesic in closed form (`geo_point_from_eye`) and the march is over-relaxed sphere tracing;
// composed with `SAMPLED_METRIC`, it is integrated with RK4 from `fk::metric`'s geodesic
// equation, in the eye's normal coordinates, without relaxation. A ray hits where the field
// comes within a few pixels' footprint, so near and far detail are found alike and a zoom
// needs no epsilon of its own. The surface takes the painted light of the raster path,
// darkened by the steps it took, and fades into the sky with distance, each as much as the
// scene asks (`march.paint`); its depth is written in
// the encoding every render path shares, so raster and ray-marched surfaces hide each other.
//
// The image's alpha is 1 on a surface; composed with `SDF_KEPT`, it is 2 where the field's
// `sdf_kept` says the surface is kept out of the mosh patches; composed with `SDF_MOSHED`, 3
// where its `sdf_moshed` says the surface carries their mosh wherever it is.
//
// Entry points: `vertex` (fullscreen triangle); `fragment` (full resolution: colour, depth,
// misses discarded); `fragment_half` (into two offscreen targets: colour with alpha 1 on a hit,
// and the distance); `composite` (those targets brought up to full resolution, with depth);
// `fragment_coarse` (one cone per block of `march.paint.z` pixels, into the coarse target the
// other two start their rays from, `coarse_start`). Rays stop where the scene marched ahead of
// this one is (`ahead_distance`, `march.paint.w`): what is behind it is never marched.

#import fk::view::{view, lighting, pixel_to_ndc}
#import fk::geometry::{
    geo_apply_iso, geo_normalize_point, geo_project_tangent, geo_inner, geo_transport_from_eye,
    geo_point_from_eye, geo_log_origin, geo_fog, geo_pixel_footprint, geo_distance_to_depth,
    geo_distance, geo_bisector_distance,
}
#import fk::sky::sky_haze
#import fk::paint::{banded, point_lights}
#import fk::march::{march, march_cone}
#import fk::sdf::{sdf_distance, sdf_color}
#ifdef SDF_KEPT
#import fk::sdf::sdf_kept
#endif
#ifdef SDF_MOSHED
#import fk::sdf::sdf_moshed
#endif
#ifdef SAMPLED_METRIC
#import fk::metric::geo_geodesic_accel
#endif

const BIG: f32 = 3.4e38;

@vertex
fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

const IDENTITY: mat4x4<f32> = mat4x4<f32>(
    vec4<f32>(1.0, 0.0, 0.0, 0.0),
    vec4<f32>(0.0, 1.0, 0.0, 0.0),
    vec4<f32>(0.0, 0.0, 1.0, 0.0),
    vec4<f32>(0.0, 0.0, 0.0, 1.0),
);

// In a quotient (`march.fold`), the deck elements the ray being traced was carried back by as
// it crossed faces of the domain, as one isometry relative to the eye: every point it
// evaluates the field at goes through it first. The identity until it crosses a face.
var<private> deck: mat4x4<f32> = IDENTITY;

// Where `p` (embedded, in the eye's frame) is in the scene's coordinates, carried by `m` first.
fn scene_point_by(m: mat4x4<f32>, p: vec4<f32>) -> vec3<f32> {
    return geo_log_origin(geo_apply_iso(march.to_scene, geo_apply_iso(m, p))) * march.settings.x;
}

// In a quotient, what carries `p` to the copy of the scene whose surface is nearest it: the
// ray's deck, or that and the step back across one face (the copies in the neighbouring
// domains, where what straddles a face sticks out).
fn nearest_copy(p: vec4<f32>) -> mat4x4<f32> {
    var best = deck;
    var nearest = sdf_distance(scene_point_by(deck, p));
    let count = u32(march.fold.count.x);
    for (var f = 0u; f < count; f++) {
        let m = march.fold.back[f] * deck;
        let d = sdf_distance(scene_point_by(m, p));
        if d < nearest {
            nearest = d;
            best = m;
        }
    }
    return best;
}

// Where `p` (embedded, in the eye's frame) is in the scene's coordinates: in the nearest copy
// of the scene, in a quotient.
fn scene_point(p: vec4<f32>) -> vec3<f32> {
    return scene_point_by(nearest_copy(p), p);
}

// How far the ray at `p` is from the nearest face of the domain (the bisector between the
// centre and a translate of it). BIG outside a quotient.
fn fold_room(p: vec4<f32>) -> f32 {
    let count = u32(march.fold.count.x);
    if count == 0u {
        return BIG;
    }
    let q = geo_normalize_point(geo_apply_iso(deck, p));
    var room = BIG;
    for (var f = 0u; f < count; f++) {
        room = min(room, geo_bisector_distance(q, march.fold.centre, march.fold.faces[f]));
    }
    return max(room, 0.0);
}

// Carries the ray at `p` back across the faces of the domain it is beyond, the farthest first.
fn fold(p: vec4<f32>) {
    let count = u32(march.fold.count.x);
    for (var crossing = 0u; crossing < 4u && count > 0u; crossing++) {
        let q = geo_normalize_point(geo_apply_iso(deck, p));
        var nearest = -1;
        var best = geo_distance(q, march.fold.centre);
        for (var f = 0u; f < count; f++) {
            let d = geo_distance(q, march.fold.faces[f]);
            if d < best {
                best = d;
                nearest = i32(f);
            }
        }
        if nearest < 0 {
            return;
        }
        deck = march.fold.back[nearest] * deck;
    }
}

// Distance, in eye units, from `p` to the surface (of the nearest copy, in a quotient).
fn scene_distance(p: vec4<f32>) -> f32 {
    var nearest = sdf_distance(scene_point_by(deck, p));
    let count = u32(march.fold.count.x);
    for (var f = 0u; f < count; f++) {
        nearest = min(nearest, sdf_distance(scene_point_by(march.fold.back[f] * deck, p)));
    }
    return nearest / march.settings.x;
}

// A ray from the eye: how far along it is, and on the sampled path its chart state.
struct Ray {
    dir: vec3<f32>,
    t: f32,
    x: vec3<f32>,
    v: vec3<f32>,
}

fn ray_point(ray: Ray) -> vec4<f32> {
#ifdef SAMPLED_METRIC
    return vec4<f32>(ray.x, 1.0);
#else
    return geo_point_from_eye(ray.dir, ray.t);
#endif
}

// Where the rays of the block round `pixel` (window pixels) start, and the steps their cone
// took to get there; the near surface and none without cones.
@group(2) @binding(2) var coarse_start: texture_2d<f32>;

fn start_at(pixel: vec2<f32>) -> vec2<f32> {
    let n = march.paint.z;
    if n < 0.5 {
        return vec2<f32>(0.0);
    }
    let size = vec2<i32>(textureDimensions(coarse_start));
    let block = clamp(vec2<i32>(floor(pixel / n)), vec2<i32>(0), size - 1);
    return textureLoad(coarse_start, block, 0).xy;
}

// The distances of the scene marched ahead of this one, 0 where it has no surface: rays stop
// there, where they would only find what it hides.
@group(2) @binding(3) var ahead_distance: texture_2d<f32>;

// How far the ray through `pixel` (window pixels) may go before the scene ahead hides it.
fn limit_at(pixel: vec2<f32>) -> f32 {
    let k = march.paint.w;
    if abs(k) < 0.5 {
        return BIG;
    }
    let size = vec2<i32>(textureDimensions(ahead_distance));
    let texel = clamp(vec2<i32>(floor(pixel / abs(k))), vec2<i32>(0), size - 1);
    let d = textureLoad(ahead_distance, texel, 0).r;
    if d <= 0.0 {
        return BIG;
    }
    // Drawn over everything, it hides all of what is under it, nearer or not.
    if k < 0.0 {
        return 0.0;
    }
    return d;
}

// The ray moved on by `s` along itself.
fn ray_advance(ray: Ray, s: f32) -> Ray {
    var next = ray;
    next.t = ray.t + s;
#ifdef SAMPLED_METRIC
    let k1x = ray.v;
    let k1v = geo_geodesic_accel(ray.x, ray.v);
    let k2x = ray.v + 0.5 * s * k1v;
    let k2v = geo_geodesic_accel(ray.x + 0.5 * s * k1x, k2x);
    let k3x = ray.v + 0.5 * s * k2v;
    let k3v = geo_geodesic_accel(ray.x + 0.5 * s * k2x, k3x);
    let k4x = ray.v + s * k3v;
    let k4v = geo_geodesic_accel(ray.x + s * k3x, k4x);
    next.x = ray.x + s / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
    next.v = ray.v + s / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
#endif
    return next;
}

// Footprint of the hit threshold at distance `t`, in eye units.
fn threshold(t: f32) -> f32 {
    return march.settings.w * march.shading.z * geo_pixel_footprint(t);
}

struct Hit {
    ray: Ray,
    // Steps taken, as a share of the budget.
    effort: f32,
    hit: bool,
}

// The ray along `dir`, from `start`: how far along it may start (its block's cone) and the
// steps that took, counted in its effort; no farther than `limit`, where it is hidden.
fn trace(dir: vec3<f32>, start: vec2<f32>, limit: f32) -> Hit {
    let far = min(view.far, limit);
    let budget = max(u32(march.settings.y), 1u);
    let longest = select(BIG, march.shading.w, march.shading.w > 0.0);
#ifdef SAMPLED_METRIC
    var omega = 1.0;
    let first = view.near;
#else
    var omega = max(march.settings.z, 1.0);
    let first = max(view.near, start.x);
#endif
    // Started by its cone, a ray steps plainly once it is within a few pixels of the field:
    // an over-relaxed step from near the surface can jump past a near miss that the steps
    // from the eye would have closed in on; farther off, it is relaxed as from the eye.
    let coned = start.x > 0.0;
    var ray = Ray(dir, first, dir * first, dir);
    deck = IDENTITY;
    var best = ray;
    var best_deck = deck;
    var best_error = BIG;
    var previous = 0.0;
    var step = 0.0;
    // The relaxation of the last step taken.
    var relaxed = 1.0;
    var taken = budget;
    var distance = scene_distance(ray_point(ray));
    // Starting inside the solid, the field is followed out rather than in.
    let side = select(-1.0, 1.0, distance >= 0.0);
    for (var i = 0u; i < budget; i++) {
        let signed = side * distance;
        let radius = abs(signed);
        // An over-relaxed step that jumped past the surface: take it back, go on unrelaxed.
        let failed = relaxed > 1.0 && radius + previous < step;
        if failed {
            step -= relaxed * step;
            omega = 1.0;
            relaxed = 1.0;
        } else {
            relaxed = select(omega, 1.0, coned && radius < 4.0 * threshold(ray.t));
            // Never farther than just past a face of the quotient's domain unfolded: the field
            // holds the copies in the neighbouring domains, not the ones beyond.
            let room = fold_room(ray_point(ray)) + threshold(ray.t);
            step = min(min(signed * relaxed, longest), room);
        }
        previous = radius;
        let error = radius / threshold(ray.t);
        if !failed && error < best_error {
            best = ray;
            best_deck = deck;
            best_error = error;
        }
        if (!failed && error < 1.0) || ray.t > far {
            taken = i;
            break;
        }
        ray = ray_advance(ray, step);
        fold(ray_point(ray));
        if i + 1u < budget {
            distance = scene_distance(ray_point(ray));
        }
    }
    deck = best_deck;
    // Out of steps short of the far surface: a crease the budget could not resolve, taken as
    // a hit and darkened by the effort.
    let hit = best.t < far && (best_error < 1.0 || taken == budget);
    let effort = min((f32(taken) + start.y) / f32(budget), 1.0);
    return Hit(best, effort, hit);
}

// How far along `dir` the cone round it, `n` pixels across, gets before a ray in it may come
// within its hit distance of the field: at each point the field's distance less the cone's
// width and the hit distance is clear for every ray in it (by the triangle inequality: a ray
// of the block is within the cone's width of this one at the same distance), so the cone steps
// by that and stops where that is no more than its width; and the steps it took. Rays of the
// block start there, none of them having come near enough to hit on the way.
fn cone(dir: vec3<f32>, n: f32) -> vec2<f32> {
    let budget = max(u32(march.settings.y), 1u);
    let longest = select(BIG, march.shading.w, march.shading.w > 0.0);
    // The cone's width and the hit distance, in pixels.
    let reach = n + march.settings.w * march.shading.z;
    var t = view.near;
    var taken = 0u;
    for (var i = 0u; i < budget; i++) {
        let wide = reach * geo_pixel_footprint(t);
        let clear = scene_distance(geo_point_from_eye(dir, t)) - wide;
        if clear < wide || t > view.far {
            break;
        }
        t += min(clear, longest);
        taken = i + 1u;
    }
    return vec2<f32>(t, f32(taken));
}

@fragment
fn fragment_coarse(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    march_cone = true;
    let n = march.paint.z;
    // The middle of the block, in window pixels.
    let ndc = pixel_to_ndc(frag.xy * n);
    return vec4<f32>(cone(pixel_dir(ndc), n), 0.0, 0.0);
}

// The surface's normal at `p`, in the eye's frame, from the field's gradient over a
// tetrahedron of size `h`.
fn normal_at(p: vec4<f32>, h: f32) -> vec4<f32> {
    let k = vec2<f32>(1.0, -1.0);
    let a = k.xyy * scene_distance(geo_normalize_point(p + vec4<f32>(k.xyy * h, 0.0)));
    let b = k.yyx * scene_distance(geo_normalize_point(p + vec4<f32>(k.yyx * h, 0.0)));
    let c = k.yxy * scene_distance(geo_normalize_point(p + vec4<f32>(k.yxy * h, 0.0)));
    let d = k.xxx * scene_distance(geo_normalize_point(p + vec4<f32>(k.xxx * h, 0.0)));
    return geo_project_tangent(p, vec4<f32>(a + b + c + d, 0.0));
}

fn shade(hit: Hit) -> vec3<f32> {
    let p = geo_normalize_point(ray_point(hit.ray));
    let t = hit.ray.t;
    let lit_by = clamp(march.paint.x, 0.0, 1.0);
    var light = vec3<f32>(1.0);
    // A scene that takes none of the light needs no normal: four evaluations of its field.
    if lit_by > 0.0 {
        let n = normal_at(p, max(0.5 * threshold(t), 1e-6 * t));
        let sun = geo_transport_from_eye(p, lighting.sun_direction);
        let facing = geo_inner(p, n, sun)
            * inverseSqrt(max(geo_inner(p, n, n) * geo_inner(p, sun, sun), 1e-30));
        let painted = mix(lighting.shadow_tint.rgb, lighting.sun_color.rgb, banded(facing))
            + point_lights(p, n);
        let occlusion = clamp(1.0 - march.shading.x * hit.effort, 0.0, 1.0);
        light = mix(vec3<f32>(1.0), painted * occlusion, lit_by);
    }
    let lit = sdf_color(scene_point(p)) * light;
    let fogged = max(geo_fog(t, lighting.fog_density), smoothstep(0.85 * view.far, view.far, t));
    let fog = fogged * clamp(march.paint.y, 0.0, 1.0);
    let haze = 1.0 - (1.0 - fog) * (1.0 - clamp(march.shading.y, 0.0, 1.0));
    return mix(lit, sky_haze(hit.ray.dir), haze);
}

// The image's alpha for a hit: 1, 2 where the field keeps its surface out of the mosh
// patches, or 3 where its surface carries their mosh wherever it is.
fn surface_alpha(hit: Hit) -> f32 {
#ifdef SDF_KEPT
    if sdf_kept(scene_point(geo_normalize_point(ray_point(hit.ray)))) {
        return 2.0;
    }
#endif
#ifdef SDF_MOSHED
    if sdf_moshed(scene_point(geo_normalize_point(ray_point(hit.ray)))) {
        return 3.0;
    }
#endif
    return 1.0;
}

// The pixel's direction at the eye; every geometry's projection maps directions alike.
fn pixel_dir(ndc: vec2<f32>) -> vec3<f32> {
    return normalize(vec3<f32>(ndc / view.proj_scale, -1.0));
}

struct Surface {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

@fragment
fn fragment(@builtin(position) frag: vec4<f32>) -> Surface {
    let ndc = pixel_to_ndc(frag.xy);
    let hit = trace(pixel_dir(ndc), start_at(frag.xy), limit_at(frag.xy));
    if !hit.hit {
        discard;
    }
    return Surface(
        vec4<f32>(shade(hit), surface_alpha(hit)),
        geo_distance_to_depth(hit.ray.t, ndc),
    );
}

struct Half {
    // Alpha 1 (or 2, kept) on a hit, 0 for a miss.
    @location(0) color: vec4<f32>,
    @location(1) distance: f32,
}

@fragment
fn fragment_half(@builtin(position) frag: vec4<f32>) -> Half {
    let ndc = pixel_to_ndc(frag.xy * march.shading.z);
    let pixel = frag.xy * march.shading.z;
    let hit = trace(pixel_dir(ndc), start_at(pixel), limit_at(pixel));
    if !hit.hit {
        return Half(vec4<f32>(0.0), 0.0);
    }
    return Half(vec4<f32>(shade(hit), surface_alpha(hit)), hit.ray.t);
}

@group(2) @binding(0) var half_color: texture_2d<f32>;
@group(2) @binding(1) var half_distance: texture_2d<f32>;

@fragment
fn composite(@builtin(position) frag: vec4<f32>) -> Surface {
    let size = vec2<i32>(textureDimensions(half_color));
    let texel = clamp(vec2<i32>(frag.xy / march.shading.z), vec2<i32>(0), size - 1);
    let color = textureLoad(half_color, texel, 0);
    if color.a < 0.5 {
        discard;
    }
    let d = textureLoad(half_distance, texel, 0).r;
    return Surface(color, geo_distance_to_depth(d, pixel_to_ndc(frag.xy)));
}
