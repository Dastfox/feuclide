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
// The image's alpha is 1 on a surface; composed with `SDF_ALPHA`, it is what the field's
// `sdf_alpha` says there (from 1 to 4), a mark for the caller's image effects to read.
//
// Entry points: `vertex` (fullscreen triangle); `fragment` (full resolution: colour, depth,
// misses discarded); `fragment_half` (into two offscreen targets: colour with alpha 1 on a hit,
// and the distance); `composite` (those targets brought up to full resolution, with depth:
// between four pixels marched that agree, what lies between them, and elsewhere, at edges, the
// pixel's own ray traced at full resolution);
// `classify`, `finalize` and `refine` (compute: the pixels whose own rays the composite
// needs, listed and traced at full resolution together);
// `fragment_coarse` (one cone per block of `march.paint.z` pixels, into the coarse target the
// other two start their rays from, `coarse_start`). Rays stop where the scene marched ahead of
// this one is (`ahead_distance`, `march.paint.w`): what is behind it is never marched.

#import fk::view::{view, lighting, pixel_to_ndc}
#import fk::geometry::{
    geo_apply_iso, geo_normalize_point, geo_project_tangent, geo_inner, geo_transport_from_eye,
    geo_point_from_eye, geo_log_origin, geo_fog, geo_pixel_footprint, geo_distance_to_depth,
    geo_distance, geo_bisector_distance, geo_project,
}
#import fk::sky::sky_haze
#import fk::portal::portal_span
#import fk::paint::{banded, point_lights}
#import fk::march::{march, march_cone}
#import fk::sdf::{sdf_distance, sdf_color}
#ifdef SDF_ALPHA
#import fk::sdf::sdf_alpha
#endif
#ifdef SAMPLED_METRIC
#import fk::metric::geo_geodesic_accel
#endif

const BIG: f32 = 3.4e38;

// Pixels across the tiles of a scene marched one tile in each square of them
// (`march.interleave`).
const INTERLEAVED: u32 = 8u;

// Added to the alpha of a pixel carried from the last frame rather than marched, while the image
// settles over frames (`march.interleave.w`); the settle pass takes it off. Over any mark a
// surface's alpha may carry (a field's `sdf_alpha`, an instance's colour's alpha).
const CARRIED: f32 = 64.0;

// The alpha `a` without the mark of a pixel carried.
fn unmarked(a: f32) -> f32 {
    return select(a, a - CARRIED, a >= CARRIED - 0.5);
}

// Which of its square of `march.interleave.x` tiles across the tile `tile` is.
fn interleaved_tile(tile: vec2<u32>) -> u32 {
    let n = u32(march.interleave.x);
    return tile.x % n + n * (tile.y % n);
}

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
    // Out of a quotient there is one copy: no need to ask the field which is nearest.
    if march.fold.count.x < 0.5 {
        return scene_point_by(deck, p);
    }
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
    var d = 0.0;
    if abs(k) < 1.5 {
        d = textureLoad(ahead_distance, clamp(vec2<i32>(floor(pixel)), vec2<i32>(0), size - 1), 0).r;
    } else {
        // Marched coarser than the window: hidden only where all four of its pixels round
        // this one are, and no nearer than the farthest of them.
        let base = vec2<i32>(floor(pixel / abs(k) - 0.5));
        var nearest = BIG;
        d = 0.0;
        for (var i = 0; i < 4; i++) {
            let texel = clamp(base + vec2<i32>(i % 2, i / 2), vec2<i32>(0), size - 1);
            let here = textureLoad(ahead_distance, texel, 0).r;
            nearest = min(nearest, here);
            d = max(d, here);
        }
        if nearest <= 0.0 {
            return BIG;
        }
    }
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
// Window pixels per pixel of the ray being traced: the scene's (`march.shading.z`), but one
// for a pixel's own ray traced at full resolution where the composite finds an edge.
var<private> traced_pixels: f32 = 0.0;

fn pixels() -> f32 {
    return select(march.shading.z, traced_pixels, traced_pixels > 0.0);
}

fn threshold(t: f32) -> f32 {
    return march.settings.w * pixels() * geo_pixel_footprint(t);
}

struct Hit {
    ray: Ray,
    // Steps taken, as a share of the budget.
    effort: f32,
    hit: bool,
    // Steps taken, as a share of the budget, its block's cone shared among the block's rays.
    own: f32,
}

// Whether each ray's steps are drawn in place of its surface (`march.debug.x`).
fn heat_on() -> bool {
    return march.debug.x > 0.5;
}

// A ray's steps, `own` a share of the budget, as a colour: blue for few, through green and
// yellow, red for all of them, by its square root.
fn heat(own: f32) -> vec3<f32> {
    let x = 4.0 * sqrt(clamp(own, 0.0, 1.0));
    return clamp(vec3<f32>(x - 1.5, 1.5 - abs(x - 2.0), 2.5 - x), vec3<f32>(0.0), vec3<f32>(1.0));
}

// The ray along `dir`, from `start`: how far along it may start (its block's cone) and the
// steps that took, counted in its effort; no farther than `limit`, where it is hidden; only
// along its stretch of the view's region of the portals' tree (`fk::portal`).
fn trace(dir: vec3<f32>, start: vec2<f32>, limit: f32) -> Hit {
    let span = portal_span(dir, view.near, view.far);
    let far = min(min(view.far, limit), span.y);
    let budget = max(u32(march.settings.y), 1u);
    let longest = select(BIG, march.shading.w, march.shading.w > 0.0);
#ifdef SAMPLED_METRIC
    var omega = 1.0;
    let first = max(view.near, span.x);
#else
    var omega = max(march.settings.z, 1.0);
    let first = max(max(view.near, start.x), span.x);
#endif
    // Started by its cone, a ray steps plainly once it is within a few pixels of the field:
    // an over-relaxed step from near the surface can jump past a near miss that the steps
    // from the eye would have closed in on; farther off, it is relaxed as from the eye.
    let coned = start.x > 0.0;
    var ray = Ray(dir, first, dir * first, dir);
    // Nothing of it in the view's region.
    if first >= far {
        let block = max(march.paint.z * march.paint.z, 1.0);
        return Hit(ray, start.y / f32(budget), false, start.y / block / f32(budget));
    }
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
            relaxed = select(omega, 1.0, coned && radius < 3.0 * threshold(ray.t));
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
    let block = max(march.paint.z * march.paint.z, 1.0);
    let own = min((f32(taken) + start.y / block) / f32(budget), 1.0);
    return Hit(best, effort, hit, own);
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
    // A block that is a tile carried this frame (`march.interleave`) marches no rays.
    if march.interleave.x > 1.5 && march.interleave.z > 0.5 && u32(n) == INTERLEAVED {
        if interleaved_tile(vec2<u32>(frag.xy)) != u32(march.interleave.y) {
            return vec4<f32>(0.0);
        }
    }
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
    if heat_on() {
        return heat(hit.own);
    }
    let p = geo_normalize_point(ray_point(hit.ray));
    let t = hit.ray.t;
    // The colour before the normal: it is asked where the march last asked the field, which
    // a field may remember.
    let base = sdf_color(scene_point(p));
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
    let lit = base * light;
    let fogged = max(geo_fog(t, lighting.fog_density), smoothstep(0.85 * view.far, view.far, t));
    let fog = fogged * clamp(march.paint.y, 0.0, 1.0);
    let haze = 1.0 - (1.0 - fog) * (1.0 - clamp(march.shading.y, 0.0, 1.0));
    return mix(lit, sky_haze(hit.ray.dir), haze);
}

// Where a ray that hit nothing is drawn while its steps are (`heat_on`): as far as it got, short
// of the far surface.
fn missed(hit: Hit) -> f32 {
    return clamp(hit.ray.t, view.near, 0.999 * view.far);
}

// The image's alpha for a hit: 1, or what the field's `sdf_alpha` says there.
fn surface_alpha(hit: Hit) -> f32 {
#ifdef SDF_ALPHA
    return clamp(sdf_alpha(scene_point(geo_normalize_point(ray_point(hit.ray)))), 1.0, 4.0);
#else
    return 1.0;
#endif
}

// Where the ray through `pixel` (window pixels, a pixel's middle) goes through it this frame, in
// normalized device coordinates: its middle, or another point of it while the image settles over
// frames (`view.jitter`).
fn ray_ndc(pixel: vec2<f32>) -> vec2<f32> {
    return pixel_to_ndc(pixel + view.jitter);
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
    let ndc = ray_ndc(frag.xy);
    let hit = trace(pixel_dir(ndc), start_at(frag.xy), limit_at(frag.xy));
    if !hit.hit {
        if heat_on() {
            return Surface(vec4<f32>(heat(hit.own), 1.0), geo_distance_to_depth(missed(hit), ndc));
        }
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
    let pixel = frag.xy * march.shading.z;
    let ndc = ray_ndc(pixel);
    let limit = limit_at(pixel);
    let hit = trace(pixel_dir(ndc), start_at(pixel), limit);
    if !hit.hit {
        if heat_on() {
            return Half(vec4<f32>(heat(hit.own), 1.0), missed(hit));
        }
        // Nothing, or, where the scene ahead stopped it, -1: hidden, not empty.
        return Half(vec4<f32>(0.0), select(0.0, -1.0, limit < BIG));
    }
    return Half(vec4<f32>(shade(hit), surface_alpha(hit)), hit.ray.t);
}

// A scene marched one tile in each square of them (`march.interleave`): its last frame's image, colour and
// distance, the pixels not marched this frame carried from it.
@group(2) @binding(12) var last_color: texture_2d<f32>;
@group(2) @binding(13) var last_distance: texture_2d<f32>;

// The pixel `pixel` (window pixels, a pixel's middle) as the last frame had it, found where the
// eye's motion took it (by the distance the last frame had there), or, come into view from out
// of the last image, the nearest it has at its edge: its colour and distance (nothing behind the
// last eye).
fn carried(pixel: vec2<f32>) -> Half {
    let size = vec2<i32>(textureDimensions(last_distance));
    var d = textureLoad(last_distance, clamp(vec2<i32>(pixel), vec2<i32>(0), size - 1), 0).r;
    if d <= 0.0 {
        d = view.far;
    }
    let point = geo_point_from_eye(pixel_dir(pixel_to_ndc(pixel)), d);
    let clip = geo_project(geo_apply_iso(march.carry, point));
    if clip.w <= 1e-4 {
        return Half(vec4<f32>(0.0), 0.0);
    }
    let last = vec2<f32>(0.5 * (clip.x / clip.w + 1.0), 0.5 * (1.0 - clip.y / clip.w)) * view.size;
    let at = clamp(vec2<i32>(floor(last)), vec2<i32>(0), size - 1);
    return Half(textureLoad(last_color, at, 0), textureLoad(last_distance, at, 0).r);
}

// Every pixel of a scene marched one tile in each square of them, carried from its last frame;
// the tiles marched this frame are drawn over them (`vertex_tiles`).
@fragment
fn fragment_carry(@builtin(position) frag: vec4<f32>) -> Half {
    var last = carried(frag.xy);
    last.color.a = unmarked(last.color.a);
    // Carried into a part of the image the view no longer draws there (a mouth came over it).
    if last.color.a > 0.5 && !in_span(pixel_to_ndc(frag.xy), last.distance) {
        return Half(vec4<f32>(0.0), 0.0);
    }
    if march.interleave.w > 0.5 {
        last.color.a += CARRIED;
    }
    return last;
}

// The tiles of a scene marched one in each square of them that are marched this frame, as two
// triangles each, `instance` the tile among them.
@vertex
fn vertex_tiles(
    @builtin(vertex_index) index: u32,
    @builtin(instance_index) instance: u32,
) -> @builtin(position) vec4<f32> {
    let n = max(u32(march.interleave.x), 1u);
    let marched = u32(march.interleave.y);
    let first = vec2<u32>(marched % n, marched / n);
    let tiles = (vec2<u32>(view.size) + INTERLEAVED - 1u) / INTERLEAVED;
    let across = max((tiles.x + n - 1u - first.x) / n, 1u);
    let tile = first + n * vec2<u32>(instance % across, instance / across);
    var corners = array<vec2<u32>, 6>(
        vec2<u32>(0u, 0u), vec2<u32>(1u, 0u), vec2<u32>(0u, 1u),
        vec2<u32>(1u, 0u), vec2<u32>(1u, 1u), vec2<u32>(0u, 1u),
    );
    let pixel = min(vec2<f32>((tile + corners[index]) * INTERLEAVED), view.size);
    let uv = pixel / view.size;
    return vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
}

@group(3) @binding(0) var half_color: texture_2d<f32>;
@group(3) @binding(1) var half_distance: texture_2d<f32>;
// The pixels whose own rays are traced at full resolution (`classify`, `refine`): their list,
// how many, the dispatch over them (x, y, z, and how many again, read by `refine`), and what
// their rays found, read back by `composite`.
@group(3) @binding(4) var<storage, read_write> edge_list: array<u32>;
@group(3) @binding(5) var<storage, read_write> edge_count: atomic<u32>;
@group(3) @binding(10) var<storage, read_write> edge_args: array<u32, 4>;
@group(3) @binding(11) var<storage, read> edge_found: array<u32, 4>;
@group(3) @binding(6) var refined_color_out: texture_storage_2d<rgba16float, write>;
@group(3) @binding(7) var refined_distance_out: texture_storage_2d<r32float, write>;
@group(3) @binding(8) var refined_color: texture_2d<f32>;
@group(3) @binding(9) var refined_distance: texture_2d<f32>;

// How far apart the four pixels round a pixel may be and still be one surface: in each
// channel of their colour (linear), and in distance, as a share of the nearest.
const COMPOSITE_COLOUR: f32 = 0.012;
const COMPOSITE_DEPTH: f32 = 0.1;

// What the composite shows at `pixel` (window pixels, a pixel's middle) of a scene marched at
// half resolution: `kind` 0 nothing (the sky, or hidden by the scene ahead), 1 what lies
// between the four pixels marched round it (`colour`, `distance`), 2 its own ray, traced at
// full resolution (an edge, a crease or a fine thing between them).
struct Between {
    kind: u32,
    colour: vec4<f32>,
    distance: f32,
}

fn between(pixel: vec2<f32>) -> Between {
    let k = march.shading.z;
    let size = vec2<i32>(textureDimensions(half_color));
    // The four pixels marched round this one, and where it falls between them.
    let at = pixel / k - 0.5;
    let base = vec2<i32>(floor(at));
    let f = at - floor(at);
    let t00 = clamp(base, vec2<i32>(0), size - 1);
    let t11 = clamp(base + vec2<i32>(1), vec2<i32>(0), size - 1);
    let t10 = vec2<i32>(t11.x, t00.y);
    let t01 = vec2<i32>(t00.x, t11.y);
    let c00 = textureLoad(half_color, t00, 0);
    let c10 = textureLoad(half_color, t10, 0);
    let c01 = textureLoad(half_color, t01, 0);
    let c11 = textureLoad(half_color, t11, 0);
    let d00 = textureLoad(half_distance, t00, 0).r;
    let d10 = textureLoad(half_distance, t10, 0).r;
    let d01 = textureLoad(half_distance, t01, 0).r;
    let d11 = textureLoad(half_distance, t11, 0).r;
    // One surface across them: all four the same kind (sky, or surface with the same alpha),
    // the same colour, and no step in distance. Then this pixel is what lies between them.
    let low = min(min(c00, c10), min(c01, c11));
    let high = max(max(c00, c10), max(c01, c11));
    let near = min(min(d00, d10), min(d01, d11));
    let far = max(max(d00, d10), max(d01, d11));
    if high.a < 0.5 {
        // No surface round it: the sky, unless the scene ahead hid one of them and not this
        // pixel, which may then show something the four did not.
        if near >= 0.0 || limit_at(pixel) < BIG {
            return Between(0u, vec4<f32>(0.0), 0.0);
        }
        return Between(2u, vec4<f32>(0.0), 0.0);
    }
    if high.a - low.a < 0.5
        && all(high.rgb - low.rgb <= vec3<f32>(COMPOSITE_COLOUR))
        && far - near <= COMPOSITE_DEPTH * near
    {
        let top = mix(c00, c10, f.x);
        let bottom = mix(c01, c11, f.x);
        let colour = vec4<f32>(mix(top.rgb, bottom.rgb, f.y), c00.a);
        let d = mix(mix(d00, d10, f.x), mix(d01, d11, f.x), f.y);
        return Between(1u, colour, d);
    }
    return Between(2u, vec4<f32>(0.0), 0.0);
}

// Lists the pixels whose own rays the composite needs, one invocation a window pixel.
@compute @workgroup_size(8, 8)
fn classify(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = vec2<u32>(view.size);
    if id.x >= size.x || id.y >= size.y {
        return;
    }
    if between(vec2<f32>(id.xy) + 0.5).kind == 2u {
        let i = atomicAdd(&edge_count, 1u);
        edge_list[i] = id.x | (id.y << 16u);
    }
}

// The dispatch over the pixels listed, 64 to a group.
@compute @workgroup_size(1)
fn finalize() {
    let n = atomicLoad(&edge_count);
    edge_args[0] = (n + 63u) / 64u;
    edge_args[1] = 1u;
    edge_args[2] = 1u;
    edge_args[3] = n;
}

// Traces the rays of the pixels listed at full resolution: none of a group's lanes waits on
// another's pixel.
@compute @workgroup_size(64)
fn refine(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= edge_found[3] {
        return;
    }
    let packed = edge_list[id.x];
    let texel = vec2<u32>(packed & 0xffffu, packed >> 16u);
    let pixel = vec2<f32>(texel) + 0.5;
    traced_pixels = 1.0;
    let ndc = ray_ndc(pixel);
    let hit = trace(pixel_dir(ndc), start_at(pixel), limit_at(pixel));
    var colour = vec4<f32>(0.0);
    var d = 0.0;
    if hit.hit {
        colour = vec4<f32>(shade(hit), surface_alpha(hit));
        d = hit.ray.t;
    } else if heat_on() {
        colour = vec4<f32>(heat(hit.own), 1.0);
        d = missed(hit);
    }
    textureStore(refined_color_out, texel, colour);
    textureStore(refined_distance_out, texel, vec4<f32>(d, 0.0, 0.0, 0.0));
}

// Whether a surface found `d` along the pixel's ray at `ndc` is in its stretch of the view's
// region: what the composite brings up from the pixels round it may not be.
fn in_span(ndc: vec2<f32>, d: f32) -> bool {
    let span = portal_span(pixel_dir(ndc), view.near, view.far);
    return d >= span.x && d <= span.y;
}

@fragment
fn composite(@builtin(position) frag: vec4<f32>) -> Surface {
    let k = march.shading.z;
    let ndc = ray_ndc(frag.xy);
    if k < 1.5 {
        // Marched at full resolution (a scene ahead): each pixel is its own.
        let size = vec2<i32>(textureDimensions(half_color));
        let texel = clamp(vec2<i32>(frag.xy), vec2<i32>(0), size - 1);
        let color = textureLoad(half_color, texel, 0);
        if unmarked(color.a) < 0.5 {
            discard;
        }
        return Surface(color, geo_distance_to_depth(textureLoad(half_distance, texel, 0).r, ndc));
    }
    let here = between(frag.xy);
    if here.kind == 1u {
        if !in_span(ndc, here.distance) {
            discard;
        }
        return Surface(here.colour, geo_distance_to_depth(here.distance, ndc));
    }
    if here.kind == 2u {
        // Its own ray, traced by `refine`.
        let texel = vec2<i32>(frag.xy);
        let colour = textureLoad(refined_color, texel, 0);
        if colour.a >= 0.5 {
            let d = textureLoad(refined_distance, texel, 0).r;
            return Surface(colour, geo_distance_to_depth(d, ndc));
        }
    }
    discard;
}
