// The settle pass: one fullscreen triangle blending the image with the last ones, after the
// opaque pass and before the ink (drawn crisp on the settled image), so that every later effect
// sees the settled image.
//
// Each frame the rays (and the raster path's vertices) go through another point of their pixel
// (`view.jitter`), so the images settled over a few frames are each pixel's average: edges and
// what is finer than a pixel stop shimmering. The last image is found again where the eye's
// motion (`settle.motion`, this frame's eye coordinates to the last's) took each surface, by
// its distance, and read between its pixels with a Catmull-Rom filter, so it stays sharp. It is
// dropped where its distance is none of those round the pixel now (something came in front, or
// went), and held to the colours round the pixel now while the eye moves, and inside surfaces
// whatever it does, so that what moves leaves no trail; at edges, the eye held still, it is not,
// and what is finer than a pixel is averaged too.
//
// A pixel a ray-marched scene carried from its last frame rather than marched (its alpha marked,
// `CARRIED` added: `fk_render::RayMarched::interleave`) brings nothing new: the last image is
// kept there as it is, so that what the scene marched every few frames settles as if marched
// every frame.
//
// It draws into a pair of targets of its own (the colour, and the distance the next frame finds
// it again by), the alpha (surface, sky, kept out of the mosh) passed on as it is now, unmarked.

#import fk::view::{view, pixel_to_ndc}
#import fk::geometry::{geo_apply_iso, geo_depth_to_distance, geo_point_from_eye, geo_project}

struct Settle {
    // Columns of the matrix taking this frame's eye coordinates to the last frame's.
    motion: array<vec4<f32>, 4>,
    // How much of this frame goes in (1 over the frames it settles over); 1 when the last
    // image is there; how far the eye's motion may carry a pixel (pixels) while it is still
    // held still; how much of it goes in while the eye moves, carried or not.
    settings: vec4<f32>,
}

@group(1) @binding(0) var scene_color: texture_2d<f32>;
// With multisampling the depth target keeps its samples; sample 0 is read.
#ifdef MULTISAMPLED
@group(1) @binding(1) var scene_depth: texture_depth_multisampled_2d;
#else
@group(1) @binding(1) var scene_depth: texture_depth_2d;
#endif
@group(1) @binding(2) var<uniform> settle: Settle;
@group(1) @binding(3) var last_color: texture_2d<f32>;
@group(1) @binding(4) var last_distance: texture_2d<f32>;
@group(1) @binding(5) var last_sampler: sampler;

@vertex
fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

// The distance along the pixel `at`'s ray, clamped into the image; the far surface on the sky.
fn distance_at(at: vec2<i32>) -> f32 {
    let size = vec2<i32>(view.size);
    let p = clamp(at, vec2<i32>(0), size - 1);
    let depth = textureLoad(scene_depth, p, 0);
    if depth <= 0.0 {
        return view.far;
    }
    return geo_depth_to_distance(depth, pixel_to_ndc(vec2<f32>(p) + 0.5));
}

// Where the surface at pixel `at`, `d` away, was in the last frame, in pixels; far outside the
// image when it was behind the last eye.
fn last_position(at: vec2<f32>, d: f32) -> vec2<f32> {
    let ndc = pixel_to_ndc(at);
    // Every geometry's projection maps directions at the eye the same way.
    let dir = normalize(vec3<f32>(ndc / view.proj_scale, -1.0));
    let m = mat4x4<f32>(settle.motion[0], settle.motion[1], settle.motion[2], settle.motion[3]);
    let clip = geo_project(geo_apply_iso(m, geo_point_from_eye(dir, d)));
    if clip.w <= 1e-4 {
        return vec2<f32>(-1e9);
    }
    let last = clip.xy / clip.w;
    return vec2<f32>(0.5 * (last.x + 1.0), 0.5 * (1.0 - last.y)) * view.size;
}

// The last image at `pixel` (pixels, between its pixels), through a Catmull-Rom filter in nine
// bilinear taps: sharp where it falls between pixels, exact on one.
fn last_at(pixel: vec2<f32>) -> vec3<f32> {
    let size = vec2<f32>(textureDimensions(last_color));
    let c = floor(pixel - 0.5) + 0.5;
    let f = pixel - c;
    let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3 = f * f * (-0.5 + 0.5 * f);
    let w12 = w1 + w2;
    let t0 = (c - 1.0) / size;
    let t12 = (c + w2 / w12) / size;
    let t3 = (c + 2.0) / size;
    var sum = vec3<f32>(0.0);
    sum += textureSampleLevel(last_color, last_sampler, vec2<f32>(t0.x, t0.y), 0.0).rgb * (w0.x * w0.y);
    sum += textureSampleLevel(last_color, last_sampler, vec2<f32>(t12.x, t0.y), 0.0).rgb * (w12.x * w0.y);
    sum += textureSampleLevel(last_color, last_sampler, vec2<f32>(t3.x, t0.y), 0.0).rgb * (w3.x * w0.y);
    sum += textureSampleLevel(last_color, last_sampler, vec2<f32>(t0.x, t12.y), 0.0).rgb * (w0.x * w12.y);
    sum += textureSampleLevel(last_color, last_sampler, vec2<f32>(t12.x, t12.y), 0.0).rgb * (w12.x * w12.y);
    sum += textureSampleLevel(last_color, last_sampler, vec2<f32>(t3.x, t12.y), 0.0).rgb * (w3.x * w12.y);
    sum += textureSampleLevel(last_color, last_sampler, vec2<f32>(t0.x, t3.y), 0.0).rgb * (w0.x * w3.y);
    sum += textureSampleLevel(last_color, last_sampler, vec2<f32>(t12.x, t3.y), 0.0).rgb * (w12.x * w3.y);
    sum += textureSampleLevel(last_color, last_sampler, vec2<f32>(t3.x, t3.y), 0.0).rgb * (w3.x * w3.y);
    return max(sum, vec3<f32>(0.0));
}

struct Settled {
    @location(0) color: vec4<f32>,
    @location(1) distance: f32,
}

// Added to the alpha of a pixel carried rather than marched: `CARRIED` in `ray_march.wgsl`.
const CARRIED: f32 = 8.0;

@fragment
fn fragment(@builtin(position) frag: vec4<f32>) -> Settled {
    let pixel = vec2<i32>(frag.xy);
    var now = textureLoad(scene_color, pixel, 0);
    let carried = now.a >= CARRIED - 0.5;
    if carried {
        now.a -= CARRIED;
    }
    let d = distance_at(pixel);
    if settle.settings.y < 0.5 {
        return Settled(now, d);
    }
    let last = last_position(frag.xy, d);
    if any(last < vec2<f32>(0.5)) || any(last > view.size - 0.5) {
        return Settled(now, d);
    }
    // The colours and distances round the pixel now.
    var low = now.rgb;
    var high = now.rgb;
    var near = d;
    var far = d;
    for (var i = 0; i < 9; i++) {
        let at = pixel + vec2<i32>(i % 3 - 1, i / 3 - 1);
        let size = vec2<i32>(view.size);
        let c = textureLoad(scene_color, clamp(at, vec2<i32>(0), size - 1), 0).rgb;
        low = min(low, c);
        high = max(high, c);
        let around = distance_at(at);
        near = min(near, around);
        far = max(far, around);
    }
    // Something came in front of what was there, or went: the last image is of something else.
    let was_at = textureLoad(last_distance, vec2<i32>(last), 0).r;
    if was_at < 0.95 * near || was_at > 1.05 * far {
        return Settled(now, d);
    }
    var was = last_at(last);
    // Held to the colours round the pixel now while the eye moves, and inside a surface (no
    // edge round the pixel, so nothing finer than it) whatever the eye does: what moves across
    // the image while the eye is still (a crawling arm) leaves no trail behind it.
    let still = settle.settings.z;
    let moving = smoothstep(still, 2.0 * still + 0.05, length(last - frag.xy));
    let inside = select(0.0, 1.0, far - near <= 0.02 * near);
    was = mix(was, clamp(was, low, high), max(moving, inside));
    // Carried, nothing new: the last image as it is, the eye still. Moving, the last images,
    // found again between their pixels, would soften it: this frame weighs more, carried or not.
    let share = select(settle.settings.x, 0.0, carried);
    let settled = mix(was, now.rgb, mix(share, max(settle.settings.w, share), moving));
    return Settled(vec4<f32>(settled, now.a), d);
}
