// The mosh pass: one fullscreen triangle rebuilding the image from the held mosh and the live
// image, like a video whose key frames were lost.
//
// Each block of the held image moves the way the camera's motion moved the live image at the
// block's centre since the last frame: the centre is put back in space from the depth buffer,
// carried into the last frame's eye and projected there. The live image's edges bleed into the
// held colours as residuals would, and a block whose turn has come shows the live image
// instead: whole regions first, scattered with single blocks. Moved pixels are read whole,
// never filtered, so the image smears instead of blurring; their alpha (surface or sky) moves
// with them.
//
// Between moshes (progress 1) the same runs only inside the patches: discs with ragged block
// edges, and the arms some reach out of them (tapering, curling, undulating tentacles), where
// the held image never stops moving, the camera's motion plus a slow swirl
// outwards from the centre, fading back into the live image smoothly, everywhere and towards
// the edge, never block by block. Outside them
// the live image is shown as it is, and so is what is kept out of them (alpha 2): it is
// neither moshed nor carried into the blocks around it. A surface that carries their mosh
// (alpha 3) is moshed wherever it is, in the first patch's mosh, block by block round it.

#import fk::view::{view, world, pixel_to_ndc}
#import fk::geometry::{geo_apply_iso, geo_depth_to_distance, geo_point_from_eye, geo_project}

struct Mosh {
    // Columns of the matrix taking this frame's eye coordinates to the last frame's.
    motion: array<vec4<f32>, 4>,
    // 0 the held image; every block is live again before 1.
    progress: f32,
    // Size of a block in pixels.
    block: f32,
    // How fast the live image's edges bleed in, per second at progress 1.
    bleed: f32,
    // Seconds since the last frame.
    delta: f32,
    // Differs from one mosh to the next.
    seed: f32,
    patch_count: u32,
    // Each patch: its centre and radius, in pixels from the top-left corner, and its block in
    // pixels.
    patches: array<vec4<f32>, 4>,
    // Each patch: its flow in pixels per second, how often a block breaks back to the live
    // image and how fast the live edges bleed in, per second, and its index.
    patch_motion: array<vec4<f32>, 4>,
    // Each patch's arms: how many, how far past its radius they reach and how wide they are at
    // the root, in pixels, and how far they curl along their length (radians at the tip).
    patch_arms: array<vec4<f32>, 4>,
    // Each patch's arms' turn about its centre and how far they undulate, radians.
    patch_turn: array<vec4<f32>, 4>,
    // 0 blocks, 1 pixels: no blocks, every pixel moved by its own motion and the live image
    // coming back in smoothly, in soft regions `block` pixels across; 2 smear: macroblocks
    // `block` pixels across torn apart, corrupt vectors, colour trails.
    kind: u32,
}

@group(1) @binding(0) var held: texture_2d<f32>;
@group(1) @binding(1) var live: texture_2d<f32>;
// With multisampling the depth target keeps its samples; sample 0 is read.
#ifdef MULTISAMPLED
@group(1) @binding(2) var live_depth: texture_depth_multisampled_2d;
#else
@group(1) @binding(2) var live_depth: texture_depth_2d;
#endif
@group(1) @binding(3) var<uniform> mosh: Mosh;

@vertex
fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

// A hash of `p` in [0, 1).
fn hash(p: vec3<f32>) -> f32 {
    var q = fract(p * 0.1031);
    q += dot(q, q.zyx + 31.32);
    return fract((q.x + q.y) * q.z);
}

// Smooth value noise in [0, 1], one lattice per `layer`.
fn noise(p: vec2<f32>, layer: f32) -> f32 {
    let i = floor(p);
    let f = p - i;
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash(vec3<f32>(i, layer));
    let b = hash(vec3<f32>(i + vec2<f32>(1.0, 0.0), layer));
    let c = hash(vec3<f32>(i + vec2<f32>(0.0, 1.0), layer));
    let d = hash(vec3<f32>(i + vec2<f32>(1.0, 1.0), layer));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Where the live image at pixel `at` was in the last frame, in pixels.
fn last_position(at: vec2<f32>) -> vec2<f32> {
    let ndc = pixel_to_ndc(at);
    let depth = textureLoad(live_depth, vec2<i32>(at), 0);
    // Every geometry's projection maps directions at the eye the same way.
    let dir = normalize(vec3<f32>(ndc / view.proj_scale, -1.0));
    let point = geo_point_from_eye(dir, geo_depth_to_distance(depth, ndc));
    let m = mat4x4<f32>(mosh.motion[0], mosh.motion[1], mosh.motion[2], mosh.motion[3]);
    let clip = geo_project(geo_apply_iso(m, point));
    if clip.w <= 1e-4 {
        // Behind the last eye: no idea where it was.
        return at;
    }
    let last = clip.xy / clip.w;
    return vec2<f32>(0.5 * (last.x + 1.0), 0.5 * (1.0 - last.y)) * view.size;
}

// Mean light of `image` over about a block around `at`: four taps at its quarter points.
fn local_luma(image: texture_2d<f32>, at: vec2<i32>, size: vec2<i32>) -> f32 {
    let r = max(i32(0.5 * mosh.block), 1);
    var sum = 0.0;
    for (var i = 0; i < 4; i++) {
        let corner = vec2<i32>(select(-r, r, (i & 1) == 1), select(-r, r, (i & 2) == 2));
        sum += luma(textureLoad(image, clamp(at + corner, vec2<i32>(0), size - 1), 0).rgb);
    }
    return 0.25 * sum;
}

// The block motion of the held image at `centre` since the last frame, in pixels.
fn block_motion(centre: vec2<f32>) -> vec2<f32> {
    let at = min(centre, view.size - 1.0);
    let limit = 0.25 * view.size.y;
    return clamp(at - last_position(at), vec2<f32>(-limit), vec2<f32>(limit));
}

// The held pixel that moved to `pixel` by `motion` (whole pixels, rounded at random per block
// and frame so slow motion still moves), with the live edges bled in at `rate` per second.
fn moved_pixel(pixel: vec2<i32>, block: vec2<f32>, motion: vec2<f32>, rate: f32) -> vec4<f32> {
    let size = vec2<i32>(textureDimensions(held));
    let jitter = vec2<f32>(
        hash(vec3<f32>(block, 61.0 * world.time)),
        hash(vec3<f32>(block.yx, 47.0 * world.time + 3.0)),
    );
    let source = clamp(pixel - vec2<i32>(floor(motion + jitter)), vec2<i32>(0), size - 1);
    let moved = textureLoad(held, source, 0);
    // Residuals: the live image's edges, its light against its surroundings, in the held
    // colours, which keep their own brightness.
    let now = luma(textureLoad(live, pixel, 0).rgb) - local_luma(live, pixel, size);
    let was = luma(moved.rgb) - local_luma(held, source, size);
    let bleed = 1.0 - exp(-rate * mosh.delta);
    return vec4<f32>(max(moved.rgb + (now - was) * bleed, vec3<f32>(0.0)), moved.a);
}

// How far inside patch `i`'s arms a point `offset` from its centre (`r` away) is, 0 outside
// to 1 in the middle of an arm near its root; its disc's `edge` is where the arms start.
fn in_arms(i: u32, offset: vec2<f32>, r: f32, edge: f32) -> f32 {
    let arms = mosh.patch_arms[i];
    let count = arms.x;
    if count < 0.5 || arms.y <= 0.0 || r < 0.5 * edge {
        return 0.0;
    }
    // How far out along an arm, 0 at the disc's edge to 1 at the tip.
    let t = (r - edge) / arms.y;
    if t >= 1.0 {
        return 0.0;
    }
    let along = clamp(t, 0.0, 1.0);
    let turn = mosh.patch_turn[i];
    // Curling along its length, waves running out along it.
    let bend = arms.w * along * along
        + turn.y * sin(6.2831853 * (1.5 * along - 0.4 * world.time)) * along;
    let step = 6.2831853 / count;
    let a = atan2(offset.y, offset.x) - turn.x - bend;
    let off = a - step * round(a / step);
    // Half its width at this far out, tapering to nothing at the tip.
    let half = 0.5 * arms.z * (1.0 - along);
    let across = abs(off) * max(r, 1.0);
    if across >= half {
        return 0.0;
    }
    return (1.0 - smoothstep(0.6 * half, half, across)) * (1.0 - smoothstep(0.7, 1.0, along));
}

// Whether an image alpha is a surface kept out of the patches (2).
fn kept(alpha: f32) -> bool {
    return abs(alpha - 2.0) < 0.5;
}

// The image at `frag` held and moving in patch `i`'s mosh, `inside` as much as it is in it.
fn held_in(i: u32, frag: vec2<f32>, now: vec4<f32>, inside: f32) -> vec4<f32> {
    let disc = mosh.patches[i];
    let motion = mosh.patch_motion[i];
    let block = floor(frag / disc.w);
    let centre = (block + 0.5) * disc.w;
    let offset = centre - disc.xy;
    let r = length(offset);
    // Outwards, turned by a slowly changing swirl.
    let a = 6.2831853 * (noise(block / 5.0 + world.time * 0.2, motion.w) - 0.5);
    let out = select(vec2<f32>(1.0, 0.0), offset / r, r > 1e-3);
    let swirl = vec2<f32>(cos(a) * out.x - sin(a) * out.y, sin(a) * out.x + cos(a) * out.y);
    let drift = swirl * motion.x * mosh.delta;
    let moved = moved_pixel(vec2<i32>(frag), block, block_motion(centre) + drift, motion.z);
    if kept(moved.a) {
        // Kept out: never smeared into its neighbours.
        return now;
    }
    // The held image fades back into the live one a little every frame, never all at once.
    let back = 1.0 - exp(-motion.y * mosh.delta);
    let shown = mix(now, moved, inside * (1.0 - back));
    // Surface or sky, never read back as kept.
    return vec4<f32>(shown.rgb, min(shown.a, 1.0));
}

// The image at `pixel` between moshes: held and moving inside a patch, and on what carries the
// mosh (alpha 3) wherever it is, block by block round it; live elsewhere.
fn patched(frag: vec2<f32>) -> vec4<f32> {
    let pixel = vec2<i32>(frag);
    let now = textureLoad(live, pixel, 0);
    if kept(now.a) {
        return now;
    }
    for (var i = 0u; i < min(mosh.patch_count, 4u); i++) {
        let disc = mosh.patches[i];
        let motion = mosh.patch_motion[i];
        let block = floor(frag / disc.w);
        let centre = (block + 0.5) * disc.w;
        let offset = centre - disc.xy;
        let r = length(offset);
        // A ragged edge, block by block, that crawls smoothly, and fades out towards it.
        let crawl = vec2<f32>(0.37, 0.23) * world.time;
        let edge = disc.z * (0.6 + 0.6 * noise(block / 3.0 + crawl, motion.w * 17.0));
        let arm = in_arms(i, offset, r, edge);
        if r >= edge && arm <= 0.0 {
            continue;
        }
        return held_in(i, frag, now, max(1.0 - smoothstep(0.7 * edge, edge, r), arm));
    }
    if mosh.patch_count > 0u {
        // Out of every patch, a block whose pixel or centre is on a surface that carries the
        // mosh is in the first patch's.
        let size = vec2<i32>(textureDimensions(live));
        let block = floor(frag / mosh.patches[0].w);
        let centre = vec2<i32>((block + 0.5) * mosh.patches[0].w);
        let marked = textureLoad(live, clamp(centre, vec2<i32>(0), size - 1), 0).a > 2.5;
        if now.a > 2.5 || marked {
            return held_in(0u, frag, now, 1.0);
        }
    }
    return now;
}

// The pixel mosh: the held pixel that the live image's motion at `frag` brought here, the live
// image coming in over it in soft regions, never block by block.
fn by_pixels(frag: vec2<f32>) -> vec4<f32> {
    let pixel = vec2<i32>(frag);
    let size = vec2<i32>(textureDimensions(held));
    let now = textureLoad(live, pixel, 0);
    let limit = 0.25 * view.size.y;
    let motion = clamp(frag - last_position(frag), vec2<f32>(-limit), vec2<f32>(limit));
    // Whole pixels, rounded the same way over the whole image each frame, so slow motion still
    // moves and the image moves as one.
    let jitter = vec2<f32>(
        hash(vec3<f32>(mosh.seed, 61.0 * world.time, 1.0)),
        hash(vec3<f32>(mosh.seed, 47.0 * world.time + 3.0, 2.0)),
    );
    let source = clamp(pixel - vec2<i32>(floor(motion + jitter)), vec2<i32>(0), size - 1);
    let moved = textureLoad(held, source, 0);
    // Residuals, a few pixels across.
    let r = 3;
    var around_now = 0.0;
    var around_was = 0.0;
    for (var i = 0; i < 4; i++) {
        let corner = vec2<i32>(select(-r, r, (i & 1) == 1), select(-r, r, (i & 2) == 2));
        around_now += luma(textureLoad(live, clamp(pixel + corner, vec2<i32>(0), size - 1), 0).rgb);
        around_was += luma(textureLoad(held, clamp(source + corner, vec2<i32>(0), size - 1), 0).rgb);
    }
    let edge = (luma(now.rgb) - 0.25 * around_now) - (luma(moved.rgb) - 0.25 * around_was);
    let bleed = 1.0 - exp(-mosh.bleed * mosh.progress * mosh.delta);
    let color = max(moved.rgb + edge * bleed, vec3<f32>(0.0));
    // Where the live image comes in: soft regions, from a tenth of the way to nearly the end.
    let at = frag / max(mosh.block, 1.0);
    let turn = 0.75 * noise(at, mosh.seed) + 0.25 * noise(4.0 * at, mosh.seed + 0.5);
    let shown = smoothstep(-0.06, 0.06, mosh.progress - (0.1 + 0.8 * turn));
    let alpha = select(moved.a, now.a, shown > 0.5);
    return vec4<f32>(mix(color, now.rgb, shown), alpha);
}

// The smear mosh: macroblocks dragged by the motion at their centres, each rounded its own way
// so they tear apart, a few thrown by corrupt vectors (more and further the faster the image
// moves); the live image's edges bled in, in full colour, faster the faster it moves, so what
// moves leaves trails; blocks snapping back to the live image one by one, in ragged regions.
fn by_smear(frag: vec2<f32>) -> vec4<f32> {
    let pixel = vec2<i32>(frag);
    let size = vec2<i32>(textureDimensions(held));
    let now = textureLoad(live, pixel, 0);
    let cell = max(mosh.block, 4.0);
    let block = floor(frag / cell);
    // Back to the live image: ragged regions of blocks, scattered with single ones, from a
    // tenth of the way to nearly the end.
    let turn = 0.6 * noise(block / 9.0, mosh.seed) + 0.4 * hash(vec3<f32>(block, mosh.seed + 0.5));
    if mosh.progress >= 0.1 + 0.85 * turn {
        return now;
    }
    var vector = block_motion(min((block + 0.5) * cell, view.size - 1.0));
    let speed = length(vector);
    // Corrupt vectors: a few blocks at a time, changing several times a second.
    let tick = floor(world.time * 7.0);
    let chance = 0.015 + 0.25 * smoothstep(0.5, 12.0, speed);
    if hash(vec3<f32>(block, tick + mosh.seed)) < chance {
        let a = 6.2831853 * hash(vec3<f32>(block.yx, tick + 7.0));
        let far = cell * (0.3 + 2.0 * hash(vec3<f32>(block, tick + 13.0))) + 2.0 * speed;
        vector += vec2<f32>(cos(a), sin(a)) * far;
    }
    // Whole pixels, rounded at random per block and frame: neighbours part along their seams.
    let jitter = vec2<f32>(
        hash(vec3<f32>(block, 61.0 * world.time)),
        hash(vec3<f32>(block.yx, 47.0 * world.time + 3.0)),
    );
    let source = clamp(pixel - vec2<i32>(floor(vector + jitter)), vec2<i32>(0), size - 1);
    let moved = textureLoad(held, source, 0);
    // Residuals in full colour, a few pixels across: the live image against its surroundings,
    // laid over the held one's.
    let r = 3;
    var around_now = vec3<f32>(0.0);
    var around_was = vec3<f32>(0.0);
    for (var i = 0; i < 4; i++) {
        let corner = vec2<i32>(select(-r, r, (i & 1) == 1), select(-r, r, (i & 2) == 2));
        around_now += textureLoad(live, clamp(pixel + corner, vec2<i32>(0), size - 1), 0).rgb;
        around_was += textureLoad(held, clamp(source + corner, vec2<i32>(0), size - 1), 0).rgb;
    }
    let residual = (now.rgb - 0.25 * around_now) - (moved.rgb - 0.25 * around_was);
    let bleed = 1.0 - exp(-mosh.bleed * (0.3 + 0.25 * speed) * mosh.delta);
    return vec4<f32>(max(moved.rgb + residual * bleed, vec3<f32>(0.0)), moved.a);
}

@fragment
fn fragment(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    if mosh.progress >= 1.0 {
        return patched(frag.xy);
    }
    if mosh.kind == 1u {
        return by_pixels(frag.xy);
    }
    if mosh.kind == 2u {
        return by_smear(frag.xy);
    }
    let pixel = vec2<i32>(frag.xy);
    let size = vec2<i32>(textureDimensions(held));
    let block = floor(frag.xy / mosh.block);

    // When the block turns live: between a tenth of the way and nearly the end.
    let turn = 0.7 * noise(block / 7.0, mosh.seed) + 0.3 * hash(vec3<f32>(block, mosh.seed + 0.5));
    if mosh.progress >= 0.1 + 0.85 * turn {
        return textureLoad(live, pixel, 0);
    }

    // The block's motion vector, from its centre.
    let centre = min((block + 0.5) * mosh.block, view.size - 1.0);
    let limit = 0.25 * view.size.y;
    let motion = clamp(centre - last_position(centre), vec2<f32>(-limit), vec2<f32>(limit));
    // Whole pixels, rounded up or down at random per block and frame, so slow motion still
    // moves.
    let jitter = vec2<f32>(
        hash(vec3<f32>(block, 61.0 * world.time)),
        hash(vec3<f32>(block.yx, 47.0 * world.time + 3.0)),
    );
    let source = clamp(pixel - vec2<i32>(floor(motion + jitter)), vec2<i32>(0), size - 1);
    let moved = textureLoad(held, source, 0);
    var color = moved.rgb;

    // Residuals: the live image's edges, its light against its surroundings, in the held
    // colours, which keep their own brightness.
    let now = luma(textureLoad(live, pixel, 0).rgb) - local_luma(live, pixel, size);
    let was = luma(color) - local_luma(held, source, size);
    let bleed = 1.0 - exp(-mosh.bleed * mosh.progress * mosh.delta);
    color = max(color + (now - was) * bleed, vec3<f32>(0.0));
    return vec4<f32>(color, moved.a);
}
