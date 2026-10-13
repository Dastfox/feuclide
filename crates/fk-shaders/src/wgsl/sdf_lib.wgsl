// Signed distance primitives and helpers for a game's fields (`fk::sdf`), each with its CPU twin
// in `fk_math::sdf` so the two agree.
#define_import_path fk::sdf_lib

// A box's signed distance: half-extents `half` about the origin.
fn cuboid(p: vec3<f32>, half: vec3<f32>) -> f32 {
    let q = abs(p) - half;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

// An upright capped cylinder about the origin's vertical: `radius`, from `bottom` to `top`.
fn column(p: vec3<f32>, radius: f32, bottom: f32, top: f32) -> f32 {
    let across = length(p.xz) - radius;
    let along = abs(p.y - (top + bottom) * 0.5) - abs(top - bottom) * 0.5;
    return length(max(vec2<f32>(across, along), vec2<f32>(0.0))) + min(max(across, along), 0.0);
}

// A cell's hash, the same bits as `fk_math::sdf::cell_hash`.
fn cell_hash(i: i32, j: i32, seed: u32) -> u32 {
    var h = (bitcast<u32>(i) * 0x8da6b343u) ^ (bitcast<u32>(j) * 0xd8163841u) ^ (seed * 0xcb1ab31fu);
    h ^= h >> 13u;
    h *= 0x5bd1e995u;
    return h ^ (h >> 15u);
}

// Which copy `q` is in, the copies repeated every `period` along each axis, one centred on every
// multiple of it.
fn repeat_index(q: vec3<f32>, period: vec3<f32>) -> vec3<f32> {
    return round(q / period);
}

// `q` from the middle of its copy (see `repeat_index`).
fn repeat_local(q: vec3<f32>, period: vec3<f32>) -> vec3<f32> {
    return q - round(q / period) * period;
}
