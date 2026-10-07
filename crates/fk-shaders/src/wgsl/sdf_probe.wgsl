// Evaluates the caller's `fk::sdf` at given points, for tests that hold it against a CPU twin.

#import fk::sdf::sdf_distance
#import fk::march::march

// Scene coordinates; w is ignored.
@group(2) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(2) @binding(1) var<storage, read_write> distances: array<f32>;

@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < arrayLength(&points) {
        distances[id.x] = sdf_distance(points[id.x].xyz);
    }
}
