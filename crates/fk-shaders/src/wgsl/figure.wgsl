// The engine's `fk::figure`: the lines of a sky body drawn as a figure, here a circle.
#define_import_path fk::figure

// Distance from `p` (radians in the plane of the sky round the body, y up) to the figure's
// lines, for a body of angular radius `radius` and the figure's own `params`.
fn figure_lines(p: vec2<f32>, radius: f32, params: vec4<f32>) -> f32 {
    return abs(length(p) - radius);
}
