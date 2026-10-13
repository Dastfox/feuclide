// The post pass's bindings (group 1), shared by its skeleton and the caller's `fk::grade`.
#define_import_path fk::post

// A segmented bar filling from the bottom.
struct Gauge {
    // Bottom-left corner and size, in image heights from the bottom-left corner.
    rect: vec4<f32>,
    // Linear RGB, and the opacity.
    color: vec4<f32>,
    // In [0, 1], shown rounded to whole segments.
    value: f32,
    segments: u32,
}

const MAX_GAUGES: u32 = 4u;

struct Post {
    mode: u32,
    band_spacing: f32,
    // Whether to apply the sRGB transfer function here, for targets that are not sRGB.
    encode_srgb: u32,
    // Blur radius in pixels.
    blur: f32,
    saturation: f32,
    vignette: f32,
    gauge_count: u32,
    gauges: array<Gauge, MAX_GAUGES>,
    // The text panel's top-left corner, in pixels from the image's top-left, and its scale.
    text_layout: vec4<f32>,
    // Its columns and lines, 0 when there is no text.
    text_size: vec4<f32>,
    text_color: vec4<f32>,
    // The panel's colour, and its opacity.
    text_background: vec4<f32>,
}

// The caller's values for its `fk::grade`, opaque to the engine.
struct Grade {
    values: array<vec4<f32>, 32>,
}

@group(1) @binding(0) var scene_color: texture_2d<f32>;
// With multisampling the depth target keeps its samples; sample 0 is read.
#ifdef MULTISAMPLED
@group(1) @binding(1) var scene_depth: texture_depth_multisampled_2d;
#else
@group(1) @binding(1) var scene_depth: texture_depth_2d;
#endif
@group(1) @binding(2) var<uniform> post: Post;
@group(1) @binding(3) var linear: sampler;
// The frame captured last (`FrameCapture`), black until one is.
@group(1) @binding(4) var captured: texture_2d<f32>;
// Only the entities on the layer, over a flat colour (`LayerRender`), while it is drawn.
@group(1) @binding(5) var layer: texture_2d<f32>;
// The font (two words a glyph, a row a byte, the top row lowest, the lowest bit leftmost),
// then the text's characters on a TEXT_COLUMNS-wide grid, four to a word.
@group(1) @binding(6) var<storage, read> text: array<u32>;
@group(1) @binding(7) var<uniform> grade: Grade;
