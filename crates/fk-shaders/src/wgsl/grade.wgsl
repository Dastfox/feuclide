// The engine's `fk::grade`: the post pass's caller hook, here leaving the image as it is.
#define_import_path fk::grade

// Where the image at `uv` is taken from; `uv` itself (exactly) for no resampling.
fn grade_source(uv: vec2<f32>) -> vec2<f32> {
    return uv;
}

// The image after the blur, before the saturation.
fn grade_blurred(color: vec3<f32>, source: vec2<f32>, frag: vec2<f32>) -> vec3<f32> {
    return color;
}

// The image after the saturation, before the vignette. `image` is the pixel as drawn: its
// alpha is 1 on surfaces and 0 on the sky.
fn grade_saturated(color: vec3<f32>, image: vec4<f32>) -> vec3<f32> {
    return color;
}

// The image after the vignette, before the gauges: `r` is the distance from the centre, 1 at
// the corners.
fn grade_over(color: vec3<f32>, uv: vec2<f32>, source: vec2<f32>, r: f32, frag: vec2<f32>) -> vec3<f32> {
    return color;
}
