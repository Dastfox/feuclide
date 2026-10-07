// The block of one ray-marched scene, bound in group 1 by the ray-march pipeline. The caller's
// `fk::sdf` reads its own values from `march.params`; the engine gives them no meaning.
#define_import_path fk::march

struct March {
    // The eye's coordinates to the scene's: the inverse of the scene's isometry relative to the
    // eye. The distance field is evaluated in the scene's normal coordinates at its origin.
    to_scene: mat4x4<f32>,
    // x: scene units per eye unit (a zoom: the field is evaluated at `scale · q` and its
    // distances divided by `scale`); y: most steps along a ray; z: over-relaxation of the
    // sphere tracing, 1 for none; w: a ray hits when the field is closer than this many pixels.
    settings: vec4<f32>,
    // x: how much the steps a ray took darken its surface (iteration-count ambient
    // occlusion); y: haze, how much more of the surface fades into the sky, 0 to 1; z: window
    // pixels per pixel drawn (1, or 2 at half resolution); w: the longest step, in eye units,
    // 0 for no limit.
    shading: vec4<f32>,
    // x: how much of the painted light and of the iteration-count darkening the surface takes,
    // 1 all of it, 0 none (drawn in its own colour); y: how much of the fog it takes.
    paint: vec4<f32>,
    // The quotient the scene is in (`fk_render::QuotientView`), in the eye's frame.
    fold: Fold,
    // The caller's.
    params: array<vec4<f32>, 192>,
}

// A quotient a ray-marched scene is folded into: rays crossing a face of its domain are
// carried back across it.
struct Fold {
    // x: how many faces, 0 for none (no quotient).
    count: vec4<f32>,
    // The domain's centre, embedded.
    centre: vec4<f32>,
    // For each face, the centre's translate across it, embedded...
    faces: array<vec4<f32>, 12>,
    // ...and the deck element carrying a point back across it.
    back: array<mat4x4<f32>, 12>,
}

@group(1) @binding(0) var<uniform> march: March;

// Whether the ray being traced is the cone of a block of rays (`fk_render::RayMarched::coarse`),
// set by the pipeline: a field that leaves out what one ray does not come near must not while
// it is, for the cone stands for every ray of its block.
var<private> march_cone: bool = false;
