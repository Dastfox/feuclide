// The default `fk::sdf`: a white ball of radius 1 at the scene's origin.
//
// A game replaces this module with its own: the same two functions, in a module with this
// import path, reading its values from `fk::march`'s `march.params`. Points are in the scene's
// coordinates, distances in scene units.
#define_import_path fk::sdf

// A lower bound of the distance from `q` to the surface, negative inside it.
fn sdf_distance(q: vec3<f32>) -> f32 {
    return length(q) - 1.0;
}

// The surface's colour at `q`, linear RGB.
fn sdf_color(q: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(1.0);
}

// Optional in a game's module: whether the surface at `q` is kept out of the mosh patches.
// Read only when the ray marcher is composed with `SDF_KEPT`.
fn sdf_kept(q: vec3<f32>) -> bool {
    return false;
}

// Optional in a game's module: whether the surface at `q` carries the mosh patches' mosh
// wherever it is, out of them too (the first patch's). Read only when the ray marcher is
// composed with `SDF_MOSHED`; `sdf_kept` comes first.
fn sdf_moshed(q: vec3<f32>) -> bool {
    return false;
}
