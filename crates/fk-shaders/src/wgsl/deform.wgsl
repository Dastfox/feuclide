// The default `fk::deform`: every mesh as it was built.
//
// A game replaces this module, for the instances it marks, with its own: one function with
// this signature, in a module with this import path, usually built from `fk::displace`. `iso`
// is the instance's camera-relative isometry and `data` its free per-instance values.
#define_import_path fk::deform

#import fk::displace::{Shaped, unshaped}

fn deform(position: vec4<f32>, normal: vec4<f32>, iso: mat4x4<f32>, data: vec4<f32>) -> Shaped {
    return unshaped(position, normal);
}
