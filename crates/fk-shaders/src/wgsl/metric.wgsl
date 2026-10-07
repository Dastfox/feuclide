// The default `fk::metric`: flat space, geodesics are straight. The ray marcher reads it when
// composed with `SAMPLED_METRIC`; a geometry's `SampledMetric::wgsl_module` replaces it.
#define_import_path fk::metric

fn geo_geodesic_accel(x: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(0.0);
}
