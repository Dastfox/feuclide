use fk_math::Real;

/// A metric known only through its geodesic equation in a chart.
///
/// This is all the ray-marching renderer needs, so it is the extension point for geometries
/// without closed-form geodesics: general Riemannian metrics, Finsler metrics, Lorentzian planes.
/// Closed-form geometries implement it too, so both render paths can be compared.
///
/// Conventions: the geodesic equation in the chart, `ẍᵏ = −Γᵏᵢⱼ(x) vⁱ vʲ` for a Riemannian
/// metric; a Lorentzian one has signature `(−, +, …, +)` with the time-like coordinate first,
/// as [`fk_math::minkowski`]; the WGSL twin works in `f32` on the same chart.
pub trait SampledMetric: Send + Sync + 'static {
    /// Number of chart coordinates.
    const CHART_DIM: usize;

    /// Writes the geodesic acceleration `ẍ` at chart position `x` with velocity `v` into `out`.
    ///
    /// For a Riemannian metric this is `ẍᵏ = -Γᵏᵢⱼ(x) vⁱ vʲ`. All slices have length
    /// [`CHART_DIM`](Self::CHART_DIM).
    fn geodesic_acceleration(&self, x: &[Real], v: &[Real], out: &mut [Real]);

    /// WGSL source defining `fn geo_geodesic_accel(x: vecN<f32>, v: vecN<f32>) -> vecN<f32>`
    /// with `N` = [`CHART_DIM`](Self::CHART_DIM), the GPU twin of
    /// [`geodesic_acceleration`](Self::geodesic_acceleration).
    fn wgsl_module(&self) -> &'static str;
}

/// Integrates the geodesic equation with classical fourth-order Runge–Kutta, in place.
///
/// # Panics
///
/// If `x` or `v` does not have length [`SampledMetric::CHART_DIM`].
pub fn integrate_geodesic<M: SampledMetric>(
    metric: &M,
    x: &mut [Real],
    v: &mut [Real],
    dt: Real,
    steps: usize,
) {
    let n = M::CHART_DIM;
    assert_eq!(x.len(), n, "position has the wrong dimension");
    assert_eq!(v.len(), n, "velocity has the wrong dimension");

    let mut k = [vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]];
    let mut kv = [vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]];
    let mut xs = vec![0.0; n];
    let mut vs = vec![0.0; n];

    for _ in 0..steps {
        k[0].copy_from_slice(v);
        metric.geodesic_acceleration(x, v, &mut kv[0]);
        for stage in 1..4 {
            let h = if stage == 3 { dt } else { 0.5 * dt };
            for i in 0..n {
                xs[i] = x[i] + h * k[stage - 1][i];
                vs[i] = v[i] + h * kv[stage - 1][i];
            }
            k[stage].copy_from_slice(&vs);
            metric.geodesic_acceleration(&xs, &vs, &mut kv[stage]);
        }
        for i in 0..n {
            x[i] += dt / 6.0 * (k[0][i] + 2.0 * k[1][i] + 2.0 * k[2][i] + k[3][i]);
            v[i] += dt / 6.0 * (kv[0][i] + 2.0 * kv[1][i] + 2.0 * kv[2][i] + kv[3][i]);
        }
    }
}
