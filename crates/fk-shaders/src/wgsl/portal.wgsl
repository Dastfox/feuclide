// The region of the portals' tree a view draws (`fk_render::PortalViews`): where its surfaces
// are kept and the regions cut out of it, from `fk::view`'s `portal`. Every region is convex,
// its forms linear on embedded points in the eye's frame: the cone's sides through the eye,
// then the plane of the mouth it is seen through, each positive inside. A view with no region
// and no cuts keeps everything, and every test here is then one comparison.
#define_import_path fk::portal

#import fk::view::view
#import fk::geometry::geo_point_from_eye

const PORTAL_FAR: f32 = 3.4e38;

// Whether all of the `count` forms from `first` are not negative at `p`.
fn portal_inside(first: u32, count: u32, p: vec4<f32>) -> bool {
    for (var i = first; i < first + count; i++) {
        if dot(view.portal_forms[i], p) < 0.0 {
            return false;
        }
    }
    return true;
}

// Whether the view keeps a surface at `p` (embedded, in the eye's frame): on the kept side of
// its clip plane (`fk_render::Elsewhere::clip`), in its region, in none of its cuts.
fn portal_keeps(p: vec4<f32>) -> bool {
    if dot(view.clip, p) < 0.0 {
        return false;
    }
    if !portal_inside(0u, u32(view.portal.x), p) {
        return false;
    }
    for (var c = 0u; c < u32(view.portal.y); c++) {
        let cut = view.portal_cuts[c];
        if portal_inside(u32(cut.x), u32(cut.y), p) {
            return false;
        }
    }
    return true;
}

// Whether the direction `dir` at the eye is in the cone of the region whose `count` forms
// start at `first`: its sides (all but the last), which go through the eye, so that their sign
// along a ray is the same all along it.
fn portal_cone_has(first: u32, count: u32, dir: vec3<f32>) -> bool {
    if count == 0u {
        return true;
    }
    let p = geo_point_from_eye(dir, view.near);
    return portal_inside(first, count - 1u, p);
}

// Whether the view sees along `dir` all the way out: the sky there is its own. In its
// region's cone and in no cut's.
fn portal_sees(dir: vec3<f32>) -> bool {
    if !portal_cone_has(0u, u32(view.portal.x), dir) {
        return false;
    }
    for (var c = 0u; c < u32(view.portal.y); c++) {
        let cut = view.portal_cuts[c];
        if portal_cone_has(u32(cut.x), u32(cut.y), dir) {
            return false;
        }
    }
    return true;
}

// The first distance along the ray `dir` from `near` to `far` where `form` is not negative,
// PORTAL_FAR for none: the first sign change among a few samples along the geodesic, then
// halved down. A plane is met once along a ray in flat and hyperbolic space and once within
// half a turn on the sphere, so the samples only find the interval it is met in.
fn portal_entry(form: vec4<f32>, dir: vec3<f32>, near: f32, far: f32) -> f32 {
    if dot(form, geo_point_from_eye(dir, near)) >= 0.0 {
        return near;
    }
    let steps = 8u;
    var a = near;
    for (var k = 1u; k <= steps; k++) {
        let b = near + (far - near) * f32(k) / f32(steps);
        if dot(form, geo_point_from_eye(dir, b)) >= 0.0 {
            var lo = a;
            var hi = b;
            for (var i = 0u; i < 24u; i++) {
                let mid = 0.5 * (lo + hi);
                if dot(form, geo_point_from_eye(dir, mid)) >= 0.0 {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            return hi;
        }
        a = b;
    }
    return PORTAL_FAR;
}

// Where along the ray `dir` the view draws, from `near` to `far`: from where it enters the
// region (its mouth's plane) to where it enters a cut it is in the cone of. Empty (x > y) out
// of the region's cone.
fn portal_span(dir: vec3<f32>, near: f32, far: f32) -> vec2<f32> {
    var span = vec2<f32>(near, far);
    let count = u32(view.portal.x);
    if count > 0u {
        if !portal_cone_has(0u, count, dir) {
            return vec2<f32>(1.0, 0.0);
        }
        span.x = portal_entry(view.portal_forms[count - 1u], dir, near, far);
    }
    for (var c = 0u; c < u32(view.portal.y); c++) {
        let cut = view.portal_cuts[c];
        let n = u32(cut.y);
        if n > 0u && portal_cone_has(u32(cut.x), n, dir) {
            span.y = min(span.y, portal_entry(view.portal_forms[u32(cut.x) + n - 1u], dir, near, far));
        }
    }
    return span;
}
