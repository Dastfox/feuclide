// The sky: a gradient from the horizon to the zenith, a sun and a moon with their glows, and
// stars. Directions are unit vectors in the eye's frame (x right, y up, −z ahead), which mean
// the same in every geometry.
#define_import_path fk::sky

#import fk::view::{view, lighting, Disc}

fn sky_gradient(dir: vec3<f32>) -> vec3<f32> {
    let h = dot(dir, lighting.sky_up.xyz);
    if h >= 0.0 {
        let t = pow(h, max(lighting.sky_up.w, 1e-3));
        return mix(lighting.sky_horizon.rgb, lighting.sky_zenith.rgb, t);
    }
    return mix(lighting.sky_horizon.rgb, lighting.sky_below.rgb, smoothstep(0.0, 0.2, -h));
}

fn disc_angle(disc: Disc, dir: vec3<f32>) -> f32 {
    return acos(clamp(dot(dir, disc.direction.xyz), -1.0, 1.0));
}

fn disc_glow(disc: Disc, dir: vec3<f32>) -> vec3<f32> {
    return disc.glow.rgb * exp(-disc_angle(disc, dir) / max(disc.glow.w, 1e-4));
}

// Angular size of one pixel at the centre of the view.
fn pixel_angle() -> f32 {
    return 2.0 / (view.proj_scale.y * view.size.y);
}

const TAU: f32 = 6.28318530718;

// `dir` in the plane of the sky around the disc, in radians: x along the horizon, y up.
fn disc_plane(disc: Disc, dir: vec3<f32>) -> vec2<f32> {
    let centre = disc.direction.xyz;
    var up = lighting.sky_up.xyz - centre * dot(lighting.sky_up.xyz, centre);
    if dot(up, up) < 1e-8 {
        // Straight overhead: any up will do.
        up = vec3<f32>(0.0, 0.0, -1.0) - centre * dot(vec3<f32>(0.0, 0.0, -1.0), centre);
    }
    up = normalize(up);
    let right = cross(up, centre);
    let off = dir - centre * dot(dir, centre);
    return vec2<f32>(dot(off, right), dot(off, up));
}

// Distance, in radians, from `p` to the lines of a star whose `points` points lie on a circle
// of radius `radius`, the first one up: concave arcs join neighbouring points. Each arc is
// centred where the circle's tangents at its two points meet, so it leaves both points along
// the radius and neighbouring arcs meet in a cusp (four points: arcs of radius `radius`
// centred on the corners of the square around the circle).
fn star_lines(p: vec2<f32>, radius: f32, points: f32) -> f32 {
    let r = length(p);
    let n = max(points, 2.0);
    let half = 0.5 * TAU / n;
    // Fold into one gap between two points, measured from the bisector of that gap.
    let angle = atan2(p.x, p.y);
    let theta = angle - (floor(angle / (2.0 * half)) * 2.0 + 1.0) * half;
    let q = vec2<f32>(r * cos(theta), r * sin(theta));
    if r <= radius {
        let centre = vec2<f32>(radius / cos(half), 0.0);
        return abs(length(q - centre) - radius * tan(half));
    }
    // Past the circle: round caps on the points.
    let tip = vec2<f32>(radius * cos(half), radius * sin(half) * sign(theta));
    return length(q - tip);
}

// How much of the pixel looking along `dir` the disc covers, with a one-pixel soft edge.
// Strokes thinner than a pixel are drawn a pixel wide and fainter.
fn disc_cover(disc: Disc, dir: vec3<f32>) -> f32 {
    let radius = disc.direction.w;
    if radius <= 0.0 || dot(dir, disc.direction.xyz) <= 0.0 {
        return 0.0;
    }
    let px = pixel_angle();
    let stroke = max(disc.shape.z, px);
    let faint = min(disc.shape.z / px, 1.0);
    let p = disc_plane(disc, dir);
    var d: f32;
    var strength = 1.0;
    switch u32(disc.shape.x) {
        case 1u: {
            d = abs(length(p) - radius) - 0.5 * stroke;
            strength = faint;
        }
        case 2u: {
            d = star_lines(p, radius, disc.shape.y) - 0.5 * stroke;
            strength = faint;
        }
        default: {
            d = length(p) - radius;
        }
    }
    return strength * (1.0 - smoothstep(-0.5 * px, 0.5 * px, d));
}

// What a far surface fades into when seen along `dir`: the gradient and the glows.
fn sky_haze(dir: vec3<f32>) -> vec3<f32> {
    return sky_gradient(dir) + disc_glow(lighting.sun_disc, dir)
        + disc_glow(lighting.moon_disc, dir);
}

fn hash3(p: vec3<f32>) -> vec3<f32> {
    var q = fract(p * vec3<f32>(0.1031, 0.1030, 0.0973));
    q += dot(q, q.yxz + 33.33);
    return fract((q.xxy + q.yxx) * q.zyx);
}

// Brightness of the stars along `dir`: one star in a few cells of a grid fixed to the turning
// sky, each at a random place in its cell, never smaller than a pixel.
fn stars_along(dir: vec3<f32>) -> f32 {
    if lighting.stars <= 0.0 {
        return 0.0;
    }
    let frame = lighting.star_frame;
    let density = lighting.star_density;
    let p = vec3<f32>(dot(dir, frame[0].xyz), dot(dir, frame[1].xyz), dot(dir, frame[2].xyz))
        * density;
    let cell = floor(p);
    let h = hash3(cell);
    let centre = cell + 0.5 + (h - 0.5) * 0.7;
    let size = max(0.04 + 0.08 * h.y, 0.7 * pixel_angle() * density);
    let d = length(p - centre);
    let present = step(0.85, h.x);
    return present * (0.25 + 0.75 * h.z * h.z) * exp(-d * d / (size * size));
}

// The sky seen along `dir`.
fn sky_color(dir: vec3<f32>) -> vec3<f32> {
    var color = sky_haze(dir);
    let above = smoothstep(-0.01, 0.02, dot(dir, lighting.sky_up.xyz));
    color += vec3<f32>(stars_along(dir) * lighting.stars * above);
    color = mix(color, lighting.moon_disc.color.rgb, disc_cover(lighting.moon_disc, dir) * above);
    color = mix(color, lighting.sun_disc.color.rgb, disc_cover(lighting.sun_disc, dir) * above);
    return color;
}
