//! Golden images: the same scenes drawn by the raster pipeline (meshes) and by the ray marcher
//! (distance fields), compared with each other and with the blessed images in `tools/golden/`.
//!
//! The ray marcher is the mathematical ground truth for the raster path: if the two agree on a
//! ball, a turned box and a little still life, the projection, the depth, the normals and the
//! painted light agree. The fourth scene is in a quotient (`fk_render::QuotientView`): the
//! raster path draws its copies at the deck translates, the ray marcher folds its rays across
//! the faces, and a box straddling a face must come out the same both ways. Images are compared perceptually ([`diff`]): CIE L\*a\*b\* differences
//! (ΔE76), each pixel against the best match within one pixel of it in the other image (so a
//! silhouette a pixel off is not a difference), both ways.
//!
//! `cargo test -p fk-tools --test golden` checks them (skipped without a GPU adapter; lavapipe
//! in CI); `cargo run -p fk-tools --bin golden -- --bless` draws them again into `tools/golden/`
//! after a change that is meant to alter them.

use std::path::{Path, PathBuf};

use bevy_ecs::world::World;
use fk_geometry::{Geometry, GroupElement};
use fk_geometry_euclidean::{E3, Se3};
use fk_math::Real;
use fk_math::nalgebra::{Point3, UnitQuaternion, Vector3};
use fk_quotient::Quotient;
use fk_render::{
    Color, Gpu, Image, Lighting, MeshBuilder, MeshInstances, Meshes, Offscreen, Paths,
    QuotientView, RayMarched, RayMarching, SdfShader, Sdfs, shapes,
};
use fk_scene::{Camera, GlobalPose};

/// Pixels across and down.
pub const SIZE: (u32, u32) = (160, 120);

/// The scenes, each drawn both ways.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scene {
    /// One ball ahead.
    Ball,
    /// A box turned two ways.
    Box,
    /// A slab, a ball and a pillar on it, in three colours.
    StillLife,
    /// In space that closes up along z every 10 m, a box straddling a face of the domain and a
    /// ball, seen from the other side of the domain looking across the face: their copies, and
    /// the copies of those down the line.
    Straddle,
}

impl Scene {
    /// Every scene.
    pub const ALL: [Self; 4] = [Self::Ball, Self::Box, Self::StillLife, Self::Straddle];

    /// Its name in file names.
    pub fn name(self) -> &'static str {
        match self {
            Self::Ball => "ball",
            Self::Box => "box",
            Self::StillLife => "still-life",
            Self::Straddle => "straddle",
        }
    }
}

/// The geometry's name in file names.
pub const GEOMETRY: &str = "e3";

/// One shape of a scene: its mesh, where it stands and its colour; the field has the same.
struct Shape {
    mesh: MeshBuilder,
    pose: Se3,
    color: Color,
}

fn at(x: Real, y: Real, z: Real) -> Se3 {
    Se3::exp(&E3::transvection(&Vector3::new(x, y, z)))
}

fn turned(pose: Se3, yaw: Real, pitch: Real) -> Se3 {
    pose.compose(&Se3 {
        rotation: UnitQuaternion::from_euler_angles(pitch, yaw, 0.0),
        translation: Vector3::zeros(),
    })
}

/// The shapes of `scene`, and its field: one `fk::sdf` module drawing all of them, in the
/// frame of the scene's root.
fn shapes_of(scene: Scene) -> (Vec<Shape>, &'static str) {
    let white = Color::hex(0xeeeae0);
    match scene {
        Scene::Straddle => (
            vec![
                Shape {
                    mesh: shapes::cuboid(Vector3::new(1.0, 1.0, 1.0)),
                    pose: at(0.6, -0.3, -4.6),
                    color: Color::hex(0xc8643c),
                },
                Shape {
                    mesh: shapes::sphere(0.7, 5),
                    pose: at(-1.6, 0.4, -2.0),
                    color: white,
                },
            ],
            r"#define_import_path fk::sdf
fn cube(q: vec3<f32>) -> f32 {
    let d = abs(q - vec3<f32>(0.6, -0.3, -4.6)) - vec3<f32>(1.0);
    return length(max(d, vec3<f32>(0.0))) + min(max(d.x, max(d.y, d.z)), 0.0);
}
fn ball(q: vec3<f32>) -> f32 {
    return length(q - vec3<f32>(-1.6, 0.4, -2.0)) - 0.7;
}
fn sdf_distance(q: vec3<f32>) -> f32 {
    return min(cube(q), ball(q));
}
fn sdf_color(q: vec3<f32>) -> vec3<f32> {
    if cube(q) <= ball(q) {
        return vec3<f32>(0.5776, 0.1248, 0.0452);
    }
    return vec3<f32>(0.8550, 0.8228, 0.7454);
}
",
        ),
        Scene::Ball => (
            vec![Shape {
                mesh: shapes::sphere(1.0, 5),
                pose: at(0.0, 0.0, -4.0),
                color: white,
            }],
            r"#define_import_path fk::sdf
fn sdf_distance(q: vec3<f32>) -> f32 {
    return length(q - vec3<f32>(0.0, 0.0, -4.0)) - 1.0;
}
fn sdf_color(q: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(0.8550, 0.8228, 0.7454);
}
",
        ),
        Scene::Box => (
            vec![Shape {
                mesh: shapes::cuboid(Vector3::new(0.8, 0.6, 0.7)),
                pose: turned(at(0.3, -0.2, -4.5), 0.6, 0.3),
                color: Color::hex(0xc8643c),
            }],
            r"#define_import_path fk::sdf
#import fk::march::march
fn sdf_distance(q: vec3<f32>) -> f32 {
    // Into the box's frame: march.params[0..3] are the rows of its turn, [3] its place.
    let r = q - march.params[3].xyz;
    let p = vec3<f32>(dot(march.params[0].xyz, r), dot(march.params[1].xyz, r), dot(march.params[2].xyz, r));
    let d = abs(p) - vec3<f32>(0.8, 0.6, 0.7);
    return length(max(d, vec3<f32>(0.0))) + min(max(d.x, max(d.y, d.z)), 0.0);
}
fn sdf_color(q: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(0.5776, 0.1248, 0.0452);
}
",
        ),
        Scene::StillLife => (
            vec![
                Shape {
                    mesh: shapes::cuboid(Vector3::new(2.5, 0.15, 1.5)),
                    pose: at(0.0, -1.2, -5.0),
                    color: Color::hex(0x5a7d4b),
                },
                Shape {
                    mesh: shapes::sphere(0.6, 5),
                    pose: at(-0.9, -0.45, -5.2),
                    color: white,
                },
                Shape {
                    mesh: shapes::cylinder(0.35, 1.6, 96),
                    pose: at(1.0, -1.05, -4.8),
                    color: Color::hex(0x3c5a96),
                },
            ],
            r"#define_import_path fk::sdf
fn slab(q: vec3<f32>) -> f32 {
    let d = abs(q - vec3<f32>(0.0, -1.2, -5.0)) - vec3<f32>(2.5, 0.15, 1.5);
    return length(max(d, vec3<f32>(0.0))) + min(max(d.x, max(d.y, d.z)), 0.0);
}
fn ball(q: vec3<f32>) -> f32 {
    return length(q - vec3<f32>(-0.9, -0.45, -5.2)) - 0.6;
}
fn pillar(q: vec3<f32>) -> f32 {
    let p = q - vec3<f32>(1.0, -1.05, -4.8);
    let d = vec2<f32>(length(p.xz) - 0.35, abs(p.y - 0.8) - 0.8);
    return length(max(d, vec2<f32>(0.0))) + min(max(d.x, d.y), 0.0);
}
fn sdf_distance(q: vec3<f32>) -> f32 {
    return min(slab(q), min(ball(q), pillar(q)));
}
fn sdf_color(q: vec3<f32>) -> vec3<f32> {
    let s = slab(q);
    let b = ball(q);
    let p = pillar(q);
    if s <= b && s <= p {
        return vec3<f32>(0.1022, 0.2051, 0.0704);
    }
    if b <= p {
        return vec3<f32>(0.8550, 0.8228, 0.7454);
    }
    return vec3<f32>(0.0452, 0.1022, 0.3049);
}
",
        ),
    }
}

/// The light every scene is drawn in: the painted light, with many bands so their edges (where
/// the two paths' normals may differ by a hair) do not dominate the comparison.
pub fn lighting() -> Lighting {
    Lighting {
        sun: Vector3::new(0.5, 0.8, 0.45),
        bands: 32,
        ..Lighting::default()
    }
}

/// How far the camera sees.
const FAR: Real = 100.0;

/// The world of `scene` drawn by `paths`: its meshes for the raster path, its field for the
/// ray marcher (one entity at the root, its values the box's frame), and its quotient.
fn world(scene: Scene, paths: Paths) -> World {
    let (shapes, field) = shapes_of(scene);
    let mut world = World::new();
    if scene == Scene::Straddle {
        let line = Quotient::<E3>::from_translations(Point3::origin(), &[Vector3::z() * 10.0]);
        world.insert_resource(QuotientView::new(line, FAR));
    }
    world.insert_resource(RayMarching {
        steps: 400,
        relaxation: 1.0,
        hit_pixels: 0.1,
        occlusion: 0.0,
        ..RayMarching::default()
    });
    if paths == Paths::RayMarch {
        let mut sdfs = Sdfs::default();
        let sdf = sdfs.add(SdfShader {
            name: format!("golden/{}.wgsl", scene.name()),
            source: field.to_owned(),
            libraries: Vec::new(),
        });
        world.insert_resource(sdfs);
        let mut marched = RayMarched::new(sdf);
        if let Some(shape) = shapes.first() {
            // The first shape's frame, for fields that turn with it: the rows of the turn into
            // its frame (the columns of its rotation) and its place.
            let rotation = shape.pose.rotation.to_rotation_matrix();
            for (i, column) in rotation.matrix().column_iter().enumerate() {
                marched.params[i] = [column[0] as f32, column[1] as f32, column[2] as f32, 0.0];
            }
            let t = shape.pose.translation;
            marched.params[3] = [t.x as f32, t.y as f32, t.z as f32, 0.0];
        }
        world.spawn((GlobalPose::<E3>::new(Se3::identity()), marched));
    } else {
        let mut meshes = Meshes::default();
        for shape in shapes {
            let mesh = meshes.add(shape.mesh.build::<E3>());
            world.spawn((
                GlobalPose::<E3>::new(shape.pose),
                MeshInstances::<E3>::single(mesh, shape.color),
            ));
        }
        world.insert_resource(meshes);
    }
    world
}

/// `scene` drawn by `paths` (raster or ray march) on `gpu`.
pub fn render(gpu: &Gpu, scene: Scene, paths: Paths) -> Image {
    let mut offscreen = Offscreen::<E3>::new(gpu, SIZE.0, SIZE.1);
    let mut world = world(scene, paths);
    // Across the domain from the straddling box, turned to look at the face it straddles.
    let eye = match scene {
        Scene::Straddle => turned(at(3.0, 2.4, -3.2), std::f64::consts::PI, -0.22),
        _ => Se3::identity(),
    };
    offscreen.draw(
        &mut world,
        &eye,
        &Camera::new(1.0, 0.1, FAR),
        &lighting(),
        paths,
    )
}

/// The blessed image of `scene` drawn by `paths`.
pub fn blessed_path(scene: Scene, paths: Paths) -> PathBuf {
    let path = match paths {
        Paths::Raster => "raster",
        Paths::RayMarch => "ray-march",
        Paths::Both => "both",
    };
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("golden")
        .join(format!("{}-{GEOMETRY}-{path}.png", scene.name()))
}

/// Reads a PNG written by [`Image::save`].
///
/// # Errors
///
/// If it cannot be read, or is not 8-bit RGBA.
pub fn read_png(path: &Path) -> Result<Image, String> {
    let file = std::fs::File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().map_err(|error| error.to_string())?;
    let mut rgba = vec![0; reader.output_buffer_size().ok_or("too large")?];
    let info = reader
        .next_frame(&mut rgba)
        .map_err(|error| error.to_string())?;
    if info.color_type != png::ColorType::Rgba || info.bit_depth != png::BitDepth::Eight {
        return Err(format!("{}: not 8-bit RGBA", path.display()));
    }
    rgba.truncate(info.buffer_size());
    Ok(Image {
        width: info.width,
        height: info.height,
        rgba,
    })
}

/// How two images differ, perceptually.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Diff {
    /// The mean ΔE76 over the pixels, each against its best match within one pixel.
    pub mean: Real,
    /// The share of the pixels whose ΔE is over [`VISIBLE`]: a difference anyone would see.
    pub visible: Real,
    /// The largest ΔE.
    pub worst: Real,
}

/// A ΔE76 anyone sees at a glance (2.3 is the least noticeable).
pub const VISIBLE: Real = 10.0;

/// `a` against `b`: see [`Diff`]. Each pixel's difference is the least ΔE between it and the
/// pixels within one of it in the other image, the larger of the two ways round.
///
/// # Panics
///
/// If they are not the same size.
pub fn diff(a: &Image, b: &Image) -> Diff {
    assert_eq!(
        (a.width, a.height),
        (b.width, b.height),
        "images of different sizes"
    );
    let (la, lb) = (lab_image(a), lab_image(b));
    let (w, h) = (a.width as i64, a.height as i64);
    let nearest = |from: &[[Real; 3]], to: &[[Real; 3]], x: i64, y: i64| {
        let here = from[(y * w + x) as usize];
        let mut best = Real::INFINITY;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (u, v) = (x + dx, y + dy);
                if u < 0 || v < 0 || u >= w || v >= h {
                    continue;
                }
                let there = to[(v * w + u) as usize];
                let d = ((here[0] - there[0]).powi(2)
                    + (here[1] - there[1]).powi(2)
                    + (here[2] - there[2]).powi(2))
                .sqrt();
                best = best.min(d);
            }
        }
        best
    };
    let (mut sum, mut visible, mut worst) = (0.0, 0usize, 0.0 as Real);
    for y in 0..h {
        for x in 0..w {
            let d = nearest(&la, &lb, x, y).max(nearest(&lb, &la, x, y));
            sum += d;
            worst = worst.max(d);
            if d > VISIBLE {
                visible += 1;
            }
        }
    }
    let n = (w * h) as Real;
    Diff {
        mean: sum / n,
        visible: visible as Real / n,
        worst,
    }
}

/// A heat map of where `a` and `b` differ, for looking: black the same, red over [`VISIBLE`].
pub fn diff_image(a: &Image, b: &Image) -> Image {
    let (la, lb) = (lab_image(a), lab_image(b));
    let rgba = la
        .iter()
        .zip(&lb)
        .flat_map(|(p, q)| {
            let d = ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt();
            let t = (d / VISIBLE).min(1.0);
            [
                (255.0 * t) as u8,
                (60.0 * (1.0 - t) * t * 4.0) as u8,
                0,
                255,
            ]
        })
        .collect();
    Image {
        width: a.width,
        height: a.height,
        rgba,
    }
}

fn lab_image(image: &Image) -> Vec<[Real; 3]> {
    image
        .rgba
        .as_chunks::<4>()
        .0
        .iter()
        .map(|[r, g, b, _]| lab([*r, *g, *b]))
        .collect()
}

/// CIE L\*a\*b\* of an sRGB colour, D65 white.
pub fn lab(srgb: [u8; 3]) -> [Real; 3] {
    let linear = srgb.map(|c| {
        let c = Real::from(c) / 255.0;
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    });
    let [r, g, b] = linear;
    let x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047;
    let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883;
    let f = |t: Real| {
        if t > 216.0 / 24389.0 {
            t.cbrt()
        } else {
            (24389.0 / 27.0 * t + 16.0) / 116.0
        }
    };
    let (fx, fy, fz) = (f(x), f(y), f(z));
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(width: u32, height: u32, rgb: [u8; 3]) -> Image {
        Image {
            width,
            height,
            rgba: (0..width * height)
                .flat_map(|_| [rgb[0], rgb[1], rgb[2], 255])
                .collect(),
        }
    }

    #[test]
    fn white_and_black_are_a_hundred_apart() {
        let [l, a, b] = lab([255, 255, 255]);
        assert!(
            (l - 100.0).abs() < 0.05 && a.abs() < 0.05 && b.abs() < 0.05,
            "{l} {a} {b}"
        );
        assert!(lab([0, 0, 0])[0].abs() < 1e-9);
    }

    #[test]
    fn the_same_image_does_not_differ_and_a_shift_of_a_pixel_hardly() {
        let mut a = flat(20, 10, [200, 30, 30]);
        assert_eq!(diff(&a, &a).mean, 0.0);
        // A dark column at x = 10 in one, at 11 in the other.
        let mut b = a.clone();
        for y in 0..10 {
            a.rgba[((y * 20 + 10) * 4) as usize..][..3].copy_from_slice(&[0, 0, 0]);
            b.rgba[((y * 20 + 11) * 4) as usize..][..3].copy_from_slice(&[0, 0, 0]);
        }
        assert_eq!(diff(&a, &b).visible, 0.0, "a pixel off is the same");
        let c = flat(20, 10, [30, 30, 200]);
        assert_eq!(diff(&a, &c).visible, 1.0, "red and blue are not");
    }
}
