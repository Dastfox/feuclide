//! The golden images: each scene drawn by the raster pipeline and by the ray marcher agree, and
//! each agrees with its blessed image. Headless; skipped without a GPU adapter (lavapipe in CI).

use fk_render::{Gpu, Paths};
use fk_tools::golden::{Scene, blessed_path, diff, read_png, render};

/// Raster against ray march: the mean ΔE, and the share of pixels visibly different.
const PATHS_MEAN: f64 = 1.0;
const PATHS_VISIBLE: f64 = 0.01;
/// Against the blessed image (another GPU, another driver): tighter.
const BLESSED_MEAN: f64 = 0.5;
const BLESSED_VISIBLE: f64 = 0.005;

#[test]
fn raster_and_ray_march_agree_and_match_their_blessed_images() {
    let Some(gpu) = Gpu::headless() else {
        eprintln!("no adapter, skipped");
        return;
    };
    let mut failures = Vec::new();
    for scene in Scene::ALL {
        let raster = render(&gpu, scene, Paths::Raster);
        let marched = render(&gpu, scene, Paths::RayMarch);
        // Not two empty skies agreeing: the scene covers a good part of the image.
        let sky = raster.pixel(0, 0);
        let covered = (0..raster.height)
            .flat_map(|y| (0..raster.width).map(move |x| (x, y)))
            .filter(|&(x, y)| raster.pixel(x, y) != sky)
            .count();
        assert!(
            covered * 20 > (raster.width * raster.height) as usize,
            "{}: the scene is hardly there",
            scene.name()
        );
        let both = diff(&raster, &marched);
        eprintln!("{}: raster against ray march {both:?}", scene.name());
        if both.mean > PATHS_MEAN || both.visible > PATHS_VISIBLE {
            failures.push(format!(
                "{}: raster and ray march differ, {both:?}",
                scene.name()
            ));
        }
        for (paths, image) in [(Paths::Raster, &raster), (Paths::RayMarch, &marched)] {
            let path = blessed_path(scene, paths);
            match read_png(&path) {
                Ok(blessed) => {
                    let d = diff(image, &blessed);
                    if d.mean > BLESSED_MEAN || d.visible > BLESSED_VISIBLE {
                        failures.push(format!("{}: {d:?}", path.display()));
                    }
                }
                Err(error) => failures.push(format!("no blessed image: {error}")),
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{}\n(cargo run -p fk-tools --bin golden -- --out <dir> writes the images and their \
         differences; --bless blesses them)",
        failures.join("\n")
    );
}
