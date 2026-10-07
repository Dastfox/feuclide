//! Draws the golden scenes by the raster pipeline and by the ray marcher, says how they differ
//! from each other and from their blessed images, and writes the images and heat maps of their
//! differences to `--out` (default `target/golden`). With `--bless`, writes the images over the
//! blessed ones in `tools/golden/` instead.
//!
//! ```text
//! cargo run -p fk-tools --bin golden -- [--bless] [--out <dir>]
//! ```

use std::path::PathBuf;
use std::process::ExitCode;

use fk_render::{Gpu, Paths};
use fk_tools::golden::{Scene, blessed_path, diff, diff_image, read_png, render};

fn main() -> ExitCode {
    let (mut bless, mut out) = (false, PathBuf::from("target/golden"));
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--bless" => bless = true,
            "--out" => match args.next() {
                Some(dir) => out = dir.into(),
                None => return usage(),
            },
            _ => return usage(),
        }
    }
    let Some(gpu) = Gpu::headless() else {
        eprintln!("no GPU adapter");
        return ExitCode::FAILURE;
    };
    for scene in Scene::ALL {
        let raster = render(&gpu, scene, Paths::Raster);
        let marched = render(&gpu, scene, Paths::RayMarch);
        println!(
            "{}: raster against ray march {:?}",
            scene.name(),
            diff(&raster, &marched)
        );
        for (paths, image) in [(Paths::Raster, &raster), (Paths::RayMarch, &marched)] {
            let blessed = blessed_path(scene, paths);
            if let Ok(old) = read_png(&blessed) {
                println!(
                    "  {}: against blessed {:?}",
                    blessed.display(),
                    diff(image, &old)
                );
            }
            let path = if bless {
                blessed
            } else {
                out.join(blessed.file_name().expect("a file name"))
            };
            if let Err(error) = image.save(&path) {
                eprintln!("{}: {error}", path.display());
                return ExitCode::FAILURE;
            }
        }
        if !bless {
            let path = out.join(format!("{}-difference.png", scene.name()));
            if let Err(error) = diff_image(&raster, &marched).save(&path) {
                eprintln!("{}: {error}", path.display());
                return ExitCode::FAILURE;
            }
        }
    }
    ExitCode::SUCCESS
}

fn usage() -> ExitCode {
    eprintln!("usage: golden [--bless] [--out <dir>]");
    ExitCode::FAILURE
}
