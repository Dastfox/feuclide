//! Refines a binary glTF for a geometry: splits it until no edge, measured between its embedded
//! ends, is longer than `--max-edge` metres, and writes it as binary glTF (positions, normals,
//! uvs; node transforms and `--scale` baked in).
//!
//! ```text
//! cargo run -p fk-tools --bin refine -- <in.glb> <out.glb> --max-edge <m> [--scale <s>] [--geometry e3]
//! ```

use std::process::ExitCode;

use fk_assets::{gltf, refine};
use fk_geometry_euclidean::E3;
use fk_math::Real;

const USAGE: &str = "usage: refine <in.glb> <out.glb> --max-edge <m> [--scale <s>] [--geometry e3]";

fn main() -> ExitCode {
    match run(std::env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: Vec<String>) -> Result<(), String> {
    let mut paths = Vec::new();
    let (mut max_edge, mut scale, mut geometry) = (None, 1.0, "e3".to_owned());
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        let mut value = |name: &str| args.next().ok_or(format!("{name} needs a value\n{USAGE}"));
        match arg.as_str() {
            "--help" | "-h" => return Err(USAGE.to_owned()),
            "--max-edge" => max_edge = Some(number(&value("--max-edge")?)?),
            "--scale" => scale = number(&value("--scale")?)?,
            "--geometry" => geometry = value("--geometry")?,
            _ => paths.push(arg),
        }
    }
    let ([input, output], Some(max_edge)) =
        (<[String; 2]>::try_from(paths).map_err(|_| USAGE)?, max_edge)
    else {
        return Err(USAGE.to_owned());
    };
    if max_edge <= 0.0 {
        return Err("--max-edge must be positive".to_owned());
    }
    let bytes = std::fs::read(&input).map_err(|error| format!("{input}: {error}"))?;
    let mesh = gltf::load(&bytes, scale).map_err(|error| format!("{input}: {error}"))?;
    let (fine, longest) = match geometry.as_str() {
        "e3" => {
            let fine = refine::<E3>(&mesh, max_edge);
            let longest = fine.longest_edge::<E3>();
            (fine, longest)
        }
        // H³ and S³ join here when their geometries land (E3 of the checklist).
        other => return Err(format!("no geometry {other:?}: e3 is the only one yet")),
    };
    std::fs::write(&output, gltf::write_glb(&fine))
        .map_err(|error| format!("{output}: {error}"))?;
    println!(
        "{input}: {} triangles -> {output}: {} triangles, longest edge {longest:.4} m",
        mesh.triangles(),
        fine.triangles()
    );
    Ok(())
}

fn number(text: &str) -> Result<Real, String> {
    text.parse().map_err(|_| format!("not a number: {text}"))
}
