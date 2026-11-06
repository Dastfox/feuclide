//! `fk-editor [scene.toml] [--tuning <folder>]`: the Feuclide editor (see the library).

use std::path::PathBuf;

const USAGE: &str = "usage: fk-editor [scene.toml] [--tuning <folder>]

  scene.toml   the scene to open and save to (default: scene.toml here)
  --tuning     the folder of TOML tuning files (default: FK_TUNING_DIR, then assets/tuning)";

fn main() -> eframe::Result {
    let mut options = fk_editor::Options::default();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            "--tuning" => match args.next() {
                Some(dir) => options.tuning = Some(PathBuf::from(dir)),
                None => {
                    eprintln!("--tuning needs a folder\n\n{USAGE}");
                    std::process::exit(2);
                }
            },
            scene if !scene.starts_with('-') && options.scene.is_none() => {
                options.scene = Some(PathBuf::from(scene));
            }
            other => {
                eprintln!("unknown argument {other}\n\n{USAGE}");
                std::process::exit(2);
            }
        }
    }
    fk_editor::run(options)
}
