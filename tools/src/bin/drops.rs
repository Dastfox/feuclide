//! Tells the frame drops the recorder wrote (`FK_DROPS`, `fk_render::drops`) as a Markdown
//! table, newest first: each drop's length against the usual frame, the build, what was most
//! over usual, the events of that frame and where the app was. `--group` counts instead how
//! often each cause and each event came with a drop.
//!
//! ```text
//! cargo run -p fk-tools --bin drops -- [dir] [--group]
//! ```
//!
//! `dir` is `target/drops` by default: the game's are in its own `target/drops`
//! (`../target/drops` from `engine/`).

use std::path::PathBuf;
use std::process::ExitCode;

use fk_tools::drops;

const USAGE: &str = "usage: drops [dir] [--group]";

fn main() -> ExitCode {
    let mut dir = PathBuf::from("target/drops");
    let mut group = false;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--help" | "-h" => {
                println!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            "--group" => group = true,
            _ => dir = PathBuf::from(arg),
        }
    }
    let reports = match drops::read_dir(&dir) {
        Ok(reports) => reports,
        Err(error) => {
            eprintln!("{error}\n{USAGE}");
            return ExitCode::FAILURE;
        }
    };
    if reports.is_empty() {
        println!("no drops in {}", dir.display());
    } else if group {
        print!("{}", drops::grouped(&reports));
    } else {
        print!("{}", drops::table(&reports));
    }
    ExitCode::SUCCESS
}
