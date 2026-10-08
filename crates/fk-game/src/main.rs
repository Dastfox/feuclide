//! Feuclide's showcase: a guided tour through every space the engine draws.
//!
//! The camera is on rails. It flies through flat space, then hyperbolic space, then the
//! 3-sphere, and through the closed manifolds made from each: the flat 3-torus, the
//! Seifert–Weber space, the Poincaré homology sphere and a lens space. Each space is a
//! station of about thirty seconds, named in the corner as it opens. The next station is
//! built on a thread while this one plays, then given the window, and the last image of this
//! one datamoshes into the next. After the last station the tour starts again.
//!
//! `cargo run -p fk-game --release [-- <station>]`, the stations numbered from 0. N goes on to
//! the next station, Esc quits. Nothing else steers: the camera is locked to its path.
//!
//! Each station is an app of its own, because an app's geometry is a type (`ScenePlugin<G>`):
//! going from H³ to S³ is a [`Handover`](fk_app::Handover) to a new app, on the same GPU
//! device.

mod station;
mod stations;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                tracing_subscriber::EnvFilter::new("info,wgpu_hal=error,wgpu_core=warn")
            }),
        )
        .init();
    // An optional first argument starts the tour at that station.
    let first = std::env::args()
        .nth(1)
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(0);
    let mut app = stations::build(first);
    app.set_window(fk_app::WindowConfig {
        title: "Feuclide · a tour of the spaces".to_owned(),
        ..Default::default()
    });
    if let Err(error) = app.run() {
        tracing::error!("{error}");
        std::process::exit(1);
    }
}
