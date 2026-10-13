//! Tuning files: constants kept in TOML files rather than in code, compiled in, overridden from
//! disk and reloaded while the app runs (the counterpart of Bevy's asset hot reload, for
//! settings).
//!
//! A [`TuningFile`] is compiled in from its file (`include_str!`), so the app always has a
//! valid one. Set the `FK_TUNING_DIR` environment variable to a directory holding edited
//! copies and they are loaded instead, then reloaded whenever they change on disk while the app
//! runs: edit a colour or a time, save, and see it on the next frame. A copy that does not
//! parse is reported and the last good values stay.
//!
//! Systems read a tuning as an ordinary resource (`Res<MyTuning>`); what is computed from it
//! once at startup keeps its old value until restarted, unless it is built again under
//! [`retuned`].

use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use bevy_ecs::prelude::*;
use fk_math::Real;
use serde::de::DeserializeOwned;

use crate::{App, PreUpdate, Time};

/// The environment variable naming the directory of edited tuning files.
pub const DIR_VARIABLE: &str = "FK_TUNING_DIR";

/// Seconds between two looks at a watched file's modification time.
const CHECK_PERIOD: Real = 0.5;

/// A tuning file: a resource read from TOML.
pub trait TuningFile: Resource + Clone + DeserializeOwned {
    /// Its file name, for example `breath.toml`: in the app's own tuning directory and in
    /// `FK_TUNING_DIR`.
    const FILE: &'static str;
    /// Its shipped contents, `include_str!` of the file.
    const SHIPPED: &'static str;

    /// Reads a tuning file's contents.
    ///
    /// # Errors
    ///
    /// When the text is not valid TOML for `Self`.
    fn parse(text: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(text)
    }

    /// The tuning shipped with the app.
    ///
    /// # Panics
    ///
    /// If the shipped file does not parse; test that it does.
    fn shipped() -> Self {
        Self::parse(Self::SHIPPED)
            .unwrap_or_else(|error| panic!("the shipped {} is invalid: {error}", Self::FILE))
    }

    /// The tuning to run with: the edited copy in `FK_TUNING_DIR` when that is set and the
    /// copy loads, the [shipped](Self::shipped) one otherwise.
    fn load() -> Self {
        edited_path::<Self>().map_or_else(Self::shipped, |path| Self::load_from(&path))
    }

    /// The tuning in the file at `path`, or the [shipped](Self::shipped) one if it does not
    /// load.
    fn load_from(path: &Path) -> Self {
        match read::<Self>(path) {
            Ok(tuning) => {
                tracing::info!("tuning from {}", path.display());
                tuning
            }
            Err(error) => {
                tracing::warn!("{}: {error}; using the shipped tuning", path.display());
                Self::shipped()
            }
        }
    }
}

/// The edited copy of `T` in `FK_TUNING_DIR` when that is set and the copy loads, `shipped`
/// otherwise: for a tuning type an app ships its own values of (a library's type with the
/// app's file).
pub fn load_or<T: TuningFile>(shipped: T) -> T {
    let Some(path) = edited_path::<T>() else {
        return shipped;
    };
    match read::<T>(&path) {
        Ok(tuning) => {
            tracing::info!("tuning from {}", path.display());
            tuning
        }
        Err(error) => {
            tracing::warn!("{}: {error}; using the shipped tuning", path.display());
            shipped
        }
    }
}

/// Where the edited copy of `T` would be, when `FK_TUNING_DIR` is set.
pub fn edited_path<T: TuningFile>() -> Option<PathBuf> {
    std::env::var_os(DIR_VARIABLE).map(|dir| Path::new(&dir).join(T::FILE))
}

fn read<T: TuningFile>(path: &Path) -> Result<T, String> {
    let text = std::fs::read_to_string(path).map_err(|error| error.to_string())?;
    T::parse(&text).map_err(|error| error.to_string())
}

fn modified(path: &Path) -> Option<SystemTime> {
    std::fs::metadata(path).and_then(|m| m.modified()).ok()
}

/// Inserts `tuning` as a resource and, when `FK_TUNING_DIR` is set, reloads it in
/// [`PreUpdate`] whenever its edited copy changes on disk.
pub fn add<T: TuningFile>(app: &mut App, tuning: T) {
    match edited_path::<T>() {
        Some(path) => add_watching(app, tuning, path),
        None => {
            app.insert_resource(tuning);
        }
    }
}

/// Inserts `tuning` as a resource and reloads it in [`PreUpdate`] whenever the file at `path`
/// changes on disk.
pub fn add_watching<T: TuningFile>(app: &mut App, tuning: T, path: PathBuf) {
    app.insert_resource(tuning)
        .insert_resource(Watched::<T> {
            modified: modified(&path),
            path,
            next_check: 0.0,
            _tuning: PhantomData,
        })
        .add_systems(PreUpdate, reload::<T>);
}

/// A run condition: the resource `T` has changed since the condition last ran, its first run
/// aside. What is built in [`Startup`](crate::Startup) from a tuning is built again under it
/// in [`Update`](crate::Update) when that tuning is reloaded, and not a second time at the
/// start.
pub fn retuned<T: Resource>(mut ran: Local<bool>, tuning: Res<T>) -> bool {
    let first = !std::mem::replace(&mut *ran, true);
    tuning.is_changed() && !first
}

#[derive(Resource)]
struct Watched<T> {
    path: PathBuf,
    modified: Option<SystemTime>,
    next_check: Real,
    _tuning: PhantomData<fn() -> T>,
}

fn reload<T: TuningFile>(mut commands: Commands, time: Res<Time>, mut watched: ResMut<Watched<T>>) {
    if time.elapsed() < watched.next_check {
        return;
    }
    watched.next_check = time.elapsed() + CHECK_PERIOD;
    let now = modified(&watched.path);
    if now == watched.modified {
        return;
    }
    watched.modified = now;
    match read::<T>(&watched.path) {
        Ok(new) => {
            tracing::info!("reloaded {}", watched.path.display());
            commands.insert_resource(new);
        }
        Err(error) => tracing::warn!(
            "{}: {error}; keeping the last values",
            watched.path.display()
        ),
    }
}

#[cfg(test)]
mod tests {
    use serde::Deserialize;

    use super::*;

    #[derive(Resource, Clone, Debug, Deserialize, PartialEq)]
    #[serde(deny_unknown_fields)]
    struct Speed {
        speed: Real,
    }

    impl TuningFile for Speed {
        const FILE: &'static str = "speed.toml";
        const SHIPPED: &'static str = "speed = 2.5\n";
    }

    #[test]
    fn a_tuning_is_shipped_and_reloads_from_its_file() {
        assert_eq!(Speed::shipped().speed, 2.5);
        assert!(Speed::parse("nonsense = 1").is_err());
        let dir = std::env::temp_dir().join(format!("fk-tuning-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(Speed::FILE);
        std::fs::write(&path, "speed = 3.0\n").unwrap();
        assert_eq!(Speed::load_from(&path).speed, 3.0);
        let mut app = App::new();
        add_watching(&mut app, Speed::shipped(), path.clone());
        app.update(1.0);
        assert_eq!(
            app.world().resource::<Speed>().speed,
            2.5,
            "until it changes"
        );
        std::fs::write(&path, "speed = 4.0\n").unwrap();
        // The modification time moves on, or the file is seen as it was.
        let later = SystemTime::now() + std::time::Duration::from_secs(5);
        std::fs::File::options()
            .write(true)
            .open(&path)
            .and_then(|file| file.set_modified(later))
            .unwrap();
        app.update(1.0);
        app.update(1.0);
        assert_eq!(app.world().resource::<Speed>().speed, 4.0, "reloaded");
        std::fs::remove_dir_all(&dir).ok();
    }
}
