//! Files an app keeps for its player between runs (its settings, a save), in TOML, usually in
//! its [`config_dir`](crate::config_dir): read at the start, written when they change.

use std::path::PathBuf;

use serde::Serialize;
use serde::de::DeserializeOwned;

/// A TOML file an app keeps, or none (a run that must not touch the player's files: a test, a
/// recording). Failures are logged, never fatal: a file that cannot be read reads as none.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TomlFile(pub Option<PathBuf>);

impl TomlFile {
    /// `file` in the config directory of the application `app_name`.
    pub fn in_config(app_name: &str, file: &str) -> Self {
        Self(crate::config_dir(app_name).map(|dir| dir.join(file)))
    }

    /// Its text, if there is a file and it can be read.
    pub fn read_text(&self) -> Option<String> {
        let path = self.0.as_ref()?;
        std::fs::read_to_string(path)
            .inspect_err(|error| {
                if error.kind() != std::io::ErrorKind::NotFound {
                    tracing::warn!("{}: {error}", path.display());
                }
            })
            .ok()
    }

    /// What it holds, if there is a file and it reads as a `T`.
    pub fn read<T: DeserializeOwned>(&self) -> Option<T> {
        let text = self.read_text()?;
        toml::from_str(&text)
            .inspect_err(|error| {
                if let Some(path) = &self.0 {
                    tracing::warn!("{}: {error}", path.display());
                }
            })
            .ok()
    }

    /// Writes `text` to it, making its directory if need be.
    pub fn write_text(&self, text: &str) {
        let Some(path) = &self.0 else { return };
        let written = path
            .parent()
            .map_or(Ok(()), std::fs::create_dir_all)
            .and_then(|()| std::fs::write(path, text));
        if let Err(error) = written {
            tracing::warn!("could not write {}: {error}", path.display());
        }
    }

    /// Writes `value` to it as TOML.
    pub fn write<T: Serialize>(&self, value: &T) {
        match toml::to_string(value) {
            Ok(text) => self.write_text(&text),
            Err(error) => tracing::warn!("could not write {:?}: {error}", self.0),
        }
    }

    /// Removes it, if it is there.
    pub fn forget(&self) {
        if let Some(path) = &self.0
            && let Err(error) = std::fs::remove_file(path)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            tracing::warn!("could not forget {}: {error}", path.display());
        }
    }
}

#[cfg(test)]
mod tests {
    use serde::Deserialize;

    use super::*;

    #[derive(Debug, Default, PartialEq, Serialize, Deserialize)]
    struct Kept {
        volume: f64,
    }

    #[test]
    fn a_value_is_written_read_back_and_forgotten() {
        let dir = std::env::temp_dir().join(format!("fk-persist-{}", std::process::id()));
        let file = TomlFile(Some(dir.join("kept.toml")));
        assert_eq!(file.read::<Kept>(), None, "nothing yet");
        file.write(&Kept { volume: 0.5 });
        assert_eq!(file.read::<Kept>(), Some(Kept { volume: 0.5 }));
        file.forget();
        assert_eq!(file.read::<Kept>(), None, "forgotten");
        assert_eq!(TomlFile(None).read::<Kept>(), None);
        std::fs::remove_dir_all(&dir).ok();
    }
}
