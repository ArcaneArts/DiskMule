use std::{env, ffi::OsStr, path::PathBuf};

use directories::ProjectDirs;

use crate::error::{AppError, Result};

pub const HOME_ENV: &str = "DISKMULE_HOME";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Paths {
    pub root: PathBuf,
    pub config: PathBuf,
    pub models: PathBuf,
    pub registry: PathBuf,
}

impl Paths {
    pub fn discover() -> Result<Self> {
        match env::var_os(HOME_ENV) {
            Some(root) => Self::from_override(&root),
            None => {
                let project = ProjectDirs::from("com", "ArcaneArts", "DiskMule")
                    .ok_or(AppError::HomeDirectoryUnavailable)?;
                Ok(Self::from_root(project.data_local_dir().to_path_buf()))
            }
        }
    }

    pub fn from_override(root: &OsStr) -> Result<Self> {
        if root.is_empty() {
            return Err(AppError::InvalidConfiguration(format!(
                "{HOME_ENV} cannot be empty"
            )));
        }
        Ok(Self::from_root(PathBuf::from(root)))
    }

    pub fn from_root(root: PathBuf) -> Self {
        Self {
            config: root.join("config.toml"),
            models: root.join("models"),
            registry: root.join("registry.json"),
            root,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{ffi::OsStr, path::PathBuf};

    use super::Paths;

    #[test]
    fn derives_all_paths_from_an_override() {
        let paths = Paths::from_override(OsStr::new("/tmp/diskmule-test")).unwrap();
        assert_eq!(paths.root, PathBuf::from("/tmp/diskmule-test"));
        assert_eq!(paths.models, paths.root.join("models"));
        assert_eq!(paths.registry, paths.root.join("registry.json"));
        assert_eq!(paths.config, paths.root.join("config.toml"));
    }

    #[test]
    fn rejects_an_empty_override() {
        assert!(Paths::from_override(OsStr::new("")).is_err());
    }
}
