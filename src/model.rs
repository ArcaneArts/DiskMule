use std::{
    collections::HashSet,
    fmt,
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::{Component, Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use crate::{config::Paths, gguf::GgufFile};

const REGISTRY_VERSION: u32 = 1;
const MAX_REGISTRY_BYTES: u64 = 16 * 1024 * 1024;
const MAX_OLLAMA_MANIFEST_BYTES: u64 = 4 * 1024 * 1024;
const OLLAMA_MODEL_MEDIA_TYPE: &str = "application/vnd.ollama.image.model";

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("could not read {path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("could not write {path}: {source}")]
    Write {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("{path} is too large to be a model registry or manifest ({size} bytes)")]
    MetadataFileTooLarge { path: PathBuf, size: u64 },

    #[error("invalid DiskMule registry {path}: {message}")]
    InvalidRegistry { path: PathBuf, message: String },

    #[error("unsupported DiskMule registry version {0}")]
    UnsupportedRegistryVersion(u32),

    #[error("model {0:?} was not found")]
    NotFound(String),

    #[error("model name {name:?} is ambiguous across {matches} catalog entries")]
    Ambiguous { name: String, matches: usize },

    #[error("model {name:?} is owned by {owner} and cannot be removed by DiskMule")]
    ExternalOwnership { name: String, owner: ModelSource },

    #[error("model {0:?} is currently loaded and cannot be removed")]
    Loaded(String),

    #[error("model {name:?} cannot be run: {status}")]
    NotRunnable { name: String, status: String },

    #[error("model {name:?} has an unsafe managed path: {reason}")]
    UnsafeManagedPath { name: String, reason: String },

    #[error("model {name:?} shares its file with another registry entry")]
    SharedManagedPath { name: String },

    #[error("could not inspect GGUF {path}: {message}")]
    InvalidGguf { path: PathBuf, message: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSource {
    DiskMule,
    Ollama,
    LocalFile,
}

impl fmt::Display for ModelSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::DiskMule => "diskmule",
            Self::Ollama => "ollama (read-only)",
            Self::LocalFile => "local file (read-only)",
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Compatibility {
    MetadataCompatible,
    Unsupported(String),
    Invalid(String),
}

impl Compatibility {
    pub fn is_metadata_compatible(&self) -> bool {
        matches!(self, Self::MetadataCompatible)
    }
}

impl fmt::Display for Compatibility {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MetadataCompatible => {
                formatter.write_str("metadata-compatible; inference pending")
            }
            Self::Unsupported(reason) => write!(formatter, "unsupported: {reason}"),
            Self::Invalid(reason) => write!(formatter, "invalid: {reason}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelRecord {
    pub name: String,
    pub path: Option<PathBuf>,
    pub architecture: Option<String>,
    pub quantization: Option<String>,
    pub size: Option<u64>,
    pub source: ModelSource,
    pub compatibility: Compatibility,
    pub gguf: Option<GgufSummary>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufSummary {
    pub version: u32,
    pub alignment: u32,
    pub data_offset: u64,
    pub tensor_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemovedModel {
    pub name: String,
    pub path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManagedRegistry {
    version: u32,
    #[serde(default)]
    models: Vec<ManagedEntry>,
}

impl Default for ManagedRegistry {
    fn default() -> Self {
        Self {
            version: REGISTRY_VERSION,
            models: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManagedEntry {
    name: String,
    path: PathBuf,
}

#[derive(Debug, Deserialize)]
struct OllamaManifest {
    #[serde(default)]
    layers: Vec<OllamaLayer>,
}

#[derive(Debug, Deserialize)]
struct OllamaLayer {
    #[serde(rename = "mediaType")]
    media_type: String,
    digest: String,
    size: Option<u64>,
}

#[derive(Debug)]
pub struct ModelCatalog {
    managed_root: PathBuf,
    registry_path: PathBuf,
    managed: ManagedRegistry,
    records: Vec<ModelRecord>,
}

impl ModelCatalog {
    pub fn discover(paths: &Paths, ollama_root: Option<PathBuf>) -> Result<Self, ModelError> {
        let managed = load_managed_registry(&paths.registry)?;
        let mut records = managed
            .models
            .iter()
            .map(|entry| inspect_managed_entry(&paths.models, entry))
            .collect::<Vec<_>>();

        if let Some(root) = ollama_root
            && root.is_dir()
        {
            records.extend(discover_ollama(&root));
        }

        records.sort_by(|left, right| {
            left.name
                .cmp(&right.name)
                .then_with(|| source_rank(left.source).cmp(&source_rank(right.source)))
        });

        Ok(Self {
            managed_root: paths.models.clone(),
            registry_path: paths.registry.clone(),
            managed,
            records,
        })
    }

    pub fn records(&self) -> &[ModelRecord] {
        &self.records
    }

    pub fn resolve(&self, name: &str) -> Result<&ModelRecord, ModelError> {
        let matches = self
            .records
            .iter()
            .filter(|record| record.name == name)
            .collect::<Vec<_>>();
        match matches.as_slice() {
            [] => Err(ModelError::NotFound(name.to_owned())),
            [record] => Ok(record),
            _ => Err(ModelError::Ambiguous {
                name: name.to_owned(),
                matches: matches.len(),
            }),
        }
    }

    pub fn resolve_for_run(&self, input: &str) -> Result<ModelRecord, ModelError> {
        match self.resolve(input) {
            Ok(record) => Ok(record.clone()),
            Err(ModelError::NotFound(_)) => {
                let path = PathBuf::from(input);
                if !path.is_file() {
                    return Err(ModelError::NotFound(input.to_owned()));
                }
                let canonical = fs::canonicalize(&path).map_err(|source| ModelError::Read {
                    path: path.clone(),
                    source,
                })?;
                Ok(inspect_model(
                    input.to_owned(),
                    canonical,
                    ModelSource::LocalFile,
                    None,
                ))
            }
            Err(error) => Err(error),
        }
    }

    pub fn remove(
        &mut self,
        name: &str,
        loaded_models: &HashSet<String>,
    ) -> Result<RemovedModel, ModelError> {
        let record = self.resolve(name)?.clone();
        if record.source != ModelSource::DiskMule {
            return Err(ModelError::ExternalOwnership {
                name: name.to_owned(),
                owner: record.source,
            });
        }
        if loaded_models.contains(name) {
            return Err(ModelError::Loaded(name.to_owned()));
        }

        let managed_indexes = self
            .managed
            .models
            .iter()
            .enumerate()
            .filter(|(_, entry)| entry.name == name)
            .collect::<Vec<_>>();
        if managed_indexes.len() != 1 {
            return Err(ModelError::Ambiguous {
                name: name.to_owned(),
                matches: managed_indexes.len(),
            });
        }
        let (managed_index, entry) = managed_indexes[0];
        let target = validate_managed_target(&self.managed_root, name, &entry.path)?;
        if self
            .managed
            .models
            .iter()
            .enumerate()
            .any(|(index, other)| index != managed_index && other.path == entry.path)
        {
            return Err(ModelError::SharedManagedPath {
                name: name.to_owned(),
            });
        }

        let quarantine = quarantine_target(&target)?;

        let removed_entry = self.managed.models.remove(managed_index);
        if let Err(error) = write_managed_registry(&self.registry_path, &self.managed) {
            self.managed.models.insert(managed_index, removed_entry);
            let _ = fs::rename(&quarantine, &target);
            return Err(error);
        }

        if let Err(source) = fs::remove_file(&quarantine) {
            return Err(ModelError::Write {
                path: quarantine,
                source,
            });
        }
        self.records.retain(|candidate| candidate.name != name);

        Ok(RemovedModel {
            name: name.to_owned(),
            path: target,
        })
    }
}

fn load_managed_registry(path: &Path) -> Result<ManagedRegistry, ModelError> {
    if !path.exists() {
        return Ok(ManagedRegistry::default());
    }
    let bytes = read_bounded(path, MAX_REGISTRY_BYTES)?;
    let registry: ManagedRegistry =
        serde_json::from_slice(&bytes).map_err(|error| ModelError::InvalidRegistry {
            path: path.to_owned(),
            message: error.to_string(),
        })?;
    if registry.version != REGISTRY_VERSION {
        return Err(ModelError::UnsupportedRegistryVersion(registry.version));
    }
    Ok(registry)
}

fn write_managed_registry(path: &Path, registry: &ManagedRegistry) -> Result<(), ModelError> {
    let parent = path.parent().ok_or_else(|| ModelError::InvalidRegistry {
        path: path.to_owned(),
        message: "registry path has no parent directory".to_owned(),
    })?;
    fs::create_dir_all(parent).map_err(|source| ModelError::Write {
        path: parent.to_owned(),
        source,
    })?;
    let mut temporary =
        tempfile::NamedTempFile::new_in(parent).map_err(|source| ModelError::Write {
            path: parent.to_owned(),
            source,
        })?;
    {
        let mut writer = BufWriter::new(temporary.as_file_mut());
        serde_json::to_writer_pretty(&mut writer, registry).map_err(|source| {
            ModelError::Write {
                path: path.to_owned(),
                source: std::io::Error::other(source),
            }
        })?;
        writer
            .write_all(b"\n")
            .map_err(|source| ModelError::Write {
                path: path.to_owned(),
                source,
            })?;
        writer.flush().map_err(|source| ModelError::Write {
            path: path.to_owned(),
            source,
        })?;
    }
    temporary
        .as_file()
        .sync_all()
        .map_err(|source| ModelError::Write {
            path: path.to_owned(),
            source,
        })?;
    temporary.persist(path).map_err(|error| ModelError::Write {
        path: path.to_owned(),
        source: error.error,
    })?;
    Ok(())
}

fn inspect_managed_entry(root: &Path, entry: &ManagedEntry) -> ModelRecord {
    match validate_managed_target(root, &entry.name, &entry.path) {
        Ok(path) => inspect_model(entry.name.clone(), path, ModelSource::DiskMule, None),
        Err(error) => ModelRecord {
            name: entry.name.clone(),
            path: Some(root.join(&entry.path)),
            architecture: None,
            quantization: None,
            size: None,
            source: ModelSource::DiskMule,
            compatibility: Compatibility::Invalid(error.to_string()),
            gguf: None,
        },
    }
}

fn discover_ollama(root: &Path) -> Vec<ModelRecord> {
    let manifest_root = root.join("manifests");
    let mut manifests = Vec::new();
    collect_files(&manifest_root, 0, &mut manifests);
    manifests.sort();
    manifests
        .into_iter()
        .filter_map(|path| ollama_name(&manifest_root, &path).map(|name| (name, path)))
        .map(|(name, manifest_path)| inspect_ollama_manifest(root, name, &manifest_path))
        .collect()
}

fn collect_files(directory: &Path, depth: usize, output: &mut Vec<PathBuf>) {
    if depth > 16 {
        return;
    }
    let Ok(entries) = fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_symlink() {
            continue;
        }
        if file_type.is_dir() {
            collect_files(&entry.path(), depth + 1, output);
        } else if file_type.is_file() {
            output.push(entry.path());
        }
    }
}

fn ollama_name(manifest_root: &Path, manifest: &Path) -> Option<String> {
    let components = manifest
        .strip_prefix(manifest_root)
        .ok()?
        .components()
        .filter_map(|component| match component {
            Component::Normal(value) => value.to_str().map(str::to_owned),
            _ => None,
        })
        .collect::<Vec<_>>();
    if components.len() < 4 {
        return None;
    }
    let namespace = &components[components.len() - 3];
    let model = &components[components.len() - 2];
    let tag = &components[components.len() - 1];
    if namespace == "library" {
        Some(format!("{model}:{tag}"))
    } else {
        Some(format!("{namespace}/{model}:{tag}"))
    }
}

fn inspect_ollama_manifest(root: &Path, name: String, manifest_path: &Path) -> ModelRecord {
    let result = (|| {
        let bytes = read_bounded(manifest_path, MAX_OLLAMA_MANIFEST_BYTES)?;
        let manifest: OllamaManifest =
            serde_json::from_slice(&bytes).map_err(|error| ModelError::InvalidRegistry {
                path: manifest_path.to_owned(),
                message: error.to_string(),
            })?;
        let layers = manifest
            .layers
            .iter()
            .filter(|layer| layer.media_type == OLLAMA_MODEL_MEDIA_TYPE)
            .collect::<Vec<_>>();
        if layers.len() != 1 {
            return Err(ModelError::InvalidRegistry {
                path: manifest_path.to_owned(),
                message: format!("expected one model layer, found {}", layers.len()),
            });
        }
        let layer = layers[0];
        let digest =
            parse_sha256_digest(&layer.digest).ok_or_else(|| ModelError::InvalidRegistry {
                path: manifest_path.to_owned(),
                message: format!("invalid model digest {:?}", layer.digest),
            })?;
        let blob = root.join("blobs").join(format!("sha256-{digest}"));
        Ok((blob, layer.size))
    })();

    match result {
        Ok((blob, declared_size)) => inspect_model(name, blob, ModelSource::Ollama, declared_size),
        Err(error) => ModelRecord {
            name,
            path: None,
            architecture: None,
            quantization: None,
            size: None,
            source: ModelSource::Ollama,
            compatibility: Compatibility::Invalid(error.to_string()),
            gguf: None,
        },
    }
}

fn inspect_model(
    name: String,
    path: PathBuf,
    source: ModelSource,
    declared_size: Option<u64>,
) -> ModelRecord {
    let actual_size = fs::metadata(&path).ok().map(|metadata| metadata.len());
    match GgufFile::inspect(&path) {
        Ok(gguf) => {
            let architecture = gguf.architecture().map(str::to_owned);
            let quantization = gguf.file_type().map(quantization_name);
            let compatibility = match architecture.as_deref() {
                Some("gemma4") => Compatibility::MetadataCompatible,
                Some(other) => Compatibility::Unsupported(format!(
                    "architecture {other}; the first pass supports gemma4 metadata"
                )),
                None => Compatibility::Invalid("missing general.architecture".to_owned()),
            };
            ModelRecord {
                name,
                path: Some(path),
                architecture,
                quantization,
                size: actual_size.or(declared_size),
                source,
                compatibility,
                gguf: Some(GgufSummary {
                    version: gguf.version,
                    alignment: gguf.alignment,
                    data_offset: gguf.data_offset,
                    tensor_count: gguf.tensors.len(),
                }),
            }
        }
        Err(error) => ModelRecord {
            name,
            path: Some(path),
            architecture: None,
            quantization: None,
            size: actual_size.or(declared_size),
            source,
            compatibility: Compatibility::Invalid(error.to_string()),
            gguf: None,
        },
    }
}

fn validate_managed_target(
    root: &Path,
    name: &str,
    relative: &Path,
) -> Result<PathBuf, ModelError> {
    if relative.as_os_str().is_empty()
        || relative.is_absolute()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(ModelError::UnsafeManagedPath {
            name: name.to_owned(),
            reason: format!(
                "registry path {:?} is not a normalized relative path",
                relative
            ),
        });
    }

    let target = root.join(relative);
    let link_metadata =
        fs::symlink_metadata(&target).map_err(|error| ModelError::UnsafeManagedPath {
            name: name.to_owned(),
            reason: format!("cannot inspect {}: {error}", target.display()),
        })?;
    if link_metadata.file_type().is_symlink() || !link_metadata.is_file() {
        return Err(ModelError::UnsafeManagedPath {
            name: name.to_owned(),
            reason: "target must be a regular non-symlink file".to_owned(),
        });
    }

    let canonical_root = fs::canonicalize(root).map_err(|error| ModelError::UnsafeManagedPath {
        name: name.to_owned(),
        reason: format!("cannot resolve managed root {}: {error}", root.display()),
    })?;
    let canonical_target =
        fs::canonicalize(&target).map_err(|error| ModelError::UnsafeManagedPath {
            name: name.to_owned(),
            reason: format!("cannot resolve target {}: {error}", target.display()),
        })?;
    if !canonical_target.starts_with(&canonical_root) {
        return Err(ModelError::UnsafeManagedPath {
            name: name.to_owned(),
            reason: format!("{} resolves outside the managed root", target.display()),
        });
    }
    Ok(canonical_target)
}

fn read_bounded(path: &Path, limit: u64) -> Result<Vec<u8>, ModelError> {
    let file = File::open(path).map_err(|source| ModelError::Read {
        path: path.to_owned(),
        source,
    })?;
    let size = file
        .metadata()
        .map_err(|source| ModelError::Read {
            path: path.to_owned(),
            source,
        })?
        .len();
    if size > limit {
        return Err(ModelError::MetadataFileTooLarge {
            path: path.to_owned(),
            size,
        });
    }
    let capacity = usize::try_from(size).map_err(|_| ModelError::MetadataFileTooLarge {
        path: path.to_owned(),
        size,
    })?;
    let mut bytes = Vec::with_capacity(capacity);
    BufReader::new(file)
        .read_to_end(&mut bytes)
        .map_err(|source| ModelError::Read {
            path: path.to_owned(),
            source,
        })?;
    Ok(bytes)
}

fn parse_sha256_digest(digest: &str) -> Option<&str> {
    let value = digest.strip_prefix("sha256:")?;
    (value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())).then_some(value)
}

fn quantization_name(file_type: u32) -> String {
    match file_type {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        30 => "BF16",
        _ => return format!("GGUF_FILE_TYPE_{file_type}"),
    }
    .to_owned()
}

fn source_rank(source: ModelSource) -> u8 {
    match source {
        ModelSource::DiskMule => 0,
        ModelSource::Ollama => 1,
        ModelSource::LocalFile => 2,
    }
}

fn quarantine_target(target: &Path) -> Result<PathBuf, ModelError> {
    let parent = target
        .parent()
        .ok_or_else(|| ModelError::UnsafeManagedPath {
            name: target.display().to_string(),
            reason: "target has no parent directory".to_owned(),
        })?;
    let temporary = tempfile::Builder::new()
        .prefix(".diskmule-removing-")
        .tempfile_in(parent)
        .map_err(|source| ModelError::Write {
            path: parent.to_owned(),
            source,
        })?;
    let quarantine = temporary.path().to_owned();
    fs::rename(target, &quarantine).map_err(|source| ModelError::Write {
        path: target.to_owned(),
        source,
    })?;
    temporary.keep().map_err(|error| ModelError::Write {
        path: quarantine.clone(),
        source: error.error,
    })?;
    Ok(quarantine)
}

pub fn format_size(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} {}", UNITS[unit])
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, fs, path::Path};

    use tempfile::TempDir;

    use super::{Compatibility, ModelCatalog, ModelError, ModelSource};
    use crate::config::Paths;

    #[test]
    fn discovers_managed_and_ollama_models_with_distinct_ownership() {
        let temp = TempDir::new().unwrap();
        let paths = Paths::from_root(temp.path().join("diskmule"));
        fs::create_dir_all(&paths.models).unwrap();
        write_fixture(&paths.models.join("managed.gguf"));
        write_registry(&paths, &[("managed:latest", "managed.gguf")]);

        let ollama = temp.path().join("ollama");
        write_ollama_model(&ollama, "gemma4", "26b");

        let catalog = ModelCatalog::discover(&paths, Some(ollama.clone())).unwrap();
        let managed = catalog.resolve("managed:latest").unwrap();
        assert_eq!(managed.source, ModelSource::DiskMule);
        assert_eq!(managed.architecture.as_deref(), Some("gemma4"));
        assert_eq!(managed.quantization.as_deref(), Some("Q4_K_M"));
        assert_eq!(managed.compatibility, Compatibility::MetadataCompatible);

        let external = catalog.resolve("gemma4:26b").unwrap();
        assert_eq!(external.source, ModelSource::Ollama);
        assert!(external.compatibility.is_metadata_compatible());
        assert!(
            external
                .path
                .as_ref()
                .unwrap()
                .starts_with(ollama.join("blobs"))
        );
    }

    #[test]
    fn resolves_a_direct_local_file_as_read_only() {
        let temp = TempDir::new().unwrap();
        let paths = Paths::from_root(temp.path().join("diskmule"));
        let local = temp.path().join("direct.gguf");
        write_fixture(&local);

        let catalog = ModelCatalog::discover(&paths, None).unwrap();
        let record = catalog.resolve_for_run(local.to_str().unwrap()).unwrap();
        assert_eq!(record.source, ModelSource::LocalFile);
        assert_eq!(record.path, Some(fs::canonicalize(local).unwrap()));
        assert_eq!(record.architecture.as_deref(), Some("gemma4"));
        assert!(record.compatibility.is_metadata_compatible());
    }

    #[test]
    fn refuses_to_remove_ollama_owned_model() {
        let temp = TempDir::new().unwrap();
        let paths = Paths::from_root(temp.path().join("diskmule"));
        let ollama = temp.path().join("ollama");
        let blob = write_ollama_model(&ollama, "gemma4", "26b");
        let before = fs::read(&blob).unwrap();

        let mut catalog = ModelCatalog::discover(&paths, Some(ollama)).unwrap();
        let error = catalog.remove("gemma4:26b", &HashSet::new()).unwrap_err();
        assert!(matches!(error, ModelError::ExternalOwnership { .. }));
        assert_eq!(fs::read(blob).unwrap(), before);
    }

    #[test]
    fn refuses_out_of_root_and_symlink_targets() {
        let temp = TempDir::new().unwrap();
        let paths = Paths::from_root(temp.path().join("diskmule"));
        fs::create_dir_all(&paths.models).unwrap();
        let outside = temp.path().join("outside.gguf");
        write_fixture(&outside);
        write_registry(&paths, &[("outside", "../outside.gguf")]);

        let mut catalog = ModelCatalog::discover(&paths, None).unwrap();
        let error = catalog.remove("outside", &HashSet::new()).unwrap_err();
        assert!(matches!(error, ModelError::UnsafeManagedPath { .. }));
        assert!(outside.exists());

        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;

            let link = paths.models.join("linked.gguf");
            symlink(&outside, &link).unwrap();
            write_registry(&paths, &[("linked", "linked.gguf")]);
            let mut catalog = ModelCatalog::discover(&paths, None).unwrap();
            let error = catalog.remove("linked", &HashSet::new()).unwrap_err();
            assert!(matches!(error, ModelError::UnsafeManagedPath { .. }));
            assert!(outside.exists());
        }
    }

    #[test]
    fn refuses_loaded_and_ambiguous_models() {
        let temp = TempDir::new().unwrap();
        let paths = Paths::from_root(temp.path().join("diskmule"));
        fs::create_dir_all(&paths.models).unwrap();
        let managed_path = paths.models.join("same.gguf");
        write_fixture(&managed_path);
        write_registry(&paths, &[("same:latest", "same.gguf")]);

        let mut loaded_catalog = ModelCatalog::discover(&paths, None).unwrap();
        let loaded = HashSet::from(["same:latest".to_owned()]);
        assert!(matches!(
            loaded_catalog.remove("same:latest", &loaded),
            Err(ModelError::Loaded(_))
        ));
        assert!(managed_path.exists());

        let ollama = temp.path().join("ollama");
        write_ollama_model(&ollama, "same", "latest");
        let mut ambiguous_catalog = ModelCatalog::discover(&paths, Some(ollama)).unwrap();
        assert!(matches!(
            ambiguous_catalog.remove("same:latest", &HashSet::new()),
            Err(ModelError::Ambiguous { matches: 2, .. })
        ));
        assert!(managed_path.exists());
    }

    #[test]
    fn removes_only_the_validated_managed_file_and_updates_registry() {
        let temp = TempDir::new().unwrap();
        let paths = Paths::from_root(temp.path().join("diskmule"));
        fs::create_dir_all(&paths.models).unwrap();
        let target = paths.models.join("mine.gguf");
        write_fixture(&target);
        let canonical_target = fs::canonicalize(&target).unwrap();
        write_registry(&paths, &[("mine:latest", "mine.gguf")]);

        let mut catalog = ModelCatalog::discover(&paths, None).unwrap();
        let removed = catalog.remove("mine:latest", &HashSet::new()).unwrap();
        assert_eq!(removed.path, canonical_target);
        assert!(!target.exists());
        assert!(matches!(
            catalog.resolve("mine:latest"),
            Err(ModelError::NotFound(_))
        ));
        let registry = fs::read_to_string(&paths.registry).unwrap();
        assert!(!registry.contains("mine:latest"));
    }

    fn write_registry(paths: &Paths, entries: &[(&str, &str)]) {
        fs::create_dir_all(&paths.root).unwrap();
        let models = entries
            .iter()
            .map(|(name, path)| serde_json::json!({ "name": name, "path": path }))
            .collect::<Vec<_>>();
        let registry = serde_json::json!({ "version": 1, "models": models });
        fs::write(
            &paths.registry,
            serde_json::to_vec_pretty(&registry).unwrap(),
        )
        .unwrap();
    }

    fn write_ollama_model(root: &Path, model: &str, tag: &str) -> std::path::PathBuf {
        let digest = "a".repeat(64);
        let blob = root.join("blobs").join(format!("sha256-{digest}"));
        fs::create_dir_all(blob.parent().unwrap()).unwrap();
        write_fixture(&blob);
        let manifest = root
            .join("manifests/registry.ollama.ai/library")
            .join(model)
            .join(tag);
        fs::create_dir_all(manifest.parent().unwrap()).unwrap();
        let size = fs::metadata(&blob).unwrap().len();
        fs::write(
            manifest,
            serde_json::to_vec(&serde_json::json!({
                "schemaVersion": 2,
                "layers": [{
                    "mediaType": "application/vnd.ollama.image.model",
                    "digest": format!("sha256:{digest}"),
                    "size": size
                }]
            }))
            .unwrap(),
        )
        .unwrap();
        blob
    }

    fn write_fixture(path: &Path) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        put_u32(&mut bytes, 3);
        put_u64(&mut bytes, 1);
        put_u64(&mut bytes, 3);
        put_string_metadata(&mut bytes, "general.architecture", "gemma4");
        put_u32_metadata(&mut bytes, "general.file_type", 15);
        put_u32_metadata(&mut bytes, "general.alignment", 32);
        put_string(&mut bytes, "weight");
        put_u32(&mut bytes, 1);
        put_u64(&mut bytes, 32);
        put_u32(&mut bytes, 2);
        put_u64(&mut bytes, 0);
        let padding = (32 - (bytes.len() % 32)) % 32;
        bytes.resize(bytes.len() + padding, 0);
        bytes.extend_from_slice(&[0_u8; 18]);
        fs::write(path, bytes).unwrap();
    }

    fn put_u32_metadata(bytes: &mut Vec<u8>, key: &str, value: u32) {
        put_string(bytes, key);
        put_u32(bytes, 4);
        put_u32(bytes, value);
    }

    fn put_string_metadata(bytes: &mut Vec<u8>, key: &str, value: &str) {
        put_string(bytes, key);
        put_u32(bytes, 8);
        put_string(bytes, value);
    }

    fn put_string(bytes: &mut Vec<u8>, value: &str) {
        put_u64(bytes, value.len() as u64);
        bytes.extend_from_slice(value.as_bytes());
    }

    fn put_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn put_u64(bytes: &mut Vec<u8>, value: u64) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
}
