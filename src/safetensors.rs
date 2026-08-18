//! Strict, payload-lazy safetensors and sharded-index reader.

use std::{
    collections::{HashMap, HashSet},
    fs::{self, File},
    io::{self, Read},
    path::{Component, Path, PathBuf},
    sync::Arc,
};

#[cfg(not(unix))]
use std::io::{Seek, SeekFrom};

use serde::{
    Deserialize,
    de::{MapAccess, Visitor},
};

const MAX_HEADER_BYTES: u64 = 256 * 1024 * 1024;
const MAX_INDEX_BYTES: u64 = 64 * 1024 * 1024;
const MAX_TENSORS: usize = 1_000_000;
const MAX_SHARDS: usize = 16_384;
const MAX_DIMENSIONS: usize = 16;

#[derive(Debug, thiserror::Error)]
pub enum SafeTensorError {
    #[error("could not read safetensors file {path}: {source}")]
    Read { path: PathBuf, source: io::Error },

    #[error("safetensors header in {path} is {size} bytes; maximum is {maximum}")]
    HeaderTooLarge {
        path: PathBuf,
        size: u64,
        maximum: u64,
    },

    #[error("invalid safetensors JSON in {path}: {message}")]
    InvalidJson { path: PathBuf, message: String },

    #[error("duplicate tensor name {name:?} in {path}")]
    DuplicateTensor { path: PathBuf, name: String },

    #[error("safetensors file {path} contains {count} tensors; maximum is {maximum}")]
    TensorLimit {
        path: PathBuf,
        count: usize,
        maximum: usize,
    },

    #[error("tensor {tensor:?} in {path} uses unsupported dtype {dtype:?}")]
    UnsupportedDtype {
        path: PathBuf,
        tensor: String,
        dtype: String,
    },

    #[error("tensor {tensor:?} in {path} has {count} dimensions; maximum is {maximum}")]
    DimensionLimit {
        path: PathBuf,
        tensor: String,
        count: usize,
        maximum: usize,
    },

    #[error("tensor {tensor:?} in {path} has an invalid or overflowing shape")]
    InvalidShape { path: PathBuf, tensor: String },

    #[error(
        "tensor {tensor:?} in {path} declares {actual} bytes but dtype and shape require {expected}"
    )]
    SizeMismatch {
        path: PathBuf,
        tensor: String,
        expected: u64,
        actual: u64,
    },

    #[error("tensor {tensor:?} range [{start}, {end}) exceeds payload size {payload} in {path}")]
    OutOfBounds {
        path: PathBuf,
        tensor: String,
        start: u64,
        end: u64,
        payload: u64,
    },

    #[error("tensor ranges {left:?} and {right:?} overlap in {path}")]
    OverlappingTensors {
        path: PathBuf,
        left: String,
        right: String,
    },

    #[error("safetensors source {0} contains no .safetensors files")]
    NoShards(PathBuf),

    #[error("safetensors source {path} contains {count} shards; maximum is {maximum}")]
    ShardLimit {
        path: PathBuf,
        count: usize,
        maximum: usize,
    },

    #[error("unsafe shard path {shard:?} in {path}")]
    UnsafeShardPath { path: PathBuf, shard: String },

    #[error("index {path} maps tensor {tensor:?} to missing shard {shard:?}")]
    MissingShard {
        path: PathBuf,
        tensor: String,
        shard: String,
    },

    #[error("index {path} maps tensor {tensor:?} to {expected:?}, but it is stored in {actual:?}")]
    WrongShard {
        path: PathBuf,
        tensor: String,
        expected: String,
        actual: String,
    },

    #[error("tensor {tensor:?} is present in shard {shard:?} but absent from index {path}")]
    UnindexedTensor {
        path: PathBuf,
        tensor: String,
        shard: String,
    },

    #[error("tensor {0:?} was not found")]
    TensorNotFound(String),

    #[error("tensor {tensor:?} read [{start}, {end}) exceeds its {length}-byte payload")]
    ReadOutOfBounds {
        tensor: String,
        start: u64,
        end: u64,
        length: u64,
    },

    #[error("safetensors integer arithmetic overflowed while computing {0}")]
    Overflow(&'static str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafeDtype {
    Bool,
    U8,
    I8,
    U16,
    I16,
    F16,
    Bf16,
    U32,
    I32,
    F32,
    U64,
    I64,
    F64,
    F8E4M3,
    F8E5M2,
}

impl SafeDtype {
    pub const fn byte_width(self) -> u64 {
        match self {
            Self::Bool | Self::U8 | Self::I8 | Self::F8E4M3 | Self::F8E5M2 => 1,
            Self::U16 | Self::I16 | Self::F16 | Self::Bf16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
        }
    }

    fn parse(dtype: &str) -> Option<Self> {
        Some(match dtype {
            "BOOL" => Self::Bool,
            "U8" => Self::U8,
            "I8" => Self::I8,
            "U16" => Self::U16,
            "I16" => Self::I16,
            "F16" => Self::F16,
            "BF16" => Self::Bf16,
            "U32" => Self::U32,
            "I32" => Self::I32,
            "F32" => Self::F32,
            "U64" => Self::U64,
            "I64" => Self::I64,
            "F64" => Self::F64,
            "F8_E4M3" | "F8_E4M3FN" => Self::F8E4M3,
            "F8_E5M2" => Self::F8E5M2,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafeTensorInfo {
    pub name: String,
    pub dtype: SafeDtype,
    pub shape: Vec<u64>,
    pub byte_len: u64,
    pub shard: usize,
    pub absolute_offset: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafeTensorShard {
    pub path: PathBuf,
    pub file_len: u64,
    pub data_offset: u64,
}

#[derive(Debug, Clone)]
pub struct SafeTensorIndex {
    tensors: Vec<SafeTensorInfo>,
    shards: Vec<SafeTensorShard>,
    names: HashMap<String, usize>,
}

impl SafeTensorIndex {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, SafeTensorError> {
        let path = path.as_ref();
        if path.is_file() {
            return Self::from_shard_paths(vec![path.to_path_buf()], None);
        }
        let index_path = path.join("model.safetensors.index.json");
        if index_path.is_file() {
            let weight_map = read_weight_map(&index_path)?;
            let shard_names = weight_map.values().cloned().collect::<HashSet<_>>();
            if shard_names.len() > MAX_SHARDS {
                return Err(SafeTensorError::ShardLimit {
                    path: index_path,
                    count: shard_names.len(),
                    maximum: MAX_SHARDS,
                });
            }
            let mut shard_names = shard_names.into_iter().collect::<Vec<_>>();
            shard_names.sort();
            let mut shards = Vec::with_capacity(shard_names.len());
            for shard in &shard_names {
                validate_shard_name(&index_path, shard)?;
                let shard_path = path.join(shard);
                if !shard_path.is_file() {
                    let tensor = weight_map
                        .iter()
                        .find(|(_, value)| *value == shard)
                        .map(|(name, _)| name.clone())
                        .unwrap_or_default();
                    return Err(SafeTensorError::MissingShard {
                        path: index_path,
                        tensor,
                        shard: shard.clone(),
                    });
                }
                shards.push(shard_path);
            }
            return Self::from_shard_paths(shards, Some((&index_path, &weight_map, &shard_names)));
        }

        let mut shards = fs::read_dir(path)
            .map_err(|source| SafeTensorError::Read {
                path: path.to_path_buf(),
                source,
            })?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| {
                path.extension()
                    .is_some_and(|extension| extension == "safetensors")
            })
            .collect::<Vec<_>>();
        shards.sort();
        if shards.is_empty() {
            return Err(SafeTensorError::NoShards(path.to_path_buf()));
        }
        if shards.len() > MAX_SHARDS {
            return Err(SafeTensorError::ShardLimit {
                path: path.to_path_buf(),
                count: shards.len(),
                maximum: MAX_SHARDS,
            });
        }
        Self::from_shard_paths(shards, None)
    }

    fn from_shard_paths(
        paths: Vec<PathBuf>,
        expected: Option<(&Path, &HashMap<String, String>, &[String])>,
    ) -> Result<Self, SafeTensorError> {
        let mut tensors = Vec::new();
        let mut shards = Vec::with_capacity(paths.len());
        let mut names = HashMap::new();
        for (shard_index, path) in paths.iter().enumerate() {
            let (shard, entries) = read_shard_header(path, shard_index)?;
            let shard_name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default();
            for tensor in entries {
                if names.contains_key(&tensor.name) {
                    return Err(SafeTensorError::DuplicateTensor {
                        path: path.clone(),
                        name: tensor.name,
                    });
                }
                if let Some((index_path, weight_map, _)) = expected {
                    match weight_map.get(&tensor.name) {
                        Some(mapped) if mapped == shard_name => {}
                        Some(mapped) => {
                            return Err(SafeTensorError::WrongShard {
                                path: index_path.to_path_buf(),
                                tensor: tensor.name,
                                expected: mapped.clone(),
                                actual: shard_name.to_owned(),
                            });
                        }
                        None => {
                            return Err(SafeTensorError::UnindexedTensor {
                                path: index_path.to_path_buf(),
                                tensor: tensor.name,
                                shard: shard_name.to_owned(),
                            });
                        }
                    }
                }
                names.insert(tensor.name.clone(), tensors.len());
                tensors.push(tensor);
            }
            shards.push(shard);
        }
        if let Some((index_path, weight_map, shard_names)) = expected {
            for (tensor, shard) in weight_map {
                if !names.contains_key(tensor) {
                    return Err(SafeTensorError::MissingShard {
                        path: index_path.to_path_buf(),
                        tensor: tensor.clone(),
                        shard: shard.clone(),
                    });
                }
            }
            debug_assert_eq!(shards.len(), shard_names.len());
        }
        tensors.sort_by(|left, right| left.name.cmp(&right.name));
        names = tensors
            .iter()
            .enumerate()
            .map(|(index, tensor)| (tensor.name.clone(), index))
            .collect();
        Ok(Self {
            tensors,
            shards,
            names,
        })
    }

    pub fn tensors(&self) -> &[SafeTensorInfo] {
        &self.tensors
    }

    pub fn shards(&self) -> &[SafeTensorShard] {
        &self.shards
    }

    pub fn tensor(&self, name: &str) -> Option<&SafeTensorInfo> {
        self.names.get(name).map(|index| &self.tensors[*index])
    }
}

#[derive(Debug, Clone)]
pub struct SafeTensorSource {
    pub index: SafeTensorIndex,
    files: Vec<Arc<File>>,
}

impl SafeTensorSource {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, SafeTensorError> {
        let index = SafeTensorIndex::open(path)?;
        let files = index
            .shards
            .iter()
            .map(|shard| {
                File::open(&shard.path)
                    .map(Arc::new)
                    .map_err(|source| SafeTensorError::Read {
                        path: shard.path.clone(),
                        source,
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { index, files })
    }

    pub fn read_tensor_at(
        &self,
        name: &str,
        offset: u64,
        destination: &mut [u8],
    ) -> Result<(), SafeTensorError> {
        let tensor = self
            .index
            .tensor(name)
            .ok_or_else(|| SafeTensorError::TensorNotFound(name.to_owned()))?;
        let length = u64::try_from(destination.len())
            .map_err(|_| SafeTensorError::Overflow("tensor read length"))?;
        let end = offset
            .checked_add(length)
            .ok_or(SafeTensorError::Overflow("tensor read end"))?;
        if end > tensor.byte_len {
            return Err(SafeTensorError::ReadOutOfBounds {
                tensor: name.to_owned(),
                start: offset,
                end,
                length: tensor.byte_len,
            });
        }
        let absolute = tensor
            .absolute_offset
            .checked_add(offset)
            .ok_or(SafeTensorError::Overflow("absolute tensor read offset"))?;
        read_exact_at(&self.files[tensor.shard], destination, absolute).map_err(|source| {
            SafeTensorError::Read {
                path: self.index.shards[tensor.shard].path.clone(),
                source,
            }
        })
    }

    pub fn shared_file(&self, shard: usize) -> Option<Arc<File>> {
        self.files.get(shard).map(Arc::clone)
    }
}

#[derive(Deserialize)]
struct RawTensor {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

struct HeaderEntries(Vec<(String, serde_json::Value)>);

impl<'de> Deserialize<'de> for HeaderEntries {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct EntriesVisitor;
        impl<'de> Visitor<'de> for EntriesVisitor {
            type Value = HeaderEntries;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a safetensors header object")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut entries = Vec::new();
                let mut names = HashSet::new();
                while let Some((name, value)) = map.next_entry::<String, serde_json::Value>()? {
                    if !names.insert(name.clone()) {
                        return Err(serde::de::Error::custom(format!(
                            "duplicate tensor name {name:?}"
                        )));
                    }
                    entries.push((name, value));
                }
                Ok(HeaderEntries(entries))
            }
        }
        deserializer.deserialize_map(EntriesVisitor)
    }
}

fn read_shard_header(
    path: &Path,
    shard_index: usize,
) -> Result<(SafeTensorShard, Vec<SafeTensorInfo>), SafeTensorError> {
    let mut file = File::open(path).map_err(|source| SafeTensorError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let file_len = file
        .metadata()
        .map_err(|source| SafeTensorError::Read {
            path: path.to_path_buf(),
            source,
        })?
        .len();
    let mut length = [0_u8; 8];
    file.read_exact(&mut length)
        .map_err(|source| SafeTensorError::Read {
            path: path.to_path_buf(),
            source,
        })?;
    let header_len = u64::from_le_bytes(length);
    if header_len > MAX_HEADER_BYTES {
        return Err(SafeTensorError::HeaderTooLarge {
            path: path.to_path_buf(),
            size: header_len,
            maximum: MAX_HEADER_BYTES,
        });
    }
    let data_offset = 8_u64
        .checked_add(header_len)
        .ok_or(SafeTensorError::Overflow("safetensors data offset"))?;
    if data_offset > file_len {
        return Err(SafeTensorError::OutOfBounds {
            path: path.to_path_buf(),
            tensor: "<header>".to_owned(),
            start: 8,
            end: data_offset,
            payload: file_len.saturating_sub(8),
        });
    }
    let header_size = usize::try_from(header_len)
        .map_err(|_| SafeTensorError::Overflow("safetensors header allocation"))?;
    let mut header = vec![0_u8; header_size];
    file.read_exact(&mut header)
        .map_err(|source| SafeTensorError::Read {
            path: path.to_path_buf(),
            source,
        })?;
    let raw: HeaderEntries =
        serde_json::from_slice(&header).map_err(|error| SafeTensorError::InvalidJson {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;
    if raw.0.len() > MAX_TENSORS + 1 {
        return Err(SafeTensorError::TensorLimit {
            path: path.to_path_buf(),
            count: raw.0.len(),
            maximum: MAX_TENSORS,
        });
    }
    let payload_len = file_len - data_offset;
    let mut tensors = Vec::new();
    for (name, value) in raw.0 {
        if name == "__metadata__" {
            continue;
        }
        let raw: RawTensor =
            serde_json::from_value(value).map_err(|error| SafeTensorError::InvalidJson {
                path: path.to_path_buf(),
                message: format!("tensor {name:?}: {error}"),
            })?;
        let dtype =
            SafeDtype::parse(&raw.dtype).ok_or_else(|| SafeTensorError::UnsupportedDtype {
                path: path.to_path_buf(),
                tensor: name.clone(),
                dtype: raw.dtype,
            })?;
        if raw.shape.len() > MAX_DIMENSIONS {
            return Err(SafeTensorError::DimensionLimit {
                path: path.to_path_buf(),
                tensor: name,
                count: raw.shape.len(),
                maximum: MAX_DIMENSIONS,
            });
        }
        let elements = raw
            .shape
            .iter()
            .try_fold(1_u64, |count, dimension| count.checked_mul(*dimension));
        let expected = elements
            .and_then(|elements| elements.checked_mul(dtype.byte_width()))
            .ok_or_else(|| SafeTensorError::InvalidShape {
                path: path.to_path_buf(),
                tensor: name.clone(),
            })?;
        let [start, end] = raw.data_offsets;
        let actual = end
            .checked_sub(start)
            .ok_or_else(|| SafeTensorError::OutOfBounds {
                path: path.to_path_buf(),
                tensor: name.clone(),
                start,
                end,
                payload: payload_len,
            })?;
        if actual != expected {
            return Err(SafeTensorError::SizeMismatch {
                path: path.to_path_buf(),
                tensor: name,
                expected,
                actual,
            });
        }
        if end > payload_len {
            return Err(SafeTensorError::OutOfBounds {
                path: path.to_path_buf(),
                tensor: name,
                start,
                end,
                payload: payload_len,
            });
        }
        let absolute_offset = data_offset
            .checked_add(start)
            .ok_or(SafeTensorError::Overflow("absolute tensor offset"))?;
        tensors.push(SafeTensorInfo {
            name,
            dtype,
            shape: raw.shape,
            byte_len: actual,
            shard: shard_index,
            absolute_offset,
        });
    }
    if tensors.len() > MAX_TENSORS {
        return Err(SafeTensorError::TensorLimit {
            path: path.to_path_buf(),
            count: tensors.len(),
            maximum: MAX_TENSORS,
        });
    }
    let mut ranges = tensors
        .iter()
        .filter(|tensor| tensor.byte_len > 0)
        .map(|tensor| {
            (
                tensor.absolute_offset,
                tensor.absolute_offset + tensor.byte_len,
                &tensor.name,
            )
        })
        .collect::<Vec<_>>();
    ranges.sort_by_key(|range| range.0);
    for pair in ranges.windows(2) {
        if pair[0].1 > pair[1].0 {
            return Err(SafeTensorError::OverlappingTensors {
                path: path.to_path_buf(),
                left: pair[0].2.clone(),
                right: pair[1].2.clone(),
            });
        }
    }
    Ok((
        SafeTensorShard {
            path: path.to_path_buf(),
            file_len,
            data_offset,
        },
        tensors,
    ))
}

#[derive(Deserialize)]
struct ShardIndex {
    weight_map: HashMap<String, String>,
}

fn read_weight_map(path: &Path) -> Result<HashMap<String, String>, SafeTensorError> {
    let mut file = File::open(path).map_err(|source| SafeTensorError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let length = file
        .metadata()
        .map_err(|source| SafeTensorError::Read {
            path: path.to_path_buf(),
            source,
        })?
        .len();
    if length > MAX_INDEX_BYTES {
        return Err(SafeTensorError::HeaderTooLarge {
            path: path.to_path_buf(),
            size: length,
            maximum: MAX_INDEX_BYTES,
        });
    }
    let mut bytes = Vec::with_capacity(
        usize::try_from(length).map_err(|_| SafeTensorError::Overflow("index allocation"))?,
    );
    file.read_to_end(&mut bytes)
        .map_err(|source| SafeTensorError::Read {
            path: path.to_path_buf(),
            source,
        })?;
    let index: ShardIndex =
        serde_json::from_slice(&bytes).map_err(|error| SafeTensorError::InvalidJson {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;
    if index.weight_map.len() > MAX_TENSORS {
        return Err(SafeTensorError::TensorLimit {
            path: path.to_path_buf(),
            count: index.weight_map.len(),
            maximum: MAX_TENSORS,
        });
    }
    Ok(index.weight_map)
}

fn validate_shard_name(index_path: &Path, shard: &str) -> Result<(), SafeTensorError> {
    let path = Path::new(shard);
    let mut components = path.components();
    if path.is_absolute()
        || !matches!(components.next(), Some(Component::Normal(_)))
        || components.next().is_some()
        || path
            .extension()
            .is_none_or(|extension| extension != "safetensors")
    {
        return Err(SafeTensorError::UnsafeShardPath {
            path: index_path.to_path_buf(),
            shard: shard.to_owned(),
        });
    }
    Ok(())
}

#[cfg(unix)]
fn read_exact_at(file: &File, mut destination: &mut [u8], mut offset: u64) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    while !destination.is_empty() {
        let read = file.read_at(destination, offset)?;
        if read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "short positional safetensors read",
            ));
        }
        destination = &mut destination[read..];
        offset = offset
            .checked_add(read as u64)
            .ok_or_else(|| io::Error::other("safetensors read offset overflow"))?;
    }
    Ok(())
}

#[cfg(not(unix))]
fn read_exact_at(file: &File, destination: &mut [u8], offset: u64) -> io::Result<()> {
    let mut file = file.try_clone()?;
    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(destination)
}

#[cfg(test)]
mod tests {
    use std::{fs, io::Write};

    use serde_json::json;
    use tempfile::TempDir;

    use super::{SafeDtype, SafeTensorError, SafeTensorIndex, SafeTensorSource};

    fn write_shard(path: &Path, entries: serde_json::Value, payload: &[u8]) {
        let header = serde_json::to_vec(&entries).unwrap();
        let mut file = fs::File::create(path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.write_all(payload).unwrap();
    }

    #[test]
    fn indexes_and_positionally_reads_valid_payloads() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("model.safetensors");
        write_shard(
            &path,
            json!({
                "__metadata__": {"format": "pt"},
                "a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
                "b": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [8, 16]}
            }),
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        );
        let source = SafeTensorSource::open(&path).unwrap();
        assert_eq!(source.index.tensors().len(), 2);
        assert_eq!(source.index.tensor("b").unwrap().dtype, SafeDtype::Bf16);
        assert!(source.index.tensor("b").unwrap().absolute_offset > 8);
        let mut bytes = [0_u8; 4];
        source.read_tensor_at("b", 2, &mut bytes).unwrap();
        assert_eq!(bytes, [11, 12, 13, 14]);
        assert!(source.read_tensor_at("b", 7, &mut bytes).is_err());
    }

    #[test]
    fn validates_sharded_index_ownership_and_paths() {
        let temp = TempDir::new().unwrap();
        write_shard(
            &temp.path().join("model-00001-of-00002.safetensors"),
            json!({"a": {"dtype": "I8", "shape": [2], "data_offsets": [0, 2]}}),
            &[1, 2],
        );
        write_shard(
            &temp.path().join("model-00002-of-00002.safetensors"),
            json!({"b": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}),
            &[3, 4],
        );
        fs::write(
            temp.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&json!({"weight_map": {
                "a": "model-00001-of-00002.safetensors",
                "b": "model-00002-of-00002.safetensors"
            }}))
            .unwrap(),
        )
        .unwrap();
        let index = SafeTensorIndex::open(temp.path()).unwrap();
        assert_eq!(index.shards().len(), 2);
        assert_eq!(index.tensor("a").unwrap().shard, 0);
        assert_eq!(index.tensor("b").unwrap().shard, 1);

        fs::write(
            temp.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&json!({"weight_map": {"a": "../escape.safetensors"}})).unwrap(),
        )
        .unwrap();
        assert!(matches!(
            SafeTensorIndex::open(temp.path()),
            Err(SafeTensorError::UnsafeShardPath { .. })
        ));
    }

    #[test]
    fn rejects_size_mismatch_overlap_and_truncation() {
        let temp = TempDir::new().unwrap();
        let mismatch = temp.path().join("mismatch.safetensors");
        write_shard(
            &mismatch,
            json!({"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}}),
            &[0; 8],
        );
        assert!(matches!(
            SafeTensorIndex::open(mismatch),
            Err(SafeTensorError::SizeMismatch { .. })
        ));

        let overlap = temp.path().join("overlap.safetensors");
        write_shard(
            &overlap,
            json!({
                "a": {"dtype": "I8", "shape": [4], "data_offsets": [0, 4]},
                "b": {"dtype": "I8", "shape": [4], "data_offsets": [2, 6]}
            }),
            &[0; 6],
        );
        assert!(matches!(
            SafeTensorIndex::open(overlap),
            Err(SafeTensorError::OverlappingTensors { .. })
        ));

        let truncated = temp.path().join("truncated.safetensors");
        fs::write(&truncated, 100_u64.to_le_bytes()).unwrap();
        assert!(SafeTensorIndex::open(truncated).is_err());
    }

    use std::path::Path;
}
