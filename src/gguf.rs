//! Bounded GGUF metadata inspection.
//!
//! This module intentionally stops at indexing tensor byte ranges. It never
//! reads tensor payloads, which keeps model discovery cheap even for very large
//! files.

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs::File,
    io::{self, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    sync::Arc,
};

#[cfg(unix)]
use std::os::unix::fs::FileExt;

const MAGIC: &[u8; 4] = b"GGUF";
const DEFAULT_ALIGNMENT: u32 = 32;
const MAX_METADATA_ENTRIES: u64 = 1_000_000;
const MAX_TENSORS: u64 = 1_000_000;
const MAX_KEY_BYTES: u64 = 64 * 1024;
const MAX_NAME_BYTES: u64 = 1024 * 1024;
const MAX_VALUE_STRING_BYTES: u64 = 64 * 1024 * 1024;
const MAX_ARRAY_ELEMENTS: u64 = 16_000_000;
const MAX_CAPTURED_ARRAY_ELEMENTS: u64 = 1_000_000;
const MAX_CAPTURED_ARRAY_BYTES: u64 = 256 * 1024 * 1024;
const MAX_DIMENSIONS: u32 = 4;

#[derive(Debug, thiserror::Error)]
pub enum GgufError {
    #[error("{path} is not a GGUF file")]
    InvalidMagic { path: String },

    #[error("unsupported GGUF version {0}; DiskMule supports versions 2 and 3")]
    UnsupportedVersion(u32),

    #[error("GGUF is truncated at byte {offset} while reading {field}")]
    Truncated { offset: u64, field: &'static str },

    #[error("{field} count {count} exceeds the inspection limit {limit}")]
    CountLimit {
        field: &'static str,
        count: u64,
        limit: u64,
    },

    #[error("{field} length {length} exceeds the inspection limit {limit}")]
    LengthLimit {
        field: &'static str,
        length: u64,
        limit: u64,
    },

    #[error("invalid UTF-8 in {field} at byte {offset}")]
    InvalidUtf8 { field: &'static str, offset: u64 },

    #[error("unknown GGUF metadata value type {kind} at byte {offset}")]
    UnknownMetadataType { kind: u32, offset: u64 },

    #[error("invalid GGUF boolean value {value} at byte {offset}")]
    InvalidBoolean { value: u8, offset: u64 },

    #[error("nested GGUF metadata arrays are not supported at byte {0}")]
    NestedArray(u64),

    #[error("duplicate GGUF metadata key {0:?}")]
    DuplicateMetadataKey(String),

    #[error("invalid general.alignment value {0}; expected a nonzero power of two")]
    InvalidAlignment(u32),

    #[error("tensor {tensor:?} has invalid dimension count {count}; maximum is {MAX_DIMENSIONS}")]
    InvalidDimensionCount { tensor: String, count: u32 },

    #[error("tensor {tensor:?} has a zero-sized dimension")]
    ZeroDimension { tensor: String },

    #[error("unsupported GGML tensor type {kind} for tensor {tensor:?}")]
    UnsupportedTensorType { tensor: String, kind: u32 },

    #[error(
        "tensor {tensor:?} first dimension {size} is not divisible by the {block}-element block size for {kind}"
    )]
    InvalidBlockShape {
        tensor: String,
        size: u64,
        block: u64,
        kind: &'static str,
    },

    #[error("duplicate tensor name {0:?}")]
    DuplicateTensorName(String),

    #[error("tensor {tensor:?} offset {offset} is not aligned to {alignment} bytes")]
    MisalignedTensor {
        tensor: String,
        offset: u64,
        alignment: u32,
    },

    #[error("tensor {tensor:?} byte range {start}..{end} exceeds the {file_len}-byte GGUF file")]
    TensorOutOfBounds {
        tensor: String,
        start: u64,
        end: u64,
        file_len: u64,
    },

    #[error("tensor {tensor:?} overlaps an earlier tensor at byte {offset}")]
    OverlappingTensor { tensor: String, offset: u64 },

    #[error("integer overflow while calculating {0}")]
    Overflow(&'static str),

    #[error("captured metadata array {key:?} is too large ({length} elements; limit {limit})")]
    CapturedArrayTooLarge {
        key: String,
        length: u64,
        limit: u64,
    },

    #[error("captured metadata array {key:?} exceeds the {limit}-byte memory budget")]
    CapturedArrayBudget { key: String, limit: u64 },

    #[error("tensor {0:?} was not found in the GGUF index")]
    TensorNotFound(String),

    #[error("requested range {offset}..{end} exceeds tensor {tensor:?} length {tensor_len}")]
    TensorReadOutOfBounds {
        tensor: String,
        offset: u64,
        end: u64,
        tensor_len: u64,
    },

    #[error("could not read GGUF: {0}")]
    Io(#[from] io::Error),
}

#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array { element_type: u32, length: u64 },
    ArrayValues(MetadataArray),
    U64(u64),
    I64(i64),
    F64(f64),
}

/// Materialized values for metadata arrays explicitly requested by a runtime.
///
/// Ordinary inspection retains only an array's type and length. This separate
/// representation lets tokenizers load vocabulary metadata without making
/// `diskmule ls` retain hundreds of thousands of strings.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataArray {
    U8(Vec<u8>),
    I8(Vec<i8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    F32(Vec<f32>),
    Bool(Vec<bool>),
    String(Vec<String>),
    U64(Vec<u64>),
    I64(Vec<i64>),
    F64(Vec<f64>),
}

impl MetadataArray {
    pub fn len(&self) -> usize {
        match self {
            Self::U8(values) => values.len(),
            Self::I8(values) => values.len(),
            Self::U16(values) => values.len(),
            Self::I16(values) => values.len(),
            Self::U32(values) => values.len(),
            Self::I32(values) => values.len(),
            Self::F32(values) => values.len(),
            Self::Bool(values) => values.len(),
            Self::String(values) => values.len(),
            Self::U64(values) => values.len(),
            Self::I64(values) => values.len(),
            Self::F64(values) => values.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_strings(&self) -> Option<&[String]> {
        match self {
            Self::String(values) => Some(values),
            _ => None,
        }
    }

    pub fn as_i32s(&self) -> Option<&[i32]> {
        match self {
            Self::I32(values) => Some(values),
            _ => None,
        }
    }

    pub fn as_f32s(&self) -> Option<&[f32]> {
        match self {
            Self::F32(values) => Some(values),
            _ => None,
        }
    }

    pub fn as_bools(&self) -> Option<&[bool]> {
        match self {
            Self::Bool(values) => Some(values),
            _ => None,
        }
    }
}

impl MetadataValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::U32(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&MetadataArray> {
        match self {
            Self::ArrayValues(values) => Some(values),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::F32(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(value) => Some(*value),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    I8,
    I16,
    I32,
    I64,
    F64,
    Bf16,
}

impl TensorType {
    fn from_ggml(kind: u32) -> Option<Self> {
        Some(match kind {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            30 => Self::Bf16,
            _ => return None,
        })
    }

    fn layout(self) -> (u64, u64) {
        match self {
            Self::F32 => (1, 4),
            Self::F16 | Self::Bf16 => (1, 2),
            Self::Q4_0 => (32, 18),
            Self::Q4_1 => (32, 20),
            Self::Q5_0 => (32, 22),
            Self::Q5_1 => (32, 24),
            Self::Q8_0 => (32, 34),
            Self::Q8_1 => (32, 40),
            Self::Q2K => (256, 84),
            Self::Q3K => (256, 110),
            Self::Q4K => (256, 144),
            Self::Q5K => (256, 176),
            Self::Q6K => (256, 210),
            Self::Q8K => (256, 292),
            Self::I8 => (1, 1),
            Self::I16 => (1, 2),
            Self::I32 => (1, 4),
            Self::I64 | Self::F64 => (1, 8),
        }
    }

    pub fn block_elements(self) -> u64 {
        self.layout().0
    }

    pub fn block_bytes(self) -> u64 {
        self.layout().1
    }

    pub fn row_byte_len(self, elements: u64) -> Option<u64> {
        let (block_elements, block_bytes) = self.layout();
        elements
            .is_multiple_of(block_elements)
            .then(|| (elements / block_elements).checked_mul(block_bytes))
            .flatten()
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2K => "Q2_K",
            Self::Q3K => "Q3_K",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
            Self::Q8K => "Q8_K",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::F64 => "F64",
            Self::Bf16 => "BF16",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub kind: TensorType,
    /// Offset relative to the beginning of the GGUF tensor data section.
    pub offset: u64,
    pub absolute_offset: u64,
    pub byte_len: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GgufFile {
    pub version: u32,
    pub alignment: u32,
    pub data_offset: u64,
    pub file_len: u64,
    pub metadata: BTreeMap<String, MetadataValue>,
    pub tensors: Vec<TensorInfo>,
}

impl GgufFile {
    pub fn inspect(path: impl AsRef<Path>) -> Result<Self, GgufError> {
        let path = path.as_ref();
        let mut file = File::open(path)?;
        let file_len = file.metadata()?.len();
        Self::read_from(&mut file, file_len, &path.display().to_string())
    }

    /// Parse a GGUF index while retaining values for only the named metadata
    /// arrays. Scalar and string metadata is always retained.
    pub fn inspect_with_arrays<I, S>(path: impl AsRef<Path>, keys: I) -> Result<Self, GgufError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let path = path.as_ref();
        let captured_keys = keys.into_iter().map(Into::into).collect();
        let mut file = File::open(path)?;
        let file_len = file.metadata()?.len();
        Self::read_from_with_arrays(
            &mut file,
            file_len,
            &path.display().to_string(),
            &captured_keys,
        )
    }

    pub fn architecture(&self) -> Option<&str> {
        self.metadata
            .get("general.architecture")
            .and_then(MetadataValue::as_str)
    }

    pub fn name(&self) -> Option<&str> {
        self.metadata
            .get("general.name")
            .and_then(MetadataValue::as_str)
    }

    pub fn file_type(&self) -> Option<u32> {
        self.metadata
            .get("general.file_type")
            .and_then(MetadataValue::as_u32)
    }

    fn read_from<R: Read + Seek>(
        reader: &mut R,
        file_len: u64,
        display_path: &str,
    ) -> Result<Self, GgufError> {
        Self::read_from_with_arrays(reader, file_len, display_path, &HashSet::new())
    }

    fn read_from_with_arrays<R: Read + Seek>(
        reader: &mut R,
        file_len: u64,
        display_path: &str,
        captured_array_keys: &HashSet<String>,
    ) -> Result<Self, GgufError> {
        let mut input = BoundedReader::new(reader, file_len);

        let magic = input.read_array::<4>("GGUF magic")?;
        if &magic != MAGIC {
            return Err(GgufError::InvalidMagic {
                path: display_path.to_owned(),
            });
        }

        let version = input.read_u32("GGUF version")?;
        if !matches!(version, 2 | 3) {
            return Err(GgufError::UnsupportedVersion(version));
        }

        let tensor_count = input.read_u64("tensor count")?;
        enforce_count("tensor", tensor_count, MAX_TENSORS)?;
        let metadata_count = input.read_u64("metadata count")?;
        enforce_count("metadata", metadata_count, MAX_METADATA_ENTRIES)?;

        let mut metadata = BTreeMap::new();
        for _ in 0..metadata_count {
            let key = input.read_string("metadata key", MAX_KEY_BYTES)?;
            let kind_offset = input.position;
            let kind = input.read_u32("metadata value type")?;
            let value = read_metadata_value(
                &mut input,
                kind,
                kind_offset,
                captured_array_keys.contains(&key).then_some(key.as_str()),
            )?;
            if metadata.insert(key.clone(), value).is_some() {
                return Err(GgufError::DuplicateMetadataKey(key));
            }
        }

        let alignment = match metadata.get("general.alignment") {
            Some(MetadataValue::U32(value)) => *value,
            Some(_) => return Err(GgufError::InvalidAlignment(0)),
            None => DEFAULT_ALIGNMENT,
        };
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(GgufError::InvalidAlignment(alignment));
        }

        let tensor_capacity = usize::try_from(tensor_count)
            .map_err(|_| GgufError::Overflow("tensor vector capacity"))?;
        let mut raw_tensors = Vec::with_capacity(tensor_capacity);
        let mut tensor_names = HashSet::with_capacity(tensor_capacity);
        for _ in 0..tensor_count {
            let name = input.read_string("tensor name", MAX_NAME_BYTES)?;
            if !tensor_names.insert(name.clone()) {
                return Err(GgufError::DuplicateTensorName(name));
            }

            let dimension_count = input.read_u32("tensor dimension count")?;
            if dimension_count > MAX_DIMENSIONS {
                return Err(GgufError::InvalidDimensionCount {
                    tensor: name,
                    count: dimension_count,
                });
            }

            let dimension_capacity = usize::try_from(dimension_count)
                .map_err(|_| GgufError::Overflow("tensor dimensions"))?;
            let mut dimensions = Vec::with_capacity(dimension_capacity);
            for _ in 0..dimension_count {
                let dimension = input.read_u64("tensor dimension")?;
                if dimension == 0 {
                    return Err(GgufError::ZeroDimension { tensor: name });
                }
                dimensions.push(dimension);
            }

            let raw_kind = input.read_u32("tensor type")?;
            let kind = TensorType::from_ggml(raw_kind).ok_or_else(|| {
                GgufError::UnsupportedTensorType {
                    tensor: name.clone(),
                    kind: raw_kind,
                }
            })?;
            let offset = input.read_u64("tensor offset")?;
            let byte_len = tensor_byte_len(&name, &dimensions, kind)?;
            raw_tensors.push((name, dimensions, kind, offset, byte_len));
        }

        let data_offset = align_up(input.position, u64::from(alignment))?;
        if data_offset > file_len {
            return Err(GgufError::Truncated {
                offset: input.position,
                field: "aligned tensor data section",
            });
        }

        let mut tensors = Vec::with_capacity(raw_tensors.len());
        for (name, dimensions, kind, offset, byte_len) in raw_tensors {
            if offset % u64::from(alignment) != 0 {
                return Err(GgufError::MisalignedTensor {
                    tensor: name,
                    offset,
                    alignment,
                });
            }
            let absolute_offset = data_offset
                .checked_add(offset)
                .ok_or(GgufError::Overflow("absolute tensor offset"))?;
            let end = absolute_offset
                .checked_add(byte_len)
                .ok_or(GgufError::Overflow("tensor byte range"))?;
            if end > file_len {
                return Err(GgufError::TensorOutOfBounds {
                    tensor: name,
                    start: absolute_offset,
                    end,
                    file_len,
                });
            }
            tensors.push(TensorInfo {
                name,
                dimensions,
                kind,
                offset,
                absolute_offset,
                byte_len,
            });
        }

        validate_non_overlapping(&tensors)?;

        Ok(Self {
            version,
            alignment,
            data_offset,
            file_len,
            metadata,
            tensors,
        })
    }
}

/// An indexed GGUF file that supports bounded positional tensor reads.
///
/// The file descriptor is shared safely between readers and every request is
/// checked against the tensor's declared byte range before any I/O occurs.
#[derive(Debug)]
pub struct TensorSource {
    path: PathBuf,
    file: Arc<File>,
    pub gguf: GgufFile,
    tensor_indices: HashMap<String, usize>,
}

impl TensorSource {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, GgufError> {
        Self::open_with_arrays(path, std::iter::empty::<String>())
    }

    pub fn open_with_arrays<I, S>(path: impl AsRef<Path>, keys: I) -> Result<Self, GgufError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let path = path.as_ref().to_path_buf();
        let captured_keys = keys.into_iter().map(Into::into).collect();
        let mut file = File::open(&path)?;
        let file_len = file.metadata()?.len();
        let gguf = GgufFile::read_from_with_arrays(
            &mut file,
            file_len,
            &path.display().to_string(),
            &captured_keys,
        )?;
        let file = Arc::new(file);
        let tensor_indices = gguf
            .tensors
            .iter()
            .enumerate()
            .map(|(index, tensor)| (tensor.name.clone(), index))
            .collect();
        Ok(Self {
            path,
            file,
            gguf,
            tensor_indices,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensor_indices
            .get(name)
            .map(|index| &self.gguf.tensors[*index])
    }

    /// Read a bounded subrange of one tensor without moving shared file state.
    pub fn read_tensor_at(
        &self,
        name: &str,
        offset: u64,
        destination: &mut [u8],
    ) -> Result<(), GgufError> {
        let tensor = self
            .tensor(name)
            .ok_or_else(|| GgufError::TensorNotFound(name.to_owned()))?;
        let length = u64::try_from(destination.len())
            .map_err(|_| GgufError::Overflow("tensor read length"))?;
        let end = offset
            .checked_add(length)
            .ok_or(GgufError::Overflow("tensor read range"))?;
        if end > tensor.byte_len {
            return Err(GgufError::TensorReadOutOfBounds {
                tensor: name.to_owned(),
                offset,
                end,
                tensor_len: tensor.byte_len,
            });
        }
        let absolute_offset = tensor
            .absolute_offset
            .checked_add(offset)
            .ok_or(GgufError::Overflow("absolute tensor read offset"))?;
        read_exact_at(&self.file, destination, absolute_offset)?;
        Ok(())
    }
}

fn enforce_count(field: &'static str, count: u64, limit: u64) -> Result<(), GgufError> {
    if count > limit {
        return Err(GgufError::CountLimit {
            field,
            count,
            limit,
        });
    }
    Ok(())
}

fn read_metadata_value<R: Read + Seek>(
    input: &mut BoundedReader<'_, R>,
    kind: u32,
    kind_offset: u64,
    captured_array_key: Option<&str>,
) -> Result<MetadataValue, GgufError> {
    Ok(match kind {
        0 => MetadataValue::U8(input.read_u8("U8 metadata value")?),
        1 => MetadataValue::I8(input.read_u8("I8 metadata value")? as i8),
        2 => MetadataValue::U16(input.read_u16("U16 metadata value")?),
        3 => MetadataValue::I16(input.read_u16("I16 metadata value")? as i16),
        4 => MetadataValue::U32(input.read_u32("U32 metadata value")?),
        5 => MetadataValue::I32(input.read_u32("I32 metadata value")? as i32),
        6 => MetadataValue::F32(f32::from_bits(input.read_u32("F32 metadata value")?)),
        7 => match input.read_u8("BOOL metadata value")? {
            0 => MetadataValue::Bool(false),
            1 => MetadataValue::Bool(true),
            value => {
                return Err(GgufError::InvalidBoolean {
                    value,
                    offset: input.position - 1,
                });
            }
        },
        8 => MetadataValue::String(input.read_string("metadata string", MAX_VALUE_STRING_BYTES)?),
        9 => {
            let element_offset = input.position;
            let element_type = input.read_u32("metadata array element type")?;
            if element_type == 9 {
                return Err(GgufError::NestedArray(element_offset));
            }
            let length = input.read_u64("metadata array length")?;
            enforce_count("metadata array element", length, MAX_ARRAY_ELEMENTS)?;
            if let Some(key) = captured_array_key {
                MetadataValue::ArrayValues(read_metadata_array(
                    input,
                    element_type,
                    element_offset,
                    length,
                    key,
                )?)
            } else {
                for _ in 0..length {
                    skip_metadata_value(input, element_type, element_offset)?;
                }
                MetadataValue::Array {
                    element_type,
                    length,
                }
            }
        }
        10 => MetadataValue::U64(input.read_u64("U64 metadata value")?),
        11 => MetadataValue::I64(input.read_u64("I64 metadata value")? as i64),
        12 => MetadataValue::F64(f64::from_bits(input.read_u64("F64 metadata value")?)),
        _ => {
            return Err(GgufError::UnknownMetadataType {
                kind,
                offset: kind_offset,
            });
        }
    })
}

fn read_metadata_array<R: Read + Seek>(
    input: &mut BoundedReader<'_, R>,
    element_type: u32,
    element_offset: u64,
    length: u64,
    key: &str,
) -> Result<MetadataArray, GgufError> {
    if length > MAX_CAPTURED_ARRAY_ELEMENTS {
        return Err(GgufError::CapturedArrayTooLarge {
            key: key.to_owned(),
            length,
            limit: MAX_CAPTURED_ARRAY_ELEMENTS,
        });
    }
    let capacity = usize::try_from(length)
        .map_err(|_| GgufError::Overflow("captured metadata array capacity"))?;
    let fixed_budget = |element_bytes: u64| -> Result<(), GgufError> {
        let bytes = length
            .checked_mul(element_bytes)
            .ok_or(GgufError::Overflow("captured metadata array size"))?;
        if bytes > MAX_CAPTURED_ARRAY_BYTES {
            return Err(GgufError::CapturedArrayBudget {
                key: key.to_owned(),
                limit: MAX_CAPTURED_ARRAY_BYTES,
            });
        }
        Ok(())
    };

    macro_rules! read_values {
        ($variant:ident, $bytes:expr, $read:expr) => {{
            fixed_budget($bytes)?;
            let mut values = Vec::with_capacity(capacity);
            for _ in 0..length {
                values.push($read?);
            }
            MetadataArray::$variant(values)
        }};
    }

    Ok(match element_type {
        0 => read_values!(U8, 1, input.read_u8("U8 metadata array element")),
        1 => read_values!(
            I8,
            1,
            input.read_u8("I8 metadata array element").map(|v| v as i8)
        ),
        2 => read_values!(U16, 2, input.read_u16("U16 metadata array element")),
        3 => read_values!(
            I16,
            2,
            input
                .read_u16("I16 metadata array element")
                .map(|v| v as i16)
        ),
        4 => read_values!(U32, 4, input.read_u32("U32 metadata array element")),
        5 => read_values!(
            I32,
            4,
            input
                .read_u32("I32 metadata array element")
                .map(|v| v as i32)
        ),
        6 => read_values!(
            F32,
            4,
            input
                .read_u32("F32 metadata array element")
                .map(f32::from_bits)
        ),
        7 => {
            fixed_budget(1)?;
            let mut values = Vec::with_capacity(capacity);
            for _ in 0..length {
                let offset = input.position;
                match input.read_u8("BOOL metadata array element")? {
                    0 => values.push(false),
                    1 => values.push(true),
                    value => return Err(GgufError::InvalidBoolean { value, offset }),
                }
            }
            MetadataArray::Bool(values)
        }
        8 => {
            fixed_budget(std::mem::size_of::<String>() as u64)?;
            let mut values = Vec::with_capacity(capacity);
            let mut retained_bytes = length
                .checked_mul(std::mem::size_of::<String>() as u64)
                .ok_or(GgufError::Overflow("captured string array size"))?;
            for _ in 0..length {
                let value = input.read_string("metadata array string", MAX_VALUE_STRING_BYTES)?;
                retained_bytes = retained_bytes
                    .checked_add(value.len() as u64)
                    .ok_or(GgufError::Overflow("captured string array size"))?;
                if retained_bytes > MAX_CAPTURED_ARRAY_BYTES {
                    return Err(GgufError::CapturedArrayBudget {
                        key: key.to_owned(),
                        limit: MAX_CAPTURED_ARRAY_BYTES,
                    });
                }
                values.push(value);
            }
            MetadataArray::String(values)
        }
        9 => return Err(GgufError::NestedArray(element_offset)),
        10 => read_values!(U64, 8, input.read_u64("U64 metadata array element")),
        11 => read_values!(
            I64,
            8,
            input
                .read_u64("I64 metadata array element")
                .map(|v| v as i64)
        ),
        12 => read_values!(
            F64,
            8,
            input
                .read_u64("F64 metadata array element")
                .map(f64::from_bits)
        ),
        _ => {
            return Err(GgufError::UnknownMetadataType {
                kind: element_type,
                offset: element_offset,
            });
        }
    })
}

#[cfg(unix)]
fn read_exact_at(file: &File, mut destination: &mut [u8], mut offset: u64) -> io::Result<()> {
    while !destination.is_empty() {
        match file.read_at(destination, offset) {
            Ok(0) => return Err(io::Error::from(io::ErrorKind::UnexpectedEof)),
            Ok(read) => {
                offset = offset
                    .checked_add(read as u64)
                    .ok_or_else(|| io::Error::other("positional read offset overflow"))?;
                destination = &mut destination[read..];
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

#[cfg(not(unix))]
fn read_exact_at(file: &File, destination: &mut [u8], offset: u64) -> io::Result<()> {
    let mut cloned = file.try_clone()?;
    cloned.seek(SeekFrom::Start(offset))?;
    cloned.read_exact(destination)
}

fn skip_metadata_value<R: Read + Seek>(
    input: &mut BoundedReader<'_, R>,
    kind: u32,
    kind_offset: u64,
) -> Result<(), GgufError> {
    let fixed_size = match kind {
        0 | 1 => 1,
        2 | 3 => 2,
        4..=6 => 4,
        10..=12 => 8,
        7 => {
            let offset = input.position;
            let value = input.read_u8("BOOL metadata array element")?;
            if value > 1 {
                return Err(GgufError::InvalidBoolean { value, offset });
            }
            return Ok(());
        }
        8 => {
            input.read_string("metadata array string", MAX_VALUE_STRING_BYTES)?;
            return Ok(());
        }
        9 => return Err(GgufError::NestedArray(kind_offset)),
        _ => {
            return Err(GgufError::UnknownMetadataType {
                kind,
                offset: kind_offset,
            });
        }
    };

    input.skip(fixed_size, "metadata array element")?;
    Ok(())
}

fn tensor_byte_len(tensor: &str, dimensions: &[u64], kind: TensorType) -> Result<u64, GgufError> {
    let element_count = dimensions.iter().try_fold(1_u64, |total, dimension| {
        total
            .checked_mul(*dimension)
            .ok_or(GgufError::Overflow("tensor element count"))
    })?;
    let (block_elements, block_bytes) = kind.layout();
    if let Some(first) = dimensions.first()
        && first % block_elements != 0
    {
        return Err(GgufError::InvalidBlockShape {
            tensor: tensor.to_owned(),
            size: *first,
            block: block_elements,
            kind: kind.name(),
        });
    }
    let blocks = element_count / block_elements;
    blocks
        .checked_mul(block_bytes)
        .ok_or(GgufError::Overflow("tensor byte length"))
}

fn align_up(value: u64, alignment: u64) -> Result<u64, GgufError> {
    let mask = alignment - 1;
    value
        .checked_add(mask)
        .map(|sum| sum & !mask)
        .ok_or(GgufError::Overflow("GGUF data alignment"))
}

fn validate_non_overlapping(tensors: &[TensorInfo]) -> Result<(), GgufError> {
    let mut ordered: Vec<_> = tensors.iter().collect();
    ordered.sort_unstable_by_key(|tensor| tensor.absolute_offset);
    let mut previous_end = 0_u64;
    for tensor in ordered {
        if tensor.absolute_offset < previous_end {
            return Err(GgufError::OverlappingTensor {
                tensor: tensor.name.clone(),
                offset: tensor.absolute_offset,
            });
        }
        previous_end = tensor.absolute_offset + tensor.byte_len;
    }
    Ok(())
}

struct BoundedReader<'a, R> {
    inner: &'a mut R,
    file_len: u64,
    position: u64,
}

impl<'a, R: Read + Seek> BoundedReader<'a, R> {
    fn new(inner: &'a mut R, file_len: u64) -> Self {
        Self {
            inner,
            file_len,
            position: 0,
        }
    }

    fn read_array<const N: usize>(&mut self, field: &'static str) -> Result<[u8; N], GgufError> {
        self.ensure_available(N as u64, field)?;
        let mut bytes = [0_u8; N];
        self.inner.read_exact(&mut bytes).map_err(|error| {
            if error.kind() == io::ErrorKind::UnexpectedEof {
                GgufError::Truncated {
                    offset: self.position,
                    field,
                }
            } else {
                GgufError::Io(error)
            }
        })?;
        self.position += N as u64;
        Ok(bytes)
    }

    fn read_u8(&mut self, field: &'static str) -> Result<u8, GgufError> {
        Ok(self.read_array::<1>(field)?[0])
    }

    fn read_u16(&mut self, field: &'static str) -> Result<u16, GgufError> {
        Ok(u16::from_le_bytes(self.read_array(field)?))
    }

    fn read_u32(&mut self, field: &'static str) -> Result<u32, GgufError> {
        Ok(u32::from_le_bytes(self.read_array(field)?))
    }

    fn read_u64(&mut self, field: &'static str) -> Result<u64, GgufError> {
        Ok(u64::from_le_bytes(self.read_array(field)?))
    }

    fn read_string(&mut self, field: &'static str, limit: u64) -> Result<String, GgufError> {
        let length = self.read_u64(field)?;
        if length > limit {
            return Err(GgufError::LengthLimit {
                field,
                length,
                limit,
            });
        }
        let length_usize =
            usize::try_from(length).map_err(|_| GgufError::Overflow("string allocation length"))?;
        self.ensure_available(length, field)?;
        let offset = self.position;
        let mut bytes = vec![0_u8; length_usize];
        self.inner.read_exact(&mut bytes).map_err(|error| {
            if error.kind() == io::ErrorKind::UnexpectedEof {
                GgufError::Truncated {
                    offset: self.position,
                    field,
                }
            } else {
                GgufError::Io(error)
            }
        })?;
        self.position += length;
        String::from_utf8(bytes).map_err(|_| GgufError::InvalidUtf8 { field, offset })
    }

    fn skip(&mut self, length: u64, field: &'static str) -> Result<(), GgufError> {
        self.ensure_available(length, field)?;
        let target = self
            .position
            .checked_add(length)
            .ok_or(GgufError::Overflow("reader position"))?;
        self.inner.seek(SeekFrom::Start(target))?;
        self.position = target;
        Ok(())
    }

    fn ensure_available(&self, length: u64, field: &'static str) -> Result<(), GgufError> {
        let end = self
            .position
            .checked_add(length)
            .ok_or(GgufError::Overflow("input byte range"))?;
        if end > self.file_len {
            return Err(GgufError::Truncated {
                offset: self.position,
                field,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashSet,
        io::{Cursor, Write},
        sync::Arc,
        thread,
    };

    use super::{GgufError, GgufFile, MetadataArray, MetadataValue, TensorSource, TensorType};

    const I32: u32 = 5;
    const U32: u32 = 4;
    const STRING: u32 = 8;
    const ARRAY: u32 = 9;

    struct Fixture {
        bytes: Vec<u8>,
    }

    impl Fixture {
        fn valid() -> Self {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(b"GGUF");
            put_u32(&mut bytes, 3);
            put_u64(&mut bytes, 1);
            put_u64(&mut bytes, 4);
            put_string_metadata(&mut bytes, "general.architecture", "gemma4");
            put_u32_metadata(&mut bytes, "general.file_type", 15);
            put_u32_metadata(&mut bytes, "general.alignment", 32);
            put_string_metadata(&mut bytes, "general.name", "fixture");
            put_string(&mut bytes, "weight");
            put_u32(&mut bytes, 2);
            put_u64(&mut bytes, 32);
            put_u64(&mut bytes, 1);
            put_u32(&mut bytes, 2);
            put_u64(&mut bytes, 0);
            pad_to(&mut bytes, 32);
            bytes.extend(0_u8..18);
            Self { bytes }
        }

        fn parse(&self) -> Result<GgufFile, GgufError> {
            let mut cursor = Cursor::new(&self.bytes);
            GgufFile::read_from(&mut cursor, self.bytes.len() as u64, "fixture.gguf")
        }

        fn tensor_descriptor_start(&self) -> usize {
            let needle = 6_u64.to_le_bytes();
            self.bytes
                .windows(needle.len())
                .rposition(|window| window == needle)
                .expect("tensor name length")
        }
    }

    #[test]
    fn parses_valid_metadata_and_tensor_bounds() {
        let fixture = Fixture::valid();
        let parsed = fixture.parse().unwrap();

        assert_eq!(parsed.version, 3);
        assert_eq!(parsed.alignment, 32);
        assert_eq!(parsed.architecture(), Some("gemma4"));
        assert_eq!(parsed.name(), Some("fixture"));
        assert_eq!(parsed.file_type(), Some(15));
        assert_eq!(parsed.tensors.len(), 1);
        assert_eq!(parsed.tensors[0].kind, TensorType::Q4_0);
        assert_eq!(parsed.tensors[0].dimensions, [32, 1]);
        assert_eq!(parsed.tensors[0].byte_len, 18);
        assert_eq!(
            parsed.metadata.get("general.file_type"),
            Some(&MetadataValue::U32(15))
        );
    }

    #[test]
    fn rejects_malformed_and_truncated_metadata() {
        let mut bad_magic = Fixture::valid();
        bad_magic.bytes[0] = b'B';
        assert!(matches!(
            bad_magic.parse(),
            Err(GgufError::InvalidMagic { .. })
        ));

        let mut truncated = Fixture::valid();
        truncated.bytes.truncate(28);
        assert!(matches!(
            truncated.parse(),
            Err(GgufError::Truncated { .. })
        ));

        let mut excessive_count = Fixture::valid();
        excessive_count.bytes[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(matches!(
            excessive_count.parse(),
            Err(GgufError::CountLimit {
                field: "metadata",
                ..
            })
        ));
    }

    #[test]
    fn rejects_invalid_alignment() {
        let mut fixture = Fixture::valid();
        let key = b"general.alignment";
        let key_position = fixture
            .bytes
            .windows(key.len())
            .position(|window| window == key)
            .unwrap();
        let value_position = key_position + key.len() + 4;
        fixture.bytes[value_position..value_position + 4].copy_from_slice(&3_u32.to_le_bytes());
        assert!(matches!(
            fixture.parse(),
            Err(GgufError::InvalidAlignment(3))
        ));
    }

    #[test]
    fn rejects_tensor_math_overflow() {
        let mut fixture = Fixture::valid();
        let descriptor = fixture.tensor_descriptor_start();
        let second_dimension = descriptor + 8 + 6 + 4 + 8;
        fixture.bytes[second_dimension..second_dimension + 8]
            .copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(matches!(fixture.parse(), Err(GgufError::Overflow(_))));
    }

    #[test]
    fn rejects_misaligned_and_out_of_bounds_tensor_ranges() {
        let fixture = Fixture::valid();
        let descriptor = fixture.tensor_descriptor_start();
        let offset_position = descriptor + 8 + 6 + 4 + 16 + 4;

        let mut misaligned = Fixture {
            bytes: fixture.bytes.clone(),
        };
        misaligned.bytes[offset_position..offset_position + 8]
            .copy_from_slice(&1_u64.to_le_bytes());
        assert!(matches!(
            misaligned.parse(),
            Err(GgufError::MisalignedTensor { .. })
        ));

        let mut out_of_bounds = fixture;
        out_of_bounds.bytes.truncate(out_of_bounds.bytes.len() - 1);
        assert!(matches!(
            out_of_bounds.parse(),
            Err(GgufError::TensorOutOfBounds { .. })
        ));
    }

    #[test]
    fn rejects_unknown_tensor_types_and_bad_quantized_shapes() {
        let fixture = Fixture::valid();
        let descriptor = fixture.tensor_descriptor_start();
        let type_position = descriptor + 8 + 6 + 4 + 16;

        let mut unknown = Fixture {
            bytes: fixture.bytes.clone(),
        };
        unknown.bytes[type_position..type_position + 4].copy_from_slice(&999_u32.to_le_bytes());
        assert!(matches!(
            unknown.parse(),
            Err(GgufError::UnsupportedTensorType { kind: 999, .. })
        ));

        let mut bad_shape = fixture;
        let first_dimension = descriptor + 8 + 6 + 4;
        bad_shape.bytes[first_dimension..first_dimension + 8]
            .copy_from_slice(&31_u64.to_le_bytes());
        assert!(matches!(
            bad_shape.parse(),
            Err(GgufError::InvalidBlockShape { .. })
        ));
    }

    #[test]
    fn rejects_truncated_array_values_without_allocating_the_array() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        put_u32(&mut bytes, 3);
        put_u64(&mut bytes, 0);
        put_u64(&mut bytes, 1);
        put_string(&mut bytes, "tokenizer.ggml.tokens");
        put_u32(&mut bytes, 9);
        put_u32(&mut bytes, STRING);
        put_u64(&mut bytes, 1);
        put_u64(&mut bytes, 100);

        let mut cursor = Cursor::new(&bytes);
        let parsed = GgufFile::read_from(&mut cursor, bytes.len() as u64, "array.gguf");
        assert!(matches!(parsed, Err(GgufError::Truncated { .. })));
    }

    #[test]
    fn captures_only_requested_metadata_arrays() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        put_u32(&mut bytes, 3);
        put_u64(&mut bytes, 0);
        put_u64(&mut bytes, 2);
        put_string_array_metadata(&mut bytes, "tokenizer.ggml.tokens", &["hello", " world"]);
        put_i32_array_metadata(&mut bytes, "tokenizer.ggml.token_type", &[1, 6]);
        pad_to(&mut bytes, 32);

        let mut summary_cursor = Cursor::new(&bytes);
        let summary =
            GgufFile::read_from(&mut summary_cursor, bytes.len() as u64, "tokenizer.gguf").unwrap();
        assert_eq!(
            summary.metadata.get("tokenizer.ggml.tokens"),
            Some(&MetadataValue::Array {
                element_type: STRING,
                length: 2,
            })
        );

        let requested = HashSet::from([
            "tokenizer.ggml.tokens".to_owned(),
            "tokenizer.ggml.token_type".to_owned(),
        ]);
        let mut captured_cursor = Cursor::new(&bytes);
        let captured = GgufFile::read_from_with_arrays(
            &mut captured_cursor,
            bytes.len() as u64,
            "tokenizer.gguf",
            &requested,
        )
        .unwrap();
        assert_eq!(
            captured
                .metadata
                .get("tokenizer.ggml.tokens")
                .and_then(MetadataValue::as_array)
                .and_then(MetadataArray::as_strings),
            Some(["hello".to_owned(), " world".to_owned()].as_slice())
        );
        assert_eq!(
            captured
                .metadata
                .get("tokenizer.ggml.token_type")
                .and_then(MetadataValue::as_array)
                .and_then(MetadataArray::as_i32s),
            Some([1, 6].as_slice())
        );
    }

    #[test]
    fn tensor_source_reads_checked_ranges_concurrently() {
        let fixture = Fixture::valid();
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&fixture.bytes).unwrap();
        file.flush().unwrap();

        let source = Arc::new(TensorSource::open(file.path()).unwrap());
        assert_eq!(source.path(), file.path());
        assert_eq!(source.tensor("weight").unwrap().byte_len, 18);

        let readers: Vec<_> = [(0_u64, vec![0, 1, 2, 3]), (10, vec![10, 11, 12, 13])]
            .into_iter()
            .map(|(offset, expected)| {
                let source = Arc::clone(&source);
                thread::spawn(move || {
                    let mut actual = vec![0_u8; expected.len()];
                    source
                        .read_tensor_at("weight", offset, &mut actual)
                        .unwrap();
                    assert_eq!(actual, expected);
                })
            })
            .collect();
        for reader in readers {
            reader.join().unwrap();
        }

        assert!(matches!(
            source.read_tensor_at("missing", 0, &mut [0]),
            Err(GgufError::TensorNotFound(_))
        ));
        assert!(matches!(
            source.read_tensor_at("weight", 17, &mut [0; 2]),
            Err(GgufError::TensorReadOutOfBounds { .. })
        ));
    }

    fn put_u32_metadata(bytes: &mut Vec<u8>, key: &str, value: u32) {
        put_string(bytes, key);
        put_u32(bytes, U32);
        put_u32(bytes, value);
    }

    fn put_string_metadata(bytes: &mut Vec<u8>, key: &str, value: &str) {
        put_string(bytes, key);
        put_u32(bytes, STRING);
        put_string(bytes, value);
    }

    fn put_string_array_metadata(bytes: &mut Vec<u8>, key: &str, values: &[&str]) {
        put_string(bytes, key);
        put_u32(bytes, ARRAY);
        put_u32(bytes, STRING);
        put_u64(bytes, values.len() as u64);
        for value in values {
            put_string(bytes, value);
        }
    }

    fn put_i32_array_metadata(bytes: &mut Vec<u8>, key: &str, values: &[i32]) {
        put_string(bytes, key);
        put_u32(bytes, ARRAY);
        put_u32(bytes, I32);
        put_u64(bytes, values.len() as u64);
        for value in values {
            put_u32(bytes, *value as u32);
        }
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

    fn pad_to(bytes: &mut Vec<u8>, alignment: usize) {
        let padding = (alignment - (bytes.len() % alignment)) % alignment;
        bytes.resize(bytes.len() + padding, 0);
    }
}
