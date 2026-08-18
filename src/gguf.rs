//! Bounded GGUF metadata inspection.
//!
//! This module intentionally stops at indexing tensor byte ranges. It never
//! reads tensor payloads, which keeps model discovery cheap even for very large
//! files.

use std::{
    collections::{BTreeMap, HashSet},
    fs::File,
    io::{self, Read, Seek, SeekFrom},
    path::Path,
};

const MAGIC: &[u8; 4] = b"GGUF";
const DEFAULT_ALIGNMENT: u32 = 32;
const MAX_METADATA_ENTRIES: u64 = 1_000_000;
const MAX_TENSORS: u64 = 1_000_000;
const MAX_KEY_BYTES: u64 = 64 * 1024;
const MAX_NAME_BYTES: u64 = 1024 * 1024;
const MAX_VALUE_STRING_BYTES: u64 = 64 * 1024 * 1024;
const MAX_ARRAY_ELEMENTS: u64 = 16_000_000;
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
    U64(u64),
    I64(i64),
    F64(f64),
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
            let value = read_metadata_value(&mut input, kind, kind_offset)?;
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
            for _ in 0..length {
                skip_metadata_value(input, element_type, element_offset)?;
            }
            MetadataValue::Array {
                element_type,
                length,
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
    use std::io::Cursor;

    use super::{GgufError, GgufFile, MetadataValue, TensorType};

    const U32: u32 = 4;
    const STRING: u32 = 8;

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
            bytes.extend_from_slice(&[0_u8; 18]);
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
