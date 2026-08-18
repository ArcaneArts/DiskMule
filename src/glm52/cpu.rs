//! Payload-lazy CPU tensor operations for GLM-5.2 safetensors weights.

use crate::safetensors::{SafeDtype, SafeTensorError, SafeTensorInfo, SafeTensorSource};

const FP8_BLOCK: usize = 128;

#[derive(Debug, thiserror::Error)]
pub enum Glm52CpuError {
    #[error(transparent)]
    SafeTensor(#[from] SafeTensorError),

    #[error("safetensors tensor index {index} is out of bounds for {count} tensors")]
    TensorIndex { index: usize, count: usize },

    #[error("tensor {tensor:?} has rank {actual}; expected rank {expected}")]
    TensorRank {
        tensor: String,
        expected: usize,
        actual: usize,
    },

    #[error("tensor {tensor:?} uses non-floating CPU dtype {dtype:?}")]
    TensorDtype { tensor: String, dtype: SafeDtype },

    #[error("tensor {tensor:?} dimension {dimension} is too large for this platform")]
    Dimension { tensor: String, dimension: u64 },

    #[error("tensor {tensor:?} row {row} is out of bounds for {rows} rows")]
    RowOutOfBounds {
        tensor: String,
        row: usize,
        rows: usize,
    },

    #[error("tensor {tensor:?} input has length {actual}; expected {expected}")]
    InputLength {
        tensor: String,
        expected: usize,
        actual: usize,
    },

    #[error("tensor {tensor:?} output has length {actual}; expected {expected}")]
    OutputLength {
        tensor: String,
        expected: usize,
        actual: usize,
    },

    #[error("FP8 tensor {tensor:?} is missing its validated scale tensor {scale:?}")]
    MissingFp8Scale { tensor: String, scale: String },

    #[error("FP8 tensor {tensor:?} contains a NaN E4M3FN value at column {column}")]
    InvalidFp8Value { tensor: String, column: usize },

    #[error("integer arithmetic overflow while reading tensor {0:?}")]
    Overflow(String),
}

/// Reads and computes directly from safetensors ranges without materializing a shard.
pub struct TensorReader<'a> {
    source: &'a SafeTensorSource,
}

impl<'a> TensorReader<'a> {
    pub const fn new(source: &'a SafeTensorSource) -> Self {
        Self { source }
    }

    pub fn vector(&self, tensor_index: usize) -> Result<Vec<f32>, Glm52CpuError> {
        let tensor = self.tensor(tensor_index)?;
        require_rank(tensor, 1)?;
        let length = dimension(tensor, tensor.shape[0])?;
        let mut values = vec![0.0; length];
        self.read_dense_values(tensor, 0, &mut values)?;
        Ok(values)
    }

    pub fn matrix_row(&self, tensor_index: usize, row: usize) -> Result<Vec<f32>, Glm52CpuError> {
        let tensor = self.tensor(tensor_index)?;
        require_rank(tensor, 2)?;
        let columns = dimension(tensor, tensor.shape[1])?;
        let mut values = vec![0.0; columns];
        self.matrix_row_into(tensor_index, row, &mut values)?;
        Ok(values)
    }

    pub fn matrix_row_into(
        &self,
        tensor_index: usize,
        row: usize,
        output: &mut [f32],
    ) -> Result<(), Glm52CpuError> {
        let tensor = self.tensor(tensor_index)?;
        require_rank(tensor, 2)?;
        let rows = dimension(tensor, tensor.shape[0])?;
        let columns = dimension(tensor, tensor.shape[1])?;
        if row >= rows {
            return Err(Glm52CpuError::RowOutOfBounds {
                tensor: tensor.name.clone(),
                row,
                rows,
            });
        }
        if output.len() != columns {
            return Err(Glm52CpuError::OutputLength {
                tensor: tensor.name.clone(),
                expected: columns,
                actual: output.len(),
            });
        }
        let offset = row
            .checked_mul(columns)
            .ok_or_else(|| Glm52CpuError::Overflow(tensor.name.clone()))?;
        match tensor.dtype {
            SafeDtype::F8E4M3 => self.read_fp8_row(tensor, row, output),
            _ => self.read_dense_values(tensor, offset, output),
        }
    }

    pub fn matvec(&self, tensor_index: usize, input: &[f32]) -> Result<Vec<f32>, Glm52CpuError> {
        let tensor = self.tensor(tensor_index)?;
        require_rank(tensor, 2)?;
        let rows = dimension(tensor, tensor.shape[0])?;
        let mut output = vec![0.0; rows];
        self.matvec_into(tensor_index, input, &mut output)?;
        Ok(output)
    }

    pub fn matvec_into(
        &self,
        tensor_index: usize,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), Glm52CpuError> {
        let tensor = self.tensor(tensor_index)?;
        require_rank(tensor, 2)?;
        let rows = dimension(tensor, tensor.shape[0])?;
        let columns = dimension(tensor, tensor.shape[1])?;
        if input.len() != columns {
            return Err(Glm52CpuError::InputLength {
                tensor: tensor.name.clone(),
                expected: columns,
                actual: input.len(),
            });
        }
        if output.len() != rows {
            return Err(Glm52CpuError::OutputLength {
                tensor: tensor.name.clone(),
                expected: rows,
                actual: output.len(),
            });
        }
        let mut row = vec![0.0; columns];
        for (row_index, result) in output.iter_mut().enumerate() {
            self.matrix_row_into(tensor_index, row_index, &mut row)?;
            *result = row
                .iter()
                .zip(input)
                .map(|(weight, value)| weight * value)
                .sum();
        }
        Ok(())
    }

    fn tensor(&self, index: usize) -> Result<&SafeTensorInfo, Glm52CpuError> {
        self.source
            .index
            .tensors()
            .get(index)
            .ok_or(Glm52CpuError::TensorIndex {
                index,
                count: self.source.index.tensors().len(),
            })
    }

    fn read_dense_values(
        &self,
        tensor: &SafeTensorInfo,
        element_offset: usize,
        output: &mut [f32],
    ) -> Result<(), Glm52CpuError> {
        let width = match tensor.dtype {
            SafeDtype::F32 => 4,
            SafeDtype::F16 | SafeDtype::Bf16 => 2,
            dtype => {
                return Err(Glm52CpuError::TensorDtype {
                    tensor: tensor.name.clone(),
                    dtype,
                });
            }
        };
        let byte_offset = element_offset
            .checked_mul(width)
            .ok_or_else(|| Glm52CpuError::Overflow(tensor.name.clone()))?;
        let byte_length = output
            .len()
            .checked_mul(width)
            .ok_or_else(|| Glm52CpuError::Overflow(tensor.name.clone()))?;
        let mut bytes = vec![0_u8; byte_length];
        self.source.read_tensor_at(
            &tensor.name,
            u64::try_from(byte_offset).map_err(|_| Glm52CpuError::Overflow(tensor.name.clone()))?,
            &mut bytes,
        )?;
        match tensor.dtype {
            SafeDtype::F32 => {
                for (value, bytes) in output.iter_mut().zip(bytes.chunks_exact(4)) {
                    *value = f32::from_le_bytes(bytes.try_into().expect("four-byte chunk"));
                }
            }
            SafeDtype::F16 => {
                for (value, bytes) in output.iter_mut().zip(bytes.chunks_exact(2)) {
                    *value = f16_to_f32(u16::from_le_bytes(
                        bytes.try_into().expect("two-byte chunk"),
                    ));
                }
            }
            SafeDtype::Bf16 => {
                for (value, bytes) in output.iter_mut().zip(bytes.chunks_exact(2)) {
                    let bits = u16::from_le_bytes(bytes.try_into().expect("two-byte chunk"));
                    *value = f32::from_bits(u32::from(bits) << 16);
                }
            }
            _ => unreachable!("dtype checked before tensor read"),
        }
        Ok(())
    }

    fn read_fp8_row(
        &self,
        tensor: &SafeTensorInfo,
        row: usize,
        output: &mut [f32],
    ) -> Result<(), Glm52CpuError> {
        let mut bytes = vec![0_u8; output.len()];
        let row_offset = row
            .checked_mul(output.len())
            .ok_or_else(|| Glm52CpuError::Overflow(tensor.name.clone()))?;
        self.source.read_tensor_at(
            &tensor.name,
            u64::try_from(row_offset).map_err(|_| Glm52CpuError::Overflow(tensor.name.clone()))?,
            &mut bytes,
        )?;

        let scale_name = format!("{}_scale_inv", tensor.name);
        let scale = self.source.index.tensor(&scale_name).ok_or_else(|| {
            Glm52CpuError::MissingFp8Scale {
                tensor: tensor.name.clone(),
                scale: scale_name.clone(),
            }
        })?;
        let scale_columns = output.len().div_ceil(FP8_BLOCK);
        if scale.dtype != SafeDtype::F32
            || scale.shape
                != [
                    tensor.shape[0].div_ceil(FP8_BLOCK as u64),
                    scale_columns as u64,
                ]
        {
            return Err(Glm52CpuError::MissingFp8Scale {
                tensor: tensor.name.clone(),
                scale: scale_name,
            });
        }
        let scale_offset = (row / FP8_BLOCK)
            .checked_mul(scale_columns)
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| Glm52CpuError::Overflow(scale.name.clone()))?;
        let mut scale_bytes = vec![0_u8; scale_columns * 4];
        self.source.read_tensor_at(
            &scale.name,
            u64::try_from(scale_offset).map_err(|_| Glm52CpuError::Overflow(scale.name.clone()))?,
            &mut scale_bytes,
        )?;
        let scales = scale_bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
            .collect::<Vec<_>>();
        for (column, (output, value)) in output.iter_mut().zip(bytes).enumerate() {
            let decoded =
                decode_f8_e4m3fn(value).ok_or_else(|| Glm52CpuError::InvalidFp8Value {
                    tensor: tensor.name.clone(),
                    column,
                })?;
            *output = decoded * scales[column / FP8_BLOCK];
        }
        Ok(())
    }
}

/// Decodes the finite-only E4M3 format used by GLM-5.2 safetensors.
pub fn decode_f8_e4m3fn(value: u8) -> Option<f32> {
    let sign = if value & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = (value >> 3) & 0x0f;
    let fraction = value & 0x07;
    if exponent == 0x0f && fraction == 0x07 {
        return None;
    }
    let magnitude = if exponent == 0 {
        f32::from(fraction) * 2.0_f32.powi(-9)
    } else {
        (1.0 + f32::from(fraction) / 8.0) * 2.0_f32.powi(i32::from(exponent) - 7)
    };
    Some(sign * magnitude)
}

fn require_rank(tensor: &SafeTensorInfo, expected: usize) -> Result<(), Glm52CpuError> {
    if tensor.shape.len() == expected {
        Ok(())
    } else {
        Err(Glm52CpuError::TensorRank {
            tensor: tensor.name.clone(),
            expected,
            actual: tensor.shape.len(),
        })
    }
}

fn dimension(tensor: &SafeTensorInfo, value: u64) -> Result<usize, Glm52CpuError> {
    usize::try_from(value).map_err(|_| Glm52CpuError::Dimension {
        tensor: tensor.name.clone(),
        dimension: value,
    })
}

fn f16_to_f32(value: u16) -> f32 {
    let sign = u32::from(value & 0x8000) << 16;
    let exponent = (value >> 10) & 0x1f;
    let fraction = u32::from(value & 0x03ff);
    match exponent {
        0 if fraction == 0 => f32::from_bits(sign),
        0 => {
            let magnitude = fraction as f32 * 2.0_f32.powi(-24);
            if sign == 0 { magnitude } else { -magnitude }
        }
        0x1f => f32::from_bits(sign | 0x7f80_0000 | (fraction << 13)),
        _ => f32::from_bits(sign | (u32::from(exponent) + 112) << 23 | fraction << 13),
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, io::Write, path::Path};

    use serde_json::{Map, json};
    use tempfile::TempDir;

    use crate::safetensors::SafeTensorSource;

    use super::{Glm52CpuError, TensorReader, decode_f8_e4m3fn};

    #[test]
    fn decodes_e4m3fn_boundaries() {
        assert_eq!(decode_f8_e4m3fn(0x00), Some(0.0));
        assert_eq!(decode_f8_e4m3fn(0x01), Some(2.0_f32.powi(-9)));
        assert_eq!(decode_f8_e4m3fn(0x38), Some(1.0));
        assert_eq!(decode_f8_e4m3fn(0xb8), Some(-1.0));
        assert_eq!(decode_f8_e4m3fn(0x7e), Some(448.0));
        assert_eq!(decode_f8_e4m3fn(0x7f), None);
        assert_eq!(decode_f8_e4m3fn(0xff), None);
    }

    #[test]
    fn reads_dense_vectors_rows_and_matvecs() {
        let temp = TempDir::new().unwrap();
        let f32_values = [1.0_f32, 2.0, 3.0, 4.0, -1.0, 0.5];
        let f32_bytes = f32_values
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let f16_bytes = [0x3c00_u16, 0xc000]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();
        let bf16_bytes = [1.5_f32, -0.25]
            .into_iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect::<Vec<_>>();
        write_safetensors(
            &temp.path().join("model.safetensors"),
            &[
                ("matrix", "F32", vec![2, 3], f32_bytes),
                ("half", "F16", vec![2], f16_bytes),
                ("brain", "BF16", vec![2], bf16_bytes),
            ],
        );
        let source = SafeTensorSource::open(temp.path()).unwrap();
        let reader = TensorReader::new(&source);
        let matrix = tensor_index(&source, "matrix");
        assert_eq!(reader.matrix_row(matrix, 1).unwrap(), [4.0, -1.0, 0.5]);
        assert_eq!(
            reader.matvec(matrix, &[2.0, -1.0, 4.0]).unwrap(),
            [12.0, 11.0]
        );
        assert_eq!(
            reader.vector(tensor_index(&source, "half")).unwrap(),
            [1.0, -2.0]
        );
        assert_eq!(
            reader.vector(tensor_index(&source, "brain")).unwrap(),
            [1.5, -0.25]
        );
    }

    #[test]
    fn applies_fp8_scales_per_128_by_128_block() {
        let temp = TempDir::new().unwrap();
        let mut weights = vec![0x38_u8; 129 * 130];
        weights[128 * 130 + 129] = 0xb8;
        let scales = [2.0_f32, 3.0, 5.0, 7.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        write_safetensors(
            &temp.path().join("model.safetensors"),
            &[
                ("weight", "F8_E4M3", vec![129, 130], weights),
                ("weight_scale_inv", "F32", vec![2, 2], scales),
            ],
        );
        let source = SafeTensorSource::open(temp.path()).unwrap();
        let reader = TensorReader::new(&source);
        let index = tensor_index(&source, "weight");
        let row = reader.matrix_row(index, 128).unwrap();
        assert_eq!(row[0], 5.0);
        assert_eq!(row[127], 5.0);
        assert_eq!(row[128], 7.0);
        assert_eq!(row[129], -7.0);

        let mut short = [0.0; 2];
        assert!(matches!(
            reader.matrix_row_into(index, 0, &mut short),
            Err(Glm52CpuError::OutputLength { .. })
        ));
    }

    fn tensor_index(source: &SafeTensorSource, name: &str) -> usize {
        source
            .index
            .tensors()
            .iter()
            .position(|tensor| tensor.name == name)
            .unwrap()
    }

    fn write_safetensors(path: &Path, tensors: &[(&str, &str, Vec<u64>, Vec<u8>)]) {
        let mut header = Map::new();
        let mut payload = Vec::new();
        for (name, dtype, shape, bytes) in tensors {
            let start = payload.len();
            payload.extend(bytes);
            header.insert(
                (*name).to_owned(),
                json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [start, payload.len()]
                }),
            );
        }
        let header = serde_json::to_vec(&header).unwrap();
        let mut file = fs::File::create(path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.write_all(&payload).unwrap();
    }
}
