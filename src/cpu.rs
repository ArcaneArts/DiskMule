//! Scalar CPU reference operators.
//!
//! These prioritize transparent, deterministic arithmetic over throughput.
//! Metal kernels and future vectorized paths are compared against this module.

use rayon::prelude::*;

use crate::gguf::{TensorSource, TensorType};

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum CpuError {
    #[error("CPU dequantization does not support {0}")]
    UnsupportedType(&'static str),

    #[error(
        "{kind} output length {elements} is not divisible by its {block_elements}-element block"
    )]
    InvalidElementCount {
        kind: &'static str,
        elements: usize,
        block_elements: u64,
    },

    #[error("{kind} requires {expected} encoded bytes for {elements} values, got {actual}")]
    InvalidByteLength {
        kind: &'static str,
        elements: usize,
        expected: usize,
        actual: usize,
    },

    #[error("matrix dimensions or encoded byte length overflowed")]
    Overflow,

    #[error("matvec input has {actual} columns; expected {expected}")]
    InvalidInputLength { expected: usize, actual: usize },

    #[error("matvec output has {actual} rows; expected {expected}")]
    InvalidOutputLength { expected: usize, actual: usize },

    #[error("{operation} requires a non-empty input")]
    EmptyInput { operation: &'static str },

    #[error("{operation} {argument} has length {actual}; expected {expected}")]
    InvalidVectorLength {
        operation: &'static str,
        argument: &'static str,
        expected: usize,
        actual: usize,
    },

    #[error("{operation} received a non-finite value")]
    NonFinite { operation: &'static str },

    #[error("top-k count {k} is outside 1..={length}")]
    InvalidTopK { k: usize, length: usize },

    #[error(
        "NeoX RoPE input length {length} does not match {heads} heads of dimension {head_dimension}"
    )]
    InvalidRopeShape {
        length: usize,
        heads: usize,
        head_dimension: usize,
    },

    #[error("NeoX RoPE requires an even head dimension and at most half its dimensions as pairs")]
    InvalidRopePairs,

    #[error("{operation} requires a finite positive parameter")]
    InvalidPositiveParameter { operation: &'static str },

    #[error("tensor index {index} is outside the {count}-tensor GGUF index")]
    InvalidTensorIndex { index: usize, count: usize },

    #[error("tensor {tensor:?} has rank {actual}; expected rank {expected}")]
    InvalidTensorRank {
        tensor: String,
        expected: usize,
        actual: usize,
    },

    #[error("tensor {tensor:?} row {row} is outside its {rows} rows")]
    InvalidTensorRow {
        tensor: String,
        row: usize,
        rows: usize,
    },

    #[error("tensor {tensor:?} expert {expert} is outside its {experts} experts")]
    InvalidTensorExpert {
        tensor: String,
        expert: usize,
        experts: usize,
    },
}

/// FP32 RMS normalization with an optional learned per-feature weight.
pub fn rms_norm(
    input: &[f32],
    weight: Option<&[f32]>,
    epsilon: f32,
    output: &mut [f32],
) -> Result<(), CpuError> {
    if input.is_empty() {
        return Err(CpuError::EmptyInput {
            operation: "RMSNorm",
        });
    }
    validate_vector_length("RMSNorm", "output", input.len(), output.len())?;
    if let Some(weight) = weight {
        validate_vector_length("RMSNorm", "weight", input.len(), weight.len())?;
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(CpuError::InvalidPositiveParameter {
            operation: "RMSNorm epsilon",
        });
    }

    let sum_squares = input
        .iter()
        .fold(0.0_f32, |sum, value| value.mul_add(*value, sum));
    let inverse_rms = (sum_squares / input.len() as f32 + epsilon).sqrt().recip();
    if !inverse_rms.is_finite() {
        return Err(CpuError::NonFinite {
            operation: "RMSNorm",
        });
    }
    match weight {
        Some(weight) => {
            for ((output, input), weight) in output.iter_mut().zip(input).zip(weight) {
                *output = input * weight * inverse_rms;
            }
        }
        None => {
            for (output, input) in output.iter_mut().zip(input) {
                *output = input * inverse_rms;
            }
        }
    }
    Ok(())
}

/// Gemma 4's `gelu_pytorch_tanh` activation.
pub fn gelu_tanh(value: f32) -> f32 {
    const SQRT_TWO_OVER_PI: f32 = 0.797_884_6;
    const CUBIC: f32 = 0.044_715;
    0.5 * value * (1.0 + (SQRT_TWO_OVER_PI * (value + CUBIC * value.powi(3))).tanh())
}

/// Apply NeoX-convention rotary embeddings to contiguous attention heads.
pub fn rope_neox_in_place(
    values: &mut [f32],
    heads: usize,
    head_dimension: usize,
    rotated_pairs: usize,
    position: usize,
    theta: f32,
) -> Result<(), CpuError> {
    if heads
        .checked_mul(head_dimension)
        .filter(|length| *length == values.len())
        .is_none()
    {
        return Err(CpuError::InvalidRopeShape {
            length: values.len(),
            heads,
            head_dimension,
        });
    }
    if !head_dimension.is_multiple_of(2) || rotated_pairs > head_dimension / 2 {
        return Err(CpuError::InvalidRopePairs);
    }
    if !theta.is_finite() || theta <= 0.0 {
        return Err(CpuError::InvalidPositiveParameter {
            operation: "NeoX RoPE theta",
        });
    }

    let half_dimension = head_dimension / 2;
    for head in values.chunks_exact_mut(head_dimension) {
        for pair in 0..rotated_pairs {
            let exponent = -((2 * pair) as f32) / head_dimension as f32;
            let angle = position as f32 * theta.powf(exponent);
            let (sin, cos) = angle.sin_cos();
            let first = head[pair];
            let second = head[half_dimension + pair];
            head[pair] = first * cos - second * sin;
            head[half_dimension + pair] = first * sin + second * cos;
        }
    }
    Ok(())
}

/// Numerically stable in-place softmax.
pub fn softmax_in_place(values: &mut [f32]) -> Result<(), CpuError> {
    if values.is_empty() {
        return Err(CpuError::EmptyInput {
            operation: "softmax",
        });
    }
    if values
        .iter()
        .any(|value| value.is_nan() || *value == f32::INFINITY)
    {
        return Err(CpuError::NonFinite {
            operation: "softmax",
        });
    }
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if maximum == f32::NEG_INFINITY {
        return Err(CpuError::NonFinite {
            operation: "softmax",
        });
    }
    let mut sum = 0.0_f32;
    for value in values.iter_mut() {
        *value = (*value - maximum).exp();
        sum += *value;
    }
    if !sum.is_finite() || sum <= 0.0 {
        return Err(CpuError::NonFinite {
            operation: "softmax",
        });
    }
    for value in values {
        *value /= sum;
    }
    Ok(())
}

/// Select the largest logits and normalize only the selected values.
/// Equal logits are ordered by ascending original index.
pub fn top_k_softmax(logits: &[f32], k: usize) -> Result<Vec<(usize, f32)>, CpuError> {
    if k == 0 || k > logits.len() {
        return Err(CpuError::InvalidTopK {
            k,
            length: logits.len(),
        });
    }
    if logits.iter().any(|value| !value.is_finite()) {
        return Err(CpuError::NonFinite {
            operation: "top-k softmax",
        });
    }
    let mut selected: Vec<_> = logits.iter().copied().enumerate().collect();
    selected.sort_unstable_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    selected.truncate(k);
    let mut probabilities: Vec<_> = selected.iter().map(|(_, value)| *value).collect();
    softmax_in_place(&mut probabilities)?;
    for ((_, value), probability) in selected.iter_mut().zip(probabilities) {
        *value = probability;
    }
    Ok(selected)
}

/// Apply Gemma's bounded final-logit transform in place.
pub fn softcap_in_place(values: &mut [f32], cap: f32) -> Result<(), CpuError> {
    if !cap.is_finite() || cap <= 0.0 {
        return Err(CpuError::InvalidPositiveParameter {
            operation: "logit softcap",
        });
    }
    for value in values {
        *value = cap * (*value / cap).tanh();
    }
    Ok(())
}

/// Return the first maximum, matching deterministic greedy decoding.
pub fn argmax(values: &[f32]) -> Result<usize, CpuError> {
    if values.is_empty() {
        return Err(CpuError::EmptyInput {
            operation: "argmax",
        });
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(CpuError::NonFinite {
            operation: "argmax",
        });
    }
    Ok(values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1).then_with(|| right.0.cmp(&left.0)))
        .expect("non-empty input")
        .0)
}

fn validate_vector_length(
    operation: &'static str,
    argument: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), CpuError> {
    if actual != expected {
        return Err(CpuError::InvalidVectorLength {
            operation,
            argument,
            expected,
            actual,
        });
    }
    Ok(())
}

/// Decode GGML tensor blocks into FP32 values.
pub fn dequantize(kind: TensorType, encoded: &[u8], output: &mut [f32]) -> Result<(), CpuError> {
    let block_elements = kind.block_elements();
    let elements = u64::try_from(output.len()).map_err(|_| CpuError::Overflow)?;
    let expected = kind
        .row_byte_len(elements)
        .ok_or(CpuError::InvalidElementCount {
            kind: kind.name(),
            elements: output.len(),
            block_elements,
        })?;
    let expected = usize::try_from(expected).map_err(|_| CpuError::Overflow)?;
    if encoded.len() != expected {
        return Err(CpuError::InvalidByteLength {
            kind: kind.name(),
            elements: output.len(),
            expected,
            actual: encoded.len(),
        });
    }

    match kind {
        TensorType::F32 => decode_f32(encoded, output),
        TensorType::F16 => decode_f16(encoded, output),
        TensorType::Q5_0 => decode_q5_0(encoded, output),
        TensorType::Q8_0 => decode_q8_0(encoded, output),
        TensorType::Q4K => decode_q4_k(encoded, output),
        TensorType::Q6K => decode_q6_k(encoded, output),
        _ => return Err(CpuError::UnsupportedType(kind.name())),
    }
    Ok(())
}

/// Reference matrix-vector multiplication for a row-major GGML tensor.
pub fn matvec(
    kind: TensorType,
    rows: usize,
    columns: usize,
    encoded: &[u8],
    input: &[f32],
    output: &mut [f32],
) -> Result<(), CpuError> {
    if input.len() != columns {
        return Err(CpuError::InvalidInputLength {
            expected: columns,
            actual: input.len(),
        });
    }
    if output.len() != rows {
        return Err(CpuError::InvalidOutputLength {
            expected: rows,
            actual: output.len(),
        });
    }
    let row_bytes = kind
        .row_byte_len(u64::try_from(columns).map_err(|_| CpuError::Overflow)?)
        .ok_or(CpuError::InvalidElementCount {
            kind: kind.name(),
            elements: columns,
            block_elements: kind.block_elements(),
        })?;
    let row_bytes = usize::try_from(row_bytes).map_err(|_| CpuError::Overflow)?;
    let matrix_bytes = rows.checked_mul(row_bytes).ok_or(CpuError::Overflow)?;
    if encoded.len() != matrix_bytes {
        return Err(CpuError::InvalidByteLength {
            kind: kind.name(),
            elements: rows.checked_mul(columns).ok_or(CpuError::Overflow)?,
            expected: matrix_bytes,
            actual: encoded.len(),
        });
    }

    encoded
        .par_chunks_exact(row_bytes)
        .zip(output.par_iter_mut())
        .try_for_each_init(
            || vec![0.0_f32; columns],
            |row, (encoded, result)| {
                dequantize(kind, encoded, row)?;
                *result = row
                    .iter()
                    .zip(input)
                    .fold(0.0_f32, |sum, (weight, value)| weight.mul_add(*value, sum));
                Ok::<(), CpuError>(())
            },
        )?;
    Ok(())
}

/// Decode a rank-one tensor from a mapped GGUF source.
pub fn tensor_vector(
    source: &TensorSource,
    tensor_index: usize,
    output: &mut [f32],
) -> Result<(), CpuError> {
    let tensor = tensor_at(source, tensor_index)?;
    require_rank(tensor.name.as_str(), tensor.dimensions.len(), 1)?;
    validate_vector_length(
        "tensor vector",
        "output",
        dimension(tensor.dimensions[0])?,
        output.len(),
    )?;
    let encoded =
        source
            .tensor_bytes_by_index(tensor_index)
            .ok_or(CpuError::InvalidTensorIndex {
                index: tensor_index,
                count: source.gguf.tensors.len(),
            })?;
    dequantize(tensor.kind, encoded, output)
}

/// Decode one row from a mapped rank-two tensor.
pub fn tensor_row(
    source: &TensorSource,
    tensor_index: usize,
    row: usize,
    output: &mut [f32],
) -> Result<(), CpuError> {
    let tensor = tensor_at(source, tensor_index)?;
    require_rank(tensor.name.as_str(), tensor.dimensions.len(), 2)?;
    let columns = dimension(tensor.dimensions[0])?;
    let rows = dimension(tensor.dimensions[1])?;
    if row >= rows {
        return Err(CpuError::InvalidTensorRow {
            tensor: tensor.name.clone(),
            row,
            rows,
        });
    }
    validate_vector_length("tensor row", "output", columns, output.len())?;
    let row_bytes = tensor
        .kind
        .row_byte_len(tensor.dimensions[0])
        .and_then(|length| usize::try_from(length).ok())
        .ok_or(CpuError::Overflow)?;
    let start = row.checked_mul(row_bytes).ok_or(CpuError::Overflow)?;
    let bytes = tensor_bytes(source, tensor_index)?;
    dequantize(tensor.kind, &bytes[start..start + row_bytes], output)
}

/// Apply a mapped rank-two GGUF matrix to one FP32 vector.
pub fn tensor_matvec(
    source: &TensorSource,
    tensor_index: usize,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), CpuError> {
    let tensor = tensor_at(source, tensor_index)?;
    require_rank(tensor.name.as_str(), tensor.dimensions.len(), 2)?;
    matvec(
        tensor.kind,
        dimension(tensor.dimensions[1])?,
        dimension(tensor.dimensions[0])?,
        tensor_bytes(source, tensor_index)?,
        input,
        output,
    )
}

/// Apply one contiguous expert matrix from a mapped rank-three GGUF tensor.
pub fn tensor_expert_matvec(
    source: &TensorSource,
    tensor_index: usize,
    expert: usize,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), CpuError> {
    let tensor = tensor_at(source, tensor_index)?;
    require_rank(tensor.name.as_str(), tensor.dimensions.len(), 3)?;
    let columns = dimension(tensor.dimensions[0])?;
    let rows = dimension(tensor.dimensions[1])?;
    let experts = dimension(tensor.dimensions[2])?;
    if expert >= experts {
        return Err(CpuError::InvalidTensorExpert {
            tensor: tensor.name.clone(),
            expert,
            experts,
        });
    }
    let row_bytes = tensor
        .kind
        .row_byte_len(tensor.dimensions[0])
        .and_then(|length| usize::try_from(length).ok())
        .ok_or(CpuError::Overflow)?;
    let expert_bytes = rows.checked_mul(row_bytes).ok_or(CpuError::Overflow)?;
    let start = expert.checked_mul(expert_bytes).ok_or(CpuError::Overflow)?;
    let bytes = tensor_bytes(source, tensor_index)?;
    matvec(
        tensor.kind,
        rows,
        columns,
        &bytes[start..start + expert_bytes],
        input,
        output,
    )
}

fn tensor_at(source: &TensorSource, index: usize) -> Result<&crate::gguf::TensorInfo, CpuError> {
    source
        .gguf
        .tensors
        .get(index)
        .ok_or(CpuError::InvalidTensorIndex {
            index,
            count: source.gguf.tensors.len(),
        })
}

fn tensor_bytes(source: &TensorSource, index: usize) -> Result<&[u8], CpuError> {
    source
        .tensor_bytes_by_index(index)
        .ok_or(CpuError::InvalidTensorIndex {
            index,
            count: source.gguf.tensors.len(),
        })
}

fn require_rank(tensor: &str, actual: usize, expected: usize) -> Result<(), CpuError> {
    if actual != expected {
        return Err(CpuError::InvalidTensorRank {
            tensor: tensor.to_owned(),
            expected,
            actual,
        });
    }
    Ok(())
}

fn dimension(value: u64) -> Result<usize, CpuError> {
    usize::try_from(value).map_err(|_| CpuError::Overflow)
}

fn decode_f32(encoded: &[u8], output: &mut [f32]) {
    for (bytes, value) in encoded.chunks_exact(4).zip(output) {
        *value = f32::from_le_bytes(bytes.try_into().expect("four-byte chunk"));
    }
}

fn decode_f16(encoded: &[u8], output: &mut [f32]) {
    for (bytes, value) in encoded.chunks_exact(2).zip(output) {
        *value = f16_to_f32(u16::from_le_bytes(
            bytes.try_into().expect("two-byte chunk"),
        ));
    }
}

fn decode_q5_0(encoded: &[u8], output: &mut [f32]) {
    for (block, values) in encoded.chunks_exact(22).zip(output.chunks_exact_mut(32)) {
        let delta = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let high_bits = u32::from_le_bytes(block[2..6].try_into().expect("four-byte high mask"));
        for index in 0..16 {
            let packed = block[6 + index];
            let low_high_bit = (((high_bits >> index) << 4) & 0x10) as u8;
            let high_high_bit = ((high_bits >> (index + 12)) & 0x10) as u8;
            let low = ((packed & 0x0f) | low_high_bit) as i32 - 16;
            let high = ((packed >> 4) | high_high_bit) as i32 - 16;
            values[index] = delta * low as f32;
            values[index + 16] = delta * high as f32;
        }
    }
}

fn decode_q8_0(encoded: &[u8], output: &mut [f32]) {
    for (block, values) in encoded.chunks_exact(34).zip(output.chunks_exact_mut(32)) {
        let delta = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for (quant, value) in block[2..].iter().zip(values) {
            *value = delta * (*quant as i8) as f32;
        }
    }
}

fn decode_q4_k(encoded: &[u8], output: &mut [f32]) {
    for (block, values) in encoded.chunks_exact(144).zip(output.chunks_exact_mut(256)) {
        let delta = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let minimum = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];
        let quants = &block[16..];
        for group_pair in 0..4 {
            let (scale_low, min_low) = scale_min_q4_k(group_pair * 2, scales);
            let (scale_high, min_high) = scale_min_q4_k(group_pair * 2 + 1, scales);
            let quant_offset = group_pair * 32;
            let value_offset = group_pair * 64;
            for index in 0..32 {
                let packed = quants[quant_offset + index];
                values[value_offset + index] =
                    delta * scale_low as f32 * (packed & 0x0f) as f32 - minimum * min_low as f32;
                values[value_offset + index + 32] =
                    delta * scale_high as f32 * (packed >> 4) as f32 - minimum * min_high as f32;
            }
        }
    }
}

fn scale_min_q4_k(index: usize, scales: &[u8]) -> (u8, u8) {
    if index < 4 {
        (scales[index] & 0x3f, scales[index + 4] & 0x3f)
    } else {
        (
            (scales[index + 4] & 0x0f) | ((scales[index - 4] >> 6) << 4),
            (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4),
        )
    }
}

fn decode_q6_k(encoded: &[u8], output: &mut [f32]) {
    for (block, values) in encoded.chunks_exact(210).zip(output.chunks_exact_mut(256)) {
        let low = &block[..128];
        let high = &block[128..192];
        let scales = &block[192..208];
        let delta = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
        for half in 0..2 {
            let low = &low[half * 64..];
            let high = &high[half * 32..];
            let scales = &scales[half * 8..];
            let values = &mut values[half * 128..];
            for index in 0..32 {
                let scale_pair = index / 16;
                let q1 = (low[index] & 0x0f) | ((high[index] & 0x03) << 4);
                let q2 = (low[index + 32] & 0x0f) | (((high[index] >> 2) & 0x03) << 4);
                let q3 = (low[index] >> 4) | (((high[index] >> 4) & 0x03) << 4);
                let q4 = (low[index + 32] >> 4) | (((high[index] >> 6) & 0x03) << 4);
                values[index] = delta * (scales[scale_pair] as i8) as f32 * (q1 as f32 - 32.0);
                values[index + 32] =
                    delta * (scales[scale_pair + 2] as i8) as f32 * (q2 as f32 - 32.0);
                values[index + 64] =
                    delta * (scales[scale_pair + 4] as i8) as f32 * (q3 as f32 - 32.0);
                values[index + 96] =
                    delta * (scales[scale_pair + 6] as i8) as f32 * (q4 as f32 - 32.0);
            }
        }
    }
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
    use super::{
        CpuError, argmax, dequantize, gelu_tanh, matvec, rms_norm, rope_neox_in_place,
        scale_min_q4_k, softcap_in_place, softmax_in_place, top_k_softmax,
    };
    use crate::gguf::TensorType;

    #[test]
    fn decodes_f32_and_f16_values() {
        let f32_bytes: Vec<_> = [1.25_f32, -2.5]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let mut values = [0.0; 2];
        dequantize(TensorType::F32, &f32_bytes, &mut values).unwrap();
        assert_eq!(values, [1.25, -2.5]);

        let f16_bytes: Vec<_> = [0x3c00_u16, 0xc000, 0x0001]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect();
        let mut values = [0.0; 3];
        dequantize(TensorType::F16, &f16_bytes, &mut values).unwrap();
        assert_eq!(values, [1.0, -2.0, 2.0_f32.powi(-24)]);
    }

    #[test]
    fn decodes_q5_0_high_bits_and_nibbles() {
        let mut block = vec![0_u8; 22];
        block[..2].copy_from_slice(&0x4000_u16.to_le_bytes());
        block[2..6].fill(0xff);
        block[6..].fill(0x10);
        let mut values = [0.0; 32];
        dequantize(TensorType::Q5_0, &block, &mut values).unwrap();
        assert_eq!(values[..16], [0.0; 16]);
        assert_eq!(values[16..], [2.0; 16]);
    }

    #[test]
    fn decodes_q8_0_signed_values() {
        let mut block = vec![0_u8; 34];
        block[..2].copy_from_slice(&0x3800_u16.to_le_bytes());
        for (index, quant) in block[2..].iter_mut().enumerate() {
            *quant = (index as i8 - 16) as u8;
        }
        let mut values = [0.0; 32];
        dequantize(TensorType::Q8_0, &block, &mut values).unwrap();
        for (index, value) in values.into_iter().enumerate() {
            assert_eq!(value, (index as f32 - 16.0) * 0.5);
        }
    }

    #[test]
    fn decodes_q4_k_group_scales_and_minima() {
        let mut block = vec![0_u8; 144];
        block[..2].copy_from_slice(&0x3800_u16.to_le_bytes());
        block[2..4].copy_from_slice(&0x3400_u16.to_le_bytes());
        block[4..8].fill(1);
        block[8..12].fill(2);
        block[12..16].fill(0x21);
        block[16..].fill(0x30);
        let mut values = [0.0; 256];
        dequantize(TensorType::Q4K, &block, &mut values).unwrap();
        for group in values.chunks_exact(64) {
            assert_eq!(group[..32], [-0.5; 32]);
            assert_eq!(group[32..], [1.0; 32]);
        }
    }

    #[test]
    fn unpacks_q4_k_high_scale_and_minimum_bits() {
        let mut scales = [0_u8; 12];
        scales[0] = 0b1100_0000;
        scales[4] = 0b1000_0000;
        scales[8] = 0b1010_0001;
        assert_eq!(scale_min_q4_k(4, &scales), (49, 42));
    }

    #[test]
    fn decodes_q6_k_six_bit_groups() {
        let mut block = vec![0_u8; 210];
        for (index, scale) in block[192..208].iter_mut().enumerate() {
            *scale = (index + 1) as u8;
        }
        block[208..].copy_from_slice(&0x3800_u16.to_le_bytes());
        let mut values = [0.0; 256];
        dequantize(TensorType::Q6K, &block, &mut values).unwrap();
        for (group_index, group) in values.chunks_exact(16).enumerate() {
            assert_eq!(group, [-16.0 * (group_index + 1) as f32; 16]);
        }
    }

    #[test]
    fn matvec_uses_row_major_tensor_layout() {
        let encoded: Vec<_> = [1.0_f32, 2.0, 3.0, 4.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let mut output = [0.0; 2];
        matvec(TensorType::F32, 2, 2, &encoded, &[0.5, -1.0], &mut output).unwrap();
        assert_eq!(output, [-1.5, -2.5]);
    }

    #[test]
    fn rejects_bad_lengths_and_unsupported_types() {
        assert!(matches!(
            dequantize(TensorType::Q4K, &[0; 144], &mut [0.0; 32]),
            Err(CpuError::InvalidElementCount { .. })
        ));
        assert!(matches!(
            dequantize(TensorType::Q5K, &[], &mut []),
            Err(CpuError::UnsupportedType("Q5_K"))
        ));
    }

    #[test]
    fn normalizes_weighted_and_unweighted_vectors() {
        let input = [3.0_f32, 4.0];
        let mut output = [0.0; 2];
        rms_norm(&input, None, 1e-6, &mut output).unwrap();
        let inverse_rms = (12.5_f32 + 1e-6).sqrt().recip();
        assert_eq!(output, [3.0 * inverse_rms, 4.0 * inverse_rms]);

        rms_norm(&input, Some(&[2.0, 0.5]), 1e-6, &mut output).unwrap();
        assert_eq!(output, [6.0 * inverse_rms, 2.0 * inverse_rms]);
        assert!(matches!(
            rms_norm(&input, Some(&[1.0]), 1e-6, &mut output),
            Err(CpuError::InvalidVectorLength {
                argument: "weight",
                ..
            })
        ));
    }

    #[test]
    fn applies_gemma_gelu_and_logit_softcap() {
        assert_eq!(gelu_tanh(0.0), 0.0);
        assert!((gelu_tanh(1.0) - 0.841_192).abs() < 1e-6);
        let mut logits = [-60.0_f32, 0.0, 60.0];
        softcap_in_place(&mut logits, 30.0).unwrap();
        assert!((logits[0] + 28.920_826).abs() < 1e-5);
        assert_eq!(logits[1], 0.0);
        assert!((logits[2] - 28.920_826).abs() < 1e-5);
    }

    #[test]
    fn rotates_neox_pairs_and_leaves_the_remainder() {
        let mut values = [1.0_f32, 2.0, 3.0, 4.0];
        rope_neox_in_place(&mut values, 1, 4, 1, 1, 1.0).unwrap();
        let (sin, cos) = 1.0_f32.sin_cos();
        assert!((values[0] - (cos - 3.0 * sin)).abs() < 1e-6);
        assert_eq!(values[1], 2.0);
        assert!((values[2] - (sin + 3.0 * cos)).abs() < 1e-6);
        assert_eq!(values[3], 4.0);
    }

    #[test]
    fn softmax_and_top_k_are_stable_and_deterministic() {
        let mut probabilities = [1_000.0_f32, 999.0, f32::NEG_INFINITY];
        softmax_in_place(&mut probabilities).unwrap();
        assert!((probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert_eq!(probabilities[2], 0.0);

        let selected = top_k_softmax(&[2.0, 3.0, 3.0, 0.0], 2).unwrap();
        assert_eq!(selected, [(1, 0.5), (2, 0.5)]);
        assert_eq!(argmax(&[1.0, 4.0, 4.0]).unwrap(), 1);
        assert!(matches!(
            top_k_softmax(&[1.0], 0),
            Err(CpuError::InvalidTopK { .. })
        ));
    }
}
