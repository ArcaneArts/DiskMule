//! Scalar CPU reference operators.
//!
//! These prioritize transparent, deterministic arithmetic over throughput.
//! Metal kernels and future vectorized paths are compared against this module.

use crate::gguf::TensorType;

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

    let mut row = vec![0.0_f32; columns];
    for (row_index, result) in output.iter_mut().enumerate() {
        let start = row_index * row_bytes;
        dequantize(kind, &encoded[start..start + row_bytes], &mut row)?;
        *result = row
            .iter()
            .zip(input)
            .fold(0.0_f32, |sum, (weight, value)| sum + weight * value);
    }
    Ok(())
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
    use super::{CpuError, dequantize, matvec, scale_min_q4_k};
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
}
