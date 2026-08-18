//! Strict detection of Colibri-compatible integer matrix layouts.

use crate::safetensors::{SafeDtype, SafeTensorIndex, SafeTensorInfo};

const GROUP_SIZES: [u64; 8] = [16, 32, 48, 64, 96, 128, 192, 256];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GlmQuantization {
    Int8Row,
    Int4Grouped { group_size: usize },
}

impl GlmQuantization {
    pub(crate) const fn row_bytes(self, columns: usize) -> usize {
        match self {
            Self::Int8Row => columns,
            Self::Int4Grouped { .. } => columns.div_ceil(2),
        }
    }

    pub(crate) const fn scales_per_row(self, columns: usize) -> usize {
        match self {
            Self::Int8Row => 1,
            Self::Int4Grouped { group_size } => columns.div_ceil(group_size),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GlmQuantizationError {
    #[error("integer arithmetic overflow while validating quantized tensor {0:?}")]
    Overflow(String),

    #[error("quantized tensor {tensor:?} requires a flat F32 scale tensor {scale:?}")]
    MissingScale { tensor: String, scale: String },

    #[error(
        "quantized tensor {tensor:?} has an invalid flat layout for logical shape [{rows}, {columns}]: {weight_bytes} weight bytes and {scale_count} scales"
    )]
    InvalidLayout {
        tensor: String,
        rows: u64,
        columns: u64,
        weight_bytes: u64,
        scale_count: u64,
    },

    #[error(
        "quantized tensor {tensor:?} has ambiguous geometry for {columns} input columns: {weight_bytes} weight bytes and {scale_count} scales"
    )]
    AmbiguousLayout {
        tensor: String,
        columns: u64,
        weight_bytes: u64,
        scale_count: u64,
    },
}

pub(crate) fn resolve_quantization(
    index: &SafeTensorIndex,
    tensor: &SafeTensorInfo,
    rows: u64,
    columns: u64,
) -> Result<GlmQuantization, GlmQuantizationError> {
    let scale_count = validate_flat_storage(index, tensor)?;
    matching_quantizations(tensor, rows, columns, scale_count)
        .into_iter()
        .next()
        .ok_or_else(|| GlmQuantizationError::InvalidLayout {
            tensor: tensor.name.clone(),
            rows,
            columns,
            weight_bytes: tensor.byte_len,
            scale_count,
        })
}

pub(crate) fn infer_quantization(
    index: &SafeTensorIndex,
    tensor: &SafeTensorInfo,
    columns: usize,
) -> Result<(usize, GlmQuantization), GlmQuantizationError> {
    let scale_count = validate_flat_storage(index, tensor)?;
    infer_quantization_from_scale_count(tensor, scale_count, columns)
}

pub(crate) fn infer_quantization_from_scale_count(
    tensor: &SafeTensorInfo,
    scale_count: u64,
    columns: usize,
) -> Result<(usize, GlmQuantization), GlmQuantizationError> {
    let columns =
        u64::try_from(columns).map_err(|_| GlmQuantizationError::Overflow(tensor.name.clone()))?;
    let mut matches = Vec::with_capacity(2);

    if columns != 0 && tensor.byte_len.is_multiple_of(columns) {
        let rows = tensor.byte_len / columns;
        for quantization in matching_quantizations(tensor, rows, columns, scale_count) {
            matches.push((rows, quantization));
        }
    }
    let packed_row_bytes = columns.div_ceil(2);
    if packed_row_bytes != 0 && tensor.byte_len.is_multiple_of(packed_row_bytes) {
        let rows = tensor.byte_len / packed_row_bytes;
        for quantization in matching_quantizations(tensor, rows, columns, scale_count) {
            if !matches.contains(&(rows, quantization)) {
                matches.push((rows, quantization));
            }
        }
    }

    if matches.len() != 1 {
        return Err(GlmQuantizationError::AmbiguousLayout {
            tensor: tensor.name.clone(),
            columns,
            weight_bytes: tensor.byte_len,
            scale_count,
        });
    }
    let (rows, quantization) = matches[0];
    Ok((
        usize::try_from(rows).map_err(|_| GlmQuantizationError::Overflow(tensor.name.clone()))?,
        quantization,
    ))
}

fn validate_flat_storage(
    index: &SafeTensorIndex,
    tensor: &SafeTensorInfo,
) -> Result<u64, GlmQuantizationError> {
    let scale_name = format!("{}.qs", tensor.name);
    let scale = index
        .tensor(&scale_name)
        .ok_or_else(|| GlmQuantizationError::MissingScale {
            tensor: tensor.name.clone(),
            scale: scale_name.clone(),
        })?;
    let scale_count = scale.byte_len.checked_div(4).filter(|_| {
        scale.dtype == SafeDtype::F32
            && scale.byte_len.is_multiple_of(4)
            && scale.shape == [scale.byte_len / 4]
    });
    if tensor.dtype != SafeDtype::U8 || tensor.shape != [tensor.byte_len] || scale_count.is_none() {
        return Err(GlmQuantizationError::MissingScale {
            tensor: tensor.name.clone(),
            scale: scale_name,
        });
    }
    Ok(scale_count.expect("validated scale count"))
}

fn matching_quantizations(
    tensor: &SafeTensorInfo,
    rows: u64,
    columns: u64,
    scale_count: u64,
) -> Vec<GlmQuantization> {
    let mut matches = Vec::with_capacity(2);
    if rows
        .checked_mul(columns)
        .is_some_and(|bytes| bytes == tensor.byte_len)
        && scale_count == rows
    {
        matches.push(GlmQuantization::Int8Row);
    }
    if rows
        .checked_mul(columns.div_ceil(2))
        .is_some_and(|bytes| bytes == tensor.byte_len)
    {
        for group_size in GROUP_SIZES {
            if rows
                .checked_mul(columns.div_ceil(group_size))
                .is_some_and(|count| count == scale_count)
            {
                matches.push(GlmQuantization::Int4Grouped {
                    group_size: group_size as usize,
                });
                break;
            }
        }
    }
    matches
}
