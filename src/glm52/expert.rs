//! Bounded, shard-aware GLM expert residency with parallel range reads.

use std::{sync::atomic::AtomicBool, time::Duration, time::Instant};

use crate::{
    glm52::{Glm52FeedForwardWeights, Glm52Weights},
    safetensors::{SafeDtype, SafeTensorError, SafeTensorInfo, SafeTensorSource},
};

const FP8_BLOCK: usize = 128;

#[derive(Debug, thiserror::Error)]
pub enum Glm52ExpertCacheError {
    #[error(transparent)]
    SafeTensor(#[from] SafeTensorError),

    #[error("GLM expert cache requires at least one slot")]
    NoSlots,

    #[error("GLM expert {expert} does not exist in sparse layer {layer}")]
    InvalidExpert { layer: usize, expert: usize },

    #[error("GLM expert tensor {tensor:?} is too large to cache")]
    TensorTooLarge { tensor: String },

    #[error("GLM expert tensor {tensor:?} has unsupported cached dtype {dtype:?}")]
    UnsupportedDtype { tensor: String, dtype: SafeDtype },

    #[error("GLM expert tensor {tensor:?} has invalid cached shape {shape:?}")]
    InvalidShape { tensor: String, shape: Vec<u64> },

    #[error("GLM expert tensor {tensor:?} is missing its FP8 scale grid")]
    MissingScale { tensor: String },

    #[error("GLM expert cache operation was cancelled")]
    Cancelled,

    #[error("GLM expert cache lock was poisoned")]
    Poisoned,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Glm52ExpertCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub reads: u64,
    pub bytes_read: u64,
    pub io_wait: Duration,
    pub resident_bytes: u64,
}

#[derive(Debug)]
pub struct Glm52ExpertCache {
    source: SafeTensorSource,
    experts: Vec<Option<Vec<[usize; 3]>>>,
    slots: Vec<Vec<CacheSlot>>,
    clock: u64,
    stats: Glm52ExpertCacheStats,
}

#[derive(Debug)]
struct CacheSlot {
    expert: Option<usize>,
    last_used: u64,
    value: Option<CachedExpert>,
}

#[derive(Debug)]
pub struct CachedExpert {
    gate: CachedTensor,
    up: CachedTensor,
    down: CachedTensor,
}

#[derive(Debug)]
struct CachedTensor {
    info: SafeTensorInfo,
    bytes: Vec<u8>,
    scales: Vec<f32>,
}

#[derive(Debug, Clone, Copy, Default)]
struct LoadMetrics {
    reads: u64,
    bytes: u64,
}

impl Glm52ExpertCache {
    pub fn new(
        source: SafeTensorSource,
        weights: &Glm52Weights,
        slot_count: usize,
    ) -> Result<Self, Glm52ExpertCacheError> {
        if slot_count == 0 {
            return Err(Glm52ExpertCacheError::NoSlots);
        }
        let experts: Vec<Option<Vec<[usize; 3]>>> = weights
            .layers
            .iter()
            .map(|layer| match &layer.feed_forward {
                Glm52FeedForwardWeights::Dense { .. } => None,
                Glm52FeedForwardWeights::Sparse { experts, .. } => Some(
                    experts
                        .iter()
                        .map(|expert| [expert.gate, expert.up, expert.down])
                        .collect(),
                ),
            })
            .collect();
        let slots = experts
            .iter()
            .map(|experts| {
                if experts.is_some() {
                    (0..slot_count)
                        .map(|_| CacheSlot {
                            expert: None,
                            last_used: 0,
                            value: None,
                        })
                        .collect()
                } else {
                    Vec::new()
                }
            })
            .collect();
        Ok(Self {
            source,
            experts,
            slots,
            clock: 0,
            stats: Glm52ExpertCacheStats::default(),
        })
    }

    pub fn stats(&self) -> Glm52ExpertCacheStats {
        self.stats
    }

    pub fn slot_count(&self) -> usize {
        self.slots.iter().map(Vec::len).max().unwrap_or(0)
    }

    pub fn with_expert<T, E, F>(
        &mut self,
        layer: usize,
        expert: usize,
        cancelled: &AtomicBool,
        use_expert: F,
    ) -> Result<T, E>
    where
        E: From<Glm52ExpertCacheError>,
        F: FnOnce(&CachedExpert) -> Result<T, E>,
    {
        if cancelled.load(std::sync::atomic::Ordering::Relaxed) {
            return Err(Glm52ExpertCacheError::Cancelled.into());
        }
        let slot = self.acquire(layer, expert, cancelled).map_err(E::from)?;
        if cancelled.load(std::sync::atomic::Ordering::Relaxed) {
            return Err(Glm52ExpertCacheError::Cancelled.into());
        }
        use_expert(
            self.slots[layer][slot]
                .value
                .as_ref()
                .expect("acquired GLM expert slot is populated"),
        )
    }

    fn acquire(
        &mut self,
        layer: usize,
        expert: usize,
        cancelled: &AtomicBool,
    ) -> Result<usize, Glm52ExpertCacheError> {
        let tensors = self
            .experts
            .get(layer)
            .and_then(Option::as_ref)
            .and_then(|experts| experts.get(expert))
            .copied()
            .ok_or(Glm52ExpertCacheError::InvalidExpert { layer, expert })?;
        self.clock = self.clock.saturating_add(1);
        let layer_slots = &mut self.slots[layer];
        if let Some((index, slot)) = layer_slots
            .iter_mut()
            .enumerate()
            .find(|(_, slot)| slot.expert == Some(expert))
        {
            slot.last_used = self.clock;
            self.stats.hits = self.stats.hits.saturating_add(1);
            return Ok(index);
        }

        self.stats.misses = self.stats.misses.saturating_add(1);
        let slot_index = layer_slots
            .iter()
            .enumerate()
            .min_by_key(|(index, slot)| (slot.expert.is_some(), slot.last_used, *index))
            .map(|(index, _)| index)
            .expect("positive GLM expert slot count");
        if layer_slots[slot_index].expert.is_some() {
            self.stats.evictions = self.stats.evictions.saturating_add(1);
        }

        let started = Instant::now();
        let ((gate, up), down) = rayon::join(
            || {
                rayon::join(
                    || CachedTensor::load(&self.source, tensors[0]),
                    || CachedTensor::load(&self.source, tensors[1]),
                )
            },
            || CachedTensor::load(&self.source, tensors[2]),
        );
        let (gate, gate_metrics) = gate?;
        let (up, up_metrics) = up?;
        let (down, down_metrics) = down?;
        if cancelled.load(std::sync::atomic::Ordering::Relaxed) {
            return Err(Glm52ExpertCacheError::Cancelled);
        }
        let metrics = gate_metrics.add(up_metrics).add(down_metrics);
        self.stats.io_wait = self.stats.io_wait.saturating_add(started.elapsed());
        self.stats.reads = self.stats.reads.saturating_add(metrics.reads);
        self.stats.bytes_read = self.stats.bytes_read.saturating_add(metrics.bytes);
        let value = CachedExpert { gate, up, down };
        let resident = value.resident_bytes();
        let evicted = layer_slots[slot_index]
            .value
            .as_ref()
            .map_or(0, CachedExpert::resident_bytes);
        self.stats.resident_bytes = self
            .stats
            .resident_bytes
            .saturating_sub(evicted)
            .saturating_add(resident);
        layer_slots[slot_index] = CacheSlot {
            expert: Some(expert),
            last_used: self.clock,
            value: Some(value),
        };
        Ok(slot_index)
    }
}

impl CachedExpert {
    pub fn mlp(&self, input: &[f32]) -> Result<Vec<f32>, Glm52ExpertCacheError> {
        let mut gate = self.gate.matvec(input)?;
        let up = self.up.matvec(input)?;
        for (gate, up) in gate.iter_mut().zip(up) {
            *gate = (*gate / (1.0 + (-*gate).exp())) * up;
        }
        self.down.matvec(&gate)
    }

    fn resident_bytes(&self) -> u64 {
        self.gate
            .resident_bytes()
            .saturating_add(self.up.resident_bytes())
            .saturating_add(self.down.resident_bytes())
    }
}

impl CachedTensor {
    fn load(
        source: &SafeTensorSource,
        tensor_index: usize,
    ) -> Result<(Self, LoadMetrics), Glm52ExpertCacheError> {
        let info = source
            .index
            .tensors()
            .get(tensor_index)
            .cloned()
            .ok_or_else(|| Glm52ExpertCacheError::TensorTooLarge {
                tensor: format!("index {tensor_index}"),
            })?;
        if info.shape.len() != 2 {
            return Err(Glm52ExpertCacheError::InvalidShape {
                tensor: info.name,
                shape: info.shape,
            });
        }
        if !matches!(
            info.dtype,
            SafeDtype::F32 | SafeDtype::F16 | SafeDtype::Bf16 | SafeDtype::F8E4M3
        ) {
            return Err(Glm52ExpertCacheError::UnsupportedDtype {
                tensor: info.name,
                dtype: info.dtype,
            });
        }
        let length =
            usize::try_from(info.byte_len).map_err(|_| Glm52ExpertCacheError::TensorTooLarge {
                tensor: info.name.clone(),
            })?;
        let mut bytes = vec![0_u8; length];
        source.read_tensor_at(&info.name, 0, &mut bytes)?;
        let mut metrics = LoadMetrics {
            reads: 1,
            bytes: info.byte_len,
        };
        let scales = if info.dtype == SafeDtype::F8E4M3 {
            let scale_name = format!("{}_scale_inv", info.name);
            let scale = source.index.tensor(&scale_name).ok_or_else(|| {
                Glm52ExpertCacheError::MissingScale {
                    tensor: info.name.clone(),
                }
            })?;
            let scale_length = usize::try_from(scale.byte_len).map_err(|_| {
                Glm52ExpertCacheError::TensorTooLarge {
                    tensor: scale.name.clone(),
                }
            })?;
            let mut scale_bytes = vec![0_u8; scale_length];
            source.read_tensor_at(&scale.name, 0, &mut scale_bytes)?;
            metrics.reads += 1;
            metrics.bytes = metrics.bytes.saturating_add(scale.byte_len);
            scale_bytes
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
                .collect()
        } else {
            Vec::new()
        };
        Ok((
            Self {
                info,
                bytes,
                scales,
            },
            metrics,
        ))
    }

    fn matvec(&self, input: &[f32]) -> Result<Vec<f32>, Glm52ExpertCacheError> {
        let rows = usize::try_from(self.info.shape[0]).map_err(|_| {
            Glm52ExpertCacheError::TensorTooLarge {
                tensor: self.info.name.clone(),
            }
        })?;
        let columns = usize::try_from(self.info.shape[1]).map_err(|_| {
            Glm52ExpertCacheError::TensorTooLarge {
                tensor: self.info.name.clone(),
            }
        })?;
        if input.len() != columns {
            return Err(Glm52ExpertCacheError::InvalidShape {
                tensor: self.info.name.clone(),
                shape: self.info.shape.clone(),
            });
        }
        let mut output = vec![0.0; rows];
        for (row, output) in output.iter_mut().enumerate() {
            let mut sum = 0.0;
            for (column, input) in input.iter().enumerate() {
                sum += self.value(row, column, columns)? * input;
            }
            *output = sum;
        }
        Ok(output)
    }

    fn value(
        &self,
        row: usize,
        column: usize,
        columns: usize,
    ) -> Result<f32, Glm52ExpertCacheError> {
        let index = row * columns + column;
        Ok(match self.info.dtype {
            SafeDtype::F32 => {
                let offset = index * 4;
                f32::from_le_bytes(self.bytes[offset..offset + 4].try_into().unwrap())
            }
            SafeDtype::F16 => {
                let offset = index * 2;
                decode_f16(u16::from_le_bytes(
                    self.bytes[offset..offset + 2].try_into().unwrap(),
                ))
            }
            SafeDtype::Bf16 => {
                let offset = index * 2;
                let value = u16::from_le_bytes(self.bytes[offset..offset + 2].try_into().unwrap());
                f32::from_bits(u32::from(value) << 16)
            }
            SafeDtype::F8E4M3 => {
                let scale_columns = columns.div_ceil(FP8_BLOCK);
                decode_f8(self.bytes[index]).ok_or_else(|| {
                    Glm52ExpertCacheError::UnsupportedDtype {
                        tensor: self.info.name.clone(),
                        dtype: self.info.dtype,
                    }
                })? * self.scales[(row / FP8_BLOCK) * scale_columns + column / FP8_BLOCK]
            }
            dtype => {
                return Err(Glm52ExpertCacheError::UnsupportedDtype {
                    tensor: self.info.name.clone(),
                    dtype,
                });
            }
        })
    }

    fn resident_bytes(&self) -> u64 {
        u64::try_from(self.bytes.len().saturating_add(self.scales.len() * 4)).unwrap_or(u64::MAX)
    }
}

impl LoadMetrics {
    fn add(self, other: Self) -> Self {
        Self {
            reads: self.reads.saturating_add(other.reads),
            bytes: self.bytes.saturating_add(other.bytes),
        }
    }
}

fn decode_f16(value: u16) -> f32 {
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

fn decode_f8(value: u8) -> Option<f32> {
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

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;

    use tempfile::TempDir;

    use crate::{
        glm52::{
            Glm52Config, Glm52FeedForwardWeights, Glm52Weights,
            cpu::{TensorReader, write_test_model},
        },
        safetensors::SafeTensorSource,
    };

    use super::{Glm52ExpertCache, Glm52ExpertCacheError};

    #[test]
    fn caches_exact_experts_with_parallel_range_reads_and_lru_eviction() {
        let temp = TempDir::new().unwrap();
        write_test_model(temp.path());
        let config = Glm52Config::from_directory(temp.path()).unwrap();
        let source = SafeTensorSource::open(temp.path()).unwrap();
        let weights = Glm52Weights::from_index(&source.index, &config).unwrap();
        let experts = match &weights.layers[1].feed_forward {
            Glm52FeedForwardWeights::Sparse { experts, .. } => experts,
            _ => panic!("fixture layer 1 must be sparse"),
        };
        let input = [0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9];
        let expected = uncached_mlp(&source, &experts[0], &input);
        let mut cache = Glm52ExpertCache::new(source, &weights, 1).unwrap();
        let cancelled = AtomicBool::new(false);
        for _ in 0..2 {
            let actual = cache
                .with_expert::<_, Glm52ExpertCacheError, _>(1, 0, &cancelled, |expert| {
                    expert.mlp(&input)
                })
                .unwrap();
            assert_close(&actual, &expected);
        }
        cache
            .with_expert::<_, Glm52ExpertCacheError, _>(1, 1, &cancelled, |expert| {
                expert.mlp(&input).map(|_| ())
            })
            .unwrap();
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 2);
        assert_eq!(stats.evictions, 1);
        assert_eq!(stats.reads, 6);
        assert!(stats.bytes_read > 0);
        assert!(stats.resident_bytes > 0);

        cancelled.store(true, std::sync::atomic::Ordering::Relaxed);
        assert!(matches!(
            cache.with_expert::<_, Glm52ExpertCacheError, _>(1, 0, &cancelled, |_| Ok(())),
            Err(Glm52ExpertCacheError::Cancelled)
        ));
    }

    fn uncached_mlp(
        source: &SafeTensorSource,
        weights: &crate::glm52::Glm52ExpertWeights,
        input: &[f32],
    ) -> Vec<f32> {
        let reader = TensorReader::new(source);
        let mut gate = reader.matvec(weights.gate, input).unwrap();
        let up = reader.matvec(weights.up, input).unwrap();
        for (gate, up) in gate.iter_mut().zip(up) {
            *gate = (*gate / (1.0 + (-*gate).exp())) * up;
        }
        reader.matvec(weights.down, &gate).unwrap()
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
    }
}
