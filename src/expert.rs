//! Bounded storage-backed expert residency for Metal execution.

use std::{
    fs::File,
    io,
    os::unix::fs::FileExt,
    sync::Arc,
    time::{Duration, Instant},
};

use crate::metal::{MetalContext, MetalError, MetalOwnedBuffer};

#[derive(Debug, thiserror::Error)]
pub enum ExpertCacheError {
    #[error(transparent)]
    Io(#[from] io::Error),

    #[error(transparent)]
    Metal(#[from] MetalError),

    #[error("expert cache requires at least one slot per layer")]
    NoSlots,

    #[error("expert cache layer {layer} is outside its {count} layers")]
    InvalidLayer { layer: usize, count: usize },

    #[error("expert {expert} is outside layer {layer}'s {count} experts")]
    InvalidExpert {
        layer: usize,
        expert: usize,
        count: usize,
    },

    #[error("all expert slots in layer {layer} are pinned by in-flight work")]
    AllSlotsPinned { layer: usize },

    #[error("expert loading was cancelled")]
    Cancelled,

    #[error("invalid expert cache layout: {0}")]
    InvalidLayout(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertLayerLayout {
    pub expert_count: usize,
    pub gate_up_offset: u64,
    pub gate_up_bytes: usize,
    pub down_offset: u64,
    pub down_bytes: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExpertCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub reads: u64,
    pub bytes_read: u64,
    pub io_wait: Duration,
    pub resident_bytes: u64,
}

#[derive(Debug)]
pub struct ExpertCache {
    file: Arc<File>,
    layers: Vec<LayerCache>,
    stats: ExpertCacheStats,
}

#[derive(Debug)]
struct LayerCache {
    layout: ExpertLayerLayout,
    slots: Vec<ExpertSlot>,
    clock: u64,
}

#[derive(Debug)]
struct ExpertSlot {
    expert: Option<usize>,
    last_used: u64,
    pinned: bool,
    gate_up: MetalOwnedBuffer,
    down: MetalOwnedBuffer,
}

impl ExpertCache {
    pub fn new(
        metal: &MetalContext,
        file: Arc<File>,
        layouts: &[ExpertLayerLayout],
        slots_per_layer: usize,
    ) -> Result<Self, ExpertCacheError> {
        if slots_per_layer == 0 {
            return Err(ExpertCacheError::NoSlots);
        }
        let mut resident_bytes = 0_u64;
        let mut layers = Vec::with_capacity(layouts.len());
        for layout in layouts {
            let mut slots = Vec::with_capacity(slots_per_layer);
            for _ in 0..slots_per_layer {
                slots.push(ExpertSlot {
                    expert: None,
                    last_used: 0,
                    pinned: false,
                    gate_up: metal.new_owned_buffer(layout.gate_up_bytes)?,
                    down: metal.new_owned_buffer(layout.down_bytes)?,
                });
                resident_bytes = resident_bytes.saturating_add(
                    u64::try_from(layout.gate_up_bytes.saturating_add(layout.down_bytes))
                        .unwrap_or(u64::MAX),
                );
            }
            layers.push(LayerCache {
                layout: *layout,
                slots,
                clock: 0,
            });
        }
        Ok(Self {
            file,
            layers,
            stats: ExpertCacheStats {
                resident_bytes,
                ..ExpertCacheStats::default()
            },
        })
    }

    pub fn stats(&self) -> ExpertCacheStats {
        self.stats
    }

    pub fn slots_per_layer(&self) -> usize {
        self.layers.first().map_or(0, |layer| layer.slots.len())
    }

    pub fn with_expert<T, E, F, C>(
        &mut self,
        layer: usize,
        expert: usize,
        cancelled: C,
        use_expert: F,
    ) -> Result<T, E>
    where
        E: From<ExpertCacheError>,
        F: FnOnce(&MetalOwnedBuffer, &MetalOwnedBuffer) -> Result<T, E>,
        C: Fn() -> bool,
    {
        if cancelled() {
            return Err(ExpertCacheError::Cancelled.into());
        }
        let slot_index = self.acquire(layer, expert).map_err(E::from)?;
        if cancelled() {
            return Err(ExpertCacheError::Cancelled.into());
        }
        self.layers[layer].slots[slot_index].pinned = true;
        let result = {
            let slot = &self.layers[layer].slots[slot_index];
            use_expert(&slot.gate_up, &slot.down)
        };
        self.layers[layer].slots[slot_index].pinned = false;
        result
    }

    fn acquire(&mut self, layer_index: usize, expert: usize) -> Result<usize, ExpertCacheError> {
        let layer_count = self.layers.len();
        let layer = self
            .layers
            .get_mut(layer_index)
            .ok_or(ExpertCacheError::InvalidLayer {
                layer: layer_index,
                count: layer_count,
            })?;
        if expert >= layer.layout.expert_count {
            return Err(ExpertCacheError::InvalidExpert {
                layer: layer_index,
                expert,
                count: layer.layout.expert_count,
            });
        }
        layer.clock = layer.clock.saturating_add(1);
        if let Some((slot_index, slot)) = layer
            .slots
            .iter_mut()
            .enumerate()
            .find(|(_, slot)| slot.expert == Some(expert))
        {
            slot.last_used = layer.clock;
            self.stats.hits = self.stats.hits.saturating_add(1);
            return Ok(slot_index);
        }

        self.stats.misses = self.stats.misses.saturating_add(1);
        let slot_index = layer
            .slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| !slot.pinned)
            .min_by_key(|(index, slot)| (slot.expert.is_some(), slot.last_used, *index))
            .map(|(index, _)| index)
            .ok_or(ExpertCacheError::AllSlotsPinned { layer: layer_index })?;
        if layer.slots[slot_index].expert.is_some() {
            self.stats.evictions = self.stats.evictions.saturating_add(1);
        }

        let layout = layer.layout;
        let gate_up_offset = expert_offset(layout.gate_up_offset, layout.gate_up_bytes, expert)?;
        let down_offset = expert_offset(layout.down_offset, layout.down_bytes, expert)?;
        let started = Instant::now();
        let slot = &mut layer.slots[slot_index];
        let gate_up = slot.gate_up.as_mut_bytes();
        let down = slot.down.as_mut_bytes();
        let (gate_result, down_result) = rayon::join(
            || read_exact_at(&self.file, gate_up_offset, gate_up),
            || read_exact_at(&self.file, down_offset, down),
        );
        gate_result?;
        down_result?;
        self.stats.io_wait = self.stats.io_wait.saturating_add(started.elapsed());
        self.stats.reads = self.stats.reads.saturating_add(2);
        self.stats.bytes_read = self.stats.bytes_read.saturating_add(
            u64::try_from(layout.gate_up_bytes.saturating_add(layout.down_bytes))
                .unwrap_or(u64::MAX),
        );
        slot.expert = Some(expert);
        slot.last_used = layer.clock;
        Ok(slot_index)
    }
}

fn expert_offset(base: u64, bytes: usize, expert: usize) -> Result<u64, ExpertCacheError> {
    let relative = expert
        .checked_mul(bytes)
        .and_then(|offset| u64::try_from(offset).ok())
        .ok_or_else(|| io::Error::other("expert file offset overflow"))?;
    base.checked_add(relative)
        .ok_or_else(|| io::Error::other("expert file offset overflow").into())
}

fn read_exact_at(file: &File, mut offset: u64, mut destination: &mut [u8]) -> io::Result<()> {
    while !destination.is_empty() {
        match file.read_at(destination, offset) {
            Ok(0) => return Err(io::Error::from(io::ErrorKind::UnexpectedEof)),
            Ok(read) => {
                offset = offset
                    .checked_add(read as u64)
                    .ok_or_else(|| io::Error::other("expert read offset overflow"))?;
                destination = &mut destination[read..];
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{io::Write, sync::Arc};

    use crate::metal::MetalContext;

    use super::{ExpertCache, ExpertCacheError, ExpertLayerLayout};

    #[test]
    fn bounded_cache_tracks_hits_misses_evictions_and_cancellation() {
        let metal = MetalContext::new().unwrap();
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(&[
            1, 2, 3, 4, 5, 6, 7, 8, // gate/up experts
            11, 12, 13, 14, 15, 16, 17, 18, // down experts
        ])
        .unwrap();
        file.flush().unwrap();
        let layout = ExpertLayerLayout {
            expert_count: 2,
            gate_up_offset: 0,
            gate_up_bytes: 4,
            down_offset: 8,
            down_bytes: 4,
        };
        let mut cache = ExpertCache::new(&metal, Arc::new(file), &[layout], 1).unwrap();
        assert_eq!(cache.slots_per_layer(), 1);

        for _ in 0..2 {
            cache
                .with_expert::<_, ExpertCacheError, _, _>(
                    0,
                    0,
                    || false,
                    |gate, down| {
                        assert_eq!(gate.as_bytes(), [1, 2, 3, 4]);
                        assert_eq!(down.as_bytes(), [11, 12, 13, 14]);
                        Ok(())
                    },
                )
                .unwrap();
        }
        cache
            .with_expert::<_, ExpertCacheError, _, _>(
                0,
                1,
                || false,
                |gate, down| {
                    assert_eq!(gate.as_bytes(), [5, 6, 7, 8]);
                    assert_eq!(down.as_bytes(), [15, 16, 17, 18]);
                    Ok(())
                },
            )
            .unwrap();
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 2);
        assert_eq!(stats.evictions, 1);
        assert_eq!(stats.reads, 4);
        assert_eq!(stats.bytes_read, 16);
        assert_eq!(stats.resident_bytes, 8);
        assert!(stats.io_wait > std::time::Duration::ZERO);

        assert!(matches!(
            cache.with_expert::<_, ExpertCacheError, _, _>(0, 0, || true, |_, _| Ok(())),
            Err(ExpertCacheError::Cancelled)
        ));
    }
}
