//! No-copy Metal matrix execution over sharded GLM safetensors files.

use std::{fs::File, sync::Arc};

use memmap2::MmapOptions;

use crate::{
    metal::{MetalContext, MetalError, MetalMappedBuffer},
    safetensors::{SafeDtype, SafeTensorSource},
};

#[derive(Debug)]
pub struct Glm52MetalWeights {
    context: MetalContext,
    mappings: Vec<MetalMappedBuffer>,
}

impl Glm52MetalWeights {
    pub fn new(source: &SafeTensorSource) -> Result<Self, MetalError> {
        let context = MetalContext::new()?;
        let mappings = source
            .index
            .shards()
            .iter()
            .map(|shard| {
                let file = File::open(&shard.path)
                    .map_err(|error| MetalError::Command(error.to_string()))?;
                // SAFETY: the model files are opened read-only and the mapping is
                // retained by its Metal buffer for the accelerator lifetime.
                let mapping = unsafe { MmapOptions::new().map(&file) }
                    .map_err(|error| MetalError::Command(error.to_string()))?;
                context.map_read_only(Arc::new(mapping))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { context, mappings })
    }

    pub fn device_name(&self) -> String {
        self.context.device_name()
    }

    pub fn matvec(
        &self,
        source: &SafeTensorSource,
        tensor_index: usize,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), MetalError> {
        let tensor = source
            .index
            .tensors()
            .get(tensor_index)
            .ok_or(MetalError::Overflow)?;
        if tensor.shape.len() != 2 {
            return Err(MetalError::UnsupportedTensorType(
                "non-matrix safetensors tensor",
            ));
        }
        let rows = usize::try_from(tensor.shape[0]).map_err(|_| MetalError::Overflow)?;
        let columns = usize::try_from(tensor.shape[1]).map_err(|_| MetalError::Overflow)?;
        let offset = usize::try_from(tensor.absolute_offset).map_err(|_| MetalError::Overflow)?;
        let length = usize::try_from(tensor.byte_len).map_err(|_| MetalError::Overflow)?;
        let mapping = self
            .mappings
            .get(tensor.shard)
            .ok_or(MetalError::Overflow)?;
        let scale = if tensor.dtype == SafeDtype::F8E4M3 {
            let scale = source
                .index
                .tensor(&format!("{}_scale_inv", tensor.name))
                .ok_or(MetalError::UnsupportedTensorType("missing FP8 scale"))?;
            Some((
                self.mappings.get(scale.shard).ok_or(MetalError::Overflow)?,
                usize::try_from(scale.absolute_offset).map_err(|_| MetalError::Overflow)?,
                usize::try_from(scale.byte_len).map_err(|_| MetalError::Overflow)?,
            ))
        } else {
            None
        };
        self.context.matvec_safetensor_mapped(
            tensor.dtype,
            rows,
            columns,
            mapping,
            offset,
            length,
            scale,
            input,
            output,
        )
    }
}
