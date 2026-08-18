//! No-copy Metal matrix execution over sharded GLM safetensors files.

use std::{fs::File, sync::Arc};

use memmap2::MmapOptions;

use crate::{
    glm52::{
        expert::{CachedExpert, Glm52ExpertCacheError},
        quant::resolve_quantization,
    },
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

    pub fn expert_mlp(
        &self,
        expert: &CachedExpert,
        input: &[f32],
    ) -> Result<Vec<f32>, Glm52ExpertCacheError> {
        expert.mlp_metal(&self.context, input)
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
        let (rows, columns) = if tensor.dtype == SafeDtype::U8 {
            resolve_quantization(
                &source.index,
                tensor,
                u64::try_from(output.len()).map_err(|_| MetalError::Overflow)?,
                u64::try_from(input.len()).map_err(|_| MetalError::Overflow)?,
            )
            .map_err(|error| MetalError::Command(error.to_string()))?;
            (output.len(), input.len())
        } else {
            if tensor.shape.len() != 2 {
                return Err(MetalError::UnsupportedTensorType(
                    "non-matrix safetensors tensor",
                ));
            }
            (
                usize::try_from(tensor.shape[0]).map_err(|_| MetalError::Overflow)?,
                usize::try_from(tensor.shape[1]).map_err(|_| MetalError::Overflow)?,
            )
        };
        let offset = usize::try_from(tensor.absolute_offset).map_err(|_| MetalError::Overflow)?;
        let length = usize::try_from(tensor.byte_len).map_err(|_| MetalError::Overflow)?;
        let mapping = self
            .mappings
            .get(tensor.shard)
            .ok_or(MetalError::Overflow)?;
        let scale_name = match tensor.dtype {
            SafeDtype::F8E4M3 => Some(format!("{}_scale_inv", tensor.name)),
            SafeDtype::U8 => Some(format!("{}.qs", tensor.name)),
            _ => None,
        };
        let scale = if let Some(scale_name) = scale_name {
            let scale =
                source
                    .index
                    .tensor(&scale_name)
                    .ok_or(MetalError::UnsupportedTensorType(
                        "missing safetensors quantization scale",
                    ))?;
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
