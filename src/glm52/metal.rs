//! No-copy Metal matrix execution over sharded GLM safetensors files.

use std::{
    collections::HashMap,
    fs::File,
    sync::{Arc, Mutex},
};

use memmap2::MmapOptions;

use crate::{
    glm52::{
        expert::{CachedExpert, Glm52ExpertCacheError},
        quant::resolve_quantization,
    },
    metal::{MetalContext, MetalError, MetalMappedBuffer},
    safetensors::{SafeDtype, SafeTensorInfo, SafeTensorSource},
};

#[derive(Debug)]
pub struct Glm52MetalWeights {
    context: MetalContext,
    files: Vec<Arc<File>>,
    mappings: Mutex<HashMap<usize, TensorMapping>>,
    page_size: usize,
}

#[derive(Debug)]
struct TensorMapping {
    encoded: MetalMappedBuffer,
    encoded_offset: usize,
    scale: Option<(MetalMappedBuffer, usize, usize)>,
}

impl Glm52MetalWeights {
    pub fn new(source: &SafeTensorSource) -> Result<Self, MetalError> {
        let context = MetalContext::new()?;
        let files = source
            .index
            .shards()
            .iter()
            .map(|shard| {
                let file = File::open(&shard.path)
                    .map_err(|error| MetalError::Command(error.to_string()))?;
                Ok(Arc::new(file))
            })
            .collect::<Result<Vec<_>, _>>()?;
        // SAFETY: sysconf reads a process-wide platform constant.
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        let page_size = usize::try_from(page_size).map_err(|_| MetalError::PageSize)?;
        if page_size == 0 {
            return Err(MetalError::PageSize);
        }
        Ok(Self {
            context,
            files,
            mappings: Mutex::new(HashMap::new()),
            page_size,
        })
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
        let length = usize::try_from(tensor.byte_len).map_err(|_| MetalError::Overflow)?;
        let scale_name = match tensor.dtype {
            SafeDtype::F8E4M3 => Some(format!("{}_scale_inv", tensor.name)),
            SafeDtype::U8 => Some(format!("{}.qs", tensor.name)),
            _ => None,
        };
        let scale = if let Some(scale_name) = scale_name {
            Some(
                source
                    .index
                    .tensor(&scale_name)
                    .ok_or(MetalError::UnsupportedTensorType(
                        "missing safetensors quantization scale",
                    ))?,
            )
        } else {
            None
        };
        let mut mappings = self
            .mappings
            .lock()
            .map_err(|_| MetalError::Command("GLM Metal mapping cache was poisoned".to_owned()))?;
        if let std::collections::hash_map::Entry::Vacant(entry) = mappings.entry(tensor_index) {
            let (encoded, encoded_offset) = self.map_tensor(tensor)?;
            let scale = scale
                .map(|scale| {
                    let scale_length =
                        usize::try_from(scale.byte_len).map_err(|_| MetalError::Overflow)?;
                    let (mapping, offset) = self.map_tensor(scale)?;
                    Ok((mapping, offset, scale_length))
                })
                .transpose()?;
            entry.insert(TensorMapping {
                encoded,
                encoded_offset,
                scale,
            });
        }
        let mapping = mappings.get(&tensor_index).ok_or(MetalError::Overflow)?;
        let scale = mapping
            .scale
            .as_ref()
            .map(|(mapping, offset, length)| (mapping, *offset, *length));
        self.context.matvec_safetensor_mapped(
            tensor.dtype,
            rows,
            columns,
            &mapping.encoded,
            mapping.encoded_offset,
            length,
            scale,
            input,
            output,
        )
    }

    fn map_tensor(
        &self,
        tensor: &SafeTensorInfo,
    ) -> Result<(MetalMappedBuffer, usize), MetalError> {
        let absolute_offset =
            usize::try_from(tensor.absolute_offset).map_err(|_| MetalError::Overflow)?;
        let tensor_length = usize::try_from(tensor.byte_len).map_err(|_| MetalError::Overflow)?;
        let mapping_offset = absolute_offset / self.page_size * self.page_size;
        let tensor_offset = absolute_offset - mapping_offset;
        let mapping_length = tensor_offset
            .checked_add(tensor_length)
            .ok_or(MetalError::Overflow)?;
        let file = self.files.get(tensor.shard).ok_or(MetalError::Overflow)?;
        // SAFETY: the model files are opened read-only and each page-aligned
        // range mapping is retained by its Metal buffer for the cache lifetime.
        let mapping = unsafe {
            MmapOptions::new()
                .offset(u64::try_from(mapping_offset).map_err(|_| MetalError::Overflow)?)
                .len(mapping_length)
                .map(file.as_ref())
        }
        .map_err(|error| MetalError::Command(error.to_string()))?;
        Ok((
            self.context.map_read_only(Arc::new(mapping))?,
            tensor_offset,
        ))
    }
}
