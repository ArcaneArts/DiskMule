//! No-copy Metal matrix execution for mapped Qwen3.5/Qwen3.8 GGUF weights.

use crate::{
    gguf::TensorSource,
    metal::{MetalContext, MetalError, MetalMappedBuffer},
};

#[derive(Debug)]
pub struct Qwen35MetalWeights {
    context: MetalContext,
    mapping: MetalMappedBuffer,
}

impl Qwen35MetalWeights {
    pub fn new(source: &TensorSource) -> Result<Self, MetalError> {
        let context = MetalContext::new()?;
        let mapping = context.map_read_only(source.shared_mapping())?;
        Ok(Self { context, mapping })
    }

    pub fn device_name(&self) -> String {
        self.context.device_name()
    }

    pub fn matvec(
        &self,
        source: &TensorSource,
        tensor_index: usize,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), MetalError> {
        let tensor = source
            .gguf
            .tensors
            .get(tensor_index)
            .ok_or(MetalError::Overflow)?;
        if tensor.dimensions.len() != 2 {
            return Err(MetalError::UnsupportedTensorType("non-matrix GGUF tensor"));
        }
        self.context.matvec_mapped(
            tensor.kind,
            dimension(tensor.dimensions[1])?,
            dimension(tensor.dimensions[0])?,
            &self.mapping,
            dimension(tensor.absolute_offset)?,
            dimension(tensor.byte_len)?,
            input,
            output,
        )
    }
}

fn dimension(value: u64) -> Result<usize, MetalError> {
    usize::try_from(value).map_err(|_| MetalError::Overflow)
}
