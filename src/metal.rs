//! Apple Metal device and command ownership.

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum MetalError {
    #[error("Metal is unavailable on this platform")]
    UnsupportedPlatform,

    #[error("no Metal device is available")]
    NoDevice,

    #[error("could not create a Metal command queue")]
    NoCommandQueue,

    #[error("could not compile DiskMule Metal shaders: {0}")]
    Library(String),

    #[error("Metal shader {0:?} is missing")]
    MissingFunction(&'static str),

    #[error("could not create Metal pipeline {name:?}: {message}")]
    Pipeline { name: &'static str, message: String },

    #[error("could not allocate a {bytes}-byte shared Metal buffer")]
    BufferAllocation { bytes: usize },

    #[error("could not determine the host memory page size")]
    PageSize,

    #[error("Metal vector lengths differ: left={left}, right={right}, output={output}")]
    VectorLength {
        left: usize,
        right: usize,
        output: usize,
    },

    #[error("Metal vector length exceeds the 32-bit kernel limit")]
    VectorTooLarge,

    #[error("{operation} requires a non-empty input")]
    EmptyInput { operation: &'static str },

    #[error("{operation} {argument} has length {actual}; expected {expected}")]
    InvalidVectorLength {
        operation: &'static str,
        argument: &'static str,
        expected: usize,
        actual: usize,
    },

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

    #[error("{operation} received a non-finite value")]
    NonFinite { operation: &'static str },

    #[error("Metal matrix-vector multiplication does not support {0}")]
    UnsupportedTensorType(&'static str),

    #[error(
        "{kind} output length {elements} is not divisible by its {block_elements}-element block"
    )]
    InvalidElementCount {
        kind: &'static str,
        elements: usize,
        block_elements: u64,
    },

    #[error("{kind} requires {expected} encoded bytes, got {actual}")]
    InvalidByteLength {
        kind: &'static str,
        expected: usize,
        actual: usize,
    },

    #[error("matrix dimensions or encoded byte length overflowed")]
    Overflow,

    #[error("matvec input has {actual} columns; expected {expected}")]
    InvalidInputLength { expected: usize, actual: usize },

    #[error("matvec output has {actual} rows; expected {expected}")]
    InvalidOutputLength { expected: usize, actual: usize },

    #[error("invalid attention shape: {0}")]
    InvalidAttentionShape(&'static str),

    #[error("top-k count {k} is outside 1..={length}")]
    InvalidTopK { k: usize, length: usize },

    #[error("mapped buffer address {address:#x} is not aligned to its {page_size}-byte host page")]
    MappedBufferAlignment { address: usize, page_size: usize },

    #[error("mapped buffer range {offset}..{end} exceeds its {length}-byte file mapping")]
    MappedRange {
        offset: usize,
        end: usize,
        length: usize,
    },

    #[error("Metal KV cache is at position {actual}; expected {expected}")]
    KvPosition { expected: usize, actual: usize },

    #[error("Metal KV cache width is {actual}; expected {expected}")]
    KvWidth { expected: usize, actual: usize },

    #[error("Metal KV cache position {position} exceeds its context limit {limit}")]
    KvContext { position: usize, limit: usize },

    #[error("could not create a Metal command buffer or compute encoder")]
    CommandEncoding,

    #[error("Metal command failed: {0}")]
    Command(String),
}

#[cfg(target_os = "macos")]
mod implementation {
    use std::{
        alloc::{Layout, alloc_zeroed, dealloc},
        ffi::c_void,
        ptr::{self, NonNull},
        slice,
        sync::Arc,
    };

    use block2::RcBlock;
    use memmap2::Mmap;
    use objc2::{rc::Retained, runtime::ProtocolObject};
    use objc2_foundation::NSString;
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
        MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
        MTLLibrary, MTLResourceOptions, MTLSize,
    };

    use super::MetalError;
    use crate::{gguf::TensorType, safetensors::SafeDtype};

    const SHADERS: &str = include_str!("../metal/kernels.metal");
    const VECTOR_ADD: &str = "vector_add";
    const RMS_NORM: &str = "rms_norm";
    const ROPE_NEOX: &str = "rope_neox";
    const GELU_MUL: &str = "gelu_mul";
    const STABLE_SOFTMAX: &str = "stable_softmax";
    const LOGIT_SOFTCAP: &str = "logit_softcap";
    const DETERMINISTIC_ARGMAX: &str = "deterministic_argmax";
    const MATVEC_F32: &str = "matvec_f32";
    const MATVEC_F16: &str = "matvec_f16";
    const MATVEC_BF16: &str = "matvec_bf16";
    const MATVEC_F8_E4M3FN: &str = "matvec_f8_e4m3fn";
    const MATVEC_Q5_0: &str = "matvec_q5_0";
    const MATVEC_Q8_0: &str = "matvec_q8_0";
    const MATVEC_Q4_K: &str = "matvec_q4_k";
    const MATVEC_Q6_K: &str = "matvec_q6_k";
    const ATTENTION_SCORES: &str = "attention_scores";
    const ATTENTION_SOFTMAX: &str = "attention_softmax";
    const ATTENTION_VALUES: &str = "attention_values";
    const TOP_K_SOFTMAX: &str = "top_k_softmax";
    const REDUCTION_THREADS: usize = 256;

    #[link(name = "CoreGraphics", kind = "framework")]
    unsafe extern "C" {}

    #[derive(Debug)]
    pub struct MetalContext {
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
        vector_add: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        rms_norm: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        rope_neox: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        gelu_mul: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        stable_softmax: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        logit_softcap: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        deterministic_argmax: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        matvec_f32: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        matvec_f16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        matvec_bf16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        matvec_f8_e4m3fn: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        matvec_q5_0: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        matvec_q8_0: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        matvec_q4_k: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        matvec_q6_k: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        attention_scores: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        attention_softmax: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        attention_values: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        top_k_softmax: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    }

    #[derive(Debug)]
    pub struct MetalMappedBuffer {
        buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        mapping: Arc<Mmap>,
    }

    #[derive(Debug)]
    pub struct MetalOwnedBuffer {
        buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        length: usize,
    }

    impl MetalOwnedBuffer {
        pub fn len(&self) -> usize {
            self.length
        }

        pub fn is_empty(&self) -> bool {
            self.length == 0
        }

        pub fn as_bytes(&self) -> &[u8] {
            // SAFETY: the retained shared buffer owns at least `length` bytes
            // and the immutable borrow prevents simultaneous host mutation.
            unsafe {
                slice::from_raw_parts(self.buffer.contents().as_ptr().cast::<u8>(), self.length)
            }
        }

        pub fn as_mut_bytes(&mut self) -> &mut [u8] {
            // SAFETY: the retained shared buffer owns at least `length` bytes,
            // and the exclusive wrapper borrow prevents simultaneous host use.
            unsafe {
                slice::from_raw_parts_mut(self.buffer.contents().as_ptr().cast::<u8>(), self.length)
            }
        }
    }

    #[derive(Debug)]
    pub struct MetalKvLayer {
        keys: Retained<ProtocolObject<dyn MTLBuffer>>,
        values: Retained<ProtocolObject<dyn MTLBuffer>>,
        width: usize,
        capacity: usize,
        max_context: usize,
        position: usize,
    }

    impl MetalKvLayer {
        pub fn position(&self) -> usize {
            self.position
        }

        pub fn capacity(&self) -> usize {
            self.capacity
        }

        pub fn allocated_bytes(&self) -> usize {
            self.capacity
                .saturating_mul(self.width)
                .saturating_mul(std::mem::size_of::<f32>())
                .saturating_mul(2)
        }

        pub fn clear(&mut self) {
            self.position = 0;
        }

        pub fn truncate(&mut self, position: usize) {
            self.position = self.position.min(position);
        }
    }

    impl MetalMappedBuffer {
        pub fn len(&self) -> usize {
            self.mapping.len()
        }

        pub fn is_empty(&self) -> bool {
            self.mapping.is_empty()
        }
    }

    impl MetalContext {
        pub fn new() -> Result<Self, MetalError> {
            let device = MTLCreateSystemDefaultDevice().ok_or(MetalError::NoDevice)?;
            let queue = device.newCommandQueue().ok_or(MetalError::NoCommandQueue)?;
            let source = NSString::from_str(SHADERS);
            let library = device
                .newLibraryWithSource_options_error(&source, None)
                .map_err(|error| MetalError::Library(error.to_string()))?;
            let vector_add = pipeline(&device, &library, VECTOR_ADD)?;
            let rms_norm = pipeline(&device, &library, RMS_NORM)?;
            let rope_neox = pipeline(&device, &library, ROPE_NEOX)?;
            let gelu_mul = pipeline(&device, &library, GELU_MUL)?;
            let stable_softmax = pipeline(&device, &library, STABLE_SOFTMAX)?;
            let logit_softcap = pipeline(&device, &library, LOGIT_SOFTCAP)?;
            let deterministic_argmax = pipeline(&device, &library, DETERMINISTIC_ARGMAX)?;
            let matvec_f32 = pipeline(&device, &library, MATVEC_F32)?;
            let matvec_f16 = pipeline(&device, &library, MATVEC_F16)?;
            let matvec_bf16 = pipeline(&device, &library, MATVEC_BF16)?;
            let matvec_f8_e4m3fn = pipeline(&device, &library, MATVEC_F8_E4M3FN)?;
            let matvec_q5_0 = pipeline(&device, &library, MATVEC_Q5_0)?;
            let matvec_q8_0 = pipeline(&device, &library, MATVEC_Q8_0)?;
            let matvec_q4_k = pipeline(&device, &library, MATVEC_Q4_K)?;
            let matvec_q6_k = pipeline(&device, &library, MATVEC_Q6_K)?;
            let attention_scores = pipeline(&device, &library, ATTENTION_SCORES)?;
            let attention_softmax = pipeline(&device, &library, ATTENTION_SOFTMAX)?;
            let attention_values = pipeline(&device, &library, ATTENTION_VALUES)?;
            let top_k_softmax = pipeline(&device, &library, TOP_K_SOFTMAX)?;
            Ok(Self {
                device,
                queue,
                vector_add,
                rms_norm,
                rope_neox,
                gelu_mul,
                stable_softmax,
                logit_softcap,
                deterministic_argmax,
                matvec_f32,
                matvec_f16,
                matvec_bf16,
                matvec_f8_e4m3fn,
                matvec_q5_0,
                matvec_q8_0,
                matvec_q4_k,
                matvec_q6_k,
                attention_scores,
                attention_softmax,
                attention_values,
                top_k_softmax,
            })
        }

        pub fn device_name(&self) -> String {
            self.device.name().to_string()
        }

        /// Wrap an immutable file mapping without copying it. The returned
        /// object retains the mapping until Metal releases its buffer view.
        pub fn map_read_only(&self, mapping: Arc<Mmap>) -> Result<MetalMappedBuffer, MetalError> {
            let page_size = host_page_size()?;
            let address = mapping.as_ptr() as usize;
            if !address.is_multiple_of(page_size) {
                return Err(MetalError::MappedBufferAlignment { address, page_size });
            }
            let mapped_length = mapping.len();
            if mapped_length == 0 {
                return Err(MetalError::BufferAllocation { bytes: 0 });
            }
            let buffer_length = mapped_length
                .checked_add(page_size - 1)
                .map(|rounded| rounded / page_size * page_size)
                .ok_or(MetalError::BufferAllocation {
                    bytes: mapped_length,
                })?;
            let pointer = NonNull::new(mapping.as_ptr().cast_mut().cast::<c_void>()).ok_or(
                MetalError::BufferAllocation {
                    bytes: mapped_length,
                },
            )?;
            // SAFETY: memmap2 creates a host-page-aligned mapping. The final
            // virtual page is mapped even when the file ends partway through
            // it, and every dispatch is range-checked against `mapped_length`.
            // The returned wrapper retains the mapping beyond the MTLBuffer.
            let buffer = unsafe {
                self.device
                    .newBufferWithBytesNoCopy_length_options_deallocator(
                        pointer,
                        buffer_length,
                        MTLResourceOptions::StorageModeShared,
                        None,
                    )
            }
            .ok_or(MetalError::BufferAllocation {
                bytes: mapped_length,
            })?;
            Ok(MetalMappedBuffer { buffer, mapping })
        }

        pub fn new_owned_buffer(&self, length: usize) -> Result<MetalOwnedBuffer, MetalError> {
            if length == 0 {
                return Err(MetalError::BufferAllocation { bytes: 0 });
            }
            Ok(MetalOwnedBuffer {
                buffer: self.empty_buffer::<u8>(length)?,
                length,
            })
        }

        pub fn add(
            &self,
            left: &[f32],
            right: &[f32],
            output: &mut [f32],
        ) -> Result<(), MetalError> {
            if left.len() != right.len() || left.len() != output.len() {
                return Err(MetalError::VectorLength {
                    left: left.len(),
                    right: right.len(),
                    output: output.len(),
                });
            }
            if left.is_empty() {
                return Ok(());
            }
            let count = u32::try_from(left.len()).map_err(|_| MetalError::VectorTooLarge)?;
            let left_buffer = self.buffer_with_data(left)?;
            let right_buffer = self.buffer_with_data(right)?;
            let output_buffer = self.empty_buffer::<f32>(output.len())?;
            let count_buffer = self.buffer_with_data(&[count])?;
            self.execute(
                &self.vector_add,
                &[&left_buffer, &right_buffer, &output_buffer, &count_buffer],
                left.len(),
                left.len().min(REDUCTION_THREADS),
            )?;
            copy_buffer(&output_buffer, output);
            Ok(())
        }

        pub fn rms_norm(
            &self,
            input: &[f32],
            weight: Option<&[f32]>,
            epsilon: f32,
            output: &mut [f32],
        ) -> Result<(), MetalError> {
            require_nonempty("RMSNorm", input)?;
            validate_vector_length("RMSNorm", "output", input.len(), output.len())?;
            if let Some(weight) = weight {
                validate_vector_length("RMSNorm", "weight", input.len(), weight.len())?;
            }
            require_positive("RMSNorm epsilon", epsilon)?;
            if input.iter().any(|value| !value.is_finite()) {
                return Err(MetalError::NonFinite {
                    operation: "RMSNorm",
                });
            }

            let count = count_u32(input.len())?;
            let input_buffer = self.buffer_with_data(input)?;
            let weight_buffer = self.buffer_with_data(weight.unwrap_or(input))?;
            let output_buffer = self.empty_buffer::<f32>(output.len())?;
            let count_buffer = self.buffer_with_data(&[count])?;
            let epsilon_buffer = self.buffer_with_data(&[epsilon])?;
            let weighted_buffer = self.buffer_with_data(&[u32::from(weight.is_some())])?;
            self.execute(
                &self.rms_norm,
                &[
                    &input_buffer,
                    &weight_buffer,
                    &output_buffer,
                    &count_buffer,
                    &epsilon_buffer,
                    &weighted_buffer,
                ],
                REDUCTION_THREADS,
                REDUCTION_THREADS,
            )?;
            copy_buffer(&output_buffer, output);
            Ok(())
        }

        pub fn rope_neox_in_place(
            &self,
            values: &mut [f32],
            heads: usize,
            head_dimension: usize,
            rotated_pairs: usize,
            position: usize,
            theta: f32,
        ) -> Result<(), MetalError> {
            if heads
                .checked_mul(head_dimension)
                .filter(|length| *length == values.len())
                .is_none()
            {
                return Err(MetalError::InvalidRopeShape {
                    length: values.len(),
                    heads,
                    head_dimension,
                });
            }
            if !head_dimension.is_multiple_of(2) || rotated_pairs > head_dimension / 2 {
                return Err(MetalError::InvalidRopePairs);
            }
            require_positive("NeoX RoPE theta", theta)?;
            if rotated_pairs == 0 || heads == 0 {
                return Ok(());
            }

            let pairs = heads
                .checked_mul(rotated_pairs)
                .ok_or(MetalError::VectorTooLarge)?;
            let values_buffer = self.buffer_with_data(values)?;
            let head_dimension_buffer = self.buffer_with_data(&[count_u32(head_dimension)?])?;
            let rotated_pairs_buffer = self.buffer_with_data(&[count_u32(rotated_pairs)?])?;
            let position_buffer = self.buffer_with_data(&[count_u32(position)?])?;
            let theta_buffer = self.buffer_with_data(&[theta])?;
            self.execute(
                &self.rope_neox,
                &[
                    &values_buffer,
                    &head_dimension_buffer,
                    &rotated_pairs_buffer,
                    &position_buffer,
                    &theta_buffer,
                ],
                pairs,
                pairs.min(REDUCTION_THREADS),
            )?;
            copy_buffer(&values_buffer, values);
            Ok(())
        }

        pub fn gelu_mul(
            &self,
            gate: &[f32],
            up: &[f32],
            output: &mut [f32],
        ) -> Result<(), MetalError> {
            require_nonempty("GELU multiply", gate)?;
            validate_vector_length("GELU multiply", "up", gate.len(), up.len())?;
            validate_vector_length("GELU multiply", "output", gate.len(), output.len())?;
            let count = count_u32(gate.len())?;
            let gate_buffer = self.buffer_with_data(gate)?;
            let up_buffer = self.buffer_with_data(up)?;
            let output_buffer = self.empty_buffer::<f32>(output.len())?;
            let count_buffer = self.buffer_with_data(&[count])?;
            self.execute(
                &self.gelu_mul,
                &[&gate_buffer, &up_buffer, &output_buffer, &count_buffer],
                gate.len(),
                gate.len().min(REDUCTION_THREADS),
            )?;
            copy_buffer(&output_buffer, output);
            Ok(())
        }

        pub fn softmax(&self, input: &[f32], output: &mut [f32]) -> Result<(), MetalError> {
            require_nonempty("softmax", input)?;
            validate_vector_length("softmax", "output", input.len(), output.len())?;
            if input.iter().any(|value| !value.is_finite()) {
                return Err(MetalError::NonFinite {
                    operation: "softmax",
                });
            }
            let count = count_u32(input.len())?;
            let input_buffer = self.buffer_with_data(input)?;
            let output_buffer = self.empty_buffer::<f32>(output.len())?;
            let count_buffer = self.buffer_with_data(&[count])?;
            self.execute(
                &self.stable_softmax,
                &[&input_buffer, &output_buffer, &count_buffer],
                REDUCTION_THREADS,
                REDUCTION_THREADS,
            )?;
            copy_buffer(&output_buffer, output);
            Ok(())
        }

        pub fn softcap(
            &self,
            input: &[f32],
            cap: f32,
            output: &mut [f32],
        ) -> Result<(), MetalError> {
            require_nonempty("logit softcap", input)?;
            validate_vector_length("logit softcap", "output", input.len(), output.len())?;
            require_positive("logit softcap", cap)?;
            let count = count_u32(input.len())?;
            let input_buffer = self.buffer_with_data(input)?;
            let output_buffer = self.empty_buffer::<f32>(output.len())?;
            let count_buffer = self.buffer_with_data(&[count])?;
            let cap_buffer = self.buffer_with_data(&[cap])?;
            self.execute(
                &self.logit_softcap,
                &[&input_buffer, &output_buffer, &count_buffer, &cap_buffer],
                input.len(),
                input.len().min(REDUCTION_THREADS),
            )?;
            copy_buffer(&output_buffer, output);
            Ok(())
        }

        pub fn argmax(&self, input: &[f32]) -> Result<usize, MetalError> {
            require_nonempty("argmax", input)?;
            if input.iter().any(|value| !value.is_finite()) {
                return Err(MetalError::NonFinite {
                    operation: "argmax",
                });
            }
            let count = count_u32(input.len())?;
            let input_buffer = self.buffer_with_data(input)?;
            let output_buffer = self.empty_buffer::<u32>(1)?;
            let count_buffer = self.buffer_with_data(&[count])?;
            self.execute(
                &self.deterministic_argmax,
                &[&input_buffer, &output_buffer, &count_buffer],
                REDUCTION_THREADS,
                REDUCTION_THREADS,
            )?;
            let mut output = [0_u32];
            copy_buffer(&output_buffer, &mut output);
            Ok(output[0] as usize)
        }

        /// Multiply a row-major GGUF matrix by one FP32 vector without
        /// dequantizing the matrix on the CPU.
        pub fn matvec(
            &self,
            kind: TensorType,
            rows: usize,
            columns: usize,
            encoded: &[u8],
            input: &[f32],
            output: &mut [f32],
        ) -> Result<(), MetalError> {
            let (pipeline, row_bytes) =
                self.prepare_matvec(kind, rows, columns, encoded.len(), input, output)?;
            if rows == 0 {
                return Ok(());
            }
            require_nonempty("matvec", input)?;

            let encoded_buffer = self.buffer_with_data(encoded)?;
            let input_buffer = self.buffer_with_data(input)?;
            let output_buffer = self.empty_buffer::<f32>(output.len())?;
            let columns_buffer = self.buffer_with_data(&[count_u32(columns)?])?;
            let row_bytes_buffer = self.buffer_with_data(&[count_u32(row_bytes)?])?;
            self.execute(
                pipeline,
                &[
                    &encoded_buffer,
                    &input_buffer,
                    &output_buffer,
                    &columns_buffer,
                    &row_bytes_buffer,
                ],
                rows,
                rows.min(REDUCTION_THREADS),
            )?;
            copy_buffer(&output_buffer, output);
            Ok(())
        }

        /// Execute GEMV directly from a retained whole-file mapping.
        #[allow(clippy::too_many_arguments)]
        pub fn matvec_mapped(
            &self,
            kind: TensorType,
            rows: usize,
            columns: usize,
            mapping: &MetalMappedBuffer,
            encoded_offset: usize,
            encoded_length: usize,
            input: &[f32],
            output: &mut [f32],
        ) -> Result<(), MetalError> {
            let (pipeline, row_bytes) =
                self.prepare_matvec(kind, rows, columns, encoded_length, input, output)?;
            let end = encoded_offset
                .checked_add(encoded_length)
                .ok_or(MetalError::Overflow)?;
            if end > mapping.len() {
                return Err(MetalError::MappedRange {
                    offset: encoded_offset,
                    end,
                    length: mapping.len(),
                });
            }
            if rows == 0 {
                return Ok(());
            }
            require_nonempty("matvec", input)?;

            let input_buffer = self.buffer_with_data(input)?;
            let output_buffer = self.empty_buffer::<f32>(output.len())?;
            let columns_buffer = self.buffer_with_data(&[count_u32(columns)?])?;
            let row_bytes_buffer = self.buffer_with_data(&[count_u32(row_bytes)?])?;
            self.execute_with_first_offset(
                pipeline,
                &mapping.buffer,
                encoded_offset,
                &[
                    &input_buffer,
                    &output_buffer,
                    &columns_buffer,
                    &row_bytes_buffer,
                ],
                rows,
                rows.min(REDUCTION_THREADS),
            )?;
            copy_buffer(&output_buffer, output);
            Ok(())
        }

        /// Execute GEMV directly from one or two retained safetensors shard mappings.
        #[allow(clippy::too_many_arguments)]
        pub fn matvec_safetensor_mapped(
            &self,
            dtype: SafeDtype,
            rows: usize,
            columns: usize,
            mapping: &MetalMappedBuffer,
            encoded_offset: usize,
            encoded_length: usize,
            scale: Option<(&MetalMappedBuffer, usize, usize)>,
            input: &[f32],
            output: &mut [f32],
        ) -> Result<(), MetalError> {
            match dtype {
                SafeDtype::F32 => {
                    return self.matvec_mapped(
                        TensorType::F32,
                        rows,
                        columns,
                        mapping,
                        encoded_offset,
                        encoded_length,
                        input,
                        output,
                    );
                }
                SafeDtype::F16 => {
                    return self.matvec_mapped(
                        TensorType::F16,
                        rows,
                        columns,
                        mapping,
                        encoded_offset,
                        encoded_length,
                        input,
                        output,
                    );
                }
                SafeDtype::Bf16 | SafeDtype::F8E4M3 => {}
                _ => return Err(MetalError::UnsupportedTensorType("safetensors dtype")),
            }
            if input.len() != columns {
                return Err(MetalError::InvalidInputLength {
                    expected: columns,
                    actual: input.len(),
                });
            }
            if output.len() != rows {
                return Err(MetalError::InvalidOutputLength {
                    expected: rows,
                    actual: output.len(),
                });
            }
            let byte_width = if dtype == SafeDtype::Bf16 { 2 } else { 1 };
            let expected = rows
                .checked_mul(columns)
                .and_then(|elements| elements.checked_mul(byte_width))
                .ok_or(MetalError::Overflow)?;
            if encoded_length != expected {
                return Err(MetalError::InvalidByteLength {
                    kind: if dtype == SafeDtype::Bf16 {
                        "BF16"
                    } else {
                        "F8_E4M3FN"
                    },
                    expected,
                    actual: encoded_length,
                });
            }
            checked_mapped_range(mapping, encoded_offset, encoded_length)?;
            if rows == 0 {
                return Ok(());
            }
            require_nonempty("safetensors matvec", input)?;
            let input_buffer = self.buffer_with_data(input)?;
            let output_buffer = self.empty_buffer::<f32>(output.len())?;
            let columns_buffer = self.buffer_with_data(&[count_u32(columns)?])?;
            if dtype == SafeDtype::Bf16 {
                let row_bytes_buffer = self.buffer_with_data(&[count_u32(columns * 2)?])?;
                self.execute_with_first_offset(
                    &self.matvec_bf16,
                    &mapping.buffer,
                    encoded_offset,
                    &[
                        &input_buffer,
                        &output_buffer,
                        &columns_buffer,
                        &row_bytes_buffer,
                    ],
                    rows,
                    rows.min(REDUCTION_THREADS),
                )?;
            } else {
                let (scale_mapping, scale_offset, scale_length) =
                    scale.ok_or(MetalError::InvalidByteLength {
                        kind: "F8_E4M3FN scale",
                        expected: rows.div_ceil(128) * columns.div_ceil(128) * 4,
                        actual: 0,
                    })?;
                let expected_scale = rows
                    .div_ceil(128)
                    .checked_mul(columns.div_ceil(128))
                    .and_then(|elements| elements.checked_mul(4))
                    .ok_or(MetalError::Overflow)?;
                if scale_length != expected_scale {
                    return Err(MetalError::InvalidByteLength {
                        kind: "F8_E4M3FN scale",
                        expected: expected_scale,
                        actual: scale_length,
                    });
                }
                checked_mapped_range(scale_mapping, scale_offset, scale_length)?;
                let scale_columns_buffer =
                    self.buffer_with_data(&[count_u32(columns.div_ceil(128))?])?;
                self.execute_with_two_offsets(
                    &self.matvec_f8_e4m3fn,
                    (&mapping.buffer, encoded_offset),
                    (&scale_mapping.buffer, scale_offset),
                    &[
                        &input_buffer,
                        &output_buffer,
                        &columns_buffer,
                        &scale_columns_buffer,
                    ],
                    rows,
                    rows.min(REDUCTION_THREADS),
                )?;
            }
            copy_buffer(&output_buffer, output);
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        pub fn matvec_owned(
            &self,
            kind: TensorType,
            rows: usize,
            columns: usize,
            encoded: &MetalOwnedBuffer,
            encoded_offset: usize,
            encoded_length: usize,
            input: &[f32],
            output: &mut [f32],
        ) -> Result<(), MetalError> {
            let (pipeline, row_bytes) =
                self.prepare_matvec(kind, rows, columns, encoded_length, input, output)?;
            let end = encoded_offset
                .checked_add(encoded_length)
                .ok_or(MetalError::Overflow)?;
            if end > encoded.len() {
                return Err(MetalError::MappedRange {
                    offset: encoded_offset,
                    end,
                    length: encoded.len(),
                });
            }
            if rows == 0 {
                return Ok(());
            }
            require_nonempty("matvec", input)?;
            let input_buffer = self.buffer_with_data(input)?;
            let output_buffer = self.empty_buffer::<f32>(output.len())?;
            let columns_buffer = self.buffer_with_data(&[count_u32(columns)?])?;
            let row_bytes_buffer = self.buffer_with_data(&[count_u32(row_bytes)?])?;
            self.execute_with_first_offset(
                pipeline,
                &encoded.buffer,
                encoded_offset,
                &[
                    &input_buffer,
                    &output_buffer,
                    &columns_buffer,
                    &row_bytes_buffer,
                ],
                rows,
                rows.min(REDUCTION_THREADS),
            )?;
            copy_buffer(&output_buffer, output);
            Ok(())
        }

        pub fn new_kv_layer(
            &self,
            width: usize,
            max_context: usize,
            sliding_window: Option<usize>,
        ) -> Result<MetalKvLayer, MetalError> {
            let capacity = sliding_window.unwrap_or(max_context).min(max_context);
            if width == 0 || capacity == 0 {
                return Err(MetalError::InvalidAttentionShape(
                    "KV width and capacity must be non-zero",
                ));
            }
            let elements = width.checked_mul(capacity).ok_or(MetalError::Overflow)?;
            Ok(MetalKvLayer {
                keys: self.empty_buffer::<f32>(elements)?,
                values: self.empty_buffer::<f32>(elements)?,
                width,
                capacity,
                max_context,
                position: 0,
            })
        }

        pub fn append_kv(
            &self,
            layer: &mut MetalKvLayer,
            position: usize,
            key: &[f32],
            value: &[f32],
        ) -> Result<(), MetalError> {
            if layer.position != position {
                return Err(MetalError::KvPosition {
                    expected: position,
                    actual: layer.position,
                });
            }
            if position >= layer.max_context {
                return Err(MetalError::KvContext {
                    position,
                    limit: layer.max_context,
                });
            }
            if key.len() != layer.width || value.len() != layer.width {
                return Err(MetalError::KvWidth {
                    expected: layer.width,
                    actual: key.len().min(value.len()),
                });
            }
            let slot = position % layer.capacity;
            let element_offset = slot.checked_mul(layer.width).ok_or(MetalError::Overflow)?;
            // SAFETY: both shared buffers contain capacity * width f32 values,
            // the selected ring slot is in range, and all Metal dispatches in
            // this context complete synchronously before host mutation.
            unsafe {
                let key_destination = layer
                    .keys
                    .contents()
                    .as_ptr()
                    .cast::<f32>()
                    .add(element_offset);
                let value_destination = layer
                    .values
                    .contents()
                    .as_ptr()
                    .cast::<f32>()
                    .add(element_offset);
                ptr::copy_nonoverlapping(key.as_ptr(), key_destination, layer.width);
                ptr::copy_nonoverlapping(value.as_ptr(), value_destination, layer.width);
            }
            layer.position += 1;
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        pub fn attention_cached(
            &self,
            query: &[f32],
            cache: &MetalKvLayer,
            query_heads: usize,
            kv_heads: usize,
            head_dimension: usize,
            output: &mut [f32],
        ) -> Result<(), MetalError> {
            if query_heads == 0 || kv_heads == 0 || !query_heads.is_multiple_of(kv_heads) {
                return Err(MetalError::InvalidAttentionShape(
                    "query heads must be a non-zero multiple of KV heads",
                ));
            }
            let query_width = query_heads
                .checked_mul(head_dimension)
                .ok_or(MetalError::Overflow)?;
            let expected_cache_width = kv_heads
                .checked_mul(head_dimension)
                .ok_or(MetalError::Overflow)?;
            if query.len() != query_width || output.len() != query_width {
                return Err(MetalError::InvalidAttentionShape(
                    "query and output must match query_heads * head_dimension",
                ));
            }
            if cache.width != expected_cache_width {
                return Err(MetalError::KvWidth {
                    expected: expected_cache_width,
                    actual: cache.width,
                });
            }
            if cache.position == 0 {
                return Err(MetalError::EmptyInput {
                    operation: "attention",
                });
            }
            let visible = cache.position.min(cache.capacity);
            let start = cache.position - visible;
            self.attention_buffers(
                query,
                &cache.keys,
                &cache.values,
                cache.position,
                cache.capacity,
                cache.width,
                query_heads,
                kv_heads,
                head_dimension,
                start,
                visible,
                output,
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn attention(
            &self,
            query: &[f32],
            keys: &[f32],
            values: &[f32],
            query_heads: usize,
            kv_heads: usize,
            head_dimension: usize,
            window: Option<usize>,
            output: &mut [f32],
        ) -> Result<(), MetalError> {
            if query_heads == 0 || kv_heads == 0 || !query_heads.is_multiple_of(kv_heads) {
                return Err(MetalError::InvalidAttentionShape(
                    "query heads must be a non-zero multiple of KV heads",
                ));
            }
            let query_width = query_heads
                .checked_mul(head_dimension)
                .ok_or(MetalError::Overflow)?;
            let cache_width = kv_heads
                .checked_mul(head_dimension)
                .ok_or(MetalError::Overflow)?;
            if query.len() != query_width || output.len() != query_width {
                return Err(MetalError::InvalidAttentionShape(
                    "query and output must match query_heads * head_dimension",
                ));
            }
            if cache_width == 0
                || keys.len() != values.len()
                || !keys.len().is_multiple_of(cache_width)
            {
                return Err(MetalError::InvalidAttentionShape(
                    "key/value caches must have equal whole-position lengths",
                ));
            }
            let sequence_length = keys.len() / cache_width;
            if sequence_length == 0 {
                return Err(MetalError::EmptyInput {
                    operation: "attention",
                });
            }
            let start = window.map_or(0, |window| sequence_length.saturating_sub(window));
            let visible = sequence_length - start;
            if visible == 0 {
                return Err(MetalError::EmptyInput {
                    operation: "attention window",
                });
            }

            let keys_buffer = self.buffer_with_data(keys)?;
            let values_buffer = self.buffer_with_data(values)?;
            self.attention_buffers(
                query,
                &keys_buffer,
                &values_buffer,
                sequence_length,
                sequence_length,
                cache_width,
                query_heads,
                kv_heads,
                head_dimension,
                start,
                visible,
                output,
            )
        }

        pub fn top_k_softmax(
            &self,
            logits: &[f32],
            k: usize,
        ) -> Result<Vec<(usize, f32)>, MetalError> {
            if k == 0 || k > logits.len() {
                return Err(MetalError::InvalidTopK {
                    k,
                    length: logits.len(),
                });
            }
            if logits.iter().any(|value| !value.is_finite()) {
                return Err(MetalError::NonFinite {
                    operation: "top-k softmax",
                });
            }
            let logits_buffer = self.buffer_with_data(logits)?;
            let indices_buffer = self.empty_buffer::<u32>(k)?;
            let probabilities_buffer = self.empty_buffer::<f32>(k)?;
            let count_buffer = self.buffer_with_data(&[count_u32(logits.len())?])?;
            let k_buffer = self.buffer_with_data(&[count_u32(k)?])?;
            self.execute(
                &self.top_k_softmax,
                &[
                    &logits_buffer,
                    &indices_buffer,
                    &probabilities_buffer,
                    &count_buffer,
                    &k_buffer,
                ],
                1,
                1,
            )?;
            let mut indices = vec![0_u32; k];
            let mut probabilities = vec![0.0_f32; k];
            copy_buffer(&indices_buffer, &mut indices);
            copy_buffer(&probabilities_buffer, &mut probabilities);
            Ok(indices
                .into_iter()
                .zip(probabilities)
                .map(|(index, probability)| (index as usize, probability))
                .collect())
        }

        #[allow(clippy::too_many_arguments)]
        fn attention_buffers(
            &self,
            query: &[f32],
            keys: &ProtocolObject<dyn MTLBuffer>,
            values: &ProtocolObject<dyn MTLBuffer>,
            sequence_length: usize,
            cache_capacity: usize,
            cache_width: usize,
            query_heads: usize,
            kv_heads: usize,
            head_dimension: usize,
            start: usize,
            visible: usize,
            output: &mut [f32],
        ) -> Result<(), MetalError> {
            let query_buffer = self.buffer_with_data(query)?;
            let scores_len = query_heads
                .checked_mul(visible)
                .ok_or(MetalError::Overflow)?;
            let scores_buffer = self.empty_buffer::<f32>(scores_len)?;
            let output_buffer = self.empty_buffer::<f32>(output.len())?;
            let sequence_buffer = self.buffer_with_data(&[count_u32(sequence_length)?])?;
            let cache_capacity_buffer = self.buffer_with_data(&[count_u32(cache_capacity)?])?;
            let cache_width_buffer = self.buffer_with_data(&[count_u32(cache_width)?])?;
            let query_heads_buffer = self.buffer_with_data(&[count_u32(query_heads)?])?;
            let kv_heads_buffer = self.buffer_with_data(&[count_u32(kv_heads)?])?;
            let head_dimension_buffer = self.buffer_with_data(&[count_u32(head_dimension)?])?;
            let start_buffer = self.buffer_with_data(&[count_u32(start)?])?;
            let visible_buffer = self.buffer_with_data(&[count_u32(visible)?])?;

            self.execute(
                &self.attention_scores,
                &[
                    &query_buffer,
                    keys,
                    &scores_buffer,
                    &sequence_buffer,
                    &cache_width_buffer,
                    &query_heads_buffer,
                    &kv_heads_buffer,
                    &head_dimension_buffer,
                    &start_buffer,
                    &cache_capacity_buffer,
                ],
                scores_len,
                scores_len.min(REDUCTION_THREADS),
            )?;
            self.execute(
                &self.attention_softmax,
                &[&scores_buffer, &visible_buffer],
                query_heads
                    .checked_mul(REDUCTION_THREADS)
                    .ok_or(MetalError::Overflow)?,
                REDUCTION_THREADS,
            )?;
            self.execute(
                &self.attention_values,
                &[
                    &scores_buffer,
                    values,
                    &output_buffer,
                    &visible_buffer,
                    &start_buffer,
                    &cache_width_buffer,
                    &query_heads_buffer,
                    &kv_heads_buffer,
                    &head_dimension_buffer,
                    &cache_capacity_buffer,
                ],
                output.len(),
                output.len().min(REDUCTION_THREADS),
            )?;
            copy_buffer(&output_buffer, output);
            Ok(())
        }

        fn prepare_matvec(
            &self,
            kind: TensorType,
            rows: usize,
            columns: usize,
            encoded_length: usize,
            input: &[f32],
            output: &[f32],
        ) -> Result<(&ProtocolObject<dyn MTLComputePipelineState>, usize), MetalError> {
            if input.len() != columns {
                return Err(MetalError::InvalidInputLength {
                    expected: columns,
                    actual: input.len(),
                });
            }
            if output.len() != rows {
                return Err(MetalError::InvalidOutputLength {
                    expected: rows,
                    actual: output.len(),
                });
            }
            let pipeline = match kind {
                TensorType::F32 => &*self.matvec_f32,
                TensorType::F16 => &*self.matvec_f16,
                TensorType::Q5_0 => &*self.matvec_q5_0,
                TensorType::Q8_0 => &*self.matvec_q8_0,
                TensorType::Q4K => &*self.matvec_q4_k,
                TensorType::Q6K => &*self.matvec_q6_k,
                _ => return Err(MetalError::UnsupportedTensorType(kind.name())),
            };
            let row_bytes = kind
                .row_byte_len(u64::try_from(columns).map_err(|_| MetalError::Overflow)?)
                .ok_or(MetalError::InvalidElementCount {
                    kind: kind.name(),
                    elements: columns,
                    block_elements: kind.block_elements(),
                })?;
            let row_bytes = usize::try_from(row_bytes).map_err(|_| MetalError::Overflow)?;
            let matrix_bytes = rows.checked_mul(row_bytes).ok_or(MetalError::Overflow)?;
            if encoded_length != matrix_bytes {
                return Err(MetalError::InvalidByteLength {
                    kind: kind.name(),
                    expected: matrix_bytes,
                    actual: encoded_length,
                });
            }
            Ok((pipeline, row_bytes))
        }

        fn execute(
            &self,
            pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
            buffers: &[&ProtocolObject<dyn MTLBuffer>],
            grid_width: usize,
            threadgroup_width: usize,
        ) -> Result<(), MetalError> {
            let command_buffer = self
                .queue
                .commandBuffer()
                .ok_or(MetalError::CommandEncoding)?;
            let encoder = command_buffer
                .computeCommandEncoder()
                .ok_or(MetalError::CommandEncoding)?;
            encoder.setComputePipelineState(pipeline);
            for (index, buffer) in buffers.iter().enumerate() {
                // SAFETY: callers retain every buffer until this synchronous
                // execution completes, and every binding uses a zero offset.
                unsafe { encoder.setBuffer_offset_atIndex(Some(buffer), 0, index) };
            }
            encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: grid_width,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: threadgroup_width,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
            command_buffer.commit();
            command_buffer.waitUntilCompleted();
            if command_buffer.status() == MTLCommandBufferStatus::Error {
                return Err(MetalError::Command(command_buffer.error().map_or_else(
                    || "unknown command-buffer error".to_owned(),
                    |error| error.to_string(),
                )));
            }
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        fn execute_with_first_offset(
            &self,
            pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
            first: &ProtocolObject<dyn MTLBuffer>,
            first_offset: usize,
            remaining: &[&ProtocolObject<dyn MTLBuffer>],
            grid_width: usize,
            threadgroup_width: usize,
        ) -> Result<(), MetalError> {
            let command_buffer = self
                .queue
                .commandBuffer()
                .ok_or(MetalError::CommandEncoding)?;
            let encoder = command_buffer
                .computeCommandEncoder()
                .ok_or(MetalError::CommandEncoding)?;
            encoder.setComputePipelineState(pipeline);
            // SAFETY: the mapped range and offset are checked by the caller,
            // and every buffer remains retained through synchronous completion.
            unsafe { encoder.setBuffer_offset_atIndex(Some(first), first_offset, 0) };
            for (index, buffer) in remaining.iter().enumerate() {
                // SAFETY: all remaining buffers are retained and bound at zero.
                unsafe { encoder.setBuffer_offset_atIndex(Some(buffer), 0, index + 1) };
            }
            encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: grid_width,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: threadgroup_width,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
            command_buffer.commit();
            command_buffer.waitUntilCompleted();
            if command_buffer.status() == MTLCommandBufferStatus::Error {
                return Err(MetalError::Command(command_buffer.error().map_or_else(
                    || "unknown command-buffer error".to_owned(),
                    |error| error.to_string(),
                )));
            }
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        fn execute_with_two_offsets(
            &self,
            pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
            first: (&ProtocolObject<dyn MTLBuffer>, usize),
            second: (&ProtocolObject<dyn MTLBuffer>, usize),
            remaining: &[&ProtocolObject<dyn MTLBuffer>],
            grid_width: usize,
            threadgroup_width: usize,
        ) -> Result<(), MetalError> {
            let command_buffer = self
                .queue
                .commandBuffer()
                .ok_or(MetalError::CommandEncoding)?;
            let encoder = command_buffer
                .computeCommandEncoder()
                .ok_or(MetalError::CommandEncoding)?;
            encoder.setComputePipelineState(pipeline);
            // SAFETY: callers range-check both retained mappings and retain all
            // buffers through this synchronous command completion.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(first.0), first.1, 0);
                encoder.setBuffer_offset_atIndex(Some(second.0), second.1, 1);
            }
            for (index, buffer) in remaining.iter().enumerate() {
                // SAFETY: each retained buffer remains live until completion.
                unsafe { encoder.setBuffer_offset_atIndex(Some(buffer), 0, index + 2) };
            }
            encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: grid_width,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: threadgroup_width,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
            command_buffer.commit();
            command_buffer.waitUntilCompleted();
            if command_buffer.status() == MTLCommandBufferStatus::Error {
                return Err(MetalError::Command(command_buffer.error().map_or_else(
                    || "unknown command-buffer error".to_owned(),
                    |error| error.to_string(),
                )));
            }
            Ok(())
        }

        fn buffer_with_data<T>(
            &self,
            values: &[T],
        ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
            let bytes = values
                .len()
                .checked_mul(std::mem::size_of::<T>())
                .ok_or(MetalError::BufferAllocation { bytes: usize::MAX })?;
            let pointer = NonNull::new(values.as_ptr().cast_mut().cast::<c_void>())
                .ok_or(MetalError::BufferAllocation { bytes })?;
            // SAFETY: `pointer` addresses `bytes` initialized bytes. Metal
            // copies them into a new retained shared buffer before returning.
            unsafe {
                self.device
                    .newBufferWithBytes_length_options(
                        pointer,
                        bytes,
                        MTLResourceOptions::StorageModeShared,
                    )
                    .ok_or(MetalError::BufferAllocation { bytes })
            }
        }

        fn empty_buffer<T>(
            &self,
            elements: usize,
        ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
            let bytes = elements
                .checked_mul(std::mem::size_of::<T>())
                .ok_or(MetalError::BufferAllocation { bytes: usize::MAX })?;
            let page_size = host_page_size()?;
            let allocated_bytes = bytes
                .checked_add(page_size - 1)
                .map(|rounded| rounded / page_size * page_size)
                .ok_or(MetalError::BufferAllocation { bytes })?;
            let layout = Layout::from_size_align(allocated_bytes, page_size)
                .map_err(|_| MetalError::BufferAllocation { bytes })?;
            // SAFETY: `layout` has non-zero size and a power-of-two alignment
            // reported by the operating system.
            let pointer = NonNull::new(unsafe { alloc_zeroed(layout) })
                .ok_or(MetalError::BufferAllocation { bytes })?;
            let opaque_pointer = pointer.cast::<c_void>();
            let deallocator: RcBlock<dyn Fn(NonNull<c_void>, usize)> =
                RcBlock::new(move |pointer: NonNull<c_void>, length: usize| {
                    let layout = Layout::from_size_align(length, page_size)
                        .expect("Metal returned the page-aligned allocation length");
                    // SAFETY: this callback runs exactly once when Metal
                    // releases the no-copy buffer created from this layout.
                    unsafe { dealloc(pointer.cast::<u8>().as_ptr(), layout) };
                });
            // SAFETY: the allocation is page-aligned, remains owned by the
            // returned buffer, and its sendable deallocator releases it.
            let buffer = unsafe {
                self.device
                    .newBufferWithBytesNoCopy_length_options_deallocator(
                        opaque_pointer,
                        allocated_bytes,
                        MTLResourceOptions::StorageModeShared,
                        Some(&deallocator),
                    )
            };
            match buffer {
                Some(buffer) => Ok(buffer),
                None => {
                    // SAFETY: Metal did not accept ownership of the allocation.
                    unsafe { dealloc(pointer.as_ptr(), layout) };
                    Err(MetalError::BufferAllocation { bytes })
                }
            }
        }
    }

    fn copy_buffer<T: Copy>(buffer: &ProtocolObject<dyn MTLBuffer>, output: &mut [T]) {
        // SAFETY: all callers wait for command completion, allocate at least
        // output.len() values of T, and copy into a disjoint Rust slice.
        unsafe {
            output.copy_from_slice(slice::from_raw_parts(
                buffer.contents().as_ptr().cast::<T>(),
                output.len(),
            ));
        }
    }

    fn checked_mapped_range(
        mapping: &MetalMappedBuffer,
        offset: usize,
        length: usize,
    ) -> Result<(), MetalError> {
        let end = offset.checked_add(length).ok_or(MetalError::Overflow)?;
        if end > mapping.len() {
            return Err(MetalError::MappedRange {
                offset,
                end,
                length: mapping.len(),
            });
        }
        Ok(())
    }

    fn count_u32(value: usize) -> Result<u32, MetalError> {
        u32::try_from(value).map_err(|_| MetalError::VectorTooLarge)
    }

    fn require_nonempty(operation: &'static str, values: &[f32]) -> Result<(), MetalError> {
        if values.is_empty() {
            return Err(MetalError::EmptyInput { operation });
        }
        Ok(())
    }

    fn require_positive(operation: &'static str, value: f32) -> Result<(), MetalError> {
        if !value.is_finite() || value <= 0.0 {
            return Err(MetalError::InvalidPositiveParameter { operation });
        }
        Ok(())
    }

    fn validate_vector_length(
        operation: &'static str,
        argument: &'static str,
        expected: usize,
        actual: usize,
    ) -> Result<(), MetalError> {
        if expected != actual {
            return Err(MetalError::InvalidVectorLength {
                operation,
                argument,
                expected,
                actual,
            });
        }
        Ok(())
    }

    fn host_page_size() -> Result<usize, MetalError> {
        // SAFETY: sysconf is thread-safe and has no pointer arguments.
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        usize::try_from(page_size)
            .ok()
            .filter(|size| size.is_power_of_two())
            .ok_or(MetalError::PageSize)
    }

    fn pipeline(
        device: &ProtocolObject<dyn MTLDevice>,
        library: &ProtocolObject<dyn MTLLibrary>,
        name: &'static str,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, MetalError> {
        let function = library
            .newFunctionWithName(&NSString::from_str(name))
            .ok_or(MetalError::MissingFunction(name))?;
        device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|error| MetalError::Pipeline {
                name,
                message: error.to_string(),
            })
    }

    #[cfg(test)]
    mod tests {
        use std::{io::Write, sync::Arc};

        use memmap2::MmapOptions;
        use objc2_metal::MTLBuffer;

        use crate::{cpu, gguf::TensorType, safetensors::SafeDtype};

        use super::MetalContext;

        fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
            assert_eq!(actual.len(), expected.len());
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "index {index}: Metal {actual} differs from CPU {expected} by more than {tolerance}"
                );
            }
        }

        fn assert_close_relative(actual: &[f32], expected: &[f32]) {
            assert_eq!(actual.len(), expected.len());
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                let tolerance = 5.0e-4_f32.max(expected.abs() * 2.0e-5);
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "index {index}: Metal {actual} differs from CPU {expected} by more than {tolerance}"
                );
            }
        }

        fn matvec_fixture(kind: TensorType, rows: usize) -> (usize, Vec<u8>) {
            match kind {
                TensorType::F32 => {
                    let columns = 64;
                    let encoded = (0..rows * columns)
                        .flat_map(|index| {
                            (((index * 17) % 41) as f32 * 0.03125 - 0.625).to_le_bytes()
                        })
                        .collect();
                    (columns, encoded)
                }
                TensorType::F16 => {
                    let columns = 64;
                    let patterns = [0x3000_u16, 0xb400, 0x3800, 0xbc00, 0x3c00, 0x4000];
                    let encoded = (0..rows * columns)
                        .flat_map(|index| patterns[index % patterns.len()].to_le_bytes())
                        .collect();
                    (columns, encoded)
                }
                TensorType::Q5_0 => {
                    let blocks = 3;
                    let mut encoded = Vec::with_capacity(rows * blocks * 22);
                    for row in 0..rows {
                        for block_index in 0..blocks {
                            let mut block = [0_u8; 22];
                            block[..2].copy_from_slice(&0x3400_u16.to_le_bytes());
                            for (index, value) in block[2..].iter_mut().enumerate() {
                                *value = (index * 29 + row * 47 + block_index * 13 + 3) as u8;
                            }
                            encoded.extend_from_slice(&block);
                        }
                    }
                    (32 * blocks, encoded)
                }
                TensorType::Q8_0 => {
                    let blocks = 3;
                    let mut encoded = Vec::with_capacity(rows * blocks * 34);
                    for row in 0..rows {
                        for block_index in 0..blocks {
                            let mut block = [0_u8; 34];
                            block[..2].copy_from_slice(&0x3000_u16.to_le_bytes());
                            for (index, value) in block[2..].iter_mut().enumerate() {
                                *value = (index as i16 * 7 - 101
                                    + row as i16 * 11
                                    + block_index as i16 * 17)
                                    as i8 as u8;
                            }
                            encoded.extend_from_slice(&block);
                        }
                    }
                    (32 * blocks, encoded)
                }
                TensorType::Q4K => {
                    let blocks = 3;
                    let mut encoded = Vec::with_capacity(rows * blocks * 144);
                    for row in 0..rows {
                        for block_index in 0..blocks {
                            let mut block = [0_u8; 144];
                            block[..2].copy_from_slice(&0x3000_u16.to_le_bytes());
                            block[2..4].copy_from_slice(&0x2800_u16.to_le_bytes());
                            for (index, value) in block[4..].iter_mut().enumerate() {
                                *value = (index * 19 + row * 31 + block_index * 43 + 5) as u8;
                            }
                            encoded.extend_from_slice(&block);
                        }
                    }
                    (256 * blocks, encoded)
                }
                TensorType::Q6K => {
                    let blocks = 3;
                    let mut encoded = Vec::with_capacity(rows * blocks * 210);
                    for row in 0..rows {
                        for block_index in 0..blocks {
                            let mut block = [0_u8; 210];
                            for (index, value) in block[..192].iter_mut().enumerate() {
                                *value = (index * 23 + row * 37 + block_index * 41 + 11) as u8;
                            }
                            for (index, value) in block[192..208].iter_mut().enumerate() {
                                *value =
                                    (index as i8 * 5 - 37 + row as i8 * 3 + block_index as i8 * 7)
                                        as u8;
                            }
                            block[208..].copy_from_slice(&0x2800_u16.to_le_bytes());
                            encoded.extend_from_slice(&block);
                        }
                    }
                    (256 * blocks, encoded)
                }
                _ => unreachable!("fixture only covers Metal-supported tensor types"),
            }
        }

        #[test]
        fn creates_device_and_adds_shared_vectors() {
            let context = MetalContext::new().unwrap();
            assert!(context.device_name().contains("Apple"));
            let page_buffer = context.empty_buffer::<f32>(1_024).unwrap();
            assert_eq!(
                page_buffer.contents().as_ptr() as usize % super::host_page_size().unwrap(),
                0
            );
            let mut output = [0.0_f32; 5];
            context
                .add(
                    &[1.0, 2.0, -3.0, 4.5, 0.25],
                    &[5.0, -2.0, 1.0, 0.5, 0.75],
                    &mut output,
                )
                .unwrap();
            assert_eq!(output, [6.0, 0.0, -2.0, 5.0, 1.0]);
        }

        #[test]
        fn gemma_primitives_match_the_cpu_reference() {
            let context = MetalContext::new().unwrap();
            let input: Vec<_> = (0..513)
                .map(|index| ((index % 31) as f32 - 15.0) * 0.071)
                .collect();
            let weight: Vec<_> = (0..input.len())
                .map(|index| 0.8 + (index % 13) as f32 * 0.025)
                .collect();

            let mut expected = vec![0.0; input.len()];
            cpu::rms_norm(&input, Some(&weight), 1.0e-6, &mut expected).unwrap();
            let mut actual = vec![0.0; input.len()];
            context
                .rms_norm(&input, Some(&weight), 1.0e-6, &mut actual)
                .unwrap();
            assert_close(&actual, &expected, 2.0e-6);

            let mut expected_rope: Vec<_> =
                (0..48).map(|index| (index as f32 - 24.0) * 0.125).collect();
            let mut actual_rope = expected_rope.clone();
            cpu::rope_neox_in_place(&mut expected_rope, 3, 16, 6, 37, 10_000.0).unwrap();
            context
                .rope_neox_in_place(&mut actual_rope, 3, 16, 6, 37, 10_000.0)
                .unwrap();
            assert_close(&actual_rope, &expected_rope, 2.0e-5);

            let gate: Vec<_> = input.iter().copied().take(67).collect();
            let up: Vec<_> = gate.iter().map(|value| 1.25 - value * 0.2).collect();
            let expected_gelu: Vec<_> = gate
                .iter()
                .zip(&up)
                .map(|(gate, up)| cpu::gelu_tanh(*gate) * up)
                .collect();
            let mut actual_gelu = vec![0.0; gate.len()];
            context.gelu_mul(&gate, &up, &mut actual_gelu).unwrap();
            assert_close(&actual_gelu, &expected_gelu, 2.0e-6);

            let large_gate = [-80.0, -51.0, 37.0, 66.0, 90.0];
            let large_up = [2.0, -3.0, 4.0, -5.0, 6.0];
            let expected_large: Vec<_> = large_gate
                .iter()
                .zip(large_up)
                .map(|(gate, up)| cpu::gelu_tanh(*gate) * up)
                .collect();
            let mut actual_large = [0.0; 5];
            context
                .gelu_mul(&large_gate, &large_up, &mut actual_large)
                .unwrap();
            assert_close(&actual_large, &expected_large, 2.0e-5);

            let mut expected_softmax = input.clone();
            cpu::softmax_in_place(&mut expected_softmax).unwrap();
            let mut actual_softmax = vec![0.0; input.len()];
            context.softmax(&input, &mut actual_softmax).unwrap();
            assert_close(&actual_softmax, &expected_softmax, 2.0e-7);

            let mut expected_softcap = input.clone();
            cpu::softcap_in_place(&mut expected_softcap, 30.0).unwrap();
            let mut actual_softcap = vec![0.0; input.len()];
            context.softcap(&input, 30.0, &mut actual_softcap).unwrap();
            assert_close(&actual_softcap, &expected_softcap, 2.0e-6);

            let logits = [-2.0, 7.5, 4.0, 7.5, 0.0, 7.49];
            assert_eq!(
                context.argmax(&logits).unwrap(),
                cpu::argmax(&logits).unwrap()
            );
            let expected_top_k = cpu::top_k_softmax(&logits, 4).unwrap();
            let actual_top_k = context.top_k_softmax(&logits, 4).unwrap();
            assert_eq!(
                actual_top_k.iter().map(|item| item.0).collect::<Vec<_>>(),
                expected_top_k.iter().map(|item| item.0).collect::<Vec<_>>()
            );
            assert_close(
                &actual_top_k.iter().map(|item| item.1).collect::<Vec<_>>(),
                &expected_top_k.iter().map(|item| item.1).collect::<Vec<_>>(),
                2.0e-7,
            );
        }

        #[test]
        fn encoded_matvec_kernels_match_the_cpu_reference() {
            let context = MetalContext::new().unwrap();
            for kind in [
                TensorType::F32,
                TensorType::F16,
                TensorType::Q5_0,
                TensorType::Q8_0,
                TensorType::Q4K,
                TensorType::Q6K,
            ] {
                let rows = 3;
                let (columns, encoded) = matvec_fixture(kind, rows);
                let input: Vec<_> = (0..columns)
                    .map(|index| ((index * 11 % 53) as f32 - 26.0) * 0.0078125)
                    .collect();
                let mut expected = vec![0.0; rows];
                cpu::matvec(kind, rows, columns, &encoded, &input, &mut expected).unwrap();
                let mut actual = vec![0.0; rows];
                context
                    .matvec(kind, rows, columns, &encoded, &input, &mut actual)
                    .unwrap();
                assert_close_relative(&actual, &expected);
            }
        }

        #[test]
        fn mapped_matvec_reads_a_nonzero_file_offset_without_copying() {
            let context = MetalContext::new().unwrap();
            let encoded: Vec<_> = [1.0_f32, 2.0, 3.0, 4.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect();
            let offset = 32;
            let mut file = tempfile::tempfile().unwrap();
            file.write_all(&vec![0_u8; offset]).unwrap();
            file.write_all(&encoded).unwrap();
            file.flush().unwrap();
            // SAFETY: the temporary file is not mutated while this immutable
            // mapping and its retained Metal view are alive.
            let mapping = Arc::new(unsafe { MmapOptions::new().map(&file).unwrap() });
            let mapped = context.map_read_only(mapping).unwrap();
            assert!(!mapped.is_empty());
            let mut output = [0.0; 2];
            context
                .matvec_mapped(
                    TensorType::F32,
                    2,
                    2,
                    &mapped,
                    offset,
                    encoded.len(),
                    &[0.5, -1.0],
                    &mut output,
                )
                .unwrap();
            assert_eq!(output, [-1.5, -2.5]);
        }

        #[test]
        fn mapped_safetensors_bf16_and_fp8_match_cpu_values() {
            let context = MetalContext::new().unwrap();
            let bf16_offset = 32;
            let bf16 = [1.0_f32, 2.0, 3.0, -1.0, 0.5, 4.0]
                .into_iter()
                .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
                .collect::<Vec<_>>();
            let fp8_offset = 64;
            let fp8 = vec![0x38_u8; 129 * 130];
            let scale_offset = fp8_offset + fp8.len() + 2;
            let scales = [2.0_f32, 3.0, 5.0, 7.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>();
            let mut contents = vec![0_u8; scale_offset + scales.len()];
            contents[bf16_offset..bf16_offset + bf16.len()].copy_from_slice(&bf16);
            contents[fp8_offset..fp8_offset + fp8.len()].copy_from_slice(&fp8);
            contents[scale_offset..scale_offset + scales.len()].copy_from_slice(&scales);
            let mut file = tempfile::tempfile().unwrap();
            file.write_all(&contents).unwrap();
            file.flush().unwrap();
            // SAFETY: the file remains immutable for the mapping lifetime.
            let mapping = Arc::new(unsafe { MmapOptions::new().map(&file).unwrap() });
            let mapped = context.map_read_only(mapping).unwrap();

            let mut bf16_output = [0.0; 2];
            context
                .matvec_safetensor_mapped(
                    SafeDtype::Bf16,
                    2,
                    3,
                    &mapped,
                    bf16_offset,
                    bf16.len(),
                    None,
                    &[2.0, -1.0, 0.5],
                    &mut bf16_output,
                )
                .unwrap();
            assert_close(&bf16_output, &[1.5, -0.5], 1e-6);

            let mut input = vec![1.0; 130];
            input[129] = 2.0;
            let mut fp8_output = vec![0.0; 129];
            context
                .matvec_safetensor_mapped(
                    SafeDtype::F8E4M3,
                    129,
                    130,
                    &mapped,
                    fp8_offset,
                    fp8.len(),
                    Some((&mapped, scale_offset, scales.len())),
                    &input,
                    &mut fp8_output,
                )
                .unwrap();
            assert_close(&fp8_output[..128], &vec![265.0; 128], 1e-5);
            assert!((fp8_output[128] - 661.0).abs() < 1e-5);
        }

        #[test]
        fn grouped_sliding_attention_matches_the_cpu_reference() {
            let context = MetalContext::new().unwrap();
            let query: [f32; 8] = [1.0, -0.5, 0.25, 0.75, -0.5, 1.0, 0.6, -0.2];
            let keys: [f32; 12] = [
                0.5, 0.1, -0.2, 0.8, 0.9, -0.4, 0.3, 0.7, -0.6, 0.2, 1.0, -0.5,
            ];
            let values: [f32; 12] = [
                10.0, 20.0, 30.0, 40.0, 1.0, 2.0, 3.0, 4.0, -2.0, 0.5, 7.0, -3.0,
            ];
            let mut expected = vec![0.0_f32; query.len()];
            for query_head in 0..4 {
                let kv_head = query_head / 2;
                let query_head_values = &query[query_head * 2..query_head * 2 + 2];
                let mut scores = Vec::new();
                for position in 1..3 {
                    let key = &keys[position * 4 + kv_head * 2..position * 4 + kv_head * 2 + 2];
                    scores.push(
                        query_head_values
                            .iter()
                            .zip(key)
                            .fold(0.0_f32, |sum, (query, key)| query.mul_add(*key, sum)),
                    );
                }
                cpu::softmax_in_place(&mut scores).unwrap();
                for (score_index, probability) in scores.into_iter().enumerate() {
                    let position = score_index + 1;
                    let value = &values[position * 4 + kv_head * 2..position * 4 + kv_head * 2 + 2];
                    for (output, value) in expected[query_head * 2..query_head * 2 + 2]
                        .iter_mut()
                        .zip(value)
                    {
                        *output = value.mul_add(probability, *output);
                    }
                }
            }
            let mut actual = vec![0.0; query.len()];
            context
                .attention(&query, &keys, &values, 4, 2, 2, Some(2), &mut actual)
                .unwrap();
            assert_close(&actual, &expected, 2.0e-6);
        }

        #[test]
        fn resident_kv_cache_wraps_at_the_sliding_window() {
            let context = MetalContext::new().unwrap();
            let mut cache = context.new_kv_layer(2, 10, Some(2)).unwrap();
            assert_eq!(cache.capacity(), 2);
            assert_eq!(cache.allocated_bytes(), 32);
            context
                .append_kv(&mut cache, 0, &[1.0, 0.0], &[100.0, 100.0])
                .unwrap();
            context
                .append_kv(&mut cache, 1, &[1.0, 0.0], &[2.0, 4.0])
                .unwrap();
            context
                .append_kv(&mut cache, 2, &[1.0, 0.0], &[6.0, 8.0])
                .unwrap();
            let mut output = [0.0; 2];
            context
                .attention_cached(&[1.0, 0.0], &cache, 1, 1, 2, &mut output)
                .unwrap();
            assert_close(&output, &[4.0, 6.0], 1.0e-6);
            assert_eq!(cache.position(), 3);
            cache.truncate(2);
            assert_eq!(cache.position(), 2);
            cache.clear();
            assert_eq!(cache.position(), 0);
        }
    }
}

#[cfg(target_os = "macos")]
pub use implementation::{MetalContext, MetalKvLayer, MetalMappedBuffer, MetalOwnedBuffer};

#[cfg(not(target_os = "macos"))]
#[derive(Debug)]
pub struct MetalContext;

#[cfg(not(target_os = "macos"))]
impl MetalContext {
    pub fn new() -> Result<Self, MetalError> {
        Err(MetalError::UnsupportedPlatform)
    }
}
