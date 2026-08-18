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
        ptr::NonNull,
        slice,
    };

    use block2::RcBlock;
    use objc2::{rc::Retained, runtime::ProtocolObject};
    use objc2_foundation::NSString;
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
        MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
        MTLLibrary, MTLResourceOptions, MTLSize,
    };

    use super::MetalError;
    use crate::gguf::TensorType;

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
    const MATVEC_Q5_0: &str = "matvec_q5_0";
    const MATVEC_Q8_0: &str = "matvec_q8_0";
    const MATVEC_Q4_K: &str = "matvec_q4_k";
    const MATVEC_Q6_K: &str = "matvec_q6_k";
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
        matvec_q5_0: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        matvec_q8_0: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        matvec_q4_k: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        matvec_q6_k: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
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
            let matvec_q5_0 = pipeline(&device, &library, MATVEC_Q5_0)?;
            let matvec_q8_0 = pipeline(&device, &library, MATVEC_Q8_0)?;
            let matvec_q4_k = pipeline(&device, &library, MATVEC_Q4_K)?;
            let matvec_q6_k = pipeline(&device, &library, MATVEC_Q6_K)?;
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
                matvec_q5_0,
                matvec_q8_0,
                matvec_q4_k,
                matvec_q6_k,
            })
        }

        pub fn device_name(&self) -> String {
            self.device.name().to_string()
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
                TensorType::F32 => &self.matvec_f32,
                TensorType::F16 => &self.matvec_f16,
                TensorType::Q5_0 => &self.matvec_q5_0,
                TensorType::Q8_0 => &self.matvec_q8_0,
                TensorType::Q4K => &self.matvec_q4_k,
                TensorType::Q6K => &self.matvec_q6_k,
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
            if encoded.len() != matrix_bytes {
                return Err(MetalError::InvalidByteLength {
                    kind: kind.name(),
                    expected: matrix_bytes,
                    actual: encoded.len(),
                });
            }
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
        use objc2_metal::MTLBuffer;

        use crate::{cpu, gguf::TensorType};

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
                    let mut encoded = Vec::with_capacity(rows * 22);
                    for row in 0..rows {
                        let mut block = [0_u8; 22];
                        block[..2].copy_from_slice(&0x3400_u16.to_le_bytes());
                        for (index, value) in block[2..].iter_mut().enumerate() {
                            *value = (index * 29 + row * 47 + 3) as u8;
                        }
                        encoded.extend_from_slice(&block);
                    }
                    (32, encoded)
                }
                TensorType::Q8_0 => {
                    let mut encoded = Vec::with_capacity(rows * 34);
                    for row in 0..rows {
                        let mut block = [0_u8; 34];
                        block[..2].copy_from_slice(&0x3000_u16.to_le_bytes());
                        for (index, value) in block[2..].iter_mut().enumerate() {
                            *value = (index as i16 * 7 - 101 + row as i16 * 11) as i8 as u8;
                        }
                        encoded.extend_from_slice(&block);
                    }
                    (32, encoded)
                }
                TensorType::Q4K => {
                    let mut encoded = Vec::with_capacity(rows * 144);
                    for row in 0..rows {
                        let mut block = [0_u8; 144];
                        block[..2].copy_from_slice(&0x3000_u16.to_le_bytes());
                        block[2..4].copy_from_slice(&0x2800_u16.to_le_bytes());
                        for (index, value) in block[4..].iter_mut().enumerate() {
                            *value = (index * 19 + row * 31 + 5) as u8;
                        }
                        encoded.extend_from_slice(&block);
                    }
                    (256, encoded)
                }
                TensorType::Q6K => {
                    let mut encoded = Vec::with_capacity(rows * 210);
                    for row in 0..rows {
                        let mut block = [0_u8; 210];
                        for (index, value) in block[..192].iter_mut().enumerate() {
                            *value = (index * 23 + row * 37 + 11) as u8;
                        }
                        for (index, value) in block[192..208].iter_mut().enumerate() {
                            *value = (index as i8 * 5 - 37 + row as i8 * 3) as u8;
                        }
                        block[208..].copy_from_slice(&0x2800_u16.to_le_bytes());
                        encoded.extend_from_slice(&block);
                    }
                    (256, encoded)
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
    }
}

#[cfg(target_os = "macos")]
pub use implementation::MetalContext;

#[cfg(not(target_os = "macos"))]
#[derive(Debug)]
pub struct MetalContext;

#[cfg(not(target_os = "macos"))]
impl MetalContext {
    pub fn new() -> Result<Self, MetalError> {
        Err(MetalError::UnsupportedPlatform)
    }
}
