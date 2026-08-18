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

    const SHADERS: &str = include_str!("../metal/kernels.metal");
    const VECTOR_ADD: &str = "vector_add";

    #[link(name = "CoreGraphics", kind = "framework")]
    unsafe extern "C" {}

    #[derive(Debug)]
    pub struct MetalContext {
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
        vector_add: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
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
            Ok(Self {
                device,
                queue,
                vector_add,
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
            let output_buffer = self.empty_buffer(output.len())?;
            let count_buffer = self.buffer_with_data(&[count])?;
            let command_buffer = self
                .queue
                .commandBuffer()
                .ok_or(MetalError::CommandEncoding)?;
            let encoder = command_buffer
                .computeCommandEncoder()
                .ok_or(MetalError::CommandEncoding)?;
            encoder.setComputePipelineState(&self.vector_add);
            // SAFETY: all buffers remain retained until the command completes,
            // offsets are zero, and their element layouts match the MSL kernel.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&left_buffer), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&right_buffer), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&output_buffer), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&count_buffer), 0, 3);
            }
            encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: left.len(),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: left.len().min(256),
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
            // SAFETY: the command has completed, the shared buffer contains at
            // least output.len() f32 values, and the destination is disjoint.
            unsafe {
                output.copy_from_slice(slice::from_raw_parts(
                    output_buffer.contents().as_ptr().cast::<f32>(),
                    output.len(),
                ));
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

        fn empty_buffer(
            &self,
            elements: usize,
        ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
            let bytes = elements
                .checked_mul(std::mem::size_of::<f32>())
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

        use super::MetalContext;

        #[test]
        fn creates_device_and_adds_shared_vectors() {
            let context = MetalContext::new().unwrap();
            assert!(context.device_name().contains("Apple"));
            let page_buffer = context.empty_buffer(1_024).unwrap();
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
