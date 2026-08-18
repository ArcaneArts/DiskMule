//! Architecture-neutral generation results, cancellation, profiling, and errors.

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use crate::{cpu, gemma4::Gemma4Error, gguf::GgufError};

#[cfg(target_os = "macos")]
use crate::{expert::ExpertCacheError, metal::MetalError};

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error(transparent)]
    Gguf(#[from] GgufError),

    #[error(transparent)]
    Gemma4(#[from] Gemma4Error),

    #[error(transparent)]
    Cpu(#[from] cpu::CpuError),

    #[error(transparent)]
    Glm52Cpu(#[from] crate::glm52::cpu::Glm52CpuError),

    #[error(transparent)]
    Glm52Tokenizer(#[from] crate::glm52::tokenizer::Glm52TokenizerError),

    #[error(transparent)]
    Qwen35(#[from] crate::qwen35::Qwen35Error),

    #[error(transparent)]
    Qwen35Tokenizer(#[from] crate::qwen35::tokenizer::Qwen35TokenizerError),

    #[cfg(target_os = "macos")]
    #[error(transparent)]
    Metal(#[from] MetalError),

    #[cfg(target_os = "macos")]
    #[error(transparent)]
    ExpertCache(#[from] ExpertCacheError),

    #[error("generation requires at least one prompt token")]
    EmptyPrompt,

    #[error("token ID {token} is outside the {vocabulary}-token vocabulary")]
    InvalidToken { token: u32, vocabulary: usize },

    #[error("context would grow to {requested} tokens, exceeding the configured limit {limit}")]
    ContextLimit { requested: usize, limit: usize },

    #[error("KV cache is at position {actual}; expected {expected}")]
    CachePosition { expected: usize, actual: usize },

    #[error("generation was cancelled")]
    Cancelled,

    #[error("invalid runtime configuration: {0}")]
    InvalidConfiguration(String),

    #[error("unsupported proportional-RoPE frequency-factor layout")]
    UnsupportedRopeFactors,

    #[error("tensor {tensor:?} produced a non-finite Metal result")]
    NonFiniteMetalTensor { tensor: String },

    #[error("tensor {tensor:?} received a non-finite Metal input")]
    NonFiniteMetalInput { tensor: String },

    #[error(
        "Metal {operation} produced a non-finite activation (left max {left_max}, right max {right_max})"
    )]
    NonFiniteMetalActivation {
        operation: String,
        left_max: f32,
        right_max: f32,
    },
}

impl RuntimeError {
    pub fn is_cancelled(&self) -> bool {
        matches!(self, Self::Cancelled)
            || matches!(
                self,
                Self::Glm52Cpu(crate::glm52::cpu::Glm52CpuError::Cancelled)
            )
    }
}

#[derive(Debug, Clone, Default)]
pub struct CancellationFlag(Arc<AtomicBool>);

impl CancellationFlag {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }

    pub(crate) fn as_atomic(&self) -> &AtomicBool {
        &self.0
    }

    pub(crate) fn check(&self) -> Result<(), RuntimeError> {
        if self.is_cancelled() {
            Err(RuntimeError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GenerationProfile {
    pub prompt_tokens: usize,
    pub reused_prompt_tokens: usize,
    pub generated_tokens: usize,
    pub mapped_bytes_touched: u64,
    pub resident_kv_bytes: u64,
    pub expert_cache_hits: u64,
    pub expert_cache_misses: u64,
    pub expert_cache_evictions: u64,
    pub expert_reads: u64,
    pub expert_bytes_read: u64,
    pub expert_io_wait: Duration,
    pub expert_resident_bytes: u64,
    pub time_to_first_token: Duration,
    pub prefill_time: Duration,
    pub decode_time: Duration,
    pub embedding_time: Duration,
    pub attention_time: Duration,
    pub feed_forward_time: Duration,
    pub output_time: Duration,
    pub total_time: Duration,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenerationResult {
    pub token_ids: Vec<u32>,
    pub text: String,
    pub stopped: bool,
    pub profile: GenerationProfile,
}
