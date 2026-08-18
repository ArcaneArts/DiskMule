//! Payload-lazy CPU execution for GLM-5.2 safetensors weights.

use std::{
    cell::Cell,
    env,
    path::Path,
    sync::Mutex,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

use crate::{
    generation::{CancellationFlag, GenerationProfile, GenerationResult, RuntimeError},
    glm52::{
        Glm52Config, Glm52Error, Glm52FeedForwardWeights, Glm52IndexerWeights, Glm52LayerWeights,
        Glm52Weights, IndexerKind,
        expert::{Glm52ExpertCache, Glm52ExpertCacheError, Glm52ExpertCacheStats},
        quant::{GlmQuantization, GlmQuantizationError, infer_quantization, resolve_quantization},
        tokenizer::{Glm52Tokenizer, Glm52TokenizerError},
    },
    safetensors::{ReadCachePolicy, SafeDtype, SafeTensorError, SafeTensorInfo, SafeTensorSource},
};

#[cfg(target_os = "macos")]
use crate::{glm52::metal::Glm52MetalWeights, metal::MetalError};

const FP8_BLOCK: usize = 128;

#[derive(Debug, thiserror::Error)]
pub enum Glm52CpuError {
    #[error(transparent)]
    Model(#[from] Glm52Error),

    #[error(transparent)]
    Tokenizer(#[from] Glm52TokenizerError),

    #[error(transparent)]
    ExpertCache(#[from] Glm52ExpertCacheError),

    #[error(transparent)]
    Quantization(#[from] GlmQuantizationError),

    #[error(transparent)]
    SafeTensor(#[from] SafeTensorError),

    #[cfg(target_os = "macos")]
    #[error(transparent)]
    Metal(#[from] MetalError),

    #[error("safetensors tensor index {index} is out of bounds for {count} tensors")]
    TensorIndex { index: usize, count: usize },

    #[error("tensor {tensor:?} has rank {actual}; expected rank {expected}")]
    TensorRank {
        tensor: String,
        expected: usize,
        actual: usize,
    },

    #[error("tensor {tensor:?} uses non-floating CPU dtype {dtype:?}")]
    TensorDtype { tensor: String, dtype: SafeDtype },

    #[error("tensor {tensor:?} dimension {dimension} is too large for this platform")]
    Dimension { tensor: String, dimension: u64 },

    #[error("tensor {tensor:?} row {row} is out of bounds for {rows} rows")]
    RowOutOfBounds {
        tensor: String,
        row: usize,
        rows: usize,
    },

    #[error("tensor {tensor:?} input has length {actual}; expected {expected}")]
    InputLength {
        tensor: String,
        expected: usize,
        actual: usize,
    },

    #[error("tensor {tensor:?} output has length {actual}; expected {expected}")]
    OutputLength {
        tensor: String,
        expected: usize,
        actual: usize,
    },

    #[error("FP8 tensor {tensor:?} is missing its validated scale tensor {scale:?}")]
    MissingFp8Scale { tensor: String, scale: String },

    #[error("FP8 tensor {tensor:?} contains a NaN E4M3FN value at column {column}")]
    InvalidFp8Value { tensor: String, column: usize },

    #[error("integer arithmetic overflow while reading tensor {0:?}")]
    Overflow(String),

    #[error("token ID {token} is outside the GLM-5.2 vocabulary of {vocabulary} tokens")]
    TokenOutOfRange { token: u32, vocabulary: usize },

    #[error("GLM-5.2 session reached its {maximum}-token context limit")]
    ContextExceeded { maximum: usize },

    #[error("GLM-5.2 session has {actual} layer caches; expected {expected}")]
    SessionLayers { expected: usize, actual: usize },

    #[error("GLM-5.2 generation was cancelled")]
    Cancelled,

    #[error("GLM-5.2 router produced a non-finite score in layer {layer}, expert {expert}")]
    InvalidRouterScore { layer: usize, expert: usize },

    #[error("GLM-5.2 shared DSA layer {layer} has no preceding full-layer selection")]
    MissingDsaSelection { layer: usize },

    #[error("invalid GLM-5.2 expert-cache configuration: {0}")]
    InvalidExpertCache(String),

    #[error("invalid GLM-5.2 I/O configuration: {0}")]
    InvalidIoConfiguration(String),
}

/// A loaded CPU model. Dense tensors remain on disk and are read by range as needed.
pub struct Glm52CpuModel {
    pub config: Glm52Config,
    pub tokenizer: Glm52Tokenizer,
    pub weights: Glm52Weights,
    source: SafeTensorSource,
    max_context: usize,
    expert_cache: Mutex<Glm52ExpertCache>,
    #[cfg(target_os = "macos")]
    metal: Option<Box<Glm52MetalWeights>>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Glm52ForwardProfile {
    pub mapped_bytes_touched: u64,
    pub resident_kv_bytes: u64,
    pub embedding_time: Duration,
    pub attention_time: Duration,
    pub feed_forward_time: Duration,
    pub output_time: Duration,
}

#[derive(Debug, Clone)]
pub struct Glm52CpuSession {
    maximum_context: usize,
    layers: Vec<Glm52LayerCache>,
    dsa_selection: Option<Vec<usize>>,
    dsa_history: Vec<Option<Vec<usize>>>,
    pub(crate) tokens: Vec<u32>,
    pub(crate) logits: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Default)]
struct Glm52LayerCache {
    latent: Vec<Vec<f32>>,
    rope: Vec<Vec<f32>>,
    index_keys: Vec<Vec<f32>>,
}

impl Glm52CpuModel {
    pub fn from_directory(directory: impl AsRef<Path>) -> Result<Self, Glm52CpuError> {
        Self::from_directory_with_context(directory, 4_096)
    }

    pub fn from_directory_with_context(
        directory: impl AsRef<Path>,
        max_context: usize,
    ) -> Result<Self, Glm52CpuError> {
        Self::build(directory.as_ref(), max_context, false)
    }

    #[cfg(target_os = "macos")]
    pub fn from_directory_with_context_and_metal(
        directory: impl AsRef<Path>,
        max_context: usize,
    ) -> Result<Self, Glm52CpuError> {
        Self::build(directory.as_ref(), max_context, true)
    }

    fn build(
        directory: &Path,
        max_context: usize,
        #[cfg_attr(not(target_os = "macos"), allow(unused_variables))] use_metal: bool,
    ) -> Result<Self, Glm52CpuError> {
        let config = Glm52Config::from_directory(directory)?;
        if max_context == 0 {
            return Err(Glm52CpuError::ContextExceeded { maximum: 0 });
        }
        let tokenizer = Glm52Tokenizer::from_directory(directory)?;
        if tokenizer.vocabulary_size() != config.vocabulary_size {
            return Err(Glm52CpuError::Tokenizer(Glm52TokenizerError::Unsupported(
                format!(
                    "tokenizer has {} IDs but config declares {}",
                    tokenizer.vocabulary_size(),
                    config.vocabulary_size
                ),
            )));
        }
        let source = SafeTensorSource::open_with_cache_policy(directory, glm_read_cache_policy()?)?;
        let weights = Glm52Weights::from_index(&source.index, &config)?;
        let expert_cache = Glm52ExpertCache::new(source.clone(), &weights, expert_cache_slots()?)?;
        #[cfg(target_os = "macos")]
        let metal = use_metal
            .then(|| Glm52MetalWeights::new(&source).map(Box::new))
            .transpose()?;
        Ok(Self {
            config,
            tokenizer,
            weights,
            source,
            max_context,
            expert_cache: Mutex::new(expert_cache),
            #[cfg(target_os = "macos")]
            metal,
        })
    }

    #[cfg(target_os = "macos")]
    pub fn metal_device_name(&self) -> Option<String> {
        self.metal.as_deref().map(Glm52MetalWeights::device_name)
    }

    pub fn new_session(&self) -> Glm52CpuSession {
        self.start_session(self.max_context)
    }

    pub fn generate_session_with_selector<F, S>(
        &self,
        session: &mut Glm52CpuSession,
        prompt_tokens: &[u32],
        maximum_new_tokens: usize,
        cancellation: &CancellationFlag,
        mut select: S,
        mut on_token: F,
    ) -> Result<GenerationResult, RuntimeError>
    where
        F: FnMut(u32, &str),
        S: FnMut(&[f32]) -> Result<u32, RuntimeError>,
    {
        if prompt_tokens.is_empty() {
            return Err(RuntimeError::EmptyPrompt);
        }
        let requested = prompt_tokens.len().checked_add(maximum_new_tokens).ok_or(
            RuntimeError::ContextLimit {
                requested: usize::MAX,
                limit: session.maximum_context(),
            },
        )?;
        if requested > session.maximum_context() {
            return Err(RuntimeError::ContextLimit {
                requested,
                limit: session.maximum_context(),
            });
        }

        let reusable = session
            .tokens
            .iter()
            .zip(prompt_tokens)
            .take_while(|(left, right)| left == right)
            .count();
        let stable_logits = (reusable == session.tokens.len())
            .then(|| session.logits.clone())
            .flatten();
        session.truncate(reusable);
        session.logits = stable_logits.clone();
        let cache_before = self.expert_cache_stats()?;

        let result = (|| {
            let started = Instant::now();
            let mut profile = GenerationProfile {
                prompt_tokens: prompt_tokens.len(),
                reused_prompt_tokens: reusable,
                ..GenerationProfile::default()
            };
            let mut forward_profile = Glm52ForwardProfile::default();
            let prefill_started = Instant::now();
            let mut logits = if reusable == prompt_tokens.len() {
                session.logits.clone().ok_or_else(|| {
                    RuntimeError::InvalidConfiguration(
                        "GLM session is missing logits for its cached prompt".to_owned(),
                    )
                })?
            } else {
                let mut final_logits = None;
                for token in prompt_tokens.iter().copied().skip(reusable) {
                    cancellation.check()?;
                    let output = self.forward_token_profiled(
                        session,
                        token,
                        cancellation.as_atomic(),
                        &mut forward_profile,
                    )?;
                    session.tokens.push(token);
                    session.logits = Some(output.clone());
                    final_logits = Some(output);
                }
                final_logits.expect("non-empty GLM prompt suffix produces logits")
            };
            profile.prefill_time = prefill_started.elapsed();

            let mut generated = Vec::with_capacity(maximum_new_tokens);
            let mut pending_utf8 = Vec::new();
            let mut stopped = false;
            for generation_index in 0..maximum_new_tokens {
                cancellation.check()?;
                let next = select(&logits)?;
                if next as usize >= logits.len() {
                    return Err(RuntimeError::InvalidToken {
                        token: next,
                        vocabulary: logits.len(),
                    });
                }
                if generation_index == 0 {
                    profile.time_to_first_token = started.elapsed();
                }
                if self.config.stop_token_ids.contains(&next) || self.tokenizer.is_special(next) {
                    stopped = true;
                    break;
                }
                generated.push(next);
                profile.generated_tokens += 1;
                pending_utf8.extend(self.tokenizer.decode_token_bytes(next, false)?);
                let piece = drain_valid_utf8(&mut pending_utf8, false);
                if !piece.is_empty() {
                    on_token(next, &piece);
                }
                if generation_index + 1 < maximum_new_tokens {
                    let decode_started = Instant::now();
                    logits = self.forward_token_profiled(
                        session,
                        next,
                        cancellation.as_atomic(),
                        &mut forward_profile,
                    )?;
                    session.tokens.push(next);
                    session.logits = Some(logits.clone());
                    profile.decode_time += decode_started.elapsed();
                }
            }
            if !pending_utf8.is_empty() {
                let piece = drain_valid_utf8(&mut pending_utf8, true);
                if !piece.is_empty() {
                    on_token(
                        *generated.last().expect("pending bytes have a token"),
                        &piece,
                    );
                }
            }
            profile.mapped_bytes_touched = forward_profile.mapped_bytes_touched;
            profile.resident_kv_bytes = session.resident_state_bytes();
            profile.embedding_time = forward_profile.embedding_time;
            profile.attention_time = forward_profile.attention_time;
            profile.feed_forward_time = forward_profile.feed_forward_time;
            profile.output_time = forward_profile.output_time;
            let cache_after = self.expert_cache_stats()?;
            profile.expert_cache_hits = cache_after.hits.saturating_sub(cache_before.hits);
            profile.expert_cache_misses = cache_after.misses.saturating_sub(cache_before.misses);
            profile.expert_cache_evictions =
                cache_after.evictions.saturating_sub(cache_before.evictions);
            profile.expert_reads = cache_after.reads.saturating_sub(cache_before.reads);
            profile.expert_bytes_read = cache_after
                .bytes_read
                .saturating_sub(cache_before.bytes_read);
            profile.expert_io_wait = cache_after
                .io_wait
                .checked_sub(cache_before.io_wait)
                .unwrap_or_default();
            profile.expert_resident_bytes = cache_after.resident_bytes;
            profile.total_time = started.elapsed();
            Ok(GenerationResult {
                text: self.tokenizer.decode(&generated, false)?,
                token_ids: generated,
                stopped,
                profile,
            })
        })();
        if result.is_err() {
            session.truncate(reusable);
            session.logits = stable_logits;
        }
        result
    }

    pub fn expert_cache_stats(&self) -> Result<Glm52ExpertCacheStats, Glm52CpuError> {
        self.expert_cache
            .lock()
            .map(|cache| cache.stats())
            .map_err(|_| Glm52ExpertCacheError::Poisoned.into())
    }

    pub const fn read_cache_policy(&self) -> ReadCachePolicy {
        self.source.cache_policy()
    }

    pub fn start_session(&self, maximum_context: usize) -> Glm52CpuSession {
        Glm52CpuSession {
            maximum_context,
            layers: vec![Glm52LayerCache::default(); self.config.layer_count],
            dsa_selection: None,
            dsa_history: Vec::new(),
            tokens: Vec::new(),
            logits: None,
        }
    }

    /// Appends one token and returns its next-token logits.
    ///
    /// Cache changes are rolled back if an I/O, validation, or cancellation error occurs.
    pub fn forward_token(
        &self,
        session: &mut Glm52CpuSession,
        token: u32,
        cancelled: &AtomicBool,
    ) -> Result<Vec<f32>, Glm52CpuError> {
        self.forward_token_profiled(
            session,
            token,
            cancelled,
            &mut Glm52ForwardProfile::default(),
        )
    }

    pub fn forward_token_profiled(
        &self,
        session: &mut Glm52CpuSession,
        token: u32,
        cancelled: &AtomicBool,
        profile: &mut Glm52ForwardProfile,
    ) -> Result<Vec<f32>, Glm52CpuError> {
        if session.layers.len() != self.config.layer_count {
            return Err(Glm52CpuError::SessionLayers {
                expected: self.config.layer_count,
                actual: session.layers.len(),
            });
        }
        if token as usize >= self.config.vocabulary_size {
            return Err(Glm52CpuError::TokenOutOfRange {
                token,
                vocabulary: self.config.vocabulary_size,
            });
        }
        let position = session.position();
        if position >= session.maximum_context {
            return Err(Glm52CpuError::ContextExceeded {
                maximum: session.maximum_context,
            });
        }
        let previous_selection = session.dsa_selection.clone();
        let result = self.forward_token_inner(session, token, position, cancelled, profile);
        if result.is_err() {
            for layer in &mut session.layers {
                layer.latent.truncate(position);
                layer.rope.truncate(position);
                layer.index_keys.truncate(position);
            }
            session.dsa_selection = previous_selection;
            session.dsa_history.truncate(position);
        }
        if result.is_ok() {
            profile.resident_kv_bytes = session.resident_state_bytes();
        }
        result
    }

    fn forward_token_inner(
        &self,
        session: &mut Glm52CpuSession,
        token: u32,
        position: usize,
        cancelled: &AtomicBool,
        profile: &mut Glm52ForwardProfile,
    ) -> Result<Vec<f32>, Glm52CpuError> {
        check_cancelled(cancelled)?;
        let mut reader = TensorReader::new(&self.source);
        #[cfg(target_os = "macos")]
        if let Some(metal) = self.metal.as_ref() {
            reader = reader.with_metal(metal);
        }
        let embedding_started = Instant::now();
        let mut hidden = reader.matrix_row_with_columns(
            self.weights.token_embedding,
            token as usize,
            self.config.hidden_size,
        )?;
        profile.embedding_time += embedding_started.elapsed();
        for (layer_index, weights) in self.weights.layers.iter().enumerate() {
            check_cancelled(cancelled)?;
            let input_norm = reader.vector(weights.input_norm)?;
            let normalized = rms_norm(&hidden, &input_norm, self.config.rms_epsilon);
            let attention_started = Instant::now();
            let attention = self.attention(
                &reader,
                session,
                layer_index,
                weights,
                &normalized,
                position,
                cancelled,
            )?;
            add_in_place(&mut hidden, &attention);
            profile.attention_time += attention_started.elapsed();

            let feed_forward_started = Instant::now();
            let post_norm = reader.vector(weights.post_attention_norm)?;
            let normalized = rms_norm(&hidden, &post_norm, self.config.rms_epsilon);
            let feed_forward = self.feed_forward(
                &reader,
                layer_index,
                &weights.feed_forward,
                &normalized,
                cancelled,
            )?;
            add_in_place(&mut hidden, &feed_forward);
            profile.feed_forward_time += feed_forward_started.elapsed();
        }
        check_cancelled(cancelled)?;
        let output_started = Instant::now();
        let final_norm = reader.vector(self.weights.output_norm)?;
        let normalized = rms_norm(&hidden, &final_norm, self.config.rms_epsilon);
        let logits = reader.matvec(self.weights.output, &normalized)?;
        profile.output_time += output_started.elapsed();
        profile.mapped_bytes_touched = profile
            .mapped_bytes_touched
            .saturating_add(reader.bytes_touched());
        session.dsa_history.push(session.dsa_selection.clone());
        Ok(logits)
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        reader: &TensorReader<'_>,
        session: &mut Glm52CpuSession,
        layer_index: usize,
        weights: &Glm52LayerWeights,
        hidden: &[f32],
        position: usize,
        cancelled: &AtomicBool,
    ) -> Result<Vec<f32>, Glm52CpuError> {
        let config = &self.config;
        let query_a = reader.matvec(weights.query_a, hidden)?;
        let query_norm = reader.vector(weights.query_a_norm)?;
        let query_latent = rms_norm(&query_a, &query_norm, config.rms_epsilon);
        let mut query = reader.matvec(weights.query_b, &query_latent)?;
        let query_head_size = config.query_key_nope_size + config.query_key_rope_size;
        for head in 0..config.attention_heads {
            let start = head * query_head_size + config.query_key_nope_size;
            rope_interleaved_prefix(
                &mut query[start..start + config.query_key_rope_size],
                config.query_key_rope_size,
                position,
                config.rope_theta,
            );
        }

        let compressed = reader.matvec(weights.kv_a, hidden)?;
        let kv_norm = reader.vector(weights.kv_a_norm)?;
        let latent = rms_norm(
            &compressed[..config.kv_lora_rank],
            &kv_norm,
            config.rms_epsilon,
        );
        let mut rope = compressed[config.kv_lora_rank..].to_vec();
        rope_interleaved_prefix(
            &mut rope,
            config.query_key_rope_size,
            position,
            config.rope_theta,
        );

        let cache = &mut session.layers[layer_index];
        cache.latent.push(latent);
        cache.rope.push(rope);

        match config.indexer_kinds[layer_index] {
            IndexerKind::Full => {
                let indexer = weights
                    .indexer
                    .as_ref()
                    .expect("validated full DSA layer has indexer weights");
                let key = self.index_key(reader, indexer, hidden, position)?;
                cache.index_keys.push(key);
                session.dsa_selection = self.index_selection(
                    reader,
                    indexer,
                    &query_latent,
                    hidden,
                    &cache.index_keys,
                    position,
                    cancelled,
                )?;
            }
            IndexerKind::Shared if position + 1 > config.index_topk => {
                if session.dsa_selection.is_none() {
                    return Err(Glm52CpuError::MissingDsaSelection { layer: layer_index });
                }
            }
            IndexerKind::Shared => {}
        }

        let positions = session
            .dsa_selection
            .clone()
            .unwrap_or_else(|| (0..=position).collect());
        let projected = positions
            .iter()
            .map(|&cached_position| reader.matvec(weights.kv_b, &cache.latent[cached_position]))
            .collect::<Result<Vec<_>, _>>()?;
        let kv_head_size = config.query_key_nope_size + config.value_head_size;
        let scale = 1.0 / (query_head_size as f32).sqrt();
        let mut context = vec![0.0; config.attention_heads * config.value_head_size];
        for head in 0..config.attention_heads {
            check_cancelled(cancelled)?;
            let query_offset = head * query_head_size;
            let kv_offset = head * kv_head_size;
            let mut scores = Vec::with_capacity(positions.len());
            for (projected, &cached_position) in projected.iter().zip(&positions) {
                let nope = dot(
                    &query[query_offset..query_offset + config.query_key_nope_size],
                    &projected[kv_offset..kv_offset + config.query_key_nope_size],
                );
                let rope = dot(
                    &query
                        [query_offset + config.query_key_nope_size..query_offset + query_head_size],
                    &cache.rope[cached_position],
                );
                scores.push((nope + rope) * scale);
            }
            softmax_in_place(&mut scores);
            let output_offset = head * config.value_head_size;
            for (score, projected) in scores.iter().zip(&projected) {
                for value in 0..config.value_head_size {
                    context[output_offset + value] +=
                        score * projected[kv_offset + config.query_key_nope_size + value];
                }
            }
        }
        reader.matvec(weights.attention_output, &context)
    }

    fn index_key(
        &self,
        reader: &TensorReader<'_>,
        weights: &Glm52IndexerWeights,
        hidden: &[f32],
        position: usize,
    ) -> Result<Vec<f32>, Glm52CpuError> {
        let mut key = reader.matvec(weights.key, hidden)?;
        let norm_weight = reader.vector(weights.key_norm_weight)?;
        let norm_bias = reader.vector(weights.key_norm_bias)?;
        layer_norm_in_place(&mut key, &norm_weight, &norm_bias, 1e-6);
        rope_interleaved_prefix(
            &mut key,
            self.config.query_key_rope_size,
            position,
            self.config.rope_theta,
        );
        Ok(key)
    }

    #[allow(clippy::too_many_arguments)]
    fn index_selection(
        &self,
        reader: &TensorReader<'_>,
        weights: &Glm52IndexerWeights,
        query_latent: &[f32],
        hidden: &[f32],
        keys: &[Vec<f32>],
        position: usize,
        cancelled: &AtomicBool,
    ) -> Result<Option<Vec<usize>>, Glm52CpuError> {
        if position < self.config.index_topk {
            return Ok(None);
        }
        let mut query = reader.matvec(weights.query, query_latent)?;
        for head in 0..self.config.index_heads {
            let start = head * self.config.index_head_size;
            rope_interleaved_prefix(
                &mut query[start..start + self.config.index_head_size],
                self.config.query_key_rope_size,
                position,
                self.config.rope_theta,
            );
        }
        let head_weights = reader.matvec(weights.projection, hidden)?;
        let head_scale = 1.0 / (self.config.index_heads as f32).sqrt();
        let dot_scale = 1.0 / (self.config.index_head_size as f32).sqrt();
        let mut scores = Vec::with_capacity(keys.len());
        for (cached_position, key) in keys.iter().enumerate() {
            check_cancelled(cancelled)?;
            let mut score = 0.0;
            for (head, weight) in head_weights.iter().enumerate() {
                let start = head * self.config.index_head_size;
                score += weight
                    * (dot(&query[start..start + self.config.index_head_size], key) * dot_scale)
                        .max(0.0);
            }
            scores.push((cached_position, score * head_scale));
        }
        scores.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        let mut selected = scores
            .into_iter()
            .take(self.config.index_topk)
            .map(|(position, _)| position)
            .collect::<Vec<_>>();
        selected.sort_unstable();
        Ok(Some(selected))
    }

    fn feed_forward(
        &self,
        reader: &TensorReader<'_>,
        layer: usize,
        weights: &Glm52FeedForwardWeights,
        hidden: &[f32],
        cancelled: &AtomicBool,
    ) -> Result<Vec<f32>, Glm52CpuError> {
        match weights {
            Glm52FeedForwardWeights::Dense { gate, up, down } => {
                mlp(reader, *gate, *up, *down, hidden)
            }
            Glm52FeedForwardWeights::Sparse {
                router,
                correction_bias,
                shared_gate,
                shared_up,
                shared_down,
                experts,
            } => {
                let router_logits = reader.matvec(*router, hidden)?;
                let correction = reader.vector(*correction_bias)?;
                let mut choices = Vec::with_capacity(experts.len());
                for expert in 0..experts.len() {
                    let weight = sigmoid(router_logits[expert]);
                    let choice = weight + correction[expert];
                    if !weight.is_finite() || !choice.is_finite() {
                        return Err(Glm52CpuError::InvalidRouterScore { layer, expert });
                    }
                    choices.push((expert, choice, weight));
                }
                choices.sort_by(|left, right| {
                    right
                        .1
                        .total_cmp(&left.1)
                        .then_with(|| left.0.cmp(&right.0))
                });
                choices.truncate(self.config.experts_per_token);
                if self.config.normalize_topk {
                    let sum = choices.iter().map(|choice| choice.2).sum::<f32>() + 1e-20;
                    for choice in &mut choices {
                        choice.2 /= sum;
                    }
                }
                let mut output = vec![0.0; self.config.hidden_size];
                for (expert, _, weight) in choices {
                    check_cancelled(cancelled)?;
                    let expert_output = self
                        .expert_cache
                        .lock()
                        .map_err(|_| Glm52ExpertCacheError::Poisoned)?
                        .with_expert(layer, expert, cancelled, |cached| {
                            #[cfg(target_os = "macos")]
                            if let Some(metal) = reader.metal {
                                return metal.expert_mlp(cached, hidden);
                            }
                            cached.mlp(hidden)
                        })?;
                    for (output, value) in output.iter_mut().zip(expert_output) {
                        *output += value * weight * self.config.routed_scale;
                    }
                }
                let shared = mlp(reader, *shared_gate, *shared_up, *shared_down, hidden)?;
                add_in_place(&mut output, &shared);
                Ok(output)
            }
        }
    }
}

fn expert_cache_slots() -> Result<usize, Glm52CpuError> {
    let Some(raw) = env::var_os("DISKMULE_GLM_EXPERT_SLOTS") else {
        return Ok(8);
    };
    raw.to_str()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| (1..=256).contains(value))
        .ok_or_else(|| {
            Glm52CpuError::InvalidExpertCache(
                "DISKMULE_GLM_EXPERT_SLOTS must be an integer in 1..=256".to_owned(),
            )
        })
}

fn glm_read_cache_policy() -> Result<ReadCachePolicy, Glm52CpuError> {
    let Some(raw) = env::var_os("DISKMULE_GLM_DIRECT") else {
        return Ok(ReadCachePolicy::Buffered);
    };
    match raw.to_str() {
        Some("0" | "false" | "no") => Ok(ReadCachePolicy::Buffered),
        Some("1" | "true" | "yes") => Ok(ReadCachePolicy::NoCache),
        _ => Err(Glm52CpuError::InvalidIoConfiguration(
            "DISKMULE_GLM_DIRECT must be 0/1, false/true, or no/yes".to_owned(),
        )),
    }
}

fn drain_valid_utf8(bytes: &mut Vec<u8>, final_piece: bool) -> String {
    match std::str::from_utf8(bytes) {
        Ok(text) => {
            let output = text.to_owned();
            bytes.clear();
            output
        }
        Err(error) if error.valid_up_to() > 0 => {
            let valid = error.valid_up_to();
            let output = String::from_utf8(bytes[..valid].to_vec())
                .expect("UTF-8 validator identified a valid prefix");
            bytes.drain(..valid);
            output
        }
        Err(_) if final_piece => {
            let output = String::from_utf8_lossy(bytes).into_owned();
            bytes.clear();
            output
        }
        Err(_) => String::new(),
    }
}

impl Glm52CpuSession {
    pub fn position(&self) -> usize {
        self.layers.first().map_or(0, |layer| layer.latent.len())
    }

    pub const fn maximum_context(&self) -> usize {
        self.maximum_context
    }

    pub fn resident_state_bytes(&self) -> u64 {
        let mut bytes = self
            .layers
            .capacity()
            .saturating_mul(std::mem::size_of::<Glm52LayerCache>());
        for layer in &self.layers {
            bytes = bytes
                .saturating_add(nested_f32_bytes(&layer.latent, layer.latent.capacity()))
                .saturating_add(nested_f32_bytes(&layer.rope, layer.rope.capacity()))
                .saturating_add(nested_f32_bytes(
                    &layer.index_keys,
                    layer.index_keys.capacity(),
                ));
        }
        bytes = bytes
            .saturating_add(self.dsa_selection.as_ref().map_or(0, |selection| {
                selection.capacity() * std::mem::size_of::<usize>()
            }))
            .saturating_add(
                self.dsa_history
                    .iter()
                    .flatten()
                    .map(|selection| selection.capacity() * std::mem::size_of::<usize>())
                    .sum::<usize>(),
            )
            .saturating_add(self.tokens.capacity() * std::mem::size_of::<u32>())
            .saturating_add(
                self.logits
                    .as_ref()
                    .map_or(0, |logits| logits.capacity() * std::mem::size_of::<f32>()),
            );
        u64::try_from(bytes).unwrap_or(u64::MAX)
    }

    pub(crate) fn truncate(&mut self, length: usize) {
        for layer in &mut self.layers {
            layer.latent.truncate(length);
            layer.rope.truncate(length);
            layer.index_keys.truncate(length);
        }
        self.dsa_history.truncate(length);
        self.dsa_selection = self.dsa_history.last().cloned().flatten();
        self.tokens.truncate(length);
    }
}

fn nested_f32_bytes(values: &[Vec<f32>], capacity: usize) -> usize {
    capacity
        .saturating_mul(std::mem::size_of::<Vec<f32>>())
        .saturating_add(
            values
                .iter()
                .map(|value| value.capacity().saturating_mul(std::mem::size_of::<f32>()))
                .sum::<usize>(),
        )
}

fn mlp(
    reader: &TensorReader<'_>,
    gate: usize,
    up: usize,
    down: usize,
    input: &[f32],
) -> Result<Vec<f32>, Glm52CpuError> {
    let mut gate = reader.matvec(gate, input)?;
    let up = reader.matvec(up, input)?;
    for (gate, up) in gate.iter_mut().zip(up) {
        *gate = silu(*gate) * up;
    }
    reader.matvec(down, &gate)
}

fn rms_norm(input: &[f32], weight: &[f32], epsilon: f32) -> Vec<f32> {
    let mean_square = input
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        / input.len() as f64;
    let scale = 1.0 / (mean_square as f32 + epsilon).sqrt();
    input
        .iter()
        .zip(weight)
        .map(|(value, weight)| value * scale * weight)
        .collect()
}

fn layer_norm_in_place(values: &mut [f32], weight: &[f32], bias: &[f32], epsilon: f32) {
    let mean = values.iter().map(|value| f64::from(*value)).sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| {
            let difference = f64::from(*value) - mean;
            difference * difference
        })
        .sum::<f64>()
        / values.len() as f64;
    let scale = 1.0 / (variance as f32 + epsilon).sqrt();
    for ((value, weight), bias) in values.iter_mut().zip(weight).zip(bias) {
        *value = (*value - mean as f32) * scale * weight + bias;
    }
}

fn rope_interleaved_prefix(values: &mut [f32], rotary: usize, position: usize, theta: f32) {
    let input = values[..rotary].to_vec();
    let half = rotary / 2;
    for index in 0..half {
        let angle = position as f32 * theta.powf(-2.0 * index as f32 / rotary as f32);
        let (sin, cos) = angle.sin_cos();
        let first = input[index * 2];
        let second = input[index * 2 + 1];
        values[index] = first * cos - second * sin;
        values[half + index] = second * cos + first * sin;
    }
}

fn softmax_in_place(values: &mut [f32]) {
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for value in values.iter_mut() {
        *value = (*value - maximum).exp();
        sum += *value;
    }
    for value in values {
        *value /= sum;
    }
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn add_in_place(target: &mut [f32], values: &[f32]) {
    for (target, value) in target.iter_mut().zip(values) {
        *target += value;
    }
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn silu(value: f32) -> f32 {
    value * sigmoid(value)
}

fn check_cancelled(cancelled: &AtomicBool) -> Result<(), Glm52CpuError> {
    if cancelled.load(Ordering::Relaxed) {
        Err(Glm52CpuError::Cancelled)
    } else {
        Ok(())
    }
}

/// Reads and computes directly from safetensors ranges without materializing a shard.
pub struct TensorReader<'a> {
    source: &'a SafeTensorSource,
    bytes_touched: Cell<u64>,
    #[cfg(target_os = "macos")]
    metal: Option<&'a Glm52MetalWeights>,
}

impl<'a> TensorReader<'a> {
    pub const fn new(source: &'a SafeTensorSource) -> Self {
        Self {
            source,
            bytes_touched: Cell::new(0),
            #[cfg(target_os = "macos")]
            metal: None,
        }
    }

    #[cfg(target_os = "macos")]
    pub const fn with_metal(mut self, metal: &'a Glm52MetalWeights) -> Self {
        self.metal = Some(metal);
        self
    }

    pub fn bytes_touched(&self) -> u64 {
        self.bytes_touched.get()
    }

    pub fn vector(&self, tensor_index: usize) -> Result<Vec<f32>, Glm52CpuError> {
        let tensor = self.tensor(tensor_index)?;
        require_rank(tensor, 1)?;
        let length = dimension(tensor, tensor.shape[0])?;
        let mut values = vec![0.0; length];
        self.read_dense_values(tensor, 0, &mut values)?;
        self.note_bytes(tensor.byte_len);
        Ok(values)
    }

    pub fn matrix_row(&self, tensor_index: usize, row: usize) -> Result<Vec<f32>, Glm52CpuError> {
        let tensor = self.tensor(tensor_index)?;
        require_rank(tensor, 2)?;
        let columns = dimension(tensor, tensor.shape[1])?;
        self.matrix_row_with_columns(tensor_index, row, columns)
    }

    fn matrix_row_with_columns(
        &self,
        tensor_index: usize,
        row: usize,
        columns: usize,
    ) -> Result<Vec<f32>, Glm52CpuError> {
        let mut values = vec![0.0; columns];
        self.matrix_row_into(tensor_index, row, &mut values)?;
        Ok(values)
    }

    pub fn matrix_row_into(
        &self,
        tensor_index: usize,
        row: usize,
        output: &mut [f32],
    ) -> Result<(), Glm52CpuError> {
        let tensor = self.tensor(tensor_index)?;
        let (rows, columns, quantization) = if tensor.dtype == SafeDtype::U8 {
            let (rows, quantization) =
                infer_quantization(&self.source.index, tensor, output.len())?;
            (rows, output.len(), Some(quantization))
        } else {
            require_rank(tensor, 2)?;
            (
                dimension(tensor, tensor.shape[0])?,
                dimension(tensor, tensor.shape[1])?,
                None,
            )
        };
        if row >= rows {
            return Err(Glm52CpuError::RowOutOfBounds {
                tensor: tensor.name.clone(),
                row,
                rows,
            });
        }
        if output.len() != columns {
            return Err(Glm52CpuError::OutputLength {
                tensor: tensor.name.clone(),
                expected: columns,
                actual: output.len(),
            });
        }
        if let Some(quantization) = quantization {
            self.read_quantized_row(tensor, row, output, quantization)?;
            let touched = quantization
                .row_bytes(columns)
                .saturating_add(quantization.scales_per_row(columns).saturating_mul(4));
            self.note_bytes(u64::try_from(touched).unwrap_or(u64::MAX));
        } else {
            let offset = row
                .checked_mul(columns)
                .ok_or_else(|| Glm52CpuError::Overflow(tensor.name.clone()))?;
            match tensor.dtype {
                SafeDtype::F8E4M3 => self.read_fp8_row(tensor, row, output),
                _ => self.read_dense_values(tensor, offset, output),
            }?;
            let row_bytes = u64::try_from(columns)
                .ok()
                .and_then(|columns| columns.checked_mul(tensor.dtype.byte_width()))
                .unwrap_or(u64::MAX);
            self.note_bytes(row_bytes);
        }
        Ok(())
    }

    pub fn matvec(&self, tensor_index: usize, input: &[f32]) -> Result<Vec<f32>, Glm52CpuError> {
        let tensor = self.tensor(tensor_index)?;
        let rows = if tensor.dtype == SafeDtype::U8 {
            infer_quantization(&self.source.index, tensor, input.len())?.0
        } else {
            require_rank(tensor, 2)?;
            dimension(tensor, tensor.shape[0])?
        };
        let mut output = vec![0.0; rows];
        self.matvec_into(tensor_index, input, &mut output)?;
        Ok(output)
    }

    pub fn matvec_into(
        &self,
        tensor_index: usize,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), Glm52CpuError> {
        let tensor = self.tensor(tensor_index)?;
        let quantization = if tensor.dtype == SafeDtype::U8 {
            Some(resolve_quantization(
                &self.source.index,
                tensor,
                u64::try_from(output.len())
                    .map_err(|_| Glm52CpuError::Overflow(tensor.name.clone()))?,
                u64::try_from(input.len())
                    .map_err(|_| Glm52CpuError::Overflow(tensor.name.clone()))?,
            )?)
        } else {
            require_rank(tensor, 2)?;
            None
        };
        let (rows, columns) = if quantization.is_some() {
            (output.len(), input.len())
        } else {
            (
                dimension(tensor, tensor.shape[0])?,
                dimension(tensor, tensor.shape[1])?,
            )
        };
        if input.len() != columns {
            return Err(Glm52CpuError::InputLength {
                tensor: tensor.name.clone(),
                expected: columns,
                actual: input.len(),
            });
        }
        if output.len() != rows {
            return Err(Glm52CpuError::OutputLength {
                tensor: tensor.name.clone(),
                expected: rows,
                actual: output.len(),
            });
        }
        self.note_bytes(tensor.byte_len);
        let scale_name = match tensor.dtype {
            SafeDtype::F8E4M3 => Some(format!("{}_scale_inv", tensor.name)),
            SafeDtype::U8 => Some(format!("{}.qs", tensor.name)),
            _ => None,
        };
        if let Some(scale) = scale_name.and_then(|name| self.source.index.tensor(&name)) {
            self.note_bytes(scale.byte_len);
        }
        #[cfg(target_os = "macos")]
        if let Some(metal) = self.metal {
            metal.matvec(self.source, tensor_index, input, output)?;
            return Ok(());
        }
        let mut row = vec![0.0; columns];
        for (row_index, result) in output.iter_mut().enumerate() {
            self.matrix_row_into_untracked(tensor, row_index, &mut row)?;
            *result = row
                .iter()
                .zip(input)
                .map(|(weight, value)| weight * value)
                .sum();
        }
        Ok(())
    }

    fn matrix_row_into_untracked(
        &self,
        tensor: &SafeTensorInfo,
        row: usize,
        output: &mut [f32],
    ) -> Result<(), Glm52CpuError> {
        if tensor.dtype == SafeDtype::U8 {
            let (_, quantization) = infer_quantization(&self.source.index, tensor, output.len())?;
            return self.read_quantized_row(tensor, row, output, quantization);
        }
        let columns = dimension(tensor, tensor.shape[1])?;
        let offset = row
            .checked_mul(columns)
            .ok_or_else(|| Glm52CpuError::Overflow(tensor.name.clone()))?;
        match tensor.dtype {
            SafeDtype::F8E4M3 => self.read_fp8_row(tensor, row, output),
            _ => self.read_dense_values(tensor, offset, output),
        }
    }

    fn note_bytes(&self, bytes: u64) {
        self.bytes_touched
            .set(self.bytes_touched.get().saturating_add(bytes));
    }

    fn tensor(&self, index: usize) -> Result<&SafeTensorInfo, Glm52CpuError> {
        self.source
            .index
            .tensors()
            .get(index)
            .ok_or(Glm52CpuError::TensorIndex {
                index,
                count: self.source.index.tensors().len(),
            })
    }

    fn read_dense_values(
        &self,
        tensor: &SafeTensorInfo,
        element_offset: usize,
        output: &mut [f32],
    ) -> Result<(), Glm52CpuError> {
        let width = match tensor.dtype {
            SafeDtype::F32 => 4,
            SafeDtype::F16 | SafeDtype::Bf16 => 2,
            dtype => {
                return Err(Glm52CpuError::TensorDtype {
                    tensor: tensor.name.clone(),
                    dtype,
                });
            }
        };
        let byte_offset = element_offset
            .checked_mul(width)
            .ok_or_else(|| Glm52CpuError::Overflow(tensor.name.clone()))?;
        let byte_length = output
            .len()
            .checked_mul(width)
            .ok_or_else(|| Glm52CpuError::Overflow(tensor.name.clone()))?;
        let mut bytes = vec![0_u8; byte_length];
        self.source.read_tensor_at(
            &tensor.name,
            u64::try_from(byte_offset).map_err(|_| Glm52CpuError::Overflow(tensor.name.clone()))?,
            &mut bytes,
        )?;
        match tensor.dtype {
            SafeDtype::F32 => {
                for (value, bytes) in output.iter_mut().zip(bytes.chunks_exact(4)) {
                    *value = f32::from_le_bytes(bytes.try_into().expect("four-byte chunk"));
                }
            }
            SafeDtype::F16 => {
                for (value, bytes) in output.iter_mut().zip(bytes.chunks_exact(2)) {
                    *value = f16_to_f32(u16::from_le_bytes(
                        bytes.try_into().expect("two-byte chunk"),
                    ));
                }
            }
            SafeDtype::Bf16 => {
                for (value, bytes) in output.iter_mut().zip(bytes.chunks_exact(2)) {
                    let bits = u16::from_le_bytes(bytes.try_into().expect("two-byte chunk"));
                    *value = f32::from_bits(u32::from(bits) << 16);
                }
            }
            _ => unreachable!("dtype checked before tensor read"),
        }
        Ok(())
    }

    fn read_fp8_row(
        &self,
        tensor: &SafeTensorInfo,
        row: usize,
        output: &mut [f32],
    ) -> Result<(), Glm52CpuError> {
        let mut bytes = vec![0_u8; output.len()];
        let row_offset = row
            .checked_mul(output.len())
            .ok_or_else(|| Glm52CpuError::Overflow(tensor.name.clone()))?;
        self.source.read_tensor_at(
            &tensor.name,
            u64::try_from(row_offset).map_err(|_| Glm52CpuError::Overflow(tensor.name.clone()))?,
            &mut bytes,
        )?;

        let scale_name = format!("{}_scale_inv", tensor.name);
        let scale = self.source.index.tensor(&scale_name).ok_or_else(|| {
            Glm52CpuError::MissingFp8Scale {
                tensor: tensor.name.clone(),
                scale: scale_name.clone(),
            }
        })?;
        let scale_columns = output.len().div_ceil(FP8_BLOCK);
        if scale.dtype != SafeDtype::F32
            || scale.shape
                != [
                    tensor.shape[0].div_ceil(FP8_BLOCK as u64),
                    scale_columns as u64,
                ]
        {
            return Err(Glm52CpuError::MissingFp8Scale {
                tensor: tensor.name.clone(),
                scale: scale_name,
            });
        }
        let scale_offset = (row / FP8_BLOCK)
            .checked_mul(scale_columns)
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| Glm52CpuError::Overflow(scale.name.clone()))?;
        let mut scale_bytes = vec![0_u8; scale_columns * 4];
        self.source.read_tensor_at(
            &scale.name,
            u64::try_from(scale_offset).map_err(|_| Glm52CpuError::Overflow(scale.name.clone()))?,
            &mut scale_bytes,
        )?;
        let scales = scale_bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
            .collect::<Vec<_>>();
        for (column, (output, value)) in output.iter_mut().zip(bytes).enumerate() {
            let decoded =
                decode_f8_e4m3fn(value).ok_or_else(|| Glm52CpuError::InvalidFp8Value {
                    tensor: tensor.name.clone(),
                    column,
                })?;
            *output = decoded * scales[column / FP8_BLOCK];
        }
        Ok(())
    }

    fn read_quantized_row(
        &self,
        tensor: &SafeTensorInfo,
        row: usize,
        output: &mut [f32],
        quantization: GlmQuantization,
    ) -> Result<(), Glm52CpuError> {
        let row_bytes = quantization.row_bytes(output.len());
        let byte_offset = row
            .checked_mul(row_bytes)
            .ok_or_else(|| Glm52CpuError::Overflow(tensor.name.clone()))?;
        let mut encoded = vec![0_u8; row_bytes];
        self.source.read_tensor_at(
            &tensor.name,
            u64::try_from(byte_offset).map_err(|_| Glm52CpuError::Overflow(tensor.name.clone()))?,
            &mut encoded,
        )?;

        let scale_name = format!("{}.qs", tensor.name);
        let scale_count = quantization.scales_per_row(output.len());
        let scale_offset = row
            .checked_mul(scale_count)
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| Glm52CpuError::Overflow(scale_name.clone()))?;
        let mut scale_bytes = vec![0_u8; scale_count.saturating_mul(4)];
        self.source.read_tensor_at(
            &scale_name,
            u64::try_from(scale_offset).map_err(|_| Glm52CpuError::Overflow(scale_name.clone()))?,
            &mut scale_bytes,
        )?;
        let scales = scale_bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
            .collect::<Vec<_>>();

        match quantization {
            GlmQuantization::Int8Row => {
                for (value, encoded) in output.iter_mut().zip(encoded) {
                    *value = f32::from(encoded as i8) * scales[0];
                }
            }
            GlmQuantization::Int4Grouped { group_size } => {
                for (column, value) in output.iter_mut().enumerate() {
                    let byte = encoded[column / 2];
                    let nibble = if column.is_multiple_of(2) {
                        byte & 0x0f
                    } else {
                        byte >> 4
                    };
                    *value = (f32::from(nibble) - 8.0) * scales[column / group_size];
                }
            }
        }
        Ok(())
    }
}

/// Decodes the finite-only E4M3 format used by GLM-5.2 safetensors.
pub fn decode_f8_e4m3fn(value: u8) -> Option<f32> {
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

fn require_rank(tensor: &SafeTensorInfo, expected: usize) -> Result<(), Glm52CpuError> {
    if tensor.shape.len() == expected {
        Ok(())
    } else {
        Err(Glm52CpuError::TensorRank {
            tensor: tensor.name.clone(),
            expected,
            actual: tensor.shape.len(),
        })
    }
}

fn dimension(tensor: &SafeTensorInfo, value: u64) -> Result<usize, Glm52CpuError> {
    usize::try_from(value).map_err(|_| Glm52CpuError::Dimension {
        tensor: tensor.name.clone(),
        dimension: value,
    })
}

fn f16_to_f32(value: u16) -> f32 {
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

#[cfg(test)]
pub(crate) fn write_test_model(directory: &Path) {
    tests::write_tiny_model(directory);
}

#[cfg(test)]
mod tests {
    use std::{fs, io::Write, path::Path, sync::atomic::AtomicBool};

    use serde_json::{Map, json};
    use tempfile::TempDir;

    use crate::safetensors::SafeTensorSource;

    use super::{
        Glm52CpuError, Glm52CpuModel, TensorReader, decode_f8_e4m3fn, rope_interleaved_prefix,
    };

    #[test]
    fn decodes_e4m3fn_boundaries() {
        assert_eq!(decode_f8_e4m3fn(0x00), Some(0.0));
        assert_eq!(decode_f8_e4m3fn(0x01), Some(2.0_f32.powi(-9)));
        assert_eq!(decode_f8_e4m3fn(0x38), Some(1.0));
        assert_eq!(decode_f8_e4m3fn(0xb8), Some(-1.0));
        assert_eq!(decode_f8_e4m3fn(0x7e), Some(448.0));
        assert_eq!(decode_f8_e4m3fn(0x7f), None);
        assert_eq!(decode_f8_e4m3fn(0xff), None);
    }

    #[test]
    fn reads_dense_vectors_rows_and_matvecs() {
        let temp = TempDir::new().unwrap();
        let f32_values = [1.0_f32, 2.0, 3.0, 4.0, -1.0, 0.5];
        let f32_bytes = f32_values
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let f16_bytes = [0x3c00_u16, 0xc000]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();
        let bf16_bytes = [1.5_f32, -0.25]
            .into_iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect::<Vec<_>>();
        write_safetensors(
            &temp.path().join("model.safetensors"),
            &[
                ("matrix", "F32", vec![2, 3], f32_bytes),
                ("half", "F16", vec![2], f16_bytes),
                ("brain", "BF16", vec![2], bf16_bytes),
            ],
        );
        let source = SafeTensorSource::open(temp.path()).unwrap();
        let reader = TensorReader::new(&source);
        let matrix = tensor_index(&source, "matrix");
        assert_eq!(reader.matrix_row(matrix, 1).unwrap(), [4.0, -1.0, 0.5]);
        assert_eq!(
            reader.matvec(matrix, &[2.0, -1.0, 4.0]).unwrap(),
            [12.0, 11.0]
        );
        assert_eq!(
            reader.vector(tensor_index(&source, "half")).unwrap(),
            [1.0, -2.0]
        );
        assert_eq!(
            reader.vector(tensor_index(&source, "brain")).unwrap(),
            [1.5, -0.25]
        );
    }

    #[test]
    fn applies_fp8_scales_per_128_by_128_block() {
        let temp = TempDir::new().unwrap();
        let mut weights = vec![0x38_u8; 129 * 130];
        weights[128 * 130 + 129] = 0xb8;
        let scales = [2.0_f32, 3.0, 5.0, 7.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        write_safetensors(
            &temp.path().join("model.safetensors"),
            &[
                ("weight", "F8_E4M3", vec![129, 130], weights),
                ("weight_scale_inv", "F32", vec![2, 2], scales),
            ],
        );
        let source = SafeTensorSource::open(temp.path()).unwrap();
        let reader = TensorReader::new(&source);
        let index = tensor_index(&source, "weight");
        let row = reader.matrix_row(index, 128).unwrap();
        assert_eq!(row[0], 5.0);
        assert_eq!(row[127], 5.0);
        assert_eq!(row[128], 7.0);
        assert_eq!(row[129], -7.0);

        let mut short = [0.0; 2];
        assert!(matches!(
            reader.matrix_row_into(index, 0, &mut short),
            Err(Glm52CpuError::OutputLength { .. })
        ));
    }

    #[test]
    fn reads_flat_row_int8_and_grouped_int4_matrices() {
        let temp = TempDir::new().unwrap();
        let int8_scales = [0.1_f32, 0.5]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let int4_scales = [0.5_f32, 2.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        write_safetensors(
            &temp.path().join("model.safetensors"),
            &[
                (
                    "int8",
                    "U8",
                    vec![6],
                    vec![(-127_i8) as u8, 0, 127, (-2_i8) as u8, 3, 4],
                ),
                ("int8.qs", "F32", vec![2], int8_scales),
                (
                    "int4",
                    "U8",
                    vec![6],
                    vec![0x40, 0xc8, 0x0f, 0xa9, 0xcb, 0x0d],
                ),
                ("int4.qs", "F32", vec![2], int4_scales),
            ],
        );
        let source = SafeTensorSource::open(temp.path()).unwrap();
        let reader = TensorReader::new(&source);

        let int8 = tensor_index(&source, "int8");
        assert_eq!(
            reader.matvec(int8, &[1.0, 2.0, -1.0]).unwrap(),
            [-25.4, 0.0]
        );
        let mut int8_row = [0.0; 3];
        reader.matrix_row_into(int8, 1, &mut int8_row).unwrap();
        assert_eq!(int8_row, [-1.0, 1.5, 2.0]);

        let int4 = tensor_index(&source, "int4");
        assert_eq!(
            reader.matvec(int4, &[1.0, 1.0, 1.0, 1.0, 1.0]).unwrap(),
            [-0.5, 30.0]
        );
        let mut int4_row = [0.0; 5];
        reader.matrix_row_into(int4, 0, &mut int4_row).unwrap();
        assert_eq!(int4_row, [-4.0, -2.0, 0.0, 2.0, 3.5]);
    }

    #[test]
    fn rotates_adjacent_pairs_into_contiguous_halves() {
        let mut values = [1.0, 2.0, 3.0, 4.0];
        rope_interleaved_prefix(&mut values, 4, 1, 100.0);
        let (sin0, cos0) = 1.0_f32.sin_cos();
        let (sin1, cos1) = 0.1_f32.sin_cos();
        let expected = [
            cos0 - 2.0 * sin0,
            3.0 * cos1 - 4.0 * sin1,
            2.0 * cos0 + sin0,
            4.0 * cos1 + 3.0 * sin1,
        ];
        for (actual, expected) in values.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
    }

    #[test]
    fn executes_dense_sparse_mla_and_shared_dsa_layers() {
        let temp = TempDir::new().unwrap();
        write_tiny_model(temp.path());
        let model = Glm52CpuModel::from_directory(temp.path()).unwrap();
        let cancelled = AtomicBool::new(false);
        let mut session = model.start_session(8);
        let mut predictions = Vec::new();
        let mut final_logits = Vec::new();
        for token in [1, 2, 3, 4] {
            final_logits = model
                .forward_token(&mut session, token, &cancelled)
                .unwrap();
            predictions.push(argmax(&final_logits));
        }
        assert_eq!(session.position(), 4);
        assert_eq!(session.layers[0].index_keys.len(), 4);
        assert!(session.layers[1].index_keys.is_empty());
        assert_eq!(session.dsa_selection.as_deref(), Some(&[2, 3][..]));
        assert_eq!(predictions, [5, 9, 10, 5]);
        assert!(
            (final_logits[0] - (-0.129_220_81)).abs() < 1e-6,
            "{}",
            final_logits[0]
        );

        let mut replay = model.start_session(8);
        let mut replay_logits = Vec::new();
        for token in [1, 2, 3, 4] {
            replay_logits = model.forward_token(&mut replay, token, &cancelled).unwrap();
        }
        assert_eq!(final_logits, replay_logits);
    }

    #[test]
    fn grouped_int4_graph_matches_its_dequantized_cpu_reference() {
        let quantized_directory = TempDir::new().unwrap();
        let reference_directory = TempDir::new().unwrap();
        write_tiny_quantized_pair(quantized_directory.path(), reference_directory.path());
        let quantized = Glm52CpuModel::from_directory(quantized_directory.path()).unwrap();
        let reference = Glm52CpuModel::from_directory(reference_directory.path()).unwrap();
        let cancelled = AtomicBool::new(false);
        let mut quantized_session = quantized.start_session(8);
        let mut reference_session = reference.start_session(8);

        for token in [1, 2, 3, 4] {
            let actual = quantized
                .forward_token(&mut quantized_session, token, &cancelled)
                .unwrap();
            let expected = reference
                .forward_token(&mut reference_session, token, &cancelled)
                .unwrap();
            assert_eq!(argmax(&actual), argmax(&expected));
            for (actual, expected) in actual.iter().zip(expected) {
                assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
            }
        }
        let stats = quantized.expert_cache_stats().unwrap();
        assert!(stats.misses > 0);
        assert!(stats.bytes_read > 0);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_mapped_matrices_match_the_cpu_transformer_graph() {
        let temp = TempDir::new().unwrap();
        write_tiny_model(temp.path());
        let cpu = Glm52CpuModel::from_directory(temp.path()).unwrap();
        let metal = Glm52CpuModel::from_directory_with_context_and_metal(temp.path(), 8).unwrap();
        assert!(metal.metal_device_name().is_some());

        let cancelled = AtomicBool::new(false);
        let mut cpu_session = cpu.start_session(8);
        let mut metal_session = metal.start_session(8);
        for token in [1, 2, 3, 4] {
            let cpu_logits = cpu
                .forward_token(&mut cpu_session, token, &cancelled)
                .unwrap();
            let metal_logits = metal
                .forward_token(&mut metal_session, token, &cancelled)
                .unwrap();
            assert_eq!(argmax(&cpu_logits), argmax(&metal_logits));
            for (cpu, metal) in cpu_logits.iter().zip(&metal_logits) {
                assert!((cpu - metal).abs() < 1e-4, "{cpu} != {metal}");
            }
        }
        assert_eq!(cpu_session.dsa_selection, metal_session.dsa_selection);
    }

    #[test]
    fn cancellation_and_context_errors_do_not_advance_the_cache() {
        let temp = TempDir::new().unwrap();
        write_tiny_model(temp.path());
        let model = Glm52CpuModel::from_directory(temp.path()).unwrap();
        let cancelled = AtomicBool::new(true);
        let mut session = model.start_session(1);
        assert!(matches!(
            model.forward_token(&mut session, 1, &cancelled),
            Err(Glm52CpuError::Cancelled)
        ));
        assert_eq!(session.position(), 0);
        cancelled.store(false, std::sync::atomic::Ordering::Relaxed);
        model.forward_token(&mut session, 1, &cancelled).unwrap();
        assert!(matches!(
            model.forward_token(&mut session, 2, &cancelled),
            Err(Glm52CpuError::ContextExceeded { maximum: 1 })
        ));
        assert_eq!(session.position(), 1);
    }

    fn tensor_index(source: &SafeTensorSource, name: &str) -> usize {
        source
            .index
            .tensors()
            .iter()
            .position(|tensor| tensor.name == name)
            .unwrap()
    }

    fn write_safetensors(path: &Path, tensors: &[(&str, &str, Vec<u64>, Vec<u8>)]) {
        let mut header = Map::new();
        let mut payload = Vec::new();
        for (name, dtype, shape, bytes) in tensors {
            let start = payload.len();
            payload.extend(bytes);
            header.insert(
                (*name).to_owned(),
                json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [start, payload.len()]
                }),
            );
        }
        let header = serde_json::to_vec(&header).unwrap();
        let mut file = fs::File::create(path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.write_all(&payload).unwrap();
    }

    fn argmax(values: &[f32]) -> usize {
        values
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .unwrap()
            .0
    }

    pub(super) fn write_tiny_model(directory: &Path) {
        fs::write(
            directory.join("config.json"),
            serde_json::to_vec(&json!({
                "model_type": "glm_moe_dsa",
                "vocab_size": 16,
                "hidden_size": 8,
                "intermediate_size": 6,
                "moe_intermediate_size": 3,
                "num_hidden_layers": 2,
                "first_k_dense_replace": 1,
                "num_attention_heads": 2,
                "n_routed_experts": 2,
                "num_experts_per_tok": 1,
                "n_shared_experts": 1,
                "q_lora_rank": 4,
                "kv_lora_rank": 4,
                "qk_nope_head_dim": 2,
                "qk_rope_head_dim": 2,
                "v_head_dim": 2,
                "index_topk": 2,
                "index_head_dim": 2,
                "index_n_heads": 2,
                "indexer_types": ["full", "shared"],
                "n_group": 1,
                "topk_group": 1,
                "norm_topk_prob": true,
                "routed_scaling_factor": 2.5,
                "rope_parameters": {"rope_theta": 10000.0},
                "rms_norm_eps": 0.00001,
                "eos_token_id": [2,3]
            }))
            .unwrap(),
        )
        .unwrap();
        let vocabulary = (0..8)
            .map(|id| {
                (
                    char::from_u32(33 + id).unwrap().to_string(),
                    serde_json::Value::from(id),
                )
            })
            .collect::<Map<_, _>>();
        let added_tokens = [
            (8, "[gMASK]"),
            (9, "<sop>"),
            (10, "<|system|>"),
            (11, "<|user|>"),
            (12, "<|assistant|>"),
            (13, "<think>"),
            (14, "</think>"),
            (15, "<|observation|>"),
        ]
        .map(|(id, content)| json!({"id": id, "content": content, "special": true}));
        fs::write(
            directory.join("tokenizer.json"),
            serde_json::to_vec(&json!({
                "model": {
                    "type": "BPE",
                    "vocab": vocabulary,
                    "merges": [],
                    "ignore_merges": true,
                    "byte_fallback": false
                },
                "added_tokens": added_tokens
            }))
            .unwrap(),
        )
        .unwrap();
        let mut tensors = Vec::new();
        add_tensor(&mut tensors, "model.embed_tokens.weight", &[16, 8]);
        add_tensor(&mut tensors, "lm_head.weight", &[16, 8]);
        add_tensor(&mut tensors, "model.norm.weight", &[8]);
        for layer in 0..2 {
            let prefix = format!("model.layers.{layer}");
            add_tensor(
                &mut tensors,
                &format!("{prefix}.input_layernorm.weight"),
                &[8],
            );
            add_tensor(
                &mut tensors,
                &format!("{prefix}.post_attention_layernorm.weight"),
                &[8],
            );
            add_tensor(
                &mut tensors,
                &format!("{prefix}.self_attn.q_a_proj.weight"),
                &[4, 8],
            );
            add_tensor(
                &mut tensors,
                &format!("{prefix}.self_attn.q_a_layernorm.weight"),
                &[4],
            );
            add_tensor(
                &mut tensors,
                &format!("{prefix}.self_attn.q_b_proj.weight"),
                &[8, 4],
            );
            add_tensor(
                &mut tensors,
                &format!("{prefix}.self_attn.kv_a_proj_with_mqa.weight"),
                &[6, 8],
            );
            add_tensor(
                &mut tensors,
                &format!("{prefix}.self_attn.kv_a_layernorm.weight"),
                &[4],
            );
            add_tensor(
                &mut tensors,
                &format!("{prefix}.self_attn.kv_b_proj.weight"),
                &[8, 4],
            );
            add_tensor(
                &mut tensors,
                &format!("{prefix}.self_attn.o_proj.weight"),
                &[8, 4],
            );
            if layer == 0 {
                add_tensor(
                    &mut tensors,
                    &format!("{prefix}.mlp.gate_proj.weight"),
                    &[6, 8],
                );
                add_tensor(
                    &mut tensors,
                    &format!("{prefix}.mlp.up_proj.weight"),
                    &[6, 8],
                );
                add_tensor(
                    &mut tensors,
                    &format!("{prefix}.mlp.down_proj.weight"),
                    &[8, 6],
                );
                add_tensor(
                    &mut tensors,
                    &format!("{prefix}.self_attn.indexer.wq_b.weight"),
                    &[4, 4],
                );
                add_tensor(
                    &mut tensors,
                    &format!("{prefix}.self_attn.indexer.wk.weight"),
                    &[2, 8],
                );
                add_tensor(
                    &mut tensors,
                    &format!("{prefix}.self_attn.indexer.weights_proj.weight"),
                    &[2, 8],
                );
                add_tensor(
                    &mut tensors,
                    &format!("{prefix}.self_attn.indexer.k_norm.weight"),
                    &[2],
                );
                add_tensor(
                    &mut tensors,
                    &format!("{prefix}.self_attn.indexer.k_norm.bias"),
                    &[2],
                );
            } else {
                add_tensor(&mut tensors, &format!("{prefix}.mlp.gate.weight"), &[2, 8]);
                add_tensor(
                    &mut tensors,
                    &format!("{prefix}.mlp.gate.e_score_correction_bias"),
                    &[2],
                );
                add_tensor(
                    &mut tensors,
                    &format!("{prefix}.mlp.shared_experts.gate_proj.weight"),
                    &[3, 8],
                );
                add_tensor(
                    &mut tensors,
                    &format!("{prefix}.mlp.shared_experts.up_proj.weight"),
                    &[3, 8],
                );
                add_tensor(
                    &mut tensors,
                    &format!("{prefix}.mlp.shared_experts.down_proj.weight"),
                    &[8, 3],
                );
                for expert in 0..2 {
                    for projection in ["gate", "up"] {
                        add_tensor(
                            &mut tensors,
                            &format!("{prefix}.mlp.experts.{expert}.{projection}_proj.weight"),
                            &[3, 8],
                        );
                    }
                    add_tensor(
                        &mut tensors,
                        &format!("{prefix}.mlp.experts.{expert}.down_proj.weight"),
                        &[8, 3],
                    );
                }
            }
        }
        write_safetensors(
            &directory.join("model.safetensors"),
            &tensors
                .iter()
                .map(|(name, shape)| {
                    let count = shape.iter().product::<u64>() as usize;
                    let seed = name.bytes().fold(0_u32, |sum, byte| sum + u32::from(byte));
                    let values = (0..count)
                        .map(|index| {
                            if name.contains("layernorm") || name == "model.norm.weight" {
                                0.9 + ((seed as usize + index * 7) % 9) as f32 * 0.025
                            } else if name.ends_with("e_score_correction_bias") {
                                if index == 0 { 0.1 } else { -0.1 }
                            } else {
                                ((seed as usize + index * 17) % 29) as f32 * 0.0125 - 0.175
                            }
                        })
                        .flat_map(f32::to_le_bytes)
                        .collect::<Vec<_>>();
                    (name.as_str(), "F32", shape.clone(), values)
                })
                .collect::<Vec<_>>(),
        );
    }

    fn write_tiny_quantized_pair(quantized_directory: &Path, reference_directory: &Path) {
        write_tiny_model(quantized_directory);
        write_tiny_model(reference_directory);
        let source = SafeTensorSource::open(quantized_directory).unwrap();
        let mut quantized = Vec::new();
        let mut reference = Vec::new();
        for tensor in source.index.tensors() {
            let mut bytes = vec![0_u8; tensor.byte_len as usize];
            source.read_tensor_at(&tensor.name, 0, &mut bytes).unwrap();
            let is_matrix = tensor.shape.len() == 2;
            let is_router = tensor.name.ends_with(".mlp.gate.weight");
            if !is_matrix || is_router {
                quantized.push((
                    tensor.name.clone(),
                    "F32",
                    tensor.shape.clone(),
                    bytes.clone(),
                ));
                reference.push((tensor.name.clone(), "F32", tensor.shape.clone(), bytes));
                continue;
            }
            let values = bytes
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>();
            let rows = tensor.shape[0] as usize;
            let columns = tensor.shape[1] as usize;
            let int8 = matches!(
                tensor.name.as_str(),
                "model.embed_tokens.weight" | "lm_head.weight"
            );
            let (encoded, scales, dequantized) = if int8 {
                quantize_rows(&values, rows, columns, 127, None)
            } else {
                quantize_rows(&values, rows, columns, 7, Some(16))
            };
            quantized.push((
                tensor.name.clone(),
                "U8",
                vec![encoded.len() as u64],
                encoded,
            ));
            quantized.push((
                format!("{}.qs", tensor.name),
                "F32",
                vec![scales.len() as u64],
                scales
                    .iter()
                    .flat_map(|scale| scale.to_le_bytes())
                    .collect(),
            ));
            reference.push((
                tensor.name.clone(),
                "F32",
                tensor.shape.clone(),
                dequantized.into_iter().flat_map(f32::to_le_bytes).collect(),
            ));
        }
        drop(source);
        write_owned_safetensors(&quantized_directory.join("model.safetensors"), &quantized);
        write_owned_safetensors(&reference_directory.join("model.safetensors"), &reference);
    }

    fn quantize_rows(
        values: &[f32],
        rows: usize,
        columns: usize,
        maximum: i8,
        group_size: Option<usize>,
    ) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
        let group_size = group_size.unwrap_or(columns);
        let groups = columns.div_ceil(group_size);
        let mut integers = vec![0_i8; values.len()];
        let mut scales = Vec::with_capacity(rows * groups);
        let mut dequantized = vec![0.0; values.len()];
        for row in 0..rows {
            for group in 0..groups {
                let start = row * columns + group * group_size;
                let end = (start + group_size).min((row + 1) * columns);
                let scale = (values[start..end]
                    .iter()
                    .fold(0.0_f32, |maximum, value| maximum.max(value.abs()))
                    / f32::from(maximum))
                .max(1e-8);
                scales.push(scale);
                for index in start..end {
                    let integer = (values[index] / scale)
                        .round()
                        .clamp(f32::from(-maximum - 1), f32::from(maximum))
                        as i8;
                    integers[index] = integer;
                    dequantized[index] = f32::from(integer) * scale;
                }
            }
        }
        let encoded = if maximum == 127 {
            integers.into_iter().map(|value| value as u8).collect()
        } else {
            let mut encoded = vec![0_u8; rows * columns.div_ceil(2)];
            for row in 0..rows {
                for column in 0..columns {
                    let nibble = (integers[row * columns + column] + 8) as u8;
                    let target = &mut encoded[row * columns.div_ceil(2) + column / 2];
                    if column.is_multiple_of(2) {
                        *target = nibble;
                    } else {
                        *target |= nibble << 4;
                    }
                }
            }
            encoded
        };
        (encoded, scales, dequantized)
    }

    fn write_owned_safetensors(path: &Path, tensors: &[(String, &str, Vec<u64>, Vec<u8>)]) {
        let borrowed = tensors
            .iter()
            .map(|(name, dtype, shape, bytes)| {
                (name.as_str(), *dtype, shape.clone(), bytes.clone())
            })
            .collect::<Vec<_>>();
        write_safetensors(path, &borrowed);
    }

    fn add_tensor(tensors: &mut Vec<(String, Vec<u64>)>, name: &str, shape: &[u64]) {
        tensors.push((name.to_owned(), shape.to_vec()));
    }
}
