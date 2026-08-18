//! Deterministic Gemma 4 CPU reference execution.

use std::{path::Path, time::Instant};

use crate::{
    cpu::{
        self, argmax, gelu_tanh, rms_norm, rope_neox_in_place, softcap_in_place, softmax_in_place,
        tensor_expert_matvec, tensor_matvec, tensor_row, tensor_vector, top_k_softmax,
    },
    gemma4::{Gemma4Config, Gemma4Tokenizer, Gemma4Weights, RUNTIME_ARRAY_KEYS},
    gguf::TensorSource,
};

pub use crate::generation::{CancellationFlag, GenerationProfile, GenerationResult, RuntimeError};

#[derive(Debug)]
pub struct Gemma4CpuModel {
    source: TensorSource,
    pub config: Gemma4Config,
    pub tokenizer: Gemma4Tokenizer,
    pub weights: Gemma4Weights,
    max_context: usize,
    full_rope_pairs: usize,
}

#[derive(Debug, Clone)]
pub struct Gemma4CpuSession {
    cache: KvCache,
    tokens: Vec<u32>,
    logits: Option<Vec<f32>>,
}

impl Gemma4CpuModel {
    pub fn open(path: impl AsRef<Path>, max_context: usize) -> Result<Self, RuntimeError> {
        let mut source = TensorSource::open_with_arrays(path, RUNTIME_ARRAY_KEYS)?;
        let config = Gemma4Config::from_gguf(&source.gguf)?;
        let weights = Gemma4Weights::from_gguf(&source.gguf, &config)?;
        let mut rope_frequencies = vec![0.0_f32; config.rope_dimensions as usize / 2];
        tensor_vector(&source, weights.rope_frequencies, &mut rope_frequencies)?;
        let full_rope_pairs = rope_frequencies
            .iter()
            .take_while(|factor| **factor == 1.0)
            .count();
        if full_rope_pairs == 0
            || rope_frequencies[full_rope_pairs..]
                .iter()
                .any(|factor| *factor != 1e30)
        {
            return Err(RuntimeError::UnsupportedRopeFactors);
        }
        let tokenizer = Gemma4Tokenizer::take_from_gguf(&mut source.gguf)?;
        let max_context = max_context.min(config.context_length as usize);
        if max_context == 0 {
            return Err(RuntimeError::ContextLimit {
                requested: 1,
                limit: 0,
            });
        }
        Ok(Self {
            source,
            config,
            tokenizer,
            weights,
            max_context,
            full_rope_pairs,
        })
    }

    pub fn source(&self) -> &TensorSource {
        &self.source
    }

    pub fn new_cache(&self) -> KvCache {
        KvCache::new(&self.config, self.max_context)
    }

    pub fn new_session(&self) -> Gemma4CpuSession {
        Gemma4CpuSession {
            cache: self.new_cache(),
            tokens: Vec::new(),
            logits: None,
        }
    }

    pub fn generate_session_with_selector<F, S>(
        &self,
        session: &mut Gemma4CpuSession,
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
        validate_generation_request(prompt_tokens, maximum_new_tokens, self.max_context)?;
        let reusable = common_prefix_length(&session.tokens, prompt_tokens);
        let stable_logits = (reusable == session.tokens.len())
            .then(|| session.logits.clone())
            .flatten();
        session.cache.truncate(reusable);
        session.tokens.truncate(reusable);
        session.logits = stable_logits.clone();

        let result = self.generate_cpu_session_inner(
            session,
            prompt_tokens,
            maximum_new_tokens,
            cancellation,
            &mut select,
            &mut on_token,
        );
        if result.is_err() {
            session.cache.truncate(reusable);
            session.tokens.truncate(reusable);
            session.logits = stable_logits;
        }
        result
    }

    fn generate_cpu_session_inner<F, S>(
        &self,
        session: &mut Gemma4CpuSession,
        prompt_tokens: &[u32],
        maximum_new_tokens: usize,
        cancellation: &CancellationFlag,
        select: &mut S,
        on_token: &mut F,
    ) -> Result<GenerationResult, RuntimeError>
    where
        F: FnMut(u32, &str),
        S: FnMut(&[f32]) -> Result<u32, RuntimeError>,
    {
        let started = Instant::now();
        let mut profile = GenerationProfile {
            prompt_tokens: prompt_tokens.len(),
            reused_prompt_tokens: session.tokens.len(),
            ..GenerationProfile::default()
        };
        let prefill_started = Instant::now();
        let mut logits = if session.tokens.len() == prompt_tokens.len() {
            session.logits.clone().ok_or_else(|| {
                RuntimeError::InvalidConfiguration(
                    "session is missing logits for its cached token prefix".to_owned(),
                )
            })?
        } else {
            let mut final_logits = None;
            for (position, token) in prompt_tokens
                .iter()
                .copied()
                .enumerate()
                .skip(session.tokens.len())
            {
                cancellation.check()?;
                let output = self.forward_token(
                    token,
                    position,
                    &mut session.cache,
                    position + 1 == prompt_tokens.len(),
                    cancellation,
                    &mut profile,
                )?;
                session.tokens.push(token);
                if let Some(output) = output {
                    session.logits = Some(output.clone());
                    final_logits = Some(output);
                } else {
                    session.logits = None;
                }
            }
            final_logits.expect("non-empty uncached prompt suffix produces logits")
        };
        profile.prefill_time = prefill_started.elapsed();

        let mut generated = Vec::with_capacity(maximum_new_tokens);
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
            if self.tokenizer.stop_token_ids.contains(&next) {
                stopped = true;
                break;
            }
            let piece = self.tokenizer.decode(&[next], true);
            on_token(next, &piece);
            generated.push(next);
            profile.generated_tokens += 1;
            if generation_index + 1 < maximum_new_tokens {
                let decode_started = Instant::now();
                logits = self
                    .forward_token(
                        next,
                        prompt_tokens.len() + generation_index,
                        &mut session.cache,
                        true,
                        cancellation,
                        &mut profile,
                    )?
                    .expect("decode requests logits");
                session.tokens.push(next);
                session.logits = Some(logits.clone());
                profile.decode_time += decode_started.elapsed();
            }
        }
        profile.total_time = started.elapsed();
        Ok(GenerationResult {
            text: self.tokenizer.decode(&generated, true),
            token_ids: generated,
            stopped,
            profile,
        })
    }

    pub fn generate_greedy<F>(
        &self,
        prompt_tokens: &[u32],
        maximum_new_tokens: usize,
        cancellation: &CancellationFlag,
        on_token: F,
    ) -> Result<GenerationResult, RuntimeError>
    where
        F: FnMut(u32, &str),
    {
        self.generate_with_selector(
            prompt_tokens,
            maximum_new_tokens,
            cancellation,
            |logits| Ok(argmax(logits)? as u32),
            on_token,
        )
    }

    pub fn generate_with_selector<F, S>(
        &self,
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
                limit: self.max_context,
            },
        )?;
        if requested > self.max_context {
            return Err(RuntimeError::ContextLimit {
                requested,
                limit: self.max_context,
            });
        }

        let started = Instant::now();
        let mut cache = self.new_cache();
        let mut profile = GenerationProfile {
            prompt_tokens: prompt_tokens.len(),
            ..GenerationProfile::default()
        };
        let prefill_started = Instant::now();
        let mut logits = self.prefill(prompt_tokens, &mut cache, cancellation, &mut profile)?;
        profile.prefill_time = prefill_started.elapsed();

        let mut generated = Vec::with_capacity(maximum_new_tokens);
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
            if self.tokenizer.stop_token_ids.contains(&next) {
                stopped = true;
                break;
            }
            let piece = self.tokenizer.decode(&[next], true);
            on_token(next, &piece);
            generated.push(next);
            profile.generated_tokens += 1;
            if generation_index + 1 < maximum_new_tokens {
                let decode_started = Instant::now();
                logits = self
                    .forward_token(
                        next,
                        prompt_tokens.len() + generation_index,
                        &mut cache,
                        true,
                        cancellation,
                        &mut profile,
                    )?
                    .expect("decode requests logits");
                profile.decode_time += decode_started.elapsed();
            }
        }
        profile.total_time = started.elapsed();
        let text = self.tokenizer.decode(&generated, true);
        Ok(GenerationResult {
            token_ids: generated,
            text,
            stopped,
            profile,
        })
    }

    /// Compute deterministic, softcapped next-token logits for one prompt.
    pub fn prompt_logits(
        &self,
        prompt_tokens: &[u32],
        cancellation: &CancellationFlag,
    ) -> Result<(Vec<f32>, GenerationProfile), RuntimeError> {
        if prompt_tokens.is_empty() {
            return Err(RuntimeError::EmptyPrompt);
        }
        if prompt_tokens.len() > self.max_context {
            return Err(RuntimeError::ContextLimit {
                requested: prompt_tokens.len(),
                limit: self.max_context,
            });
        }
        let started = Instant::now();
        let mut cache = self.new_cache();
        let mut profile = GenerationProfile {
            prompt_tokens: prompt_tokens.len(),
            ..GenerationProfile::default()
        };
        let prefill_started = Instant::now();
        let logits = self.prefill(prompt_tokens, &mut cache, cancellation, &mut profile)?;
        profile.prefill_time = prefill_started.elapsed();
        profile.total_time = started.elapsed();
        Ok((logits, profile))
    }

    fn prefill(
        &self,
        prompt_tokens: &[u32],
        cache: &mut KvCache,
        cancellation: &CancellationFlag,
        profile: &mut GenerationProfile,
    ) -> Result<Vec<f32>, RuntimeError> {
        let mut logits = None;
        for (position, token) in prompt_tokens.iter().copied().enumerate() {
            cancellation.check()?;
            logits = self.forward_token(
                token,
                position,
                cache,
                position + 1 == prompt_tokens.len(),
                cancellation,
                profile,
            )?;
        }
        Ok(logits.expect("non-empty prefill produces final logits"))
    }

    fn forward_token(
        &self,
        token: u32,
        position: usize,
        cache: &mut KvCache,
        produce_logits: bool,
        cancellation: &CancellationFlag,
        profile: &mut GenerationProfile,
    ) -> Result<Option<Vec<f32>>, RuntimeError> {
        if token as usize >= self.tokenizer.vocab_size() {
            return Err(RuntimeError::InvalidToken {
                token,
                vocabulary: self.tokenizer.vocab_size(),
            });
        }
        if position >= self.max_context {
            return Err(RuntimeError::ContextLimit {
                requested: position + 1,
                limit: self.max_context,
            });
        }
        let actual = cache.position()?;
        if actual != position {
            return Err(RuntimeError::CachePosition {
                expected: position,
                actual,
            });
        }

        let result = self.forward_token_inner(
            token,
            position,
            cache,
            produce_logits,
            cancellation,
            profile,
        );
        if result.is_err() {
            cache.truncate(position);
        }
        result
    }

    fn forward_token_inner(
        &self,
        token: u32,
        position: usize,
        cache: &mut KvCache,
        produce_logits: bool,
        cancellation: &CancellationFlag,
        profile: &mut GenerationProfile,
    ) -> Result<Option<Vec<f32>>, RuntimeError> {
        let hidden_size = self.config.hidden_size as usize;
        let embedding_started = Instant::now();
        let mut hidden = vec![0.0_f32; hidden_size];
        tensor_row(
            &self.source,
            self.weights.token_embedding,
            token as usize,
            &mut hidden,
        )?;
        profile.mapped_bytes_touched = profile
            .mapped_bytes_touched
            .saturating_add(self.tensor_row_bytes(self.weights.token_embedding));
        let embedding_scale = (hidden_size as f32).sqrt();
        for value in &mut hidden {
            *value *= embedding_scale;
        }
        profile.embedding_time += embedding_started.elapsed();

        for (layer_index, layer) in self.weights.layers.iter().enumerate() {
            cancellation.check()?;
            let attention_started = Instant::now();
            let residual = hidden;
            let attention_input = self.weighted_norm(&residual, layer.attention_norm, profile)?;
            let sliding = self.config.sliding_layers[layer_index];
            let head_dimension = if sliding {
                self.config.key_length_swa as usize
            } else {
                self.config.key_length as usize
            };
            let query_heads = self.config.attention_heads as usize;
            let kv_heads = self.config.kv_heads_by_layer[layer_index] as usize;
            let mut query = vec![0.0_f32; query_heads * head_dimension];
            let mut key = vec![0.0_f32; kv_heads * head_dimension];
            self.matvec(layer.attention_query, &attention_input, &mut query, profile)?;
            self.matvec(layer.attention_key, &attention_input, &mut key, profile)?;
            let mut value = if let Some(value_weight) = layer.attention_value {
                let mut value = vec![0.0_f32; kv_heads * head_dimension];
                self.matvec(value_weight, &attention_input, &mut value, profile)?;
                value
            } else {
                key.clone()
            };
            let query_norm = self.vector(layer.query_norm, head_dimension, profile)?;
            let key_norm = self.vector(layer.key_norm, head_dimension, profile)?;
            let query_input = query.clone();
            normalize_heads(
                &query_input,
                Some(&query_norm),
                self.config.rms_epsilon,
                head_dimension,
                &mut query,
            )?;
            let key_input = key.clone();
            normalize_heads(
                &key_input,
                Some(&key_norm),
                self.config.rms_epsilon,
                head_dimension,
                &mut key,
            )?;
            let value_input = value.clone();
            normalize_heads(
                &value_input,
                None,
                self.config.rms_epsilon,
                head_dimension,
                &mut value,
            )?;
            let rotated_pairs = if sliding {
                self.config.rope_dimensions_swa as usize / 2
            } else {
                self.full_rope_pairs
            };
            let rope_theta = if sliding {
                self.config.rope_frequency_base_swa
            } else {
                self.config.rope_frequency_base
            };
            rope_neox_in_place(
                &mut query,
                query_heads,
                head_dimension,
                rotated_pairs,
                position,
                rope_theta,
            )?;
            rope_neox_in_place(
                &mut key,
                kv_heads,
                head_dimension,
                rotated_pairs,
                position,
                rope_theta,
            )?;
            cache.layers[layer_index].append(position, &key, &value)?;
            let attention = attend(
                &query,
                &cache.layers[layer_index],
                query_heads,
                kv_heads,
                head_dimension,
                sliding.then_some(self.config.sliding_window as usize),
            )?;
            let mut projected_attention = vec![0.0_f32; hidden_size];
            self.matvec(
                layer.attention_output,
                &attention,
                &mut projected_attention,
                profile,
            )?;
            let projected_attention =
                self.weighted_norm(&projected_attention, layer.post_attention_norm, profile)?;
            let attention_output = add(&residual, &projected_attention);
            profile.attention_time += attention_started.elapsed();

            let feed_forward_started = Instant::now();
            let shared_input =
                self.weighted_norm(&attention_output, layer.feed_forward_norm, profile)?;
            let shared = self.feed_forward(
                &shared_input,
                layer.feed_forward_gate,
                layer.feed_forward_up,
                layer.feed_forward_down,
                self.config.feed_forward_length as usize,
                profile,
            )?;
            let shared = self.weighted_norm(&shared, layer.shared_post_norm, profile)?;

            let routed_input =
                self.weighted_norm(&attention_output, layer.routed_pre_norm, profile)?;
            let mut router_input = vec![0.0_f32; hidden_size];
            rms_norm(
                &attention_output,
                None,
                self.config.rms_epsilon,
                &mut router_input,
            )?;
            let router_scale = self.vector(layer.router_scale, hidden_size, profile)?;
            let inverse_hidden_scale = (hidden_size as f32).sqrt().recip();
            for (value, scale) in router_input.iter_mut().zip(router_scale) {
                *value *= scale * inverse_hidden_scale;
            }
            let mut routing_logits = vec![0.0_f32; self.config.expert_count as usize];
            self.matvec(layer.router, &router_input, &mut routing_logits, profile)?;
            let selected = top_k_softmax(&routing_logits, self.config.experts_used as usize)?;
            let expert_scale = self.vector(
                layer.expert_down_scale,
                self.config.expert_count as usize,
                profile,
            )?;
            let expert_width = self.config.expert_feed_forward_length as usize;
            let mut routed = vec![0.0_f32; hidden_size];
            for (expert, probability) in selected {
                cancellation.check()?;
                let mut gate_up = vec![0.0_f32; expert_width * 2];
                self.expert_matvec(
                    layer.expert_gate_up,
                    expert,
                    &routed_input,
                    &mut gate_up,
                    profile,
                )?;
                let (gate, up) = gate_up.split_at(expert_width);
                let activated: Vec<_> = gate
                    .iter()
                    .zip(up)
                    .map(|(gate, up)| gelu_tanh(*gate) * up)
                    .collect();
                let mut expert_output = vec![0.0_f32; hidden_size];
                self.expert_matvec(
                    layer.expert_down,
                    expert,
                    &activated,
                    &mut expert_output,
                    profile,
                )?;
                let scale = probability * expert_scale[expert];
                for (output, expert_value) in routed.iter_mut().zip(expert_output) {
                    *output = expert_value.mul_add(scale, *output);
                }
            }
            let routed = self.weighted_norm(&routed, layer.routed_post_norm, profile)?;
            let combined = add(&shared, &routed);
            let combined = self.weighted_norm(&combined, layer.post_feed_forward_norm, profile)?;
            hidden = add(&attention_output, &combined);
            let layer_scale = self.vector(layer.layer_output_scale, 1, profile)?[0];
            for value in &mut hidden {
                *value *= layer_scale;
            }
            profile.feed_forward_time += feed_forward_started.elapsed();
        }

        if !produce_logits {
            return Ok(None);
        }
        cancellation.check()?;
        let output_started = Instant::now();
        let normalized = self.weighted_norm(&hidden, self.weights.output_norm, profile)?;
        let mut logits = vec![0.0_f32; self.tokenizer.vocab_size()];
        self.matvec(self.weights.output, &normalized, &mut logits, profile)?;
        softcap_in_place(&mut logits, self.config.final_logit_softcap)?;
        profile.output_time += output_started.elapsed();
        Ok(Some(logits))
    }

    fn vector(
        &self,
        index: usize,
        length: usize,
        profile: &mut GenerationProfile,
    ) -> Result<Vec<f32>, RuntimeError> {
        let mut output = vec![0.0_f32; length];
        tensor_vector(&self.source, index, &mut output)?;
        profile.mapped_bytes_touched = profile
            .mapped_bytes_touched
            .saturating_add(self.source.gguf.tensors[index].byte_len);
        Ok(output)
    }

    fn weighted_norm(
        &self,
        input: &[f32],
        weight_index: usize,
        profile: &mut GenerationProfile,
    ) -> Result<Vec<f32>, RuntimeError> {
        let weight = self.vector(weight_index, input.len(), profile)?;
        let mut output = vec![0.0_f32; input.len()];
        rms_norm(input, Some(&weight), self.config.rms_epsilon, &mut output)?;
        Ok(output)
    }

    fn matvec(
        &self,
        index: usize,
        input: &[f32],
        output: &mut [f32],
        profile: &mut GenerationProfile,
    ) -> Result<(), RuntimeError> {
        tensor_matvec(&self.source, index, input, output)?;
        profile.mapped_bytes_touched = profile
            .mapped_bytes_touched
            .saturating_add(self.source.gguf.tensors[index].byte_len);
        Ok(())
    }

    fn expert_matvec(
        &self,
        index: usize,
        expert: usize,
        input: &[f32],
        output: &mut [f32],
        profile: &mut GenerationProfile,
    ) -> Result<(), RuntimeError> {
        tensor_expert_matvec(&self.source, index, expert, input, output)?;
        let experts = self.source.gguf.tensors[index].dimensions[2];
        profile.mapped_bytes_touched = profile
            .mapped_bytes_touched
            .saturating_add(self.source.gguf.tensors[index].byte_len / experts);
        Ok(())
    }

    fn feed_forward(
        &self,
        input: &[f32],
        gate_index: usize,
        up_index: usize,
        down_index: usize,
        width: usize,
        profile: &mut GenerationProfile,
    ) -> Result<Vec<f32>, RuntimeError> {
        let mut gate = vec![0.0_f32; width];
        let mut up = vec![0.0_f32; width];
        self.matvec(gate_index, input, &mut gate, profile)?;
        self.matvec(up_index, input, &mut up, profile)?;
        for (gate, up) in gate.iter_mut().zip(up) {
            *gate = gelu_tanh(*gate) * up;
        }
        let mut output = vec![0.0_f32; self.config.hidden_size as usize];
        self.matvec(down_index, &gate, &mut output, profile)?;
        Ok(output)
    }

    fn tensor_row_bytes(&self, index: usize) -> u64 {
        let tensor = &self.source.gguf.tensors[index];
        tensor
            .kind
            .row_byte_len(tensor.dimensions[0])
            .unwrap_or_default()
    }
}

#[derive(Debug, Clone)]
pub struct KvCache {
    pub(super) layers: Vec<LayerKv>,
    max_context: usize,
}

pub(super) fn common_prefix_length(left: &[u32], right: &[u32]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left, right)| left == right)
        .count()
}

pub(super) fn validate_generation_request(
    prompt_tokens: &[u32],
    maximum_new_tokens: usize,
    max_context: usize,
) -> Result<(), RuntimeError> {
    if prompt_tokens.is_empty() {
        return Err(RuntimeError::EmptyPrompt);
    }
    let requested =
        prompt_tokens
            .len()
            .checked_add(maximum_new_tokens)
            .ok_or(RuntimeError::ContextLimit {
                requested: usize::MAX,
                limit: max_context,
            })?;
    if requested > max_context {
        return Err(RuntimeError::ContextLimit {
            requested,
            limit: max_context,
        });
    }
    Ok(())
}

impl KvCache {
    pub(super) fn new(config: &Gemma4Config, max_context: usize) -> Self {
        let layers = config
            .sliding_layers
            .iter()
            .enumerate()
            .map(|(layer, sliding)| {
                let head_dimension = if *sliding {
                    config.key_length_swa as usize
                } else {
                    config.key_length as usize
                };
                LayerKv::new(config.kv_heads_by_layer[layer] as usize * head_dimension)
            })
            .collect();
        Self {
            layers,
            max_context,
        }
    }

    pub fn len(&self) -> usize {
        self.layers.first().map_or(0, LayerKv::len)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn max_context(&self) -> usize {
        self.max_context
    }

    pub fn clear(&mut self) {
        self.truncate(0);
    }

    pub(super) fn position(&self) -> Result<usize, RuntimeError> {
        let position = self.len();
        if self.layers.iter().all(|layer| layer.len() == position) {
            Ok(position)
        } else {
            Err(RuntimeError::CachePosition {
                expected: position,
                actual: self
                    .layers
                    .iter()
                    .map(LayerKv::len)
                    .find(|length| *length != position)
                    .unwrap_or(position),
            })
        }
    }

    pub(super) fn truncate(&mut self, length: usize) {
        for layer in &mut self.layers {
            layer.truncate(length);
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct LayerKv {
    pub(super) width: usize,
    pub(super) keys: Vec<f32>,
    pub(super) values: Vec<f32>,
}

impl LayerKv {
    fn new(width: usize) -> Self {
        Self {
            width,
            keys: Vec::new(),
            values: Vec::new(),
        }
    }

    fn len(&self) -> usize {
        self.keys.len() / self.width
    }

    pub(super) fn append(
        &mut self,
        position: usize,
        key: &[f32],
        value: &[f32],
    ) -> Result<(), RuntimeError> {
        let actual = self.len();
        if actual != position {
            return Err(RuntimeError::CachePosition {
                expected: position,
                actual,
            });
        }
        if key.len() != self.width || value.len() != self.width {
            return Err(RuntimeError::CachePosition {
                expected: self.width,
                actual: key.len().min(value.len()),
            });
        }
        self.keys.extend_from_slice(key);
        self.values.extend_from_slice(value);
        Ok(())
    }

    fn truncate(&mut self, length: usize) {
        self.keys.truncate(length.saturating_mul(self.width));
        self.values.truncate(length.saturating_mul(self.width));
    }
}

fn normalize_heads(
    input: &[f32],
    weight: Option<&[f32]>,
    epsilon: f32,
    head_dimension: usize,
    output: &mut [f32],
) -> Result<(), cpu::CpuError> {
    if head_dimension == 0
        || weight.is_some_and(|weight| weight.len() != head_dimension)
        || !input.len().is_multiple_of(head_dimension)
        || output.len() != input.len()
    {
        return Err(cpu::CpuError::InvalidVectorLength {
            operation: "per-head RMSNorm",
            argument: "output",
            expected: input.len(),
            actual: output.len(),
        });
    }
    for (input, output) in input
        .chunks_exact(head_dimension)
        .zip(output.chunks_exact_mut(head_dimension))
    {
        rms_norm(input, weight, epsilon, output)?;
    }
    Ok(())
}

fn attend(
    query: &[f32],
    cache: &LayerKv,
    query_heads: usize,
    kv_heads: usize,
    head_dimension: usize,
    window: Option<usize>,
) -> Result<Vec<f32>, cpu::CpuError> {
    let sequence_length = cache.len();
    if sequence_length == 0 {
        return Err(cpu::CpuError::EmptyInput {
            operation: "attention",
        });
    }
    let group_size = query_heads / kv_heads;
    let start = window.map_or(0, |window| sequence_length.saturating_sub(window));
    let mut output = vec![0.0_f32; query.len()];
    for query_head in 0..query_heads {
        let kv_head = query_head / group_size;
        let query_start = query_head * head_dimension;
        let query = &query[query_start..query_start + head_dimension];
        let mut scores = Vec::with_capacity(sequence_length - start);
        for position in start..sequence_length {
            let key_start = position * cache.width + kv_head * head_dimension;
            let key = &cache.keys[key_start..key_start + head_dimension];
            scores.push(
                query
                    .iter()
                    .zip(key)
                    .fold(0.0_f32, |sum, (query, key)| query.mul_add(*key, sum)),
            );
        }
        softmax_in_place(&mut scores)?;
        let output = &mut output[query_start..query_start + head_dimension];
        for (score_index, probability) in scores.into_iter().enumerate() {
            let position = start + score_index;
            let value_start = position * cache.width + kv_head * head_dimension;
            let value = &cache.values[value_start..value_start + head_dimension];
            for (output, value) in output.iter_mut().zip(value) {
                *output = value.mul_add(probability, *output);
            }
        }
    }
    Ok(output)
}

fn add(left: &[f32], right: &[f32]) -> Vec<f32> {
    left.iter()
        .zip(right)
        .map(|(left, right)| left + right)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{CancellationFlag, LayerKv, RuntimeError, attend, common_prefix_length};

    #[test]
    fn cancellation_is_shared_and_observable() {
        let flag = CancellationFlag::default();
        let clone = flag.clone();
        assert!(!clone.is_cancelled());
        flag.cancel();
        assert!(clone.is_cancelled());
        assert!(matches!(clone.check(), Err(RuntimeError::Cancelled)));
    }

    #[test]
    fn grouped_attention_uses_the_matching_kv_head() {
        let mut cache = LayerKv::new(4);
        cache
            .append(0, &[1.0, 0.0, 0.0, 1.0], &[2.0, 3.0, 5.0, 7.0])
            .unwrap();
        let query = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        let output = attend(&query, &cache, 4, 2, 2, None).unwrap();
        assert_eq!(output, [2.0, 3.0, 2.0, 3.0, 5.0, 7.0, 5.0, 7.0]);
    }

    #[test]
    fn sliding_attention_ignores_values_outside_the_window() {
        let mut cache = LayerKv::new(2);
        cache.append(0, &[1.0, 0.0], &[100.0, 100.0]).unwrap();
        cache.append(1, &[1.0, 0.0], &[2.0, 4.0]).unwrap();
        let output = attend(&[1.0, 0.0], &cache, 1, 1, 2, Some(1)).unwrap();
        assert_eq!(output, [2.0, 4.0]);
    }

    #[test]
    fn failed_append_does_not_mutate_the_cache() {
        let mut cache = LayerKv::new(2);
        assert!(matches!(
            cache.append(1, &[1.0, 0.0], &[1.0, 0.0]),
            Err(RuntimeError::CachePosition { .. })
        ));
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn session_prefix_reuse_stops_at_the_first_changed_token() {
        assert_eq!(common_prefix_length(&[1, 2, 3, 4], &[1, 2, 9, 4]), 2);
        assert_eq!(common_prefix_length(&[1, 2], &[1, 2, 3]), 2);
        assert_eq!(common_prefix_length(&[], &[1]), 0);
    }
}
