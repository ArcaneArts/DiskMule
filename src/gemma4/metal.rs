//! Gemma 4 execution backed by the validated Metal operators.

use std::{path::Path, time::Instant};

use crate::{
    cpu::{tensor_row, tensor_vector},
    gemma4::{Gemma4Config, Gemma4Tokenizer, Gemma4Weights, RUNTIME_ARRAY_KEYS},
    gguf::TensorSource,
    metal::{MetalContext, MetalMappedBuffer},
};

use super::cpu::{CancellationFlag, GenerationProfile, GenerationResult, KvCache, RuntimeError};

#[derive(Debug)]
pub struct Gemma4MetalModel {
    source: TensorSource,
    pub config: Gemma4Config,
    pub tokenizer: Gemma4Tokenizer,
    pub weights: Gemma4Weights,
    metal: MetalContext,
    mapped_weights: MetalMappedBuffer,
    max_context: usize,
    full_rope_pairs: usize,
}

impl Gemma4MetalModel {
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
        let metal = MetalContext::new()?;
        let mapped_weights = metal.map_read_only(source.shared_mapping())?;
        Ok(Self {
            source,
            config,
            tokenizer,
            weights,
            metal,
            mapped_weights,
            max_context,
            full_rope_pairs,
        })
    }

    pub fn source(&self) -> &TensorSource {
        &self.source
    }

    pub fn device_name(&self) -> String {
        self.metal.device_name()
    }

    pub fn new_cache(&self) -> KvCache {
        KvCache::new(&self.config, self.max_context)
    }

    pub fn generate_greedy<F>(
        &self,
        prompt_tokens: &[u32],
        maximum_new_tokens: usize,
        cancellation: &CancellationFlag,
        mut on_token: F,
    ) -> Result<GenerationResult, RuntimeError>
    where
        F: FnMut(u32, &str),
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
        let mut logits = self.prefill(prompt_tokens, &mut cache, cancellation, &mut profile)?;
        let mut generated = Vec::with_capacity(maximum_new_tokens);
        let mut stopped = false;
        for generation_index in 0..maximum_new_tokens {
            cancellation.check()?;
            let next = self.metal.argmax(&logits)? as u32;
            if self.tokenizer.stop_token_ids.contains(&next) {
                stopped = true;
                break;
            }
            let piece = self.tokenizer.decode(&[next], true);
            on_token(next, &piece);
            generated.push(next);
            profile.generated_tokens += 1;
            if generation_index + 1 < maximum_new_tokens {
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
        let logits = self.prefill(prompt_tokens, &mut cache, cancellation, &mut profile)?;
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
            self.normalize_heads(&query_input, Some(&query_norm), head_dimension, &mut query)?;
            let key_input = key.clone();
            self.normalize_heads(&key_input, Some(&key_norm), head_dimension, &mut key)?;
            let value_input = value.clone();
            self.normalize_heads(&value_input, None, head_dimension, &mut value)?;
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
            self.metal.rope_neox_in_place(
                &mut query,
                query_heads,
                head_dimension,
                rotated_pairs,
                position,
                rope_theta,
            )?;
            self.metal.rope_neox_in_place(
                &mut key,
                kv_heads,
                head_dimension,
                rotated_pairs,
                position,
                rope_theta,
            )?;
            cache.layers[layer_index].append(position, &key, &value)?;
            let layer_cache = &cache.layers[layer_index];
            let mut attention = vec![0.0_f32; query.len()];
            self.metal.attention(
                &query,
                &layer_cache.keys,
                &layer_cache.values,
                query_heads,
                kv_heads,
                head_dimension,
                sliding.then_some(self.config.sliding_window as usize),
                &mut attention,
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
            let attention_output = self.add(&residual, &projected_attention)?;
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
            self.metal.rms_norm(
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
            let selected = self
                .metal
                .top_k_softmax(&routing_logits, self.config.experts_used as usize)?;
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
                let mut activated = vec![0.0_f32; expert_width];
                self.metal.gelu_mul(gate, up, &mut activated)?;
                self.require_finite_activation(
                    &format!("layer {layer_index} expert {expert} GELU"),
                    gate,
                    up,
                    &activated,
                )?;
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
            let combined = self.add(&shared, &routed)?;
            let combined = self.weighted_norm(&combined, layer.post_feed_forward_norm, profile)?;
            hidden = self.add(&attention_output, &combined)?;
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
        let raw_logits = logits;
        let mut logits = vec![0.0_f32; raw_logits.len()];
        self.metal
            .softcap(&raw_logits, self.config.final_logit_softcap, &mut logits)?;
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
        self.metal
            .rms_norm(input, Some(&weight), self.config.rms_epsilon, &mut output)?;
        Ok(output)
    }

    fn normalize_heads(
        &self,
        input: &[f32],
        weight: Option<&[f32]>,
        head_dimension: usize,
        output: &mut [f32],
    ) -> Result<(), RuntimeError> {
        if head_dimension == 0
            || weight.is_some_and(|weight| weight.len() != head_dimension)
            || !input.len().is_multiple_of(head_dimension)
            || output.len() != input.len()
        {
            return Err(crate::metal::MetalError::InvalidVectorLength {
                operation: "per-head RMSNorm",
                argument: "output",
                expected: input.len(),
                actual: output.len(),
            }
            .into());
        }
        for (input, output) in input
            .chunks_exact(head_dimension)
            .zip(output.chunks_exact_mut(head_dimension))
        {
            self.metal
                .rms_norm(input, weight, self.config.rms_epsilon, output)?;
        }
        Ok(())
    }

    fn matvec(
        &self,
        index: usize,
        input: &[f32],
        output: &mut [f32],
        profile: &mut GenerationProfile,
    ) -> Result<(), RuntimeError> {
        let tensor = &self.source.gguf.tensors[index];
        if input.iter().any(|value| !value.is_finite()) {
            return Err(RuntimeError::NonFiniteMetalInput {
                tensor: format!("{} ({})", tensor.name, tensor.kind.name()),
            });
        }
        self.metal.matvec_mapped(
            tensor.kind,
            dimension(tensor.dimensions[1])?,
            dimension(tensor.dimensions[0])?,
            &self.mapped_weights,
            dimension(tensor.absolute_offset)?,
            dimension(tensor.byte_len)?,
            input,
            output,
        )?;
        if output.iter().any(|value| !value.is_finite()) {
            return Err(RuntimeError::NonFiniteMetalTensor {
                tensor: format!("{} ({})", tensor.name, tensor.kind.name()),
            });
        }
        profile.mapped_bytes_touched = profile.mapped_bytes_touched.saturating_add(tensor.byte_len);
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
        let tensor = &self.source.gguf.tensors[index];
        if input.iter().any(|value| !value.is_finite()) {
            return Err(RuntimeError::NonFiniteMetalInput {
                tensor: format!("{} expert {expert} ({})", tensor.name, tensor.kind.name()),
            });
        }
        let columns = dimension(tensor.dimensions[0])?;
        let rows = dimension(tensor.dimensions[1])?;
        let experts = dimension(tensor.dimensions[2])?;
        if expert >= experts {
            return Err(crate::cpu::CpuError::InvalidTensorExpert {
                tensor: tensor.name.clone(),
                expert,
                experts,
            }
            .into());
        }
        let row_bytes = tensor
            .kind
            .row_byte_len(tensor.dimensions[0])
            .and_then(|length| usize::try_from(length).ok())
            .ok_or(crate::metal::MetalError::Overflow)?;
        let expert_bytes = rows
            .checked_mul(row_bytes)
            .ok_or(crate::metal::MetalError::Overflow)?;
        let start = expert
            .checked_mul(expert_bytes)
            .ok_or(crate::metal::MetalError::Overflow)?;
        let absolute_offset = dimension(tensor.absolute_offset)?
            .checked_add(start)
            .ok_or(crate::metal::MetalError::Overflow)?;
        self.metal.matvec_mapped(
            tensor.kind,
            rows,
            columns,
            &self.mapped_weights,
            absolute_offset,
            expert_bytes,
            input,
            output,
        )?;
        if output.iter().any(|value| !value.is_finite()) {
            return Err(RuntimeError::NonFiniteMetalTensor {
                tensor: format!("{} expert {expert} ({})", tensor.name, tensor.kind.name()),
            });
        }
        profile.mapped_bytes_touched = profile
            .mapped_bytes_touched
            .saturating_add(tensor.byte_len / experts as u64);
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
        let mut activated = vec![0.0_f32; width];
        self.metal.gelu_mul(&gate, &up, &mut activated)?;
        self.require_finite_activation("shared GELU", &gate, &up, &activated)?;
        let mut output = vec![0.0_f32; self.config.hidden_size as usize];
        self.matvec(down_index, &activated, &mut output, profile)?;
        Ok(output)
    }

    fn add(&self, left: &[f32], right: &[f32]) -> Result<Vec<f32>, RuntimeError> {
        let mut output = vec![0.0_f32; left.len()];
        self.metal.add(left, right, &mut output)?;
        Ok(output)
    }

    fn require_finite_activation(
        &self,
        operation: &str,
        left: &[f32],
        right: &[f32],
        output: &[f32],
    ) -> Result<(), RuntimeError> {
        if output.iter().any(|value| !value.is_finite()) {
            return Err(RuntimeError::NonFiniteMetalActivation {
                operation: operation.to_owned(),
                left_max: maximum_magnitude(left),
                right_max: maximum_magnitude(right),
            });
        }
        Ok(())
    }

    fn tensor_row_bytes(&self, index: usize) -> u64 {
        let tensor = &self.source.gguf.tensors[index];
        tensor
            .kind
            .row_byte_len(tensor.dimensions[0])
            .unwrap_or_default()
    }
}

fn dimension(value: u64) -> Result<usize, RuntimeError> {
    usize::try_from(value).map_err(|_| crate::metal::MetalError::Overflow.into())
}

fn maximum_magnitude(values: &[f32]) -> f32 {
    values.iter().copied().map(f32::abs).fold(0.0, f32::max)
}
