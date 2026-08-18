//! Correctness-first scalar execution for Qwen3.5/Qwen3.8 text models.

use std::{path::Path, time::Instant};

use crate::{
    cpu::{
        self, argmax, dequantize, rms_norm, softmax_in_place, tensor_matvec, tensor_row,
        tensor_vector,
    },
    gemma4::cpu::{CancellationFlag, GenerationProfile, GenerationResult, RuntimeError},
    gguf::TensorSource,
    qwen35::{
        Qwen35AttentionWeights, Qwen35Config, Qwen35LayerWeights, Qwen35Weights,
        RUNTIME_ARRAY_KEYS, tokenizer::Qwen35Tokenizer,
    },
};

#[derive(Debug)]
pub struct Qwen35CpuModel {
    source: TensorSource,
    pub config: Qwen35Config,
    pub tokenizer: Qwen35Tokenizer,
    pub weights: Qwen35Weights,
    max_context: usize,
    #[cfg(target_os = "macos")]
    metal: Option<crate::qwen35::metal::Qwen35MetalWeights>,
}

#[derive(Debug)]
pub struct Qwen35CpuSession {
    state: Qwen35State,
    tokens: Vec<u32>,
    logits: Option<Vec<f32>>,
}

#[derive(Debug)]
struct Qwen35State {
    layers: Vec<Qwen35LayerState>,
    position: usize,
}

#[derive(Debug)]
enum Qwen35LayerState {
    Recurrent {
        /// Oldest-to-newest raw QKV projections for the causal depthwise convolution.
        convolution: Vec<f32>,
        /// Key-major matrices: `[value_head][key_component][value_component]`.
        delta: Vec<f32>,
    },
    Full {
        keys: Vec<f32>,
        values: Vec<f32>,
    },
}

impl Qwen35CpuModel {
    pub fn open(path: impl AsRef<Path>, max_context: usize) -> Result<Self, RuntimeError> {
        let source = TensorSource::open_with_arrays(path, RUNTIME_ARRAY_KEYS)?;
        Self::from_source(
            source,
            max_context,
            #[cfg(target_os = "macos")]
            None,
        )
    }

    #[cfg(target_os = "macos")]
    pub fn open_metal(path: impl AsRef<Path>, max_context: usize) -> Result<Self, RuntimeError> {
        let source = TensorSource::open_with_arrays(path, RUNTIME_ARRAY_KEYS)?;
        let metal = crate::qwen35::metal::Qwen35MetalWeights::new(&source)?;
        Self::from_source(source, max_context, Some(metal))
    }

    fn from_source(
        source: TensorSource,
        max_context: usize,
        #[cfg(target_os = "macos")] metal: Option<crate::qwen35::metal::Qwen35MetalWeights>,
    ) -> Result<Self, RuntimeError> {
        let config = Qwen35Config::from_gguf(&source.gguf)?;
        let weights = Qwen35Weights::from_gguf(&source.gguf, &config)?;
        let tokenizer = Qwen35Tokenizer::from_gguf(&source.gguf)?;
        let max_context = max_context.min(config.context_length);
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
            #[cfg(target_os = "macos")]
            metal,
        })
    }

    pub fn backend_label(&self) -> String {
        #[cfg(target_os = "macos")]
        if let Some(metal) = &self.metal {
            return format!("Metal ({})", metal.device_name());
        }
        "CPU".to_owned()
    }

    pub fn source(&self) -> &TensorSource {
        &self.source
    }

    pub fn new_session(&self) -> Qwen35CpuSession {
        Qwen35CpuSession {
            state: Qwen35State::new(&self.config),
            tokens: Vec::new(),
            logits: None,
        }
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
        let mut session = self.new_session();
        self.generate_session_with_selector(
            &mut session,
            prompt_tokens,
            maximum_new_tokens,
            cancellation,
            |logits| Ok(argmax(logits)? as u32),
            on_token,
        )
    }

    pub fn generate_session_with_selector<F, S>(
        &self,
        session: &mut Qwen35CpuSession,
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
        self.validate_request(prompt_tokens, maximum_new_tokens)?;

        let common = common_prefix_length(&session.tokens, prompt_tokens);
        if common != session.tokens.len() {
            // Recurrent state cannot be truncated cheaply. A diverging prompt is
            // rebuilt from token zero; exact extensions retain all hybrid state.
            session.clear();
        }
        let result = self.generate_session_inner(
            session,
            prompt_tokens,
            maximum_new_tokens,
            cancellation,
            &mut select,
            &mut on_token,
        );
        if result.is_err() {
            // A token touches many recurrent layers. Clearing on failure keeps a
            // cancelled or failed transaction from leaking partial state.
            session.clear();
        }
        result
    }

    fn generate_session_inner<F, S>(
        &self,
        session: &mut Qwen35CpuSession,
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
        let reused = session.tokens.len();
        let mut profile = GenerationProfile {
            prompt_tokens: prompt_tokens.len(),
            reused_prompt_tokens: reused,
            ..GenerationProfile::default()
        };
        let prefill_started = Instant::now();
        let mut logits = if reused == prompt_tokens.len() {
            session.logits.clone().ok_or_else(|| {
                RuntimeError::InvalidConfiguration(
                    "Qwen session is missing logits for its cached prefix".to_owned(),
                )
            })?
        } else {
            let mut final_logits = None;
            for (position, token) in prompt_tokens.iter().copied().enumerate().skip(reused) {
                cancellation.check()?;
                let output = self.forward_token(
                    token,
                    position,
                    &mut session.state,
                    position + 1 == prompt_tokens.len(),
                    cancellation,
                    &mut profile,
                )?;
                session.tokens.push(token);
                session.logits = output.clone();
                if output.is_some() {
                    final_logits = output;
                }
            }
            final_logits.expect("a non-empty uncached Qwen prompt suffix produces logits")
        };
        profile.prefill_time = prefill_started.elapsed();

        let mut generated = Vec::with_capacity(maximum_new_tokens);
        let mut stopped = false;
        for generation_index in 0..maximum_new_tokens {
            cancellation.check()?;
            let next = select(&logits)?;
            if next as usize >= self.tokenizer.vocabulary_size() {
                return Err(RuntimeError::InvalidToken {
                    token: next,
                    vocabulary: self.tokenizer.vocabulary_size(),
                });
            }
            if generation_index == 0 {
                profile.time_to_first_token = started.elapsed();
            }
            if self.tokenizer.stop_token_ids().contains(&next) {
                stopped = true;
                break;
            }
            let piece = self.tokenizer.decode(&[next], false)?;
            on_token(next, &piece);
            generated.push(next);
            profile.generated_tokens += 1;
            if generation_index + 1 < maximum_new_tokens {
                let decode_started = Instant::now();
                logits = self
                    .forward_token(
                        next,
                        prompt_tokens.len() + generation_index,
                        &mut session.state,
                        true,
                        cancellation,
                        &mut profile,
                    )?
                    .expect("Qwen decode requests logits");
                session.tokens.push(next);
                session.logits = Some(logits.clone());
                profile.decode_time += decode_started.elapsed();
            }
        }
        profile.resident_kv_bytes = session.state.resident_bytes();
        profile.total_time = started.elapsed();
        Ok(GenerationResult {
            text: self.tokenizer.decode(&generated, false)?,
            token_ids: generated,
            stopped,
            profile,
        })
    }

    pub fn prompt_logits(
        &self,
        prompt_tokens: &[u32],
        cancellation: &CancellationFlag,
    ) -> Result<(Vec<f32>, GenerationProfile), RuntimeError> {
        self.validate_request(prompt_tokens, 0)?;
        let started = Instant::now();
        let mut state = Qwen35State::new(&self.config);
        let mut profile = GenerationProfile {
            prompt_tokens: prompt_tokens.len(),
            ..GenerationProfile::default()
        };
        let prefill_started = Instant::now();
        let mut logits = None;
        for (position, token) in prompt_tokens.iter().copied().enumerate() {
            logits = self.forward_token(
                token,
                position,
                &mut state,
                position + 1 == prompt_tokens.len(),
                cancellation,
                &mut profile,
            )?;
        }
        profile.prefill_time = prefill_started.elapsed();
        profile.resident_kv_bytes = state.resident_bytes();
        profile.total_time = started.elapsed();
        Ok((
            logits.expect("non-empty Qwen prompt produces logits"),
            profile,
        ))
    }

    fn validate_request(
        &self,
        prompt_tokens: &[u32],
        maximum_new_tokens: usize,
    ) -> Result<(), RuntimeError> {
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
        Ok(())
    }

    fn forward_token(
        &self,
        token: u32,
        position: usize,
        state: &mut Qwen35State,
        produce_logits: bool,
        cancellation: &CancellationFlag,
        profile: &mut GenerationProfile,
    ) -> Result<Option<Vec<f32>>, RuntimeError> {
        if token as usize >= self.tokenizer.vocabulary_size() {
            return Err(RuntimeError::InvalidToken {
                token,
                vocabulary: self.tokenizer.vocabulary_size(),
            });
        }
        if position >= self.max_context {
            return Err(RuntimeError::ContextLimit {
                requested: position + 1,
                limit: self.max_context,
            });
        }
        if state.position != position {
            return Err(RuntimeError::CachePosition {
                expected: position,
                actual: state.position,
            });
        }

        let embedding_started = Instant::now();
        let mut hidden = vec![0.0_f32; self.config.hidden_size];
        tensor_row(
            &self.source,
            self.weights.token_embedding,
            token as usize,
            &mut hidden,
        )?;
        self.touch_row(self.weights.token_embedding, profile);
        profile.embedding_time += embedding_started.elapsed();

        for (layer_index, layer) in self.weights.layers.iter().enumerate() {
            cancellation.check()?;
            let attention_started = Instant::now();
            let residual = hidden;
            let normalized = self.weighted_norm(&residual, layer.attention_norm, profile)?;
            let mixed = match (&layer.attention, &mut state.layers[layer_index]) {
                (Qwen35AttentionWeights::Recurrent { .. }, Qwen35LayerState::Recurrent { .. }) => {
                    self.recurrent_attention(
                        &normalized,
                        layer,
                        &mut state.layers[layer_index],
                        profile,
                    )?
                }
                (Qwen35AttentionWeights::Full { .. }, Qwen35LayerState::Full { .. }) => self
                    .full_attention(
                        &normalized,
                        layer,
                        &mut state.layers[layer_index],
                        position,
                        profile,
                    )?,
                _ => {
                    return Err(RuntimeError::InvalidConfiguration(format!(
                        "Qwen layer {layer_index} state does not match its architecture"
                    )));
                }
            };
            hidden = add(&residual, &mixed);
            profile.attention_time += attention_started.elapsed();

            let feed_forward_started = Instant::now();
            let residual = hidden;
            let normalized = self.weighted_norm(&residual, layer.post_attention_norm, profile)?;
            let mut gate = vec![0.0_f32; self.config.feed_forward_size];
            let mut up = vec![0.0_f32; self.config.feed_forward_size];
            self.matvec(layer.feed_forward_gate, &normalized, &mut gate, profile)?;
            self.matvec(layer.feed_forward_up, &normalized, &mut up, profile)?;
            for (gate, up) in gate.iter_mut().zip(up) {
                *gate = silu(*gate) * up;
            }
            let mut projected = vec![0.0_f32; self.config.hidden_size];
            self.matvec(layer.feed_forward_down, &gate, &mut projected, profile)?;
            hidden = add(&residual, &projected);
            profile.feed_forward_time += feed_forward_started.elapsed();
        }
        state.position += 1;
        profile.resident_kv_bytes = state.resident_bytes();

        if !produce_logits {
            return Ok(None);
        }
        cancellation.check()?;
        let output_started = Instant::now();
        let normalized = self.weighted_norm(&hidden, self.weights.output_norm, profile)?;
        let mut logits = vec![0.0_f32; self.tokenizer.vocabulary_size()];
        self.matvec(self.weights.output, &normalized, &mut logits, profile)?;
        profile.output_time += output_started.elapsed();
        Ok(Some(logits))
    }

    fn full_attention(
        &self,
        input: &[f32],
        layer: &Qwen35LayerWeights,
        state: &mut Qwen35LayerState,
        position: usize,
        profile: &mut GenerationProfile,
    ) -> Result<Vec<f32>, RuntimeError> {
        let Qwen35AttentionWeights::Full {
            query_gate,
            key,
            value,
            query_norm,
            key_norm,
            output,
        } = layer.attention
        else {
            unreachable!("full attention called with a recurrent layer")
        };
        let Qwen35LayerState::Full { keys, values } = state else {
            unreachable!("full attention called with recurrent state")
        };
        let head_size = self.config.attention_head_size;
        let query_width = self.config.attention_heads * head_size;
        let kv_width = self.config.kv_heads * head_size;
        if keys.len() != position * kv_width || values.len() != position * kv_width {
            return Err(RuntimeError::CachePosition {
                expected: position,
                actual: keys.len() / kv_width,
            });
        }

        let mut query_and_gate = vec![0.0_f32; query_width * 2];
        let mut current_key = vec![0.0_f32; kv_width];
        let mut current_value = vec![0.0_f32; kv_width];
        self.matvec(query_gate, input, &mut query_and_gate, profile)?;
        self.matvec(key, input, &mut current_key, profile)?;
        self.matvec(value, input, &mut current_value, profile)?;

        let query_weight = self.vector(query_norm, head_size, profile)?;
        let key_weight = self.vector(key_norm, head_size, profile)?;
        let mut query = vec![0.0_f32; query_width];
        let mut gate = vec![0.0_f32; query_width];
        for head in 0..self.config.attention_heads {
            let source = &query_and_gate[head * head_size * 2..(head + 1) * head_size * 2];
            let target = &mut query[head * head_size..(head + 1) * head_size];
            rms_norm(
                &source[..head_size],
                Some(&query_weight),
                self.config.rms_epsilon,
                target,
            )?;
            gate[head * head_size..(head + 1) * head_size].copy_from_slice(&source[head_size..]);
        }
        let unnormalized_key = current_key;
        let mut current_key = vec![0.0_f32; kv_width];
        for head in 0..self.config.kv_heads {
            let range = head * head_size..(head + 1) * head_size;
            rms_norm(
                &unnormalized_key[range.clone()],
                Some(&key_weight),
                self.config.rms_epsilon,
                &mut current_key[range],
            )?;
        }
        partial_rope_neox_in_place(
            &mut query,
            self.config.attention_heads,
            head_size,
            self.config.rope_dimensions,
            position,
            self.config.rope_frequency_base,
        )?;
        partial_rope_neox_in_place(
            &mut current_key,
            self.config.kv_heads,
            head_size,
            self.config.rope_dimensions,
            position,
            self.config.rope_frequency_base,
        )?;
        keys.extend_from_slice(&current_key);
        values.extend_from_slice(&current_value);

        let tokens = position + 1;
        let query_groups = self.config.attention_heads / self.config.kv_heads;
        let scale = (head_size as f32).sqrt().recip();
        let mut attention = vec![0.0_f32; query_width];
        for query_head in 0..self.config.attention_heads {
            let kv_head = query_head / query_groups;
            let query_head_values = &query[query_head * head_size..(query_head + 1) * head_size];
            let mut scores = vec![0.0_f32; tokens];
            for (token_index, score) in scores.iter_mut().enumerate() {
                let start = token_index * kv_width + kv_head * head_size;
                *score = dot(query_head_values, &keys[start..start + head_size]) * scale;
            }
            softmax_in_place(&mut scores)?;
            let target = &mut attention[query_head * head_size..(query_head + 1) * head_size];
            for (token_index, probability) in scores.into_iter().enumerate() {
                let start = token_index * kv_width + kv_head * head_size;
                for (target, value) in target.iter_mut().zip(&values[start..start + head_size]) {
                    *target = probability.mul_add(*value, *target);
                }
            }
            for (value, gate) in target
                .iter_mut()
                .zip(&gate[query_head * head_size..(query_head + 1) * head_size])
            {
                *value *= sigmoid(*gate);
            }
        }
        let mut projected = vec![0.0_f32; self.config.hidden_size];
        self.matvec(output, &attention, &mut projected, profile)?;
        Ok(projected)
    }

    fn recurrent_attention(
        &self,
        input: &[f32],
        layer: &Qwen35LayerWeights,
        state: &mut Qwen35LayerState,
        profile: &mut GenerationProfile,
    ) -> Result<Vec<f32>, RuntimeError> {
        let Qwen35AttentionWeights::Recurrent {
            qkv,
            gate,
            convolution,
            alpha,
            beta,
            time_bias,
            decay,
            norm,
            output,
        } = layer.attention
        else {
            unreachable!("recurrent attention called with a full layer")
        };
        let Qwen35LayerState::Recurrent {
            convolution: convolution_state,
            delta,
        } = state
        else {
            unreachable!("recurrent attention called with full state")
        };
        let projection_size = self.config.recurrent_projection_size();
        let inner = self.config.recurrent_inner_size;
        let value_heads = self.config.recurrent_value_heads;
        let key_heads = self.config.recurrent_key_heads;
        let head_size = self.config.recurrent_state_size;
        let kernel = self.config.convolution_kernel;

        let mut projected_qkv = vec![0.0_f32; projection_size];
        let mut z = vec![0.0_f32; inner];
        let mut alpha_values = vec![0.0_f32; value_heads];
        let mut beta_values = vec![0.0_f32; value_heads];
        self.matvec(qkv, input, &mut projected_qkv, profile)?;
        self.matvec(gate, input, &mut z, profile)?;
        self.matvec(alpha, input, &mut alpha_values, profile)?;
        self.matvec(beta, input, &mut beta_values, profile)?;
        let time_bias = self.vector(time_bias, value_heads, profile)?;
        let decay = self.vector(decay, value_heads, profile)?;
        let convolution_weights =
            self.matrix_values(convolution, kernel * projection_size, profile)?;

        let mut convolved = vec![0.0_f32; projection_size];
        for channel in 0..projection_size {
            let mut sum = 0.0_f32;
            for tap in 0..kernel.saturating_sub(1) {
                sum = convolution_state[tap * projection_size + channel]
                    .mul_add(convolution_weights[channel * kernel + tap], sum);
            }
            convolved[channel] = projected_qkv[channel]
                .mul_add(convolution_weights[channel * kernel + kernel - 1], sum);
        }
        for value in &mut convolved {
            *value = silu(*value);
        }
        if kernel > 1 {
            if kernel > 2 {
                convolution_state.copy_within(projection_size.., 0);
            }
            let start = (kernel - 2) * projection_size;
            convolution_state[start..start + projection_size].copy_from_slice(&projected_qkv);
        }

        let key_width = head_size * key_heads;
        let (queries, rest) = convolved.split_at_mut(key_width);
        let (keys, values) = rest.split_at_mut(key_width);
        for head in 0..key_heads {
            l2_normalize_in_place(
                &mut queries[head * head_size..(head + 1) * head_size],
                self.config.rms_epsilon,
            );
            l2_normalize_in_place(
                &mut keys[head * head_size..(head + 1) * head_size],
                self.config.rms_epsilon,
            );
        }
        for head in 0..value_heads {
            alpha_values[head] = decay[head] * softplus(alpha_values[head] + time_bias[head]);
            beta_values[head] = sigmoid(beta_values[head]);
        }

        let mut recurrent_output = vec![0.0_f32; inner];
        gated_delta_update(
            GatedDeltaInput {
                queries,
                keys,
                values,
                gates: &alpha_values,
                betas: &beta_values,
                key_heads,
                value_heads,
                head_size,
            },
            delta,
            &mut recurrent_output,
        );

        let norm_weight = self.vector(norm, head_size, profile)?;
        let raw_output = recurrent_output;
        let mut recurrent_output = vec![0.0_f32; inner];
        for head in 0..value_heads {
            let range = head * head_size..(head + 1) * head_size;
            rms_norm(
                &raw_output[range.clone()],
                Some(&norm_weight),
                self.config.rms_epsilon,
                &mut recurrent_output[range.clone()],
            )?;
            for (value, gate) in recurrent_output[range.clone()].iter_mut().zip(&z[range]) {
                *value *= silu(*gate);
            }
        }
        let mut projected = vec![0.0_f32; self.config.hidden_size];
        self.matvec(output, &recurrent_output, &mut projected, profile)?;
        Ok(projected)
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

    fn vector(
        &self,
        index: usize,
        length: usize,
        profile: &mut GenerationProfile,
    ) -> Result<Vec<f32>, RuntimeError> {
        let mut output = vec![0.0_f32; length];
        tensor_vector(&self.source, index, &mut output)?;
        self.touch(index, profile);
        Ok(output)
    }

    fn matrix_values(
        &self,
        index: usize,
        length: usize,
        profile: &mut GenerationProfile,
    ) -> Result<Vec<f32>, RuntimeError> {
        let tensor = &self.source.gguf.tensors[index];
        let bytes =
            self.source
                .tensor_bytes_by_index(index)
                .ok_or(cpu::CpuError::InvalidTensorIndex {
                    index,
                    count: self.source.gguf.tensors.len(),
                })?;
        let mut output = vec![0.0_f32; length];
        dequantize(tensor.kind, bytes, &mut output)?;
        self.touch(index, profile);
        Ok(output)
    }

    fn matvec(
        &self,
        index: usize,
        input: &[f32],
        output: &mut [f32],
        profile: &mut GenerationProfile,
    ) -> Result<(), RuntimeError> {
        #[cfg(target_os = "macos")]
        if let Some(metal) = &self.metal {
            let tensor = &self.source.gguf.tensors[index];
            if input.iter().any(|value| !value.is_finite()) {
                return Err(RuntimeError::NonFiniteMetalInput {
                    tensor: format!("{} ({})", tensor.name, tensor.kind.name()),
                });
            }
            metal.matvec(&self.source, index, input, output)?;
            if output.iter().any(|value| !value.is_finite()) {
                return Err(RuntimeError::NonFiniteMetalTensor {
                    tensor: format!("{} ({})", tensor.name, tensor.kind.name()),
                });
            }
            self.touch(index, profile);
            return Ok(());
        }
        tensor_matvec(&self.source, index, input, output)?;
        self.touch(index, profile);
        Ok(())
    }

    fn touch(&self, index: usize, profile: &mut GenerationProfile) {
        profile.mapped_bytes_touched = profile
            .mapped_bytes_touched
            .saturating_add(self.source.gguf.tensors[index].byte_len);
    }

    fn touch_row(&self, index: usize, profile: &mut GenerationProfile) {
        let tensor = &self.source.gguf.tensors[index];
        profile.mapped_bytes_touched = profile.mapped_bytes_touched.saturating_add(
            tensor
                .kind
                .row_byte_len(tensor.dimensions[0])
                .unwrap_or_default(),
        );
    }
}

impl Qwen35CpuSession {
    pub fn clear(&mut self) {
        self.state.clear();
        self.tokens.clear();
        self.logits = None;
    }

    pub fn cached_tokens(&self) -> &[u32] {
        &self.tokens
    }
}

impl Qwen35State {
    fn new(config: &Qwen35Config) -> Self {
        let layers = (0..config.layer_count)
            .map(|layer| match config.layer_kind(layer) {
                crate::qwen35::Qwen35LayerKind::Recurrent => Qwen35LayerState::Recurrent {
                    convolution: vec![
                        0.0;
                        (config.convolution_kernel - 1)
                            * config.recurrent_projection_size()
                    ],
                    delta: vec![
                        0.0;
                        config.recurrent_value_heads
                            * config.recurrent_state_size
                            * config.recurrent_value_head_size()
                    ],
                },
                crate::qwen35::Qwen35LayerKind::FullAttention => Qwen35LayerState::Full {
                    keys: Vec::new(),
                    values: Vec::new(),
                },
            })
            .collect();
        Self {
            layers,
            position: 0,
        }
    }

    fn clear(&mut self) {
        for layer in &mut self.layers {
            match layer {
                Qwen35LayerState::Recurrent { convolution, delta } => {
                    convolution.fill(0.0);
                    delta.fill(0.0);
                }
                Qwen35LayerState::Full { keys, values } => {
                    keys.clear();
                    values.clear();
                }
            }
        }
        self.position = 0;
    }

    fn resident_bytes(&self) -> u64 {
        self.layers
            .iter()
            .map(|layer| match layer {
                Qwen35LayerState::Recurrent { convolution, delta } => {
                    convolution.len().saturating_add(delta.len())
                }
                Qwen35LayerState::Full { keys, values } => keys.len().saturating_add(values.len()),
            })
            .fold(0_u64, |total, elements| {
                total.saturating_add((elements as u64).saturating_mul(4))
            })
    }
}

struct GatedDeltaInput<'a> {
    queries: &'a [f32],
    keys: &'a [f32],
    values: &'a [f32],
    gates: &'a [f32],
    betas: &'a [f32],
    key_heads: usize,
    value_heads: usize,
    head_size: usize,
}

fn gated_delta_update(input: GatedDeltaInput<'_>, state: &mut [f32], output: &mut [f32]) {
    let GatedDeltaInput {
        queries,
        keys,
        values,
        gates,
        betas,
        key_heads,
        value_heads,
        head_size,
    } = input;
    let query_scale = (head_size as f32).sqrt().recip();
    for value_head in 0..value_heads {
        // GGML's fused GDN broadcasts key heads cyclically (`h_v % h_k`).
        let key_head = value_head % key_heads;
        let query = &queries[key_head * head_size..(key_head + 1) * head_size];
        let key = &keys[key_head * head_size..(key_head + 1) * head_size];
        let value = &values[value_head * head_size..(value_head + 1) * head_size];
        let matrix = &mut state
            [value_head * head_size * head_size..(value_head + 1) * head_size * head_size];
        let decay = gates[value_head].exp();
        for element in matrix.iter_mut() {
            *element *= decay;
        }
        for value_component in 0..head_size {
            let mut prediction = 0.0_f32;
            for key_component in 0..head_size {
                prediction = matrix[key_component * head_size + value_component]
                    .mul_add(key[key_component], prediction);
            }
            let delta = (value[value_component] - prediction) * betas[value_head];
            for key_component in 0..head_size {
                matrix[key_component * head_size + value_component] += key[key_component] * delta;
            }
        }
        let target = &mut output[value_head * head_size..(value_head + 1) * head_size];
        for value_component in 0..head_size {
            let mut sum = 0.0_f32;
            for key_component in 0..head_size {
                sum = matrix[key_component * head_size + value_component]
                    .mul_add(query[key_component] * query_scale, sum);
            }
            target[value_component] = sum;
        }
    }
}

fn partial_rope_neox_in_place(
    values: &mut [f32],
    heads: usize,
    head_size: usize,
    rotary_size: usize,
    position: usize,
    theta: f32,
) -> Result<(), RuntimeError> {
    if heads.checked_mul(head_size) != Some(values.len())
        || rotary_size > head_size
        || !rotary_size.is_multiple_of(2)
        || rotary_size == 0
        || !theta.is_finite()
        || theta <= 0.0
    {
        return Err(RuntimeError::InvalidConfiguration(
            "invalid Qwen partial-RoPE dimensions or frequency base".to_owned(),
        ));
    }
    let half = rotary_size / 2;
    for head in values.chunks_exact_mut(head_size) {
        for pair in 0..half {
            let exponent = -((2 * pair) as f32) / rotary_size as f32;
            let angle = position as f32 * theta.powf(exponent);
            let (sin, cos) = angle.sin_cos();
            let first = head[pair];
            let second = head[half + pair];
            head[pair] = first * cos - second * sin;
            head[half + pair] = first * sin + second * cos;
        }
    }
    Ok(())
}

fn l2_normalize_in_place(values: &mut [f32], epsilon: f32) {
    let norm = values
        .iter()
        .fold(0.0_f32, |sum, value| value.mul_add(*value, sum))
        .sqrt()
        .max(epsilon);
    for value in values {
        *value /= norm;
    }
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .fold(0.0_f32, |sum, (left, right)| left.mul_add(*right, sum))
}

fn add(left: &[f32], right: &[f32]) -> Vec<f32> {
    left.iter().zip(right).map(|(a, b)| a + b).collect()
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

fn silu(value: f32) -> f32 {
    value * sigmoid(value)
}

fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else if value < -20.0 {
        value.exp()
    } else {
        value.exp().ln_1p()
    }
}

fn common_prefix_length(left: &[u32], right: &[u32]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left, right)| left == right)
        .count()
}

#[cfg(test)]
mod tests {
    use super::{GatedDeltaInput, gated_delta_update, partial_rope_neox_in_place};

    #[test]
    fn gated_delta_rule_decays_updates_then_queries_state() {
        let mut state = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 2];
        gated_delta_update(
            GatedDeltaInput {
                queries: &[2.0, -1.0],
                keys: &[0.5, -0.5],
                values: &[3.0, -2.0],
                gates: &[0.0],
                betas: &[0.25],
                key_heads: 1,
                value_heads: 1,
                head_size: 2,
            },
            &mut state,
            &mut output,
        );
        assert_close(&state, &[1.5, 1.875, 2.5, 4.125], 1e-6);
        assert_close(&output, &[0.353_553_38, -0.265_165_03], 1e-6);
    }

    #[test]
    fn recurrent_key_heads_repeat_cyclically_like_ggml() {
        let mut state = vec![0.0; 4];
        let mut output = vec![0.0; 4];
        gated_delta_update(
            GatedDeltaInput {
                queries: &[1.0, 2.0],
                keys: &[1.0, 1.0],
                values: &[1.0, 2.0, 3.0, 4.0],
                gates: &[0.0; 4],
                betas: &[1.0; 4],
                key_heads: 2,
                value_heads: 4,
                head_size: 1,
            },
            &mut state,
            &mut output,
        );
        assert_eq!(output, [1.0, 4.0, 3.0, 8.0]);
    }

    #[test]
    fn partial_rope_rotates_only_the_configured_prefix() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        partial_rope_neox_in_place(&mut values, 1, 6, 4, 1, 10_000.0).unwrap();
        let (sin, cos) = 1.0_f32.sin_cos();
        assert_close(
            &values,
            &[
                cos - 3.0 * sin,
                2.0 * 0.999_95 - 4.0 * 0.009_999_833,
                sin + 3.0 * cos,
                2.0 * 0.009_999_833 + 4.0 * 0.999_95,
                5.0,
                6.0,
            ],
            2e-5,
        );
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "value {index}: {actual} != {expected}"
            );
        }
    }
}
