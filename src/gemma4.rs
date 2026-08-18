//! Gemma 4 architecture metadata, tokenizer, and chat framing.

pub mod cpu;
#[cfg(target_os = "macos")]
pub mod metal;

use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::gguf::{GgufFile, MetadataArray, MetadataValue, TensorInfo};

pub const RUNTIME_ARRAY_KEYS: [&str; 6] = [
    "gemma4.attention.head_count_kv",
    "gemma4.attention.sliding_window_pattern",
    "tokenizer.ggml.eos_token_ids",
    "tokenizer.ggml.scores",
    "tokenizer.ggml.token_type",
    "tokenizer.ggml.tokens",
];

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum Gemma4Error {
    #[error("missing Gemma 4 GGUF metadata {0:?}")]
    MissingMetadata(&'static str),

    #[error("Gemma 4 GGUF metadata {key:?} has the wrong type; expected {expected}")]
    MetadataType {
        key: &'static str,
        expected: &'static str,
    },

    #[error("Gemma 4 26B metadata {key:?} is {actual}; expected {expected}")]
    MetadataMismatch {
        key: &'static str,
        expected: String,
        actual: String,
    },

    #[error("tokenizer array {key:?} has {actual} entries; expected {expected}")]
    TokenizerLength {
        key: &'static str,
        expected: usize,
        actual: usize,
    },

    #[error("tokenizer contains duplicate token text {0:?}")]
    DuplicateToken(String),

    #[error("tokenizer special token {token:?} is missing")]
    MissingSpecialToken { token: &'static str },

    #[error("token ID {id} from {key:?} is outside the {vocab_size}-token vocabulary")]
    InvalidTokenId {
        key: &'static str,
        id: i64,
        vocab_size: usize,
    },

    #[error("chat system message must be first")]
    SystemMessageOrder,

    #[error("missing Gemma 4 tensor {0:?}")]
    MissingTensor(String),

    #[error("Gemma 4 tensor {tensor:?} has shape {actual:?}; expected {expected:?}")]
    TensorShape {
        tensor: String,
        expected: Vec<u64>,
        actual: Vec<u64>,
    },

    #[error("Gemma 4 full-attention layer {layer} must derive V from K, but {tensor:?} exists")]
    UnexpectedValueTensor { layer: usize, tensor: String },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Gemma4Config {
    pub context_length: u32,
    pub hidden_size: u32,
    pub layer_count: u32,
    pub attention_heads: u32,
    pub kv_heads_by_layer: Vec<u32>,
    pub key_length: u32,
    pub key_length_swa: u32,
    pub value_length: u32,
    pub value_length_swa: u32,
    pub shared_kv_layers: u32,
    pub rms_epsilon: f32,
    pub sliding_window: u32,
    pub sliding_layers: Vec<bool>,
    pub feed_forward_length: u32,
    pub expert_feed_forward_length: u32,
    pub expert_count: u32,
    pub experts_used: u32,
    pub final_logit_softcap: f32,
    pub rope_dimensions: u32,
    pub rope_dimensions_swa: u32,
    pub rope_frequency_base: f32,
    pub rope_frequency_base_swa: f32,
    pub per_layer_embedding_length: u32,
}

impl Gemma4Config {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, Gemma4Error> {
        expect_string(gguf, "general.architecture", "gemma4")?;
        let kv_heads_by_layer = required_i32_array(gguf, "gemma4.attention.head_count_kv")?
            .iter()
            .map(|value| {
                u32::try_from(*value).map_err(|_| Gemma4Error::MetadataMismatch {
                    key: "gemma4.attention.head_count_kv",
                    expected: "non-negative head counts".to_owned(),
                    actual: value.to_string(),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let config = Self {
            context_length: required_u32(gguf, "gemma4.context_length")?,
            hidden_size: required_u32(gguf, "gemma4.embedding_length")?,
            layer_count: required_u32(gguf, "gemma4.block_count")?,
            attention_heads: required_u32(gguf, "gemma4.attention.head_count")?,
            kv_heads_by_layer,
            key_length: required_u32(gguf, "gemma4.attention.key_length")?,
            key_length_swa: required_u32(gguf, "gemma4.attention.key_length_swa")?,
            value_length: required_u32(gguf, "gemma4.attention.value_length")?,
            value_length_swa: required_u32(gguf, "gemma4.attention.value_length_swa")?,
            shared_kv_layers: required_u32(gguf, "gemma4.attention.shared_kv_layers")?,
            rms_epsilon: required_f32(gguf, "gemma4.attention.layer_norm_rms_epsilon")?,
            sliding_window: required_u32(gguf, "gemma4.attention.sliding_window")?,
            sliding_layers: required_bool_array(gguf, "gemma4.attention.sliding_window_pattern")?
                .to_vec(),
            feed_forward_length: required_u32(gguf, "gemma4.feed_forward_length")?,
            expert_feed_forward_length: required_u32(gguf, "gemma4.expert_feed_forward_length")?,
            expert_count: required_u32(gguf, "gemma4.expert_count")?,
            experts_used: required_u32(gguf, "gemma4.expert_used_count")?,
            final_logit_softcap: required_f32(gguf, "gemma4.final_logit_softcapping")?,
            rope_dimensions: required_u32(gguf, "gemma4.rope.dimension_count")?,
            rope_dimensions_swa: required_u32(gguf, "gemma4.rope.dimension_count_swa")?,
            rope_frequency_base: required_f32(gguf, "gemma4.rope.freq_base")?,
            rope_frequency_base_swa: required_f32(gguf, "gemma4.rope.freq_base_swa")?,
            per_layer_embedding_length: required_u32(
                gguf,
                "gemma4.embedding_length_per_layer_input",
            )?,
        };
        config.validate_26b()?;
        Ok(config)
    }

    fn validate_26b(&self) -> Result<(), Gemma4Error> {
        expect_field("gemma4.context_length", self.context_length, 262_144)?;
        expect_field("gemma4.embedding_length", self.hidden_size, 2_816)?;
        expect_field("gemma4.block_count", self.layer_count, 30)?;
        expect_field("gemma4.attention.head_count", self.attention_heads, 16)?;
        expect_field("gemma4.attention.key_length", self.key_length, 512)?;
        expect_field("gemma4.attention.key_length_swa", self.key_length_swa, 256)?;
        expect_field("gemma4.attention.value_length", self.value_length, 512)?;
        expect_field(
            "gemma4.attention.value_length_swa",
            self.value_length_swa,
            256,
        )?;
        expect_field(
            "gemma4.attention.shared_kv_layers",
            self.shared_kv_layers,
            0,
        )?;
        expect_field(
            "gemma4.embedding_length_per_layer_input",
            self.per_layer_embedding_length,
            0,
        )?;
        expect_field(
            "gemma4.attention.sliding_window",
            self.sliding_window,
            1_024,
        )?;
        expect_field(
            "gemma4.feed_forward_length",
            self.feed_forward_length,
            2_112,
        )?;
        expect_field(
            "gemma4.expert_feed_forward_length",
            self.expert_feed_forward_length,
            704,
        )?;
        expect_field("gemma4.expert_count", self.expert_count, 128)?;
        expect_field("gemma4.expert_used_count", self.experts_used, 8)?;
        expect_float(
            "gemma4.attention.layer_norm_rms_epsilon",
            self.rms_epsilon,
            1e-6,
        )?;
        expect_float(
            "gemma4.final_logit_softcapping",
            self.final_logit_softcap,
            30.0,
        )?;
        expect_field("gemma4.rope.dimension_count", self.rope_dimensions, 512)?;
        expect_field(
            "gemma4.rope.dimension_count_swa",
            self.rope_dimensions_swa,
            256,
        )?;
        expect_float(
            "gemma4.rope.freq_base",
            self.rope_frequency_base,
            1_000_000.0,
        )?;
        expect_float(
            "gemma4.rope.freq_base_swa",
            self.rope_frequency_base_swa,
            10_000.0,
        )?;

        let expected_sliding: Vec<_> = (0..30).map(|layer| layer % 6 != 5).collect();
        if self.sliding_layers != expected_sliding {
            return Err(mismatch(
                "gemma4.attention.sliding_window_pattern",
                format!("{expected_sliding:?}"),
                format!("{:?}", self.sliding_layers),
            ));
        }
        let expected_kv: Vec<_> = expected_sliding
            .iter()
            .map(|sliding| if *sliding { 8 } else { 2 })
            .collect();
        if self.kv_heads_by_layer != expected_kv {
            return Err(mismatch(
                "gemma4.attention.head_count_kv",
                format!("{expected_kv:?}"),
                format!("{:?}", self.kv_heads_by_layer),
            ));
        }
        Ok(())
    }
}

/// Indices for the text tower's tensors in a parsed GGUF index.
///
/// Keeping indices instead of borrowing tensor descriptors lets the runtime
/// own this validated map alongside a [`crate::gguf::TensorSource`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gemma4Weights {
    pub token_embedding: usize,
    pub output_norm: usize,
    pub output: usize,
    pub rope_frequencies: usize,
    pub tied_output: bool,
    pub layers: Vec<Gemma4LayerWeights>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gemma4LayerWeights {
    pub attention_norm: usize,
    pub attention_query: usize,
    pub attention_key: usize,
    pub attention_value: Option<usize>,
    pub attention_output: usize,
    pub query_norm: usize,
    pub key_norm: usize,
    pub post_attention_norm: usize,
    pub feed_forward_norm: usize,
    pub feed_forward_gate: usize,
    pub feed_forward_up: usize,
    pub feed_forward_down: usize,
    pub shared_post_norm: usize,
    pub post_feed_forward_norm: usize,
    pub routed_pre_norm: usize,
    pub routed_post_norm: usize,
    pub router: usize,
    pub router_scale: usize,
    pub expert_gate_up: usize,
    pub expert_down: usize,
    pub expert_down_scale: usize,
    pub layer_output_scale: usize,
}

impl Gemma4Weights {
    /// Validate and bind the complete Gemma 4 text-tower tensor contract.
    pub fn from_gguf(gguf: &GgufFile, config: &Gemma4Config) -> Result<Self, Gemma4Error> {
        let hidden = u64::from(config.hidden_size);
        let vocabulary = required_array_length(gguf, "tokenizer.ggml.tokens")?;
        let token_embedding = required_tensor(gguf, "token_embd.weight", &[hidden, vocabulary])?;
        let output_norm = required_tensor(gguf, "output_norm.weight", &[hidden])?;
        let rope_frequencies = required_tensor(
            gguf,
            "rope_freqs.weight",
            &[u64::from(config.rope_dimensions) / 2],
        )?;
        let (output, tied_output) = match tensor_index(gguf, "output.weight") {
            Some(index) => {
                validate_shape(gguf, index, &[hidden, vocabulary])?;
                (index, false)
            }
            None => (token_embedding, true),
        };

        let mut layers = Vec::with_capacity(config.layer_count as usize);
        for layer in 0..config.layer_count as usize {
            let sliding = config.sliding_layers[layer];
            let head_dimension = if sliding {
                u64::from(config.key_length_swa)
            } else {
                u64::from(config.key_length)
            };
            let attention_width = u64::from(config.attention_heads) * head_dimension;
            let kv_width = u64::from(config.kv_heads_by_layer[layer]) * head_dimension;
            let prefix = format!("blk.{layer}");
            let name = |suffix: &str| format!("{prefix}.{suffix}");

            let value_name = name("attn_v.weight");
            let attention_value = if sliding {
                Some(required_tensor(gguf, &value_name, &[hidden, kv_width])?)
            } else if tensor_index(gguf, &value_name).is_some() {
                return Err(Gemma4Error::UnexpectedValueTensor {
                    layer,
                    tensor: value_name,
                });
            } else {
                None
            };

            layers.push(Gemma4LayerWeights {
                attention_norm: required_tensor(gguf, &name("attn_norm.weight"), &[hidden])?,
                attention_query: required_tensor(
                    gguf,
                    &name("attn_q.weight"),
                    &[hidden, attention_width],
                )?,
                attention_key: required_tensor(gguf, &name("attn_k.weight"), &[hidden, kv_width])?,
                attention_value,
                attention_output: required_tensor(
                    gguf,
                    &name("attn_output.weight"),
                    &[attention_width, hidden],
                )?,
                query_norm: required_tensor(gguf, &name("attn_q_norm.weight"), &[head_dimension])?,
                key_norm: required_tensor(gguf, &name("attn_k_norm.weight"), &[head_dimension])?,
                post_attention_norm: required_tensor(
                    gguf,
                    &name("post_attention_norm.weight"),
                    &[hidden],
                )?,
                feed_forward_norm: required_tensor(gguf, &name("ffn_norm.weight"), &[hidden])?,
                feed_forward_gate: required_tensor(
                    gguf,
                    &name("ffn_gate.weight"),
                    &[hidden, u64::from(config.feed_forward_length)],
                )?,
                feed_forward_up: required_tensor(
                    gguf,
                    &name("ffn_up.weight"),
                    &[hidden, u64::from(config.feed_forward_length)],
                )?,
                feed_forward_down: required_tensor(
                    gguf,
                    &name("ffn_down.weight"),
                    &[u64::from(config.feed_forward_length), hidden],
                )?,
                shared_post_norm: required_tensor(
                    gguf,
                    &name("post_ffw_norm_1.weight"),
                    &[hidden],
                )?,
                post_feed_forward_norm: required_tensor(
                    gguf,
                    &name("post_ffw_norm.weight"),
                    &[hidden],
                )?,
                routed_pre_norm: required_tensor(gguf, &name("pre_ffw_norm_2.weight"), &[hidden])?,
                routed_post_norm: required_tensor(
                    gguf,
                    &name("post_ffw_norm_2.weight"),
                    &[hidden],
                )?,
                router: required_tensor(
                    gguf,
                    &name("ffn_gate_inp.weight"),
                    &[hidden, u64::from(config.expert_count)],
                )?,
                router_scale: required_tensor(gguf, &name("ffn_gate_inp.scale"), &[hidden])?,
                expert_gate_up: required_tensor(
                    gguf,
                    &name("ffn_gate_up_exps.weight"),
                    &[
                        hidden,
                        2 * u64::from(config.expert_feed_forward_length),
                        u64::from(config.expert_count),
                    ],
                )?,
                expert_down: required_tensor(
                    gguf,
                    &name("ffn_down_exps.weight"),
                    &[
                        u64::from(config.expert_feed_forward_length),
                        hidden,
                        u64::from(config.expert_count),
                    ],
                )?,
                expert_down_scale: required_tensor(
                    gguf,
                    &name("ffn_down_exps.scale"),
                    &[u64::from(config.expert_count)],
                )?,
                layer_output_scale: required_tensor(
                    gguf,
                    &name("layer_output_scale.weight"),
                    &[1],
                )?,
            });
        }

        Ok(Self {
            token_embedding,
            output_norm,
            output,
            rope_frequencies,
            tied_output,
            layers,
        })
    }

    pub fn tensor<'a>(&self, gguf: &'a GgufFile, index: usize) -> &'a TensorInfo {
        &gguf.tensors[index]
    }
}

fn required_tensor(gguf: &GgufFile, name: &str, dimensions: &[u64]) -> Result<usize, Gemma4Error> {
    let index =
        tensor_index(gguf, name).ok_or_else(|| Gemma4Error::MissingTensor(name.to_owned()))?;
    validate_shape(gguf, index, dimensions)?;
    Ok(index)
}

fn tensor_index(gguf: &GgufFile, name: &str) -> Option<usize> {
    gguf.tensors.iter().position(|tensor| tensor.name == name)
}

fn validate_shape(gguf: &GgufFile, index: usize, expected: &[u64]) -> Result<(), Gemma4Error> {
    let tensor = &gguf.tensors[index];
    if tensor.dimensions != expected {
        return Err(Gemma4Error::TensorShape {
            tensor: tensor.name.clone(),
            expected: expected.to_vec(),
            actual: tensor.dimensions.clone(),
        });
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum TokenType {
    Undefined = 0,
    Normal = 1,
    Unknown = 2,
    Control = 3,
    UserDefined = 4,
    Unused = 5,
    Byte = 6,
}

impl TokenType {
    fn from_i32(value: i32) -> Option<Self> {
        Some(match value {
            0 => Self::Undefined,
            1 => Self::Normal,
            2 => Self::Unknown,
            3 => Self::Control,
            4 => Self::UserDefined,
            5 => Self::Unused,
            6 => Self::Byte,
            _ => return None,
        })
    }

    fn is_special(self) -> bool {
        matches!(self, Self::Control | Self::UserDefined)
    }
}

#[derive(Debug)]
pub struct Gemma4Tokenizer {
    tokens: Vec<String>,
    token_types: Vec<TokenType>,
    scores: Vec<f32>,
    token_to_id: HashMap<String, u32>,
    byte_tokens: [Option<u32>; 256],
    specials_by_first_byte: HashMap<u8, Vec<(String, u32)>>,
    pub bos_id: u32,
    pub eos_id: u32,
    pub padding_id: u32,
    pub unknown_id: u32,
    pub end_of_turn_id: u32,
    pub stop_token_ids: HashSet<u32>,
}

impl Gemma4Tokenizer {
    /// Build the tokenizer and move its large arrays out of the GGUF metadata
    /// map so the runtime does not keep duplicate vocabulary and score tables.
    pub fn take_from_gguf(gguf: &mut GgufFile) -> Result<Self, Gemma4Error> {
        expect_string(gguf, "tokenizer.ggml.model", "llama")?;
        expect_string(gguf, "tokenizer.ggml.pre", "gemma4")?;
        let bos_id = required_u32(gguf, "tokenizer.ggml.bos_token_id")?;
        let eos_id = required_u32(gguf, "tokenizer.ggml.eos_token_id")?;
        let padding_id = required_u32(gguf, "tokenizer.ggml.padding_token_id")?;
        let unknown_id = required_u32(gguf, "tokenizer.ggml.unknown_token_id")?;
        let add_bos = required_bool(gguf, "tokenizer.ggml.add_bos_token")?;
        let add_eos = required_bool(gguf, "tokenizer.ggml.add_eos_token")?;
        if add_bos || add_eos {
            return Err(mismatch(
                if add_bos {
                    "tokenizer.ggml.add_bos_token"
                } else {
                    "tokenizer.ggml.add_eos_token"
                },
                "false".to_owned(),
                "true".to_owned(),
            ));
        }

        let tokens = take_array(gguf, "tokenizer.ggml.tokens")?;
        let tokens = match tokens {
            MetadataArray::String(values) => values,
            _ => return Err(wrong_type("tokenizer.ggml.tokens", "string array")),
        };
        let scores = take_array(gguf, "tokenizer.ggml.scores")?;
        let scores = match scores {
            MetadataArray::F32(values) => values,
            _ => return Err(wrong_type("tokenizer.ggml.scores", "F32 array")),
        };
        let raw_types = take_array(gguf, "tokenizer.ggml.token_type")?;
        let raw_types = match raw_types {
            MetadataArray::I32(values) => values,
            _ => return Err(wrong_type("tokenizer.ggml.token_type", "I32 array")),
        };
        let eos_ids = take_array(gguf, "tokenizer.ggml.eos_token_ids")?;
        let eos_ids = match eos_ids {
            MetadataArray::I32(values) => values,
            _ => return Err(wrong_type("tokenizer.ggml.eos_token_ids", "I32 array")),
        };
        let vocab_size = tokens.len();
        expect_length("tokenizer.ggml.tokens", vocab_size, 262_144)?;
        expect_length("tokenizer.ggml.scores", scores.len(), vocab_size)?;
        expect_length("tokenizer.ggml.token_type", raw_types.len(), vocab_size)?;

        let mut token_types = Vec::with_capacity(raw_types.len());
        for value in raw_types {
            token_types.push(TokenType::from_i32(value).ok_or_else(|| {
                Gemma4Error::MetadataMismatch {
                    key: "tokenizer.ggml.token_type",
                    expected: "token types in 0..=6".to_owned(),
                    actual: value.to_string(),
                }
            })?);
        }
        let mut token_to_id = HashMap::with_capacity(tokens.len());
        for (id, token) in tokens.iter().enumerate() {
            let id = u32::try_from(id).expect("Gemma vocabulary fits u32");
            if token_to_id.insert(token.clone(), id).is_some() {
                return Err(Gemma4Error::DuplicateToken(token.clone()));
            }
        }
        let mut byte_tokens = [None; 256];
        for (id, token) in tokens.iter().enumerate() {
            if let Some(byte) = parse_byte_token(token) {
                byte_tokens[byte as usize] = Some(id as u32);
            }
        }

        for (key, id) in [
            ("tokenizer.ggml.bos_token_id", i64::from(bos_id)),
            ("tokenizer.ggml.eos_token_id", i64::from(eos_id)),
            ("tokenizer.ggml.padding_token_id", i64::from(padding_id)),
            ("tokenizer.ggml.unknown_token_id", i64::from(unknown_id)),
        ] {
            validate_id(key, id, vocab_size)?;
        }
        let mut stop_token_ids = HashSet::new();
        for id in eos_ids {
            validate_id("tokenizer.ggml.eos_token_ids", i64::from(id), vocab_size)?;
            stop_token_ids.insert(id as u32);
        }
        if !stop_token_ids.contains(&eos_id) {
            return Err(mismatch(
                "tokenizer.ggml.eos_token_ids",
                format!("an array containing {eos_id}"),
                format!("{stop_token_ids:?}"),
            ));
        }

        let end_of_turn_id = required_special(&token_to_id, "<turn|>")?;
        for (special, expected_id) in [
            ("<pad>", padding_id),
            ("<eos>", eos_id),
            ("<bos>", bos_id),
            ("<unk>", unknown_id),
        ] {
            let actual_id = required_special(&token_to_id, special)?;
            if actual_id != expected_id {
                return Err(mismatch(
                    "tokenizer.ggml.tokens",
                    format!("{special} at ID {expected_id}"),
                    format!("{special} at ID {actual_id}"),
                ));
            }
        }
        for special in ["<|turn>", "<|channel>", "<channel|>"] {
            required_special(&token_to_id, special)?;
        }
        if !stop_token_ids.contains(&end_of_turn_id) {
            return Err(mismatch(
                "tokenizer.ggml.eos_token_ids",
                format!("an array containing <turn|> ID {end_of_turn_id}"),
                format!("{stop_token_ids:?}"),
            ));
        }
        let mut specials_by_first_byte: HashMap<u8, Vec<(String, u32)>> = HashMap::new();
        for (id, (token, kind)) in tokens.iter().zip(&token_types).enumerate() {
            if kind.is_special() && !token.is_empty() {
                specials_by_first_byte
                    .entry(token.as_bytes()[0])
                    .or_default()
                    .push((token.clone(), id as u32));
            }
        }
        for candidates in specials_by_first_byte.values_mut() {
            candidates.sort_unstable_by(|left, right| right.0.len().cmp(&left.0.len()));
        }

        Ok(Self {
            tokens,
            token_types,
            scores,
            token_to_id,
            byte_tokens,
            specials_by_first_byte,
            bos_id,
            eos_id,
            padding_id,
            unknown_id,
            end_of_turn_id,
            stop_token_ids,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<u32> {
        let mut output = Vec::new();
        if add_bos {
            output.push(self.bos_id);
        }
        let mut raw_start = 0;
        let mut position = 0;
        while position < text.len() {
            let matched = self
                .specials_by_first_byte
                .get(&text.as_bytes()[position])
                .and_then(|candidates| {
                    candidates
                        .iter()
                        .find(|(special, _)| text[position..].starts_with(special))
                });
            if let Some((special, id)) = matched {
                self.encode_raw(&text[raw_start..position], &mut output);
                output.push(*id);
                position += special.len();
                raw_start = position;
            } else {
                position += text[position..]
                    .chars()
                    .next()
                    .expect("valid position")
                    .len_utf8();
            }
        }
        self.encode_raw(&text[raw_start..], &mut output);
        output
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        let mut bytes = Vec::new();
        for id in ids {
            let Some(token) = self.tokens.get(*id as usize) else {
                continue;
            };
            let kind = self.token_types[*id as usize];
            if skip_special && kind.is_special() {
                continue;
            }
            if let Some(byte) = parse_byte_token(token) {
                bytes.push(byte);
            } else {
                bytes.extend_from_slice(token.replace('▁', " ").as_bytes());
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn encode_raw(&self, raw: &str, output: &mut Vec<u32>) {
        if raw.is_empty() {
            return;
        }
        let normalized = format!("▁{}", raw.replace(' ', "▁"));
        self.encode_spm(&normalized, output);
    }

    fn encode_spm(&self, text: &str, output: &mut Vec<u32>) {
        let mut symbols: Vec<Symbol> = text
            .chars()
            .enumerate()
            .map(|(index, character)| Symbol {
                text: character.to_string(),
                previous: index.checked_sub(1),
                next: None,
                active: true,
            })
            .collect();
        for index in 0..symbols.len().saturating_sub(1) {
            symbols[index].next = Some(index + 1);
        }
        let mut work: BinaryHeap<ScoredMerge> = BinaryHeap::new();
        for index in 0..symbols.len().saturating_sub(1) {
            self.queue_spm_merge(&symbols, index, index + 1, &mut work);
        }
        while let Some(candidate) = work.pop() {
            let left = candidate.left;
            let right = candidate.right;
            if !symbols[left].active
                || !symbols[right].active
                || symbols[left].next != Some(right)
                || symbols[left].text.len() + symbols[right].text.len() != candidate.byte_len
            {
                continue;
            }
            let right_text = std::mem::take(&mut symbols[right].text);
            symbols[left].text.push_str(&right_text);
            symbols[right].active = false;
            let next = symbols[right].next;
            symbols[left].next = next;
            if let Some(next) = next {
                symbols[next].previous = Some(left);
            }
            if let Some(previous) = symbols[left].previous {
                self.queue_spm_merge(&symbols, previous, left, &mut work);
            }
            if let Some(next) = symbols[left].next {
                self.queue_spm_merge(&symbols, left, next, &mut work);
            }
        }
        let mut current = (!symbols.is_empty()).then_some(0);
        while let Some(index) = current {
            self.emit_spm_symbol(index, &symbols, output);
            current = symbols[index].next;
        }
    }

    fn queue_spm_merge(
        &self,
        symbols: &[Symbol],
        left: usize,
        right: usize,
        work: &mut BinaryHeap<ScoredMerge>,
    ) {
        let combined = format!("{}{}", symbols[left].text, symbols[right].text);
        if let Some(id) = self.token_to_id.get(&combined) {
            work.push(ScoredMerge {
                score: self.scores[*id as usize],
                left,
                right,
                byte_len: combined.len(),
            });
        }
    }

    fn emit_spm_symbol(&self, index: usize, symbols: &[Symbol], output: &mut Vec<u32>) {
        let text = &symbols[index].text;
        if let Some(id) = self.token_to_id.get(text) {
            output.push(*id);
        } else {
            for byte in text.as_bytes() {
                output.push(self.byte_tokens[*byte as usize].unwrap_or(self.unknown_id));
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ScoredMerge {
    score: f32,
    left: usize,
    right: usize,
    byte_len: usize,
}

impl PartialEq for ScoredMerge {
    fn eq(&self, other: &Self) -> bool {
        self.score.to_bits() == other.score.to_bits()
            && self.left == other.left
            && self.right == other.right
            && self.byte_len == other.byte_len
    }
}

impl Eq for ScoredMerge {}

impl PartialOrd for ScoredMerge {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredMerge {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.left.cmp(&self.left))
    }
}

#[derive(Debug)]
struct Symbol {
    text: String,
    previous: Option<usize>,
    next: Option<usize>,
    active: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: ChatRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }
}

pub fn render_chat(messages: &[ChatMessage]) -> Result<String, Gemma4Error> {
    let mut rendered = String::from("<bos>");
    for (index, message) in messages.iter().enumerate() {
        if message.role == ChatRole::System && index != 0 {
            return Err(Gemma4Error::SystemMessageOrder);
        }
        let role = match message.role {
            ChatRole::System => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "model",
        };
        rendered.push_str("<|turn>");
        rendered.push_str(role);
        rendered.push('\n');
        rendered.push_str(message.content.trim());
        rendered.push_str("<turn|>\n");
    }
    rendered.push_str("<|turn>model\n<|channel>thought\n<channel|>");
    Ok(rendered)
}

fn required_u32(gguf: &GgufFile, key: &'static str) -> Result<u32, Gemma4Error> {
    required(gguf, key)?
        .as_u32()
        .ok_or_else(|| wrong_type(key, "U32"))
}

fn required_f32(gguf: &GgufFile, key: &'static str) -> Result<f32, Gemma4Error> {
    required(gguf, key)?
        .as_f32()
        .ok_or_else(|| wrong_type(key, "F32"))
}

fn required_bool(gguf: &GgufFile, key: &'static str) -> Result<bool, Gemma4Error> {
    required(gguf, key)?
        .as_bool()
        .ok_or_else(|| wrong_type(key, "BOOL"))
}

fn required_array_length(gguf: &GgufFile, key: &'static str) -> Result<u64, Gemma4Error> {
    match required(gguf, key)? {
        MetadataValue::Array { length, .. } => Ok(*length),
        MetadataValue::ArrayValues(values) => Ok(values.len() as u64),
        _ => Err(wrong_type(key, "array")),
    }
}

fn required_i32_array<'a>(gguf: &'a GgufFile, key: &'static str) -> Result<&'a [i32], Gemma4Error> {
    required(gguf, key)?
        .as_array()
        .and_then(MetadataArray::as_i32s)
        .ok_or_else(|| wrong_type(key, "captured I32 array"))
}

fn required_bool_array<'a>(
    gguf: &'a GgufFile,
    key: &'static str,
) -> Result<&'a [bool], Gemma4Error> {
    required(gguf, key)?
        .as_array()
        .and_then(MetadataArray::as_bools)
        .ok_or_else(|| wrong_type(key, "captured BOOL array"))
}

fn required<'a>(gguf: &'a GgufFile, key: &'static str) -> Result<&'a MetadataValue, Gemma4Error> {
    gguf.metadata
        .get(key)
        .ok_or(Gemma4Error::MissingMetadata(key))
}

fn take_array(gguf: &mut GgufFile, key: &'static str) -> Result<MetadataArray, Gemma4Error> {
    match gguf
        .metadata
        .remove(key)
        .ok_or(Gemma4Error::MissingMetadata(key))?
    {
        MetadataValue::ArrayValues(values) => Ok(values),
        _ => Err(wrong_type(key, "captured array")),
    }
}

fn expect_string(
    gguf: &GgufFile,
    key: &'static str,
    expected: &'static str,
) -> Result<(), Gemma4Error> {
    let actual = required(gguf, key)?
        .as_str()
        .ok_or_else(|| wrong_type(key, "string"))?;
    if actual != expected {
        return Err(mismatch(key, expected.to_owned(), actual.to_owned()));
    }
    Ok(())
}

fn expect_field<T>(key: &'static str, actual: T, expected: T) -> Result<(), Gemma4Error>
where
    T: PartialEq + ToString,
{
    if actual != expected {
        return Err(mismatch(key, expected.to_string(), actual.to_string()));
    }
    Ok(())
}

fn expect_float(key: &'static str, actual: f32, expected: f32) -> Result<(), Gemma4Error> {
    if actual.to_bits() != expected.to_bits() {
        return Err(mismatch(key, expected.to_string(), actual.to_string()));
    }
    Ok(())
}

fn expect_length(key: &'static str, actual: usize, expected: usize) -> Result<(), Gemma4Error> {
    if actual != expected {
        return Err(Gemma4Error::TokenizerLength {
            key,
            expected,
            actual,
        });
    }
    Ok(())
}

fn validate_id(key: &'static str, id: i64, vocab_size: usize) -> Result<(), Gemma4Error> {
    if id < 0 || usize::try_from(id).map_or(true, |id| id >= vocab_size) {
        return Err(Gemma4Error::InvalidTokenId {
            key,
            id,
            vocab_size,
        });
    }
    Ok(())
}

fn required_special(
    token_to_id: &HashMap<String, u32>,
    token: &'static str,
) -> Result<u32, Gemma4Error> {
    token_to_id
        .get(token)
        .copied()
        .ok_or(Gemma4Error::MissingSpecialToken { token })
}

fn wrong_type(key: &'static str, expected: &'static str) -> Gemma4Error {
    Gemma4Error::MetadataType { key, expected }
}

fn mismatch(key: &'static str, expected: String, actual: String) -> Gemma4Error {
    Gemma4Error::MetadataMismatch {
        key,
        expected,
        actual,
    }
}

fn parse_byte_token(token: &str) -> Option<u8> {
    if token.len() == 6 && token.starts_with("<0x") && token.ends_with('>') {
        u8::from_str_radix(&token[3..5], 16).ok()
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use super::{
        ChatMessage, ChatRole, Gemma4Error, Gemma4Tokenizer, TokenType, parse_byte_token,
        render_chat,
    };

    #[test]
    fn renders_the_pinned_text_chat_template() {
        let rendered = render_chat(&[
            ChatMessage::new(ChatRole::System, " Be terse. "),
            ChatMessage::new(ChatRole::User, " Hello "),
            ChatMessage::new(ChatRole::Assistant, " Hi "),
        ])
        .unwrap();
        assert_eq!(
            rendered,
            "<bos><|turn>system\nBe terse.<turn|>\n\
             <|turn>user\nHello<turn|>\n\
             <|turn>model\nHi<turn|>\n\
             <|turn>model\n<|channel>thought\n<channel|>"
        );
    }

    #[test]
    fn rejects_a_late_system_message() {
        assert_eq!(
            render_chat(&[
                ChatMessage::new(ChatRole::User, "hello"),
                ChatMessage::new(ChatRole::System, "late"),
            ]),
            Err(Gemma4Error::SystemMessageOrder)
        );
    }

    #[test]
    fn recognizes_spm_byte_fallback_tokens() {
        assert_eq!(parse_byte_token("<0x00>"), Some(0));
        assert_eq!(parse_byte_token("<0xFF>"), Some(255));
        assert_eq!(parse_byte_token("hello"), None);
    }

    #[test]
    fn spm_merges_by_score_and_partitions_special_tokens() {
        let tokens = [
            "<pad>",
            "<eos>",
            "<bos>",
            "<unk>",
            "a",
            "b",
            "ab",
            "▁",
            "▁a",
            "<turn|>",
            "<|turn>",
            "<|channel>",
            "<channel|>",
        ]
        .map(str::to_owned)
        .to_vec();
        let token_types = vec![
            TokenType::Control,
            TokenType::Control,
            TokenType::Control,
            TokenType::Unknown,
            TokenType::Normal,
            TokenType::Normal,
            TokenType::Normal,
            TokenType::Normal,
            TokenType::Normal,
            TokenType::Control,
            TokenType::Control,
            TokenType::Control,
            TokenType::Control,
        ];
        let token_to_id = tokens
            .iter()
            .enumerate()
            .map(|(id, token)| (token.clone(), id as u32))
            .collect();
        let specials_by_first_byte = HashMap::from([(
            b'<',
            vec![
                ("<|channel>".to_owned(), 11),
                ("<channel|>".to_owned(), 12),
                ("<|turn>".to_owned(), 10),
                ("<turn|>".to_owned(), 9),
                ("<pad>".to_owned(), 0),
                ("<eos>".to_owned(), 1),
                ("<bos>".to_owned(), 2),
            ],
        )]);
        let tokenizer = Gemma4Tokenizer {
            tokens,
            token_types,
            scores: vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0,
            ],
            token_to_id,
            byte_tokens: [None; 256],
            specials_by_first_byte,
            bos_id: 2,
            eos_id: 1,
            padding_id: 0,
            unknown_id: 3,
            end_of_turn_id: 9,
            stop_token_ids: HashSet::from([1, 9]),
        };

        assert_eq!(tokenizer.encode("ab a", true), [2, 8, 5, 8]);
        assert_eq!(tokenizer.decode(&[8, 5, 8], true), " ab a");
        assert_eq!(tokenizer.encode("<bos>ab", false), [2, 8, 5]);
    }
}
