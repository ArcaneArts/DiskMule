//! GLM-5.2 model-specific configuration and safetensors mapping.

use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
};

use serde::Deserialize;

use crate::safetensors::{SafeDtype, SafeTensorIndex, SafeTensorInfo};

pub mod cpu;
pub mod expert;
#[cfg(target_os = "macos")]
pub mod metal;
pub mod quant;
pub mod tokenizer;

const MAX_CONFIG_BYTES: u64 = 16 * 1024 * 1024;
const FP8_BLOCK: u64 = 128;

#[derive(Debug, thiserror::Error)]
pub enum Glm52Error {
    #[error("could not read GLM-5.2 configuration {path}: {source}")]
    ReadConfig {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("GLM-5.2 configuration {path} is {size} bytes; maximum is {maximum}")]
    ConfigTooLarge {
        path: PathBuf,
        size: u64,
        maximum: u64,
    },

    #[error("invalid GLM-5.2 configuration {path}: {message}")]
    InvalidConfig { path: PathBuf, message: String },

    #[error("GLM-5.2 field {field} has invalid value {value}: {reason}")]
    InvalidField {
        field: &'static str,
        value: String,
        reason: &'static str,
    },

    #[error("missing GLM-5.2 tensor {0:?}")]
    MissingTensor(String),

    #[error("GLM-5.2 tensor {tensor:?} has shape {actual:?}; expected {expected:?}")]
    TensorShape {
        tensor: String,
        expected: Vec<u64>,
        actual: Vec<u64>,
    },

    #[error("GLM-5.2 tensor {tensor:?} uses unsupported runtime dtype {dtype:?}")]
    TensorDtype { tensor: String, dtype: SafeDtype },

    #[error("FP8 tensor {tensor:?} requires F32 block scale {scale:?} with shape {expected:?}")]
    MissingFp8Scale {
        tensor: String,
        scale: String,
        expected: Vec<u64>,
    },

    #[error(transparent)]
    Quantization(#[from] quant::GlmQuantizationError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexerKind {
    Full,
    Shared,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Glm52Config {
    pub vocabulary_size: usize,
    pub hidden_size: usize,
    pub layer_count: usize,
    pub attention_heads: usize,
    pub routed_experts: usize,
    pub experts_per_token: usize,
    pub routed_intermediate_size: usize,
    pub dense_intermediate_size: usize,
    pub first_sparse_layer: usize,
    pub query_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub query_key_nope_size: usize,
    pub query_key_rope_size: usize,
    pub value_head_size: usize,
    pub shared_experts: usize,
    pub expert_groups: usize,
    pub selected_groups: usize,
    pub normalize_topk: bool,
    pub rms_epsilon: f32,
    pub rope_theta: f32,
    pub routed_scale: f32,
    pub index_topk: usize,
    pub index_heads: usize,
    pub index_head_size: usize,
    pub indexer_kinds: Vec<IndexerKind>,
    pub stop_token_ids: Vec<u32>,
}

#[derive(Debug, Deserialize)]
struct RawConfig {
    #[serde(default)]
    model_type: String,
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    n_routed_experts: usize,
    num_experts_per_tok: usize,
    moe_intermediate_size: usize,
    intermediate_size: usize,
    first_k_dense_replace: usize,
    q_lora_rank: usize,
    kv_lora_rank: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    v_head_dim: usize,
    n_shared_experts: usize,
    n_group: usize,
    topk_group: usize,
    norm_topk_prob: bool,
    #[serde(default = "default_epsilon")]
    rms_norm_eps: f32,
    #[serde(default = "default_routed_scale")]
    routed_scaling_factor: f32,
    #[serde(default)]
    rope_parameters: RawRope,
    #[serde(default)]
    index_topk: usize,
    #[serde(default)]
    index_n_heads: usize,
    #[serde(default)]
    index_head_dim: usize,
    #[serde(default)]
    indexer_types: Vec<String>,
    #[serde(default = "default_index_frequency")]
    index_topk_freq: usize,
    #[serde(default = "default_index_offset")]
    index_skip_topk_offset: usize,
    #[serde(default)]
    eos_token_id: Option<TokenIds>,
}

#[derive(Debug, Default, Deserialize)]
struct RawRope {
    #[serde(default = "default_rope_theta")]
    rope_theta: f32,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum TokenIds {
    One(u32),
    Many(Vec<u32>),
}

#[derive(Debug, Default, Deserialize)]
struct GenerationConfig {
    #[serde(default)]
    eos_token_id: Option<TokenIds>,
}

impl Glm52Config {
    pub fn from_directory(directory: impl AsRef<Path>) -> Result<Self, Glm52Error> {
        let directory = directory.as_ref();
        let path = directory.join("config.json");
        let raw: RawConfig = read_json(&path)?;
        let generation_path = directory.join("generation_config.json");
        let generation = if generation_path.is_file() {
            read_json::<GenerationConfig>(&generation_path)?
        } else {
            GenerationConfig::default()
        };
        Self::from_raw(raw, generation)
    }

    fn from_raw(raw: RawConfig, generation: GenerationConfig) -> Result<Self, Glm52Error> {
        if !matches!(raw.model_type.as_str(), "" | "glm_moe_dsa") {
            return Err(invalid(
                "model_type",
                raw.model_type,
                "expected glm_moe_dsa",
            ));
        }
        bounded("vocab_size", raw.vocab_size, 1, 1 << 24)?;
        bounded("hidden_size", raw.hidden_size, 1, 1 << 20)?;
        bounded("num_hidden_layers", raw.num_hidden_layers, 1, 128)?;
        bounded("num_attention_heads", raw.num_attention_heads, 1, 1_024)?;
        bounded("n_routed_experts", raw.n_routed_experts, 1, 4_096)?;
        bounded("num_experts_per_tok", raw.num_experts_per_tok, 1, 64)?;
        bounded(
            "moe_intermediate_size",
            raw.moe_intermediate_size,
            1,
            1 << 20,
        )?;
        bounded("intermediate_size", raw.intermediate_size, 1, 1 << 24)?;
        bounded(
            "first_k_dense_replace",
            raw.first_k_dense_replace,
            0,
            raw.num_hidden_layers,
        )?;
        bounded("q_lora_rank", raw.q_lora_rank, 1, 1 << 20)?;
        bounded("kv_lora_rank", raw.kv_lora_rank, 1, 1 << 20)?;
        bounded("qk_nope_head_dim", raw.qk_nope_head_dim, 1, 1 << 16)?;
        bounded("qk_rope_head_dim", raw.qk_rope_head_dim, 2, 1 << 16)?;
        bounded("v_head_dim", raw.v_head_dim, 1, 1 << 16)?;
        bounded("n_shared_experts", raw.n_shared_experts, 1, 64)?;
        bounded("index_topk", raw.index_topk, 1, 1 << 20)?;
        bounded("index_n_heads", raw.index_n_heads, 1, 1_024)?;
        bounded("index_head_dim", raw.index_head_dim, 2, 1 << 16)?;
        if raw.num_experts_per_tok > raw.n_routed_experts {
            return Err(invalid(
                "num_experts_per_tok",
                raw.num_experts_per_tok,
                "cannot exceed n_routed_experts",
            ));
        }
        if raw.n_group != 1 || raw.topk_group != 1 {
            return Err(invalid(
                "n_group/topk_group",
                format!("{}/{}", raw.n_group, raw.topk_group),
                "DiskMule's GLM-5.2 contract requires both values to be 1",
            ));
        }
        if !raw.qk_rope_head_dim.is_multiple_of(2) || !raw.index_head_dim.is_multiple_of(2) {
            return Err(invalid(
                "rotary dimensions",
                format!("{}/{}", raw.qk_rope_head_dim, raw.index_head_dim),
                "interleaved RoPE dimensions must be even",
            ));
        }
        if raw.index_head_dim < raw.qk_rope_head_dim {
            return Err(invalid(
                "index_head_dim",
                raw.index_head_dim,
                "must be at least qk_rope_head_dim",
            ));
        }
        for (field, value) in [
            ("rms_norm_eps", raw.rms_norm_eps),
            ("rope_theta", raw.rope_parameters.rope_theta),
            ("routed_scaling_factor", raw.routed_scaling_factor),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(invalid(field, value, "must be finite and positive"));
            }
        }

        let indexer_kinds = if raw.indexer_types.is_empty() {
            (0..raw.num_hidden_layers)
                .map(|layer| {
                    let shifted =
                        layer.saturating_sub(raw.index_skip_topk_offset.saturating_sub(1));
                    if shifted.is_multiple_of(raw.index_topk_freq.max(1)) {
                        IndexerKind::Full
                    } else {
                        IndexerKind::Shared
                    }
                })
                .collect()
        } else {
            if raw.indexer_types.len() != raw.num_hidden_layers {
                return Err(invalid(
                    "indexer_types",
                    raw.indexer_types.len(),
                    "must contain one entry per layer",
                ));
            }
            raw.indexer_types
                .into_iter()
                .map(|kind| match kind.as_str() {
                    "full" => Ok(IndexerKind::Full),
                    "shared" => Ok(IndexerKind::Shared),
                    _ => Err(invalid(
                        "indexer_types",
                        kind,
                        "entries must be full or shared",
                    )),
                })
                .collect::<Result<Vec<_>, _>>()?
        };
        let mut stop_token_ids = Vec::new();
        append_token_ids(&mut stop_token_ids, raw.eos_token_id);
        append_token_ids(&mut stop_token_ids, generation.eos_token_id);
        let mut seen = HashSet::new();
        stop_token_ids.retain(|token| seen.insert(*token));
        if stop_token_ids
            .iter()
            .any(|token| *token as usize >= raw.vocab_size)
        {
            return Err(invalid(
                "eos_token_id",
                format!("{stop_token_ids:?}"),
                "all stop IDs must be inside the vocabulary",
            ));
        }

        Ok(Self {
            vocabulary_size: raw.vocab_size,
            hidden_size: raw.hidden_size,
            layer_count: raw.num_hidden_layers,
            attention_heads: raw.num_attention_heads,
            routed_experts: raw.n_routed_experts,
            experts_per_token: raw.num_experts_per_tok,
            routed_intermediate_size: raw.moe_intermediate_size,
            dense_intermediate_size: raw.intermediate_size,
            first_sparse_layer: raw.first_k_dense_replace,
            query_lora_rank: raw.q_lora_rank,
            kv_lora_rank: raw.kv_lora_rank,
            query_key_nope_size: raw.qk_nope_head_dim,
            query_key_rope_size: raw.qk_rope_head_dim,
            value_head_size: raw.v_head_dim,
            shared_experts: raw.n_shared_experts,
            expert_groups: raw.n_group,
            selected_groups: raw.topk_group,
            normalize_topk: raw.norm_topk_prob,
            rms_epsilon: raw.rms_norm_eps,
            rope_theta: raw.rope_parameters.rope_theta,
            routed_scale: raw.routed_scaling_factor,
            index_topk: raw.index_topk,
            index_heads: raw.index_n_heads,
            index_head_size: raw.index_head_dim,
            indexer_kinds,
            stop_token_ids,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Glm52Weights {
    pub token_embedding: usize,
    pub output: usize,
    pub output_norm: usize,
    pub layers: Vec<Glm52LayerWeights>,
    pub dsa_enabled: bool,
}

#[derive(Debug, Clone)]
pub struct Glm52LayerWeights {
    pub input_norm: usize,
    pub post_attention_norm: usize,
    pub query_a: usize,
    pub query_a_norm: usize,
    pub query_b: usize,
    pub kv_a: usize,
    pub kv_a_norm: usize,
    pub kv_b: usize,
    pub attention_output: usize,
    pub feed_forward: Glm52FeedForwardWeights,
    pub indexer: Option<Glm52IndexerWeights>,
}

#[derive(Debug, Clone)]
pub enum Glm52FeedForwardWeights {
    Dense {
        gate: usize,
        up: usize,
        down: usize,
    },
    Sparse {
        router: usize,
        correction_bias: usize,
        shared_gate: usize,
        shared_up: usize,
        shared_down: usize,
        experts: Vec<Glm52ExpertWeights>,
    },
}

#[derive(Debug, Clone)]
pub struct Glm52ExpertWeights {
    pub gate: usize,
    pub up: usize,
    pub down: usize,
}

#[derive(Debug, Clone)]
pub struct Glm52IndexerWeights {
    pub query: usize,
    pub key: usize,
    pub projection: usize,
    pub key_norm_weight: usize,
    pub key_norm_bias: usize,
}

impl Glm52Weights {
    pub fn from_index(index: &SafeTensorIndex, config: &Glm52Config) -> Result<Self, Glm52Error> {
        let hidden = config.hidden_size as u64;
        let vocab = config.vocabulary_size as u64;
        let token_embedding =
            required_matrix(index, "model.embed_tokens.weight", &[vocab, hidden])?;
        let output = required_matrix(index, "lm_head.weight", &[vocab, hidden])?;
        let output_norm = required_vector(index, "model.norm.weight", hidden)?;
        let dsa_enabled = indexer_availability(index, config)?;
        let mut layers = Vec::with_capacity(config.layer_count);
        for layer in 0..config.layer_count {
            let prefix = format!("model.layers.{layer}");
            let name = |suffix: &str| format!("{prefix}.{suffix}");
            let query_head = (config.query_key_nope_size + config.query_key_rope_size) as u64;
            let query_width = config.attention_heads as u64 * query_head;
            let kv_output = (config.kv_lora_rank + config.query_key_rope_size) as u64;
            let kv_b_output = config.attention_heads as u64
                * (config.query_key_nope_size + config.value_head_size) as u64;
            let input_norm = required_vector(index, &name("input_layernorm.weight"), hidden)?;
            let post_attention_norm =
                required_vector(index, &name("post_attention_layernorm.weight"), hidden)?;
            let query_a = required_matrix(
                index,
                &name("self_attn.q_a_proj.weight"),
                &[config.query_lora_rank as u64, hidden],
            )?;
            let query_a_norm = required_vector(
                index,
                &name("self_attn.q_a_layernorm.weight"),
                config.query_lora_rank as u64,
            )?;
            let query_b = required_matrix(
                index,
                &name("self_attn.q_b_proj.weight"),
                &[query_width, config.query_lora_rank as u64],
            )?;
            let kv_a = required_matrix(
                index,
                &name("self_attn.kv_a_proj_with_mqa.weight"),
                &[kv_output, hidden],
            )?;
            let kv_a_norm = required_vector(
                index,
                &name("self_attn.kv_a_layernorm.weight"),
                config.kv_lora_rank as u64,
            )?;
            let kv_b = required_matrix(
                index,
                &name("self_attn.kv_b_proj.weight"),
                &[kv_b_output, config.kv_lora_rank as u64],
            )?;
            let attention_output = required_matrix(
                index,
                &name("self_attn.o_proj.weight"),
                &[
                    hidden,
                    config.attention_heads as u64 * config.value_head_size as u64,
                ],
            )?;
            let feed_forward = if layer < config.first_sparse_layer {
                Glm52FeedForwardWeights::Dense {
                    gate: required_matrix(
                        index,
                        &name("mlp.gate_proj.weight"),
                        &[config.dense_intermediate_size as u64, hidden],
                    )?,
                    up: required_matrix(
                        index,
                        &name("mlp.up_proj.weight"),
                        &[config.dense_intermediate_size as u64, hidden],
                    )?,
                    down: required_matrix(
                        index,
                        &name("mlp.down_proj.weight"),
                        &[hidden, config.dense_intermediate_size as u64],
                    )?,
                }
            } else {
                let mut experts = Vec::with_capacity(config.routed_experts);
                for expert in 0..config.routed_experts {
                    let expert_name = |projection: &str| {
                        name(&format!("mlp.experts.{expert}.{projection}_proj.weight"))
                    };
                    experts.push(Glm52ExpertWeights {
                        gate: required_matrix(
                            index,
                            &expert_name("gate"),
                            &[config.routed_intermediate_size as u64, hidden],
                        )?,
                        up: required_matrix(
                            index,
                            &expert_name("up"),
                            &[config.routed_intermediate_size as u64, hidden],
                        )?,
                        down: required_matrix(
                            index,
                            &expert_name("down"),
                            &[hidden, config.routed_intermediate_size as u64],
                        )?,
                    });
                }
                let shared_width = (config.shared_experts * config.routed_intermediate_size) as u64;
                Glm52FeedForwardWeights::Sparse {
                    router: required_vector_or_matrix(
                        index,
                        &name("mlp.gate.weight"),
                        &[config.routed_experts as u64, hidden],
                    )?,
                    correction_bias: required_vector(
                        index,
                        &name("mlp.gate.e_score_correction_bias"),
                        config.routed_experts as u64,
                    )?,
                    shared_gate: required_matrix(
                        index,
                        &name("mlp.shared_experts.gate_proj.weight"),
                        &[shared_width, hidden],
                    )?,
                    shared_up: required_matrix(
                        index,
                        &name("mlp.shared_experts.up_proj.weight"),
                        &[shared_width, hidden],
                    )?,
                    shared_down: required_matrix(
                        index,
                        &name("mlp.shared_experts.down_proj.weight"),
                        &[hidden, shared_width],
                    )?,
                    experts,
                }
            };
            let indexer = (dsa_enabled && config.indexer_kinds[layer] == IndexerKind::Full)
                .then(|| {
                    Ok::<_, Glm52Error>(Glm52IndexerWeights {
                        query: required_matrix(
                            index,
                            &name("self_attn.indexer.wq_b.weight"),
                            &[
                                (config.index_heads * config.index_head_size) as u64,
                                config.query_lora_rank as u64,
                            ],
                        )?,
                        key: required_matrix(
                            index,
                            &name("self_attn.indexer.wk.weight"),
                            &[config.index_head_size as u64, hidden],
                        )?,
                        projection: required_matrix(
                            index,
                            &name("self_attn.indexer.weights_proj.weight"),
                            &[config.index_heads as u64, hidden],
                        )?,
                        key_norm_weight: required_vector(
                            index,
                            &name("self_attn.indexer.k_norm.weight"),
                            config.index_head_size as u64,
                        )?,
                        key_norm_bias: required_vector(
                            index,
                            &name("self_attn.indexer.k_norm.bias"),
                            config.index_head_size as u64,
                        )?,
                    })
                })
                .transpose()?;
            layers.push(Glm52LayerWeights {
                input_norm,
                post_attention_norm,
                query_a,
                query_a_norm,
                query_b,
                kv_a,
                kv_a_norm,
                kv_b,
                attention_output,
                feed_forward,
                indexer,
            });
        }
        Ok(Self {
            token_embedding,
            output,
            output_norm,
            layers,
            dsa_enabled,
        })
    }
}

fn indexer_availability(index: &SafeTensorIndex, config: &Glm52Config) -> Result<bool, Glm52Error> {
    let mut required = Vec::new();
    for (layer, kind) in config.indexer_kinds.iter().enumerate() {
        if *kind != IndexerKind::Full {
            continue;
        }
        let prefix = format!("model.layers.{layer}.self_attn.indexer");
        for suffix in [
            "wq_b.weight",
            "wk.weight",
            "weights_proj.weight",
            "k_norm.weight",
            "k_norm.bias",
        ] {
            required.push(format!("{prefix}.{suffix}"));
        }
    }
    let present = required
        .iter()
        .filter(|name| index.tensor(name).is_some())
        .count();
    if present == 0 {
        return Ok(false);
    }
    if let Some(missing) = required
        .into_iter()
        .find(|name| index.tensor(name).is_none())
    {
        return Err(Glm52Error::MissingTensor(missing));
    }
    Ok(true)
}

fn required_vector(index: &SafeTensorIndex, name: &str, length: u64) -> Result<usize, Glm52Error> {
    required_tensor(index, name, &[length], false)
}

fn required_vector_or_matrix(
    index: &SafeTensorIndex,
    name: &str,
    shape: &[u64],
) -> Result<usize, Glm52Error> {
    required_tensor(index, name, shape, false)
}

fn required_matrix(
    index: &SafeTensorIndex,
    name: &str,
    shape: &[u64],
) -> Result<usize, Glm52Error> {
    required_tensor(index, name, shape, true)
}

fn required_tensor(
    index: &SafeTensorIndex,
    name: &str,
    expected: &[u64],
    allow_quantized: bool,
) -> Result<usize, Glm52Error> {
    let tensor_index = index
        .tensors()
        .iter()
        .position(|tensor| tensor.name == name)
        .ok_or_else(|| Glm52Error::MissingTensor(name.to_owned()))?;
    let tensor = &index.tensors()[tensor_index];
    if tensor.dtype == SafeDtype::U8 && allow_quantized && expected.len() == 2 {
        quant::resolve_quantization(index, tensor, expected[0], expected[1])?;
        return Ok(tensor_index);
    }
    if tensor.shape != expected {
        return Err(Glm52Error::TensorShape {
            tensor: name.to_owned(),
            expected: expected.to_vec(),
            actual: tensor.shape.clone(),
        });
    }
    match tensor.dtype {
        SafeDtype::F32 | SafeDtype::F16 | SafeDtype::Bf16 => {}
        SafeDtype::F8E4M3 if allow_quantized => validate_fp8_scale(index, tensor)?,
        dtype => {
            return Err(Glm52Error::TensorDtype {
                tensor: name.to_owned(),
                dtype,
            });
        }
    }
    Ok(tensor_index)
}

fn validate_fp8_scale(index: &SafeTensorIndex, tensor: &SafeTensorInfo) -> Result<(), Glm52Error> {
    let scale = format!("{}_scale_inv", tensor.name);
    let expected = vec![
        tensor.shape[0].div_ceil(FP8_BLOCK),
        tensor.shape[1].div_ceil(FP8_BLOCK),
    ];
    if !index
        .tensor(&scale)
        .is_some_and(|scale| scale.dtype == SafeDtype::F32 && scale.shape == expected)
    {
        return Err(Glm52Error::MissingFp8Scale {
            tensor: tensor.name.clone(),
            scale,
            expected,
        });
    }
    Ok(())
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, Glm52Error> {
    let metadata = fs::metadata(path).map_err(|source| Glm52Error::ReadConfig {
        path: path.to_path_buf(),
        source,
    })?;
    if metadata.len() > MAX_CONFIG_BYTES {
        return Err(Glm52Error::ConfigTooLarge {
            path: path.to_path_buf(),
            size: metadata.len(),
            maximum: MAX_CONFIG_BYTES,
        });
    }
    let bytes = fs::read(path).map_err(|source| Glm52Error::ReadConfig {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_slice(&bytes).map_err(|error| Glm52Error::InvalidConfig {
        path: path.to_path_buf(),
        message: error.to_string(),
    })
}

fn append_token_ids(target: &mut Vec<u32>, ids: Option<TokenIds>) {
    match ids {
        Some(TokenIds::One(token)) => target.push(token),
        Some(TokenIds::Many(tokens)) => target.extend(tokens),
        None => {}
    }
}

fn bounded(
    field: &'static str,
    value: usize,
    minimum: usize,
    maximum: usize,
) -> Result<(), Glm52Error> {
    if (minimum..=maximum).contains(&value) {
        Ok(())
    } else {
        Err(invalid(field, value, "outside the supported bounded range"))
    }
}

fn invalid(field: &'static str, value: impl ToString, reason: &'static str) -> Glm52Error {
    Glm52Error::InvalidField {
        field,
        value: value.to_string(),
        reason,
    }
}

const fn default_epsilon() -> f32 {
    1e-5
}

const fn default_rope_theta() -> f32 {
    10_000.0
}

const fn default_routed_scale() -> f32 {
    1.0
}

const fn default_index_frequency() -> usize {
    1
}

const fn default_index_offset() -> usize {
    2
}

#[cfg(test)]
mod tests {
    use std::{fs, io::Write, path::Path};

    use serde_json::{Map, Value, json};
    use tempfile::TempDir;

    use crate::safetensors::SafeTensorIndex;

    use super::{Glm52Config, Glm52Error, Glm52FeedForwardWeights, Glm52Weights, IndexerKind};

    #[test]
    fn validates_tiny_config_stop_union_and_indexer_schedule() {
        let temp = TempDir::new().unwrap();
        write_config(temp.path());
        fs::write(
            temp.path().join("generation_config.json"),
            r#"{"eos_token_id":[3,5]}"#,
        )
        .unwrap();
        let config = Glm52Config::from_directory(temp.path()).unwrap();
        assert_eq!(config.stop_token_ids, [2, 3, 5]);
        assert_eq!(
            config.indexer_kinds,
            [IndexerKind::Full, IndexerKind::Shared]
        );
        assert_eq!(config.kv_lora_rank + config.query_key_rope_size, 6);
    }

    #[test]
    fn maps_dense_sparse_expert_and_dsa_tensors() {
        let temp = TempDir::new().unwrap();
        write_config(temp.path());
        let config = Glm52Config::from_directory(temp.path()).unwrap();
        let mut tensors = Vec::new();
        add(&mut tensors, "model.embed_tokens.weight", &[16, 8]);
        add(&mut tensors, "lm_head.weight", &[16, 8]);
        add(&mut tensors, "model.norm.weight", &[8]);
        for layer in 0..2 {
            let p = format!("model.layers.{layer}");
            add(&mut tensors, &format!("{p}.input_layernorm.weight"), &[8]);
            add(
                &mut tensors,
                &format!("{p}.post_attention_layernorm.weight"),
                &[8],
            );
            add(
                &mut tensors,
                &format!("{p}.self_attn.q_a_proj.weight"),
                &[4, 8],
            );
            add(
                &mut tensors,
                &format!("{p}.self_attn.q_a_layernorm.weight"),
                &[4],
            );
            add(
                &mut tensors,
                &format!("{p}.self_attn.q_b_proj.weight"),
                &[8, 4],
            );
            add(
                &mut tensors,
                &format!("{p}.self_attn.kv_a_proj_with_mqa.weight"),
                &[6, 8],
            );
            add(
                &mut tensors,
                &format!("{p}.self_attn.kv_a_layernorm.weight"),
                &[4],
            );
            add(
                &mut tensors,
                &format!("{p}.self_attn.kv_b_proj.weight"),
                &[8, 4],
            );
            add(
                &mut tensors,
                &format!("{p}.self_attn.o_proj.weight"),
                &[8, 4],
            );
            if layer == 0 {
                add(&mut tensors, &format!("{p}.mlp.gate_proj.weight"), &[6, 8]);
                add(&mut tensors, &format!("{p}.mlp.up_proj.weight"), &[6, 8]);
                add(&mut tensors, &format!("{p}.mlp.down_proj.weight"), &[8, 6]);
                add(
                    &mut tensors,
                    &format!("{p}.self_attn.indexer.wq_b.weight"),
                    &[4, 4],
                );
                add(
                    &mut tensors,
                    &format!("{p}.self_attn.indexer.wk.weight"),
                    &[2, 8],
                );
                add(
                    &mut tensors,
                    &format!("{p}.self_attn.indexer.weights_proj.weight"),
                    &[2, 8],
                );
                add(
                    &mut tensors,
                    &format!("{p}.self_attn.indexer.k_norm.weight"),
                    &[2],
                );
                add(
                    &mut tensors,
                    &format!("{p}.self_attn.indexer.k_norm.bias"),
                    &[2],
                );
            } else {
                add(&mut tensors, &format!("{p}.mlp.gate.weight"), &[2, 8]);
                add(
                    &mut tensors,
                    &format!("{p}.mlp.gate.e_score_correction_bias"),
                    &[2],
                );
                add(
                    &mut tensors,
                    &format!("{p}.mlp.shared_experts.gate_proj.weight"),
                    &[3, 8],
                );
                add(
                    &mut tensors,
                    &format!("{p}.mlp.shared_experts.up_proj.weight"),
                    &[3, 8],
                );
                add(
                    &mut tensors,
                    &format!("{p}.mlp.shared_experts.down_proj.weight"),
                    &[8, 3],
                );
                for expert in 0..2 {
                    add(
                        &mut tensors,
                        &format!("{p}.mlp.experts.{expert}.gate_proj.weight"),
                        &[3, 8],
                    );
                    add(
                        &mut tensors,
                        &format!("{p}.mlp.experts.{expert}.up_proj.weight"),
                        &[3, 8],
                    );
                    add(
                        &mut tensors,
                        &format!("{p}.mlp.experts.{expert}.down_proj.weight"),
                        &[8, 3],
                    );
                }
            }
        }
        write_safetensors(temp.path(), &tensors);
        let index = SafeTensorIndex::open(temp.path()).unwrap();
        let weights = Glm52Weights::from_index(&index, &config).unwrap();
        assert_eq!(
            index.tensors()[weights.token_embedding].dtype,
            crate::safetensors::SafeDtype::F8E4M3
        );
        assert_eq!(weights.layers.len(), 2);
        assert!(weights.dsa_enabled);
        assert!(weights.layers[0].indexer.is_some());
        assert!(weights.layers[1].indexer.is_none());
        assert!(matches!(
            &weights.layers[1].feed_forward,
            Glm52FeedForwardWeights::Sparse { experts, .. } if experts.len() == 2
        ));
    }

    #[test]
    fn rejects_invalid_relationships_and_tensor_shapes() {
        let temp = TempDir::new().unwrap();
        write_config(temp.path());
        let mut value: Value =
            serde_json::from_slice(&fs::read(temp.path().join("config.json")).unwrap()).unwrap();
        value["num_experts_per_tok"] = json!(3);
        fs::write(
            temp.path().join("config.json"),
            serde_json::to_vec(&value).unwrap(),
        )
        .unwrap();
        assert!(matches!(
            Glm52Config::from_directory(temp.path()),
            Err(Glm52Error::InvalidField { .. })
        ));
    }

    fn write_config(directory: &Path) {
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
                "index_topk": 4,
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
    }

    fn add(tensors: &mut Vec<(String, Vec<u64>)>, name: &str, shape: &[u64]) {
        tensors.push((name.to_owned(), shape.to_vec()));
    }

    fn write_safetensors(directory: &Path, tensors: &[(String, Vec<u64>)]) {
        let mut header = Map::new();
        let mut offset = 0_u64;
        for (name, shape) in tensors {
            let fp8 = name == "model.embed_tokens.weight";
            let length = shape.iter().product::<u64>() * if fp8 { 1 } else { 4 };
            header.insert(
                name.clone(),
                json!({
                    "dtype": if fp8 { "F8_E4M3" } else { "F32" },
                    "shape":shape,
                    "data_offsets":[offset, offset+length]
                }),
            );
            offset += length;
            if fp8 {
                header.insert(
                    format!("{name}_scale_inv"),
                    json!({"dtype":"F32", "shape":[1,1], "data_offsets":[offset, offset+4]}),
                );
                offset += 4;
            }
        }
        let header = serde_json::to_vec(&header).unwrap();
        let mut file = fs::File::create(directory.join("model.safetensors")).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.write_all(&vec![0_u8; offset as usize]).unwrap();
    }
}
