//! Qwen3.5/Qwen3.8 hybrid recurrent-attention architecture contract.

pub mod cpu;
#[cfg(target_os = "macos")]
pub mod metal;
pub mod tokenizer;

use crate::gguf::{GgufFile, MetadataValue, TensorType};

pub const RUNTIME_ARRAY_KEYS: [&str; 3] = [
    "tokenizer.ggml.merges",
    "tokenizer.ggml.token_type",
    "tokenizer.ggml.tokens",
];

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum Qwen35Error {
    #[error("missing Qwen3.5 GGUF metadata {0:?}")]
    MissingMetadata(&'static str),

    #[error("Qwen3.5 GGUF metadata {key:?} has the wrong type; expected {expected}")]
    MetadataType {
        key: &'static str,
        expected: &'static str,
    },

    #[error("invalid Qwen3.5 configuration: {0}")]
    InvalidConfiguration(String),

    #[error("missing Qwen3.5 tensor {0:?}")]
    MissingTensor(String),

    #[error("Qwen3.5 tensor {tensor:?} has shape {actual:?}; expected {expected:?}")]
    TensorShape {
        tensor: String,
        expected: Vec<u64>,
        actual: Vec<u64>,
    },

    #[error("Qwen3.5 tensor {tensor:?} uses unsupported runtime type {kind:?}")]
    TensorType { tensor: String, kind: TensorType },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35LayerKind {
    Recurrent,
    FullAttention,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35Config {
    pub context_length: usize,
    pub hidden_size: usize,
    pub layer_count: usize,
    pub nextn_layers: usize,
    pub attention_heads: usize,
    pub kv_heads: usize,
    pub attention_head_size: usize,
    pub rope_dimensions: usize,
    pub rope_frequency_base: f32,
    pub rms_epsilon: f32,
    pub feed_forward_size: usize,
    pub full_attention_interval: usize,
    pub recurrent_inner_size: usize,
    pub recurrent_state_size: usize,
    pub recurrent_value_heads: usize,
    pub recurrent_key_heads: usize,
    pub convolution_kernel: usize,
}

impl Qwen35Config {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, Qwen35Error> {
        let architecture = required_string(gguf, "general.architecture")?;
        if architecture != "qwen35" {
            return Err(Qwen35Error::InvalidConfiguration(format!(
                "general.architecture is {architecture:?}; expected \"qwen35\""
            )));
        }
        let total_layers = usize_field(gguf, "qwen35.block_count")?;
        let nextn_layers = optional_usize(gguf, "qwen35.nextn_predict_layers")?.unwrap_or(0);
        let layer_count = total_layers.checked_sub(nextn_layers).ok_or_else(|| {
            Qwen35Error::InvalidConfiguration("nextn_predict_layers exceeds block_count".to_owned())
        })?;
        let key_size = usize_field(gguf, "qwen35.attention.key_length")?;
        let value_size = usize_field(gguf, "qwen35.attention.value_length")?;
        if key_size != value_size {
            return Err(Qwen35Error::InvalidConfiguration(format!(
                "attention key length {key_size} differs from value length {value_size}"
            )));
        }
        let config = Self {
            context_length: usize_field(gguf, "qwen35.context_length")?,
            hidden_size: usize_field(gguf, "qwen35.embedding_length")?,
            layer_count,
            nextn_layers,
            attention_heads: usize_field(gguf, "qwen35.attention.head_count")?,
            kv_heads: usize_field(gguf, "qwen35.attention.head_count_kv")?,
            attention_head_size: key_size,
            rope_dimensions: usize_field(gguf, "qwen35.rope.dimension_count")?,
            rope_frequency_base: f32_field(gguf, "qwen35.rope.freq_base")?,
            rms_epsilon: f32_field(gguf, "qwen35.attention.layer_norm_rms_epsilon")?,
            feed_forward_size: usize_field(gguf, "qwen35.feed_forward_length")?,
            full_attention_interval: usize_field(gguf, "qwen35.full_attention_interval")?,
            recurrent_inner_size: usize_field(gguf, "qwen35.ssm.inner_size")?,
            recurrent_state_size: usize_field(gguf, "qwen35.ssm.state_size")?,
            recurrent_value_heads: usize_field(gguf, "qwen35.ssm.time_step_rank")?,
            recurrent_key_heads: usize_field(gguf, "qwen35.ssm.group_count")?,
            convolution_kernel: usize_field(gguf, "qwen35.ssm.conv_kernel")?,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), Qwen35Error> {
        if self.context_length == 0
            || self.hidden_size == 0
            || self.layer_count == 0
            || self.attention_heads == 0
            || self.kv_heads == 0
            || self.attention_head_size == 0
            || self.feed_forward_size == 0
            || self.full_attention_interval == 0
            || self.recurrent_inner_size == 0
            || self.recurrent_state_size == 0
            || self.recurrent_value_heads == 0
            || self.recurrent_key_heads == 0
            || self.convolution_kernel == 0
        {
            return Err(Qwen35Error::InvalidConfiguration(
                "all runtime dimensions must be positive".to_owned(),
            ));
        }
        if !self.attention_heads.is_multiple_of(self.kv_heads) {
            return Err(Qwen35Error::InvalidConfiguration(
                "attention head count must be divisible by KV head count".to_owned(),
            ));
        }
        if self.rope_dimensions > self.attention_head_size
            || !self.rope_dimensions.is_multiple_of(2)
        {
            return Err(Qwen35Error::InvalidConfiguration(
                "RoPE dimensions must be even and no larger than the attention head".to_owned(),
            ));
        }
        if !self
            .recurrent_inner_size
            .is_multiple_of(self.recurrent_value_heads)
            || !self
                .recurrent_value_heads
                .is_multiple_of(self.recurrent_key_heads)
        {
            return Err(Qwen35Error::InvalidConfiguration(
                "recurrent inner/value/key head dimensions are incompatible".to_owned(),
            ));
        }
        if self.recurrent_state_size != self.recurrent_value_head_size() {
            return Err(Qwen35Error::InvalidConfiguration(format!(
                "recurrent state size {} differs from value head size {}",
                self.recurrent_state_size,
                self.recurrent_value_head_size()
            )));
        }
        if !self.rms_epsilon.is_finite()
            || self.rms_epsilon <= 0.0
            || !self.rope_frequency_base.is_finite()
            || self.rope_frequency_base <= 0.0
        {
            return Err(Qwen35Error::InvalidConfiguration(
                "RMS epsilon and RoPE frequency base must be finite and positive".to_owned(),
            ));
        }
        Ok(())
    }

    pub fn layer_kind(&self, layer: usize) -> Qwen35LayerKind {
        if (layer + 1).is_multiple_of(self.full_attention_interval) {
            Qwen35LayerKind::FullAttention
        } else {
            Qwen35LayerKind::Recurrent
        }
    }

    pub fn recurrent_value_head_size(&self) -> usize {
        self.recurrent_inner_size / self.recurrent_value_heads
    }

    pub fn recurrent_projection_size(&self) -> usize {
        2 * self.recurrent_state_size * self.recurrent_key_heads + self.recurrent_inner_size
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35Weights {
    pub token_embedding: usize,
    pub output_norm: usize,
    pub output: usize,
    pub layers: Vec<Qwen35LayerWeights>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35LayerWeights {
    pub attention_norm: usize,
    pub attention: Qwen35AttentionWeights,
    pub post_attention_norm: usize,
    pub feed_forward_gate: usize,
    pub feed_forward_up: usize,
    pub feed_forward_down: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Qwen35AttentionWeights {
    Recurrent {
        qkv: usize,
        gate: usize,
        convolution: usize,
        alpha: usize,
        beta: usize,
        time_bias: usize,
        decay: usize,
        norm: usize,
        output: usize,
    },
    Full {
        query_gate: usize,
        key: usize,
        value: usize,
        query_norm: usize,
        key_norm: usize,
        output: usize,
    },
}

impl Qwen35Weights {
    pub fn from_gguf(gguf: &GgufFile, config: &Qwen35Config) -> Result<Self, Qwen35Error> {
        let hidden = config.hidden_size as u64;
        let vocabulary = tokenizer_vocabulary(gguf)? as u64;
        let token_embedding = matrix(gguf, "token_embd.weight", &[hidden, vocabulary])?;
        let output_norm = vector(gguf, "output_norm.weight", hidden)?;
        let output = matrix(gguf, "output.weight", &[hidden, vocabulary])?;
        let attention_width = (config.attention_heads * config.attention_head_size) as u64;
        let kv_width = (config.kv_heads * config.attention_head_size) as u64;
        let recurrent_projection = config.recurrent_projection_size() as u64;
        let recurrent_inner = config.recurrent_inner_size as u64;
        let recurrent_heads = config.recurrent_value_heads as u64;
        let recurrent_head_size = config.recurrent_value_head_size() as u64;
        let convolution_channels = recurrent_projection;
        let feed_forward = config.feed_forward_size as u64;
        let mut layers = Vec::with_capacity(config.layer_count);
        for layer in 0..config.layer_count {
            let prefix = format!("blk.{layer}");
            let name = |suffix: &str| format!("{prefix}.{suffix}");
            let attention = match config.layer_kind(layer) {
                Qwen35LayerKind::Recurrent => Qwen35AttentionWeights::Recurrent {
                    qkv: matrix(
                        gguf,
                        &name("attn_qkv.weight"),
                        &[hidden, recurrent_projection],
                    )?,
                    gate: matrix(gguf, &name("attn_gate.weight"), &[hidden, recurrent_inner])?,
                    convolution: matrix(
                        gguf,
                        &name("ssm_conv1d.weight"),
                        &[config.convolution_kernel as u64, convolution_channels],
                    )?,
                    alpha: matrix(gguf, &name("ssm_alpha.weight"), &[hidden, recurrent_heads])?,
                    beta: matrix(gguf, &name("ssm_beta.weight"), &[hidden, recurrent_heads])?,
                    time_bias: vector(gguf, &name("ssm_dt.bias"), recurrent_heads)?,
                    decay: vector(gguf, &name("ssm_a"), recurrent_heads)?,
                    norm: vector(gguf, &name("ssm_norm.weight"), recurrent_head_size)?,
                    output: matrix(gguf, &name("ssm_out.weight"), &[recurrent_inner, hidden])?,
                },
                Qwen35LayerKind::FullAttention => Qwen35AttentionWeights::Full {
                    query_gate: matrix(
                        gguf,
                        &name("attn_q.weight"),
                        &[hidden, attention_width * 2],
                    )?,
                    key: matrix(gguf, &name("attn_k.weight"), &[hidden, kv_width])?,
                    value: matrix(gguf, &name("attn_v.weight"), &[hidden, kv_width])?,
                    query_norm: vector(
                        gguf,
                        &name("attn_q_norm.weight"),
                        config.attention_head_size as u64,
                    )?,
                    key_norm: vector(
                        gguf,
                        &name("attn_k_norm.weight"),
                        config.attention_head_size as u64,
                    )?,
                    output: matrix(
                        gguf,
                        &name("attn_output.weight"),
                        &[attention_width, hidden],
                    )?,
                },
            };
            layers.push(Qwen35LayerWeights {
                attention_norm: vector(gguf, &name("attn_norm.weight"), hidden)?,
                attention,
                post_attention_norm: vector(gguf, &name("post_attention_norm.weight"), hidden)?,
                feed_forward_gate: matrix(gguf, &name("ffn_gate.weight"), &[hidden, feed_forward])?,
                feed_forward_up: matrix(gguf, &name("ffn_up.weight"), &[hidden, feed_forward])?,
                feed_forward_down: matrix(gguf, &name("ffn_down.weight"), &[feed_forward, hidden])?,
            });
        }
        Ok(Self {
            token_embedding,
            output_norm,
            output,
            layers,
        })
    }
}

fn required_string<'a>(gguf: &'a GgufFile, key: &'static str) -> Result<&'a str, Qwen35Error> {
    match gguf.metadata.get(key) {
        Some(MetadataValue::String(value)) => Ok(value),
        Some(_) => Err(Qwen35Error::MetadataType {
            key,
            expected: "string",
        }),
        None => Err(Qwen35Error::MissingMetadata(key)),
    }
}

fn usize_field(gguf: &GgufFile, key: &'static str) -> Result<usize, Qwen35Error> {
    let value = match gguf.metadata.get(key) {
        Some(MetadataValue::U32(value)) => u64::from(*value),
        Some(MetadataValue::U64(value)) => *value,
        Some(_) => {
            return Err(Qwen35Error::MetadataType {
                key,
                expected: "unsigned integer",
            });
        }
        None => return Err(Qwen35Error::MissingMetadata(key)),
    };
    usize::try_from(value).map_err(|_| {
        Qwen35Error::InvalidConfiguration(format!("metadata {key:?} exceeds platform limits"))
    })
}

fn optional_usize(gguf: &GgufFile, key: &'static str) -> Result<Option<usize>, Qwen35Error> {
    if gguf.metadata.contains_key(key) {
        usize_field(gguf, key).map(Some)
    } else {
        Ok(None)
    }
}

fn f32_field(gguf: &GgufFile, key: &'static str) -> Result<f32, Qwen35Error> {
    match gguf.metadata.get(key) {
        Some(MetadataValue::F32(value)) => Ok(*value),
        Some(_) => Err(Qwen35Error::MetadataType {
            key,
            expected: "F32",
        }),
        None => Err(Qwen35Error::MissingMetadata(key)),
    }
}

fn tokenizer_vocabulary(gguf: &GgufFile) -> Result<usize, Qwen35Error> {
    match gguf.metadata.get("tokenizer.ggml.tokens") {
        Some(MetadataValue::ArrayValues(values)) => Ok(values.len()),
        Some(MetadataValue::Array { length, .. }) => usize::try_from(*length).map_err(|_| {
            Qwen35Error::InvalidConfiguration("tokenizer vocabulary is too large".to_owned())
        }),
        Some(_) => Err(Qwen35Error::MetadataType {
            key: "tokenizer.ggml.tokens",
            expected: "array",
        }),
        None => Err(Qwen35Error::MissingMetadata("tokenizer.ggml.tokens")),
    }
}

fn tensor_index(gguf: &GgufFile, name: &str) -> Option<usize> {
    gguf.tensors.iter().position(|tensor| tensor.name == name)
}

fn matrix(gguf: &GgufFile, name: &str, expected: &[u64]) -> Result<usize, Qwen35Error> {
    let index =
        tensor_index(gguf, name).ok_or_else(|| Qwen35Error::MissingTensor(name.to_owned()))?;
    validate_tensor(gguf, index, expected, false)?;
    Ok(index)
}

fn vector(gguf: &GgufFile, name: &str, length: u64) -> Result<usize, Qwen35Error> {
    let index =
        tensor_index(gguf, name).ok_or_else(|| Qwen35Error::MissingTensor(name.to_owned()))?;
    validate_tensor(gguf, index, &[length], true)?;
    Ok(index)
}

fn validate_tensor(
    gguf: &GgufFile,
    index: usize,
    expected: &[u64],
    vector: bool,
) -> Result<(), Qwen35Error> {
    let tensor = &gguf.tensors[index];
    if tensor.dimensions != expected {
        return Err(Qwen35Error::TensorShape {
            tensor: tensor.name.clone(),
            expected: expected.to_vec(),
            actual: tensor.dimensions.clone(),
        });
    }
    let supported = if vector {
        matches!(tensor.kind, TensorType::F32 | TensorType::F16)
    } else {
        matches!(
            tensor.kind,
            TensorType::F32
                | TensorType::F16
                | TensorType::Q4K
                | TensorType::Q5_0
                | TensorType::Q6K
                | TensorType::Q8_0
        )
    };
    if !supported {
        return Err(Qwen35Error::TensorType {
            tensor: tensor.name.clone(),
            kind: tensor.kind,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::gguf::{GgufFile, MetadataArray, MetadataValue, TensorInfo, TensorType};

    use super::{Qwen35AttentionWeights, Qwen35Config, Qwen35LayerKind, Qwen35Weights};

    #[test]
    fn validates_hybrid_layer_and_tensor_contract() {
        let gguf = tiny_gguf();
        let config = Qwen35Config::from_gguf(&gguf).unwrap();
        assert_eq!(config.layer_count, 2);
        assert_eq!(config.nextn_layers, 1);
        assert_eq!(config.layer_kind(0), Qwen35LayerKind::Recurrent);
        assert_eq!(config.layer_kind(1), Qwen35LayerKind::FullAttention);
        assert_eq!(config.recurrent_projection_size(), 8);
        let weights = Qwen35Weights::from_gguf(&gguf, &config).unwrap();
        assert!(matches!(
            weights.layers[0].attention,
            Qwen35AttentionWeights::Recurrent { .. }
        ));
        assert!(matches!(
            weights.layers[1].attention,
            Qwen35AttentionWeights::Full { .. }
        ));
    }

    #[test]
    fn rejects_missing_or_misshaped_hybrid_tensors() {
        let mut gguf = tiny_gguf();
        gguf.tensors
            .iter_mut()
            .find(|tensor| tensor.name == "blk.0.ssm_conv1d.weight")
            .unwrap()
            .dimensions = vec![8, 2];
        let config = Qwen35Config::from_gguf(&gguf).unwrap();
        assert!(Qwen35Weights::from_gguf(&gguf, &config).is_err());
    }

    fn tiny_gguf() -> GgufFile {
        let mut metadata = [
            (
                "general.architecture",
                MetadataValue::String("qwen35".to_owned()),
            ),
            ("qwen35.context_length", MetadataValue::U32(32)),
            ("qwen35.embedding_length", MetadataValue::U32(4)),
            ("qwen35.block_count", MetadataValue::U32(3)),
            ("qwen35.nextn_predict_layers", MetadataValue::U32(1)),
            ("qwen35.attention.head_count", MetadataValue::U32(2)),
            ("qwen35.attention.head_count_kv", MetadataValue::U32(1)),
            ("qwen35.attention.key_length", MetadataValue::U32(2)),
            ("qwen35.attention.value_length", MetadataValue::U32(2)),
            ("qwen35.rope.dimension_count", MetadataValue::U32(2)),
            ("qwen35.rope.freq_base", MetadataValue::F32(10_000.0)),
            (
                "qwen35.attention.layer_norm_rms_epsilon",
                MetadataValue::F32(1e-6),
            ),
            ("qwen35.feed_forward_length", MetadataValue::U32(3)),
            ("qwen35.full_attention_interval", MetadataValue::U32(2)),
            ("qwen35.ssm.inner_size", MetadataValue::U32(4)),
            ("qwen35.ssm.state_size", MetadataValue::U32(2)),
            ("qwen35.ssm.time_step_rank", MetadataValue::U32(2)),
            ("qwen35.ssm.group_count", MetadataValue::U32(1)),
            ("qwen35.ssm.conv_kernel", MetadataValue::U32(2)),
        ]
        .into_iter()
        .map(|(key, value)| (key.to_owned(), value))
        .collect::<BTreeMap<_, _>>();
        metadata.insert(
            "tokenizer.ggml.tokens".to_owned(),
            MetadataValue::ArrayValues(MetadataArray::String(
                (0..8).map(|token| format!("t{token}")).collect(),
            )),
        );
        let mut tensors = Vec::new();
        push(&mut tensors, "token_embd.weight", &[4, 8]);
        push(&mut tensors, "output_norm.weight", &[4]);
        push(&mut tensors, "output.weight", &[4, 8]);
        for layer in 0..2 {
            let prefix = format!("blk.{layer}");
            push(&mut tensors, &format!("{prefix}.attn_norm.weight"), &[4]);
            push(
                &mut tensors,
                &format!("{prefix}.post_attention_norm.weight"),
                &[4],
            );
            push(&mut tensors, &format!("{prefix}.ffn_gate.weight"), &[4, 3]);
            push(&mut tensors, &format!("{prefix}.ffn_up.weight"), &[4, 3]);
            push(&mut tensors, &format!("{prefix}.ffn_down.weight"), &[3, 4]);
        }
        for (name, shape) in [
            ("attn_qkv.weight", vec![4, 8]),
            ("attn_gate.weight", vec![4, 4]),
            ("ssm_conv1d.weight", vec![2, 8]),
            ("ssm_alpha.weight", vec![4, 2]),
            ("ssm_beta.weight", vec![4, 2]),
            ("ssm_dt.bias", vec![2]),
            ("ssm_a", vec![2]),
            ("ssm_norm.weight", vec![2]),
            ("ssm_out.weight", vec![4, 4]),
        ] {
            push(&mut tensors, &format!("blk.0.{name}"), &shape);
        }
        for (name, shape) in [
            ("attn_q.weight", vec![4, 8]),
            ("attn_k.weight", vec![4, 2]),
            ("attn_v.weight", vec![4, 2]),
            ("attn_q_norm.weight", vec![2]),
            ("attn_k_norm.weight", vec![2]),
            ("attn_output.weight", vec![4, 4]),
        ] {
            push(&mut tensors, &format!("blk.1.{name}"), &shape);
        }
        GgufFile {
            version: 3,
            alignment: 32,
            data_offset: 0,
            file_len: 0,
            metadata,
            tensors,
        }
    }

    fn push(tensors: &mut Vec<TensorInfo>, name: &str, shape: &[u64]) {
        tensors.push(TensorInfo {
            name: name.to_owned(),
            dimensions: shape.to_vec(),
            kind: TensorType::F32,
            offset: 0,
            absolute_offset: 0,
            byte_len: shape.iter().product::<u64>() * 4,
        });
    }
}
