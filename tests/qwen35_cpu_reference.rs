use std::{fs, path::Path};

use diskmule::{
    cpu::argmax,
    gemma4::cpu::CancellationFlag,
    qwen35::cpu::Qwen35CpuModel,
    runtime::{BackendSelection, GenerationEngine, GenerationOptions},
};
use tempfile::TempDir;

#[test]
fn mapped_hybrid_graph_generates_deterministically() {
    let fixture = TinyQwen::write();
    let model = Qwen35CpuModel::open(&fixture.path, 16).unwrap();
    let (logits, profile) = model
        .prompt_logits(&[0, 1], &CancellationFlag::default())
        .unwrap();
    assert_eq!(logits.len(), 8);
    assert!(logits.iter().all(|value| value.is_finite()));
    assert!(profile.mapped_bytes_touched > 0);
    assert!(profile.resident_kv_bytes > 0);

    let generated = model
        .generate_greedy(&[0, 1], 3, &CancellationFlag::default(), |_, _| {})
        .unwrap();
    assert_eq!(generated.token_ids, [4]);
    assert_eq!(generated.text, "e");
    assert!(generated.stopped);
    assert_eq!(generated.profile.prompt_tokens, 2);
    assert_eq!(generated.profile.generated_tokens, 1);
}

#[test]
fn shared_generation_engine_loads_and_runs_qwen35() {
    let fixture = TinyQwen::write();
    let engine = GenerationEngine::open(
        "tiny-qwen",
        &fixture.path,
        "qwen35",
        16,
        BackendSelection::Cpu,
    )
    .unwrap();
    let result = engine
        .generate_tokens(
            &[0, 1],
            GenerationOptions {
                maximum_new_tokens: 3,
                ..GenerationOptions::default()
            },
            &CancellationFlag::default(),
            |_, _| {},
        )
        .unwrap();

    assert_eq!(engine.name(), "tiny-qwen");
    assert_eq!(engine.backend(), BackendSelection::Cpu);
    assert_eq!(engine.capabilities().architecture, "qwen35");
    assert!(engine.capabilities().persistent_sessions);
    assert_eq!(result.token_ids, [4]);
    assert_eq!(result.text, "e");
}

#[test]
fn hybrid_session_reuses_extensions_and_rebuilds_divergent_prompts() {
    let fixture = TinyQwen::write();
    let model = Qwen35CpuModel::open(&fixture.path, 16).unwrap();
    let cancellation = CancellationFlag::default();
    let mut session = model.new_session();
    model
        .generate_session_with_selector(
            &mut session,
            &[0, 1],
            0,
            &cancellation,
            |logits| Ok(argmax(logits).unwrap() as u32),
            |_, _| {},
        )
        .unwrap();
    assert_eq!(session.cached_tokens(), [0, 1]);

    let reused = model
        .generate_session_with_selector(
            &mut session,
            &[0, 1],
            2,
            &cancellation,
            |logits| Ok(argmax(logits).unwrap() as u32),
            |_, _| {},
        )
        .unwrap();
    let fresh = model
        .generate_greedy(&[0, 1], 2, &cancellation, |_, _| {})
        .unwrap();
    assert_eq!(reused.token_ids, fresh.token_ids);
    assert_eq!(reused.profile.reused_prompt_tokens, 2);

    let rebuilt = model
        .generate_session_with_selector(
            &mut session,
            &[0, 2],
            1,
            &cancellation,
            |logits| Ok(argmax(logits).unwrap() as u32),
            |_, _| {},
        )
        .unwrap();
    let fresh = model
        .generate_greedy(&[0, 2], 1, &cancellation, |_, _| {})
        .unwrap();
    assert_eq!(rebuilt.token_ids, fresh.token_ids);
    assert_eq!(rebuilt.profile.reused_prompt_tokens, 0);

    let cancelled = CancellationFlag::default();
    cancelled.cancel();
    assert!(
        model
            .generate_session_with_selector(
                &mut session,
                &[0, 2],
                1,
                &cancelled,
                |logits| Ok(argmax(logits).unwrap() as u32),
                |_, _| {},
            )
            .unwrap_err()
            .is_cancelled()
    );
    assert!(session.cached_tokens().is_empty());
}

#[cfg(target_os = "macos")]
#[test]
fn metal_matrix_offload_matches_the_scalar_hybrid_graph() {
    let fixture = TinyQwen::write();
    let cpu = Qwen35CpuModel::open(&fixture.path, 16).unwrap();
    let metal = Qwen35CpuModel::open_metal(&fixture.path, 16).unwrap();
    assert!(metal.backend_label().starts_with("Metal ("));
    let cancellation = CancellationFlag::default();
    let (cpu_logits, _) = cpu.prompt_logits(&[0, 1, 2], &cancellation).unwrap();
    let (metal_logits, _) = metal.prompt_logits(&[0, 1, 2], &cancellation).unwrap();
    let maximum_error = cpu_logits
        .iter()
        .zip(&metal_logits)
        .map(|(cpu, metal)| (cpu - metal).abs())
        .fold(0.0_f32, f32::max);
    assert!(maximum_error < 1e-5, "maximum logit error: {maximum_error}");
    let cpu_tokens = cpu
        .generate_greedy(&[0, 1], 3, &cancellation, |_, _| {})
        .unwrap()
        .token_ids;
    let metal_tokens = metal
        .generate_greedy(&[0, 1], 3, &cancellation, |_, _| {})
        .unwrap()
        .token_ids;
    assert_eq!(metal_tokens, cpu_tokens);
}

struct TinyQwen {
    _directory: TempDir,
    path: std::path::PathBuf,
}

impl TinyQwen {
    fn write() -> Self {
        let directory = TempDir::new().unwrap();
        let path = directory.path().join("tiny-qwen35.gguf");
        write_tiny_gguf(&path);
        Self {
            _directory: directory,
            path,
        }
    }
}

#[derive(Clone)]
struct Tensor {
    name: String,
    dimensions: Vec<u64>,
    values: Vec<f32>,
    offset: u64,
}

enum Metadata<'a> {
    U32(&'a str, u32),
    F32(&'a str, f32),
    String(&'a str, &'a str),
    Strings(&'a str, Vec<&'a str>),
    I32s(&'a str, Vec<i32>),
}

fn write_tiny_gguf(path: &Path) {
    let metadata = vec![
        Metadata::String("general.architecture", "qwen35"),
        Metadata::U32("general.alignment", 32),
        Metadata::U32("qwen35.context_length", 32),
        Metadata::U32("qwen35.embedding_length", 4),
        Metadata::U32("qwen35.block_count", 3),
        Metadata::U32("qwen35.nextn_predict_layers", 1),
        Metadata::U32("qwen35.attention.head_count", 2),
        Metadata::U32("qwen35.attention.head_count_kv", 1),
        Metadata::U32("qwen35.attention.key_length", 2),
        Metadata::U32("qwen35.attention.value_length", 2),
        Metadata::U32("qwen35.rope.dimension_count", 2),
        Metadata::F32("qwen35.rope.freq_base", 10_000.0),
        Metadata::F32("qwen35.attention.layer_norm_rms_epsilon", 1e-6),
        Metadata::U32("qwen35.feed_forward_length", 3),
        Metadata::U32("qwen35.full_attention_interval", 2),
        Metadata::U32("qwen35.ssm.inner_size", 4),
        Metadata::U32("qwen35.ssm.state_size", 2),
        Metadata::U32("qwen35.ssm.time_step_rank", 2),
        Metadata::U32("qwen35.ssm.group_count", 1),
        Metadata::U32("qwen35.ssm.conv_kernel", 2),
        Metadata::String("tokenizer.ggml.model", "gpt2"),
        Metadata::String("tokenizer.ggml.pre", "qwen35"),
        Metadata::Strings(
            "tokenizer.ggml.tokens",
            vec!["a", "b", "c", "d", "e", "f", "g", "<eos>"],
        ),
        Metadata::I32s("tokenizer.ggml.token_type", vec![1, 1, 1, 1, 1, 1, 1, 3]),
        Metadata::Strings("tokenizer.ggml.merges", Vec::new()),
        Metadata::U32("tokenizer.ggml.eos_token_id", 7),
    ];
    let mut tensors = tiny_tensors();
    let mut next_offset = 0_u64;
    for tensor in &mut tensors {
        tensor.offset = next_offset;
        next_offset = align(next_offset + tensor.values.len() as u64 * 4, 32);
    }

    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    put_u32(&mut bytes, 3);
    put_u64(&mut bytes, tensors.len() as u64);
    put_u64(&mut bytes, metadata.len() as u64);
    for value in metadata {
        write_metadata(&mut bytes, value);
    }
    for tensor in &tensors {
        put_string(&mut bytes, &tensor.name);
        put_u32(&mut bytes, tensor.dimensions.len() as u32);
        for dimension in &tensor.dimensions {
            put_u64(&mut bytes, *dimension);
        }
        put_u32(&mut bytes, 0); // GGML_TYPE_F32
        put_u64(&mut bytes, tensor.offset);
    }
    pad_to(&mut bytes, 32);
    let data_offset = bytes.len();
    for tensor in tensors {
        bytes.resize(data_offset + tensor.offset as usize, 0);
        for value in tensor.values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    fs::write(path, bytes).unwrap();
}

fn tiny_tensors() -> Vec<Tensor> {
    let mut tensors = Vec::new();
    push_tensor(&mut tensors, "token_embd.weight", &[4, 8]);
    push_tensor(&mut tensors, "output_norm.weight", &[4]);
    push_tensor(&mut tensors, "output.weight", &[4, 8]);
    for layer in 0..2 {
        let prefix = format!("blk.{layer}");
        push_tensor(&mut tensors, &format!("{prefix}.attn_norm.weight"), &[4]);
        push_tensor(
            &mut tensors,
            &format!("{prefix}.post_attention_norm.weight"),
            &[4],
        );
        push_tensor(&mut tensors, &format!("{prefix}.ffn_gate.weight"), &[4, 3]);
        push_tensor(&mut tensors, &format!("{prefix}.ffn_up.weight"), &[4, 3]);
        push_tensor(&mut tensors, &format!("{prefix}.ffn_down.weight"), &[3, 4]);
    }
    for (name, dimensions) in [
        ("attn_qkv.weight", &[4, 8][..]),
        ("attn_gate.weight", &[4, 4]),
        ("ssm_conv1d.weight", &[2, 8]),
        ("ssm_alpha.weight", &[4, 2]),
        ("ssm_beta.weight", &[4, 2]),
        ("ssm_dt.bias", &[2]),
        ("ssm_a", &[2]),
        ("ssm_norm.weight", &[2]),
        ("ssm_out.weight", &[4, 4]),
    ] {
        push_tensor(&mut tensors, &format!("blk.0.{name}"), dimensions);
    }
    for (name, dimensions) in [
        ("attn_q.weight", &[4, 8][..]),
        ("attn_k.weight", &[4, 2]),
        ("attn_v.weight", &[4, 2]),
        ("attn_q_norm.weight", &[2]),
        ("attn_k_norm.weight", &[2]),
        ("attn_output.weight", &[4, 4]),
    ] {
        push_tensor(&mut tensors, &format!("blk.1.{name}"), dimensions);
    }
    tensors
}

fn push_tensor(tensors: &mut Vec<Tensor>, name: &str, dimensions: &[u64]) {
    let count = dimensions.iter().product::<u64>() as usize;
    let seed = name.bytes().map(usize::from).sum::<usize>();
    let values = if name.ends_with("norm.weight") {
        vec![1.0; count]
    } else if name.ends_with("ssm_a") {
        vec![-0.5; count]
    } else if name.ends_with("ssm_dt.bias") {
        vec![0.1; count]
    } else if name.ends_with("ssm_conv1d.weight") {
        (0..count)
            .map(|index| if index.is_multiple_of(2) { 0.2 } else { 0.8 })
            .collect()
    } else {
        (0..count)
            .map(|index| ((seed + index * 17) % 31) as f32 * 0.01 - 0.15)
            .collect()
    };
    tensors.push(Tensor {
        name: name.to_owned(),
        dimensions: dimensions.to_vec(),
        values,
        offset: 0,
    });
}

fn write_metadata(bytes: &mut Vec<u8>, metadata: Metadata<'_>) {
    match metadata {
        Metadata::U32(key, value) => {
            put_string(bytes, key);
            put_u32(bytes, 4);
            put_u32(bytes, value);
        }
        Metadata::F32(key, value) => {
            put_string(bytes, key);
            put_u32(bytes, 6);
            put_u32(bytes, value.to_bits());
        }
        Metadata::String(key, value) => {
            put_string(bytes, key);
            put_u32(bytes, 8);
            put_string(bytes, value);
        }
        Metadata::Strings(key, values) => {
            put_string(bytes, key);
            put_u32(bytes, 9);
            put_u32(bytes, 8);
            put_u64(bytes, values.len() as u64);
            for value in values {
                put_string(bytes, value);
            }
        }
        Metadata::I32s(key, values) => {
            put_string(bytes, key);
            put_u32(bytes, 9);
            put_u32(bytes, 5);
            put_u64(bytes, values.len() as u64);
            for value in values {
                put_u32(bytes, value as u32);
            }
        }
    }
}

fn put_string(bytes: &mut Vec<u8>, value: &str) {
    put_u64(bytes, value.len() as u64);
    bytes.extend_from_slice(value.as_bytes());
}

fn put_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn align(value: u64, alignment: u64) -> u64 {
    (value + alignment - 1) & !(alignment - 1)
}

fn pad_to(bytes: &mut Vec<u8>, alignment: usize) {
    bytes.resize((bytes.len() + alignment - 1) & !(alignment - 1), 0);
}
