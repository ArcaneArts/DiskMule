#![cfg(target_os = "macos")]

use std::{env, fs, path::PathBuf};

use diskmule::{
    cpu::argmax,
    gemma4::{
        cpu::{CancellationFlag, Gemma4CpuModel},
        metal::{ExpertResidency, Gemma4MetalModel},
    },
};
use serde::Deserialize;

const EXPECTED_DIGEST: &str =
    "sha256:7121486771cbfe218851513210c40b35dbdee93ab1ef43fe36283c883980f0df";
const MODEL_MEDIA_TYPE: &str = "application/vnd.ollama.image.model";
const CPU_METAL_TOLERANCE: f32 = 0.001;

#[derive(Deserialize)]
struct Manifest {
    layers: Vec<Layer>,
}

#[derive(Deserialize)]
struct Layer {
    #[serde(rename = "mediaType")]
    media_type: String,
    digest: String,
}

#[test]
#[ignore = "requires the installed 17 GB gemma4:26b and a full Metal forward pass"]
fn greedy_metal_decode_matches_the_cpu_and_ollama_oracles() {
    let Some(root) = ollama_root() else {
        eprintln!("skipping: Ollama model root is unavailable");
        return;
    };
    let manifest_path = root.join("manifests/registry.ollama.ai/library/gemma4/26b");
    if !manifest_path.is_file() {
        eprintln!("skipping: gemma4:26b is not installed");
        return;
    }
    let manifest: Manifest = serde_json::from_slice(&fs::read(manifest_path).unwrap()).unwrap();
    let layer = manifest
        .layers
        .iter()
        .find(|layer| layer.media_type == MODEL_MEDIA_TYPE)
        .unwrap();
    if layer.digest != EXPECTED_DIGEST {
        eprintln!("skipping: local model digest differs from the pinned oracle");
        return;
    }
    let blob = root.join("blobs").join(EXPECTED_DIGEST.replace(':', "-"));
    let cpu_model = Gemma4CpuModel::open(&blob, 8).unwrap();
    let model = Gemma4MetalModel::open(&blob, 8).unwrap();
    eprintln!("DiskMule Metal device: {}", model.device_name());
    let cancellation = CancellationFlag::default();
    let (cpu_logits, _) = cpu_model
        .prompt_logits(&[cpu_model.tokenizer.bos_id], &cancellation)
        .unwrap();
    let (metal_logits, metal_logits_profile) = model
        .prompt_logits(&[model.tokenizer.bos_id], &cancellation)
        .unwrap();
    let maximum_logit_error = cpu_logits
        .iter()
        .zip(&metal_logits)
        .map(|(cpu, metal)| (cpu - metal).abs())
        .fold(0.0_f32, f32::max);
    let cpu_winner = argmax(&cpu_logits).unwrap();
    let metal_winner = argmax(&metal_logits).unwrap();
    let cpu_log_probability = normalized_log_probability(&cpu_logits, cpu_winner);
    let metal_log_probability = normalized_log_probability(&metal_logits, metal_winner);
    eprintln!("maximum CPU/Metal logit error: {maximum_logit_error}");
    eprintln!(
        "winning CPU/Metal log probabilities: {cpu_log_probability} / {metal_log_probability}"
    );
    eprintln!("DiskMule Metal logits profile: {metal_logits_profile:#?}");
    assert_eq!(metal_winner, cpu_winner);
    assert!(
        maximum_logit_error < CPU_METAL_TOLERANCE,
        "CPU/Metal maximum logit error {maximum_logit_error} exceeds {CPU_METAL_TOLERANCE}"
    );
    assert!(
        (metal_log_probability - cpu_log_probability).abs() < CPU_METAL_TOLERANCE,
        "CPU/Metal winning log-probability error exceeds {CPU_METAL_TOLERANCE}"
    );
    let result = model
        .generate_greedy(
            &[model.tokenizer.bos_id],
            4,
            &CancellationFlag::default(),
            |_, _| {},
        )
        .unwrap();
    eprintln!("DiskMule Metal token IDs: {:?}", result.token_ids);
    eprintln!("DiskMule Metal profile: {:#?}", result.profile);
    assert_eq!(result.token_ids, [47_610, 1_852, 2_624, 1_852]);
    assert_eq!(result.text.trim_start(), "hedron ownley own");
    assert_eq!(result.profile.prompt_tokens, 1);
    assert_eq!(result.profile.generated_tokens, 4);

    let mut session = model.new_session().unwrap();
    let first_turn = model
        .generate_session_with_selector(
            &mut session,
            &[model.tokenizer.bos_id],
            2,
            &CancellationFlag::default(),
            |logits| Ok(argmax(logits).unwrap() as u32),
            |_, _| {},
        )
        .unwrap();
    let second_prompt = [
        model.tokenizer.bos_id,
        first_turn.token_ids[0],
        first_turn.token_ids[1],
    ];
    let second_turn = model
        .generate_session_with_selector(
            &mut session,
            &second_prompt,
            2,
            &CancellationFlag::default(),
            |logits| Ok(argmax(logits).unwrap() as u32),
            |_, _| {},
        )
        .unwrap();
    assert_eq!(first_turn.token_ids, result.token_ids[..2]);
    assert_eq!(second_turn.token_ids, result.token_ids[2..]);
    assert_eq!(second_turn.profile.reused_prompt_tokens, 2);

    let streamed = Gemma4MetalModel::open_with_expert_residency(
        &blob,
        8,
        ExpertResidency::Streamed { slots_per_layer: 1 },
    )
    .unwrap();
    let (streamed_logits, streamed_logits_profile) = streamed
        .prompt_logits(&[streamed.tokenizer.bos_id], &cancellation)
        .unwrap();
    let streamed_maximum_error = metal_logits
        .iter()
        .zip(&streamed_logits)
        .map(|(resident, streamed)| (resident - streamed).abs())
        .fold(0.0_f32, f32::max);
    eprintln!("maximum resident/streamed logit error: {streamed_maximum_error}");
    eprintln!("DiskMule streamed logits profile: {streamed_logits_profile:#?}");
    assert!(streamed_maximum_error < CPU_METAL_TOLERANCE);
    assert!(streamed_logits_profile.expert_cache_misses > 0);
    assert_eq!(
        streamed_logits_profile.expert_reads,
        streamed_logits_profile.expert_cache_misses * 2
    );
    assert!(streamed_logits_profile.expert_bytes_read > 0);
    assert!(streamed_logits_profile.expert_io_wait > std::time::Duration::ZERO);
    assert!(streamed_logits_profile.expert_resident_bytes > 0);
    let streamed_result = streamed
        .generate_greedy(
            &[streamed.tokenizer.bos_id],
            4,
            &CancellationFlag::default(),
            |_, _| {},
        )
        .unwrap();
    eprintln!("DiskMule streamed profile: {:#?}", streamed_result.profile);
    assert_eq!(streamed_result.token_ids, result.token_ids);
    assert_eq!(streamed_result.text, result.text);
}

fn normalized_log_probability(logits: &[f32], index: usize) -> f32 {
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    logits[index]
        - (logits
            .iter()
            .map(|logit| (*logit - maximum).exp())
            .sum::<f32>()
            .ln()
            + maximum)
}

fn ollama_root() -> Option<PathBuf> {
    env::var_os("OLLAMA_MODELS")
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".ollama/models")))
}
