use std::{env, fs, path::PathBuf};

use diskmule::gemma4::cpu::{CancellationFlag, Gemma4CpuModel};
use serde::Deserialize;

const EXPECTED_DIGEST: &str =
    "sha256:7121486771cbfe218851513210c40b35dbdee93ab1ef43fe36283c883980f0df";
const MODEL_MEDIA_TYPE: &str = "application/vnd.ollama.image.model";

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

/// Expensive, hardware-local oracle captured from Ollama 0.32.14 with the
/// exact model digest above, raw `<bos>`, temperature zero, and one output.
#[test]
#[ignore = "requires the installed 17 GB gemma4:26b and a full CPU forward pass"]
fn greedy_kv_decode_matches_the_local_ollama_reference() {
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
    let model = Gemma4CpuModel::open(blob, 8).unwrap();
    let (logits, logits_profile) = model
        .prompt_logits(&[model.tokenizer.bos_id], &CancellationFlag::default())
        .unwrap();
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let log_sum_exp = logits
        .iter()
        .map(|logit| (*logit - maximum).exp())
        .sum::<f32>()
        .ln()
        + maximum;
    let winning_log_probability = logits[47_610] - log_sum_exp;
    eprintln!("DiskMule winning logprob: {winning_log_probability}");
    eprintln!("DiskMule logits profile: {logits_profile:#?}");
    assert!(
        (winning_log_probability - -4.024_969_6).abs() < 0.02,
        "DiskMule/Ollama logprob mismatch: {winning_log_probability} versus -4.0249696"
    );
    let result = model
        .generate_greedy(
            &[model.tokenizer.bos_id],
            4,
            &CancellationFlag::default(),
            |_, _| {},
        )
        .unwrap();
    eprintln!("DiskMule token IDs: {:?}", result.token_ids);
    eprintln!("DiskMule profile: {:#?}", result.profile);
    assert_eq!(result.token_ids, [47_610, 1_852, 2_624, 1_852]);
    assert_eq!(result.text.trim_start(), "hedron ownley own");
    assert_eq!(result.profile.prompt_tokens, 1);
    assert_eq!(result.profile.generated_tokens, 4);
}

fn ollama_root() -> Option<PathBuf> {
    env::var_os("OLLAMA_MODELS")
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".ollama/models")))
}
