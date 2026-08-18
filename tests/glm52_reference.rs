use std::{env, path::PathBuf};

use diskmule::{
    generation::{CancellationFlag, RuntimeError},
    glm52::{Glm52Config, Glm52FeedForwardWeights, Glm52Weights},
    safetensors::{SafeDtype, SafeTensorIndex},
};

#[cfg(target_os = "macos")]
use diskmule::glm52::cpu::Glm52CpuModel;

fn snapshot_directory() -> Option<PathBuf> {
    env::var_os("DISKMULE_GLM52_SNAPSHOT").map(PathBuf::from)
}

/// Payload-lazy identity and tensor-contract oracle for the pinned local
/// Colibri grouped-INT4 checkpoint.
#[test]
#[ignore = "requires the complete 400 GB GLM-5.2 Colibri snapshot"]
fn pinned_colibri_snapshot_matches_the_runtime_contract() {
    let Some(directory) = snapshot_directory() else {
        eprintln!("skipping: DISKMULE_GLM52_SNAPSHOT is unavailable");
        return;
    };

    let config = Glm52Config::from_directory(&directory).unwrap();
    let index = SafeTensorIndex::open(&directory).unwrap();
    assert_eq!(index.shards().len(), 142);
    assert_eq!(index.tensors().len(), 118_478);
    assert_eq!(
        index
            .tensors()
            .iter()
            .filter(|tensor| tensor.dtype == SafeDtype::F32)
            .count(),
        59_475
    );
    assert_eq!(
        index
            .tensors()
            .iter()
            .filter(|tensor| tensor.dtype == SafeDtype::U8)
            .count(),
        59_003
    );

    let weights = Glm52Weights::from_index(&index, &config).unwrap();
    assert_eq!(weights.layers.len(), 78);
    assert!(!weights.dsa_enabled);
    assert_eq!(
        weights
            .layers
            .iter()
            .filter(|layer| matches!(layer.feed_forward, Glm52FeedForwardWeights::Dense { .. }))
            .count(),
        3
    );
    for layer in &weights.layers[3..] {
        let Glm52FeedForwardWeights::Sparse { experts, .. } = &layer.feed_forward else {
            panic!("post-dense GLM layer must be sparse");
        };
        assert_eq!(experts.len(), 256);
    }
}

/// Expensive real-model oracle captured from Colibri at repository revision
/// fd9b461ac7cae4b921470d0db12230c6505bd03c with greedy decoding, MTP off,
/// and the exact grouped-INT4 checkpoint validated above.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires the complete 400 GB GLM-5.2 checkpoint and a full Metal forward pass"]
fn forced_streaming_metal_decode_matches_the_colibri_oracle() {
    let Some(directory) = snapshot_directory() else {
        eprintln!("skipping: DISKMULE_GLM52_SNAPSHOT is unavailable");
        return;
    };
    assert_eq!(
        env::var("DISKMULE_GLM_EXPERT_SLOTS").as_deref(),
        Ok("1"),
        "run the forced-streaming oracle with DISKMULE_GLM_EXPERT_SLOTS=1"
    );

    let model = Glm52CpuModel::from_directory_with_context_and_metal(&directory, 64).unwrap();
    let prompt = model
        .tokenizer
        .encode("[gMASK]<sop><|user|>Hi<|assistant|><think></think>")
        .unwrap();
    assert_eq!(
        prompt,
        [154_822, 154_824, 154_827, 13_041, 154_828, 154_841, 154_842]
    );

    let mut session = model.new_session();
    let result = model
        .generate_session_with_selector(
            &mut session,
            &prompt,
            4,
            &CancellationFlag::default(),
            greedy,
            |_, _| {},
        )
        .unwrap();
    eprintln!("DiskMule GLM token IDs: {:?}", result.token_ids);
    eprintln!("DiskMule GLM profile: {:#?}", result.profile);
    assert_eq!(result.token_ids, [13_041, 0, 358, 2_776]);
    assert_eq!(result.text, "Hi! I'm");
    assert!(result.profile.mapped_bytes_touched > 0);
    assert!(result.profile.expert_cache_misses > 0);
    assert!(result.profile.expert_cache_evictions > 0);
    assert!(result.profile.expert_reads > 0);
    assert!(result.profile.expert_bytes_read > 0);
    assert!(result.profile.expert_resident_bytes > 0);
    assert!(result.profile.resident_kv_bytes > 0);
}

#[cfg(target_os = "macos")]
fn greedy(logits: &[f32]) -> Result<u32, RuntimeError> {
    logits
        .iter()
        .enumerate()
        .filter(|(_, logit)| logit.is_finite())
        .max_by(|left, right| left.1.total_cmp(right.1).then_with(|| right.0.cmp(&left.0)))
        .map(|(token, _)| token as u32)
        .ok_or_else(|| {
            RuntimeError::InvalidConfiguration("GLM logits are all non-finite".to_owned())
        })
}
