use std::{env, path::PathBuf};

use diskmule::{
    glm52::{Glm52Config, Glm52FeedForwardWeights, Glm52Weights},
    safetensors::{SafeDtype, SafeTensorIndex},
};

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
