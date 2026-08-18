use std::{
    collections::BTreeMap,
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

use serde::Deserialize;
use tempfile::TempDir;

use diskmule::gguf::{MetadataArray, MetadataValue, TensorSource};

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

#[test]
fn installed_gemma_is_listed_inspected_and_protected() {
    let Some(ollama_root) = ollama_root() else {
        eprintln!("skipping local Ollama smoke test: home directory unavailable");
        return;
    };
    let manifest_path = ollama_root.join("manifests/registry.ollama.ai/library/gemma4/26b");
    if !manifest_path.is_file() {
        eprintln!("skipping local Ollama smoke test: gemma4:26b is not installed");
        return;
    }

    let blob = model_blob(&ollama_root, &manifest_path);
    let before = fingerprint(&blob);
    let isolated_home = TempDir::new().unwrap();

    let source = TensorSource::open_with_arrays(&blob, ["tokenizer.ggml.tokens"]).unwrap();
    assert_eq!(source.gguf.architecture(), Some("gemma4"));
    let tokens = source
        .gguf
        .metadata
        .get("tokenizer.ggml.tokens")
        .and_then(MetadataValue::as_array)
        .and_then(MetadataArray::as_strings)
        .expect("Gemma GGUF should expose its tokenizer vocabulary");
    assert_eq!(tokens.len(), 262_144);
    let tensor_type_counts =
        source
            .gguf
            .tensors
            .iter()
            .fold(BTreeMap::<_, usize>::new(), |mut counts, tensor| {
                *counts.entry(tensor.kind.name()).or_default() += 1;
                counts
            });
    assert_eq!(
        tensor_type_counts,
        BTreeMap::from([
            ("F16", 191),
            ("F32", 557),
            ("Q4_K", 193),
            ("Q5_0", 32),
            ("Q6_K", 13),
            ("Q8_0", 28),
        ])
    );
    for kind in [
        diskmule::gguf::TensorType::F16,
        diskmule::gguf::TensorType::F32,
        diskmule::gguf::TensorType::Q4K,
        diskmule::gguf::TensorType::Q5_0,
        diskmule::gguf::TensorType::Q6K,
        diskmule::gguf::TensorType::Q8_0,
    ] {
        let tensor = source
            .gguf
            .tensors
            .iter()
            .find(|tensor| tensor.kind == kind)
            .unwrap();
        let columns = tensor.dimensions[0];
        let mut encoded = vec![0_u8; kind.row_byte_len(columns).unwrap() as usize];
        source
            .read_tensor_at(&tensor.name, 0, &mut encoded)
            .unwrap();
        let mut decoded = vec![0.0_f32; columns as usize];
        diskmule::cpu::dequantize(kind, &encoded, &mut decoded).unwrap();
        assert!(
            decoded.iter().all(|value| value.is_finite()),
            "{} decoded non-finite values from {}",
            kind.name(),
            tensor.name
        );
    }
    let first_tensor = &source.gguf.tensors[0];
    let mut tensor_prefix = [0_u8; 16];
    source
        .read_tensor_at(&first_tensor.name, 0, &mut tensor_prefix)
        .unwrap();

    let list = diskmule(isolated_home.path(), ["ls"]);
    assert!(list.status.success(), "ls failed: {}", stderr(&list));
    let listing = String::from_utf8(list.stdout).unwrap();
    let gemma = listing
        .lines()
        .find(|line| line.starts_with("gemma4:26b\t"))
        .expect("gemma4:26b should be listed");
    assert!(gemma.contains("\tgemma4\tQ4_K_M\t"));
    assert!(gemma.contains("\tollama (read-only)\t"));
    assert!(gemma.contains("metadata-compatible; inference pending"));

    let run = diskmule(isolated_home.path(), ["run", "gemma4:26b"]);
    assert_eq!(run.status.code(), Some(1));
    let inspection = String::from_utf8_lossy(&run.stdout);
    assert!(inspection.contains("Model: gemma4:26b"));
    assert!(inspection.contains("Source: ollama (read-only)"));
    assert!(inspection.contains("Architecture: gemma4"));
    assert!(inspection.contains("Quantization: Q4_K_M"));
    assert!(inspection.contains("GGUF version: 3"));
    assert!(stderr(&run).contains("model inference is not implemented yet"));

    let remove = diskmule(isolated_home.path(), ["rm", "gemma4:26b"]);
    assert_eq!(remove.status.code(), Some(1));
    assert!(stderr(&remove).contains("owned by ollama (read-only)"));
    assert_eq!(fingerprint(&blob), before);
}

fn diskmule<const N: usize>(home: &Path, arguments: [&str; N]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_diskmule"))
        .args(arguments)
        .env("DISKMULE_HOME", home)
        .output()
        .unwrap()
}

fn ollama_root() -> Option<PathBuf> {
    env::var_os("OLLAMA_MODELS")
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".ollama/models")))
}

fn model_blob(root: &Path, manifest_path: &Path) -> PathBuf {
    let manifest: Manifest = serde_json::from_slice(&fs::read(manifest_path).unwrap()).unwrap();
    let layer = manifest
        .layers
        .iter()
        .find(|layer| layer.media_type == MODEL_MEDIA_TYPE)
        .expect("Ollama manifest should have a model layer");
    let digest = layer
        .digest
        .strip_prefix("sha256:")
        .expect("model layer should use a sha256 digest");
    root.join("blobs").join(format!("sha256-{digest}"))
}

fn fingerprint(path: &Path) -> (u64, std::time::SystemTime, u64) {
    use std::os::unix::fs::MetadataExt;

    let metadata = fs::metadata(path).unwrap();
    (metadata.len(), metadata.modified().unwrap(), metadata.ino())
}

fn stderr(output: &std::process::Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}
