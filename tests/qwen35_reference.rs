use std::{env, fs, path::Path, path::PathBuf, process::Command};

use diskmule::{
    gemma4::{ChatMessage, ChatRole},
    gguf::TensorSource,
    qwen35::{
        Qwen35Config, Qwen35Weights, RUNTIME_ARRAY_KEYS,
        tokenizer::{Qwen35Tokenizer, render_chat},
    },
};
use serde::Deserialize;

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
fn installed_qwen35_contract_and_tokenizer_match_llama_cpp() {
    let Some(root) = ollama_root() else {
        eprintln!("skipping: Ollama model root is unavailable");
        return;
    };
    let manifest = root.join("manifests/registry.ollama.ai/library/qwen3.8/latest");
    if !manifest.is_file() {
        eprintln!("skipping: qwen3.8:latest is not installed");
        return;
    }
    let blob = model_blob(&root, &manifest);
    let source = TensorSource::open_with_arrays(&blob, RUNTIME_ARRAY_KEYS).unwrap();
    let config = Qwen35Config::from_gguf(&source.gguf).unwrap();
    let weights = Qwen35Weights::from_gguf(&source.gguf, &config).unwrap();
    assert_eq!(config.layer_count, 64);
    assert_eq!(config.nextn_layers, 1);
    assert_eq!(weights.layers.len(), 64);
    let tokenizer = Qwen35Tokenizer::from_gguf(&source.gguf).unwrap();
    assert_eq!(tokenizer.vocabulary_size(), 248_320);
    assert_eq!(tokenizer.token_id("<|im_start|>"), Some(248_045));
    assert!(tokenizer.stop_token_ids().contains(&248_046));

    for prompt in [
        "Hello, world! 1234 café 世界",
        "can't we're 0001234\n\nnext",
        "<|im_start|>user\nHello<|im_end|>\n",
    ] {
        let actual = tokenizer.encode(prompt).unwrap();
        if let Some(expected) = reference_tokenize(&blob, prompt) {
            assert_eq!(actual, expected, "token mismatch for {prompt:?}");
        }
        assert_eq!(tokenizer.decode(&actual, true).unwrap(), prompt);
    }

    let chat = render_chat(
        &[
            ChatMessage::new(ChatRole::System, "Be terse."),
            ChatMessage::new(ChatRole::User, "Hello"),
        ],
        false,
    )
    .unwrap();
    if let Some(expected) = reference_tokenize(&blob, &chat) {
        assert_eq!(tokenizer.encode(&chat).unwrap(), expected);
    }
}

fn reference_tokenize(model: &Path, prompt: &str) -> Option<Vec<u32>> {
    let output = match Command::new("llama-tokenize")
        .arg("--model")
        .arg(model)
        .arg("--prompt")
        .arg(prompt)
        .arg("--ids")
        .arg("--no-escape")
        .arg("--no-bos")
        .arg("--log-disable")
        .output()
    {
        Ok(output) => output,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("skipping tokenizer oracle: llama-tokenize is unavailable");
            return None;
        }
        Err(error) => panic!("could not run llama-tokenize: {error}"),
    };
    assert!(
        output.status.success(),
        "llama-tokenize failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Some(serde_json::from_slice(&output.stdout).unwrap())
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
    root.join("blobs").join(layer.digest.replacen(':', "-", 1))
}
