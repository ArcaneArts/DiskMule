#![cfg(target_os = "macos")]

use std::{env, fs, net::SocketAddr, path::PathBuf};

use diskmule::{
    config::Paths,
    runtime::{BackendSelection, RuntimeLimits, RuntimeService},
    server::serve_listener,
};
use serde::Deserialize;
use tempfile::TempDir;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::oneshot,
    time::{Duration, timeout},
};

const EXPECTED_DIGEST: &str =
    "sha256:f5f1dd8920d417aac2718b0bda3403da274301efdd6760b4f0f4b864ff2ad57d";
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires the pinned 16 GB qwen3.8 model and Metal hybrid inference"]
async fn qwen35_streaming_and_non_streaming_share_the_runtime_session() {
    let Some(root) = pinned_ollama_root() else {
        eprintln!("skipping: pinned qwen3.8:latest is unavailable");
        return;
    };
    let temp = TempDir::new().unwrap();
    let runtime = RuntimeService::new(
        Paths::from_root(temp.path().to_path_buf()),
        Some(root),
        BackendSelection::from_environment().unwrap(),
        RuntimeLimits {
            context: 64,
            maximum_loaded_models: 1,
            request_queue: 2,
            token_buffer: 2,
            maximum_sessions_per_model: 1,
        },
    )
    .unwrap();
    let listener = TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0)))
        .await
        .unwrap();
    let address = listener.local_addr().unwrap();
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let server = tokio::spawn(serve_listener(listener, runtime, async move {
        let _ = shutdown_rx.await;
    }));

    let non_streaming = request(address, "/api/chat", &chat_body(false))
        .await
        .unwrap();
    assert!(non_streaming.starts_with("HTTP/1.1 200 OK"));
    assert!(non_streaming.contains(r#""content":"Hello""#));
    assert!(non_streaming.contains(r#""done":true"#));

    let streaming = request(address, "/api/chat", &chat_body(true))
        .await
        .unwrap();
    assert!(streaming.starts_with("HTTP/1.1 200 OK"));
    assert!(streaming.contains("application/x-ndjson"));
    assert!(streaming.contains(r#""content":"Hello""#));
    assert!(streaming.contains(r#""done":true"#));

    let loaded = request(address, "/api/loaded", "").await.unwrap();
    assert!(loaded.contains(r#""name":"qwen3.8:latest""#));
    assert!(loaded.contains(r#""architecture":"qwen35""#));
    assert!(loaded.contains(r#""backend":"Metal ("#));
    assert!(loaded.contains(r#""status":"ready""#));

    shutdown_tx.send(()).unwrap();
    timeout(Duration::from_secs(5), server)
        .await
        .expect("server and Qwen worker should stop promptly")
        .expect("server task should not panic")
        .expect("server should shut down cleanly");
}

fn chat_body(stream: bool) -> String {
    serde_json::json!({
        "model": "qwen3.8:latest",
        "session": "qwen-conformance",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": stream,
        "options": {"temperature": 0, "num_predict": 1}
    })
    .to_string()
}

async fn request(address: SocketAddr, path: &str, body: &str) -> std::io::Result<String> {
    let mut stream = TcpStream::connect(address).await?;
    let method = if body.is_empty() { "GET" } else { "POST" };
    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(request.as_bytes()).await?;
    let mut response = Vec::new();
    timeout(Duration::from_secs(60), stream.read_to_end(&mut response))
        .await
        .map_err(std::io::Error::other)??;
    String::from_utf8(response).map_err(std::io::Error::other)
}

fn pinned_ollama_root() -> Option<PathBuf> {
    let root = env::var_os("OLLAMA_MODELS")
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".ollama/models")))?;
    let manifest_path = root.join("manifests/registry.ollama.ai/library/qwen3.8/latest");
    let manifest: Manifest = serde_json::from_slice(&fs::read(manifest_path).ok()?).ok()?;
    manifest
        .layers
        .iter()
        .any(|layer| layer.media_type == MODEL_MEDIA_TYPE && layer.digest == EXPECTED_DIGEST)
        .then_some(root)
}
