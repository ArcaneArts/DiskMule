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
    time::{Duration, sleep, timeout},
};

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

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires the installed 17 GB gemma4:26b and Metal server inference"]
async fn concurrent_streaming_disconnect_and_shutdown_are_safe() {
    let Some(root) = pinned_ollama_root() else {
        eprintln!("skipping: pinned gemma4:26b is unavailable");
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
            maximum_sessions_per_model: 2,
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

    let non_streaming_body = chat_body(false, "server-a", 1);
    let streaming_body = chat_body(true, "server-b", 1);
    let (non_streaming, streaming) = tokio::join!(
        request(address, "/api/chat", &non_streaming_body),
        request(address, "/api/chat", &streaming_body),
    );
    let non_streaming = non_streaming.unwrap();
    let streaming = streaming.unwrap();
    assert!(non_streaming.starts_with("HTTP/1.1 200 OK"));
    assert!(non_streaming.contains(r#""content":"Hello""#));
    assert!(non_streaming.contains(r#""done":true"#));
    assert!(streaming.starts_with("HTTP/1.1 200 OK"));
    assert!(streaming.contains("application/x-ndjson"));
    assert!(streaming.contains(r#""content":"Hello""#));
    assert!(streaming.contains(r#""done":true"#));

    let mut disconnected = TcpStream::connect(address).await.unwrap();
    let disconnect_body = chat_body(true, "server-a", 32);
    write_request(&mut disconnected, "/api/chat", &disconnect_body)
        .await
        .unwrap();
    drop(disconnected);
    sleep(Duration::from_millis(250)).await;

    let recovery = request(address, "/api/chat", &chat_body(false, "server-a", 1))
        .await
        .unwrap();
    assert!(recovery.starts_with("HTTP/1.1 200 OK"));
    assert!(recovery.contains(r#""done":true"#));
    let loaded = request(address, "/api/loaded", "").await.unwrap();
    assert!(loaded.contains(r#""name":"gemma4:26b""#));
    assert!(loaded.contains(r#""status":"ready""#));

    shutdown_tx.send(()).unwrap();
    timeout(Duration::from_secs(5), server)
        .await
        .expect("server and model worker should stop promptly")
        .expect("server task should not panic")
        .expect("server should shut down cleanly");
}

fn chat_body(stream: bool, session: &str, tokens: usize) -> String {
    serde_json::json!({
        "model": "gemma4:26b",
        "session": session,
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": stream,
        "options": {"temperature": 0, "num_predict": tokens}
    })
    .to_string()
}

async fn request(address: SocketAddr, path: &str, body: &str) -> std::io::Result<String> {
    let mut stream = TcpStream::connect(address).await?;
    write_request(&mut stream, path, body).await?;
    let mut response = Vec::new();
    timeout(Duration::from_secs(60), stream.read_to_end(&mut response))
        .await
        .map_err(std::io::Error::other)??;
    String::from_utf8(response).map_err(std::io::Error::other)
}

async fn write_request(stream: &mut TcpStream, path: &str, body: &str) -> std::io::Result<()> {
    let method = if body.is_empty() { "GET" } else { "POST" };
    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(request.as_bytes()).await
}

fn pinned_ollama_root() -> Option<PathBuf> {
    let root = env::var_os("OLLAMA_MODELS")
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".ollama/models")))?;
    let manifest_path = root.join("manifests/registry.ollama.ai/library/gemma4/26b");
    let manifest: Manifest = serde_json::from_slice(&fs::read(manifest_path).ok()?).ok()?;
    manifest
        .layers
        .iter()
        .any(|layer| layer.media_type == MODEL_MEDIA_TYPE && layer.digest == EXPECTED_DIGEST)
        .then_some(root)
}
