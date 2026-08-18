use std::{
    convert::Infallible,
    env,
    future::Future,
    io,
    net::{Ipv4Addr, SocketAddr, SocketAddrV4},
};

use axum::{
    Json, Router,
    body::{Body, Bytes},
    extract::{State, rejection::JsonRejection},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use futures_util::stream;
use serde::{Deserialize, Serialize};
use time::{OffsetDateTime, format_description::well_known::Rfc3339};
use tokio::net::TcpListener;

use crate::{
    gemma4::{ChatMessage, ChatRole, cpu::GenerationProfile},
    model::ModelError,
    runtime::{
        GenerationEvent, GenerationOptions, GenerationTicket, RuntimeService, SamplingConfig,
        ServiceError,
    },
};

pub const DEFAULT_BIND: SocketAddr = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 11_435));

#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    #[error("could not bind DiskMule server to {address}: {source}")]
    Bind {
        address: SocketAddr,
        source: io::Error,
    },

    #[error("DiskMule server failed: {0}")]
    Serve(io::Error),

    #[error("DISKMULE_BIND must be a socket address such as 127.0.0.1:11435")]
    InvalidBind,
}

#[derive(Debug, Serialize)]
struct Health {
    status: &'static str,
    service: &'static str,
    version: &'static str,
    loaded_models: usize,
}

#[derive(Debug, Deserialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<ApiMessage>,
    #[serde(default = "default_stream")]
    stream: bool,
    #[serde(default)]
    options: OllamaOptions,
    #[serde(default)]
    session: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ApiMessage {
    role: ApiRole,
    content: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum ApiRole {
    System,
    User,
    Assistant,
}

#[derive(Debug, Default, Deserialize)]
struct OllamaOptions {
    temperature: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    seed: Option<u64>,
    num_predict: Option<usize>,
}

#[derive(Debug, Serialize)]
struct OllamaChatResponse {
    model: String,
    created_at: String,
    message: ApiMessage,
    done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    done_reason: Option<&'static str>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    metrics: Option<ResponseMetrics>,
}

#[derive(Debug, Serialize)]
struct ResponseMetrics {
    total_duration: u64,
    load_duration: u64,
    prompt_eval_count: usize,
    prompt_cached_count: usize,
    prompt_eval_duration: u64,
    eval_count: usize,
    eval_duration: u64,
}

#[derive(Debug, Serialize)]
struct ApiError {
    error: String,
}

pub fn bind_from_environment() -> Result<SocketAddr, ServerError> {
    let Some(raw) = env::var_os("DISKMULE_BIND") else {
        return Ok(DEFAULT_BIND);
    };
    raw.to_str()
        .and_then(|value| value.parse().ok())
        .ok_or(ServerError::InvalidBind)
}

pub async fn serve(
    address: SocketAddr,
    runtime: RuntimeService,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<(), ServerError> {
    let listener = TcpListener::bind(address)
        .await
        .map_err(|source| ServerError::Bind { address, source })?;
    serve_listener(listener, runtime, shutdown).await
}

pub async fn serve_listener(
    listener: TcpListener,
    runtime: RuntimeService,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<(), ServerError> {
    let address = listener.local_addr().map_err(|source| ServerError::Bind {
        address: DEFAULT_BIND,
        source,
    })?;
    tracing::info!(%address, "DiskMule server listening");
    axum::serve(listener, router(runtime))
        .with_graceful_shutdown(shutdown)
        .await
        .map_err(ServerError::Serve)
}

fn router(runtime: RuntimeService) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/api/loaded", get(loaded_models))
        .route("/api/chat", post(chat))
        .with_state(runtime)
}

async fn health(State(runtime): State<RuntimeService>) -> Json<Health> {
    Json(Health {
        status: "ok",
        service: "diskmule",
        version: env!("CARGO_PKG_VERSION"),
        loaded_models: runtime.loaded_models().len(),
    })
}

async fn loaded_models(State(runtime): State<RuntimeService>) -> impl IntoResponse {
    Json(serde_json::json!({ "models": runtime.loaded_models() }))
}

async fn chat(
    State(runtime): State<RuntimeService>,
    request: Result<Json<OllamaChatRequest>, JsonRejection>,
) -> Response {
    let Json(request) = match request {
        Ok(request) => request,
        Err(error) => return json_error(StatusCode::BAD_REQUEST, error.body_text()),
    };
    let (messages, options) = match prepare_request(&request) {
        Ok(prepared) => prepared,
        Err(error) => return json_error(StatusCode::BAD_REQUEST, error),
    };
    let ticket = match &request.session {
        Some(session) => runtime.generate_in_session(&request.model, session, messages, options),
        None => runtime.generate(&request.model, messages, options),
    };
    let ticket = match ticket {
        Ok(ticket) => ticket,
        Err(error) => return service_error_response(error),
    };
    if request.stream {
        streaming_response(request.model, ticket)
    } else {
        non_streaming_response(request.model, ticket).await
    }
}

fn prepare_request(
    request: &OllamaChatRequest,
) -> Result<(Vec<ChatMessage>, GenerationOptions), String> {
    if request.model.trim().is_empty() {
        return Err("model must not be empty".to_owned());
    }
    let messages = request
        .messages
        .iter()
        .map(|message| {
            ChatMessage::new(
                match message.role {
                    ApiRole::System => ChatRole::System,
                    ApiRole::User => ChatRole::User,
                    ApiRole::Assistant => ChatRole::Assistant,
                },
                &message.content,
            )
        })
        .collect::<Vec<_>>();
    let defaults = SamplingConfig::default();
    let options = GenerationOptions {
        maximum_new_tokens: request
            .options
            .num_predict
            .unwrap_or(GenerationOptions::default().maximum_new_tokens),
        sampling: SamplingConfig {
            temperature: request.options.temperature.unwrap_or(defaults.temperature),
            top_k: request.options.top_k.unwrap_or(defaults.top_k),
            top_p: request.options.top_p.unwrap_or(defaults.top_p),
            seed: request.options.seed.unwrap_or(defaults.seed),
        },
    };
    options
        .validate()
        .map_err(|error| error.to_string())
        .map(|options| (messages, options))
}

fn streaming_response(model: String, ticket: GenerationTicket) -> Response {
    let output = stream::unfold(
        (ticket, model, false),
        |(mut ticket, model, finished)| async move {
            if finished {
                return None;
            }
            let response = match ticket.recv().await {
                Some(GenerationEvent::Token { text, .. }) => OllamaChatResponse {
                    model: model.clone(),
                    created_at: created_at(),
                    message: ApiMessage {
                        role: ApiRole::Assistant,
                        content: text,
                    },
                    done: false,
                    done_reason: None,
                    metrics: None,
                },
                Some(GenerationEvent::Complete(result)) => OllamaChatResponse {
                    model: model.clone(),
                    created_at: created_at(),
                    message: ApiMessage {
                        role: ApiRole::Assistant,
                        content: String::new(),
                    },
                    done: true,
                    done_reason: Some(if result.stopped { "stop" } else { "length" }),
                    metrics: Some(metrics(&result.profile)),
                },
                Some(GenerationEvent::Error { message, .. }) => {
                    let line = json_line(&ApiError { error: message });
                    return Some((
                        Ok::<_, Infallible>(Bytes::from(line)),
                        (ticket, model, true),
                    ));
                }
                None => return None,
            };
            let done = response.done;
            let line = json_line(&response);
            Some((
                Ok::<_, Infallible>(Bytes::from(line)),
                (ticket, model, done),
            ))
        },
    );
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/x-ndjson")
        .body(Body::from_stream(output))
        .expect("static streaming response is valid")
}

async fn non_streaming_response(model: String, mut ticket: GenerationTicket) -> Response {
    let mut text = String::new();
    while let Some(event) = ticket.recv().await {
        match event {
            GenerationEvent::Token { text: piece, .. } => text.push_str(&piece),
            GenerationEvent::Complete(result) => {
                return Json(OllamaChatResponse {
                    model,
                    created_at: created_at(),
                    message: ApiMessage {
                        role: ApiRole::Assistant,
                        content: text,
                    },
                    done: true,
                    done_reason: Some(if result.stopped { "stop" } else { "length" }),
                    metrics: Some(metrics(&result.profile)),
                })
                .into_response();
            }
            GenerationEvent::Error { message, cancelled } => {
                let status = if cancelled {
                    StatusCode::REQUEST_TIMEOUT
                } else {
                    StatusCode::INTERNAL_SERVER_ERROR
                };
                return json_error(status, message);
            }
        }
    }
    json_error(
        StatusCode::INTERNAL_SERVER_ERROR,
        "generation worker closed without a final response",
    )
}

fn metrics(profile: &GenerationProfile) -> ResponseMetrics {
    ResponseMetrics {
        total_duration: duration_nanos(profile.total_time),
        load_duration: 0,
        prompt_eval_count: profile
            .prompt_tokens
            .saturating_sub(profile.reused_prompt_tokens),
        prompt_cached_count: profile.reused_prompt_tokens,
        prompt_eval_duration: duration_nanos(profile.prefill_time),
        eval_count: profile.generated_tokens,
        eval_duration: duration_nanos(profile.decode_time),
    }
}

fn duration_nanos(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn json_line(value: &impl Serialize) -> Vec<u8> {
    let mut line = serde_json::to_vec(value).expect("API response serialization cannot fail");
    line.push(b'\n');
    line
}

fn service_error_response(error: ServiceError) -> Response {
    let status = match &error {
        ServiceError::Model(ModelError::NotFound(_)) => StatusCode::NOT_FOUND,
        ServiceError::QueueFull { .. } => StatusCode::TOO_MANY_REQUESTS,
        ServiceError::ModelLimit { .. } => StatusCode::SERVICE_UNAVAILABLE,
        ServiceError::InvalidRequest(_) => StatusCode::BAD_REQUEST,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    json_error(status, error.to_string())
}

fn json_error(status: StatusCode, message: impl Into<String>) -> Response {
    (
        status,
        Json(ApiError {
            error: message.into(),
        }),
    )
        .into_response()
}

fn created_at() -> String {
    OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .expect("RFC 3339 formatting uses no fallible external state")
}

const fn default_stream() -> bool {
    true
}

pub async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};

        let mut terminate = signal(SignalKind::terminate()).ok();
        tokio::select! {
            result = tokio::signal::ctrl_c() => {
                if let Err(error) = result {
                    tracing::warn!(%error, "could not listen for Ctrl-C");
                }
            }
            _ = async {
                if let Some(signal) = &mut terminate {
                    signal.recv().await;
                } else {
                    std::future::pending::<()>().await;
                }
            } => {}
        }
    }

    #[cfg(not(unix))]
    if let Err(error) = tokio::signal::ctrl_c().await {
        tracing::warn!(%error, "could not listen for Ctrl-C");
    }

    tracing::info!("DiskMule server shutting down");
}

#[cfg(test)]
mod tests {
    use std::net::SocketAddr;

    use tempfile::TempDir;
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::{TcpListener, TcpStream},
        sync::oneshot,
        time::{Duration, timeout},
    };

    use crate::{
        config::Paths,
        runtime::{BackendSelection, RuntimeLimits, RuntimeService},
    };

    use super::serve_listener;

    fn test_runtime(temp: &TempDir) -> RuntimeService {
        RuntimeService::new(
            Paths::from_root(temp.path().to_path_buf()),
            None,
            BackendSelection::Cpu,
            RuntimeLimits::default(),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn health_and_loaded_endpoints_are_json_and_server_stops_gracefully() {
        let temp = TempDir::new().unwrap();
        let listener = TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0)))
            .await
            .unwrap();
        let address = listener.local_addr().unwrap();
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let task = tokio::spawn(serve_listener(listener, test_runtime(&temp), async move {
            let _ = shutdown_rx.await;
        }));

        for (path, body) in [
            ("/health", r#"{"status":"ok","service":"diskmule""#),
            ("/api/loaded", r#"{"models":[]}"#),
        ] {
            let response = request(address, &format!("GET {path}"), "").await;
            assert!(response.starts_with("HTTP/1.1 200 OK"));
            assert!(response.contains("content-type: application/json"));
            assert!(response.contains(body));
        }

        shutdown_tx.send(()).unwrap();
        timeout(Duration::from_secs(2), task)
            .await
            .expect("server should stop after shutdown signal")
            .expect("server task should not panic")
            .expect("server should shut down cleanly");
    }

    #[tokio::test]
    async fn malformed_and_missing_model_requests_return_json_errors() {
        let temp = TempDir::new().unwrap();
        let listener = TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0)))
            .await
            .unwrap();
        let address = listener.local_addr().unwrap();
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let task = tokio::spawn(serve_listener(listener, test_runtime(&temp), async move {
            let _ = shutdown_rx.await;
        }));

        let malformed = request(address, "POST /api/chat", "{").await;
        assert!(malformed.starts_with("HTTP/1.1 400 Bad Request"));
        assert!(malformed.contains("\"error\""));

        let missing = request(
            address,
            "POST /api/chat",
            r#"{"model":"absent","messages":[{"role":"user","content":"hello"}],"stream":false}"#,
        )
        .await;
        assert!(missing.starts_with("HTTP/1.1 404 Not Found"));
        assert!(missing.contains(r#"model \"absent\" was not found"#));

        let invalid_session = request(
            address,
            "POST /api/chat",
            r#"{"model":"absent","session":"../escape","messages":[{"role":"user","content":"hello"}]}"#,
        )
        .await;
        assert!(invalid_session.starts_with("HTTP/1.1 400 Bad Request"));
        assert!(invalid_session.contains("session ID must use"));

        shutdown_tx.send(()).unwrap();
        timeout(Duration::from_secs(2), task)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
    }

    async fn request(address: SocketAddr, request_line: &str, body: &str) -> String {
        let mut stream = TcpStream::connect(address).await.unwrap();
        let request = format!(
            "{request_line} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(request.as_bytes()).await.unwrap();
        let mut response = Vec::new();
        timeout(Duration::from_secs(2), stream.read_to_end(&mut response))
            .await
            .unwrap()
            .unwrap();
        String::from_utf8(response).unwrap()
    }
}
