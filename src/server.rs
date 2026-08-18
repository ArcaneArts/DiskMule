use std::{
    future::Future,
    io,
    net::{Ipv4Addr, SocketAddr, SocketAddrV4},
};

use axum::{Json, Router, routing::get};
use serde::Serialize;
use tokio::net::TcpListener;

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
}

#[derive(Debug, Serialize)]
struct Health {
    status: &'static str,
    service: &'static str,
    version: &'static str,
}

pub async fn serve(
    address: SocketAddr,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<(), ServerError> {
    let listener = TcpListener::bind(address)
        .await
        .map_err(|source| ServerError::Bind { address, source })?;
    serve_listener(listener, shutdown).await
}

pub async fn serve_listener(
    listener: TcpListener,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<(), ServerError> {
    let address = listener.local_addr().map_err(|source| ServerError::Bind {
        address: DEFAULT_BIND,
        source,
    })?;
    tracing::info!(%address, "DiskMule server listening");
    axum::serve(listener, router())
        .with_graceful_shutdown(shutdown)
        .await
        .map_err(ServerError::Serve)
}

fn router() -> Router {
    Router::new().route("/health", get(health))
}

async fn health() -> Json<Health> {
    Json(Health {
        status: "ok",
        service: "diskmule",
        version: env!("CARGO_PKG_VERSION"),
    })
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

    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::{TcpListener, TcpStream},
        sync::oneshot,
        time::{Duration, timeout},
    };

    use super::serve_listener;

    #[tokio::test]
    async fn health_endpoint_is_json_and_server_stops_gracefully() {
        let listener = TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0)))
            .await
            .unwrap();
        let address = listener.local_addr().unwrap();
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let task = tokio::spawn(serve_listener(listener, async move {
            let _ = shutdown_rx.await;
        }));

        let mut stream = TcpStream::connect(address).await.unwrap();
        stream
            .write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            .await
            .unwrap();
        let mut response = Vec::new();
        timeout(Duration::from_secs(2), stream.read_to_end(&mut response))
            .await
            .unwrap()
            .unwrap();
        let response = String::from_utf8(response).unwrap();
        assert!(response.starts_with("HTTP/1.1 200 OK"));
        assert!(response.contains("content-type: application/json"));
        assert!(response.contains(r#"{"status":"ok","service":"diskmule""#));

        shutdown_tx.send(()).unwrap();
        timeout(Duration::from_secs(2), task)
            .await
            .expect("server should stop after shutdown signal")
            .expect("server task should not panic")
            .expect("server should shut down cleanly");
    }
}
