//! The embed path rides through a 429 instead of failing the build.
//!
//! Drives a real `VoyageProvider` against a throwaway local server that
//! throttles the first request (`429` + `Retry-After: 0`) and serves the
//! embedding on the retry — exercising the provider wiring end to end, not just
//! the private backoff helpers (those are unit-tested in `src/model/mod.rs`).

use std::time::Duration;

use semcast::model::{ModelProvider, VoyageProvider};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// Accept one connection, drain the request (best-effort), and write `head`
/// (status line + any extra header lines) followed by `body`.
async fn respond(listener: &TcpListener, head: &str, body: &str) {
    let (mut socket, _) = listener.accept().await.expect("accept");
    let mut buf = vec![0u8; 4096];
    let _ = socket.read(&mut buf).await; // let the client finish sending
    let response = format!(
        "{head}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    );
    socket.write_all(response.as_bytes()).await.expect("write");
    let _ = socket.shutdown().await;
}

#[tokio::test]
async fn embed_retries_after_a_429() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");

    let server = tokio::spawn(async move {
        // First attempt is throttled; Retry-After: 0 keeps the test instant.
        respond(
            &listener,
            "HTTP/1.1 429 Too Many Requests\r\nretry-after: 0",
            "",
        )
        .await;
        // The retry succeeds.
        respond(
            &listener,
            "HTTP/1.1 200 OK\r\ncontent-type: application/json",
            r#"{"data":[{"embedding":[0.1,0.2],"index":0}],"usage":{"total_tokens":1}}"#,
        )
        .await;
    });

    let provider =
        VoyageProvider::new("test-key", "voyage-4-large").with_base_url(format!("http://{addr}"));

    let embeddings = tokio::time::timeout(
        Duration::from_secs(10),
        provider.embed(vec!["hello".to_owned()]),
    )
    .await
    .expect("embed did not hang")
    .expect("embed succeeds after the retry");

    assert_eq!(embeddings, vec![vec![0.1, 0.2]]);
    server.await.expect("server task");
}
