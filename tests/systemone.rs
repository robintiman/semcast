//! `SystemOneProvider` over the wire: a real provider against a throwaway
//! local server speaking the two endpoints it uses.

use semcast::model::{CompletionRequest, ModelProvider, SystemOneProvider, YesNoAsk};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// Read one HTTP request (head and body) and return it as text.
async fn read_request(socket: &mut TcpStream) -> String {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        let n = socket.read(&mut chunk).await.expect("read");
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
        let text = String::from_utf8_lossy(&buf);
        let Some(head_end) = text.find("\r\n\r\n") else {
            continue;
        };
        let length = text[..head_end]
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().ok())?
            })
            .unwrap_or(0);
        if buf.len() >= head_end + 4 + length {
            break;
        }
    }
    String::from_utf8_lossy(&buf).into_owned()
}

/// Serve `requests` connections: the model list, a yes for the angry
/// ticket, a no for anything else, and a 400 for the "broken" one.
async fn serve(listener: TcpListener, requests: usize) {
    for _ in 0..requests {
        let (mut socket, _) = listener.accept().await.expect("accept");
        let request = read_request(&mut socket).await;
        let (status, body) = if request.starts_with("GET /v1/models") {
            (
                "200 OK",
                r#"{"object":"list","data":[{"id":"models/span-02"}]}"#,
            )
        } else if !request.starts_with("POST /v1/systemone") {
            ("404 Not Found", "{}")
        } else if request.contains("broken") {
            ("400 Bad Request", r#"{"error":{"message":"bad state"}}"#)
        } else if request.contains("swapped") {
            // The server was restarted with another model.
            (
                "200 OK",
                r#"{"model":"models/other","answers":{"q":{"type":"noul","noul":0.9}},"usage":{}}"#,
            )
        } else if request.contains("unacceptable") {
            (
                "200 OK",
                r#"{"model":"models/span-02","answers":{"q":{"type":"noul","noul":0.97}},"usage":{"input_tokens":40,"output_tokens":0}}"#,
            )
        } else {
            (
                "200 OK",
                r#"{"answers":{"q":{"type":"noul","noul":0.02}},"usage":{"input_tokens":30,"output_tokens":0}}"#,
            )
        };
        let response = format!(
            "HTTP/1.1 {status}\r\ncontent-type: application/json\r\n\
             content-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        );
        socket.write_all(response.as_bytes()).await.expect("write");
        let _ = socket.shutdown().await;
    }
}

fn ask(input: &str) -> CompletionRequest {
    CompletionRequest {
        system: "unused by a decision model".to_owned(),
        input: input.to_owned(),
        max_tokens: 8,
        schema: None,
        logprobs: true,
        ask: Some(YesNoAsk {
            condition: "the customer is angry".to_owned(),
            excerpts: false,
        }),
    }
}

#[tokio::test]
async fn connect_names_the_served_model_and_answers_with_probabilities() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let url = format!("http://{}", listener.local_addr().expect("addr"));
    let server = tokio::spawn(serve(listener, 5));

    let provider = SystemOneProvider::connect(&url, None)
        .await
        .expect("connect");
    // Named by the file, not the path the server was started with.
    assert_eq!(provider.id().0, "systemone/span-02/ask-v1");

    let answers = provider
        .complete(vec![
            ask("this is unacceptable"),
            ask("thanks, all good"),
            ask("broken"),
            ask("swapped"),
        ])
        .await;
    let yes = answers[0].as_ref().expect("yes");
    assert_eq!(yes.text, "yes");
    assert_eq!(yes.yes_probability, Some(0.97));
    assert_eq!(yes.input_tokens, 40);
    let no = answers[1].as_ref().expect("no");
    assert_eq!(no.text, "no");
    assert_eq!(no.yes_probability, Some(0.02));
    // Rows fail, queries don't: the bad row is an error of its own.
    assert!(answers[2].is_err());
    // Another model's answer is not stored under this one's name.
    let swapped = answers[3].as_ref().expect_err("swapped");
    assert!(swapped.to_string().contains("models/other"), "{swapped}");

    server.await.expect("server");
}

#[tokio::test]
async fn connect_fails_when_the_server_serves_no_model() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let url = format!("http://{}", listener.local_addr().expect("addr"));
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept");
        read_request(&mut socket).await;
        let body = r#"{"object":"list","data":[]}"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\
             content-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        );
        socket.write_all(response.as_bytes()).await.expect("write");
        let _ = socket.shutdown().await;
    });
    let error = SystemOneProvider::connect(&url, None)
        .await
        .expect_err("no model");
    assert!(error.to_string().contains("serves no model"), "{error}");
    server.await.expect("server");
}

#[tokio::test]
async fn a_named_model_must_be_served_and_is_sent_with_each_question() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let url = format!("http://{}", listener.local_addr().expect("addr"));
    let server = tokio::spawn(async move {
        // The list four times (four connects), then one question.
        for _ in 0..4 {
            let (mut socket, _) = listener.accept().await.expect("accept");
            read_request(&mut socket).await;
            respond(
                &mut socket,
                r#"{"object":"list","data":[{"id":"models/span-02"},{"id":"d3:latest"}]}"#,
            )
            .await;
        }
        let (mut socket, _) = listener.accept().await.expect("accept");
        let request = read_request(&mut socket).await;
        respond(
            &mut socket,
            r#"{"answers":{"q":{"type":"noul","noul":0.9}},"usage":{}}"#,
        )
        .await;
        request
    });

    let error = SystemOneProvider::connect(&url, Some("clef"))
        .await
        .expect_err("not served");
    assert!(error.to_string().contains("clef"), "{error}");

    // Several models and none named: no guessing.
    let error = SystemOneProvider::connect(&url, None)
        .await
        .expect_err("ambiguous");
    assert!(
        error.to_string().contains("name the decision model"),
        "{error}"
    );

    // The file name semcast reports names the model too.
    let provider = SystemOneProvider::connect(&url, Some("span-02"))
        .await
        .expect("by file name");
    assert_eq!(provider.id().0, "systemone/span-02/ask-v1");

    // Ollama lists `d3` as `d3:latest`.
    let provider = SystemOneProvider::connect(&url, Some("d3"))
        .await
        .expect("connect");
    assert_eq!(provider.id().0, "systemone/d3/ask-v1");
    let answers = provider.complete(vec![ask("anything")]).await;
    assert_eq!(answers[0].as_ref().expect("yes").text, "yes");

    let request = server.await.expect("server");
    assert!(request.contains(r#""model":"d3""#), "{request}");
}

async fn respond(socket: &mut TcpStream, body: &str) {
    let response = format!(
        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\
         content-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    );
    socket.write_all(response.as_bytes()).await.expect("write");
    let _ = socket.shutdown().await;
}
