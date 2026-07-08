//! pgwire glue: the simple-protocol handler that splits, routes, and runs
//! statements, pumping engine progress into NOTICE messages while model
//! calls are in flight. All protocol types stay in this file and `mod.rs`.

use std::fmt::Debug;
use std::sync::Arc;

use async_trait::async_trait;
use futures::sink::{Sink, SinkExt};
use pgwire::api::auth::noop::NoopStartupHandler;
use pgwire::api::query::SimpleQueryHandler;
use pgwire::api::results::{Response, Tag};
use pgwire::api::store::PortalStore;
use pgwire::api::{ClientInfo, ClientPortalStore, PgWireServerHandlers};
use pgwire::error::{ErrorInfo, PgWireError, PgWireResult};
use pgwire::messages::PgWireBackendMessage;
use pgwire::messages::response::NoticeResponse;
use tokio::sync::mpsc;

use super::encode;
use super::engine::{QueryEngine, StatementOutcome};
use super::router::{self, Route};

pub struct SemcastHandler {
    engine: Arc<QueryEngine>,
}

impl SemcastHandler {
    pub fn new(engine: Arc<QueryEngine>) -> Self {
        Self { engine }
    }
}

/// Trust auth: connections are accepted as-is. The server binds localhost
/// by default; auth is a follow-up alongside TLS.
impl NoopStartupHandler for SemcastHandler {}

/// Per-connection handler factory handed to `process_socket`.
#[derive(Clone)]
pub struct SemcastServer {
    handler: Arc<SemcastHandler>,
}

impl SemcastServer {
    pub fn new(engine: Arc<QueryEngine>) -> Self {
        Self {
            handler: Arc::new(SemcastHandler::new(engine)),
        }
    }
}

impl PgWireServerHandlers for SemcastServer {
    fn simple_query_handler(&self) -> Arc<impl SimpleQueryHandler> {
        Arc::clone(&self.handler)
    }

    fn startup_handler(&self) -> Arc<impl pgwire::api::auth::StartupHandler> {
        Arc::clone(&self.handler)
    }
}

#[async_trait]
impl SimpleQueryHandler for SemcastHandler {
    async fn do_query<C>(&self, client: &mut C, query: &str) -> PgWireResult<Vec<Response>>
    where
        C: ClientInfo + ClientPortalStore + Sink<PgWireBackendMessage> + Unpin + Send + Sync,
        C::PortalStore: PortalStore,
        C::Error: Debug,
        PgWireError: From<<C as Sink<PgWireBackendMessage>>::Error>,
    {
        let mut responses = Vec::new();
        for statement in router::split_statements(query) {
            match router::classify(statement) {
                Route::NoOp(tag) => responses.push(Response::Execution(Tag::new(tag))),
                Route::CannedTransactionIsolation => {
                    responses.push(encode::canned_response(
                        "transaction_isolation",
                        "read uncommitted",
                    )?);
                }
                Route::CannedServerVersion => {
                    responses.push(encode::canned_response(
                        "server_version",
                        concat!("16.6 (semcast ", env!("CARGO_PKG_VERSION"), ")"),
                    )?);
                }
                Route::Engine => match self.run_statement(client, statement).await? {
                    Ok(response) => responses.push(response),
                    // Postgres aborts the rest of a multi-statement string
                    // on the first error.
                    Err(error) => {
                        responses.push(Response::Error(Box::new(error)));
                        break;
                    }
                },
            }
        }
        if responses.is_empty() {
            responses.push(Response::EmptyQuery);
        }
        Ok(responses)
    }
}

impl SemcastHandler {
    /// Run one engine statement, forwarding progress as NOTICE messages
    /// while it executes. Statement failures come back as `Ok(Err(..))` so
    /// the caller can keep earlier statements' responses.
    async fn run_statement<C>(
        &self,
        client: &mut C,
        statement: &str,
    ) -> PgWireResult<Result<Response, ErrorInfo>>
    where
        C: ClientInfo + Sink<PgWireBackendMessage> + Unpin + Send + Sync,
        C::Error: Debug,
        PgWireError: From<<C as Sink<PgWireBackendMessage>>::Error>,
    {
        let (events, mut progress) = mpsc::channel::<String>(16);
        let mut execution = std::pin::pin!(self.engine.execute_statement(statement, events));
        let outcome = loop {
            tokio::select! {
                biased;
                outcome = &mut execution => break outcome,
                line = progress.recv() => {
                    if let Some(line) = line {
                        // `send`, not `feed`: flush immediately so progress
                        // shows while the model runs.
                        client
                            .send(PgWireBackendMessage::NoticeResponse(notice(&line)))
                            .await?;
                    }
                }
            }
        };
        while let Ok(line) = progress.try_recv() {
            client
                .send(PgWireBackendMessage::NoticeResponse(notice(&line)))
                .await?;
        }
        match outcome {
            Ok(StatementOutcome::Rows { schema, batches }) => {
                Ok(Ok(encode::rows_response(&schema, &batches)?))
            }
            Ok(StatementOutcome::Command { tag }) => Ok(Ok(Response::Execution(Tag::new(&tag)))),
            Err(error) => Ok(Err(user_error(statement, &error))),
        }
    }
}

fn notice(line: &str) -> NoticeResponse {
    ErrorInfo::new("NOTICE".to_owned(), "00000".to_owned(), line.to_owned()).into()
}

fn user_error(statement: &str, error: &crate::SemcastError) -> ErrorInfo {
    let lower = statement.to_ascii_lowercase();
    let message = if lower.contains("pg_catalog") || lower.contains("from pg_") {
        "pg_catalog introspection is not supported yet (psql \\d commands won't work)".to_owned()
    } else {
        error.to_string()
    };
    ErrorInfo::new("ERROR".to_owned(), "XX000".to_owned(), message)
}
