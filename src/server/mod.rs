//! The pgwire frontend: semcast as a served database. `psql` (or any
//! simple-protocol client) connects, every statement routes through
//! [`crate::sql`], and funnel progress streams back as NOTICE messages
//! while model calls run.

pub mod encode;
pub mod engine;
pub mod handler;
pub mod progress;
pub mod router;

use std::sync::Arc;

use tokio::net::TcpListener;

pub use engine::QueryEngine;
pub use handler::SemcastServer;

/// Accept connections forever, one task per client. Takes a bound listener
/// so callers (and tests) control the address.
pub async fn serve(listener: TcpListener, engine: Arc<QueryEngine>) -> std::io::Result<()> {
    loop {
        let (socket, _) = listener.accept().await?;
        let handler = SemcastServer::new(Arc::clone(&engine));
        tokio::spawn(async move {
            if let Err(error) = pgwire::tokio::process_socket(socket, None, handler).await {
                tracing::error!("semcast: connection error: {error}");
            }
        });
    }
}
