//! `semcast serve` — the pgwire server. Connect with any Postgres
//! simple-protocol client: `psql -h 127.0.0.1 -p 5433`.

use std::path::PathBuf;
use std::sync::Arc;

use clap::{Args, Parser, Subcommand, ValueEnum};
use semcast::SemcastContextBuilder;
use semcast::model::{AnthropicProvider, ModelProvider, OllamaProvider, VoyageProvider};
use semcast::server::{JobRegistry, QueryEngine, jobs, serve};

#[derive(Parser)]
#[command(
    name = "semcast",
    version,
    about = "Semantic SQL, served over the Postgres wire protocol"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Start the server.
    Serve(ServeArgs),
}

#[derive(Args)]
struct ServeArgs {
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    /// 5433 by default so a local Postgres on 5432 keeps working.
    #[arg(long, default_value_t = 5433)]
    port: u16,
    /// Which provider runs completions (MEANS verify, extraction).
    /// `ollama` (the default) runs locally; `anthropic` needs
    /// ANTHROPIC_API_KEY exported.
    #[arg(long, value_enum, default_value_t = CompletionProvider::Ollama)]
    provider: CompletionProvider,
    /// Chat model used to verify MEANS predicates. Defaults per provider:
    /// claude-haiku-4-5 for `anthropic`, gemma4:e4b for `ollama`.
    #[arg(long)]
    model: Option<String>,
    /// Small Ollama chat model that pre-filters a MEANS no semantic index
    /// covers, under WITH RECALL: it scores every row by its P(yes), and the
    /// main model verifies only rows above a floor calibrated to the recall
    /// target. Off if unset.
    #[arg(long)]
    scorer_model: Option<String>,
    /// Ollama embedding model used by semantic indexes
    /// (with `--embed-provider ollama`).
    #[arg(long, default_value = semcast::model::DEFAULT_EMBED_MODEL)]
    embed_model: String,
    #[arg(long, default_value = semcast::model::DEFAULT_OLLAMA_URL)]
    ollama_url: String,
    /// Which provider embeds text for semantic indexes. `ollama` (the
    /// default) runs locally; `voyage` needs VOYAGE_API_KEY exported.
    #[arg(long, value_enum, default_value_t = EmbedProvider::Ollama)]
    embed_provider: EmbedProvider,
    /// Voyage embedding model (with `--embed-provider voyage`).
    #[arg(long, default_value = semcast::model::DEFAULT_VOYAGE_MODEL)]
    voyage_model: String,
    /// No longer used: semantic indexes live in the --store file (or in
    /// memory without one). Accepted so existing scripts keep starting.
    #[arg(long, hide = true)]
    index_dir: Option<PathBuf>,
    /// SQLite file that keeps model verdicts, semantic-index vectors and
    /// definitions (tables, views, semantic types and indexes) across
    /// restarts, created if missing. A
    /// restarted server has the same tables, and a query answered once costs
    /// no model calls. Both live in memory only if unset.
    #[arg(long)]
    store: Option<PathBuf>,
    /// Most model requests in flight at once, across all queries; a separate
    /// embedder gets the same cap of its own. Identical requests in flight
    /// are sent once.
    #[arg(long, default_value_t = semcast::model::DEFAULT_MAX_IN_FLIGHT)]
    max_in_flight: usize,
    /// Where batch job state and results are stored; temp dir if unset.
    /// Jobs submitted with `SUBMIT` survive the connection that started
    /// them, and finished jobs stay queryable across restarts.
    #[arg(long)]
    jobs_dir: Option<PathBuf>,
    /// How many submitted jobs run at once; the rest wait in `queued`.
    /// They share one model backend, so set this to what the provider
    /// serves concurrently — against a single local model, more jobs is
    /// slower, not faster.
    #[arg(long, default_value_t = 4)]
    max_concurrent_jobs: usize,
}

#[derive(Clone, Copy, ValueEnum)]
enum CompletionProvider {
    Anthropic,
    Ollama,
}

#[derive(Clone, Copy, ValueEnum)]
enum EmbedProvider {
    Ollama,
    Voyage,
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    semcast::telemetry::init();
    let Cli {
        command: Command::Serve(args),
    } = Cli::parse();

    let ollama = || {
        OllamaProvider::new(
            args.model
                .as_deref()
                .unwrap_or(semcast::model::DEFAULT_CHAT_MODEL),
        )
        .with_base_url(&args.ollama_url)
        .with_embed_model(&args.embed_model)
    };

    // A missing API key fails here, before the listener binds — not lazily
    // at the first MEANS predicate or CREATE SEMANTIC INDEX.
    let model: Arc<dyn ModelProvider> = match args.provider {
        CompletionProvider::Ollama => Arc::new(ollama()),
        CompletionProvider::Anthropic => {
            let mut anthropic =
                AnthropicProvider::from_env().map_err(|e| std::io::Error::other(e.to_string()))?;
            if let Some(model) = &args.model {
                anthropic = anthropic.with_model(model);
            }
            Arc::new(anthropic)
        }
    };

    let mut builder = SemcastContextBuilder::new(model)
        .with_information_schema(true)
        .with_max_in_flight(args.max_in_flight);
    match args.embed_provider {
        EmbedProvider::Voyage => {
            let voyage = VoyageProvider::from_env()
                .map_err(|e| std::io::Error::other(e.to_string()))?
                .with_model(&args.voyage_model);
            builder = builder.with_embedder(Arc::new(voyage));
        }
        // With an Ollama completion provider the session model already
        // embeds; Anthropic has no embedding models, so hand the builder a
        // dedicated Ollama embedder.
        EmbedProvider::Ollama => {
            if let CompletionProvider::Anthropic = args.provider {
                builder = builder.with_embedder(Arc::new(ollama()));
            }
        }
    }
    if let Some(scorer) = &args.scorer_model {
        builder = builder.with_scorer(Arc::new(
            OllamaProvider::new(scorer).with_base_url(&args.ollama_url),
        ));
    }
    if args.index_dir.is_some() {
        eprintln!(
            "warning: --index-dir is no longer used; semantic indexes are kept in the \
             --store file, or in memory without one"
        );
    }
    // Kept beside the context so shutdown can flush it: connections hold the
    // session, so the cache is never dropped while the server runs.
    let mut store = None;
    if let Some(path) = &args.store {
        let opened = semcast::cache::sqlite::SqliteCache::open(path).map_err(|error| {
            std::io::Error::other(format!("opening {}: {error}", path.display()))
        })?;
        let opened = Arc::new(opened);
        builder = builder.with_cache(Arc::clone(&opened) as Arc<dyn semcast::cache::SemanticCache>);
        let chunks = semcast::index::stored::SqliteChunkStore::open(path).map_err(|error| {
            std::io::Error::other(format!("opening {}: {error}", path.display()))
        })?;
        builder = builder.with_chunk_store(Arc::new(chunks));
        store = Some(opened);
        let catalog = semcast::catalog::Catalog::open(path).map_err(|error| {
            std::io::Error::other(format!("opening {}: {error}", path.display()))
        })?;
        builder = builder.with_catalog(catalog);
    }
    let ctx = Arc::new(builder.build());

    // Jobs must be registered on the built context: `enable_url_table()`
    // rebuilds it, so anything registered inside the builder would be lost.
    let jobs_dir = args
        .jobs_dir
        .clone()
        .unwrap_or_else(|| std::env::temp_dir().join("semcast-jobs"));
    let jobs = Arc::new(JobRegistry::new(&jobs_dir, args.max_concurrent_jobs)?);
    let recovered = jobs.recover()?;
    jobs::register(&ctx, &jobs).map_err(|e| std::io::Error::other(e.to_string()))?;
    tracing::info!(
        "semcast: batch jobs in {} ({recovered} recovered, {} at a time)",
        jobs_dir.display(),
        args.max_concurrent_jobs,
    );

    // After everything is registered, so a saved view may read any of it.
    let restored = semcast::restore_catalog(&ctx)
        .await
        .map_err(|e| std::io::Error::other(e.to_string()))?;
    if args.store.is_some() {
        tracing::info!(
            "semcast: restored {} definitions from the store ({} could not be restored)",
            restored.restored,
            restored.failed.len(),
        );
    }

    let engine = Arc::new(QueryEngine::new(ctx).with_jobs(jobs));

    let listener = tokio::net::TcpListener::bind((args.host.as_str(), args.port)).await?;
    tracing::info!(
        "semcast: listening on {} — connect with: psql -h {} -p {}",
        listener.local_addr()?,
        args.host,
        args.port,
    );
    tokio::select! {
        result = serve(listener, engine) => result,
        () = shutdown_signal() => {
            tracing::info!("semcast: shutting down");
            if let Some(store) = store {
                // Commit verdicts still queued, so a query answered just
                // before shutdown is free after the restart too.
                tokio::task::spawn_blocking(move || store.flush())
                    .await
                    .map_err(std::io::Error::other)?;
            }
            Ok(())
        }
    }
}

/// Ctrl-C, or SIGTERM where there is one (what a service manager sends).
async fn shutdown_signal() {
    let interrupt = async {
        if tokio::signal::ctrl_c().await.is_err() {
            std::future::pending::<()>().await;
        }
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut terminate) => {
                terminate.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        () = interrupt => {}
        () = terminate => {}
    }
}
