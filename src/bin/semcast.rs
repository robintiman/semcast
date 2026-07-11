//! `semcast serve` — the pgwire server. Connect with any Postgres
//! simple-protocol client: `psql -h 127.0.0.1 -p 5433`.

use std::path::PathBuf;
use std::sync::Arc;

use clap::{Args, Parser, Subcommand, ValueEnum};
use semcast::SemcastContextBuilder;
use semcast::model::{ModelProvider, OllamaProvider, VoyageProvider};
use semcast::server::{QueryEngine, serve};

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
    /// Ollama chat model used to verify MEANS predicates.
    #[arg(long, default_value = "gemma4:e4b")]
    model: String,
    /// Ollama embedding model used by semantic indexes.
    #[arg(long, default_value = semcast::model::DEFAULT_EMBED_MODEL)]
    embed_model: String,
    #[arg(long, default_value = semcast::model::DEFAULT_OLLAMA_URL)]
    ollama_url: String,
    /// Which provider embeds text for semantic indexes. `voyage` needs
    /// VOYAGE_API_KEY exported; completions stay on Ollama.
    #[arg(long, value_enum, default_value_t = EmbedProvider::Ollama)]
    embed_provider: EmbedProvider,
    /// Voyage embedding model (with `--embed-provider voyage`).
    #[arg(long, default_value = semcast::model::DEFAULT_VOYAGE_MODEL)]
    voyage_model: String,
    /// Where semantic indexes are stored; temp dir if unset.
    #[arg(long)]
    index_dir: Option<PathBuf>,
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

    let model: Arc<dyn ModelProvider> = Arc::new(
        OllamaProvider::new(&args.model)
            .with_base_url(&args.ollama_url)
            .with_embed_model(&args.embed_model),
    );

    let mut builder = SemcastContextBuilder::new(model).with_information_schema(true);
    // A missing VOYAGE_API_KEY fails here, before the listener binds — not
    // lazily at the first CREATE SEMANTIC INDEX.
    if let EmbedProvider::Voyage = args.embed_provider {
        let voyage = VoyageProvider::from_env()
            .map_err(|e| std::io::Error::other(e.to_string()))?
            .with_model(&args.voyage_model);
        builder = builder.with_embedder(Arc::new(voyage));
    }
    if let Some(dir) = &args.index_dir {
        builder = builder.with_index_root(dir);
    }
    let engine = Arc::new(QueryEngine::new(Arc::new(builder.build())));

    let listener = tokio::net::TcpListener::bind((args.host.as_str(), args.port)).await?;
    tracing::info!(
        "semcast: listening on {} — connect with: psql -h {} -p {}",
        listener.local_addr()?,
        args.host,
        args.port,
    );
    serve(listener, engine).await
}
