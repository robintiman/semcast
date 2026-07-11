//! `semcast serve` — the pgwire server. Connect with any Postgres
//! simple-protocol client: `psql -h 127.0.0.1 -p 5433`.

use std::path::PathBuf;
use std::sync::Arc;

use clap::{Args, Parser, Subcommand, ValueEnum};
use semcast::SemcastContextBuilder;
use semcast::model::{AnthropicProvider, ModelProvider, OllamaProvider, VoyageProvider};
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
    /// Which provider runs completions (MEANS verify, extraction).
    /// `anthropic` needs ANTHROPIC_API_KEY exported.
    #[arg(long, value_enum, default_value_t = CompletionProvider::Anthropic)]
    provider: CompletionProvider,
    /// Chat model used to verify MEANS predicates. Defaults per provider:
    /// claude-haiku-4-5 for `anthropic`, gemma4:e4b for `ollama`.
    #[arg(long)]
    model: Option<String>,
    /// Ollama embedding model used by semantic indexes
    /// (with `--embed-provider ollama`).
    #[arg(long, default_value = semcast::model::DEFAULT_EMBED_MODEL)]
    embed_model: String,
    #[arg(long, default_value = semcast::model::DEFAULT_OLLAMA_URL)]
    ollama_url: String,
    /// Which provider embeds text for semantic indexes. `voyage` needs
    /// VOYAGE_API_KEY exported.
    #[arg(long, value_enum, default_value_t = EmbedProvider::Voyage)]
    embed_provider: EmbedProvider,
    /// Voyage embedding model (with `--embed-provider voyage`).
    #[arg(long, default_value = semcast::model::DEFAULT_VOYAGE_MODEL)]
    voyage_model: String,
    /// Where semantic indexes are stored; temp dir if unset.
    #[arg(long)]
    index_dir: Option<PathBuf>,
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

    let mut builder = SemcastContextBuilder::new(model).with_information_schema(true);
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
