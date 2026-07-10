<a id="readme-top"></a>

<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Apache-2.0 License][license-shield]][license-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">SemCast</h3>

  <p align="center">
    Planner-integrated semantic operators for <a href="https://datafusion.apache.org/">Apache DataFusion</a>.
    <br />
    <a href="#getting-started"><strong>Get started »</strong></a>
    <br />
    <br />
    <a href="https://github.com/robintiman/semcast/issues/new?labels=bug">Report Bug</a>
    &middot;
    <a href="https://github.com/robintiman/semcast/issues/new?labels=enhancement">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#the-idea">The idea</a></li>
        <li><a href="#where-this-bites">Where this bites</a></li>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#serve-it">Serve it</a></li>
        <li><a href="#load-data">Load data</a></li>
      </ul>
    </li>
    <li><a href="#execution-semantics">Execution semantics</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

LLM calls today live in the application layer (Marvin, BAML) or behind opaque
SQL functions (FlockMTL, `ai_query`, Cortex) — invisible to the query
optimizer, so it can't make them cheaper. **semcast puts the model call inside
the planner as a first-class operator** DataFusion can prune, reorder, and
cache.

Closest relative: [LOTUS](https://github.com/lotus-data/lotus), which pioneered
semantic operators with accuracy guarantees as a Python dataframe library.
semcast bets the same ideas belong *inside a SQL planner*, where they compose
with ordinary relations, indexes, and every other optimizer rule for free.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### The idea

One operator — `text MEANS 'a natural-language condition'` — and the planner
builds the cheapest plan that still answers the question. You declare intent
and an accuracy target; the funnel is derived, never hand-written.

* **Cheap-then-verify** — pre-filter on free signals (structured columns, a
  semantic index), spend the LLM only on survivors, under a declared recall
  target.
* **Field-level caching** — pay once per `(type, field, value, model, prompt
  version)`, shared across every query that ever asks again.
* **Agg split** — quantitative rollups run in SQL; the model touches only free text.
* **Field pushdown** — generate only the fields the query uses: finer cache
  keys, per-field routing (`BOOL` to a small model, `TEXT` to a strong one).

Net effect: a semantic query over ~20k documents costs tens of model calls —
not tens of thousands reading everything.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Where this bites

Large corpus, expensive per-document review, ad-hoc questions — index and
cache compound across queries.

* **eDiscovery / legal review** — a recall target is defensibility, not a nicety.
* **Contract analytics** — extract typed fields once, `SUM` exposure in SQL.
* **Literature screening** — `is_survey BOOL`, `audience LEVEL`, filter.
* **Support-ticket mining** — classify once, trend forever; new questions
  re-slice the cache.
* **Entity resolution** — semantic join without n×m model calls.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* [![Rust][Rust-badge]][Rust-url]
* [![Apache DataFusion][DataFusion-badge]][DataFusion-url]
* [![Lance][Lance-badge]][Lance-url]
* [![Tokio][Tokio-badge]][Tokio-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

Building needs `protoc` (a Lance requirement):

```sh
brew install protobuf
```

Pick a provider:

* **Ollama** (local, free) — `ollama pull gemma4:31b`, plus `nomic-embed-text`
  for the semantic index.
* **Anthropic** — `export ANTHROPIC_API_KEY=...`; defaults to Haiku, the right
  tier for one-word verify calls. No embeddings, so bring an Ollama embedder in
  `IndexOptions` to index.

### Installation

Not on crates.io yet — depend on it from git:

```toml
[dependencies]
semcast    = { git = "https://github.com/robintiman/semcast" }
datafusion = "54"
tokio      = { version = "1", features = ["rt-multi-thread", "macros"] }
```

Or clone and try it with no setup:

```sh
git clone https://github.com/robintiman/semcast && cd semcast
cargo run --example meetings                       # deterministic mock model
cargo test                                         # full suite, no network
cargo test --test live_ollama -- --ignored         # end-to-end against local Ollama
                                                   # (gemma4:31b + nomic-embed-text)
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

```rust
use std::sync::Arc;
use semcast::{model::OllamaProvider, semcast_context};
// or: semcast::model::AnthropicProvider::from_env()?

#[tokio::main]
async fn main() -> datafusion::error::Result<()> {
    let ctx = semcast_context(Arc::new(OllamaProvider::new("gemma4:31b")));
    ctx.register_csv("meetings", "meetings.csv", Default::default()).await?;

    // Optional but what makes it cheap: prunes candidates by vector
    // similarity so the model reads chunks of survivors, not every row.
    semcast::sql(&ctx, "CREATE SEMANTIC INDEX ON meetings(transcript)").await?;

    semcast::sql(
        &ctx,
        "SELECT meeting_id, title FROM meetings
         WHERE held_at >= CAST('2026-01-01' AS TIMESTAMP)
           AND transcript MEANS 'discussed the launch of offline sync in Atlas'",
    )
    .await?
    .show()
    .await?;

    Ok(())
}
```

Infix `MEANS` and trailing `WITH RECALL` need `semcast::sql` (DataFusion's
`ctx.sql` can't take a custom dialect); through `ctx.sql`, write
`means(text, 'condition', 0.9)` — the optional third argument is the recall
target. Either way `MEANS` is allowed in top-level `AND` conjuncts of `WHERE`
only; anything else (`OR`, `NOT`, the `SELECT` list) fails at plan time
rather than silently costing a call per row.

What runs today: `MEANS` rewrites to a `SemFilter` above your free
predicates (so they run first), survivors are verified with batched async
calls, and verdicts are cached by provenance — reruns and narrower
follow-ups cost zero new calls. With an index (the DDL above, or
`create_semantic_index(...)` from Rust), the planner adds the cheap stage:
chunks are embedded once into a Lance dataset, one embedding call per query
prunes non-candidates by vector similarity, and the verify model reads each
survivor's top-3 chunks instead of the whole document. Rows the index has
never seen pass through to full-text verify — never silently dropped;
`refresh_semantic_index` picks them up.

Add `WITH RECALL 0.9` and the pruning threshold is calibrated instead of
guessed: the scan labels a sample of surviving rows (≤64 full-text calls,
shared with the verdict cache, so repeat questions relabel for free) and sets
the floor so ≥90% of the sample's true matches survive. Without the clause
thresholds are best-effort, and `EXPLAIN` says which you're getting:

```text
VerifyExec: MEANS('discussed the launch of offline sync in Atlas') model=ollama/gemma4:31b reads top-3 chunks per doc   ~3 model calls
  IndexScanExec: MEANS('discussed the launch of offline sync in Atlas') embed_model=ollama/gemma4:31b floor=calibrated(recall≥0.90, sample≤64) top-3 chunks
```

### Serve it

semcast is meant to be run as a service — any Postgres simple-protocol
client connects (`psql` works; DBeaver needs the extended protocol, still on
the roadmap):

```sh
cargo run --features server -- serve               # Ollama provider
cargo run --features server -- serve --mock sync   # no model needed
psql -h 127.0.0.1 -p 5433
```

Funnel progress streams back as NOTICE messages while the model runs:

```text
semcast=> SELECT meeting_id FROM meetings
          WHERE transcript MEANS 'offline sync' WITH RECALL 0.9;
NOTICE:  funnel: IndexScanExec: MEANS('offline sync') embed_model=ollama/nomic-embed-text floor=calibrated(recall≥0.90, sample≤64) top-3 chunks
NOTICE:  funnel: VerifyExec: MEANS('offline sync') model=ollama/gemma4:31b reads top-3 chunks per doc   ≤47 model calls
NOTICE:  funnel done — index scan: 47 hits, 3053 pruned; verify: 47 model calls, 12 cache hits, 35 dropped
```

`semcast serve --help` lists the knobs: `--port` (5433), `--model`,
`--embed-model`, `--ollama-url`, `--index-dir`, `--mock`.

### Load data

Local Parquet and CSV, DuckDB-style — query files by path (globs work),
mount them as tables, or materialize into memory:

```sql
SELECT * FROM 'data/meetings.parquet';
SELECT count(*) FROM 'data/part-*.parquet';

CREATE EXTERNAL TABLE meetings STORED AS CSV LOCATION 'data/meetings.csv'
  OPTIONS ('format.delimiter' ';');            -- header on by default

CREATE TABLE mem AS SELECT * FROM 'data/meetings.csv';
COPY mem TO 'out/meetings.parquet' STORED AS PARQUET;
```

Paths resolve on the server, and any client can read any file the process
can. Object storage (s3) is on the roadmap.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- EXECUTION SEMANTICS -->
## Execution semantics

LLMs break two database assumptions — determinism, and evaluation that doesn't
fail halfway. semcast answers explicitly:

* **Full-provenance cache keys** — `(type version, field, input value, model,
  prompt version)`. Editing one field's doc line invalidates exactly that field.
* **First evaluation wins** — re-running a query is deterministic even though
  the model isn't.
* **Rows fail, queries don't** — a row that errors after retries yields `NULL`
  plus an error column. The cache doubles as a checkpoint for resumed jobs.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to
learn, inspire, and create. Any contributions you make are **greatly
appreciated**.

If you have a suggestion that would make this better, please fork the repo and
create a pull request. You can also simply open an issue with the tag
"enhancement".

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the Apache-2.0 License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Robin Timan — robintiman@gmail.com

Project Link: [https://github.com/robintiman/semcast](https://github.com/robintiman/semcast)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [LOTUS](https://github.com/lotus-data/lotus)
* [Apache DataFusion](https://datafusion.apache.org/)
* [Lance](https://lancedb.github.io/lance/)
* [Best-README-Template](https://github.com/othneildrew/Best-README-Template)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/robintiman/semcast.svg?style=for-the-badge
[contributors-url]: https://github.com/robintiman/semcast/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/robintiman/semcast.svg?style=for-the-badge
[forks-url]: https://github.com/robintiman/semcast/network/members
[stars-shield]: https://img.shields.io/github/stars/robintiman/semcast.svg?style=for-the-badge
[stars-url]: https://github.com/robintiman/semcast/stargazers
[issues-shield]: https://img.shields.io/github/issues/robintiman/semcast.svg?style=for-the-badge
[issues-url]: https://github.com/robintiman/semcast/issues
[license-shield]: https://img.shields.io/github/license/robintiman/semcast.svg?style=for-the-badge
[license-url]: https://github.com/robintiman/semcast/blob/main/LICENSE
[Rust-badge]: https://img.shields.io/badge/Rust-000000?style=for-the-badge&logo=rust&logoColor=white
[Rust-url]: https://www.rust-lang.org/
[DataFusion-badge]: https://img.shields.io/badge/Apache%20DataFusion-E25A1C?style=for-the-badge&logo=apache&logoColor=white
[DataFusion-url]: https://datafusion.apache.org/
[Lance-badge]: https://img.shields.io/badge/Lance-4B8BBE?style=for-the-badge
[Lance-url]: https://lancedb.github.io/lance/
[Tokio-badge]: https://img.shields.io/badge/Tokio-463EE0?style=for-the-badge
[Tokio-url]: https://tokio.rs/
