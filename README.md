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
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/semcast-logo-dark.svg">
    <img alt="sem::cast" src="assets/semcast-logo.svg" width="240">
  </picture>

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

Semcast is a query engine built with support for semantic typing and filtering.

The LLM lives inside the query planner, so the model call is prunable, reorderable, and cacheable like any other operator.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* [![Apache DataFusion][DataFusion-badge]][DataFusion-url]
* [![Lance][Lance-badge]][Lance-url]
* [![Tokio][Tokio-badge]][Tokio-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running, follow these steps.

### Prerequisites

Pick a provider:

* **Anthropic** (default for completions) — `export ANTHROPIC_API_KEY=...`;
  defaults to Haiku, the right tier for one-word verify calls. No embeddings,
  so bring an Ollama or Voyage embedder to index.
* **Voyage** (default for embeddings) — `export VOYAGE_API_KEY=...`; hosted
  embeddings for the semantic index, paired with an Ollama or Anthropic model
  for verify calls.
* **Ollama** (local, free) — `ollama pull gemma4:e4b`, plus `nomic-embed-text`
  for the semantic index. Select with `--provider ollama --embed-provider
  ollama`.

### Installation

#### Run the server (prebuilt binary)

Grab a prebuilt `semcast` binary (Linux x86_64, macOS Apple Silicon) — no
Rust toolchain needed:

```sh
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/robintiman/semcast/releases/latest/download/semcast-installer.sh | sh
```

Or download a tarball from the
[latest release](https://github.com/robintiman/semcast/releases/latest).
Then start the server and connect with any Postgres client:

```sh
semcast serve                                      # Anthropic + Voyage by default
semcast serve --provider ollama --embed-provider ollama    # fully local
psql -h 127.0.0.1 -p 5433
```

`semcast serve --help` lists the knobs: `--port` (5433), `--provider`,
`--model`, `--embed-provider`, `--embed-model`, `--ollama-url`, `--index-dir`.

#### Build from source

```sh
git clone https://github.com/robintiman/semcast && cd semcast
cargo run -- serve                                 # build and start the server
cargo test                                         # full suite, no network
cargo test --test live_ollama -- --ignored         # end-to-end against local Ollama
                                                   # (gemma4:e4b + nomic-embed-text)
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

Start the server, connect with any Postgres client, and query with plain
SQL plus the `MEANS` predicate:

```sql
CREATE EXTERNAL TABLE meetings STORED AS CSV LOCATION 'meetings.csv';

-- Optional but what makes it cheap: prunes candidates by vector
-- similarity so the model reads chunks of survivors, not every row.
CREATE SEMANTIC INDEX ON meetings(transcript);

SELECT meeting_id, title FROM meetings
WHERE held_at >= CAST('2026-01-01' AS TIMESTAMP)
  AND transcript MEANS 'discussed the launch of offline sync in Atlas';
```

`MEANS` is allowed in top-level `AND` conjuncts of `WHERE` only; anything
else (`OR`, `NOT`, the `SELECT` list) fails at plan time rather than
silently costing a call per row.

What runs today: `MEANS` rewrites to a `SemFilter` above your free
predicates (so they run first), survivors are verified with batched async
calls, and verdicts are cached by provenance — reruns and narrower
follow-ups cost zero new calls. With an index (the DDL above), the planner
adds the cheap stage: chunks are embedded once into a Lance dataset, one
embedding call per query prunes non-candidates by vector similarity, and
the verify model reads each survivor's top-3 chunks instead of the whole
document. Rows the index has never seen pass through to full-text verify —
never silently dropped; re-running `CREATE SEMANTIC INDEX` rebuilds the
index and picks them up.

Add `WITH RECALL 0.9` and the pruning threshold is calibrated instead of
guessed: the scan labels a sample of surviving rows (≤64 full-text calls,
shared with the verdict cache, so repeat questions relabel for free) and sets
the floor so ≥90% of the sample's true matches survive. Without the clause
thresholds are best-effort, and `EXPLAIN` says which you're getting:

```text
VerifyExec: MEANS('discussed the launch of offline sync in Atlas') model=ollama/gemma4:e4b reads top-3 chunks per doc   ~3 model calls
  IndexScanExec: MEANS('discussed the launch of offline sync in Atlas') embed_model=ollama/gemma4:e4b floor=calibrated(recall≥0.90, sample≤64) top-3 chunks
```

### Serve it

semcast is meant to be run as a service — any Postgres simple-protocol
client connects (`psql` works; DBeaver needs the extended protocol, still on
the roadmap):

```sh
semcast serve                                      # prebuilt binary, Ollama provider
cargo run -- serve                                 # or from a checkout
psql -h 127.0.0.1 -p 5433
```

Funnel progress streams back as NOTICE messages while the model runs:

```text
semcast=> SELECT meeting_id FROM meetings
          WHERE transcript MEANS 'offline sync' WITH RECALL 0.9;
NOTICE:  funnel: IndexScanExec: MEANS('offline sync') embed_model=ollama/nomic-embed-text floor=calibrated(recall≥0.90, sample≤64) top-3 chunks
NOTICE:  funnel: VerifyExec: MEANS('offline sync') model=ollama/gemma4:e4b reads top-3 chunks per doc   ≤47 model calls
NOTICE:  funnel done — index scan: 47 hits, 3053 pruned; verify: 47 model calls, 12 cache hits, 35 dropped
```

`semcast serve --help` lists the knobs: `--port` (5433), `--provider`
(`anthropic` by default, which needs `ANTHROPIC_API_KEY`, or `ollama`),
`--model`, `--embed-model`, `--ollama-url`, `--embed-provider` (`voyage` by
default, which needs `VOYAGE_API_KEY`, or `ollama`), `--voyage-model`,
`--index-dir`. Indexes record
which embedder built them, so switching `--embed-provider` against an existing
`--index-dir` refuses to open the old indexes rather than search them with
mismatched vectors.

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
