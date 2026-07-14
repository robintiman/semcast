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
</div>

```sql
SELECT * FROM reviews
WHERE review MEANS 'disappointed after an update'; 
```

<div align="center">
  <p align="center">
    A semantic SQL query engine you connect to with any Postgres client.
    <br />
    <br />
    <a href="https://github.com/robintiman/semcast/issues/new?labels=bug">Report Bug</a>
    &middot;
    <a href="https://github.com/robintiman/semcast/issues/new?labels=enhancement">Request Feature</a>
  </p>
</div>

## About

Semcast is a semantic SQL query engine served over the Postgres wire
protocol. It adds meaning to SQL: filter text by what it says, extract
typed fields from it, index it by similarity.

The LLM lives inside the query planner, so a model call is an operator —
prunable, reorderable, and cacheable like any other.

## Getting Started

### Prerequisites

Pick a provider:

* **Anthropic** (default for completions) — `export ANTHROPIC_API_KEY=...`;
  defaults to Haiku, the right tier for one-word verify calls. No embeddings,
  so pair with Ollama or Voyage to index.
* **Voyage** (default for embeddings) — `export VOYAGE_API_KEY=...`; hosted
  embeddings for the semantic index.
* **Ollama** (local, free) — `ollama pull gemma4:e4b` plus `nomic-embed-text`.
  Select with `--provider ollama --embed-provider ollama`.

### Installation

Prebuilt binary (Linux x86_64, macOS Apple Silicon), no Rust toolchain needed:

```sh
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/robintiman/semcast/releases/latest/download/semcast-installer.sh | sh
```

Or download a tarball from the
[latest release](https://github.com/robintiman/semcast/releases/latest).
Then start the server and connect with any Postgres client:

```sh
semcast serve                                             # Anthropic + Voyage
semcast serve --provider ollama --embed-provider ollama   # fully local
psql -h 127.0.0.1 -p 5433
```

`semcast serve --help` lists the knobs: `--port` (5433), `--provider`,
`--model`, `--embed-provider`, `--embed-model`, `--ollama-url`, `--index-dir`.

Build from source:

```sh
git clone https://github.com/robintiman/semcast && cd semcast
cargo run -- serve                                 # build and start the server
cargo test                                         # full suite, no network
cargo test --test live_ollama -- --ignored         # end-to-end against local Ollama
```

## Usage

### SQL reference

Everything semcast adds to SQL. All other SQL is DataFusion, unchanged. See full SQL reference [here](https://datafusion.apache.org/user-guide/sql/index.html).

#### `MEANS`

```sql
<text_column> MEANS '<natural-language condition>'
```

True when a model reading the text would say the condition holds. Allowed
only as a top-level `AND` conjunct of `WHERE`. Any predicates in the same `WHERE` run first, so the model only sees
their survivors.

```sql
SELECT meeting_id, title FROM meetings
WHERE held_at >= CAST('2026-01-01' AS TIMESTAMP)
  AND transcript MEANS 'discussed the launch of offline sync in Atlas';
```

#### `WITH RECALL`

```sql
<statement> WITH RECALL <fraction>
```

Requires a `MEANS`. Calibrates the index-pruning
threshold instead of guessing it. The scan labels a sample of surviving rows and sets the floor so
the given fraction of true matches survive. Without it, thresholds are best-effort.

```sql
SELECT meeting_id FROM meetings
WHERE transcript MEANS 'discussed offline sync'
WITH RECALL 0.9;
```

#### `CREATE SEMANTIC INDEX`

```sql
CREATE SEMANTIC INDEX ON <table>(<column>);
```

Chunks each value, embeds the chunks into a Lance dataset, and gives `MEANS`
a cheap stage: one embedding call per query prunes non-candidates by vector
similarity, and the verify model reads each survivor's top-3 chunks instead
of the whole document. Optional but highly recommended. 

#### `CREATE SEMANTIC TYPE`

```sql
CREATE SEMANTIC TYPE <Name> AS (
  <field> <type> '<doc line>',
  ...,
  TOGETHER ( <field> <type> '<doc>', <field> <type> '<doc>' )
);
```

Names a recurring extraction: fields, types, one doc line each (required).
semcast synthesizes the prompt and constrains decoding.

| Field type | Meaning |
|------------|---------|
| `TEXT` | prose |
| `INT`, `BOOL` | plain values |
| `REAL` | number; `CHECK (a..b)` validates at decode time |
| `ONEOF(a, b, c)` | closed category; `GROUP BY`-able |
| `LEVEL(a, b, c)` | ordered low→high |
| `T[]` | list of any of the above |

Fields are extracted independently. `TOGETHER` groups (two or more fields) are generated in
one shot, never pruned apart.

#### `CAST` and `EXTRACT`

```sql
CAST(<text_column> AS <SemanticType>)          -- all fields, as a struct
CAST(<text_column> AS <SemanticType>).<field>  -- one field
EXTRACT(<field> <type> '<doc>' FROM <text_column>)  -- one-off, no type needed
```

Run the extraction a semantic type describes; `EXTRACT` inlines a single
field without declaring a type first. Allowed in the `SELECT` list only.
Standard `EXTRACT(YEAR FROM ts)` is untouched.

```sql
SELECT CAST(transcript AS MeetingFacts).launch_stage FROM meetings;
SELECT EXTRACT(products TEXT[] 'product names discussed' FROM transcript) FROM meetings;
```

### Examples

**Ingest** 

Local Parquet and CSV, mount them as tables, or materialize into memory. Paths resolve on
the server. Object storage (s3) is on the roadmap.

```sql
SELECT * FROM 'data/meetings.parquet';
SELECT count(*) FROM 'data/part-*.parquet';

CREATE EXTERNAL TABLE meetings STORED AS CSV LOCATION 'data/meetings.csv'
  OPTIONS ('format.delimiter' ';');            -- header on by default

CREATE TABLE mem AS SELECT * FROM 'data/meetings.csv';
```

**Index**

```sql
CREATE SEMANTIC INDEX ON meetings(transcript);
```

**Query** 

The date filter runs free, the index prunes by similarity, the
model verifies the few survivors.

```sql
SELECT meeting_id, title FROM meetings
WHERE held_at >= CAST('2026-01-01' AS TIMESTAMP)
  AND transcript MEANS 'discussed the launch of offline sync in Atlas'
WITH RECALL 0.9;
```

Follow up with typed extraction — the `MEANS` verdicts are already cached,
so the model runs only to extract from the survivors:

```sql
CREATE SEMANTIC TYPE MeetingFacts AS (
  products  TEXT[] 'product names discussed in this meeting',
  decisions TEXT[] 'concrete decisions that were made',
  TOGETHER (
    launch_stage ONEOF(none, idea, planned, scheduled, shipped)
                       'the furthest launch stage discussed',
    stage_quote  TEXT  'the transcript line that shows that stage'
  )
);

SELECT meeting_id, CAST(transcript AS MeetingFacts).launch_stage AS stage
FROM meetings
WHERE transcript MEANS 'discussed the launch of offline sync in Atlas';
```

**Persist** 

Save results as tables or files:

```sql
CREATE TABLE launches AS
  SELECT meeting_id, title FROM meetings
  WHERE transcript MEANS 'discussed the launch of offline sync in Atlas';

COPY launches TO 'out/launches.parquet' STORED AS PARQUET;
```

## Roadmap

- [x] `MEANS` — planner-integrated semantic predicate with batched verify
- [x] `CREATE SEMANTIC INDEX` on Lance — vector pre-filter stage
- [x] `WITH RECALL` — sampled threshold calibration
- [x] Semantic types — `CREATE SEMANTIC TYPE`, `CAST`/`EXTRACT`, constrained decoding, field pushdown
- [x] pgwire server (simple protocol) with funnel progress as NOTICE messages
- [x] Ingestion — Parquet/CSV on disk: path-literal `SELECT`, `CREATE EXTERNAL TABLE`, CTAS, `COPY TO`
- [ ] Persistent, cross-session verdict cache
- [ ] Eval harness — calls saved and recall vs. the LLM-on-every-row baseline
- [ ] Extended protocol (DBeaver, Grafana, JDBC)
- [ ] Object storage (s3)
- [ ] Classify / rank / cluster — semantic `CASE`, `ORDER BY … RELEVANCE TO … LIMIT k`, `GROUP BY MEANING OF`, `SEMANTIC DISTINCT`
- [ ] `CREATE SEMANTIC PREDICATE` — reusable templates with `CHEAP USING` shortcuts
- [ ] Nested semantic types, `LEVEL` ordering semantics
- [ ] `BUDGET` — hard cost caps per query

Want something moved up? [Open an issue](https://github.com/robintiman/semcast/issues).

## Contributing

Feedback is as valuable as code. What are you querying? What syntax did you
expect that didn't work? [Bug reports][bug-url], [feature
requests][feature-url], and real usage examples all shape the roadmap.

Code contributions:

1. Fork, branch (`git checkout -b feature/thing`)
2. `cargo test`
3. Open a pull request

## License

Distributed under the Apache-2.0 License. See `LICENSE` for more information.

## Contact

Robin Timan — robintiman@gmail.com

Project Link: [https://github.com/robintiman/semcast](https://github.com/robintiman/semcast)

## Acknowledgments

* [LOTUS](https://github.com/lotus-data/lotus) — pioneered the calibrated-cascade technique
* [Apache DataFusion](https://datafusion.apache.org/) — the query engine semcast extends
* [Lance](https://lancedb.github.io/lance/) — backs the semantic index

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
[bug-url]: https://github.com/robintiman/semcast/issues/new?labels=bug
[feature-url]: https://github.com/robintiman/semcast/issues/new?labels=enhancement
