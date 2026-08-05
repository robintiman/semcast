# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

semcast is a semantic SQL query engine: Apache DataFusion extended with
planner-integrated LLM operators (`MEANS`, `RELEVANCE TO`, `GROUP BY MEANING OF`,
`SEMANTIC DISTINCT ON`, `CAST(... AS <SemanticType>)`), served over the Postgres
wire protocol via pgwire. Single crate, Rust 2024 edition, MSRV 1.85.

`README.md` is the user-facing SQL reference and the roadmap; module doc comments
in `src/` carry the design rationale and are the best source when changing a
subsystem. Read the target module's `//!` header before editing it.

## Commands

```sh
cargo test                                  # full suite: mock model, no network
cargo test --test means                     # one integration test file
cargo test --test means -- name_of_test     # one test
cargo test --test live_ollama -- --ignored  # live, needs a running Ollama
cargo test --test live_voyage -- --ignored  # live, needs VOYAGE_API_KEY

cargo fmt --all --check                     # CI gate
cargo clippy --all-targets -- -D warnings   # CI gate
cargo run -- serve                          # start the server on 127.0.0.1:5433
cargo run --example meetings                # README walk-through against the mock
```

Building needs a system `protoc` (lance's `prost-build`); CI installs it via
`arduino/setup-protoc`. Live tests read `SEMCAST_OLLAMA_MODEL` (default
`gemma4:e4b`), `ANTHROPIC_API_KEY`, `VOYAGE_API_KEY`.

Releases go through `./scripts/release.sh <version>` (try `--dry-run` first) — it
owns both the `Cargo.toml` bump and the tag so the two cannot disagree. Never bump
the version or push a `v*` tag by hand.

## Architecture

Every semantic operator travels the same five-stage path. Adding or changing one
means touching all five layers, one file each:

```
SQL text
  → src/sql/       parse (dialect or AST pass) → a marker scalar UDF
  → src/optimizer/ OptimizerRule rewrites the marker into an extension node
  → src/logical/   UserDefinedLogicalNodeCore, appears as LogicalPlan::Extension
  → src/physical/  ExtensionPlanner maps the node to an ExecutionPlan
  → execution      the only place model calls are spent
```

| Operator | sql/ | optimizer/ | logical/ | physical/ |
|---|---|---|---|---|
| `MEANS` (filter) | `dialect.rs`, `means_udf.rs` | `rewrite.rs` | `sem_filter.rs` | `verify.rs` + `index_scan.rs` |
| `MEANS` (label / `CASE`) | same | `rewrite.rs` → `classify.rs` | `sem_classify.rs` | `classify.rs` |
| `RELEVANCE TO` | `dialect.rs`, `rank.rs`, `rank_udf.rs` | `rank.rs` | `sem_rank.rs` | `rank.rs` |
| `GROUP BY MEANING OF` | `cluster.rs`, `cluster_udf.rs` | `cluster.rs` | `sem_cluster.rs` | `cluster.rs` |
| `SEMANTIC DISTINCT ON` | `distinct.rs`, `distinct_udf.rs` | `distinct.rs` | `sem_distinct.rs` | `distinct.rs` |
| `CAST`/`EXTRACT` typed | `typed.rs`, `extract_udf.rs` | `extract.rs` | `sem_extract.rs` | `extract.rs` |

Wiring lives in `SemcastContextBuilder::build` (`src/lib.rs`): registers the
optimizer rules, the marker UDFs, and `SemcastQueryPlanner`. `enable_url_table()`
must stay the last call — it consumes and rebuilds the context.

### Two parse surfaces, and why `semcast::sql()` exists

`SemcastDialect` (`src/sql/dialect.rs`) handles *infix* syntax (`MEANS`,
`RELEVANCE TO`) by desugaring at parse time into marker UDF calls. Everything
statement-level has no dialect hook and is handled by `semcast::sql()` in
`src/lib.rs`, in order: `CREATE SEMANTIC ...` DDL interception, the `SEMANTIC`
keyword strip for `DISTINCT ON`, trailing `WITH RECALL` / `WITH SIMILARITY`,
`CAST(col AS Type).field` desugaring, relevance sort-direction defaulting, and
`MEANING OF ... AS label` binding.

Consequence: **`ctx.sql()` cannot parse semcast syntax** — always go through
`semcast::sql(&ctx, query)`. Plain SQL still works either way. Any new
statement-level clause belongs in that function, alphabetically after parsing but
before `statement_to_plan`.

### Session state

`SemcastRuntime` (`src/index/registry.rs`) is a `SessionConfig` extension holding
the completion model, the embedder (separate — Anthropic can't embed, Voyage can't
complete), the index root, the `(table, column) → index` map, and the semantic type
registry. Both the public API and the physical planner reach it via
`state.config().get_extension::<SemcastRuntime>()`; that's the plumbing to reuse
rather than threading new parameters through.

### The funnel

For `MEANS`, `SemcastExtensionPlanner` (`src/physical/planner.rs`) builds
`IndexScanExec → VerifyExec` when a semantic index covers the column, and bare
`VerifyExec` otherwise (correct, full price). The two operators share a
planning-time `ChunkEvidence` channel: the scan fills it with each surviving
document's top chunks, verify reads it per row, keyed by `doc_hash` so it survives
repartitioning. `WITH RECALL` makes the scan calibrate its own threshold at
execution time (`src/optimizer/calibrate.rs`) by labeling a sample.

`src/optimizer/funnel.rs::derive_funnel` is an unimplemented stub — the funnel is
assembled in the physical planner, not there.

### Invariants worth not breaking

- `index::doc_hash` is FNV-1a and must never change: index entries outlive the
  process. `DefaultHasher` (the in-memory verdict cache) is only stable within one
  binary and must not leak into the index.
- Cache keys carry a prompt version (`MEANS_PROMPT_VERSION`,
  `EXTRACT_PROMPT_VERSION`, `SemanticType::version`). Change a synthesized prompt
  and bump its constant, so stale verdicts are invalidated honestly.
- Rows fail, queries don't: a row the model errors on falls through
  (`CachedValue::Error`) rather than failing the statement.
- An index gap degrades, never drops: unindexed documents pass the pre-filter
  through to full-text verify; `SEMANTIC DISTINCT` falls back to exact-match
  dedupe. Never silently discard rows the index hasn't seen.
- Illegal marker positions (a `means()` under `OR`/`NOT`, in `GROUP BY`, …) are
  plan-time errors, not silent per-row model calls.

### Server

`src/server/` is the pgwire frontend. `router.rs` splits the simple-protocol
statement string and answers client handshake chatter (`SET`, `BEGIN`,
`SHOW transaction_isolation`) with canned responses — the `SessionContext` is
shared across connections, so `SET` must never reach it. `engine.rs` executes a
statement into memory or Parquet, decoupled from the wire. `progress.rs` walks the
running physical plan's `ExecutionPlan::metrics()` to emit funnel NOTICE lines —
it reads metrics only, no hooks inside operators, so a new operator becomes
visible by exposing counters. `jobs/` implements `SUBMIT`, the `semcast_jobs`
table, and `job_result()`.

## Testing conventions

Integration tests in `tests/` run against `MockModel` (`src/model/mock.rs`):
deterministic, free, offline — answers "yes" when the input contains a configured
substring, embeds by byte histogram, and can serve typed extraction via a JSON
responder. It also counts calls, which is how tests assert the funnel actually
pruned rather than just returned the right rows. Assert on call counts, not only
on result sets.

Live tests are `#[ignore]`d with a reason string and follow the `live_ollama.rs`
convention (env-var model override, doc comment showing the exact command).

Logging: `semcast::telemetry::init()` is idempotent; `RUST_LOG=semcast=debug`
(the default filter) prints prompts and responses, `semcast=info` prints the
optimised plan and stage boundaries only.
