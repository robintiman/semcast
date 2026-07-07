# semcast

Planner-integrated semantic operators for [Apache DataFusion](https://datafusion.apache.org/).

LLM calls today live in the application layer (Marvin, BAML) or behind opaque
SQL functions (FlockMTL, `ai_query`, Cortex) — invisible to the query
optimizer, so it can't make them cheaper. **semcast puts the model call inside
the planner as a first-class operator** DataFusion can prune, reorder, and
cache.

Closest relative: [LOTUS](https://github.com/lotus-data/lotus), which pioneered
semantic operators with accuracy guarantees as a Python dataframe library.
semcast bets the same ideas belong *inside a SQL planner*, where they compose
with ordinary relations, indexes, and every other optimizer rule for free.

## The idea

One operator — `text MEANS 'a natural-language condition'` — and the planner
builds the cheapest plan that still answers the question. You declare intent
and an accuracy target; the funnel is derived, never hand-written.

- **Cheap-then-verify** — pre-filter on free signals (structured columns, a
  semantic index), spend the LLM only on survivors, under a declared recall
  target.
- **Field-level caching** — pay once per `(type, field, value, model, prompt
  version)`, shared across every query that ever asks again.
- **Agg split** — quantitative rollups run in SQL; the model touches only free text.
- **Field pushdown** — generate only the fields the query uses: finer cache
  keys, per-field routing (`BOOL` to a small model, `TEXT` to a strong one).

Net effect: a semantic query over ~20k documents costs tens of model calls —
not tens of thousands reading everything.

---

## Walk-through (design target)

This section is the design the roadmap builds toward; what runs today is under
[Getting started](#getting-started).

*"Which meetings in the last 6 months discussed launching offline sync in
Atlas?"* — 18,400 transcripts, ad-hoc question, no pipeline.

```sql
-- meetings(meeting_id, title, held_at, attendees, transcript) — your existing table
CREATE SEMANTIC INDEX ON meetings(transcript);

SELECT meeting_id, title, held_at
FROM meetings
WHERE held_at >= now() - INTERVAL '6 months'
  AND transcript MEANS 'discussed the launch of offline sync in Atlas'
WITH RECALL 0.9;
```

The index chunks each transcript (~512-token slices), embeds every chunk, and
stays fresh incrementally. It's optional — without it the planner warns the
plan has no cheap stage. `MEANS` is ground truth: *a model reading the full
transcript would say yes*; everything cheaper is an approximation managed
under your recall target. `EXPLAIN` shows the derived funnel:

```text
SemFilter: MEANS('discussed the launch of offline sync in Atlas')   recall ≥ 0.90
├─ held_at filter        18,400 → 3,100 rows    $0     plain SQL, runs first
├─ semantic index scan    3,100 →    47 rows    $0     threshold set by calibration
└─ verify (small model)      47 calls         ~$0.04   reads top-3 chunks per meeting
```

**47 model calls, not 18,400 — none reading a whole transcript.** Because:

- To the optimizer, `MEANS` is simply a very expensive predicate; the free
  date filter runs first. Predicate ordering is just predicate ordering.
- The chunk vectors that pre-filter 3,100 → 47 also pick which three chunks
  the verify model reads.
- The index scan has false negatives, so `WITH RECALL` calibrates: sample
  date-surviving rows, get ground-truth labels, set the threshold so ≥90% of
  true matches survive (the cascade technique pioneered by LOTUS). Omit the
  clause and thresholds are best-effort — `EXPLAIN` says so.

Follow-ups reuse cached verdicts — the filter below costs zero new calls; the
model runs only to extract from the ~12 survivors:

```sql
SELECT held_at,
       EXTRACT(decisions TEXT[] 'concrete decisions made' FROM transcript) AS decisions
FROM meetings
WHERE transcript MEANS 'discussed the launch of offline sync in Atlas';
```

Recurring questions become macros; next month's variant reuses every embedding
for free:

```sql
CREATE SEMANTIC PREDICATE discussed_launch(t, feature, product) AS
  t MEANS 'discussed the launch of {feature} in {product}';
```

### Know the bill

`EXPLAIN` prices every stage — calls and dollars — before a token is spent,
from cached history or a sampled slice. Estimates are estimates, so queries
take a hard cap:

```sql
SELECT ... BUDGET 1.00 USD;   -- stop at the cap; partial rows + a warning
```

### Typed extraction

A **semantic type** names a recurring extraction: field names, types, one doc
line each. semcast synthesizes the prompt, constrains decoding, and turns as
much of a downstream aggregate as possible into plain SQL.

```sql
CREATE SEMANTIC TYPE MeetingFacts AS (
  products  TEXT[]   'product names discussed in this meeting',
  decisions TEXT[]   'concrete decisions that were made',
  TOGETHER (                               -- co-generated: a stage needs its evidence
    launch_stage ONEOF(none, idea, planned, scheduled, shipped)
                       'the furthest launch stage discussed',
    stage_quote  TEXT  'the transcript line that shows that stage'
  )
);

SELECT CAST(transcript AS MeetingFacts).launch_stage FROM meetings WHERE ...;
```

| Field type | Meaning |
|------------|---------|
| `TEXT` | prose; stays with the model |
| `INT`, `REAL` | aggregate in SQL — no LLM at rollup |
| `REAL CHECK (a..b)` | validated at decode time |
| `BOOL` | becomes a plain predicate |
| `ONEOF(a, b, c)` | closed category; `GROUP BY`-able |
| `LEVEL(a, b, c)` | ordered low→high; comparable, rankable |
| `T[]` | list of any above |
| `<AnotherType>` | nested semantic type |

Fields are independent by default — that's what enables pushdown. `TOGETHER`
groups are generated in one shot, never pruned apart.

### Operators

| Kind | Signature | Example |
|------|-----------|---------|
| **Predicate** | `text → bool` | `transcript MEANS 'discussed a launch'` |
| **Map** | `text → typed fields` | `EXTRACT(...)`, `CAST(... AS MeetingFacts)` |
| **Aggregate** | `set<text> → summary` | `sem_summary(transcript, 'recurring blockers')` |
| **Join** | `text × text → bool` | `ON (c.notes, v.description) MEANS 'the same company'` |

All plan the same way: cheap stage, calibrated threshold, verify, cache. Joins
block on vector proximity first — never O(n×m) model calls. For shortcuts the
planner can't guess, `CHEAP USING <expr>` pins one on a named predicate:

```sql
CREATE SEMANTIC PREDICATE discussed_launch(t, feature, product) AS
  t MEANS 'discussed the launch of {feature} in {product}'
  CHEAP USING attendees @> ARRAY['launch-committee'];
```

---

## Execution semantics

LLMs break two database assumptions — determinism, and evaluation that doesn't
fail halfway. semcast answers explicitly:

- **Full-provenance cache keys** — `(type version, field, input value, model,
  prompt version)`. Editing one field's doc line invalidates exactly that field.
- **First evaluation wins** — re-running a query is deterministic even though
  the model isn't.
- **Rows fail, queries don't** — a row that errors after retries yields `NULL`
  plus an error column. The cache doubles as a checkpoint for resumed jobs.

## Architecture

| Piece | DataFusion hook | Status |
|-------|-----------------|--------|
| `MEANS` logical operator | `UserDefinedLogicalNodeCore` → `LogicalPlan::Extension` | ✅ `SemFilter` |
| Infix `text MEANS '...'` syntax | custom sqlparser `Dialect` | ✅ via `semcast::sql` |
| `means()` → `SemFilter` rewrite | `OptimizerRule` | ✅ |
| Verify stage + call estimate | `ExecutionPlan` via `ExtensionPlanner` | ✅ `VerifyExec` |
| Async batched model calls | `tokio` + `reqwest` | ✅ Ollama, Anthropic |
| Verdict cache | provenance-keyed, in-memory | ✅ |
| `CREATE SEMANTIC INDEX` syntax | parser extension | ✅ (`TYPE` / `PREDICATE` planned) |
| Semantic index + pre-filter stage | [Lance](https://lancedb.github.io/lance/) (Arrow-native) | ✅ `IndexScanExec` |
| Calibration, field pushdown, agg split | `OptimizerRule` / `PhysicalOptimizerRule` | planned |

## Where this bites

Large corpus, expensive per-document review, ad-hoc questions — index and
cache compound across queries.

- **eDiscovery / legal review** — a recall target is defensibility, not a nicety.
- **Contract analytics** — extract typed fields once, `SUM` exposure in SQL.
- **Literature screening** — `is_survey BOOL`, `audience LEVEL`, filter.
- **Support-ticket mining** — classify once, trend forever; new questions
  re-slice the cache.
- **Entity resolution** — semantic join without n×m model calls.

## Getting started

Not on crates.io yet — depend on it from git:

```toml
[dependencies]
semcast    = { git = "https://github.com/robintiman/semcast" }
datafusion = "54"
tokio      = { version = "1", features = ["rt-multi-thread", "macros"] }
```

Building needs `protoc` (Lance requirement): `brew install protobuf`.

Pick a provider: **Ollama** (local, free — `ollama pull gemma4:31b`, plus
`nomic-embed-text` for the semantic index) or **Anthropic**
(`export ANTHROPIC_API_KEY=...`; defaults to Haiku, the right tier for
one-word verify calls — no embeddings, so bring an Ollama embedder in
`IndexOptions` to index).

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

Infix `MEANS` needs `semcast::sql` (DataFusion's `ctx.sql` can't take a custom
dialect); through `ctx.sql`, write it as `means(text, 'condition')`. Either
way it's allowed in top-level `AND` conjuncts of `WHERE` only; anything else
(`OR`, `NOT`, the `SELECT` list) fails at plan time rather than silently
costing a call per row. `WITH RECALL` isn't parsed yet.

What runs today: `MEANS` rewrites to a `SemFilter` above your free
predicates (so they run first), survivors are verified with batched async
calls, and verdicts are cached by provenance — reruns and narrower
follow-ups cost zero new calls. With an index (the DDL above, or
`create_semantic_index(...)` from Rust), the planner adds the cheap stage:
chunks are embedded once into a Lance dataset, one embedding call per query
prunes non-candidates by vector similarity, and the verify model reads each
survivor's top-3 chunks instead of the whole document. Rows the index has
never seen pass through to full-text verify — never silently dropped;
`refresh_semantic_index` picks them up. Thresholds are best-effort until
`WITH RECALL` lands, and `EXPLAIN` says so:

```text
VerifyExec: MEANS('discussed the launch of offline sync in Atlas') model=ollama/gemma4:31b reads top-3 chunks per doc   ~3 model calls
  IndexScanExec: MEANS('discussed the launch of offline sync in Atlas') embed_model=ollama/gemma4:31b floor=0.35 top-3 chunks (threshold best-effort — no WITH RECALL)
```

Try it with no setup:

```sh
git clone https://github.com/robintiman/semcast && cd semcast
cargo run --example meetings                       # deterministic mock model
cargo test                                         # full suite, no network
cargo test --test live_ollama -- --ignored         # end-to-end against local Ollama
                                                   # (gemma4:31b + nomic-embed-text)
```

## Status

Early / experimental. Order of attack:

1. ~~`MEANS` logical operator + verify-only physical plan~~ **done**
2. ~~`CREATE SEMANTIC INDEX` on Lance + the index pre-filter stage~~ **done**
3. `WITH RECALL` — sampled threshold calibration — **next**
4. Field-level cache with provenance keys — **in-memory done**; persistent,
   cross-session cache on disk
5. Eval harness: labeled corpus, reporting **calls saved and recall** against
   the LLM-on-every-row baseline — so the headline claim stays falsifiable

## License

Apache-2.0
