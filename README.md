# semcast

Planner-integrated semantic operators for [Apache DataFusion](https://datafusion.apache.org/).

LLM calls today live in the application layer (Marvin, BAML) or as opaque SQL
functions (FlockMTL, Databricks `ai_query`, Snowflake Cortex). Either way the
query optimizer can't see them, so it can't make them cheaper. **semcast puts the
model call inside the planner as a first-class operator** — so DataFusion can
prune, reorder, and cache it like any other.

The closest relative is [LOTUS](https://github.com/lotus-data/lotus), which
pioneered semantic operators with statistical accuracy guarantees — as a Python
dataframe library. semcast bets the same ideas belong *inside a SQL planner*,
where they compose with ordinary relations, indexes, and every other optimizer
rule for free.

## The idea

You write one operator — `text MEANS 'a natural-language condition'` — and the
planner does what planners have always done with expensive predicates: build the
cheapest plan that still answers the question. You declare *intent* and an
*accuracy target*; the funnel is derived, never hand-written.

- **Derived cheap-then-verify** — the planner pre-filters on free signals
  (structured columns, a semantic index) and spends the LLM only on the
  survivors, under a declared recall target — because a lossy pre-filter is an
  approximation, and semcast treats it as one.
- **Field-level caching** — pay once per `(type, field, value, model, prompt
  version)`, shared across every query that ever asks again.
- **Agg split** — quantitative rollups run in SQL; the model touches only free text.
- **Field pushdown** — generate only the fields the query actually uses. The
  honest win here isn't tokens (for documents, input dwarfs output) — it's
  finer-grained cache keys and per-field model routing: a `BOOL` field can go to
  a small model while `TEXT` goes to a strong one.

The net effect: a semantic query over ~20k documents costs tens of model calls —
each reading a few hundred tokens — not tens of thousands reading everything.

---

## Walk-through: "which meetings in the last 6 months discussed launching *offline sync* in *Atlas*?"

The setting: your company transcribes every meeting — 18,400 transcripts and
counting. The question is ad-hoc; next week it's a different feature. Nothing
below is a pipeline built for this one question.

### 0. Your table is just your table

```sql
-- meetings(meeting_id, title, held_at, attendees, transcript)
```

No bespoke ingest, no embedding columns to compute and babysit. semcast works
on tables you already have.

### 1. One line of setup: a semantic index

```sql
CREATE SEMANTIC INDEX ON meetings(transcript);
```

The database idiom that already means "maintain a cheap search structure so
queries don't scan everything." semcast chunks each transcript (~512-token
slices), embeds every chunk, and keeps the index fresh as meetings arrive —
exactly as incremental as any other index. It's also optional: queries work
without it, the planner just warns you the plan has no cheap stage.

### 2. The query is the whole program

```sql
SELECT meeting_id, title, held_at
FROM meetings
WHERE held_at >= now() - INTERVAL '6 months'
  AND transcript MEANS 'discussed the launch of offline sync in Atlas'
WITH RECALL 0.9;
```

`MEANS` is defined as ground truth: *a model reading the full transcript would
say yes*. Everything the planner substitutes for that is an approximation,
managed under your recall target. `EXPLAIN` shows the funnel it derived:

```text
SemFilter: MEANS('discussed the launch of offline sync in Atlas')   recall ≥ 0.90
├─ held_at filter        18,400 → 3,100 rows    $0     plain SQL, runs first
├─ semantic index scan    3,100 →    47 rows    $0     threshold set by calibration
└─ verify (small model)      47 calls         ~$0.04   reads top-3 chunks per meeting
```

**47 model calls, not 18,400 — and none of them reads a whole transcript.**

Three things earned that number:

- **Predicate ordering is just predicate ordering.** The date filter is free and
  runs first; to the optimizer, `MEANS` is simply a very expensive predicate in
  a framework that has always reordered predicates by cost.
- **The index pulls double duty.** The same chunk vectors that pre-filter
  3,100 → 47 also tell the verify step *which three chunks* to show the model.
- **`WITH RECALL` keeps the shortcut honest.** The index scan has false
  negatives — a meeting that discussed the launch in oblique language can score
  below threshold and vanish — so unlike classic pushdown, this rewrite changes
  the answer. Given a target, the planner samples a few hundred date-surviving
  rows, gets ground-truth labels, and sets the threshold so ≥90% of true
  matches survive (the cascade technique pioneered by LOTUS). Calibration cost
  appears in `EXPLAIN` and is cached for repeat questions of the same shape.
  Omit the clause and thresholds are best-effort — fine for exploration, and
  `EXPLAIN` says so.

### 3. Follow-up extraction, inline

```sql
SELECT held_at,
       EXTRACT(decisions TEXT[] 'concrete decisions made' FROM transcript) AS decisions
FROM meetings
WHERE held_at >= now() - INTERVAL '6 months'
  AND transcript MEANS 'discussed the launch of offline sync in Atlas';
```

`EXTRACT(x FROM y)` is already SQL; here it takes a typed field spec. The
`MEANS` verdicts are cached from the previous query, so the filter costs zero
new model calls — the model runs only to extract `decisions` from the ~12
surviving meetings. Extraction happens *after* the funnel, never before it.

### 4. Reuse when a question recurs

```sql
CREATE SEMANTIC PREDICATE discussed_launch(t, feature, product) AS
  t MEANS 'discussed the launch of {feature} in {product}';

SELECT meeting_id FROM meetings
WHERE discussed_launch(transcript, 'offline sync', 'Beacon');
```

A macro, nothing more. Next month's Beacon question reuses the date filter and
every embedding for free; only fresh verify calls are paid.

---

## Know the bill before you run

The funnel above is real output: `EXPLAIN` prices every plan — stage by stage,
in model calls and dollars — before a single token is spent. Estimates come
from cached history when it exists, otherwise from sampling a small slice of
the corpus. And because estimates are estimates, queries take a hard cap:

```sql
SELECT ... BUDGET 1.00 USD;   -- stop at the cap; return partial rows + a warning
```

---

## Typed extraction

Inline `EXTRACT` covers one-off fields. When the same extraction recurs, name
it — a **semantic type** is the whole specification: field names, types, and a
one-line doc per field. semcast synthesizes the prompt from this; you never
write one. The type also drives constrained decoding (the model *must* return
conforming output) and decides how much of a downstream aggregate becomes
plain SQL.

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

| Field type | Meaning | Why it matters |
|------------|---------|----------------|
| `TEXT` | free-form string | the only truly "prose" field; stays with the model |
| `INT`, `REAL` | numbers | aggregate in SQL (`avg`, `sum`) — no LLM at rollup |
| `REAL CHECK (a..b)` | bounded number | validated at decode time |
| `BOOL` | yes/no | becomes a plain predicate |
| `ONEOF(a, b, c)` | closed category | classification; `GROUP BY`-able |
| `LEVEL(a, b, c)` | ordered category (declared low→high) | ordinal — comparable and rankable |
| `T[]` | list of any above | multi-valued extraction (e.g. `decisions`) |
| `<AnotherType>` | nested semantic type | compose structured extractions |

Fields are **independent by default** — that's what enables field pushdown.
`TOGETHER(...)` marks a group that must be generated in one shot; the planner
never prunes one member without the others.

## Kinds of semantic operator

| Kind | Signature | Example |
|------|-----------|---------|
| **Predicate** | `text → bool` | `transcript MEANS 'discussed a launch'` |
| **Map** (extract) | `text → typed fields` | `EXTRACT(... FROM transcript)`, `CAST(transcript AS MeetingFacts)` |
| **Aggregate** | `set<text> → summary` | `sem_summary(transcript, 'recurring blockers')` (hierarchical fold) |
| **Join** | `text × text → bool` | `JOIN ... ON (c.notes, v.description) MEANS 'the same company'` |

All of them plan the same way: derived cheap stage, calibrated threshold,
model verify, field-level cache.

Semantic join is where planner integration pays off hardest: evaluated naively
it costs O(n×m) model calls, so the planner always *blocks* first — semantic
indexes on both sides, join on vector proximity, LLM adjudication only on the
surviving candidate pairs.

For power users, `CHEAP` / `VERIFY` clauses exist as **overrides** on a named
predicate — pin a shortcut the planner can't guess, like a metadata column that
happens to encode the answer:

```sql
CREATE SEMANTIC PREDICATE discussed_launch(t, feature, product) AS
  t MEANS 'discussed the launch of {feature} in {product}'
  CHEAP USING attendees @> ARRAY['launch-committee'];   -- optional hint, not homework
```

---

## Execution semantics

LLMs break two assumptions databases have always made: functions are
deterministic, and evaluation doesn't fail halfway through. semcast picks
explicit answers instead of inheriting silent ones:

- **Cache keys are full provenance** — `(type version, field, input value,
  model id, prompt-synthesis version)`. Editing one field's doc line invalidates
  exactly that field's entries and nothing else.
- **First evaluation wins** — results are cached, so re-running a query is
  deterministic even though the model isn't.
- **Rows fail, queries don't** — a row that still errors after retries yields
  `NULL` plus an error column. The cache doubles as a checkpoint: a 10k-row job
  killed at row 6,000 resumes for the cost of the rows it hadn't reached.

---

## Architecture

| Piece | DataFusion hook |
|-------|-----------------|
| `MEANS` / `EXTRACT` logical operators | `UserDefinedLogicalNodeCore` → `LogicalPlan::Extension` |
| `CREATE SEMANTIC INDEX / TYPE / PREDICATE` syntax | `RelationPlanner` |
| Semantic index (chunk, embed, incremental maintenance) | [Lance](https://lancedb.github.io/lance/) (Arrow-native) |
| Funnel derivation, threshold calibration, field pushdown, agg split | `OptimizerRule` / `PhysicalOptimizerRule` |
| Physical ops (stream / batch / cascade / cached) | custom `ExecutionPlan` via `ExtensionPlanner` |
| Async batched model calls | `tokio` |

## Where this bites

The pattern that pays: a large corpus, per-document review that's expensive
(human or LLM), and *ad-hoc* questions rather than one fixed pipeline — because
the index and cache compound across queries.

- **eDiscovery / legal review** — "find documents responsive to this request"
  over millions of emails. A recall target there isn't a nicety; it's
  defensibility.
- **Contract analytics** — extract renewal dates and liability caps into typed
  fields once, then `SUM` exposure and `GROUP BY` governing law in plain SQL.
- **Literature screening** — cast papers to typed fields (`is_survey BOOL`,
  `audience LEVEL`) and filter; systematic reviews start exactly this way.
- **Support-ticket mining** — classify once into `ONEOF` categories, trend by
  week forever; new questions re-slice the cache instead of re-labeling.
- **Entity resolution** — the semantic-join case: match a CRM against a
  purchased dataset without n×m model calls.

## Status

Early / experimental. Order of attack:

1. `MEANS` logical operator + verify-only physical plan — correct first, cheap later
2. `CREATE SEMANTIC INDEX` on Lance + the index pre-filter stage
3. `WITH RECALL` — sampled threshold calibration
4. Field-level cache with provenance keys
5. Eval harness: a labeled corpus, reporting **calls saved and recall** against
   the LLM-on-every-row baseline — so the headline claim stays falsifiable

## License

Apache-2.0
