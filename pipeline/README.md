# semcast-hn

A daily, manually triggered batch job: fetch a day of Hacker News, keep only
what actually discusses semantic data processing, and extract a structured
digest — using semcast for all the semantic work.

Orchestrated with Dagster, using **daily partitions**, so "re-run last
Tuesday" and "backfill the week" are built in. No schedule is attached: you
materialize a partition from the UI or the CLI. Adding a `ScheduleDefinition`
in `definitions.py` is the only change needed to make it automatic.

## Topology

Dagster must run **on the same host as the semcast server**. Paths in semcast
SQL (`FROM '<parquet>'`, `COPY ... TO`) resolve on the *server*, so the fetch
step has to write where semcast can read. Split them across hosts and you
need a shared mount plus divergent path config.

```
hn_raw ──> hn_staged ──> hn_relevant ──┬──> hn_digest  ──> out/…/digest.parquet
(Algolia)  (CTAS +       (MEANS +      └──> hn_topics  ──> out/…/topics.parquet
            index)        dedupe)
```

| Asset | What it does | Cost |
|---|---|---|
| `hn_raw` | Algolia API → Parquet | none |
| `hn_staged` | length filter, then `CREATE SEMANTIC INDEX` | one embedding per row |
| `hn_relevant` | `MEANS` filter + `SEMANTIC DISTINCT ON` | one model call per surviving row |
| `hn_digest` | `CAST(... AS HnFinding)` extraction | one model call per field per row |
| `hn_topics` | `GROUP BY MEANING OF` rollup | one model call per group |

The `LENGTH` filter in `01_stage.sql` is the cheapest thing in the pipeline
and the one that most affects the bill — it runs before the index is built,
so every row it drops is an embedding never paid for. Tune it with
`HN_MIN_TEXT_LEN` before you tune anything else.

## Setup

```sh
cd pipeline
uv venv && uv pip install -e .

# semcast, on this host, with persistent index and job dirs
semcast serve --index-dir ~/semcast/idx --jobs-dir ~/semcast/jobs

export SEMCAST_DATA_ROOT=/abs/path/to/data   # absolute: both processes use it
export DAGSTER_HOME=$PWD/dagster_home
dagster dev
```

Then materialize a day from the UI, or:

```sh
dagster asset materialize --select '*' --partition 2026-08-05 -m semcast_hn.definitions
```

Every knob is an environment variable — see `config.py`. The one worth
editing first is `HN_MEANS`, the predicate defining what counts as worth
reading; it is the whole point of the pipeline.

## Notes on talking to semcast

These are the things that cost time to discover, all verified against
semcast v0.3.0.

**Use psycopg2, not psycopg3 or asyncpg.** semcast implements
`SimpleQueryHandler` only. Clients that bind parameters server-side over the
extended protocol cannot connect.

**Leading comments break `CREATE SEMANTIC ...`.** The custom DDL dispatches
on a statement's first tokens without skipping comments, so a documented
template fails with `Expected: an object type after CREATE, found: SEMANTIC`.
The templates here stay commented for humans; `sql.render()` strips leading
`--` lines before sending. Ordinary DataFusion DDL is unbothered.

**`SEMANTIC DISTINCT ON` cannot be combined with `ORDER BY`.** The parser
wraps each `DISTINCT ON` expression in a `semantic_key()` marker, so
DataFusion's "DISTINCT ON expressions must match initial ORDER BY
expressions" rule can never be satisfied — and naming the marker explicitly
is rejected too. Consequence: which row of a duplicate group survives is not
controllable. This contradicts the README in the repo root and is worth
filing upstream; `03_relevant.sql` has the details and the ORDER BY to
restore if it is fixed.

**Failed extraction is silent.** A `CAST(... AS <SemanticType>)` against an
unreachable model provider returns NULLs and a *successful* query. `hn_digest`
therefore checks the extraction rate and fails loudly before persisting
anything — without that guard, a dead provider produces a green asset holding
an empty digest.

**Everything is one global namespace, and it is in memory.** semcast builds a
single `SessionContext` for the whole process, so these assets can each open
their own connection and still see the previous one's tables. But there is no
isolation between concurrent runs, which is why every table and semantic type
here is suffixed with the partition key — otherwise a two-day backfill would
collide. And `CREATE TABLE AS` dies with the process, which is why each
output is `COPY`d to Parquet: anything wanted tomorrow has to hit disk today.

**Semantic types cannot be redeclared.** There is no `CREATE OR REPLACE`, so
a bare `HnFinding` would fail on day two. The partition suffix also means an
edit to the doc lines in `05_type.sql` takes effect on the next run rather
than being silently ignored in favour of yesterday's definition.

**Set `--max-concurrent-jobs` to what your provider actually serves.** It
defaults to 4. Against a single local Ollama, more concurrent jobs is slower,
not faster — use 1 and let Dagster serialize.

## Layout

```
semcast_hn/
  config.py          every knob, env-overridable
  hn.py              Algolia fetch + HTML stripping + Arrow schema
  semcast_client.py  psycopg2 wrapper: run / submit / poll / wait
  sql.py             template loading, rendering, literal quoting
  assets.py          the five Dagster assets
  definitions.py     Dagster entrypoint
  sql/*.sql          one statement per file
```
