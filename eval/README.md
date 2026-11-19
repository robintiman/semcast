# Eval

Runs [SemBench](https://sembench.github.io/SemBench/) against a semcast server
and scores it with SemBench's own evaluator.

```sh
./eval/setup.sh                          # once: fetch SemBench at a pinned commit
cargo build --release
uv run --project eval eval/sembench.py   # Movie, all queries, one cold run
```

Needs a running Ollama with the server's default models, or
`--provider anthropic` with `ANTHROPIC_API_KEY` set.

## Options

| Flag | Effect |
|---|---|
| `--queries 2 3` | Run only these queries |
| `--repeats 5` | Repeat the run; the summary averages them |
| `--model <name>` | Completion model passed to `semcast serve` |
| `--server-arg=<arg>` | Extra `semcast serve` argument, repeatable |
| `--warm` | One server for the whole run, so later queries reuse verdicts |
| `--connect host:port` | Use a server that is already running |
| `--system <name>` | File results under another name, to compare variants |
| `--queries-dir <dir>` | Use another query set |
| `--timeout <seconds>` | Per query; the server is killed and restarted, and the query scores 0 |

## What it measures

- **Quality**: SemBench's metric per query. F1 for retrieval,
  1/(1 + relative error) for aggregates, Spearman for ranking (which can be
  negative). A failed query scores 0.
- **Seconds**: wall time of the query's statements over the wire.
- **Model calls** and **cache hits**: from the server's closing funnel NOTICE.

Seconds, calls and hits are averaged over the runs that finished.

By default every query gets a fresh server, so none is cheaper because an
earlier one filled the verdict cache. Passing `--server-arg=--store=<file>`
undoes that.

Results land in `eval/results/<scenario>/<system>/`: `summary.json`, and per
run the result CSVs and `metrics.json`.

## Scenarios

Only Movie is wired up: it is SemBench's text-only scenario. Queries are in
`queries/movie/`, one file per SemBench query, plus `setup.sql`.

- Q5–Q7 ask for review pairs with the same or opposite sentiment. semcast has
  no semantic join, so they label each review once and join on the label.
- The ground-truth columns are removed before the server sees the data:
  `scoreSentiment`, `reviewState`, `originalScore`, `audienceScore` and
  `tomatoMeter`.
- Q5–Q7 cost one call per review, so their call counts are not comparable
  with systems that judge every pair.

## Limits

- Token counts and cost are not reported: the server does not expose them.
- Only the `sf_2000` data SemBench ships in its repository is used. SemBench
  asks for regenerated data before publishing numbers.
- A prompt in a query file must not contain a semicolon.

## Store hits

`storehit.py` replays a session against one server and one fresh `--store`
file, and reports the share of semantic questions the store answered rather
than the model.

```sh
uv run --project eval eval/storehit.py movie     # SemBench Movie, Q1–Q10 in order
uv run --project eval eval/storehit.py cfpb --data data/cfpb_complaints.csv
```

A session file in `queries/sessions/` marks each statement with `-- setup` or
`-- step: <label>`; `{data}` is replaced by `--data`. Per-step counts land in
`results/storehit/<session>.json`.

Counts are per operator, as the funnel NOTICE reports them, so read the share
with care:

- A CTE referenced twice is planned twice: the second copy's answers count as
  stored though no earlier step paid for them. Movie Q5–Q7 self-join one, so
  each counts its 256 reviews twice; without that the Movie share is 28.0%,
  not the 37.1% the script prints.
- Units are rows, not distinct texts. Verify and classify ask a text once
  per batch, EXTRACT once per row, and the store counts every row. Half of
  the reviews of the movie Q5–Q9 ask about are exact duplicates; counted by
  distinct text, Movie is about 21%.
- A multi-branch `CASE` or multi-field EXTRACT counts stored answers per
  branch or field but calls per row.
- A failed step reports nothing, though the calls it sent were spent and
  later steps find their answers stored. The summary line flags failed steps.
- With `--scorer`, scorer calls and hits count as questions too.

## CUAD recall

```sh
cargo build --release
uv run --project eval eval/cuad_recall.py --model gemma4:e4b
```

Does `WITH RECALL r WITH CONFIDENCE c` hold end to end on CUAD (509
contracts, 5 clauses from 5.5% to 73% selectivity), and at what cost? For
each clause, a `MEANS … WITH RECALL` over an indexed copy of the contracts
(the funnel) runs first, on a fresh store. Then the same `MEANS` over an
unindexed copy asks about every contract's full text (the reference the
promise is about), reusing the funnel's full-text labels.

`--data` (default `data/cuad`) holds CUAD v1 as three files:
`contracts.json` (one `{"id", "text"}` per line), `clauses.csv` (`clause`,
`prompt`: the `MEANS` condition) and `clause_labels.csv` (`id`, `clause`,
`present`: the lawyers' labels). `--limit N` is a smoke run over the first N
contracts. Ollama only: `--ollama-url` is the upstream, and the default
`--num-ctx` of 131072 fits every CUAD contract.

Recall is the share of the reference's matches the funnel returned. Cost is
counted twice:

- model calls, from the funnel NOTICE, against one per contract;
- input tokens, from Ollama's own `prompt_eval_count`, counted by a proxy the
  script puts between semcast and Ollama. A full scan's tokens are measured:
  the funnel's full-text calls and the reference's together ask every
  answerable contract once. Index embedding tokens are reported apart.

Seconds depend on Ollama's prompt cache: a re-run over the same contracts is
much faster than the first. The lawyers' labels are scored too, for context
only. Results go to `results/cuad_recall/`, rewritten after every clause; a
failed statement is recorded and the run goes on.

### Replay

```sh
uv run --project eval eval/cuad_recall.py --model gemma4:e4b \
    --replay data/cuad_runs/2026-11-17/store.db
```

Reruns the same statements against an earlier run's `store.db` instead of a
model: seconds instead of hours, no model cost, the same answers. Use it to
measure a funnel change. The server (`--provider replay`) answers each
`MEANS` request from the recording, keyed exactly as verify keys it, and
starts from a copy of the recording with every answer, statistic and
definition removed, so the funnel runs as it would cold. The vectors are
kept, so the index build embeds nothing; Ollama still embeds the few query
strings.

Cost is priced in input characters (system prompt plus input), which tokens
track: on the 2026-11-17 run the shares of a full scan matched the measured
token shares to the percent. A request the recording lacks fails its row and
is counted as a miss, for example a chunk set the run never asked. A replay
with misses is not exact, and the summary says so. Replaying the run it
recorded reproduces every funnel line, row and call.
