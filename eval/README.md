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
- A text repeated within a batch is asked once but stored per row, and a
  multi-branch `CASE` counts stored answers per branch but calls per row.
- With `--scorer`, scorer calls and hits count as questions too.
