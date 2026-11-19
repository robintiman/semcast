#!/usr/bin/env python3
"""Phase 4 exit check: does `WITH RECALL` hold end to end on CUAD, and at what cost?

    cargo build --release
    uv run --project eval eval/cuad_recall.py --model gemma4:e4b

For each clause: a `MEANS … WITH RECALL r WITH CONFIDENCE c` over an indexed
copy of the contracts (the funnel), then the same `MEANS` over an unindexed
copy, which asks the model about every contract's full text (the reference
the promise is about). Recall is the share of the reference's matches the
funnel returned. Funnels run first, on a fresh store, so none is cheaper for
a reference's answers; the references then reuse the funnels' full-text
labels.

Cost is counted in model calls (one per contract is a full scan) and in
tokens, by a proxy between semcast and Ollama that sums Ollama's own
`prompt_eval_count` and `eval_count`, so a chunk check is priced as the few
hundred tokens it is. A full scan's tokens are measured, not estimated: the
funnel's full-text calls and its reference's together ask each answerable
contract once.

CUAD's lawyer labels are scored too, for context: the promise is about the
model's own full-text answers, not the lawyers'.

Replay (`--replay STORE`) reruns the same statements against an earlier
run's store instead of a model, in seconds and for free:

    uv run --project eval eval/cuad_recall.py --model gemma4:e4b \
        --replay data/cuad_runs/2026-11-17/store.db

The server answers every MEANS request from the recording
(`--provider replay`) and starts from a copy of it with the answers removed
and the vectors kept, so the funnel runs as it would cold and the index
build embeds nothing. Cost is priced in input characters, which tokens
track; a request the recording lacks (a chunk set the run never asked, or a
row the run itself failed on) fails its row and is reported as a miss, and a
replay with misses is not exact. Results go to `--out`/replay unless `--out`
is given, so a replay never overwrites a live run's.
Ollama must still be running, for the few query embeddings.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import duckdb
import pandas as pd
import psycopg2

from sembench import REPO, Notices, Server

csv.field_size_limit(sys.maxsize)

# Spread over CUAD's selectivity range, rare to common.
CLAUSES = [
    "Most Favored Nation",  # 5.5%
    "Liquidated Damages",  # 12%
    "Non-Compete",  # 23%
    "Audit Rights",  # 42%
    "Anti-Assignment",  # 73%
]

# How semcast's MEANS system prompts open (src/physical/verify.rs): over the
# full text, or over a document's top chunks.
FULL_TEXT_PROMPT = "You are evaluating a predicate over a document."
CHUNKED_PROMPT = "You are evaluating a predicate over excerpts from a document"


class TokenProxy:
    """Forwards semcast's Ollama requests and counts the tokens Ollama reports."""

    FIELDS = ("full_text_in", "chunked_in", "other_in", "out", "embed_in")

    def __init__(self, upstream: str) -> None:
        self.upstream = upstream.rstrip("/")
        self.lock = threading.Lock()
        self.counts = dict.fromkeys(self.FIELDS, 0)
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args) -> None:
                pass

            def _forward(self, method: str) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else None
                request = urllib.request.Request(
                    proxy.upstream + self.path,
                    data=body,
                    method=method,
                    headers={"Content-Type": self.headers.get("Content-Type", "application/json")},
                )
                try:
                    with urllib.request.urlopen(request, timeout=3600) as response:
                        status, payload = response.status, response.read()
                except urllib.error.HTTPError as error:
                    status, payload = error.code, error.read()
                except (urllib.error.URLError, OSError) as error:
                    # Ollama unreachable or too slow: a gateway error semcast
                    # retries, instead of a dropped connection.
                    status, payload = 502, json.dumps({"error": f"proxy: {error}"}).encode()
                if status == 200 and body:
                    proxy._count(self.path, body, payload)
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # semcast gave up on the call; it was counted if answered

            def do_POST(self) -> None:
                self._forward("POST")

            def do_GET(self) -> None:
                self._forward("GET")

        class Server(ThreadingHTTPServer):
            request_queue_size = 64
            daemon_threads = True

        self.server = Server(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def _count(self, path: str, body: bytes, payload: bytes) -> None:
        try:
            answer = json.loads(payload)
            request = json.loads(body)
        except ValueError:
            return
        tokens_in = int(answer.get("prompt_eval_count") or 0)
        with self.lock:
            if path.startswith("/api/chat"):
                system = next(
                    (m.get("content", "") for m in request.get("messages", []) if m.get("role") == "system"),
                    "",
                )
                if system.startswith(FULL_TEXT_PROMPT):
                    self.counts["full_text_in"] += tokens_in
                elif system.startswith(CHUNKED_PROMPT):
                    self.counts["chunked_in"] += tokens_in
                else:
                    self.counts["other_in"] += tokens_in
                self.counts["out"] += int(answer.get("eval_count") or 0)
            elif path.startswith("/api/embed"):
                self.counts["embed_in"] += tokens_in

    def snapshot(self) -> dict[str, int]:
        with self.lock:
            return dict(self.counts)

    def stop(self) -> None:
        self.server.shutdown()


class ReplayMeter:
    """Prices a replay: input characters per request, from the provider's log.

    Same fields as [`TokenProxy`], counted in characters for chat; embedding
    tokens still come from the proxy, which forwards them to Ollama.
    """

    def __init__(self, log: Path, proxy: TokenProxy) -> None:
        self.log = log
        self.proxy = proxy

    def snapshot(self) -> dict[str, int]:
        counts = self.proxy.snapshot()
        counts["misses"] = 0
        if self.log.exists():
            for line in self.log.read_text().splitlines():
                request = json.loads(line)
                field = {"full": "full_text_in", "chunked": "chunked_in"}.get(request["kind"], "other_in")
                counts[field] += request["chars"]
                counts["misses"] += not request["answered"]
        return counts

    def stop(self) -> None:
        self.proxy.stop()


# What a replay deletes from its copy of the recording: every answer and
# what was learned from them, and the definitions, which the run recreates.
# Kept: the vectors (`embeddings`), so the index build embeds nothing.
REPLAY_CLEARED = ("verdicts", "ask_stats", "prune_curves", "catalog", "index_chunks")


def prepare_replay(recording: Path, work: Path) -> tuple[Path, Path]:
    """Copy the recording twice: the answers the provider reads, and the
    session store, cleared of answers."""
    copies = []
    for name in ("recording.db", "store.db"):
        copy = work / name
        for suffix in ("", "-wal"):
            source = Path(f"{recording}{suffix}")
            if source.exists():
                shutil.copyfile(source, f"{copy}{suffix}")
        copies.append(copy)
    oracle, store = copies
    db = sqlite3.connect(store)
    try:
        with db:
            for table in REPLAY_CLEARED:
                db.execute(f"DELETE FROM {table}")
    finally:
        db.close()
    return oracle, store


def done_line(notices: list[str]) -> str:
    return " ".join(line for line in notices if "funnel done" in line)


def number(pattern: str, text: str) -> int:
    return sum(int(n) for n in re.findall(pattern, text))


class Statement:
    """One statement's rows, funnel line, seconds, tokens, and error if it failed."""

    def __init__(self) -> None:
        self.ids: set[str] = set()
        self.line = ""
        self.seconds = 0.0
        self.tokens: dict[str, int] = {}
        self.error: str | None = None

    @property
    def input_tokens(self) -> int:
        return sum(self.tokens.get(key, 0) for key in ("full_text_in", "chunked_in", "other_in"))


def run(server: Server, proxy: TokenProxy | ReplayMeter, sql: str) -> Statement:
    """Run `sql` on a fresh connection; a failure is recorded, not raised."""
    result = Statement()
    before = proxy.snapshot()
    started = time.monotonic()
    try:
        server.ensure()
        connection = server.connect()
        try:
            notices = Notices()
            connection.notices = notices
            cursor = connection.cursor()
            cursor.execute(sql)
            if cursor.description is not None:
                result.ids = {row[0] for row in cursor.fetchall()}
            result.line = done_line(notices.lines)
        finally:
            connection.close()
    except psycopg2.Error as error:
        result.error = str(error).strip()
    result.seconds = time.monotonic() - started
    after = proxy.snapshot()
    result.tokens = {key: after[key] - before[key] for key in after}
    return result


def summarize(results: list[dict], args: argparse.Namespace, rows: int, index: dict) -> str:
    unit = "Input chars" if args.replay else "Input tokens"
    lines = [
        f"| Clause | Matches | Funnel rows | Recall | Calls | Of {rows} | {unit} | Of full scan | Full text |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if r["recall"] is None:
            lines.append(f"| {r['clause']} | failed: {r['error'][:80]} |||||||| ")
            continue
        share = (
            f"{r['funnel_input_tokens'] / r['full_scan_input_tokens']:.0%}"
            if r["full_scan_input_tokens"]
            else f"n/a: {r['error']}"
        )
        lines.append(
            f"| {r['clause']} | {r['reference_matches']} | {r['funnel_rows']} | {r['recall']:.3f} | "
            f"{r['funnel_model_calls']} | {r['funnel_model_calls'] / rows:.0%} | "
            f"{r['funnel_input_tokens']} | {share} | {r['full_text']} |"
        )
    met = sum(r["recall"] is not None and r["recall"] >= args.recall for r in results)
    if args.replay:
        misses = sum(r["replay_misses"] for r in results)
        lines.append(
            f"\nReplay of {args.replay}: {misses} requests the recording lacked"
            + (" (exact)." if not misses else ", each a failed row: not exact.")
        )
    lines.append(
        f"\nIndex build: {index['seconds']:.0f} s, {index['embed_tokens']} embedding tokens "
        "(a full scan needs none)."
    )
    lines.append(
        f"Target recall {args.recall} at confidence {args.confidence}: met on {met} of {len(results)}."
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=REPO / "data" / "cuad")
    parser.add_argument("--clauses", nargs="*", default=CLAUSES)
    parser.add_argument("--limit", type=int, help="only the first N contracts (a smoke run)")
    parser.add_argument("--recall", type=float, default=0.9)
    parser.add_argument("--confidence", type=float, default=0.9)
    parser.add_argument("--num-ctx", type=int, default=131072)
    parser.add_argument("--model")
    parser.add_argument("--provider", default="ollama", choices=["ollama"])
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--server", type=Path, default=REPO / "target" / "release" / "semcast")
    parser.add_argument("--out", type=Path, help="default eval/results/cuad_recall, /replay under it for --replay")
    parser.add_argument(
        "--replay",
        type=Path,
        help="an earlier run's store.db: answer from it instead of a model (see above)",
    )
    args = parser.parse_args()
    if args.replay and not args.model:
        parser.error("--replay needs --model, the model the recording asked")
    if args.replay and not args.replay.exists():
        parser.error(f"{args.replay} is missing")
    if args.out is None:
        args.out = REPO / "eval" / "results" / "cuad_recall"
        if args.replay:
            args.out /= "replay"
    if not args.server.exists():
        parser.error(f"{args.server} is missing; run cargo build --release")

    args.connect = None
    args.work = Path(tempfile.mkdtemp(prefix="cuad-recall-"))
    proxy: TokenProxy | ReplayMeter = TokenProxy(args.ollama_url)
    args.server_arg = [
        f"--store={args.work / 'store.db'}",
        f"--ollama-num-ctx={args.num_ctx}",
        f"--ollama-url={proxy.url}",
    ]
    if args.replay:
        oracle, _ = prepare_replay(args.replay, args.work)
        log = args.work / "replay.jsonl"
        proxy = ReplayMeter(log, proxy)
        # Keys carry the provider's model id; the recording's is Ollama's.
        args.provider, args.model = "replay", f"ollama/{args.model}"
        args.server_arg += [f"--replay-store={oracle}", f"--replay-log={log}"]

    with open(args.data / "clauses.csv") as file:
        prompts = {r["clause"]: r["prompt"] for r in csv.DictReader(file)}
    with open(args.data / "contracts.json") as file:
        contracts = pd.DataFrame([{"id": c["id"], "body": c["text"]} for c in map(json.loads, file)])
    if args.limit:
        contracts = contracts.head(args.limit)
    kept = set(contracts["id"])
    lawyers: dict[str, set[str]] = {}
    with open(args.data / "clause_labels.csv") as file:
        for r in csv.DictReader(file):
            if r["present"] == "true" and r["id"] in kept:
                lawyers.setdefault(r["clause"], set()).add(r["id"])
    # Parquet, which keeps each contract's line breaks inside its value.
    table = args.work / "contracts.parquet"
    duckdb.from_df(contracts).write_parquet(str(table))
    rows = len(contracts)
    setup = [
        # IF NOT EXISTS: a server restarted after a crash restores them from
        # its store's catalog.
        f"CREATE EXTERNAL TABLE IF NOT EXISTS contracts STORED AS PARQUET LOCATION '{table}'",
        f"CREATE EXTERNAL TABLE IF NOT EXISTS contracts_plain STORED AS PARQUET LOCATION '{table}'",
    ]
    server = Server(args, setup)
    args.out.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    try:
        built = run(server, proxy, "CREATE SEMANTIC INDEX ON contracts(body)")
        if built.error:
            sys.exit(f"index build failed: {built.error}")
        index = {"seconds": built.seconds, "embed_tokens": built.tokens.get("embed_in", 0)}
        print(f"index built in {built.seconds:.0f} s, {index['embed_tokens']} embedding tokens", flush=True)

        target = f"WITH RECALL {args.recall} WITH CONFIDENCE {args.confidence}"
        funnels = {}
        for clause in args.clauses:
            condition = prompts[clause].replace("'", "''")
            funnel = run(server, proxy, f"SELECT id FROM contracts WHERE body MEANS '{condition}' {target}")
            funnels[clause] = funnel
            print(
                f"funnel    {clause}: {len(funnel.ids)} rows, {funnel.seconds:.0f} s, "
                f"{funnel.input_tokens} input {'chars' if args.replay else 'tokens'} | "
                f"{funnel.error or funnel.line}",
                flush=True,
            )
        for clause in args.clauses:
            condition = prompts[clause].replace("'", "''")
            reference = run(server, proxy, f"SELECT id FROM contracts_plain WHERE body MEANS '{condition}'")
            funnel = funnels[clause]
            error = funnel.error or reference.error
            truth = lawyers.get(clause, set())
            row = {
                "clause": clause,
                "error": error,
                "reference_matches": len(reference.ids),
                "funnel_rows": len(funnel.ids),
                "recall": (len(funnel.ids & reference.ids) / len(reference.ids) if reference.ids else 1.0),
                "funnel_model_calls": number(r"(\d+) model calls", funnel.line),
                "funnel_labels": number(r"(\d+) rows labeled", funnel.line),
                "funnel_seconds": round(funnel.seconds),
                "funnel_input_tokens": funnel.input_tokens,
                "funnel_full_text_tokens": funnel.tokens.get("full_text_in", 0),
                "funnel_chunked_tokens": funnel.tokens.get("chunked_in", 0),
                "funnel_output_tokens": funnel.tokens.get("out", 0),
                "funnel_embed_tokens": funnel.tokens.get("embed_in", 0),
                # Every answerable contract asked once on its full text: the
                # funnel's full-text calls, and the reference's for the rest.
                "full_scan_input_tokens": (
                    funnel.tokens.get("full_text_in", 0) + reference.tokens.get("full_text_in", 0)
                ),
                # A replay prices in characters, not tokens; see the docstring.
                "unit": "chars" if args.replay else "tokens",
                "replay_misses": funnel.tokens.get("misses", 0) + reference.tokens.get("misses", 0),
                "full_text": "verify reads full text" in funnel.line,
                "reference_errors": number(r"(\d+) dropped", reference.line),
                "funnel_errors": number(r"(\d+) dropped", funnel.line),
                "lawyer_matches": len(truth),
                "reference_vs_lawyers_recall": len(reference.ids & truth) / len(truth) if truth else None,
                "funnel_vs_lawyers_recall": len(funnel.ids & truth) / len(truth) if truth else None,
                "funnel_line": funnel.line,
                "reference_line": reference.line,
            }
            if error:
                # Nothing measured: no number that could pass for one.
                for key in ("recall", "full_scan_input_tokens", "reference_vs_lawyers_recall", "funnel_vs_lawyers_recall"):
                    row[key] = None
            elif row["full_scan_input_tokens"] == 0 and reference.ids:
                row["error"] = (
                    "no full-text tokens counted: has verify.rs's MEANS prompt changed? "
                    f"(FULL_TEXT_PROMPT = {FULL_TEXT_PROMPT!r})"
                )
            results.append(row)
            # Written after every clause, so a failure later keeps these.
            (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
            print(
                f"reference {clause}: {len(reference.ids)} matches, recall {row['recall']:.3f}, "
                f"funnel {row['funnel_model_calls']} calls | {error or reference.line}",
                flush=True,
            )
    finally:
        server.stop()
        proxy.stop()

    other = proxy.snapshot().get("other_in", 0)
    if other:
        print(
            f"warning: {other} chat input tokens matched neither MEANS prompt; "
            "check FULL_TEXT_PROMPT and CHUNKED_PROMPT against src/physical/verify.rs",
            file=sys.stderr,
        )
    summary = summarize(results, args, rows, index)
    (args.out / "summary.md").write_text(summary + "\n")
    print(summary)
    print(f"wrote {args.out}; server log in {args.work / 'server.log'}")


if __name__ == "__main__":
    main()
