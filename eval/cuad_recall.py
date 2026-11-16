#!/usr/bin/env python3
"""Phase 4 exit check: does `WITH RECALL` hold end to end on CUAD, and at what cost?

    cargo build --release
    uv run --project eval eval/cuad_recall.py --data data/cuad

For each clause: a `MEANS … WITH RECALL r WITH CONFIDENCE c` over an indexed
copy of the contracts (the funnel), then the same `MEANS` over an unindexed
copy, which asks the model about every contract's full text (the reference
the promise is about). Recall is the share of the reference's matches the
funnel returned; cost is the funnel's model calls against one per contract.
Funnels run first, on a fresh store, so none is cheaper for a reference's
answers; the references then reuse the funnels' full-text labels.

CUAD's lawyer labels are scored too, for context: the promise is about the
model's own full-text answers, not the lawyers'.

Tokens are counted by a proxy between semcast and Ollama, from Ollama's own
`prompt_eval_count` and `eval_count`, so a chunk check is priced as the few
hundred tokens it is, not as a call. The full-scan baseline (every contract
asked on its full text) is estimated from the run's own calls: their prompt
tokens per character, times every contract's length.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
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


class TokenProxy:
    """Forwards semcast's Ollama requests and counts the tokens Ollama reports."""

    def __init__(self, upstream: str) -> None:
        self.upstream = upstream.rstrip("/")
        self.lock = threading.Lock()
        self.chat_in = self.chat_out = self.embed_in = 0
        # (input characters, prompt tokens) of every chat request.
        self.calls: list[tuple[int, int]] = []
        proxy = self

        class Handler(BaseHTTPRequestHandler):
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
                if status == 200 and body:
                    proxy._count(self.path, body, payload)
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self) -> None:
                self._forward("POST")

            def do_GET(self) -> None:
                self._forward("GET")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def _count(self, path: str, body: bytes, payload: bytes) -> None:
        try:
            answer = json.loads(payload)
        except ValueError:
            return
        tokens_in = int(answer.get("prompt_eval_count") or 0)
        with self.lock:
            if path.startswith("/api/chat"):
                self.chat_in += tokens_in
                self.chat_out += int(answer.get("eval_count") or 0)
                messages = json.loads(body).get("messages", [])
                self.calls.append((sum(len(m.get("content", "")) for m in messages), tokens_in))
            elif path.startswith("/api/embed"):
                self.embed_in += tokens_in

    def totals(self) -> tuple[int, int, int]:
        with self.lock:
            return self.chat_in, self.chat_out, self.embed_in

    def tokens_per_char(self) -> float | None:
        """Prompt tokens per input character, over every chat call."""
        with self.lock:
            chars = sum(c for c, _ in self.calls)
            tokens = sum(t for _, t in self.calls)
        return tokens / chars if chars else None

    def stop(self) -> None:
        self.server.shutdown()


def done_line(notices: list[str]) -> str:
    return " ".join(line for line in notices if "funnel done" in line)


def number(pattern: str, text: str) -> int:
    return sum(int(n) for n in re.findall(pattern, text))


def run(cursor, connection, proxy: TokenProxy, sql: str) -> tuple[list[str], str, float, tuple]:
    notices = Notices()
    connection.notices = notices
    before = proxy.totals()
    started = time.monotonic()
    cursor.execute(sql)
    ids = [row[0] for row in cursor.fetchall()]
    spent = tuple(after - b for after, b in zip(proxy.totals(), before))
    return ids, done_line(notices.lines), time.monotonic() - started, spent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=REPO / "data" / "cuad")
    parser.add_argument("--clauses", nargs="*", default=CLAUSES)
    parser.add_argument("--limit", type=int, help="only the first N contracts (a smoke run)")
    parser.add_argument("--recall", type=float, default=0.9)
    parser.add_argument("--confidence", type=float, default=0.9)
    parser.add_argument("--num-ctx", type=int, default=131072)
    parser.add_argument("--model")
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--server", type=Path, default=REPO / "target" / "release" / "semcast")
    parser.add_argument("--out", type=Path, default=REPO / "eval" / "results" / "cuad_recall")
    args = parser.parse_args()
    if not args.server.exists():
        parser.error(f"{args.server} is missing; run cargo build --release")

    args.connect = None
    args.work = Path(tempfile.mkdtemp(prefix="cuad-recall-"))
    proxy = TokenProxy(args.ollama_url)
    args.server_arg = [
        f"--store={args.work / 'store.db'}",
        f"--ollama-num-ctx={args.num_ctx}",
        f"--ollama-url={proxy.url}",
    ]

    prompts = {r["clause"]: r["prompt"] for r in csv.DictReader(open(args.data / "clauses.csv"))}
    lawyers: dict[str, set[str]] = {}
    for r in csv.DictReader(open(args.data / "clause_labels.csv")):
        if r["present"] == "true":
            lawyers.setdefault(r["clause"], set()).add(r["id"])
    # Parquet, which keeps each contract's line breaks inside its value.
    table = args.work / "contracts.parquet"
    contracts = pd.DataFrame(
        [
            {"id": c["id"], "body": c["text"]}
            for c in map(json.loads, open(args.data / "contracts.json"))
        ]
    )
    if args.limit:
        contracts = contracts.head(args.limit)
    duckdb.from_df(contracts).write_parquet(str(table))
    rows = len(contracts)
    contract_chars = int(contracts["body"].str.len().sum())
    setup = [
        f"CREATE EXTERNAL TABLE contracts STORED AS PARQUET LOCATION '{table}'",
        f"CREATE EXTERNAL TABLE contracts_plain STORED AS PARQUET LOCATION '{table}'",
    ]
    server = Server(args, setup)
    results = []
    try:
        server.ensure()
        connection = server.connect()
        cursor = connection.cursor()
        started = time.monotonic()
        cursor.execute("CREATE SEMANTIC INDEX ON contracts(body)")
        print(
            f"index built in {time.monotonic() - started:.0f} s, "
            f"{proxy.totals()[2]} embedding tokens",
            flush=True,
        )

        target = f"WITH RECALL {args.recall} WITH CONFIDENCE {args.confidence}"
        funnels = {}
        for clause in args.clauses:
            condition = prompts[clause].replace("'", "''")
            ids, line, seconds, spent = run(
                cursor,
                connection,
                proxy,
                f"SELECT id FROM contracts WHERE body MEANS '{condition}' {target}",
            )
            funnels[clause] = (set(ids), line, seconds, spent)
            print(
                f"funnel    {clause}: {len(ids)} rows, {seconds:.0f} s, "
                f"{spent[0]} input tokens | {line}",
                flush=True,
            )
        for clause in args.clauses:
            condition = prompts[clause].replace("'", "''")
            ids, line, seconds, _ = run(
                cursor, connection, proxy, f"SELECT id FROM contracts_plain WHERE body MEANS '{condition}'"
            )
            funnel, funnel_line, funnel_seconds, spent = funnels[clause]
            reference = set(ids)
            truth = lawyers.get(clause, set())
            row = {
                "clause": clause,
                "reference_matches": len(reference),
                "funnel_rows": len(funnel),
                "recall": len(funnel & reference) / len(reference) if reference else 1.0,
                "funnel_model_calls": number(r"(\d+) model calls", funnel_line),
                "funnel_labels": number(r"(\d+) rows labeled", funnel_line),
                "funnel_seconds": round(funnel_seconds),
                "funnel_input_tokens": spent[0],
                "funnel_output_tokens": spent[1],
                "full_text": "verify reads full text" in funnel_line,
                "reference_errors": number(r"(\d+) dropped", line),
                "funnel_errors": number(r"(\d+) dropped", funnel_line),
                "lawyer_matches": len(truth),
                "reference_vs_lawyers_recall": len(reference & truth) / len(truth) if truth else None,
                "funnel_vs_lawyers_recall": len(funnel & truth) / len(truth) if truth else None,
                "funnel_line": funnel_line,
                "reference_line": line,
            }
            results.append(row)
            print(
                f"reference {clause}: {len(reference)} matches, recall {row['recall']:.3f}, "
                f"funnel {row['funnel_model_calls']} calls | {line}",
                flush=True,
            )
        connection.close()
    finally:
        server.stop()
        proxy.stop()

    # Every call's prompt is mostly contract text (whole, or its chunks).
    ratio = proxy.tokens_per_char()
    full_scan_tokens = round(ratio * contract_chars) if ratio else None
    for r in results:
        r["full_scan_input_tokens_estimate"] = full_scan_tokens

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    met = sum(r["recall"] >= args.recall for r in results)
    lines = [
        f"| Clause | Matches | Funnel rows | Recall | Funnel calls | Of {rows} | Input tokens | Of full scan | Full text |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        share = (
            f"{r['funnel_input_tokens'] / full_scan_tokens:.0%}" if full_scan_tokens else "n/a"
        )
        lines.append(
            f"| {r['clause']} | {r['reference_matches']} | {r['funnel_rows']} | {r['recall']:.3f} | "
            f"{r['funnel_model_calls']} | {r['funnel_model_calls'] / rows:.0%} | "
            f"{r['funnel_input_tokens']} | {share} | {r['full_text']} |"
        )
    lines.append(f"\nFull scan, estimated: {full_scan_tokens} input tokens.")
    lines.append(f"\nTarget recall {args.recall} at confidence {args.confidence}: met on {met} of {len(results)}.")
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {args.out}; server log in {args.work / 'server.log'}")


if __name__ == "__main__":
    main()
