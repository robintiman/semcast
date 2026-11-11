#!/usr/bin/env python3
"""Run SemBench against a semcast server and score it with SemBench's evaluator.

    ./eval/setup.sh                              # once: fetch SemBench
    cargo build --release
    uv run --project eval eval/sembench.py       # Movie, all queries, one cold run

Each query gets a fresh server by default, so no query pays less because an
earlier one filled the verdict cache. `--warm` keeps one server for the run.
Queries go over the Postgres wire, the surface users see; model calls are
read from the funnel NOTICE the server sends when a statement finishes.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pandas as pd
import psycopg2

EVAL = Path(__file__).resolve().parent
REPO = EVAL.parent
SEMBENCH = EVAL / ".sembench"
# Columns the benchmarked system must not see: the ground-truth labels.
HIDDEN = {"movie": {"Reviews.csv": ["scoreSentiment"]}}


class Notices:
    """psycopg2 trims a plain list to 50 notices; a long funnel sends more."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def append(self, line: str) -> None:
        self.lines.append(line.strip())


class Server:
    """A semcast server this run owns, or one that is already listening."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.process: subprocess.Popen | None = None
        self.host, self.port = "127.0.0.1", 0
        if args.connect:
            self.host, port = args.connect.rsplit(":", 1)
            self.port = int(port)

    def start(self) -> None:
        if self.args.connect:
            return
        with socket.socket() as probe:
            probe.bind((self.host, 0))
            self.port = probe.getsockname()[1]
        command = [str(self.args.server), "serve", "--port", str(self.port)]
        command += ["--provider", self.args.provider]
        if self.args.model:
            command += ["--model", self.args.model]
        command += self.args.server_arg
        log = open(self.args.work / "server.log", "ab")
        self.process = subprocess.Popen(command, stdout=log, stderr=log, cwd=REPO)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"server exited, see {log.name}")
            try:
                socket.create_connection((self.host, self.port), 0.2).close()
                return
            except OSError:
                time.sleep(0.1)
        raise RuntimeError("server did not start listening within 60 s")

    def stop(self) -> None:
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

    def connect(self):
        connection = psycopg2.connect(
            host=self.host, port=self.port, user="semcast", dbname="semcast"
        )
        connection.autocommit = True
        return connection


def statements(path: Path, **placeholders: str) -> list[str]:
    """The file's statements. Prompts must not contain a semicolon."""
    lines = [l for l in path.read_text().splitlines() if not l.lstrip().startswith("--")]
    text = "\n".join(lines)
    for key, value in placeholders.items():
        text = text.replace("{" + key + "}", value)
    return [s.strip() for s in text.split(";") if s.strip()]


def prepare_data(scenario: str, scale_factor: int, work: Path) -> Path:
    """Copy the scenario's tables without their ground-truth columns."""
    source = SEMBENCH / "files" / scenario / "data" / f"sf_{scale_factor}"
    if not source.is_dir():
        sys.exit(f"{source} is missing; run eval/setup.sh (it ships sf_2000 only)")
    target = work / f"{scenario}_sf{scale_factor}"
    target.mkdir(parents=True, exist_ok=True)
    for table in source.glob("*.csv"):
        frame = pd.read_csv(table)
        frame = frame.drop(columns=HIDDEN.get(scenario, {}).get(table.name, []))
        frame.to_csv(target / table.name, index=False)
    return target


def funnel_counts(notices: list[str]) -> dict:
    """Model calls and cache hits from each statement's closing funnel line."""
    done = [line for line in notices if "funnel done" in line]
    text = " ".join(done)
    calls = sum(int(n) for n in re.findall(r"(\d+) model calls", text))
    calls += sum(int(n) for n in re.findall(r"scorer: (\d+) calls", text))
    hits = sum(int(n) for n in re.findall(r"(\d+) cache hits", text))
    return {"model_calls": calls, "cache_hits": hits, "funnel": done}


def run_query(server: Server, query: Path, setup: list[str], timeout: float) -> dict:
    """Execute one query file; the last statement's rows are the result."""
    timed_out = threading.Event()

    def expire() -> None:
        timed_out.set()
        server.stop()

    timer = threading.Timer(timeout, expire)
    notices = Notices()
    metric: dict = {"status": "failed", "results": pd.DataFrame()}
    try:
        connection = server.connect()
        cursor = connection.cursor()
        for statement in setup:
            cursor.execute(statement)
        connection.notices = notices
        if server.process is not None:
            timer.start()
        started = time.monotonic()
        for statement in statements(query):
            cursor.execute(statement)
        columns = [column.name for column in cursor.description or []]
        rows = cursor.fetchall() if cursor.description else []
        metric["execution_time"] = time.monotonic() - started
        metric["results"] = pd.DataFrame(rows, columns=columns)
        metric["status"] = "success"
        connection.close()
    except Exception as error:  # a failed query is a result, not a crash
        metric["error"] = f"timed out after {timeout:.0f} s" if timed_out.is_set() else str(error).strip()
    finally:
        timer.cancel()
    metric.update(funnel_counts(notices.lines))
    return metric


def evaluator(scenario: str, scale_factor: int):
    """SemBench's own evaluator for the scenario, unmodified."""
    # generic_evaluator imports cdlib at module level for scenarios this
    # harness does not run; a stub spares a heavy dependency.
    stub = types.ModuleType("cdlib")
    stub.NodeClustering = stub.evaluation = None
    sys.modules.setdefault("cdlib", stub)
    sys.path.insert(0, str(SEMBENCH / "src"))
    if scenario == "movie":
        from scenario.movie.evaluation.evaluate import MovieEvaluator

        return MovieEvaluator(scenario, scale_factor)
    sys.exit(f"no evaluator wired up for scenario {scenario}")


def quality(metric: dict) -> float | None:
    """SemBench's per-query score in [0, 1]; a failed query scores 0."""
    if metric.get("status") != "success" or "error" in metric:
        return 0.0
    if "f1_score" in metric:
        return metric["f1_score"]
    if "relative_error" in metric:
        return 1.0 / (1.0 + metric["relative_error"])
    if "spearman_correlation" in metric:
        return metric["spearman_correlation"]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenario", default="movie", choices=["movie"])
    parser.add_argument("--scale-factor", type=int, default=2000)
    parser.add_argument("--queries", type=int, nargs="+", help="query numbers; all if unset")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--system", default="semcast", help="name the results are filed under")
    parser.add_argument("--queries-dir", type=Path, help="default eval/queries/<scenario>")
    parser.add_argument("--provider", default="ollama", choices=["ollama", "anthropic"])
    parser.add_argument("--model", help="completion model; the server's default if unset")
    parser.add_argument("--server", type=Path, default=REPO / "target/release/semcast")
    parser.add_argument("--server-arg", action="append", default=[], help="extra `semcast serve` argument, repeatable")
    parser.add_argument("--connect", help="host:port of a running server; implies --warm, no timeout")
    parser.add_argument("--warm", action="store_true", help="one server for the whole run")
    parser.add_argument("--timeout", type=float, default=3600, help="seconds per query")
    args = parser.parse_args()

    if not SEMBENCH.is_dir():
        sys.exit("SemBench is missing; run eval/setup.sh")
    if not args.connect and not args.server.exists():
        sys.exit(f"{args.server} is missing; run cargo build --release")
    args.work = EVAL / ".work"
    args.work.mkdir(exist_ok=True)
    queries_dir = args.queries_dir or EVAL / "queries" / args.scenario
    query_ids = args.queries or sorted(
        int(f.stem[1:]) for f in queries_dir.glob("Q*.sql") if f.stem[1:].isdigit()
    )
    data_dir = prepare_data(args.scenario, args.scale_factor, args.work)
    setup = statements(queries_dir / "setup.sql", data_dir=str(data_dir))
    warm = args.warm or bool(args.connect)

    files = SEMBENCH / "files" / args.scenario
    raw = files / "raw_results" / args.system
    metrics_file = files / "metrics" / f"{args.system}.json"
    model = args.model or f"{args.provider} default"
    out = EVAL / "results" / args.scenario / args.system
    scorer = evaluator(args.scenario, args.scale_factor)
    runs: list[dict] = []

    for repeat in range(1, args.repeats + 1):
        shutil.rmtree(raw, ignore_errors=True)
        raw.mkdir(parents=True)
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        server = Server(args)
        run: dict = {}
        if warm:
            server.start()
        for query_id in query_ids:
            if not warm:
                server.start()
            metric = run_query(server, queries_dir / f"Q{query_id}.sql", setup, args.timeout)
            if not warm:
                server.stop()
            results = metric.pop("results")
            results.to_csv(raw / f"Q{query_id}.csv", index=False)
            metric.update(query_id=str(query_id), row_count=len(results), model_name=model,
                          scale_factor=args.scale_factor, cold=not warm)
            run[f"Q{query_id}"] = metric
            print(f"run {repeat} Q{query_id}: {metric['status']}, {len(results)} rows, "
                  f"{metric.get('execution_time', 0):.1f} s, {metric['model_calls']} model calls"
                  + (f" — {metric['error']}" if "error" in metric else ""), flush=True)
        server.stop()
        metrics_file.write_text(json.dumps(run, indent=2))
        scorer.evaluate_system(args.system, queries=query_ids)
        run = json.loads(metrics_file.read_text())
        for metric in run.values():
            metric["quality"] = quality(metric)
        target = out / f"run{repeat}"
        shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(raw, target)
        (target / "metrics.json").write_text(json.dumps(run, indent=2))
        runs.append(run)

    def mean(key: str, name: str) -> float | None:
        values = [run[name][key] for run in runs if run[name].get(key) is not None]
        return statistics.fmean(values) if values else None

    summary = {}
    print(f"\n{args.scenario} sf={args.scale_factor}, {model}, "
          f"{'warm' if warm else 'cold'}, {args.repeats} run(s)")
    print(f"{'query':<6}{'quality':>9}{'seconds':>10}{'calls':>8}{'cache hits':>12}  status")
    for query_id in query_ids:
        name = f"Q{query_id}"
        row = {key: mean(key, name) for key in ("quality", "execution_time", "model_calls", "cache_hits")}
        row["failed_runs"] = sum(run[name]["status"] != "success" for run in runs)
        summary[name] = row
        cells = [f"{row[k]:>{w}.{p}f}" if row[k] is not None else " " * (w - 1) + "-"
                 for k, w, p in (("quality", 9, 3), ("execution_time", 10, 1), ("model_calls", 8, 0), ("cache_hits", 12, 0))]
        status = "ok" if not row["failed_runs"] else f"{row['failed_runs']} failed"
        print(f"{name:<6}{''.join(cells)}  {status}")
    scored = [row["quality"] for row in summary.values() if row["quality"] is not None]
    if scored:
        print(f"{'mean':<6}{statistics.fmean(scored):>9.3f}")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"results in {out}")


if __name__ == "__main__":
    main()
