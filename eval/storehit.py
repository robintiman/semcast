#!/usr/bin/env python3
"""Measure the share of semantic questions the store answers over a session.

    cargo build --release
    uv run --project eval eval/storehit.py movie
    uv run --project eval eval/storehit.py cfpb --data data/cfpb_complaints.csv

One server, one fresh store file, the session's statements in order. A
question is one (condition, text) the engine needs answered: either the store
had it, or the model was asked. Read from each statement's funnel NOTICE.
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
import time
from pathlib import Path

from sembench import EVAL, REPO, Notices, Server, prepare_data, statements


def movie_session(args: argparse.Namespace) -> tuple[list[str], list[tuple[str, str]]]:
    """SemBench Movie's queries in order, every statement of each."""
    data = prepare_data("movie", 2000, args.work)
    folder = EVAL / "queries" / "movie"
    setup = statements(folder / "setup.sql", data_dir=str(data))
    steps = []
    for query in sorted(folder.glob("Q*.sql"), key=lambda p: int(p.stem[1:])):
        steps += [(query.stem, s) for s in statements(query)]
    return setup, steps


def file_session(path: Path, data: str) -> tuple[list[str], list[tuple[str, str]]]:
    """Statements after `-- setup` or `-- step: <label>` lines."""
    setup, steps, label, lines = [], [], None, []

    def flush() -> None:
        text = "\n".join(lines).strip().rstrip(";").replace("{data}", data)
        if label == "setup":
            setup.append(text)
        elif label is not None:
            steps.append((label, text))

    for line in path.read_text().splitlines():
        marker = re.match(r"--\s*(setup|step:\s*(.*))\s*$", line)
        if marker:
            flush()
            label, lines = marker.group(2) or "setup", []
        elif not line.lstrip().startswith("--"):
            lines.append(line)
    flush()
    return setup, steps


def counts(notices: list[str]) -> dict:
    done = " ".join(line for line in notices if "funnel done" in line)
    stored = sum(int(n) for n in re.findall(r"store: (\d+) of", done))
    stored += sum(int(n) for n in re.findall(r"(\d+) cache hits", done))
    asked = sum(int(n) for n in re.findall(r"(\d+) model calls", done))
    asked += sum(int(n) for n in re.findall(r"scorer: (\d+) calls", done))
    return {"stored": stored, "asked": asked}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("session", help="movie, or a file name under queries/sessions/")
    parser.add_argument("--data", help="CSV the session's {data} placeholder names")
    parser.add_argument("--model", help="completion model passed to semcast serve")
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--server", type=Path, default=REPO / "target" / "release" / "semcast")
    parser.add_argument("--server-arg", action="append", default=[])
    args = parser.parse_args()
    args.connect = None
    args.work = Path(tempfile.mkdtemp(prefix="storehit-"))
    if any(a.startswith("--store") for a in args.server_arg):
        parser.error("the session owns --store: it starts from an empty one")
    if args.session != "movie" and not args.data:
        parser.error("this session needs --data")
    if not args.server.exists():
        parser.error(f"{args.server} is missing; run cargo build --release")
    args.server_arg.append(f"--store={args.work / 'store.db'}")

    if args.session == "movie":
        setup, steps = movie_session(args)
    else:
        path = EVAL / "queries" / "sessions" / f"{args.session}.sql"
        setup, steps = file_session(path, str(Path(args.data).resolve()))

    server = Server(args, setup)
    rows = []
    try:
        server.ensure()
        connection = server.connect()
        cursor = connection.cursor()
        for label, statement in steps:
            notices = Notices()
            connection.notices = notices
            started = time.monotonic()
            row = {"step": label}
            try:
                cursor.execute(statement)
                if cursor.description:
                    cursor.fetchall()
            except Exception as error:  # a failed step is a result, not a crash
                row["error"] = str(error).strip()
            row["seconds"] = round(time.monotonic() - started, 1)
            row.update(counts(notices.lines))
            rows.append(row)
            failed = f"  FAILED: {row['error']}" if "error" in row else ""
            print(f"{row['stored']:>6} stored {row['asked']:>6} asked {row['seconds']:>7.1f} s  {label}{failed}", flush=True)
        connection.close()
    finally:
        server.stop()

    stored = sum(r["stored"] for r in rows)
    asked = sum(r["asked"] for r in rows)
    share = stored / (stored + asked) if stored + asked else 0.0
    print(f"store answered {stored} of {stored + asked} questions: {share:.1%}")
    out = EVAL / "results" / "storehit" / f"{args.session}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"stored": stored, "asked": asked, "share": share, "steps": rows}, indent=2) + "\n")
    print(f"wrote {out.relative_to(REPO)}; server log in {args.work / 'server.log'}")


if __name__ == "__main__":
    main()
