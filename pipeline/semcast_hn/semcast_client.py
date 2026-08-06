"""A thin psycopg2 wrapper that speaks semcast's dialect of Postgres.

Two things here that a normal Postgres helper would not need:

* **psycopg2, not psycopg3.** semcast implements `SimpleQueryHandler` only,
  so a client that binds parameters server-side cannot connect. psycopg2
  interpolates client-side and sends a plain string, which is exactly what
  the simple protocol wants.
* **submit/poll instead of blocking.** A `MEANS` scan is one model call per
  surviving row and a `CREATE SEMANTIC INDEX` embeds a whole corpus — minutes
  to hours. `SUBMIT` detaches the work so a dropped connection doesn't throw
  away model calls already paid for; `semcast_jobs` is an ordinary table that
  reports on it.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Callable

import psycopg2

from . import config

# Statuses from `JobStatus::as_str` in src/server/jobs/mod.rs.
TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}

_IDENT = re.compile(r"\A[A-Za-z_][A-Za-z0-9_]*\Z")


class SemcastJobError(RuntimeError):
    """A submitted job reached a terminal status other than `succeeded`."""


@dataclass
class JobRecord:
    job_id: str
    status: str
    rows: int | None
    command_tag: str | None
    error: str | None
    progress: str | None
    model_calls: int
    extract_model_calls: int
    index_hits: int
    rows_pruned: int
    cache_hits: int
    result_path: str | None

    @property
    def total_model_calls(self) -> int:
        return self.model_calls + self.extract_model_calls


def ident(value: str) -> str:
    """Validate a string that will be interpolated as a SQL identifier.

    Identifiers cannot be passed as parameters, and these are built from
    partition keys, so check rather than trust.
    """
    if not _IDENT.match(value):
        raise ValueError(f"unsafe SQL identifier: {value!r}")
    return value


class Semcast:
    def __init__(self, log: Callable[[str], None] = print) -> None:
        self._log = log
        self._conn = psycopg2.connect(
            host=config.SEMCAST_HOST,
            port=config.SEMCAST_PORT,
            user=config.SEMCAST_USER,
            dbname=config.SEMCAST_DBNAME,
        )
        # BEGIN/COMMIT are no-ops server-side (src/server/router.rs), so an
        # implicit transaction buys nothing but a round trip.
        self._conn.autocommit = True

    def __enter__(self) -> "Semcast":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    def run(self, sql: str) -> list[dict[str, Any]]:
        """Execute one statement and return its rows as dicts."""
        with self._conn.cursor() as cur:
            cur.execute(sql)
            if cur.description is None:
                return []
            names = [c.name for c in cur.description]
            return [dict(zip(names, row)) for row in cur.fetchall()]

    def submit(self, sql: str) -> str:
        """Detach one statement as a batch job and return its id."""
        statement = sql.strip().rstrip(";")
        rows = self.run(f"SUBMIT {statement}")
        if not rows or "job_id" not in rows[0]:
            raise SemcastJobError(f"SUBMIT returned no job_id: {rows!r}")
        job_id = rows[0]["job_id"]
        self._log(f"submitted {job_id}")
        return job_id

    def poll(self, job_id: str) -> JobRecord:
        rows = self.run(
            "SELECT job_id, status, rows, command_tag, error, progress, "
            "model_calls, extract_model_calls, index_hits, rows_pruned, "
            "cache_hits, result_path "
            f"FROM semcast_jobs WHERE job_id = '{ident_literal(job_id)}'"
        )
        if not rows:
            raise SemcastJobError(f"no such job: {job_id}")
        return JobRecord(**rows[0])

    def wait(self, job_id: str) -> JobRecord:
        """Block until a job finishes, logging progress as it goes."""
        last = None
        while True:
            record = self.poll(job_id)
            if record.progress and record.progress != last:
                self._log(f"{job_id}: {record.progress}")
                last = record.progress
            if record.status in TERMINAL:
                break
            time.sleep(config.POLL_SECONDS)

        if record.status != "succeeded":
            raise SemcastJobError(
                f"{job_id} ended as {record.status}: {record.error or 'no detail'}"
            )
        self._log(
            f"{job_id} succeeded: {record.rows} rows, "
            f"{record.total_model_calls} model calls, "
            f"{record.cache_hits} cache hits, {record.rows_pruned} rows pruned"
        )
        return record

    def submit_and_wait(self, sql: str) -> JobRecord:
        return self.wait(self.submit(sql))


def ident_literal(value: str) -> str:
    """Escape a value destined for single quotes in generated SQL."""
    return value.replace("'", "''")
