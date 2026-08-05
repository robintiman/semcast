"""The daily pipeline, as five partitioned Dagster assets.

Partition keys are dates, so "re-run last Tuesday" and "backfill the week"
are built in — worth having the first time a provider times out mid-run.

Every semcast object is suffixed with the partition key. semcast builds one
`SessionContext` for the whole process (src/bin/semcast.rs), which is what
lets these assets each open their own connection and still see the previous
one's tables — but it also means one global namespace with no isolation, so
a backfill running two days at once would collide on unsuffixed names.
"""

# No `from __future__ import annotations` here: it stringifies the
# `AssetExecutionContext` annotation, which Dagster resolves at decoration
# time and rejects. Python 3.11 handles the built-in generics below natively.
from pathlib import Path

import pyarrow.parquet as pq
from dagster import (
    AssetExecutionContext,
    DailyPartitionsDefinition,
    MaterializeResult,
    MetadataValue,
    asset,
)

from . import config, hn, sql
from .semcast_client import JobRecord, Semcast, ident

daily = DailyPartitionsDefinition(start_date=config.START_DATE)


def _suffix(day: str) -> str:
    """`2026-08-05` -> `20260805`, validated for use as an identifier."""
    return ident("d" + day.replace("-", ""))


def _names(day: str) -> dict[str, str]:
    s = _suffix(day)
    return {
        "staged": f"hn_staged_{s}",
        "relevant": f"hn_relevant_{s}",
        "digest": f"hn_digest_{s}",
        "topics": f"hn_topics_{s}",
        "type_name": f"HnFinding_{s}",
    }


def _raw_path(day: str) -> Path:
    return config.RAW_ROOT / f"dt={day}" / "items.parquet"


def _out_path(day: str, name: str) -> Path:
    return config.OUT_ROOT / f"dt={day}" / f"{name}.parquet"


def _job_metadata(record: JobRecord) -> dict[str, MetadataValue]:
    """Cost and selectivity, surfaced on the asset rather than buried in logs.

    `model_calls` is the daily bill; `cache_hits` and `rows_pruned` are what
    kept it down.
    """
    return {
        "rows": MetadataValue.int(record.rows or 0),
        "model_calls": MetadataValue.int(record.total_model_calls),
        "cache_hits": MetadataValue.int(record.cache_hits),
        "rows_pruned": MetadataValue.int(record.rows_pruned),
        "index_hits": MetadataValue.int(record.index_hits),
    }


@asset(partitions_def=daily, group_name="hn", compute_kind="python")
def hn_raw(context: AssetExecutionContext) -> MaterializeResult:
    """A day of Hacker News stories and comments, as Parquet."""
    day = context.partition_key
    table = hn.fetch_day(day)

    path = _raw_path(day)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)

    # semcast resolves this path itself, so log where it actually landed.
    context.log.info(f"wrote {table.num_rows} rows to {path.resolve()}")
    return MaterializeResult(
        metadata={
            "rows": MetadataValue.int(table.num_rows),
            "path": MetadataValue.path(str(path.resolve())),
        }
    )


@asset(partitions_def=daily, deps=[hn_raw], group_name="hn", compute_kind="semcast")
def hn_staged(context: AssetExecutionContext) -> MaterializeResult:
    """The day's corpus, length-filtered and semantically indexed."""
    day = context.partition_key
    names = _names(day)

    with Semcast(log=context.log.info) as db:
        db.run(
            sql.render(
                "01_stage",
                staged=names["staged"],
                raw_path=sql.quote(str(_raw_path(day))),
                min_len=config.MIN_TEXT_LEN,
            )
        )
        staged_rows = db.run(f"SELECT count(*) AS n FROM {names['staged']}")[0]["n"]
        context.log.info(f"staged {staged_rows} rows; embedding them now")

        record = db.submit_and_wait(sql.render("02_index", staged=names["staged"]))

    return MaterializeResult(
        metadata={"staged_rows": MetadataValue.int(staged_rows), **_job_metadata(record)}
    )


@asset(partitions_def=daily, deps=[hn_staged], group_name="hn", compute_kind="semcast")
def hn_relevant(context: AssetExecutionContext) -> MaterializeResult:
    """Posts that actually discuss semantic data processing, deduplicated."""
    day = context.partition_key
    names = _names(day)

    with Semcast(log=context.log.info) as db:
        record = db.submit_and_wait(
            sql.render(
                "03_relevant",
                relevant=names["relevant"],
                staged=names["staged"],
                predicate=sql.quote(config.MEANS_PREDICATE),
                recall=config.RECALL,
                similarity=config.SIMILARITY,
            )
        )
        # Needed by the topic rollup: an index is per table(column), and
        # clustering should describe the survivors, not the whole corpus.
        db.submit_and_wait(sql.render("04_index_relevant", relevant=names["relevant"]))

    return MaterializeResult(metadata=_job_metadata(record))


@asset(partitions_def=daily, deps=[hn_relevant], group_name="hn", compute_kind="semcast")
def hn_digest(context: AssetExecutionContext) -> MaterializeResult:
    """Structured findings — claim, tools, maturity — one row per post."""
    day = context.partition_key
    names = _names(day)
    out = _out_path(day, "digest")
    out.parent.mkdir(parents=True, exist_ok=True)

    with Semcast(log=context.log.info) as db:
        db.run(sql.render("05_type", type_name=names["type_name"]))
        record = db.submit_and_wait(
            sql.render(
                "06_digest",
                digest=names["digest"],
                relevant=names["relevant"],
                type_name=names["type_name"],
            )
        )
        # Guard before persisting. A failed extraction does not fail the
        # query — it yields NULLs — so an unreachable model provider would
        # otherwise land a green asset holding an empty digest.
        counts = db.run(
            f"SELECT count(*) AS total, count(claim) AS extracted FROM {names['digest']}"
        )[0]
        total, extracted = counts["total"], counts["extracted"]
        if total and not extracted:
            raise RuntimeError(
                f"extraction produced no values across {total} rows — "
                "the model provider is probably unreachable"
            )
        null_rate = 1 - (extracted / total) if total else 0.0
        if null_rate > 0.5:
            context.log.warning(
                f"{null_rate:.0%} of rows extracted no claim; check the model provider"
            )

        db.run(
            sql.render("08_copy", table=names["digest"], path=sql.quote(str(out)))
        )
        preview = db.run(
            f"SELECT title, claim, maturity FROM {names['digest']} "
            "ORDER BY points DESC NULLS LAST LIMIT 10"
        )

    return MaterializeResult(
        metadata={
            **_job_metadata(record),
            "path": MetadataValue.path(str(out.resolve())),
            "extraction_null_rate": MetadataValue.float(round(null_rate, 3)),
            "top_findings": MetadataValue.md(_as_markdown(preview)),
        }
    )


@asset(partitions_def=daily, deps=[hn_relevant], group_name="hn", compute_kind="semcast")
def hn_topics(context: AssetExecutionContext) -> MaterializeResult:
    """What the day was about, clustered and named by the model."""
    day = context.partition_key
    names = _names(day)
    out = _out_path(day, "topics")
    out.parent.mkdir(parents=True, exist_ok=True)

    with Semcast(log=context.log.info) as db:
        record = db.submit_and_wait(
            sql.render(
                "07_topics",
                topics=names["topics"],
                relevant=names["relevant"],
                k=config.TOPIC_K,
            )
        )
        db.run(sql.render("08_copy", table=names["topics"], path=sql.quote(str(out))))
        rollup = db.run(f"SELECT topic, posts FROM {names['topics']} ORDER BY posts DESC")

    return MaterializeResult(
        metadata={
            **_job_metadata(record),
            "path": MetadataValue.path(str(out.resolve())),
            "topics": MetadataValue.md(_as_markdown(rollup)),
        }
    )


def _as_markdown(rows: list[dict[str, object]]) -> str:
    """Render query rows as a Markdown table for the Dagster asset page."""
    if not rows:
        return "_no rows_"
    headers = list(rows[0])

    def cell(value: object) -> str:
        return str("" if value is None else value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines += ["| " + " | ".join(cell(row[h]) for h in headers) + " |" for row in rows]
    return "\n".join(lines)
