"""Fetch a day of Hacker News from the Algolia API and normalize it.

Algolia rather than the Firebase API because this pipeline is time-windowed:
`search_by_date` takes a `created_at_i` range directly, where Firebase would
mean walking item ids and filtering client-side. No key, no rate limit worth
worrying about at one run a day.
"""

from __future__ import annotations

import html
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator

import pyarrow as pa
import requests

SEARCH_URL = "https://hn.algolia.com/api/v1/search_by_date"
PAGE_SIZE = 1000
TIMEOUT = 30

SCHEMA = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("kind", pa.string(), nullable=False),
        pa.field("title", pa.string()),
        pa.field("url", pa.string()),
        pa.field("author", pa.string()),
        # Named `body`, not `text`: it is the argument to CAST(.. AS <Type>)
        # and EXTRACT(.. FROM ..), where `text` reads ambiguously against the
        # TEXT type keyword.
        pa.field("body", pa.string(), nullable=False),
        pa.field("points", pa.int64()),
        pa.field("created_at", pa.timestamp("s", tz="UTC"), nullable=False),
        pa.field("story_id", pa.string()),
    ]
)

_TAG = re.compile(r"<[^>]+>")


def day_bounds(day: str) -> tuple[int, int]:
    """UTC epoch-second bounds for a `YYYY-MM-DD` partition key."""
    start = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(start.timestamp()), int((start + timedelta(days=1)).timestamp())


def _strip_html(text: str | None) -> str:
    """Algolia returns comment bodies as HTML; the model should read prose."""
    if not text:
        return ""
    text = text.replace("<p>", "\n\n").replace("</p>", "")
    return html.unescape(_TAG.sub("", text)).strip()


def _fetch_tag(tag: str, start: int, end: int) -> Iterator[dict[str, Any]]:
    """Walk one tag's results backwards in time.

    Cursor pagination on `created_at_i` rather than `page`, because Algolia
    caps how deep paging can go and a busy day of comments is deeper than
    that cap.
    """
    cursor = end
    seen: set[str] = set()
    while cursor > start:
        response = requests.get(
            SEARCH_URL,
            params={
                "tags": tag,
                "numericFilters": f"created_at_i>{start},created_at_i<{cursor}",
                "hitsPerPage": PAGE_SIZE,
            },
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        hits = response.json().get("hits", [])
        fresh = [h for h in hits if h["objectID"] not in seen]
        if not fresh:
            return
        for hit in fresh:
            seen.add(hit["objectID"])
            yield hit
        # Algolia sorts newest-first, so the oldest hit is the next cursor.
        oldest = min(h["created_at_i"] for h in fresh)
        if oldest >= cursor:
            return
        cursor = oldest


def _normalize(hit: dict[str, Any], kind: str) -> dict[str, Any] | None:
    if kind == "story":
        body = "\n\n".join(
            part for part in (hit.get("title"), _strip_html(hit.get("story_text"))) if part
        )
        title, url, story_id = hit.get("title"), hit.get("url"), None
    else:
        body = _strip_html(hit.get("comment_text"))
        title = hit.get("story_title")
        url = hit.get("story_url")
        story_id = str(hit["story_id"]) if hit.get("story_id") is not None else None

    if not body:
        return None
    return {
        "id": str(hit["objectID"]),
        "kind": kind,
        "title": title,
        "url": url,
        "author": hit.get("author"),
        "body": body,
        "points": hit.get("points"),
        "created_at": datetime.fromtimestamp(hit["created_at_i"], tz=timezone.utc),
        "story_id": story_id,
    }


def fetch_day(day: str) -> pa.Table:
    """Every story and comment created on `day`, as one Arrow table."""
    start, end = day_bounds(day)
    rows = [
        row
        for kind in ("story", "comment")
        for hit in _fetch_tag(kind, start, end)
        if (row := _normalize(hit, kind)) is not None
    ]
    return pa.Table.from_pylist(rows, schema=SCHEMA)
