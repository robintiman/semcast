"""Load and render the SQL templates in ``sql/``.

One statement per file, deliberately: semcast speaks the simple query
protocol, and splitting a multi-statement string on ``;`` would break the
moment a string literal contained one — which the tunable MEANS predicate
easily could.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

SQL_DIR = Path(__file__).parent / "sql"


@lru_cache(maxsize=None)
def _template(name: str) -> str:
    return (SQL_DIR / f"{name}.sql").read_text()


def quote(value: str) -> str:
    """Escape a value being interpolated into a single-quoted SQL literal."""
    return value.replace("'", "''")


def _strip_leading_comments(sql: str) -> str:
    """Drop leading `--` lines before sending.

    semcast's custom DDL (`CREATE SEMANTIC INDEX`, `CREATE SEMANTIC TYPE`)
    dispatches on the statement's first tokens and does not skip comments, so
    a documented template fails to parse with "Expected: an object type after
    CREATE, found: SEMANTIC". Ordinary DataFusion DDL is unbothered, but this
    applies to every template so no file is a special case.

    Only *leading* lines are examined, which is what makes this safe: nothing
    before the first statement token can be inside a string literal.
    """
    lines = sql.splitlines()
    start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("--"):
            start = i
            break
    return "\n".join(lines[start:]).strip()


def render(name: str, **params: object) -> str:
    """Render a template. Identifiers must already be validated by `ident`."""
    return _strip_leading_comments(_template(name)).format(**params)
