"""Settings, all overridable by environment variable.

The one that matters is ``SEMCAST_DATA_ROOT``. Paths in semcast SQL resolve
on the *server*, not on the client, so this is the root as the semcast
process sees it. Dagster and semcast are assumed to be co-located, so the
same string works for both the Parquet writer here and the SQL over there.
Split them across hosts and these two become different values pointing at
one shared mount.
"""

import os
from pathlib import Path

# --- connection -----------------------------------------------------------

# semcast implements only the simple query protocol (`SimpleQueryHandler`),
# so the client has to be psycopg2 — psycopg3 and asyncpg bind parameters
# server-side over the extended protocol and will fail to connect.
SEMCAST_HOST = os.environ.get("SEMCAST_HOST", "127.0.0.1")
SEMCAST_PORT = int(os.environ.get("SEMCAST_PORT", "5433"))
SEMCAST_USER = os.environ.get("SEMCAST_USER", "semcast")
SEMCAST_DBNAME = os.environ.get("SEMCAST_DBNAME", "semcast")

# --- paths (server-side) --------------------------------------------------

DATA_ROOT = Path(os.environ.get("SEMCAST_DATA_ROOT", "data"))
RAW_ROOT = DATA_ROOT / "hn"
OUT_ROOT = Path(os.environ.get("SEMCAST_OUT_ROOT", "out")) / "hn_digest"

# --- pipeline knobs -------------------------------------------------------

START_DATE = os.environ.get("HN_START_DATE", "2026-08-01")

# What counts as worth reading. This string is the whole point of the
# pipeline; tune it here rather than in the SQL.
MEANS_PREDICATE = os.environ.get(
    "HN_MEANS",
    "substantive discussion of semantic data processing: embeddings, vector "
    "search, retrieval, semantic indexing, or using language models to filter, "
    "extract from, or transform data",
)

# Drop one-liners before spending a single embedding on them. The cheapest
# filter in the pipeline and the one that most affects the bill.
MIN_TEXT_LEN = int(os.environ.get("HN_MIN_TEXT_LEN", "120"))

# Fraction of true matches WITH RECALL should preserve when calibrating the
# index-pruning threshold.
RECALL = float(os.environ.get("HN_RECALL", "0.9"))

# Higher is stricter: how alike two posts must be to count as the same take.
SIMILARITY = float(os.environ.get("HN_SIMILARITY", "0.85"))

# Number of topic clusters for the rollup.
TOPIC_K = int(os.environ.get("HN_TOPIC_K", "6"))

# How often to poll `semcast_jobs` while a submitted job runs.
POLL_SECONDS = float(os.environ.get("SEMCAST_POLL_SECONDS", "10"))
