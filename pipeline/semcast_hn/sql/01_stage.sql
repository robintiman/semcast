-- Land the day's raw Parquet as a table, dropping one-liners first.
--
-- This LENGTH filter is the cheapest thing in the pipeline and the one that
-- most affects the bill: it runs before the semantic index is built, so
-- every row it drops is an embedding never paid for.
CREATE TABLE {staged} AS
SELECT id, kind, title, url, author, body, points, created_at, story_id
FROM '{raw_path}'
WHERE length(body) > {min_len};
