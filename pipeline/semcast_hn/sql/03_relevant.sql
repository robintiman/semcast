-- Filter to what's worth reading and collapse repeated takes, in one pass.
--
-- Both trailing knobs ride the same statement (src/sql/recall.rs, test
-- `knobs_compose`): RECALL calibrates how aggressively the index prunes
-- before the model verifies, SIMILARITY sets how alike two posts must be to
-- count as the same take. HN says the same thing many times a day, so the
-- dedupe is doing real work here — and it costs no model calls at all,
-- because the index already knows how alike two documents are.
--
-- Note the missing ORDER BY. What this wants is `ORDER BY points DESC NULLS
-- LAST`, so the most-upvoted member of each duplicate group is the one that
-- survives. semcast v0.3.0 rejects SEMANTIC DISTINCT ON combined with *any*
-- ORDER BY: the parser wraps each DISTINCT ON expression in a
-- `semantic_key()` marker (src/sql/distinct.rs), so DataFusion's rule that
-- DISTINCT ON expressions must match the leading ORDER BY expressions can
-- never be satisfied — `ORDER BY body` does not match `semantic_key(body)`,
-- and naming the marker explicitly fails too ("semantic_key() is supported
-- in a DISTINCT ON list"). So which row survives is the engine's choice.
-- If that is fixed upstream, add the ORDER BY back.
CREATE TABLE {relevant} AS
SELECT SEMANTIC DISTINCT ON (body)
       id, kind, title, url, author, body, points, created_at
FROM {staged}
WHERE body MEANS '{predicate}'
WITH RECALL {recall}
WITH SIMILARITY {similarity};
