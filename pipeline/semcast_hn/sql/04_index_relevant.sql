-- Re-index the survivors. An index is per table(column), and the topic
-- rollup clusters this table rather than the staged one — grouping should
-- describe what got through, not what got filtered out.
--
-- Cheap despite being a second embed pass: it covers the handful of rows
-- that survived MEANS, not the day's corpus.
CREATE SEMANTIC INDEX ON {relevant}(body);
