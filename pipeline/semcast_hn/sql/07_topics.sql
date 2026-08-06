-- What the day was about, without deciding the categories in advance.
--
-- Clusters the index's vectors, then spends one model call per group to name
-- it — cost tracks the number of groups, not the number of rows. Seeded, so
-- the same corpus always yields the same groups.
CREATE TABLE {topics} AS
SELECT meaning_of(body, {k}) AS topic, count(*) AS posts
FROM {relevant}
GROUP BY meaning_of(body, {k})
ORDER BY posts DESC;
