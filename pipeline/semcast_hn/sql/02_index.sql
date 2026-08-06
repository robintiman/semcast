-- Embed the day's corpus. The expensive step, and the reason everything
-- downstream is cheap: MEANS gets a similarity pre-filter instead of reading
-- every row, and SEMANTIC DISTINCT ON gets its notion of "alike" for free.
CREATE SEMANTIC INDEX ON {staged}(body);
