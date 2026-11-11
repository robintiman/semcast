-- semcast has no semantic join: label each review once, then join on the label.
-- Pairs are unordered and review ids repeat in the data: return each pair once.
WITH labeled AS (
  SELECT id, "reviewId", "reviewText" MEANS 'the movie review is clearly positive' AS positive
  FROM reviews WHERE id = 'ant_man_and_the_wasp_quantumania'
)
SELECT DISTINCT r1.id, r1."reviewId" AS "reviewId1", r2."reviewId" AS "reviewId2"
FROM labeled AS r1
JOIN labeled AS r2 ON r1.id = r2.id AND r1."reviewId" < r2."reviewId"
WHERE r1.positive <> r2.positive
LIMIT 10;
