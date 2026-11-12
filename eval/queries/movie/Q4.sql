SELECT AVG(CASE WHEN positive THEN 1.0 ELSE 0.0 END) AS positivity_ratio
FROM (
  SELECT "reviewText" MEANS 'the movie review is clearly positive' AS positive
  FROM reviews WHERE id = 'taken_3'
) AS labeled;
