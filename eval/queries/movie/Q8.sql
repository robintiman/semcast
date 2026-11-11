SELECT "scoreSentiment", COUNT(*) AS count
FROM (
  SELECT CASE WHEN "reviewText" MEANS 'the movie review is clearly positive' THEN 'POSITIVE' ELSE 'NEGATIVE' END AS "scoreSentiment"
  FROM reviews WHERE id = 'taken_3'
) AS labeled
GROUP BY "scoreSentiment";
