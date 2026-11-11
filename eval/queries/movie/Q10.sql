SELECT id AS "movieId", AVG(score) AS "movieScore"
FROM (
  SELECT id, EXTRACT(score INT 'how much the reviewer liked the movie, from 1 to 5. 5 is very positive: strong positive sentiment, high satisfaction. 4 is positive: noticeably positive sentiment, general satisfaction. 3 is neutral: no clear positive or negative sentiment. 2 is negative: noticeably negative sentiment, some dissatisfaction. 1 is very negative: strong negative sentiment, frustration or anger' FROM "reviewText") AS score
  FROM reviews
) AS scored
GROUP BY id;
