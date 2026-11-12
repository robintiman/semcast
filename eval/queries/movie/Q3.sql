SELECT COUNT(*) AS positive_review_cnt FROM reviews
WHERE id = 'taken_3' AND "reviewText" MEANS 'the movie review is clearly positive';
