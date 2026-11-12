-- An analyst on a credit-card team, two weeks of work over CFPB complaints.
-- Written before any run and not tuned after: the store-hit gate measures it.
-- Each statement follows a `-- step:` line naming what an analyst is doing.
-- Week 1 sees complaints received before 2026-06-08 (841 rows), week 2 all
-- 1123: the rows in between are the week's new data.

-- setup
CREATE EXTERNAL TABLE complaints STORED AS CSV LOCATION '{data}';
-- setup
CREATE VIEW cc_week1 AS SELECT * FROM complaints
WHERE product = 'Credit card' AND date_received < '2026-06-08';
-- setup
CREATE VIEW cc AS SELECT * FROM complaints WHERE product = 'Credit card';

-- step: explore a question, reading examples
SELECT complaint_id, narrative FROM cc_week1
WHERE narrative MEANS 'the customer was charged a fee they did not expect'
LIMIT 20;

-- step: same question, now counted by issue
SELECT issue, COUNT(*) AS n FROM cc_week1
WHERE narrative MEANS 'the customer was charged a fee they did not expect'
GROUP BY issue ORDER BY n DESC;

-- step: reword the question after reading the examples
SELECT issue, COUNT(*) AS n FROM cc_week1
WHERE narrative MEANS 'the customer disputes a fee or interest charge'
GROUP BY issue ORDER BY n DESC;

-- step: a second question
SELECT company, COUNT(*) AS n FROM cc_week1
WHERE narrative MEANS 'the customer says the company ignored or rejected their dispute'
GROUP BY company ORDER BY n DESC;

-- step: both questions together
SELECT complaint_id, company FROM cc_week1
WHERE narrative MEANS 'the customer disputes a fee or interest charge'
  AND narrative MEANS 'the customer says the company ignored or rejected their dispute';

-- step: the second question as a rate per company, for a chart
SELECT company,
       AVG(CASE WHEN narrative MEANS 'the customer says the company ignored or rejected their dispute'
                THEN 1.0 ELSE 0.0 END) AS ignored_rate,
       COUNT(*) AS n
FROM cc_week1 GROUP BY company HAVING COUNT(*) >= 10 ORDER BY ignored_rate DESC;

-- step: week 2, the fee dashboard reruns over new data
SELECT issue, COUNT(*) AS n FROM cc
WHERE narrative MEANS 'the customer disputes a fee or interest charge'
GROUP BY issue ORDER BY n DESC;

-- step: week 2, the rate chart reruns over new data
SELECT company,
       AVG(CASE WHEN narrative MEANS 'the customer says the company ignored or rejected their dispute'
                THEN 1.0 ELSE 0.0 END) AS ignored_rate,
       COUNT(*) AS n
FROM cc GROUP BY company HAVING COUNT(*) >= 10 ORDER BY ignored_rate DESC;

-- step: week 2, a new question from a manager
SELECT state, COUNT(*) AS n FROM cc
WHERE narrative MEANS 'the customer reports fraud or identity theft'
GROUP BY state ORDER BY n DESC;

-- step: week 2, the new question split by the old one
SELECT narrative MEANS 'the customer says the company ignored or rejected their dispute' AS ignored,
       COUNT(*) AS n
FROM cc WHERE narrative MEANS 'the customer reports fraud or identity theft'
GROUP BY 1;
