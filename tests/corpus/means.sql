-- The binder-gate corpus: MEANS statements a user writes, one per entry.
-- Measured by tests/binder_coverage.rs: the share that plans with every
-- MEANS turned into an ASK node is the phase 2 binder gate (>= 90%).
--
-- Entries are separated by a line holding only `--- <source>`, which names
-- where the statement comes from. Deliberate error tests are left out: they
-- probe the error, not a query anyone means to run. The `analyst` entries
-- are hand-written to cover shapes the other sources don't, including ones
-- semcast rejects today; they are what keeps the number honest.

--- README: MEANS
SELECT meeting_id, title FROM meetings
WHERE held_at >= CAST('2026-01-01' AS TIMESTAMP)
  AND transcript MEANS 'discussed the launch of offline sync in Atlas'
--- README: MEANS in a SELECT list
SELECT id, CASE WHEN status = 'closed'             THEN 'archived'
                WHEN body MEANS 'customer is angry' THEN 'escalate'
                WHEN body MEANS 'asks about pricing' THEN 'sales'
                ELSE 'other' END AS route
FROM tickets
--- README: MEANS in a SELECT list
SELECT id, body MEANS 'customer is angry' AS is_angry FROM tickets
--- README: CONFIDENCE()
SELECT id, CONFIDENCE(body MEANS 'customer is angry') AS anger
FROM tickets
WHERE CONFIDENCE(body MEANS 'customer is angry') > 0.8
ORDER BY anger DESC
--- README: UPDATE/DELETE workaround
SELECT * FROM tickets
WHERE id NOT IN (SELECT id FROM tickets WHERE body MEANS 'resolved')
--- README: WITH RECALL
SELECT meeting_id FROM meetings
WHERE transcript MEANS 'discussed offline sync'
WITH RECALL 0.9
--- README: WITH CONFIDENCE
SELECT meeting_id FROM meetings
WHERE transcript MEANS 'discussed offline sync'
WITH RECALL 0.9 WITH CONFIDENCE 0.95
--- README: typed extraction
SELECT meeting_id, CAST(transcript AS MeetingFacts).launch_stage AS stage
FROM meetings
WHERE transcript MEANS 'discussed the launch of offline sync in Atlas'
--- README: CREATE TABLE AS
SELECT meeting_id, title FROM meetings
WHERE transcript MEANS 'discussed the launch of offline sync in Atlas'
--- examples/complaints.rs
SELECT id FROM tickets
WHERE priority = 'high'
  AND body MEANS 'the consumer reports identity theft or an account they never opened'
--- tests: two conditions, one column
SELECT id FROM tickets WHERE body MEANS 'is angry' AND body MEANS 'asks for a refund'
--- tests: two columns
SELECT id FROM tickets WHERE body MEANS 'is angry' AND subject MEANS 'billing'
--- tests: CASE label
SELECT id, CASE WHEN body MEANS 'asks for a refund' THEN 1 END FROM tickets
--- tests: CONFIDENCE in ORDER BY
SELECT id FROM tickets WHERE body IS NOT NULL
ORDER BY CONFIDENCE(body MEANS 'is angry') ASC
--- tests: CONFIDENCE in a CASE
SELECT id, CASE WHEN id = 2 THEN 'skip'
                WHEN CONFIDENCE(body MEANS 'is angry') > 0.5 THEN 'angry' END
FROM tickets
--- analyst: negation
SELECT id FROM tickets WHERE NOT body MEANS 'is spam'
--- analyst: either condition
SELECT id FROM tickets
WHERE body MEANS 'mentions a competitor' OR subject MEANS 'mentions a competitor'
--- analyst: a cheap condition or a semantic one
SELECT id FROM tickets WHERE priority = 'urgent' OR body MEANS 'threatens to cancel'
--- analyst: join, filter on one side
SELECT c.name, t.id FROM tickets t JOIN customers c ON t.customer_id = c.id
WHERE t.body MEANS 'asks for a refund'
--- analyst: join, filter on both sides
SELECT c.name, t.id FROM tickets t JOIN customers c ON t.customer_id = c.id
WHERE t.body MEANS 'asks for a refund' AND c.notes MEANS 'enterprise account'
--- analyst: CTE
WITH angry AS (SELECT * FROM tickets WHERE body MEANS 'is angry')
SELECT customer_id, count(*) AS n FROM angry GROUP BY customer_id
--- analyst: count by label
SELECT CASE WHEN body MEANS 'is angry' THEN 'angry' ELSE 'calm' END AS mood,
       count(*) AS n
FROM tickets GROUP BY 1
--- analyst: aggregate over a filtered set
SELECT customer_id, count(*) FROM tickets
WHERE body MEANS 'reports a bug' GROUP BY customer_id HAVING count(*) > 2
--- analyst: semantic condition in HAVING
SELECT customer_id, string_agg(body, ' ') AS all_text FROM tickets
GROUP BY customer_id HAVING string_agg(body, ' ') MEANS 'a pattern of complaints'
--- analyst: EXISTS subquery
SELECT c.name FROM customers c
WHERE EXISTS (SELECT 1 FROM tickets t WHERE t.customer_id = c.id
              AND t.body MEANS 'threatens to cancel')
--- analyst: IN subquery
SELECT name FROM customers
WHERE id IN (SELECT customer_id FROM tickets WHERE body MEANS 'asks for a refund')
--- analyst: scalar subquery
SELECT c.name,
       (SELECT count(*) FROM tickets t
        WHERE t.customer_id = c.id AND t.body MEANS 'is angry') AS angry_tickets
FROM customers c
--- analyst: UNION
SELECT id FROM tickets WHERE body MEANS 'is angry'
UNION
SELECT id FROM tickets WHERE subject MEANS 'urgent'
--- analyst: ORDER BY and LIMIT
SELECT id, created_at FROM tickets WHERE body MEANS 'reports an outage'
ORDER BY created_at DESC LIMIT 10
--- analyst: computed text
SELECT id FROM tickets WHERE subject || ' ' || body MEANS 'is about billing'
--- analyst: label and filter together
SELECT id, body MEANS 'is urgent' AS urgent FROM tickets
WHERE body MEANS 'reports a bug'
--- analyst: window over a label
SELECT id, sum(CASE WHEN body MEANS 'is angry' THEN 1 ELSE 0 END)
       OVER (PARTITION BY customer_id) AS angry_for_customer
FROM tickets
--- analyst: average confidence per customer
SELECT customer_id, avg(anger) FROM (
  SELECT customer_id, CONFIDENCE(body MEANS 'is angry') AS anger FROM tickets
) GROUP BY customer_id
--- analyst: average confidence, inline
SELECT customer_id, avg(CONFIDENCE(body MEANS 'is angry')) FROM tickets
GROUP BY customer_id
--- analyst: DISTINCT
SELECT DISTINCT customer_id FROM tickets WHERE body MEANS 'asks for a refund'
--- analyst: view
SELECT id FROM angry_tickets WHERE created_at > CAST('2026-01-01' AS TIMESTAMP)
--- analyst: semantic join condition
SELECT t.id, c.id FROM tickets t JOIN customers c
  ON t.customer_id = c.id AND t.body MEANS 'mentions the account manager by name'
--- analyst: IN list of a label
SELECT id FROM tickets
WHERE CASE WHEN body MEANS 'is angry' THEN 'a' ELSE 'b' END IN ('a')
--- analyst: NOT with a cheap condition
SELECT id FROM tickets WHERE status = 'open' AND NOT body MEANS 'already resolved'
--- analyst: LATERAL-free correlated filter
SELECT id FROM tickets t
WHERE t.body MEANS 'is angry'
  AND t.customer_id IN (SELECT id FROM customers WHERE notes MEANS 'at risk of churn')
