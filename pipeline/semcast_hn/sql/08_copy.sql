-- Persist. Not bookkeeping: CREATE TABLE AS lands in an in-memory table
-- that dies with the server process, so anything wanted tomorrow has to hit
-- disk today. Everything downstream reads this Parquet, never the live
-- table.
COPY {table} TO '{path}' STORED AS PARQUET;
