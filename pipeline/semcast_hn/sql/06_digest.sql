-- Extract the structured digest, one row per surviving post.
--
-- The MEANS verdicts from step 03 are already cached, so the model runs here
-- only to extract — it does not re-decide relevance.
CREATE TABLE {digest} AS
SELECT id,
       kind,
       title,
       url,
       author,
       points,
       created_at,
       CAST(body AS {type_name}).claim    AS claim,
       CAST(body AS {type_name}).tools    AS tools,
       CAST(body AS {type_name}).maturity AS maturity,
       CAST(body AS {type_name}).evidence AS evidence,
       body
FROM {relevant};
