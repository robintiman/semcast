-- What to pull out of each surviving post.
--
-- The name carries the partition suffix because a semantic type is defined
-- once per session and there is no CREATE OR REPLACE
-- (src/types/registry.rs) — on a long-lived server, a bare `HnFinding`
-- would fail on the second day's run. Suffixing also means an edit to these
-- doc lines takes effect on the next run instead of being silently ignored
-- in favour of yesterday's definition.
--
-- The doc line on each field *is* the prompt, so it earns its wording.
-- TOGETHER keeps maturity and its supporting quote in one generation: a
-- rating produced apart from its evidence tends not to match it.
CREATE SEMANTIC TYPE {type_name} AS (
  claim    TEXT   'the main technical claim or finding, in one sentence',
  tools    TEXT[] 'names of specific tools, libraries, models, or databases mentioned',
  TOGETHER (
    maturity LEVEL(speculation, experiment, production)
                    'how battle-tested the thing described is',
    evidence TEXT    'the sentence that shows that maturity'
  )
);
