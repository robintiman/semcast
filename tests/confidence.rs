//! Verdict probabilities: a yes/no verdict is cached with the P("yes") the
//! model reported for it, without changing the verdict entry itself.

use std::sync::{Arc, Mutex};

use semcast::SemcastContextBuilder;
use semcast::cache::{CacheKey, CachedValue, InMemoryCache, SemanticCache};
use semcast::model::MockModel;

/// An in-memory cache that also remembers every entry it was given.
#[derive(Debug, Default)]
struct Recording {
    inner: InMemoryCache,
    puts: Mutex<Vec<(CacheKey, CachedValue)>>,
}

impl SemanticCache for Recording {
    fn get(&self, key: &CacheKey) -> Option<CachedValue> {
        self.inner.get(key)
    }

    fn put(&self, key: CacheKey, value: CachedValue) {
        self.puts.lock().unwrap().push((key.clone(), value.clone()));
        self.inner.put(key, value);
    }
}

impl Recording {
    fn values(&self, field: &str) -> Vec<String> {
        let mut values: Vec<String> = self
            .puts
            .lock()
            .unwrap()
            .iter()
            .filter(|(key, _)| key.field == field)
            .map(|(_, value)| match value {
                CachedValue::Value(value) => value.clone(),
                CachedValue::Error(error) => format!("error: {error}"),
            })
            .collect();
        values.sort();
        values
    }
}

#[tokio::test]
async fn a_verdict_is_cached_with_its_probability() {
    let cache = Arc::new(Recording::default());
    let model = MockModel::answering_yes_to(["furious"])
        .with_yes_probability(|input| if input.contains("furious") { 0.8 } else { 0.3 });
    let ctx = SemcastContextBuilder::new(Arc::new(model))
        .with_cache(Arc::clone(&cache) as _)
        .build();
    ctx.sql(
        "CREATE TABLE tickets AS SELECT * FROM (VALUES
             (1, 'I am furious about the outage'),
             (2, 'thanks, all good now')
         ) AS t(id, body)",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    for sql in [
        "SELECT id FROM tickets WHERE body MEANS 'the customer is angry'",
        "SELECT id, CASE WHEN body MEANS 'asks for a refund' THEN 1 END FROM tickets",
    ] {
        semcast::sql(&ctx, sql)
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
    }
    assert_eq!(
        cache.values("means"),
        ["no", "no", "yes", "yes"],
        "verdicts are stored as before (the mock says yes to anything furious)"
    );
    assert_eq!(
        cache.values("means.yes_probability"),
        ["no 0.3", "no 0.3", "yes 0.8", "yes 0.8"],
        "each probability names the verdict it was measured with"
    );
}

#[tokio::test]
async fn a_fused_classify_stores_no_probability() {
    let cache = Arc::new(Recording::default());
    let ctx = SemcastContextBuilder::new(Arc::new(MockModel::answering_yes_to(["furious"])))
        .with_cache(Arc::clone(&cache) as _)
        .build();
    ctx.sql("CREATE TABLE tickets AS VALUES ('I am furious')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    semcast::sql(
        &ctx,
        "SELECT CASE WHEN column1 MEANS 'is angry' THEN 'a' \
                     WHEN column1 MEANS 'wants a refund' THEN 'r' END FROM tickets",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
    assert_eq!(
        cache.values("means").len(),
        2,
        "two verdicts from one fused call"
    );
    assert!(
        cache.values("means.yes_probability").is_empty(),
        "a fused answer has no single yes/no token"
    );
}
