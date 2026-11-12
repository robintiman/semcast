//! Statistics order the stages: what the store has seen of each condition
//! decides which filter runs first, so the one that drops the most for the
//! least goes early. Results never change, only calls.

use std::sync::Arc;

use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::execution::context::SessionContext;
use semcast::SemcastContextBuilder;
use semcast::cache::{AskStats, InMemoryCache, SemanticCache};
use semcast::model::{MockModel, ModelId};
use semcast::physical::verify::stats_key;

/// `name(id, subject, body)`: `subject` mentions "refund" in most rows,
/// `body` in one in ten. The mock answers yes to "refund" in either.
async fn table(ctx: &SessionContext, name: &str, rows: usize) {
    let values: Vec<String> = (0..rows)
        .map(|i| {
            let subject = if i % 10 == 9 { "hello" } else { "refund" };
            let body = if i % 10 == 0 { "refund" } else { "nothing" };
            format!("({i}, '{name} {subject} {i}', '{name} {body} {i}')")
        })
        .collect();
    ctx.sql(&format!(
        "CREATE TABLE {name} AS SELECT * FROM (VALUES {}) AS t(id, subject, body)",
        values.join(", ")
    ))
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();
}

async fn run(ctx: &SessionContext, sql: &str) -> String {
    let batches = semcast::sql(ctx, sql)
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    pretty_format_batches(&batches).unwrap().to_string()
}

const QUERY: &str = "SELECT count(*) AS n FROM tickets \
     WHERE subject MEANS 'asks for money back' AND body MEANS 'mentions a refund'";

#[tokio::test]
async fn statistics_put_the_rarer_filter_first() {
    // History from the table's first 50 rows; then 100 new rows arrive. In
    // written order the new rows cost every subject (100), then the 90 that
    // matched: 190 calls. Seen from history, `body` is the rarer: asked
    // first, it leaves 10 rows for `subject`, 110 calls. Old rows are cached
    // either way.
    let model = Arc::new(MockModel::answering_yes_to(["refund"]));
    let ctx = SemcastContextBuilder::new(Arc::clone(&model) as _).build();
    table(&ctx, "tickets", 50).await;
    run(
        &ctx,
        "SELECT count(*) FROM tickets WHERE subject MEANS 'asks for money back'",
    )
    .await;
    run(
        &ctx,
        "SELECT count(*) FROM tickets WHERE body MEANS 'mentions a refund'",
    )
    .await;
    table(&ctx, "fresh", 100).await;
    run(
        &ctx,
        "INSERT INTO tickets SELECT id + 1000, subject, body FROM fresh",
    )
    .await;
    let seeded = model.completion_calls();

    let out = run(&ctx, QUERY).await;
    assert!(out.contains("| 15 "), "{out}");
    assert_eq!(model.completion_calls() - seeded, 110);
}

#[tokio::test]
async fn without_statistics_the_written_order_stands() {
    let model = Arc::new(MockModel::answering_yes_to(["refund"]));
    let ctx = SemcastContextBuilder::new(Arc::clone(&model) as _).build();
    table(&ctx, "tickets", 100).await;
    let out = run(&ctx, QUERY).await;
    assert!(out.contains("| 10 "), "{out}");
    assert_eq!(model.completion_calls(), 190);
}

#[tokio::test]
async fn stacked_conditions_on_one_column_go_rarest_first() {
    // WITH RECALL keeps same-column conditions stacked rather than fused;
    // the rarer one goes innermost, so it runs first.
    let cache = Arc::new(InMemoryCache::default());
    let model = ModelId("mock".to_owned());
    let seed = |condition: &str, matched: u64| {
        cache.record_stats(
            stats_key(condition, &model, "tickets.body"),
            AskStats {
                asked: 100,
                matched,
                input_chars: 4000,
            },
        );
    };
    seed("is common", 90);
    seed("is rare", 5);
    let ctx = SemcastContextBuilder::new(Arc::new(MockModel::answering_yes_to(["x"])))
        .with_cache(Arc::clone(&cache) as _)
        .build();
    table(&ctx, "tickets", 10).await;
    let plan = run(
        &ctx,
        "EXPLAIN SELECT id FROM tickets \
         WHERE body MEANS 'is common' AND body MEANS 'is rare' WITH RECALL 0.9",
    )
    .await;
    let common = plan.find("SemFilter: MEANS('is common')").expect("planned");
    let rare = plan.find("SemFilter: MEANS('is rare')").expect("planned");
    // EXPLAIN prints the outermost stage first: the rare one runs first, so
    // it sits below.
    assert!(common < rare, "{plan}");
}

#[test]
fn statistics_survive_reopening_the_store() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("store.db");
    {
        let store = semcast::cache::sqlite::SqliteCache::open(&path).unwrap();
        store.record_stats(
            7,
            AskStats {
                asked: 3,
                matched: 1,
                input_chars: 30,
            },
        );
        store.record_stats(
            7,
            AskStats {
                asked: 1,
                matched: 1,
                input_chars: 10,
            },
        );
        assert_eq!(
            store.stats(7),
            Some(AskStats {
                asked: 4,
                matched: 2,
                input_chars: 40
            }),
            "pending answers count before they are committed"
        );
    }
    let store = semcast::cache::sqlite::SqliteCache::open(&path).unwrap();
    assert_eq!(
        store.stats(7),
        Some(AskStats {
            asked: 4,
            matched: 2,
            input_chars: 40
        })
    );
    assert_eq!(store.stats(8), None);
}

#[tokio::test]
async fn a_condition_without_history_does_not_jump_a_known_rare_one() {
    // `body` is known to match 5% of long texts; `subject` has no history.
    // Assumed to read as much as `body` and to match half, it stays behind.
    let cache = Arc::new(InMemoryCache::default());
    cache.record_stats(
        stats_key("is rare", &ModelId("mock".to_owned()), "tickets.body"),
        AskStats {
            asked: 100,
            matched: 5,
            input_chars: 400_000,
        },
    );
    let ctx = SemcastContextBuilder::new(Arc::new(MockModel::answering_yes_to(["x"])))
        .with_cache(Arc::clone(&cache) as _)
        .build();
    table(&ctx, "tickets", 10).await;
    let plan = run(
        &ctx,
        "EXPLAIN SELECT id FROM tickets \
         WHERE subject MEANS 'is half' AND body MEANS 'is rare'",
    )
    .await;
    let half = plan.find("SemFilter: MEANS('is half')").expect("planned");
    let rare = plan.find("SemFilter: MEANS('is rare')").expect("planned");
    assert!(half < rare, "the rare filter runs first, below: {plan}");
}
