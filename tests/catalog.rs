//! The durable catalog: what a session defines — tables, views, semantic
//! types and indexes — is back after a restart, and indexes come back
//! without embedding anything.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::execution::context::SessionContext;
use semcast::catalog::{Catalog, EntryKind};
use semcast::index::registry::SemcastRuntime;
use semcast::model::{Completion, CompletionRequest, Embedding, MockModel, ModelId, ModelProvider};
use semcast::{Result, SemcastContextBuilder};

/// One process's worth of state on disk: the store file, the index root and
/// the data files tables read.
struct Disk {
    dir: tempfile::TempDir,
}

impl Disk {
    fn new() -> Self {
        let disk = Self {
            dir: tempfile::tempdir().unwrap(),
        };
        std::fs::write(
            disk.csv("tickets"),
            "id,body\n1,I am furious about the outage\n2,thanks all good\n3,furious again\n",
        )
        .unwrap();
        disk
    }

    fn store(&self) -> PathBuf {
        self.dir.path().join("store.db")
    }

    fn csv(&self, name: &str) -> PathBuf {
        self.dir.path().join(format!("{name}.csv"))
    }

    /// A fresh process, as far as semcast can tell: only the disk carries
    /// over.
    async fn session(&self, model: Arc<dyn ModelProvider>) -> SessionContext {
        let ctx = SemcastContextBuilder::new(model)
            .with_catalog(Catalog::open(self.store()).unwrap())
            .with_chunk_store(Arc::new(
                semcast::index::stored::SqliteChunkStore::open(self.store()).unwrap(),
            ))
            .build();
        let restored = semcast::restore_catalog(&ctx).await.unwrap();
        assert!(restored.failed.is_empty(), "{:?}", restored.failed);
        ctx
    }

    fn entries(&self) -> Vec<(EntryKind, String)> {
        Catalog::open(self.store())
            .unwrap()
            .entries()
            .unwrap()
            .into_iter()
            .map(|entry| (entry.kind, entry.name))
            .collect()
    }

    fn create_tickets(&self) -> String {
        format!(
            "CREATE EXTERNAL TABLE tickets STORED AS CSV LOCATION '{}' \
             OPTIONS ('format.has_header' 'true')",
            self.csv("tickets").display()
        )
    }
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

fn runtime(ctx: &SessionContext) -> Arc<SemcastRuntime> {
    ctx.state()
        .config()
        .get_extension::<SemcastRuntime>()
        .unwrap()
}

const ANGRY: &str = "SELECT id FROM tickets WHERE body MEANS 'the customer is angry' ORDER BY id";

#[tokio::test]
async fn definitions_survive_a_restart_and_indexes_are_not_rebuilt() {
    let disk = Disk::new();
    let first = disk
        .session(Arc::new(MockModel::answering_yes_to(["furious"])))
        .await;
    run(&first, &disk.create_tickets()).await;
    run(
        &first,
        "CREATE VIEW calm AS SELECT id FROM tickets WHERE body NOT LIKE '%furious%'",
    )
    .await;
    run(
        &first,
        "CREATE SEMANTIC TYPE Ticket AS (mood ONEOF(angry, calm) 'how the customer feels')",
    )
    .await;
    run(&first, "CREATE SEMANTIC INDEX ON tickets(body)").await;
    let angry = run(&first, ANGRY).await;
    let calm = run(&first, "SELECT id FROM calm ORDER BY id").await;
    drop(first);

    let model = Arc::new(MockModel::answering_yes_to(["furious"]));
    let second = disk
        .session(Arc::clone(&model) as Arc<dyn ModelProvider>)
        .await;
    assert_eq!(model.embed_calls(), 0, "reattaching embeds nothing");
    assert!(
        runtime(&second).index_for("tickets", "body").is_some(),
        "the index is reattached to its column"
    );
    assert!(runtime(&second).type_registry().get("ticket").is_some());
    assert_eq!(run(&second, ANGRY).await, angry);
    assert_eq!(run(&second, "SELECT id FROM calm ORDER BY id").await, calm);
    // The query embedded only itself: the documents came from disk.
    assert_eq!(model.embed_calls(), 1, "one embed call, for the query");
}

#[tokio::test]
async fn drop_forgets_a_table_and_its_indexes() {
    let disk = Disk::new();
    let first = disk.session(Arc::new(MockModel::default())).await;
    run(&first, &disk.create_tickets()).await;
    run(&first, "CREATE SEMANTIC INDEX ON tickets(body)").await;
    assert_eq!(disk.entries().len(), 2);
    run(&first, "DROP TABLE tickets").await;
    assert!(runtime(&first).index_for("tickets", "body").is_none());
    assert_eq!(disk.entries(), vec![]);
    drop(first);

    let second = disk.session(Arc::new(MockModel::default())).await;
    assert!(!second.table_exist("tickets").unwrap());
}

#[tokio::test]
async fn an_entry_that_will_not_restore_is_kept_until_dropped() {
    let disk = Disk::new();
    let first = disk.session(Arc::new(MockModel::default())).await;
    run(&first, &disk.create_tickets()).await;
    drop(first);
    std::fs::remove_file(disk.csv("tickets")).unwrap();

    let ctx = SemcastContextBuilder::new(Arc::new(MockModel::default()))
        .with_catalog(Catalog::open(disk.store()).unwrap())
        .build();
    let restored = semcast::restore_catalog(&ctx).await.unwrap();
    assert_eq!(restored.restored, 0);
    assert_eq!(restored.failed.len(), 1, "{:?}", restored.failed);
    assert_eq!(disk.entries().len(), 1, "a failed entry stays saved");

    run(&ctx, "DROP TABLE IF EXISTS tickets").await;
    assert_eq!(disk.entries(), vec![], "DROP IF EXISTS forgets it");
}

#[tokio::test]
async fn a_view_restores_even_when_its_table_was_replaced_after_it() {
    let disk = Disk::new();
    let first = disk.session(Arc::new(MockModel::default())).await;
    run(&first, &disk.create_tickets()).await;
    run(&first, "CREATE VIEW ids AS SELECT id FROM tickets").await;
    let replace = disk
        .create_tickets()
        .replacen("CREATE", "CREATE OR REPLACE", 1);
    run(&first, &replace).await;
    // The replaced table now replays after the view that reads it.
    assert_eq!(
        disk.entries(),
        vec![
            (EntryKind::Relation, "datafusion.public.ids".to_owned()),
            (EntryKind::Relation, "datafusion.public.tickets".to_owned()),
        ]
    );
    drop(first);

    let second = disk.session(Arc::new(MockModel::default())).await;
    assert!(
        run(&second, "SELECT count(*) AS n FROM ids")
            .await
            .contains("| 3 |")
    );
}

#[tokio::test]
async fn in_memory_tables_are_not_saved_and_replacing_with_one_forgets() {
    let disk = Disk::new();
    let first = disk.session(Arc::new(MockModel::default())).await;
    run(&first, "CREATE TABLE scratch AS VALUES (1)").await;
    assert_eq!(disk.entries(), vec![], "an in-memory table is not saved");

    run(&first, &disk.create_tickets()).await;
    run(&first, "CREATE OR REPLACE TABLE tickets AS VALUES (1)").await;
    assert_eq!(
        disk.entries(),
        vec![],
        "the saved definition was replaced, so replaying it would be wrong"
    );
}

#[tokio::test]
async fn if_not_exists_on_a_defined_name_keeps_the_saved_definition() {
    let disk = Disk::new();
    let first = disk.session(Arc::new(MockModel::default())).await;
    run(&first, &disk.create_tickets()).await;
    std::fs::write(disk.csv("other"), "id\n9\n").unwrap();
    run(
        &first,
        &format!(
            "CREATE EXTERNAL TABLE IF NOT EXISTS tickets STORED AS CSV LOCATION '{}' \
             OPTIONS ('format.has_header' 'true')",
            disk.csv("other").display()
        ),
    )
    .await;
    drop(first);

    let second = disk.session(Arc::new(MockModel::default())).await;
    assert!(
        run(&second, "SELECT count(*) AS n FROM tickets")
            .await
            .contains("| 3 |"),
        "the first definition is the one that took effect, and the one saved"
    );
}

#[tokio::test]
async fn schemas_restore_before_the_tables_in_them() {
    let disk = Disk::new();
    let first = disk.session(Arc::new(MockModel::default())).await;
    run(&first, "CREATE SCHEMA support").await;
    run(
        &first,
        &disk
            .create_tickets()
            .replacen("TABLE tickets", "TABLE support.tickets", 1),
    )
    .await;
    drop(first);

    let second = disk.session(Arc::new(MockModel::default())).await;
    assert!(
        run(&second, "SELECT count(*) AS n FROM support.tickets")
            .await
            .contains("| 3 |")
    );
}

/// [`MockModel`] under another name: a different embed model, as far as an
/// index's provenance check can tell.
#[derive(Debug)]
struct Renamed(MockModel);

#[async_trait]
impl ModelProvider for Renamed {
    fn id(&self) -> ModelId {
        ModelId("another-embedder".to_owned())
    }

    async fn complete(&self, requests: Vec<CompletionRequest>) -> Vec<Result<Completion>> {
        self.0.complete(requests).await
    }

    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Embedding>> {
        self.0.embed(texts).await
    }
}

#[tokio::test]
async fn an_index_from_another_embedder_is_not_reattached() {
    let disk = Disk::new();
    let first = disk.session(Arc::new(MockModel::default())).await;
    run(&first, &disk.create_tickets()).await;
    run(&first, "CREATE SEMANTIC INDEX ON tickets(body)").await;
    drop(first);

    let ctx = SemcastContextBuilder::new(Arc::new(Renamed(MockModel::default())))
        .with_catalog(Catalog::open(disk.store()).unwrap())
        .build();
    let restored = semcast::restore_catalog(&ctx).await.unwrap();
    assert_eq!(restored.restored, 1, "the table restores");
    let [(entry, error)] = restored.failed.as_slice() else {
        panic!("expected one failure: {:?}", restored.failed);
    };
    assert_eq!(entry.kind, EntryKind::SemanticIndex);
    assert!(error.contains("another-embedder"), "{error}");
    assert!(runtime(&ctx).index_for("tickets", "body").is_none());
}

#[tokio::test]
async fn a_drop_that_drops_nothing_forgets_nothing() {
    let disk = Disk::new();
    let first = disk.session(Arc::new(MockModel::default())).await;
    run(&first, &disk.create_tickets()).await;
    run(&first, "CREATE VIEW ids AS SELECT id FROM tickets").await;
    run(&first, "CREATE SEMANTIC INDEX ON tickets(body)").await;
    // Each succeeds without dropping anything: the type doesn't match.
    run(&first, "DROP TABLE IF EXISTS ids").await;
    run(&first, "DROP VIEW IF EXISTS tickets").await;
    assert!(first.table_exist("ids").unwrap());
    assert!(runtime(&first).index_for("tickets", "body").is_some());
    assert_eq!(disk.entries().len(), 3);
    drop(first);

    let second = disk.session(Arc::new(MockModel::default())).await;
    assert!(second.table_exist("ids").unwrap());
    assert!(runtime(&second).index_for("tickets", "body").is_some());
}

#[tokio::test]
async fn dropping_a_same_named_table_elsewhere_keeps_the_index() {
    let disk = Disk::new();
    let first = disk.session(Arc::new(MockModel::default())).await;
    run(&first, &disk.create_tickets()).await;
    run(&first, "CREATE SEMANTIC INDEX ON tickets(body)").await;
    run(&first, "CREATE SCHEMA staging").await;
    run(&first, "DROP TABLE IF EXISTS staging.tickets").await;
    let staged = disk
        .create_tickets()
        .replacen("TABLE tickets", "TABLE staging.tickets", 1);
    run(&first, &staged).await;
    run(&first, "DROP TABLE staging.tickets").await;
    assert!(
        runtime(&first).index_for("tickets", "body").is_some(),
        "public.tickets still has its index"
    );
    drop(first);

    let second = disk.session(Arc::new(MockModel::default())).await;
    assert!(runtime(&second).index_for("tickets", "body").is_some());
}

#[tokio::test]
async fn if_not_exists_over_an_in_memory_table_saves_nothing() {
    let disk = Disk::new();
    let first = disk.session(Arc::new(MockModel::default())).await;
    run(&first, "CREATE TABLE tickets AS VALUES (42)").await;
    let if_not_exists =
        disk.create_tickets()
            .replacen("TABLE tickets", "TABLE IF NOT EXISTS tickets", 1);
    run(&first, &if_not_exists).await;
    assert_eq!(
        disk.entries(),
        vec![],
        "the CSV definition never took effect"
    );
}

#[tokio::test]
async fn redefining_a_name_that_failed_to_restore_saves_the_new_definition() {
    let disk = Disk::new();
    let first = disk.session(Arc::new(MockModel::default())).await;
    run(&first, &disk.create_tickets()).await;
    drop(first);
    let moved = disk.csv("moved");
    std::fs::rename(disk.csv("tickets"), &moved).unwrap();

    let ctx = SemcastContextBuilder::new(Arc::new(MockModel::default()))
        .with_catalog(Catalog::open(disk.store()).unwrap())
        .build();
    let restored = semcast::restore_catalog(&ctx).await.unwrap();
    assert_eq!(restored.failed.len(), 1);
    run(
        &ctx,
        &format!(
            "CREATE EXTERNAL TABLE IF NOT EXISTS tickets STORED AS CSV LOCATION '{}' \
             OPTIONS ('format.has_header' 'true')",
            moved.display()
        ),
    )
    .await;
    drop(ctx);

    let third = disk.session(Arc::new(MockModel::default())).await;
    assert!(
        run(&third, "SELECT count(*) AS n FROM tickets")
            .await
            .contains("| 3 |")
    );
}

#[tokio::test]
async fn an_index_on_an_in_memory_table_is_not_saved() {
    let disk = Disk::new();
    let first = disk.session(Arc::new(MockModel::default())).await;
    run(&first, "CREATE TABLE notes AS VALUES ('furious'), ('calm')").await;
    run(&first, "CREATE SEMANTIC INDEX ON notes(column1)").await;
    assert!(runtime(&first).index_for("notes", "column1").is_some());
    assert_eq!(disk.entries(), vec![]);
}

#[tokio::test]
async fn replacing_a_saved_table_with_an_in_memory_one_forgets_its_indexes() {
    let disk = Disk::new();
    let first = disk.session(Arc::new(MockModel::default())).await;
    run(&first, &disk.create_tickets()).await;
    run(&first, "CREATE SEMANTIC INDEX ON tickets(body)").await;
    run(&first, "CREATE OR REPLACE TABLE tickets AS VALUES (1)").await;
    assert_eq!(disk.entries(), vec![]);
}

#[tokio::test]
async fn index_names_fold_case_like_datafusion() {
    let disk = Disk::new();
    let first = disk.session(Arc::new(MockModel::default())).await;
    run(&first, &disk.create_tickets()).await;
    run(&first, "CREATE SEMANTIC INDEX ON Tickets(Body)").await;
    assert!(runtime(&first).index_for("tickets", "body").is_some());
    run(&first, "DROP TABLE tickets").await;
    assert_eq!(disk.entries(), vec![], "the drop forgets the index");
}

#[tokio::test]
async fn a_dropped_semantic_type_stays_dropped() {
    let disk = Disk::new();
    let first = disk.session(Arc::new(MockModel::default())).await;
    run(
        &first,
        "CREATE SEMANTIC TYPE Ticket AS (mood ONEOF(angry, calm) 'how the customer feels')",
    )
    .await;
    run(&first, "DROP SEMANTIC TYPE ticket").await;
    assert!(runtime(&first).type_registry().get("ticket").is_none());
    assert!(
        semcast::sql(&first, "DROP SEMANTIC TYPE Ticket")
            .await
            .is_err()
    );
    run(&first, "DROP SEMANTIC TYPE IF EXISTS Ticket").await;
    assert_eq!(disk.entries(), vec![]);
    drop(first);

    let second = disk.session(Arc::new(MockModel::default())).await;
    assert!(runtime(&second).type_registry().get("ticket").is_none());
}

#[tokio::test]
async fn a_relative_location_is_saved_absolute() {
    let disk = Disk::new();
    let cwd = std::env::current_dir().unwrap();
    let relative = pathdiff(&disk.csv("tickets"), &cwd);
    let first = disk.session(Arc::new(MockModel::default())).await;
    run(
        &first,
        &format!(
            "CREATE EXTERNAL TABLE tickets STORED AS CSV LOCATION '{relative}' \
             OPTIONS ('format.has_header' 'true')"
        ),
    )
    .await;
    let [entry] = Catalog::open(disk.store())
        .unwrap()
        .entries()
        .unwrap()
        .try_into()
        .unwrap();
    let saved = entry.definition.split('\'').nth(1).unwrap();
    assert!(Path::new(saved).is_absolute(), "{}", entry.definition);
    assert_eq!(
        std::fs::canonicalize(saved).unwrap(),
        std::fs::canonicalize(disk.csv("tickets")).unwrap()
    );
}

/// `path` relative to `base`, through `..` as needed.
fn pathdiff(path: &Path, base: &Path) -> String {
    let path = std::fs::canonicalize(path).unwrap();
    let base = std::fs::canonicalize(base).unwrap();
    let common = path
        .components()
        .zip(base.components())
        .take_while(|(a, b)| a == b)
        .count();
    let ups = base.components().count() - common;
    let mut relative = std::path::PathBuf::new();
    for _ in 0..ups {
        relative.push("..");
    }
    for part in path.components().skip(common) {
        relative.push(part);
    }
    relative.to_str().unwrap().to_owned()
}

#[tokio::test]
async fn drop_if_exists_in_a_missing_schema_succeeds() {
    // With and without a store: DataFusion accepts it, so semcast must.
    let plain = semcast::semcast_context(Arc::new(MockModel::default()));
    run(&plain, "DROP TABLE IF EXISTS nosuch.t").await;
    run(&plain, "DROP VIEW IF EXISTS nosuch.v").await;
    let disk = Disk::new();
    let stored = disk.session(Arc::new(MockModel::default())).await;
    run(&stored, "DROP TABLE IF EXISTS nosuch.t").await;
}

#[tokio::test]
async fn a_database_restores_before_the_schemas_and_tables_in_it() {
    let disk = Disk::new();
    let first = disk.session(Arc::new(MockModel::default())).await;
    run(&first, "CREATE DATABASE crm").await;
    run(&first, "CREATE SCHEMA crm.sales").await;
    let in_crm = disk
        .create_tickets()
        .replacen("TABLE tickets", "TABLE crm.sales.tickets", 1);
    run(&first, &in_crm).await;
    drop(first);

    let second = disk.session(Arc::new(MockModel::default())).await;
    assert!(
        run(&second, "SELECT count(*) AS n FROM crm.sales.tickets")
            .await
            .contains("| 3 |")
    );
}

#[tokio::test]
async fn a_view_over_a_relative_path_is_refused_when_saving() {
    let disk = Disk::new();
    let cwd = std::env::current_dir().unwrap();
    let relative = pathdiff(&disk.csv("tickets"), &cwd);
    let view = format!("CREATE VIEW raw AS SELECT * FROM '{relative}'");

    let first = disk.session(Arc::new(MockModel::default())).await;
    let error = semcast::sql(&first, &view).await.unwrap_err().to_string();
    assert!(error.contains("relative path"), "{error}");
    assert!(!first.table_exist("raw").unwrap(), "refused before it ran");

    let absolute = std::fs::canonicalize(disk.csv("tickets")).unwrap();
    run(
        &first,
        &format!("CREATE VIEW raw AS SELECT * FROM '{}'", absolute.display()),
    )
    .await;
    run(&first, "CREATE VIEW ids AS SELECT id FROM raw").await;
    assert_eq!(
        disk.entries().len(),
        2,
        "absolute paths and tables are fine"
    );

    // Without a store there is nothing to replay, so nothing to refuse.
    let plain = semcast::semcast_context(Arc::new(MockModel::default()));
    run(&plain, &view).await;
}

#[tokio::test]
async fn an_index_is_not_saved_for_a_same_named_table_in_another_schema() {
    let disk = Disk::new();
    let first = disk.session(Arc::new(MockModel::default())).await;
    run(&first, "CREATE SCHEMA s").await;
    let elsewhere = disk
        .create_tickets()
        .replacen("TABLE tickets", "TABLE s.tickets", 1);
    run(&first, &elsewhere).await;
    run(
        &first,
        "CREATE TABLE tickets AS SELECT column1 AS body FROM (VALUES ('furious'))",
    )
    .await;
    run(&first, "CREATE SEMANTIC INDEX ON tickets(body)").await;
    assert!(
        !disk
            .entries()
            .iter()
            .any(|(kind, _)| *kind == EntryKind::SemanticIndex),
        "public.tickets is in memory: {:?}",
        disk.entries()
    );
}

#[tokio::test]
async fn replacing_a_table_with_an_in_memory_one_unregisters_its_indexes() {
    let disk = Disk::new();
    let first = disk.session(Arc::new(MockModel::default())).await;
    run(&first, &disk.create_tickets()).await;
    run(&first, "CREATE SEMANTIC INDEX ON tickets(body)").await;
    run(&first, "CREATE OR REPLACE TABLE tickets AS VALUES (1)").await;
    assert!(runtime(&first).index_for("tickets", "body").is_none());
}

#[tokio::test]
async fn a_slow_create_table_as_does_not_hold_up_other_ddl() {
    let disk = Disk::new();
    let latency = std::time::Duration::from_secs(2);
    let model = MockModel::answering_yes_to(["furious"]).with_latency(latency);
    let ctx = disk.session(Arc::new(model)).await;
    run(&ctx, &disk.create_tickets()).await;

    let slow_ctx = ctx.clone();
    let slow = tokio::spawn(async move {
        run(
            &slow_ctx,
            "CREATE TABLE angry AS SELECT id FROM tickets WHERE body MEANS 'angry'",
        )
        .await
    });
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    let started = std::time::Instant::now();
    run(&ctx, "CREATE VIEW ids AS SELECT id FROM tickets").await;
    assert!(
        started.elapsed() < latency / 2,
        "the view waited {:?} on the running query",
        started.elapsed()
    );
    slow.await.unwrap();
}

#[tokio::test]
async fn a_failed_replace_forgets_the_definition_it_dropped() {
    let disk = Disk::new();
    let first = disk.session(Arc::new(MockModel::default())).await;
    run(&first, &disk.create_tickets()).await;
    run(&first, "CREATE SEMANTIC INDEX ON tickets(body)").await;
    let broken = "CREATE OR REPLACE EXTERNAL TABLE tickets STORED AS CSV \
                  LOCATION '/nonexistent/semcast/x.csv'";
    assert!(semcast::sql(&first, broken).await.is_err());
    // DataFusion dropped the old table before the new one failed to build.
    assert!(!first.table_exist("tickets").unwrap());
    assert!(runtime(&first).index_for("tickets", "body").is_none());
    assert_eq!(
        disk.entries(),
        vec![],
        "the session lost it, so the disk must"
    );
}

#[tokio::test]
async fn views_over_table_functions_are_saved() {
    let disk = Disk::new();
    let first = disk.session(Arc::new(MockModel::default())).await;
    run(
        &first,
        "CREATE VIEW three AS SELECT * FROM generate_series(1, 3)",
    )
    .await;
    drop(first);

    let second = disk.session(Arc::new(MockModel::default())).await;
    assert!(
        run(&second, "SELECT count(*) AS n FROM three")
            .await
            .contains("| 3 |")
    );
}

#[tokio::test]
async fn a_view_over_an_in_memory_table_is_not_saved() {
    let disk = Disk::new();
    let first = disk.session(Arc::new(MockModel::default())).await;
    run(&first, "CREATE TABLE scratch AS VALUES (1)").await;
    run(&first, "CREATE VIEW direct AS SELECT * FROM scratch").await;
    run(&first, "CREATE VIEW nested AS SELECT * FROM direct").await;
    assert_eq!(disk.entries(), vec![], "neither view would restore");
}
