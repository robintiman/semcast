//! The durable catalog: what a session defines — tables, views, semantic
//! types and indexes — is back after a restart, and indexes come back
//! without embedding anything.

use std::path::PathBuf;
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
            .with_index_root(self.dir.path().join("indexes"))
            .with_catalog(Catalog::open(self.store()).unwrap())
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
