//! The durable catalog: definitions that outlive the process.
//!
//! The verdict store keeps what the model answered; this keeps what the
//! user defined — schemas, external tables, views, semantic types and
//! semantic indexes — in a `catalog` table of the same SQLite file. Each
//! successful `CREATE` saves a row and each `DROP` deletes it; [`restore`]
//! replays the rows into a fresh context at startup.
//!
//! A row holds the statement's own SQL text, replayed through
//! [`crate::sql`], so a definition restores exactly as it was first planned.
//! The exception is a semantic index: replaying `CREATE SEMANTIC INDEX`
//! would embed the whole column again, so its row holds where the index
//! lives on disk and [`restore`] reattaches it without an embedding call.
//!
//! What is not saved: in-memory tables (`CREATE TABLE ... AS`, `INSERT`ed
//! rows), whose data lives in the process, and anything `TEMPORARY`.
//!
//! An entry that fails to restore (its file is gone, the session embeds
//! with another model) is logged and kept, so it comes back once the cause
//! is fixed; `DROP ... IF EXISTS` forgets it for good.

use std::path::Path;
use std::sync::{Arc, Mutex};

use datafusion::execution::context::SessionContext;
use datafusion::logical_expr::{DdlStatement, LogicalPlan};
use rusqlite::{Connection, params};

use crate::cache::sqlite::{open_store, store_err};
use crate::{Result, SemcastError};

/// What a catalog row defines. Names are unique per kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryKind {
    /// `CREATE SCHEMA`, keyed `catalog.schema`.
    Schema,
    /// An external table or a view, keyed `catalog.schema.table`. One kind
    /// for both, as in DataFusion: they share a namespace.
    Relation,
    /// `CREATE SEMANTIC TYPE`, keyed by the lowercased type name.
    SemanticType,
    /// A semantic index, keyed `table(column)`; its definition is JSON.
    SemanticIndex,
}

impl EntryKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Schema => "schema",
            Self::Relation => "relation",
            Self::SemanticType => "semantic_type",
            Self::SemanticIndex => "semantic_index",
        }
    }

    fn parse(kind: &str) -> Option<Self> {
        [
            Self::Schema,
            Self::Relation,
            Self::SemanticType,
            Self::SemanticIndex,
        ]
        .into_iter()
        .find(|known| known.as_str() == kind)
    }
}

/// One saved definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub kind: EntryKind,
    pub name: String,
    /// SQL text to replay, or a semantic index's JSON.
    pub definition: String,
}

/// One edit to the catalog. A statement's edits commit together.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Change {
    /// Save `entry`. With `replace`, a row of the same name is overwritten
    /// and moves to the end of the replay order; without, an existing row
    /// wins (`IF NOT EXISTS` on a name already defined changed nothing).
    Put { entry: Entry, replace: bool },
    /// Delete the row of this kind and name, if any.
    Forget { kind: EntryKind, name: String },
    /// Delete every row of this kind whose name starts with `prefix`.
    ForgetPrefix { kind: EntryKind, prefix: String },
}

/// The catalog half of a store file opened with
/// [`SqliteCache`](crate::cache::sqlite::SqliteCache): open both on the same
/// path to keep verdicts and definitions together.
#[derive(Debug, Clone)]
pub struct Catalog {
    connection: Arc<Mutex<Connection>>,
}

impl Catalog {
    /// Open (or create) the catalog in the store file at `path`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            connection: Arc::new(Mutex::new(open_store(path.as_ref())?)),
        })
    }

    /// Every saved definition, in the order it was saved.
    pub fn entries(&self) -> Result<Vec<Entry>> {
        let connection = self.connection.lock().expect("catalog lock poisoned");
        let mut statement = connection
            .prepare("SELECT kind, name, definition FROM catalog ORDER BY seq")
            .map_err(store_err)?;
        let rows = statement
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .map_err(store_err)?;
        let mut entries = Vec::new();
        for row in rows {
            let (kind, name, definition) = row.map_err(store_err)?;
            // A kind from a newer build: skip it rather than refuse the file.
            match EntryKind::parse(&kind) {
                Some(kind) => entries.push(Entry {
                    kind,
                    name,
                    definition,
                }),
                None => tracing::warn!("catalog: skipping {name}, of unknown kind {kind}"),
            }
        }
        Ok(entries)
    }

    /// Apply `changes` in one transaction, off the async threads: a busy
    /// file can make SQLite wait.
    pub(crate) async fn apply(&self, changes: Vec<Change>) -> Result<()> {
        if changes.is_empty() {
            return Ok(());
        }
        let connection = Arc::clone(&self.connection);
        tokio::task::spawn_blocking(move || {
            let mut connection = connection.lock().expect("catalog lock poisoned");
            apply_changes(&mut connection, &changes)
        })
        .await
        .map_err(|error| SemcastError::Store(error.to_string()))?
    }
}

fn apply_changes(connection: &mut Connection, changes: &[Change]) -> Result<()> {
    let transaction = connection
        .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
        .map_err(store_err)?;
    for change in changes {
        match change {
            Change::Put { entry, replace } => {
                if *replace {
                    transaction
                        .execute(
                            "DELETE FROM catalog WHERE kind = ?1 AND name = ?2",
                            params![entry.kind.as_str(), entry.name],
                        )
                        .map_err(store_err)?;
                }
                transaction
                    .execute(
                        "INSERT OR IGNORE INTO catalog (kind, name, definition) \
                         VALUES (?1, ?2, ?3)",
                        params![entry.kind.as_str(), entry.name, entry.definition],
                    )
                    .map_err(store_err)?;
            }
            Change::Forget { kind, name } => {
                transaction
                    .execute(
                        "DELETE FROM catalog WHERE kind = ?1 AND name = ?2",
                        params![kind.as_str(), name],
                    )
                    .map_err(store_err)?;
            }
            Change::ForgetPrefix { kind, prefix } => {
                // substr, not LIKE: a name may hold `%` or `_`.
                transaction
                    .execute(
                        "DELETE FROM catalog \
                         WHERE kind = ?1 AND substr(name, 1, length(?2)) = ?2",
                        params![kind.as_str(), prefix],
                    )
                    .map_err(store_err)?;
            }
        }
    }
    transaction.commit().map_err(store_err)
}

/// The catalog name of a semantic index on `table.column`.
pub(crate) fn index_name(table: &str, column: &str) -> String {
    format!("{table}({column})")
}

/// The edits a DDL plan makes to the catalog once it has run, or none for a
/// plan that defines nothing durable. `sql` is the statement's text.
pub(crate) fn changes_for(ctx: &SessionContext, plan: &LogicalPlan, sql: &str) -> Vec<Change> {
    let LogicalPlan::Ddl(ddl) = plan else {
        return Vec::new();
    };
    let state = ctx.state();
    let options = &state.config().options().catalog;
    let relation = |name: &datafusion::common::TableReference| {
        name.clone()
            .resolve(&options.default_catalog, &options.default_schema)
            .to_string()
    };
    let schema = |name: &str| {
        if name.contains('.') {
            name.to_owned()
        } else {
            format!("{}.{name}", options.default_catalog)
        }
    };
    let put = |kind, name, replace| Change::Put {
        entry: Entry {
            kind,
            name,
            definition: sql.to_owned(),
        },
        replace,
    };
    match ddl {
        DdlStatement::CreateExternalTable(create) if !create.temporary => {
            vec![put(
                EntryKind::Relation,
                relation(&create.name),
                !create.if_not_exists,
            )]
        }
        DdlStatement::CreateView(create) if !create.temporary => {
            vec![put(EntryKind::Relation, relation(&create.name), true)]
        }
        // Not saved, but a `CREATE OR REPLACE` of a saved name replaced it:
        // replaying the old definition would bring back what is gone.
        DdlStatement::CreateExternalTable(create) => {
            replaced(create.or_replace, relation(&create.name))
        }
        DdlStatement::CreateView(create) => replaced(create.or_replace, relation(&create.name)),
        DdlStatement::CreateMemoryTable(create) => {
            replaced(create.or_replace, relation(&create.name))
        }
        DdlStatement::CreateCatalogSchema(create) => vec![put(
            EntryKind::Schema,
            schema(&create.schema_name),
            !create.if_not_exists,
        )],
        DdlStatement::DropTable(drop) => dropped_relation(&drop.name, relation(&drop.name)),
        DdlStatement::DropView(drop) => dropped_relation(&drop.name, relation(&drop.name)),
        // No `DropCatalogSchema`: DataFusion cannot deregister a schema.
        _ => Vec::new(),
    }
}

fn replaced(or_replace: bool, name: String) -> Vec<Change> {
    if or_replace {
        vec![Change::Forget {
            kind: EntryKind::Relation,
            name,
        }]
    } else {
        Vec::new()
    }
}

/// A dropped relation takes its semantic indexes with it. The index map is
/// keyed by bare table name, so the indexes are too.
fn dropped_relation(table: &datafusion::common::TableReference, name: String) -> Vec<Change> {
    vec![
        Change::Forget {
            kind: EntryKind::Relation,
            name,
        },
        Change::ForgetPrefix {
            kind: EntryKind::SemanticIndex,
            prefix: format!("{}(", table.table()),
        },
    ]
}

/// What [`restore`] did.
#[derive(Debug, Default)]
pub struct Restored {
    /// Definitions now live in the context.
    pub restored: usize,
    /// Definitions that would not restore, with why. They stay saved.
    pub failed: Vec<(Entry, String)>,
}

/// Replay the catalog of `ctx`'s runtime into `ctx`. Call once, on a fresh
/// context, before serving queries.
///
/// Entries replay in the order they were saved. One that fails is retried
/// after the rest, for as long as a pass restores anything: a view saved
/// before a `CREATE OR REPLACE` of the table it reads still comes back.
pub async fn restore(ctx: &SessionContext) -> Result<Restored> {
    let runtime = crate::index::runtime(ctx)?;
    let Some(catalog) = runtime.catalog() else {
        return Ok(Restored::default());
    };
    let catalog = catalog.clone();
    let mut pending = tokio::task::spawn_blocking(move || catalog.entries())
        .await
        .map_err(|error| SemcastError::Store(error.to_string()))??;
    let mut restored = 0;
    loop {
        let mut failed = Vec::new();
        let mut progressed = false;
        for entry in pending {
            match restore_entry(ctx, &entry).await {
                Ok(()) => {
                    restored += 1;
                    progressed = true;
                }
                Err(error) => failed.push((entry, error.to_string())),
            }
        }
        if failed.is_empty() || !progressed {
            for (entry, error) in &failed {
                tracing::warn!(
                    "catalog: could not restore {} {}: {error}",
                    entry.kind.as_str(),
                    entry.name
                );
            }
            return Ok(Restored { restored, failed });
        }
        pending = failed.into_iter().map(|(entry, _)| entry).collect();
    }
}

async fn restore_entry(ctx: &SessionContext, entry: &Entry) -> Result<()> {
    match entry.kind {
        EntryKind::SemanticIndex => crate::index::reattach_semantic_index(ctx, &entry.definition)
            .await
            .map(|_| ()),
        _ => crate::run_sql(ctx, &entry.definition, false)
            .await
            .map(|_| ()),
    }
}
