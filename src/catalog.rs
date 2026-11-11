//! The durable catalog: definitions that outlive the process.
//!
//! The verdict store keeps what the model answered; this keeps what the
//! user defined — databases, schemas, external tables, views, semantic
//! types and semantic indexes — in a `catalog` table of the same SQLite file. Each
//! `CREATE` that takes effect saves a row and each `DROP` that removes
//! something deletes it; [`restore`] replays the rows into a fresh context
//! at startup.
//!
//! What is saved follows what a statement *did*, not what it said: the
//! name is looked up before and after the statement runs. `IF NOT EXISTS`
//! on a defined name, or `DROP TABLE IF EXISTS` on a view, changes nothing
//! in the session, so it changes nothing on disk either.
//!
//! A row holds the statement's own SQL text, replayed through
//! [`crate::sql`], so a restored definition looks its names up again. That
//! differs from the session in one way: a DataFusion view keeps reading the
//! table it was created over, even after that name is replaced; restored,
//! it reads whatever the name means now — or fails, if the name became an
//! in-memory table or was dropped. A relative `LOCATION` is saved made absolute, so a server started from
//! another directory reads the same file, and a view that reads a file by
//! relative path is refused. The exception is a semantic
//! index: replaying `CREATE SEMANTIC INDEX` would embed the whole column
//! again, so its row holds where the index lives on disk and [`restore`]
//! reattaches it without an embedding call.
//!
//! What is not saved: in-memory tables (`CREATE TABLE ... AS`, `INSERT`ed
//! rows), whose data lives in the process, anything `TEMPORARY`, and
//! the views and semantic indexes over either. Defining one of these under a saved name
//! forgets the saved definition, which would otherwise replace it after a
//! restart.
//!
//! An entry that fails to restore (its file is gone, the session embeds
//! with another model) is logged and kept, so it comes back once the cause
//! is fixed — or is overwritten when the name is defined again.

use std::path::Path;
use std::sync::{Arc, Mutex};

use datafusion::common::TableReference;
use datafusion::dataframe::DataFrame;
use datafusion::execution::context::SessionContext;
use datafusion::logical_expr::{DdlStatement, LogicalPlan};
use rusqlite::{Connection, params};

use crate::cache::sqlite::{open_store, store_err};
use crate::{Result, SemcastError};

/// What a catalog row defines. Names are unique per kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryKind {
    /// `CREATE DATABASE`, keyed by its name.
    Catalog,
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
            Self::Catalog => "catalog",
            Self::Schema => "schema",
            Self::Relation => "relation",
            Self::SemanticType => "semantic_type",
            Self::SemanticIndex => "semantic_index",
        }
    }

    fn parse(kind: &str) -> Option<Self> {
        [
            Self::Catalog,
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
    /// Save `entry`, overwriting a row of the same name; it moves to the end
    /// of the replay order, after whatever it reads.
    Put(Entry),
    /// Delete the row of this kind and name, if any.
    Forget { kind: EntryKind, name: String },
    /// Delete the saved indexes on bare table name `table`, unless a saved
    /// relation of that bare name remains — the index map is keyed by bare
    /// name, so such a relation still owns them.
    ForgetOrphanIndexes { table: String },
}

/// The catalog half of a store file opened with
/// [`SqliteCache`](crate::cache::sqlite::SqliteCache): open both on the same
/// path to keep verdicts and definitions together.
#[derive(Debug, Clone)]
pub struct Catalog {
    connection: Arc<Mutex<Connection>>,
    /// Held across a statement and its save, so concurrent DDL reaches the
    /// disk in the order it took effect in memory.
    ddl: Arc<tokio::sync::Mutex<()>>,
}

impl Catalog {
    /// Open (or create) the catalog in the store file at `path`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            connection: Arc::new(Mutex::new(open_store(path.as_ref())?)),
            ddl: Arc::new(tokio::sync::Mutex::new(())),
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

    /// Serialize a definition change with its save. Hold the guard from
    /// before the change takes effect until [`Catalog::apply`] returns.
    pub(crate) async fn lock(&self) -> tokio::sync::MutexGuard<'_, ()> {
        self.ddl.lock().await
    }

    /// Whether the relation of resolved name `name` is saved.
    pub(crate) async fn has_relation(&self, name: &str) -> Result<bool> {
        let connection = Arc::clone(&self.connection);
        let name = name.to_owned();
        tokio::task::spawn_blocking(move || {
            let connection = connection.lock().expect("catalog lock poisoned");
            connection
                .query_row(
                    "SELECT EXISTS (SELECT 1 FROM catalog \
                     WHERE kind = 'relation' AND name = ?1)",
                    params![name],
                    |row| row.get(0),
                )
                .map_err(store_err)
        })
        .await
        .map_err(|error| SemcastError::Store(error.to_string()))?
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
            Change::Put(entry) => {
                transaction
                    .execute(
                        "DELETE FROM catalog WHERE kind = ?1 AND name = ?2",
                        params![entry.kind.as_str(), entry.name],
                    )
                    .map_err(store_err)?;
                transaction
                    .execute(
                        "INSERT INTO catalog (kind, name, definition) VALUES (?1, ?2, ?3)",
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
            Change::ForgetOrphanIndexes { table } => {
                // substr, not LIKE: a name may hold `%` or `_`.
                transaction
                    .execute(
                        "DELETE FROM catalog \
                         WHERE kind = 'semantic_index' \
                           AND substr(name, 1, length(?1)) = ?1 \
                           AND NOT EXISTS (SELECT 1 FROM catalog \
                               WHERE kind = 'relation' \
                                 AND substr(name, -length(?2)) = ?2)",
                        params![format!("{table}("), format!(".{table}")],
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

/// Save `changes`, or say what the session and the disk now disagree on.
pub(crate) async fn save(catalog: &Catalog, changes: Vec<Change>, failure: &str) -> Result<()> {
    catalog.apply(changes).await.map_err(|error| {
        SemcastError::Store(format!(
            "{failure}: the store could not save the change: {error}"
        ))
    })
}

/// What a DDL plan would do to a name, read before it runs.
enum Effect {
    /// Defines a relation that is saved.
    SaveRelation {
        name: TableReference,
        replaces: bool,
        location: Option<String>,
    },
    /// Defines a relation that is not saved (in memory, or temporary).
    UnsavedRelation {
        name: TableReference,
        replaces: bool,
    },
    SaveSchema {
        name: String,
    },
    SaveCatalog {
        name: String,
    },
    DropRelation {
        name: TableReference,
    },
    /// Nothing the catalog keeps.
    Other,
}

impl Effect {
    fn of(plan: &LogicalPlan) -> Self {
        let LogicalPlan::Ddl(ddl) = plan else {
            return Self::Other;
        };
        match ddl {
            DdlStatement::CreateExternalTable(create) if !create.temporary => Self::SaveRelation {
                name: create.name.clone(),
                replaces: create.or_replace,
                location: Some(create.location.clone()),
            },
            DdlStatement::CreateView(create) if !create.temporary => Self::SaveRelation {
                name: create.name.clone(),
                replaces: create.or_replace,
                location: None,
            },
            DdlStatement::CreateExternalTable(create) => Self::UnsavedRelation {
                name: create.name.clone(),
                replaces: create.or_replace,
            },
            DdlStatement::CreateView(create) => Self::UnsavedRelation {
                name: create.name.clone(),
                replaces: create.or_replace,
            },
            DdlStatement::CreateMemoryTable(create) => Self::UnsavedRelation {
                name: create.name.clone(),
                replaces: create.or_replace,
            },
            DdlStatement::CreateCatalogSchema(create) => Self::SaveSchema {
                name: create.schema_name.clone(),
            },
            DdlStatement::CreateCatalog(create) => Self::SaveCatalog {
                name: create.catalog_name.clone(),
            },
            DdlStatement::DropTable(drop) => Self::DropRelation {
                name: drop.name.clone(),
            },
            DdlStatement::DropView(drop) => Self::DropRelation {
                name: drop.name.clone(),
            },
            // No `DropCatalogSchema`: DataFusion cannot deregister a schema.
            _ => Self::Other,
        }
    }
}

/// Run a DDL `plan` and bring the session's indexes and catalog in line
/// with what it did. `sql` is the statement's text; with `record` unset
/// (replaying the catalog) nothing is saved.
pub(crate) async fn execute_ddl(
    ctx: &SessionContext,
    plan: LogicalPlan,
    sql: &str,
    record: bool,
) -> Result<DataFrame> {
    let runtime = ctx
        .state()
        .config()
        .get_extension::<crate::index::registry::SemcastRuntime>();
    let catalog = runtime
        .as_ref()
        .and_then(|runtime| runtime.catalog().cloned())
        .filter(|_| record);
    // `CREATE TABLE ... AS` runs its whole query, model calls included, and
    // is never saved: lock only for its save, not while it runs. Everything
    // else is quick, and holds the lock throughout so disk order matches
    // the order the statements took effect.
    let slow = matches!(&plan, LogicalPlan::Ddl(DdlStatement::CreateMemoryTable(_)));
    let mut guard = match &catalog {
        Some(catalog) if !slow => Some(catalog.lock().await),
        _ => None,
    };
    let mut effect = Effect::of(&plan);
    // A view over an in-memory table would restore without it, and fail on
    // every start: keep such a view for the session, like its table.
    if let Effect::SaveRelation {
        name,
        replaces,
        location: None,
    } = &effect
        && reads_memory_table(&plan)?
    {
        effect = Effect::UnsavedRelation {
            name: name.clone(),
            replaces: *replaces,
        };
    }
    // Before running: a definition that can't be saved faithfully refuses
    // the statement rather than define something lost on restart.
    let definition = match &effect {
        Effect::SaveRelation {
            location: Some(location),
            ..
        } if catalog.is_some() => absolute_location(sql, location)?,
        Effect::SaveRelation { location: None, .. } if catalog.is_some() => {
            refuse_relative_paths(ctx, &plan)?;
            sql.to_owned()
        }
        _ => sql.to_owned(),
    };
    let existed = exists(ctx, &effect);
    let result = ctx.execute_logical_plan(plan).await;
    // DDL has run by now: `execute_logical_plan` executes it eagerly.
    if guard.is_none()
        && let Some(catalog) = &catalog
    {
        guard = Some(catalog.lock().await);
    }
    let _guard = guard;
    let exists_now = exists(ctx, &effect);
    let df = match result {
        Ok(df) => df,
        Err(error) => {
            // DataFusion's `CREATE OR REPLACE` drops the old definition
            // before building the new one; a failed build leaves the name
            // gone. Forget it too, or a restart brings it back.
            if existed
                && !exists_now
                && let Some(name) = effect.relation()
            {
                // Report the statement's own error first; a store failure
                // on top of it is the second half of the message.
                if let Err(store) =
                    forget_relation(ctx, runtime.as_deref(), catalog.as_ref(), name).await
                {
                    return Err(SemcastError::Store(format!("{error}; then {store}")));
                }
            }
            return Err(error.into());
        }
    };
    let (changes, failure) = match effect {
        // Defined, unless the name was there and nothing replaced it.
        Effect::SaveRelation { name, replaces, .. } if !existed || replaces => {
            let name = resolve(ctx, &name);
            let failure =
                format!("{name} is defined for this session but will be gone after a restart");
            let entry = Entry {
                kind: EntryKind::Relation,
                name,
                definition,
            };
            (vec![Change::Put(entry)], failure)
        }
        Effect::UnsavedRelation { name, replaces } if !existed || replaces => {
            // Unlocked while it ran: another statement may have redefined
            // the name since, and its save stands.
            if slow && !is_memory_table(ctx, &name).await {
                return Ok(df);
            }
            let table = name.table().to_owned();
            // A replaced table's indexes described the old rows.
            if existed && let Some(runtime) = &runtime {
                runtime.unregister_indexes(&table).await;
            }
            let name = resolve(ctx, &name);
            let failure = format!(
                "{name} is defined for this session, but an earlier saved definition \
                 will replace it after a restart"
            );
            let forget = Change::Forget {
                kind: EntryKind::Relation,
                name,
            };
            (vec![forget, Change::ForgetOrphanIndexes { table }], failure)
        }
        Effect::SaveSchema { name } if !existed => {
            let name = qualified_schema(ctx, &name);
            let failure = format!(
                "schema {name} is defined for this session but will be gone after a restart"
            );
            let entry = Entry {
                kind: EntryKind::Schema,
                name,
                definition,
            };
            (vec![Change::Put(entry)], failure)
        }
        Effect::SaveCatalog { name } if !existed => {
            let failure = format!(
                "database {name} is defined for this session but will be gone after a restart"
            );
            let entry = Entry {
                kind: EntryKind::Catalog,
                name,
                definition,
            };
            (vec![Change::Put(entry)], failure)
        }
        // Forgotten only if the name is gone: `DROP TABLE IF EXISTS` on a
        // view succeeds without dropping it. A name that was never live —
        // a saved entry that failed to restore — is forgotten too.
        Effect::DropRelation { name } if !exists_now => {
            forget_relation(ctx, runtime.as_deref(), catalog.as_ref(), &name).await?;
            return Ok(df);
        }
        _ => return Ok(df),
    };
    if let Some(catalog) = &catalog {
        save(catalog, changes, &failure).await?;
    }
    Ok(df)
}

impl Effect {
    /// The relation this statement defines or drops, if any.
    fn relation(&self) -> Option<&TableReference> {
        match self {
            Self::SaveRelation { name, .. }
            | Self::UnsavedRelation { name, .. }
            | Self::DropRelation { name } => Some(name),
            _ => None,
        }
    }
}

/// `name` is gone from the session: drop its indexes from the session
/// unless a same-named table elsewhere still uses them, and from the
/// catalog unless a saved one does.
async fn forget_relation(
    ctx: &SessionContext,
    runtime: Option<&crate::index::registry::SemcastRuntime>,
    catalog: Option<&Catalog>,
    name: &TableReference,
) -> Result<()> {
    let table = name.table().to_owned();
    if let Some(runtime) = runtime
        && !bare_name_in_use(ctx, &table)
    {
        runtime.unregister_indexes(&table).await;
    }
    let Some(catalog) = catalog else {
        return Ok(());
    };
    let name = resolve(ctx, name);
    let failure = format!("{name} is gone from this session but will come back after a restart");
    let forget = Change::Forget {
        kind: EntryKind::Relation,
        name,
    };
    save(
        catalog,
        vec![forget, Change::ForgetOrphanIndexes { table }],
        &failure,
    )
    .await
}

/// Whether `name` is an in-memory table in `ctx` right now.
async fn is_memory_table(ctx: &SessionContext, name: &TableReference) -> bool {
    ctx.table_provider(name.clone())
        .await
        .is_ok_and(|provider| {
            provider
                .downcast_ref::<datafusion::datasource::MemTable>()
                .is_some()
        })
}

/// Whether `plan` reads an in-memory table, directly or through views.
fn reads_memory_table(plan: &LogicalPlan) -> Result<bool> {
    use datafusion::common::tree_node::TreeNodeRecursion;
    let mut found = false;
    let mut nested = Vec::new();
    plan.apply_with_subqueries(|node| {
        if let LogicalPlan::TableScan(scan) = node
            && let Ok(provider) = datafusion::datasource::source_as_provider(&scan.source)
        {
            if provider
                .downcast_ref::<datafusion::datasource::MemTable>()
                .is_some()
            {
                found = true;
                return Ok(TreeNodeRecursion::Stop);
            }
            if let Some(view) = provider.downcast_ref::<datafusion::datasource::ViewTable>() {
                nested.push(view.logical_plan().clone());
            }
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    if found {
        return Ok(true);
    }
    for view in nested {
        if reads_memory_table(&view)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Whether the name `effect` is about exists in `ctx` right now. A name in
/// a schema or catalog that doesn't exist doesn't exist either: `DROP TABLE
/// IF EXISTS nosuch.t` succeeds in DataFusion, so it must here.
fn exists(ctx: &SessionContext, effect: &Effect) -> bool {
    match effect {
        Effect::SaveRelation { name, .. }
        | Effect::UnsavedRelation { name, .. }
        | Effect::DropRelation { name } => ctx.table_exist(name.clone()).unwrap_or(false),
        Effect::SaveSchema { name } => {
            let qualified = qualified_schema(ctx, name);
            let (catalog, schema) = qualified
                .split_once('.')
                .expect("a qualified schema name has a dot");
            ctx.catalog(catalog)
                .is_some_and(|catalog| catalog.schema(schema).is_some())
        }
        Effect::SaveCatalog { name } => ctx.catalog(name).is_some(),
        Effect::Other => false,
    }
}

/// Refuse a view that reads a file by relative path (`FROM 'data/t.csv'`):
/// replayed from another working directory it would read another file, or
/// none. Such a scan reads a listing table under a name no table is
/// registered as — which tells it from a table with a path-like name, and
/// from a table function (`generate_series(1, 3)`), which reads no file.
fn refuse_relative_paths(ctx: &SessionContext, plan: &LogicalPlan) -> Result<()> {
    use datafusion::common::tree_node::TreeNodeRecursion;
    let mut relative = None;
    plan.apply_with_subqueries(|node| {
        if let LogicalPlan::TableScan(scan) = node {
            let name = scan.table_name.table();
            let is_path = !name.contains("://") && Path::new(name).is_relative();
            let reads_files =
                datafusion::datasource::source_as_provider(&scan.source).is_ok_and(|provider| {
                    provider
                        .downcast_ref::<datafusion::datasource::listing::ListingTable>()
                        .is_some()
                });
            if is_path && reads_files && !ctx.table_exist(scan.table_name.clone()).unwrap_or(false)
            {
                relative = Some(name.to_owned());
                return Ok(TreeNodeRecursion::Stop);
            }
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    match relative {
        Some(path) => Err(SemcastError::Store(format!(
            "this view reads '{path}' by relative path, which a restarted server \
             would resolve from its own directory; give an absolute path, or \
             read the file through CREATE EXTERNAL TABLE"
        ))),
        None => Ok(()),
    }
}

/// Whether any schema of any catalog still has a table of bare name `table`.
fn bare_name_in_use(ctx: &SessionContext, table: &str) -> bool {
    ctx.catalog_names().iter().any(|name| {
        ctx.catalog(name).is_some_and(|catalog| {
            catalog.schema_names().iter().any(|schema| {
                catalog
                    .schema(schema)
                    .is_some_and(|schema| schema.table_exist(table))
            })
        })
    })
}

fn resolve(ctx: &SessionContext, name: &TableReference) -> String {
    let state = ctx.state();
    let options = &state.config().options().catalog;
    name.clone()
        .resolve(&options.default_catalog, &options.default_schema)
        .to_string()
}

fn qualified_schema(ctx: &SessionContext, name: &str) -> String {
    if name.contains('.') {
        name.to_owned()
    } else {
        let state = ctx.state();
        format!(
            "{}.{name}",
            state.config().options().catalog.default_catalog
        )
    }
}

/// `sql` with its `LOCATION` made absolute, when it is a relative local
/// path: DataFusion resolves one against the working directory, which a
/// restarted server need not share.
fn absolute_location(sql: &str, location: &str) -> Result<String> {
    let path = Path::new(location);
    if location.contains("://") || !path.is_relative() {
        return Ok(sql.to_owned());
    }
    let mut absolute = std::path::absolute(path)
        .map_err(|error| SemcastError::Store(format!("resolving {location}: {error}")))?
        .to_str()
        .ok_or_else(|| SemcastError::Store(format!("{location} is not valid UTF-8")))?
        .to_owned();
    // A trailing slash marks a directory of files to DataFusion.
    if location.ends_with('/') && !absolute.ends_with('/') {
        absolute.push('/');
    }
    let quote = |text: &str| format!("'{}'", text.replace('\'', "''"));
    let literal = quote(location);
    if sql.matches(&literal).count() != 1 {
        return Err(SemcastError::Store(format!(
            "cannot save LOCATION {location} as an absolute path; give an absolute \
             path, so a restarted server finds the same file"
        )));
    }
    Ok(sql.replacen(&literal, &quote(&absolute), 1))
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A `CREATE TABLE ... AS` runs unlocked. If another statement
    /// redefines its name before it gets the lock to save, that statement's
    /// saved definition must stand.
    #[tokio::test]
    async fn a_create_table_as_does_not_forget_a_name_redefined_while_it_waited() {
        let dir = tempfile::tempdir().unwrap();
        let csv = dir.path().join("t.csv");
        std::fs::write(&csv, "id\n1\n2\n3\n").unwrap();
        let catalog = Catalog::open(dir.path().join("store.db")).unwrap();
        let ctx = crate::SemcastContextBuilder::new(Arc::new(crate::model::MockModel::default()))
            .with_catalog(catalog.clone())
            .build();

        let held = catalog.lock().await;
        let ctas = {
            let ctx = ctx.clone();
            tokio::spawn(async move { crate::sql(&ctx, "CREATE TABLE t AS VALUES (1)").await })
        };
        while !ctx.table_exist("t").unwrap() {
            tokio::task::yield_now().await;
        }
        // The competing statement, as it would run under the lock.
        let external = format!(
            "CREATE OR REPLACE EXTERNAL TABLE t STORED AS CSV LOCATION '{}' \
             OPTIONS ('format.has_header' 'true')",
            csv.display()
        );
        ctx.sql(&external).await.unwrap();
        let entry = Entry {
            kind: EntryKind::Relation,
            name: "datafusion.public.t".to_owned(),
            definition: external,
        };
        catalog.apply(vec![Change::Put(entry)]).await.unwrap();
        drop(held);
        ctas.await.unwrap().unwrap();

        let saved: Vec<_> = catalog
            .entries()
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        assert_eq!(saved, vec!["datafusion.public.t".to_owned()]);
    }

    #[test]
    fn a_relative_location_is_saved_absolute() {
        let sql = "CREATE EXTERNAL TABLE t STORED AS CSV LOCATION 'data/t.csv'";
        let saved = absolute_location(sql, "data/t.csv").unwrap();
        let expected = std::path::absolute("data/t.csv").unwrap();
        assert!(
            saved.ends_with(&format!("LOCATION '{}'", expected.display())),
            "{saved}"
        );
    }

    #[test]
    fn absolute_paths_and_urls_are_saved_as_written() {
        for location in ["/data/t.csv", "s3://bucket/t.csv", "file:///data/t.csv"] {
            let sql = format!("CREATE EXTERNAL TABLE t STORED AS CSV LOCATION '{location}'");
            assert_eq!(absolute_location(&sql, location).unwrap(), sql);
        }
    }

    #[test]
    fn a_directory_location_keeps_its_trailing_slash() {
        let sql = "CREATE EXTERNAL TABLE t STORED AS PARQUET LOCATION 'data/'";
        assert!(
            absolute_location(sql, "data/")
                .unwrap()
                .ends_with("/data/'")
        );
    }

    #[test]
    fn an_ambiguous_location_is_refused_rather_than_guessed() {
        let sql = "CREATE EXTERNAL TABLE t STORED AS CSV LOCATION 'a' OPTIONS ('x' 'a')";
        assert!(absolute_location(sql, "a").is_err());
    }
}
