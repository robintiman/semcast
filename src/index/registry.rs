//! Session-level state the planner needs beyond the plan itself: the model
//! provider and which semantic indexes exist. Stored as a `SessionConfig`
//! extension so both the public API (`create_semantic_index`) and the
//! physical planner reach it without new plumbing.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use datafusion::common::{Column, DFSchema, DataFusionError, Result};
use datafusion::logical_expr::Expr;

use crate::model::ModelProvider;
use crate::types::registry::TypeRegistry;

use super::SemanticIndex;

#[derive(Debug)]
pub struct SemcastRuntime {
    /// The session's model — verify calls.
    pub model: Arc<dyn ModelProvider>,
    /// What `create_semantic_index` embeds with when `IndexOptions.embedder`
    /// is unset. The session model itself unless the builder brought a
    /// dedicated one (Voyage, or Ollama next to an Anthropic session model).
    embedder: Arc<dyn ModelProvider>,
    /// Where semantic indexes keep their chunks and vectors: memory unless
    /// the builder brought the store file.
    chunk_store: Arc<dyn crate::index::stored::ChunkStore>,
    /// Registered indexes keyed by `(table, column)`. Resolution is exact:
    /// a same-named column on a different table never borrows an index.
    indexes: Mutex<HashMap<(String, String), Arc<dyn SemanticIndex>>>,
    /// `CREATE SEMANTIC TYPE` definitions. Shared with the marker UDFs, which
    /// resolve a type's fields at plan time — hence an `Arc`, not inline.
    types: Arc<TypeRegistry>,
    /// Where definitions are saved so they survive a restart; `None` keeps
    /// them for the process only.
    catalog: Option<crate::catalog::Catalog>,
}

impl SemcastRuntime {
    pub fn new(model: Arc<dyn ModelProvider>) -> Self {
        Self {
            embedder: Arc::clone(&model),
            model,
            chunk_store: Arc::new(crate::index::stored::MemoryChunkStore::default()),
            indexes: Mutex::new(HashMap::new()),
            types: Arc::new(TypeRegistry::default()),
            catalog: None,
        }
    }

    /// Embed through `embedder` instead of the session model.
    pub fn with_embedder(mut self, embedder: Arc<dyn ModelProvider>) -> Self {
        self.embedder = embedder;
        self
    }

    /// The session's default embedding provider.
    pub fn embedder(&self) -> &Arc<dyn ModelProvider> {
        &self.embedder
    }

    /// Share the type registry with a caller-provided `Arc` — the builder
    /// creates one registry and hands the same handle to the marker UDFs.
    pub fn with_type_registry(mut self, types: Arc<TypeRegistry>) -> Self {
        self.types = types;
        self
    }

    /// The session's semantic type registry.
    pub fn type_registry(&self) -> &Arc<TypeRegistry> {
        &self.types
    }

    /// Keep semantic-index chunks and vectors in `store` instead of memory.
    pub fn with_chunk_store(mut self, store: Arc<dyn crate::index::stored::ChunkStore>) -> Self {
        self.chunk_store = store;
        self
    }

    /// Where semantic indexes keep their chunks and vectors.
    pub fn chunk_store(&self) -> &Arc<dyn crate::index::stored::ChunkStore> {
        &self.chunk_store
    }

    /// Save definitions to `catalog` as they are made.
    pub fn with_catalog(mut self, catalog: crate::catalog::Catalog) -> Self {
        self.catalog = Some(catalog);
        self
    }

    /// The durable catalog, when the session has one.
    pub fn catalog(&self) -> Option<&crate::catalog::Catalog> {
        self.catalog.as_ref()
    }

    pub fn register_index(&self, table: &str, column: &str, index: Arc<dyn SemanticIndex>) {
        self.indexes
            .lock()
            .expect("index registry poisoned")
            .insert((table.to_owned(), column.to_owned()), index);
    }

    /// Forget every index on `table` — the table was dropped, and an index
    /// must not outlive it into a same-named table created later.
    pub fn unregister_indexes(&self, table: &str) {
        self.indexes
            .lock()
            .expect("index registry poisoned")
            .retain(|(indexed, _), _| indexed != table);
    }

    pub fn index_for(&self, table: &str, column: &str) -> Option<Arc<dyn SemanticIndex>> {
        self.indexes
            .lock()
            .expect("index registry poisoned")
            .get(&(table.to_owned(), column.to_owned()))
            .cloned()
    }

    /// The index covering whatever column `text` reads, or `None` when the
    /// expression is computed, unqualified, or simply unindexed.
    ///
    /// Shared by the two places that need to know whether a `MEANS` gets a
    /// funnel: the physical planner, which builds one, and the rewrite rule,
    /// which orders the cheap predicates first.
    pub fn index_for_text(&self, text: &Expr, schema: &DFSchema) -> Option<Arc<dyn SemanticIndex>> {
        let column = column_behind_casts(text)?;
        let (table, field) = qualified_name(column, schema).ok()?;
        self.index_for(&table, &field)
    }

    /// Whether a semantic index covers `text` — the plan-time signal for
    /// "this predicate prunes before it pays".
    pub fn covers(&self, text: &Expr, schema: &DFSchema) -> bool {
        self.index_for_text(text, schema).is_some()
    }
}

/// The `(table, column)` a column reference names, as the index registry
/// keys them.
pub fn qualified_name(column: &Column, schema: &DFSchema) -> Result<(String, String)> {
    let (qualifier, field) = schema.qualified_field_from_column(column)?;
    let table = qualifier
        .ok_or_else(|| {
            DataFusionError::Plan(format!(
                "cannot tell which table `{column}` belongs to; qualify it as \
                 <table>.<column>"
            ))
        })?
        .table()
        .to_owned();
    Ok((table, field.name().clone()))
}

/// The column a text expression reads, seen through casts and aliases —
/// type coercion wraps string columns in `CAST(... AS Utf8)` for the
/// `means` UDF signature, and a cast between string types doesn't change
/// which document the text is (both stages hash the evaluated text, so the
/// index keys still line up). Anything else is a computed expression: no
/// index.
pub fn column_behind_casts(expr: &Expr) -> Option<&Column> {
    match expr {
        Expr::Column(column) => Some(column),
        Expr::Cast(cast) => column_behind_casts(&cast.expr),
        Expr::TryCast(cast) => column_behind_casts(&cast.expr),
        Expr::Alias(alias) => column_behind_casts(&alias.expr),
        _ => None,
    }
}
