//! Session-level state the planner needs beyond the plan itself: the model
//! provider and which semantic indexes exist. Stored as a `SessionConfig`
//! extension so both the public API (`create_semantic_index`) and the
//! physical planner reach it without new plumbing.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
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
    index_root: PathBuf,
    /// Registered indexes keyed by `(table, column)`. Resolution is exact:
    /// a same-named column on a different table never borrows an index.
    indexes: Mutex<HashMap<(String, String), Arc<dyn SemanticIndex>>>,
    /// `CREATE SEMANTIC TYPE` definitions. Shared with the marker UDFs, which
    /// resolve a type's fields at plan time — hence an `Arc`, not inline.
    types: Arc<TypeRegistry>,
}

impl SemcastRuntime {
    pub fn new(model: Arc<dyn ModelProvider>) -> Self {
        Self {
            embedder: Arc::clone(&model),
            model,
            index_root: std::env::temp_dir().join("semcast-indexes"),
            indexes: Mutex::new(HashMap::new()),
            types: Arc::new(TypeRegistry::default()),
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

    /// Store Lance datasets under `root` instead of the temp-dir default.
    pub fn with_index_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.index_root = root.into();
        self
    }

    /// Where Lance datasets go when `IndexOptions.path` is not set.
    pub fn index_root(&self) -> &Path {
        &self.index_root
    }

    pub fn register_index(&self, table: &str, column: &str, index: Arc<dyn SemanticIndex>) {
        self.indexes
            .lock()
            .expect("index registry poisoned")
            .insert((table.to_owned(), column.to_owned()), index);
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
