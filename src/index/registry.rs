//! Session-level state the planner needs beyond the plan itself: the model
//! provider and which semantic indexes exist. Stored as a `SessionConfig`
//! extension so both the public API (`create_semantic_index`) and the
//! physical planner reach it without new plumbing.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::model::ModelProvider;

use super::SemanticIndex;

#[derive(Debug)]
pub struct SemcastRuntime {
    /// The session's model — verify calls and the default embedder.
    pub model: Arc<dyn ModelProvider>,
    index_root: PathBuf,
    /// Registered indexes keyed by `(table, column)`. Resolution is exact:
    /// a same-named column on a different table never borrows an index.
    indexes: Mutex<HashMap<(String, String), Arc<dyn SemanticIndex>>>,
}

impl SemcastRuntime {
    pub fn new(model: Arc<dyn ModelProvider>) -> Self {
        Self {
            model,
            index_root: std::env::temp_dir().join("semcast-indexes"),
            indexes: Mutex::new(HashMap::new()),
        }
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
}
