//! `CREATE SEMANTIC INDEX / TYPE / PREDICATE` statements.
//!
//! DataFusion hook: a custom statement parser (sqlparser dialect extension)
//! feeding a planner extension. Roadmap: `CREATE SEMANTIC INDEX` arrives with
//! step 2; `TYPE` and `PREDICATE` with typed extraction.

use crate::types::SemanticType;

#[derive(Debug, Clone, PartialEq)]
pub enum SemanticDdl {
    /// `CREATE SEMANTIC INDEX ON table(column)`
    CreateIndex { table: String, column: String },
    /// `CREATE SEMANTIC TYPE Name AS (...)`
    CreateType(SemanticType),
    /// `CREATE SEMANTIC PREDICATE name(params) AS template [CHEAP USING ...]`
    CreatePredicate {
        name: String,
        params: Vec<String>,
        /// `MEANS` template with `{param}` placeholders.
        template: String,
        /// Optional `CHEAP USING` override — a user-pinned pre-filter.
        cheap_using: Option<String>,
    },
}

/// Recognize a semantic DDL statement; `Ok(None)` means "not ours, let
/// DataFusion parse it".
pub fn parse_semantic_ddl(_sql: &str) -> crate::Result<Option<SemanticDdl>> {
    todo!("CREATE SEMANTIC ... parsing (roadmap step 2 for INDEX)")
}
