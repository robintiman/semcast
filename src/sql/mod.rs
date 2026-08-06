//! SQL surface: how semantic operators enter a query.
//!
//! Two layers, both landing on the same marker: [`SemcastDialect`] parses
//! infix `text MEANS 'condition'` and desugars it to the `means()` scalar
//! UDF, which the optimizer rewrites into a [`SemFilterNode`] before anything
//! tries to evaluate it. `text RELEVANCE TO 'query'` works the same way
//! through `relevance()` and [`SemRankNode`]. Calling either marker directly
//! works too.
//!
//! Statement-level syntax has no dialect hook, so [`statement`] owns the
//! parser instead and dispatches on its token stream: the `CREATE SEMANTIC ...`
//! grammar lives in [`ddl`], the trailing `WITH RECALL` / `WITH SIMILARITY`
//! knobs in [`recall`]. [`head`] reads a statement's leading words for the few
//! callers that must recognize one without parsing it. AST passes that run
//! before planning live in [`typed`] and [`rank`]. Still to come: `BUDGET`.
//!
//! [`SemFilterNode`]: crate::logical::SemFilterNode
//! [`SemRankNode`]: crate::logical::SemRankNode

pub mod cluster;
pub mod cluster_udf;
pub mod ddl;
pub mod dialect;
pub mod distinct;
pub mod distinct_udf;
pub mod extract_udf;
pub(crate) mod head;
pub mod means_udf;
pub mod rank;
pub mod rank_udf;
pub mod recall;
pub mod statement;
pub mod typed;

pub use dialect::SemcastDialect;
pub use statement::{SemcastStatement, parse_semcast_statement};
