//! SQL surface: how semantic operators enter a query.
//!
//! Two layers, both landing on the same marker: [`SemcastDialect`] parses
//! infix `text MEANS 'condition'` and desugars it to the `means()` scalar
//! UDF, which the optimizer rewrites into a [`SemFilterNode`] before anything
//! tries to evaluate it. `text RELEVANCE TO 'query'` works the same way
//! through `relevance()` and [`SemRankNode`]. Calling either marker directly
//! works too. Statement-level syntax — trailing `WITH RECALL` ([`recall`])
//! and the `CREATE SEMANTIC ...` statements ([`ddl`]) — is wrapped-parser
//! extension, not a dialect hook. AST passes that run before planning live in
//! [`typed`] and [`rank`]. Still to come: `BUDGET`.
//!
//! [`SemFilterNode`]: crate::logical::SemFilterNode
//! [`SemRankNode`]: crate::logical::SemRankNode

pub mod cluster;
pub mod cluster_udf;
pub mod ddl;
pub mod dialect;
pub mod extract_udf;
pub mod means_udf;
pub mod rank;
pub mod rank_udf;
pub mod recall;
pub mod typed;

pub use dialect::SemcastDialect;
