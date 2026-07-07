//! SQL surface: how semantic operators enter a query.
//!
//! Two layers, both landing on the same marker: [`SemcastDialect`] parses
//! infix `text MEANS 'condition'` and desugars it to the `means()` scalar
//! UDF, which the optimizer rewrites into a [`SemFilterNode`] before anything
//! tries to evaluate it. Calling `means(text, 'condition')` directly works
//! too. Still to come: `WITH RECALL`, `BUDGET`, and the `CREATE SEMANTIC ...`
//! statements (a wrapped-parser extension, not a dialect hook).
//!
//! [`SemFilterNode`]: crate::logical::SemFilterNode

pub mod ddl;
pub mod dialect;
pub mod means_udf;

pub use dialect::SemcastDialect;
