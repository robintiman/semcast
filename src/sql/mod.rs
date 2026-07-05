//! SQL surface: how semantic operators enter a query.
//!
//! For now the entry point is a plain scalar UDF, `means(text, 'condition')`,
//! which the optimizer rewrites into a [`SemFilterNode`] before anything
//! tries to evaluate it. The real surface syntax — infix `MEANS`,
//! `WITH RECALL`, `BUDGET`, and the `CREATE SEMANTIC ...` statements — needs
//! parser extensions and lands after roadmap step 1.
//!
//! [`SemFilterNode`]: crate::logical::SemFilterNode

pub mod ddl;
pub mod means_udf;
