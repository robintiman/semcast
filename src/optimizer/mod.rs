//! Logical optimizer rules (`OptimizerRule`).
//!
//! To the optimizer, `MEANS` is simply a very expensive predicate in a
//! framework that has always reordered predicates by cost. These rules turn
//! the marker UDFs into extension nodes ([`rewrite`], [`extract`], [`rank`]),
//! derive the cheap-then-verify funnel ([`funnel`]), and keep lossy shortcuts
//! honest under a recall target ([`calibrate`]).

pub mod calibrate;
pub mod classify;
pub mod cluster;
pub mod distinct;
pub mod extract;
pub mod funnel;
pub mod rank;
pub mod rewrite;
