//! Physical execution — `QueryPlanner`, `ExtensionPlanner`, and the custom
//! `ExecutionPlan`s that actually spend model calls.

pub mod index_scan;
pub mod planner;
pub mod verify;

pub use index_scan::IndexScanExec;
pub use verify::VerifyExec;
