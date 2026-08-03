//! Physical execution — `QueryPlanner`, `ExtensionPlanner`, and the custom
//! `ExecutionPlan`s that actually spend model calls.

pub mod extract;
pub mod index_scan;
pub mod planner;
pub mod rank;
pub mod trace;
pub mod verify;

pub use extract::SemExtractExec;
pub use index_scan::IndexScanExec;
pub use rank::SemRankExec;
pub use verify::VerifyExec;
