//! Physical execution — `QueryPlanner`, `ExtensionPlanner`, and the custom
//! `ExecutionPlan`s that actually spend model calls.

pub mod planner;
pub mod verify;

pub use verify::VerifyExec;
