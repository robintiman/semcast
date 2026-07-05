//! Logical extension nodes — `UserDefinedLogicalNodeCore` implementations
//! that appear in plans as `LogicalPlan::Extension`.

pub mod sem_extract;
pub mod sem_filter;

pub use sem_extract::SemExtractNode;
pub use sem_filter::SemFilterNode;
