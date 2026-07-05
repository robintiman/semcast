//! `WITH RECALL` — sampled threshold calibration (roadmap step 3).
//!
//! A lossy pre-filter is an approximation, and semcast treats it as one:
//! given a recall target, sample a few hundred rows that survive the free
//! predicates, get ground-truth labels from the model, and set the index
//! threshold so the target fraction of true matches survives (the cascade
//! technique pioneered by LOTUS). Calibration cost appears in `EXPLAIN` and
//! is cached for repeat questions of the same shape.

#[derive(Debug, Clone, PartialEq)]
pub struct Calibration {
    /// Index score threshold that meets the recall target on the sample.
    pub threshold: f32,
    /// Rows labeled to establish it — this is the calibration cost.
    pub sampled_rows: usize,
    pub estimated_recall: f64,
}

pub fn calibrate_threshold(
    _target_recall: f64,
    _sample_size: usize,
) -> crate::Result<Calibration> {
    todo!("sampled threshold calibration (roadmap step 3)")
}
