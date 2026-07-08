//! `WITH RECALL` — sampled threshold calibration (roadmap step 3).
//!
//! A lossy pre-filter is an approximation, and semcast treats it as one:
//! given a recall target, sample rows that survive the free predicates, get
//! ground-truth labels from the model reading the full text, and set the
//! index threshold so the target fraction of true matches survives (the
//! cascade technique pioneered by LOTUS). Labels are ordinary full-text
//! verify verdicts, so they land in — and draw from — the verdict cache.
//!
//! v1 is a point estimate over a small sample. Deliberately deferred:
//! LOTUS-style importance sampling and confidence intervals on the recall
//! estimate, and a per-query sample-size knob (`WITH RECALL 0.9 SAMPLE n`).

/// How many free-predicate-surviving documents get ground-truth labels.
/// The calibration cost ceiling: one full-text model call per uncached doc.
pub const DEFAULT_CALIBRATION_SAMPLE: usize = 64;

#[derive(Debug, Clone, PartialEq)]
pub struct Calibration {
    /// Index score threshold that meets the recall target on the sample.
    pub threshold: f32,
    /// Rows labeled to establish it — this is the calibration cost.
    pub sampled_rows: usize,
    pub estimated_recall: f64,
}

/// Per-document outcomes of labeling a sample against the index.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct SampledScores {
    /// Best chunk score of each positive-labeled document the index search
    /// returned. Order doesn't matter.
    pub positive_scores: Vec<f32>,
    /// Positive documents the index has never seen — the scan passes them
    /// through to verify at any floor, so they help recall for free.
    pub positive_unindexed: usize,
    /// Positive documents that are indexed but fell outside the search's
    /// `fetch_k` — lost at any floor.
    pub positive_lost: usize,
    /// Total documents labeled, positive and negative.
    pub sampled: usize,
}

/// Pick the highest score floor that keeps at least `target_recall` of the
/// sample's true matches in the funnel. The sample is the authority: no
/// clamping toward `default_floor`, which is only the fallback when the
/// sample says nothing (no positives at all, or none the floor can affect).
pub fn calibrate_threshold(
    target_recall: f64,
    default_floor: f32,
    scores: &SampledScores,
) -> Calibration {
    let mut positive = scores.positive_scores.clone();
    positive.sort_by(|a, b| b.total_cmp(a));
    let total = positive.len() + scores.positive_unindexed + scores.positive_lost;
    if total == 0 {
        // No evidence to raise the floor with — keep the default; recall is
        // vacuously met.
        return Calibration {
            threshold: default_floor,
            sampled_rows: scores.sampled,
            estimated_recall: 1.0,
        };
    }

    // How many scored positives must survive, after the unindexed ones that
    // survive any floor are counted toward the target.
    let needed =
        ((target_recall * total as f64).ceil() as usize).saturating_sub(scores.positive_unindexed);
    let threshold = if needed == 0 {
        // Passthroughs alone meet the target — a degenerate sample, not a
        // license to prune aggressively.
        default_floor
    } else if needed > positive.len() {
        // Unachievable: positives beyond fetch_k are lost at any floor.
        // Keep every scored positive and report the shortfall honestly.
        positive.last().copied().unwrap_or(default_floor)
    } else {
        // Survival downstream is `score >= threshold`, so the needed-th
        // highest positive score keeps at least `needed` positives; ties
        // only help.
        positive[needed - 1]
    };

    let surviving = positive.iter().filter(|score| **score >= threshold).count();
    Calibration {
        threshold,
        sampled_rows: scores.sampled,
        estimated_recall: (surviving + scores.positive_unindexed) as f64 / total as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scores(positive_scores: &[f32]) -> SampledScores {
        SampledScores {
            positive_scores: positive_scores.to_vec(),
            sampled: positive_scores.len(),
            ..Default::default()
        }
    }

    #[test]
    fn picks_the_kth_highest_positive_score() {
        // 10 positives, target 0.9 → keep 9 → threshold is the 9th highest.
        let sample = scores(&[0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]);
        let calibration = calibrate_threshold(0.9, 0.35, &sample);
        assert_eq!(calibration.threshold, 0.55);
        assert_eq!(calibration.estimated_recall, 0.9);
    }

    #[test]
    fn full_recall_keeps_every_scored_positive() {
        let sample = scores(&[0.9, 0.2, 0.6]);
        let calibration = calibrate_threshold(1.0, 0.35, &sample);
        assert_eq!(calibration.threshold, 0.2);
        assert_eq!(calibration.estimated_recall, 1.0);
    }

    #[test]
    fn ties_at_the_threshold_survive() {
        // Keep 2 of 4 → threshold 0.7, but three positives sit at ≥ 0.7.
        let sample = scores(&[0.7, 0.7, 0.7, 0.4]);
        let calibration = calibrate_threshold(0.5, 0.35, &sample);
        assert_eq!(calibration.threshold, 0.7);
        assert_eq!(calibration.estimated_recall, 0.75);
    }

    #[test]
    fn high_scoring_positives_raise_the_floor() {
        // The payoff case: every positive scores far above the default floor.
        let sample = scores(&[0.98, 0.97, 0.96]);
        let calibration = calibrate_threshold(0.9, 0.35, &sample);
        assert_eq!(calibration.threshold, 0.96);
    }

    #[test]
    fn no_positives_falls_back_to_the_default_floor() {
        let sample = SampledScores {
            sampled: 12,
            ..Default::default()
        };
        let calibration = calibrate_threshold(0.9, 0.35, &sample);
        assert_eq!(calibration.threshold, 0.35);
        assert_eq!(calibration.estimated_recall, 1.0);
        assert_eq!(calibration.sampled_rows, 12);
    }

    #[test]
    fn unindexed_positives_count_toward_the_target() {
        // 1 scored + 9 unindexed, target 0.9 → the passthroughs already
        // cover it; don't prune off a degenerate sample.
        let sample = SampledScores {
            positive_scores: vec![0.4],
            positive_unindexed: 9,
            sampled: 10,
            ..Default::default()
        };
        let calibration = calibrate_threshold(0.9, 0.35, &sample);
        assert_eq!(calibration.threshold, 0.35);
        assert_eq!(calibration.estimated_recall, 1.0);
    }

    #[test]
    fn lost_positives_make_the_target_unachievable_but_honest() {
        // 1 scored + 3 lost beyond fetch_k: best achievable recall is 0.25.
        let sample = SampledScores {
            positive_scores: vec![0.8],
            positive_lost: 3,
            sampled: 8,
            ..Default::default()
        };
        let calibration = calibrate_threshold(0.9, 0.35, &sample);
        assert_eq!(calibration.threshold, 0.8);
        assert_eq!(calibration.estimated_recall, 0.25);
    }

    #[test]
    fn negative_similarity_scores_are_kept_when_the_target_demands_it() {
        // Cosine scores live in [-1, 1]; a floor below zero is legal.
        let sample = scores(&[0.6, -0.2]);
        let calibration = calibrate_threshold(1.0, 0.35, &sample);
        assert_eq!(calibration.threshold, -0.2);
    }
}
