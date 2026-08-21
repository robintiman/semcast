//! `WITH RECALL` — sampled threshold calibration (roadmap step 3).
//!
//! A lossy pre-filter is an approximation, and semcast treats it as one:
//! given a recall target, sample rows that survive the free predicates, get
//! ground-truth labels from the model reading the full text, and set the
//! index threshold so the target fraction of true matches survives (the
//! cascade technique pioneered by LOTUS). Labels are ordinary full-text
//! verify verdicts, so they land in — and draw from — the verdict cache.
//!
//! Two ways to read the same sample:
//!
//! - **Point estimate** (`WITH RECALL` alone). The floor is the one that hits
//!   the target *on the sample*. Cheap and unbiased, but a sample of three
//!   positives will happily report 0.9 — the number is an estimate and the
//!   plan says so.
//! - **Certified** (`WITH RECALL … WITH CONFIDENCE c`). The floor is the one
//!   whose Wilson lower bound on recall clears the target at confidence `c`,
//!   so the guarantee holds for the population and not just the rows that
//!   happened to be drawn. Strictly more conservative: it prunes less, and on
//!   a sample too small to certify anything it prunes nothing at all rather
//!   than reporting a number it cannot defend.
//!
//! The bound is Bonferroni-corrected across the candidate floors, because the
//! floor is chosen using the same sample that scores it — testing every
//! candidate at the full confidence would certify one of them by luck.
//!
//! Deliberately deferred: LOTUS-style importance sampling, and BARGAIN's
//! adaptive betting confidence sequences, which spend a label budget more
//! efficiently than a fixed split does.

/// How many free-predicate-surviving documents get ground-truth labels before
/// the first calibration attempt. The uncertified path stops here.
pub const DEFAULT_CALIBRATION_SAMPLE: usize = 64;

/// Ceiling on labels a *certified* calibration will buy. Certifying a bound
/// needs more evidence than estimating one, so the sample grows — a tranche at
/// a time, stopping the moment the target certifies.
pub const CALIBRATION_LABEL_BUDGET: usize = 256;

#[derive(Debug, Clone, PartialEq)]
pub struct Calibration {
    /// Index score threshold that meets the recall target on the sample.
    pub threshold: f32,
    /// Rows labeled to establish it — this is the calibration cost.
    pub sampled_rows: usize,
    /// Recall at `threshold` on the sample itself.
    pub estimated_recall: f64,
    /// The recall this sample *certifies* at the requested confidence, when
    /// one was requested. `None` means the point-estimate path, where nothing
    /// is certified and `estimated_recall` is all there is.
    pub certified_recall: Option<f64>,
}

impl Calibration {
    /// Whether the target is actually met — by the certified bound when one
    /// was asked for, by the point estimate otherwise. A false here is not an
    /// error: the floor still keeps every positive it can, and the caller
    /// reports the shortfall rather than pretending.
    pub fn meets(&self, target_recall: f64) -> bool {
        self.certified_recall.unwrap_or(self.estimated_recall) >= target_recall
    }
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
/// sample's true matches in the funnel.
///
/// `confidence` picks the reading: `None` estimates, `Some(c)` certifies at
/// `c`. The sample is the authority either way: no clamping toward
/// `default_floor`, which is only the fallback when the sample says nothing
/// (no positives at all, or none the floor can affect).
pub fn calibrate_threshold(
    target_recall: f64,
    confidence: Option<f64>,
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
            certified_recall: confidence.map(|_| 1.0),
        };
    }

    let threshold = match confidence {
        None => point_estimate_floor(target_recall, default_floor, &positive, scores, total),
        Some(confidence) => {
            certified_floor(target_recall, confidence, &positive, scores, total).unwrap_or_else(
                || {
                    // Nothing certifies: keep every positive the floor can
                    // reach and report the shortfall honestly.
                    positive.last().copied().unwrap_or(default_floor)
                },
            )
        }
    };

    let surviving = positive.iter().filter(|score| **score >= threshold).count();
    let kept = surviving + scores.positive_unindexed;
    Calibration {
        threshold,
        sampled_rows: scores.sampled,
        estimated_recall: kept as f64 / total as f64,
        certified_recall: confidence
            .map(|confidence| wilson_lower_bound(kept, total, 1.0 - confidence)),
    }
}

/// The floor that hits the target on the sample, counted exactly.
fn point_estimate_floor(
    target_recall: f64,
    default_floor: f32,
    positive: &[f32],
    scores: &SampledScores,
    total: usize,
) -> f32 {
    // How many scored positives must survive, after the unindexed ones that
    // survive any floor are counted toward the target.
    let needed =
        ((target_recall * total as f64).ceil() as usize).saturating_sub(scores.positive_unindexed);
    if needed == 0 {
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
    }
}

/// The highest floor whose Wilson lower bound on recall clears the target.
///
/// Every distinct positive score is a candidate floor, and each is tested at
/// `δ/m` rather than `δ` so the whole search holds at `1 - δ` — the floor is
/// picked with the same sample that scores it, and without the correction one
/// of `m` candidates would clear the bar by luck.
///
/// `None` when no candidate certifies.
fn certified_floor(
    target_recall: f64,
    confidence: f64,
    positive: &[f32],
    scores: &SampledScores,
    total: usize,
) -> Option<f32> {
    // Descending and deduped: `positive` is already sorted high to low.
    let mut candidates: Vec<f32> = Vec::new();
    for score in positive {
        if candidates.last() != Some(score) {
            candidates.push(*score);
        }
    }
    if candidates.is_empty() {
        return None;
    }
    let alpha = (1.0 - confidence) / candidates.len() as f64;

    // Highest floor first: the first that certifies prunes the most.
    candidates.into_iter().find(|floor| {
        let kept =
            positive.iter().filter(|score| *score >= floor).count() + scores.positive_unindexed;
        wilson_lower_bound(kept, total, alpha) >= target_recall
    })
}

/// One-sided lower confidence bound on a binomial proportion, Wilson score.
///
/// Chosen over the textbook normal approximation because it stays inside
/// `[0, 1]` and stays sane at `k == n` — which is exactly where a small
/// calibration sample lives, and where the normal interval would claim a
/// bound of 1.0 from three observations.
fn wilson_lower_bound(successes: usize, trials: usize, alpha: f64) -> f64 {
    if trials == 0 {
        return 0.0;
    }
    let n = trials as f64;
    let p = successes as f64 / n;
    let z = normal_quantile(1.0 - alpha);
    let z2 = z * z;
    let center = p + z2 / (2.0 * n);
    let margin = z * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt();
    ((center - margin) / (1.0 + z2 / n)).clamp(0.0, 1.0)
}

/// Inverse standard-normal CDF — Acklam's rational approximation, accurate to
/// roughly 1.15e-9 over `(0, 1)`, which is far past what a threshold search
/// over a few hundred labels can notice.
///
/// The coefficients are transcribed from the published approximation rather
/// than rounded to the shortest literal that round-trips, so they can be
/// checked against the source; `excessive_precision` is allowed for that
/// reason, not overlooked.
#[allow(clippy::excessive_precision)]
fn normal_quantile(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    /// Where the central rational approximation gives way to the tail one.
    const BREAK: f64 = 0.02425;

    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if !(BREAK..=1.0 - BREAK).contains(&p) {
        // Tails, mirrored around the median.
        let upper = p > 0.5;
        let q = (-2.0 * if upper { 1.0 - p } else { p }.ln()).sqrt();
        let x = (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
        return if upper { -x } else { x };
    }
    let q = p - 0.5;
    let r = q * q;
    (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
        / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
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

    /// The uncertified path, unchanged: `WITH RECALL` alone must cost and
    /// prune exactly what it always did.
    mod point_estimate {
        use super::*;

        #[test]
        fn picks_the_kth_highest_positive_score() {
            // 10 positives, target 0.9 → keep 9 → threshold is the 9th highest.
            let sample = scores(&[0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]);
            let calibration = calibrate_threshold(0.9, None, 0.35, &sample);
            assert_eq!(calibration.threshold, 0.55);
            assert_eq!(calibration.estimated_recall, 0.9);
            assert_eq!(calibration.certified_recall, None);
        }

        #[test]
        fn full_recall_keeps_every_scored_positive() {
            let sample = scores(&[0.9, 0.2, 0.6]);
            let calibration = calibrate_threshold(1.0, None, 0.35, &sample);
            assert_eq!(calibration.threshold, 0.2);
            assert_eq!(calibration.estimated_recall, 1.0);
        }

        #[test]
        fn ties_at_the_threshold_survive() {
            // Keep 2 of 4 → threshold 0.7, but three positives sit at ≥ 0.7.
            let sample = scores(&[0.7, 0.7, 0.7, 0.4]);
            let calibration = calibrate_threshold(0.5, None, 0.35, &sample);
            assert_eq!(calibration.threshold, 0.7);
            assert_eq!(calibration.estimated_recall, 0.75);
        }

        #[test]
        fn high_scoring_positives_raise_the_floor() {
            // The payoff case: every positive scores far above the default floor.
            let sample = scores(&[0.98, 0.97, 0.96]);
            let calibration = calibrate_threshold(0.9, None, 0.35, &sample);
            assert_eq!(calibration.threshold, 0.96);
        }

        #[test]
        fn no_positives_falls_back_to_the_default_floor() {
            let sample = SampledScores {
                sampled: 12,
                ..Default::default()
            };
            let calibration = calibrate_threshold(0.9, None, 0.35, &sample);
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
            let calibration = calibrate_threshold(0.9, None, 0.35, &sample);
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
            let calibration = calibrate_threshold(0.9, None, 0.35, &sample);
            assert_eq!(calibration.threshold, 0.8);
            assert_eq!(calibration.estimated_recall, 0.25);
            assert!(!calibration.meets(0.9));
        }

        #[test]
        fn negative_similarity_scores_are_kept_when_the_target_demands_it() {
            // Cosine scores live in [-1, 1]; a floor below zero is legal.
            let sample = scores(&[0.6, -0.2]);
            let calibration = calibrate_threshold(1.0, None, 0.35, &sample);
            assert_eq!(calibration.threshold, -0.2);
        }
    }

    mod certified {
        use super::*;

        /// The headline property: certifying never keeps fewer true matches
        /// than estimating, because it must defend the number against the
        /// sample size rather than read it off.
        ///
        /// Stated as recall, not as the floor. A higher floor is not the same
        /// as worse recall — the point estimate's "passthroughs already cover
        /// the target" shortcut returns the *default* floor rather than the
        /// highest floor that still keeps every positive, so certification can
        /// land above it while retaining exactly the same matches. Pruning
        /// more negatives at identical recall is the good direction.
        #[test]
        fn never_keeps_fewer_matches_than_the_point_estimate() {
            let samples = [
                scores(&[0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]),
                scores(&[0.98, 0.97, 0.96]),
                scores(&[0.7, 0.7, 0.7, 0.4]),
                SampledScores {
                    positive_scores: vec![0.9, 0.5],
                    positive_unindexed: 3,
                    sampled: 20,
                    ..Default::default()
                },
            ];
            for sample in samples {
                for target in [0.5, 0.8, 0.9, 0.95] {
                    let estimated = calibrate_threshold(target, None, 0.35, &sample);
                    let certified = calibrate_threshold(target, Some(0.95), 0.35, &sample);
                    assert!(
                        certified.estimated_recall >= estimated.estimated_recall,
                        "target {target}, sample {sample:?}: certified recall \
                         {} fell below the point estimate's {}",
                        certified.estimated_recall,
                        estimated.estimated_recall,
                    );
                }
            }
        }

        /// Three positives cannot certify 90% recall at 95% confidence, so
        /// the floor drops to keep them all rather than claiming a bound.
        #[test]
        fn a_tiny_sample_certifies_nothing_and_prunes_nothing() {
            let sample = scores(&[0.98, 0.97, 0.96]);
            let calibration = calibrate_threshold(0.9, Some(0.95), 0.35, &sample);
            assert_eq!(
                calibration.threshold, 0.96,
                "every positive kept when nothing certifies",
            );
            assert!(!calibration.meets(0.9), "and it says so");
        }

        /// With enough evidence the bound clears and the floor does rise.
        #[test]
        fn a_large_sample_certifies_and_prunes() {
            // 200 positives, the bottom 4 scoring low: dropping them costs 2%
            // recall, which 196/200 certifies comfortably at 80%.
            let mut positive_scores = vec![0.9_f32; 196];
            positive_scores.extend([0.1, 0.1, 0.1, 0.1]);
            let sample = SampledScores {
                sampled: 400,
                positive_scores,
                ..Default::default()
            };
            let calibration = calibrate_threshold(0.8, Some(0.95), 0.35, &sample);
            assert_eq!(calibration.threshold, 0.9, "the low scorers are pruned");
            assert!(calibration.meets(0.8));
            let certified = calibration.certified_recall.unwrap();
            assert!(
                certified >= 0.8 && certified < calibration.estimated_recall,
                "a bound sits below the point estimate: {certified} vs {}",
                calibration.estimated_recall,
            );
        }

        #[test]
        fn no_positives_certifies_vacuously() {
            let sample = SampledScores {
                sampled: 12,
                ..Default::default()
            };
            let calibration = calibrate_threshold(0.9, Some(0.95), 0.35, &sample);
            assert_eq!(calibration.threshold, 0.35);
            assert_eq!(calibration.certified_recall, Some(1.0));
        }

        /// More confidence demanded means less pruning, never more.
        #[test]
        fn higher_confidence_is_monotonically_more_conservative() {
            let mut positive_scores = vec![0.9_f32; 90];
            positive_scores.extend(vec![0.2_f32; 10]);
            let sample = SampledScores {
                sampled: 200,
                positive_scores,
                ..Default::default()
            };
            let floors: Vec<f32> = [0.8, 0.9, 0.99]
                .into_iter()
                .map(|c| calibrate_threshold(0.85, Some(c), 0.35, &sample).threshold)
                .collect();
            assert!(
                floors[0] >= floors[1] && floors[1] >= floors[2],
                "floors must not rise with confidence: {floors:?}",
            );
        }
    }

    mod statistics {
        use super::*;

        /// Textbook one-sided critical values.
        #[test]
        fn normal_quantile_matches_known_values() {
            for (p, expected) in [
                (0.90, 1.281_551_6),
                (0.95, 1.644_853_6),
                (0.975, 1.959_964_0),
                (0.99, 2.326_347_9),
                (0.999, 3.090_232_3),
            ] {
                let z = normal_quantile(p);
                assert!(
                    (z - expected).abs() < 1e-6,
                    "quantile({p}) = {z}, expected {expected}",
                );
            }
        }

        #[test]
        fn normal_quantile_is_symmetric_about_the_median() {
            assert!(normal_quantile(0.5).abs() < 1e-12);
            for p in [0.001, 0.02, 0.2, 0.4] {
                let (low, high) = (normal_quantile(p), normal_quantile(1.0 - p));
                assert!((low + high).abs() < 1e-6, "{low} vs {high}");
            }
        }

        /// A perfect sample still cannot certify 1.0 — the whole point of a
        /// bound over a point estimate.
        #[test]
        fn a_perfect_small_sample_does_not_certify_certainty() {
            let bound = wilson_lower_bound(3, 3, 0.05);
            assert!(bound > 0.0 && bound < 1.0, "3/3 certified {bound}");
            assert!(
                wilson_lower_bound(300, 300, 0.05) > bound,
                "more evidence must certify more",
            );
        }

        #[test]
        fn the_bound_never_exceeds_the_point_estimate() {
            for (k, n) in [(0, 10), (1, 10), (5, 10), (9, 10), (10, 10), (50, 100)] {
                let bound = wilson_lower_bound(k, n, 0.05);
                assert!(
                    bound <= k as f64 / n as f64 + 1e-12,
                    "{k}/{n} bound {bound} exceeded the estimate",
                );
                assert!((0.0..=1.0).contains(&bound), "{k}/{n} bound {bound}");
            }
        }

        #[test]
        fn an_empty_sample_certifies_nothing() {
            assert_eq!(wilson_lower_bound(0, 0, 0.05), 0.0);
        }
    }
}
