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
//! What gets thresholded is [`crate::optimizer::proxy`]'s business: raw
//! similarity, or a classifier fitted to these same labels. Either way the
//! target is enforced here, which is what makes trying a better proxy safe.
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
    /// Positive documents that are indexed but have no score — lost at any
    /// floor. The pre-filter scores every chunk, so it reports none.
    pub positive_lost: usize,
    /// Best chunk score of each scored document whose label failed — a model
    /// error or an unparseable answer. Its truth is unknown, so each floor
    /// assumes the worst for it: below the floor, a match the floor loses;
    /// at or above it, kept, so a non-match that changes nothing. Dropping
    /// them from the sample instead would bias it toward whatever the model
    /// answers reliably. (A failed document the index has never seen passes
    /// through at any floor and is left out; one indexed but unscored counts
    /// in `positive_lost`.)
    pub unverified_scores: Vec<f32>,
    /// Total documents labeled, positive and negative.
    pub sampled: usize,
}

/// Pick the highest score floor that keeps at least `target_recall` of the
/// sample's true matches in the funnel.
///
/// `confidence` picks the reading: `None` estimates, `Some(c)` certifies at
/// `c`. The sample is the authority either way: no clamping toward
/// `default_floor`, which is only the fallback when the sample has no known
/// positive to set a floor by — lowered, if failed labels sit below it, until
/// it keeps them.
pub fn calibrate_threshold(
    target_recall: f64,
    confidence: Option<f64>,
    default_floor: f32,
    scores: &SampledScores,
) -> Calibration {
    let mut positive = scores.positive_scores.clone();
    positive.sort_by(|a, b| b.total_cmp(a));
    let known = positive.len() + scores.positive_unindexed + scores.positive_lost;
    // A labeled sample without a single known match certifies nothing —
    // failed labels or not: the matches it missed are exactly what a bound
    // is about. Prune nothing and report the target unmet, so a certifying
    // caller keeps drawing instead of stopping on an empty proof.
    if known == 0 && confidence.is_some() && scores.sampled > 0 {
        return Calibration {
            threshold: f32::NEG_INFINITY,
            sampled_rows: scores.sampled,
            estimated_recall: 1.0,
            certified_recall: Some(0.0),
        };
    }
    if known == 0 && scores.unverified_scores.is_empty() {
        // No evidence to raise the floor with — keep the default; recall is
        // vacuously met.
        return Calibration {
            threshold: default_floor,
            sampled_rows: scores.sampled,
            estimated_recall: 1.0,
            certified_recall: confidence.map(|_| 1.0),
        };
    }

    let recall_at = |floor: f32| recall_at(floor, &positive, scores);
    // The floor the known labels support, lowered past failed labels until
    // the worst case for them still meets the target.
    let estimated_floor = || {
        let floor = point_estimate_floor(target_recall, default_floor, &positive, scores, known);
        let (kept, total) = recall_at(floor);
        if meets_target(kept, total, target_recall) {
            floor
        } else {
            candidate_floors(&positive, scores)
                .into_iter()
                .filter(|candidate| *candidate < floor)
                .find(|candidate| {
                    let (kept, total) = recall_at(*candidate);
                    meets_target(kept, total, target_recall)
                })
                .unwrap_or(f32::NEG_INFINITY)
        }
    };
    let threshold = match confidence {
        None => estimated_floor(),
        Some(confidence) => {
            // Nothing certifies: prune nothing. A floor at the lowest sampled
            // positive would still drop every unsampled match below it, on a
            // bound the sample just failed to support.
            certified_floor(target_recall, confidence, &positive, scores)
                .unwrap_or(f32::NEG_INFINITY)
        }
    };

    let (kept, total) = recall_at(threshold);
    if total == 0 {
        return Calibration {
            threshold,
            sampled_rows: scores.sampled,
            estimated_recall: 1.0,
            certified_recall: confidence.map(|_| 1.0),
        };
    }
    Calibration {
        threshold,
        sampled_rows: scores.sampled,
        estimated_recall: kept as f64 / total as f64,
        certified_recall: confidence
            .map(|confidence| wilson_lower_bound(kept, total, 1.0 - confidence)),
    }
}

/// Positives kept, and positives in all, at `floor`, assuming the worst for
/// failed labels: those below the floor are lost matches, those at or above
/// it are kept non-matches and count for nothing.
fn recall_at(floor: f32, positive: &[f32], scores: &SampledScores) -> (usize, usize) {
    let kept = positive.iter().filter(|score| **score >= floor).count() + scores.positive_unindexed;
    let unverified_missed = scores
        .unverified_scores
        .iter()
        .filter(|score| **score < floor)
        .count();
    let total =
        positive.len() + scores.positive_unindexed + scores.positive_lost + unverified_missed;
    (kept, total)
}

fn meets_target(kept: usize, total: usize, target_recall: f64) -> bool {
    total == 0 || kept as f64 / total as f64 >= target_recall
}

/// Every floor that changes what survives: each distinct score of a positive
/// or of a failed label, highest first.
fn candidate_floors(positive: &[f32], scores: &SampledScores) -> Vec<f32> {
    let mut candidates: Vec<f32> = positive
        .iter()
        .chain(&scores.unverified_scores)
        .copied()
        .collect();
    candidates.sort_by(|a, b| b.total_cmp(a));
    candidates.dedup();
    candidates
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
        // Unachievable: lost positives are gone at any floor. Prune nothing
        // rather than pile pruning losses on top, and report the shortfall.
        f32::NEG_INFINITY
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
) -> Option<f32> {
    // A failed label can only lower the floor, never raise it: a candidate
    // above every scored positive keeps none of them, and would certify on
    // the strength of unindexed positives alone. With no scored positive,
    // there is no floor to lower, as before.
    let ceiling = positive.first().copied();
    let candidates: Vec<f32> = candidate_floors(positive, scores)
        .into_iter()
        .filter(|candidate| ceiling.is_some_and(|top| *candidate <= top))
        .collect();
    if candidates.is_empty() {
        return None;
    }
    let alpha = (1.0 - confidence) / candidates.len() as f64;

    // Highest floor first: the first that certifies prunes the most.
    candidates.into_iter().find(|floor| {
        let (kept, total) = recall_at(*floor, positive, scores);
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

    #[test]
    fn a_sample_without_matches_certifies_nothing() {
        let none = SampledScores {
            sampled: 64,
            ..Default::default()
        };
        let certified = calibrate_threshold(0.9, Some(0.95), 0.5, &none);
        assert_eq!(certified.threshold, f32::NEG_INFINITY, "prunes nothing");
        assert!(!certified.meets(0.9), "an empty proof is no proof");
        // Estimating keeps the caller's default, as before.
        let estimated = calibrate_threshold(0.9, None, 0.5, &none);
        assert_eq!(estimated.threshold, 0.5);
        // Nothing sampled at all — no rows — is vacuously met.
        let empty = calibrate_threshold(0.9, Some(0.95), 0.5, &SampledScores::default());
        assert!(empty.meets(0.9));
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
            // 1 scored + 3 lost: best achievable recall is 0.25.
            let sample = SampledScores {
                positive_scores: vec![0.8],
                positive_lost: 3,
                sampled: 8,
                ..Default::default()
            };
            let calibration = calibrate_threshold(0.9, None, 0.35, &sample);
            assert_eq!(
                calibration.threshold,
                f32::NEG_INFINITY,
                "an unreachable target prunes nothing",
            );
            assert_eq!(calibration.estimated_recall, 0.25);
            assert!(!calibration.meets(0.9));
        }

        /// A failed label below the floor is a possible lost match, so the
        /// floor drops until it keeps the failures too. Ten positives at 0.9
        /// and two failures at 0.4: a 0.9 floor loses both (10/12 < 0.9), so
        /// the floor falls to 0.4. Dropped from the sample instead, the 0.9
        /// floor would have claimed full recall and pruned them.
        #[test]
        fn failed_labels_below_the_floor_pull_it_down() {
            let sample = SampledScores {
                positive_scores: vec![0.9; 10],
                unverified_scores: vec![0.4, 0.4],
                sampled: 20,
                ..Default::default()
            };
            let calibration = calibrate_threshold(0.9, None, 0.35, &sample);
            assert_eq!(calibration.threshold, 0.4);
            assert_eq!(calibration.estimated_recall, 1.0);

            // Certified: 30 positives at 0.9 and two failures at 0.4. At a 0.9
            // floor the failures are lost and 30/32 does not certify 0.9; at
            // 0.4 they are kept and 30/30 does.
            let sample = SampledScores {
                positive_scores: vec![0.9; 30],
                unverified_scores: vec![0.4, 0.4],
                sampled: 40,
                ..Default::default()
            };
            let certified = calibrate_threshold(0.9, Some(0.9), 0.35, &sample);
            assert_eq!(certified.threshold, 0.4, "{certified:?}");
            assert!(certified.meets(0.9));
        }

        /// A failed label scoring above every positive cannot raise the
        /// certified floor. Here 400 unindexed positives certify on their own;
        /// without the cap, the failure's 0.95 would become the floor and prune
        /// every unsampled document below it.
        #[test]
        fn a_failed_label_never_raises_the_certified_floor() {
            let base = SampledScores {
                positive_scores: vec![0.5, 0.5],
                positive_unindexed: 400,
                sampled: 500,
                ..Default::default()
            };
            let without = calibrate_threshold(0.9, Some(0.95), 0.35, &base);
            let with = calibrate_threshold(
                0.9,
                Some(0.95),
                0.35,
                &SampledScores {
                    unverified_scores: vec![0.95],
                    ..base.clone()
                },
            );
            assert_eq!(without.threshold, 0.5);
            assert_eq!(with.threshold, 0.5, "{with:?}");
        }

        /// A sample with no known positive and one failed label. The estimate
        /// keeps the default floor, lowered only if the failure sits below
        /// it, rather than turning pruning off over one failed call. A
        /// certificate has nothing to stand on either way, so it prunes
        /// nothing and reports the target unmet.
        #[test]
        fn a_lone_failed_label_estimates_but_certifies_nothing() {
            for failed_score in [0.8_f32, 0.2] {
                let sample = SampledScores {
                    unverified_scores: vec![failed_score],
                    sampled: 64,
                    ..Default::default()
                };
                let estimated = calibrate_threshold(0.9, None, 0.35, &sample);
                assert_eq!(estimated.threshold, failed_score.min(0.35));
                assert!(estimated.meets(0.9));
                let certified = calibrate_threshold(0.9, Some(0.9), 0.35, &sample);
                assert_eq!(certified.threshold, f32::NEG_INFINITY);
                assert!(!certified.meets(0.9));
            }
        }

        /// A failed label the floor keeps anyway is no miss: at worst it is a
        /// non-match, which changes nothing. The floor stays where the known
        /// labels put it.
        #[test]
        fn failed_labels_above_the_floor_change_nothing() {
            let sample = SampledScores {
                positive_scores: vec![0.9; 10],
                unverified_scores: vec![0.95, 0.95],
                sampled: 20,
                ..Default::default()
            };
            let with_failures = calibrate_threshold(0.9, None, 0.35, &sample);
            let without = calibrate_threshold(
                0.9,
                None,
                0.35,
                &SampledScores {
                    unverified_scores: Vec::new(),
                    ..sample.clone()
                },
            );
            assert_eq!(with_failures, without);
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
        /// the floor drops away entirely rather than claiming a bound: a
        /// floor at the lowest sampled positive would still prune unsampled
        /// matches below it.
        #[test]
        fn a_tiny_sample_certifies_nothing_and_prunes_nothing() {
            let sample = scores(&[0.98, 0.97, 0.96]);
            let calibration = calibrate_threshold(0.9, Some(0.95), 0.35, &sample);
            assert_eq!(
                calibration.threshold,
                f32::NEG_INFINITY,
                "no floor when nothing certifies",
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
        fn no_positives_certifies_nothing_even_with_failed_labels() {
            let sample = SampledScores {
                sampled: 64,
                unverified_scores: vec![0.9],
                ..Default::default()
            };
            let calibration = calibrate_threshold(0.9, Some(0.95), 0.35, &sample);
            assert_eq!(calibration.threshold, f32::NEG_INFINITY);
            assert!(!calibration.meets(0.9));
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
