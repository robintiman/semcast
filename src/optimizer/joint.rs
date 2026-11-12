//! Joint thresholds for a two-stage funnel: an index floor, then a scorer
//! floor, then verify (phase 4 step 4, after CascadeFilter's design).
//!
//! Calibrating each stage alone splits the recall target between them before
//! anything is known about which stage prunes more cheaply. Choosing the
//! pair together on one labeled sample lets the loss go where it buys the
//! most: a column whose index separates matches well can take a high index
//! floor and skip the scorer; one whose index is weak can lean on the scorer.
//!
//! For every candidate pair `(a, b)` the sample says how many matches survive
//! (`index >= a`, `scorer >= b`, and verify's own answer, when it reads
//! chunks that can miss), and the population's index scores say how many
//! rows reach the scorer. The pair that certifies the target at the least
//! expected cost wins: scorer calls on every row above `a`, verify calls on
//! the share of those the sample shows passing `b`. A scorer floor of `-∞`
//! means no scorer stage at all, so the index alone is always a candidate,
//! and `(-∞, -∞)`, pruning nothing, is the fallback when no pair certifies.
//!
//! Failed labels get the same worst case as a single floor's calibration: a
//! failed document the pair would prune is a lost match, one it keeps is a
//! kept non-match. The pair is picked with the sample that scores it, so a
//! certified search tests each pair at `δ/m` over the `m` pairs it tries.

use crate::optimizer::calibrate::wilson_lower_bound;

/// How many floors per stage the search tries, at most: quantiles of the
/// sampled matches' scores. Bounds the Bonferroni correction (at most
/// `(GRID + 1)^2` pairs) so the search doesn't spend its confidence on
/// near-identical floors.
pub const GRID: usize = 8;

/// A scorer call's cost relative to a verify call's. The scorer is a small
/// model reading the same text, so it costs a fraction of the session model;
/// a fixed guess until the price table covers local models.
pub const SCORER_COST: f64 = 0.2;

/// One labeled document of the sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampledDoc {
    /// Index score; `None` for a document the index never saw, which passes
    /// the index stage at any floor.
    pub index: Option<f32>,
    /// Scorer probability; `None` when the scorer gave none, which passes
    /// the scorer stage at any floor.
    pub scorer: Option<f32>,
    /// The full-text label: `None` when the labeling call failed.
    pub label: Option<bool>,
    /// Verify would reject this document whatever the floors (chunked
    /// verify said no or failed).
    pub verify_miss: bool,
}

/// The chosen floors and what they are expected to cost.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JointChoice {
    pub index_floor: f32,
    /// `-∞` means no scorer stage: nothing is scored.
    pub scorer_floor: f32,
    pub estimated_recall: f64,
    pub certified_recall: Option<f64>,
    /// Rows the scorer reads and rows verify reads, expected.
    pub scorer_calls: f64,
    pub verify_calls: f64,
}

impl JointChoice {
    pub fn meets(&self, target_recall: f64) -> bool {
        self.certified_recall.unwrap_or(self.estimated_recall) >= target_recall
    }

    pub fn uses_scorer(&self) -> bool {
        self.scorer_floor > f32::NEG_INFINITY
    }
}

/// Worst-case `(kept, total)` matches of the sample at `(a, b)`.
fn recall_at(docs: &[SampledDoc], a: f32, b: f32) -> (usize, usize) {
    let passes = |doc: &SampledDoc| {
        doc.index.is_none_or(|s| s >= a) && doc.scorer.is_none_or(|p| p >= b) && !doc.verify_miss
    };
    let (mut kept, mut total) = (0, 0);
    for doc in docs {
        match doc.label {
            Some(true) => {
                total += 1;
                if passes(doc) {
                    kept += 1;
                }
            }
            // Unknown truth: pruned, it may be a lost match; kept, it is at
            // worst a kept non-match.
            None if !passes(doc) => total += 1,
            _ => {}
        }
    }
    (kept, total)
}

/// Up to `GRID` floors from `values`, highest first, plus `-∞`.
fn grid(mut values: Vec<f32>) -> Vec<f32> {
    values.retain(|v| v.is_finite());
    values.sort_by(|x, y| y.total_cmp(x));
    values.dedup();
    let mut floors: Vec<f32> = if values.len() <= GRID {
        values
    } else {
        (0..GRID)
            .map(|i| values[(i * (values.len() - 1)) / (GRID - 1)])
            .collect()
    };
    floors.dedup();
    floors.push(f32::NEG_INFINITY);
    floors
}

/// Choose `(index floor, scorer floor)`.
///
/// `population` holds the index score of every indexed row the funnel will
/// see (one per row), `unindexed` the rows it has never seen, which pass the
/// index at any floor. `scorer_cost` is a scorer call's cost relative to a
/// verify call's.
pub fn choose(
    target_recall: f64,
    confidence: Option<f64>,
    docs: &[SampledDoc],
    population: &[f32],
    unindexed: usize,
    scorer_cost: f64,
) -> JointChoice {
    // Floors that change what survives: the scores of matches and of
    // documents whose truth is unknown.
    let relevant = |doc: &&SampledDoc| doc.label != Some(false);
    let index_floors = grid(
        docs.iter()
            .filter(relevant)
            .filter_map(|d| d.index)
            .collect(),
    );
    let scorer_floors = grid(
        docs.iter()
            .filter(relevant)
            .filter_map(|d| d.scorer)
            .collect(),
    );
    let pairs = index_floors.len() * scorer_floors.len();
    let alpha = confidence.map(|c| (1.0 - c) / pairs as f64);

    let mut best: Option<JointChoice> = None;
    for &a in &index_floors {
        let reaching = population.iter().filter(|s| **s >= a).count() + unindexed;
        // The sample's documents that reach the scorer at this index floor:
        // the share of them passing `b` estimates verify's share.
        let reached: Vec<&SampledDoc> = docs
            .iter()
            .filter(|d| d.index.is_none_or(|s| s >= a))
            .collect();
        for &b in &scorer_floors {
            let (kept, total) = recall_at(docs, a, b);
            let estimated = if total == 0 {
                1.0
            } else {
                kept as f64 / total as f64
            };
            let certified = alpha.map(|alpha| {
                if total == 0 {
                    1.0
                } else {
                    wilson_lower_bound(kept, total, alpha)
                }
            });
            let choice = JointChoice {
                index_floor: a,
                scorer_floor: b,
                estimated_recall: estimated,
                certified_recall: certified,
                scorer_calls: 0.0,
                verify_calls: 0.0,
            };
            if !choice.meets(target_recall) {
                continue;
            }
            let passing = if b == f32::NEG_INFINITY || reached.is_empty() {
                1.0
            } else {
                reached
                    .iter()
                    .filter(|d| d.scorer.is_none_or(|p| p >= b))
                    .count() as f64
                    / reached.len() as f64
            };
            let scorer_calls = if b == f32::NEG_INFINITY {
                0.0
            } else {
                reaching as f64
            };
            let choice = JointChoice {
                scorer_calls,
                verify_calls: reaching as f64 * passing,
                ..choice
            };
            let cost = |c: &JointChoice| c.verify_calls + scorer_cost * c.scorer_calls;
            // Cheaper wins; on a tie, the stricter pair (it was found first,
            // floors run highest first) stays.
            if best
                .as_ref()
                .is_none_or(|current| cost(&choice) < cost(current))
            {
                best = Some(choice);
            }
        }
    }
    best.unwrap_or_else(|| {
        let (kept, total) = recall_at(docs, f32::NEG_INFINITY, f32::NEG_INFINITY);
        JointChoice {
            index_floor: f32::NEG_INFINITY,
            scorer_floor: f32::NEG_INFINITY,
            estimated_recall: if total == 0 {
                1.0
            } else {
                kept as f64 / total as f64
            },
            certified_recall: alpha.map(|alpha| {
                if total == 0 {
                    1.0
                } else {
                    wilson_lower_bound(kept, total, alpha)
                }
            }),
            scorer_calls: 0.0,
            verify_calls: (population.len() + unindexed) as f64,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(index: f32, scorer: f32, label: bool) -> SampledDoc {
        SampledDoc {
            index: Some(index),
            scorer: Some(scorer),
            label: Some(label),
            verify_miss: false,
        }
    }

    #[test]
    fn a_separating_index_needs_no_scorer() {
        // Matches score 0.9 on the index, non-matches 0.1: the index floor
        // alone keeps every match and prunes the rest, for no scorer calls.
        let mut docs: Vec<SampledDoc> = (0..40).map(|_| doc(0.9, 0.5, true)).collect();
        docs.extend((0..60).map(|_| doc(0.1, 0.5, false)));
        let population: Vec<f32> = docs.iter().map(|d| d.index.unwrap()).collect();
        let choice = choose(0.9, None, &docs, &population, 0, 0.2);
        assert_eq!(choice.index_floor, 0.9);
        assert!(!choice.uses_scorer());
        assert_eq!(choice.verify_calls, 40.0);
    }

    #[test]
    fn a_weak_index_leans_on_the_scorer() {
        // The index can't tell matches apart; the scorer can. Scoring all
        // 100 rows at 0.2 a call and verifying 40 beats verifying 100.
        let mut docs: Vec<SampledDoc> = (0..40).map(|_| doc(0.5, 0.9, true)).collect();
        docs.extend((0..60).map(|_| doc(0.5, 0.1, false)));
        let population = vec![0.5; 100];
        let choice = choose(0.9, None, &docs, &population, 0, 0.2);
        assert_eq!(choice.scorer_floor, 0.9);
        assert_eq!(choice.verify_calls, 40.0);
        assert_eq!(choice.scorer_calls, 100.0);
    }

    #[test]
    fn an_expensive_scorer_is_not_worth_it() {
        let mut docs: Vec<SampledDoc> = (0..40).map(|_| doc(0.5, 0.9, true)).collect();
        docs.extend((0..60).map(|_| doc(0.5, 0.1, false)));
        let population = vec![0.5; 100];
        // At 0.7 a call, scoring 100 and verifying 40 (110) loses to
        // verifying 100.
        let choice = choose(0.9, None, &docs, &population, 0, 0.7);
        assert!(!choice.uses_scorer());
    }

    #[test]
    fn verify_misses_and_failed_labels_count_against_a_pair() {
        // Every match is one chunked verify misses: nothing meets the
        // target, so nothing is pruned.
        let docs: Vec<SampledDoc> = (0..40)
            .map(|_| SampledDoc {
                verify_miss: true,
                ..doc(0.9, 0.9, true)
            })
            .collect();
        let choice = choose(0.9, None, &docs, &[0.9; 40], 0, 0.2);
        assert_eq!(choice.index_floor, f32::NEG_INFINITY);
        assert!(!choice.meets(0.9));

        // A failed label under the index floor is a possible lost match.
        let mut docs: Vec<SampledDoc> = (0..9).map(|_| doc(0.9, 0.9, true)).collect();
        docs.push(SampledDoc {
            label: None,
            ..doc(0.1, 0.9, true)
        });
        docs.extend((0..10).map(|_| doc(0.1, 0.1, false)));
        let choice = choose(0.95, None, &docs, &[0.5; 20], 0, 0.2);
        assert!(
            choice.index_floor <= 0.1,
            "kept the failed document: {choice:?}"
        );
    }

    #[test]
    fn a_certified_search_corrects_for_the_pairs_it_tries() {
        let mut docs: Vec<SampledDoc> = (0..80).map(|_| doc(0.9, 0.9, true)).collect();
        docs.extend((0..80).map(|_| doc(0.1, 0.1, false)));
        let population = vec![0.5; 160];
        let estimated = choose(0.9, None, &docs, &population, 0, 0.2);
        let certified = choose(0.9, Some(0.95), &docs, &population, 0, 0.2);
        assert!(estimated.meets(0.9));
        // 80 of 80 kept: the bound at δ/4 over 4 pairs still clears 0.9.
        assert!(certified.certified_recall.unwrap() >= 0.9, "{certified:?}");
        assert!(certified.index_floor > f32::NEG_INFINITY);
    }
}
