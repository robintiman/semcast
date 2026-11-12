//! The statement's recall budget: `WITH RECALL r` is a promise about the
//! rows the whole statement returns, not about each `MEANS` filter in it.
//!
//! A filter loses matches only through a lossy stage: an index floor, or a
//! scorer's floor. One verified on full text alone is exact. With `m` lossy
//! filters, each gets recall `1 - (1 - r)/m` and confidence `1 - (1 - c)/m`.
//! By the union bound all `m` certificates hold together with confidence `c`,
//! and the statement loses at most the sum of what each filter loses, `1 - r`.
//!
//! That last step assumes a filter misses the statement's matches (rows that
//! meet every condition) no more often than its own matches. A filter's floor
//! scores its own condition, so nothing ties its losses to the other
//! conditions, but nothing rules it out either: certifying the statement
//! directly needs every condition's label on one sample, which the joint
//! calibration (phase 4 step 4) does. The additive split needs no
//! independence between filters, where a product split (`r^(1/m)` each)
//! would.
//!
//! The split is applied last, over the optimized plan, because only then is
//! it settled which filters have an index or a scorer: counting too early
//! (through a view, say) could miss a lossy filter and promise more than the
//! plan delivers. Each filter keeps the statement's own target, so running
//! the rule again recomputes the same split.
//!
//! Every lossy filter in the plan is counted and given its share, including
//! one `WITH RECALL` never reached: a semantic column's `MEANS` is expanded
//! after the target is attached, and an index prunes it at a best-effort
//! floor unless the share gives it a calibrated one.
//!
//! Every target in a plan must be the statement's own: `semcast::sql`
//! refuses `WITH RECALL` on a view and any `means()` carrying a recall
//! argument, written by hand or stored in a view (embedding code's
//! `DataFrame::into_view` of a `WITH RECALL` query, say), since views are
//! inlined before the check. A statement through `ctx.sql` skips that check
//! and takes such a target as its own.

use std::sync::Arc;

use datafusion::common::Result;
use datafusion::common::tree_node::{Transformed, TreeNodeRecursion};
use datafusion::logical_expr::{Extension, LogicalPlan};
use datafusion::optimizer::optimizer::ApplyOrder;
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

use crate::index::registry::SemcastRuntime;
use crate::logical::SemFilterNode;

#[derive(Debug)]
pub struct RecallBudgetRule {
    runtime: Arc<SemcastRuntime>,
}

impl RecallBudgetRule {
    pub fn new(runtime: Arc<SemcastRuntime>) -> Self {
        Self { runtime }
    }

    /// Whether the physical planner will give this filter a lossy stage once
    /// it carries the statement's target: the same test it applies (an index
    /// on the text, or a scorer, which it uses for any filter with a target).
    /// An index prunes even without a target, so a covered filter the
    /// statement's `WITH RECALL` never reached (a semantic column, expanded
    /// after it; a subquery) is lossy too, and gets a share like the rest.
    fn lossy(&self, filter: &SemFilterNode) -> bool {
        self.runtime.covers(&filter.text, filter.input.schema()) || self.runtime.has_scorer()
    }

    /// Where the filter's pruning curve is kept: its statistics key.
    fn curve_key(&self, filter: &SemFilterNode) -> u128 {
        crate::physical::verify::stats_key(
            &filter.condition,
            &self.runtime.model.id(),
            &filter.text.to_string(),
        )
    }
}

/// A target as EXPLAIN prints it: two decimals, more when rounding would
/// misstate it (a share of 0.9967 must not read as 1.00).
pub fn fraction(value: f64) -> String {
    let mut decimals = 2;
    loop {
        let text = format!("{value:.decimals$}");
        let exact = (text.parse::<f64>().unwrap_or(value) - value).abs() < 1e-12;
        let misreads_as_one = value < 1.0 && text.parse::<f64>().is_ok_and(|v| v >= 1.0);
        if exact || (decimals >= 4 && !misreads_as_one) || decimals >= 12 {
            let trimmed = text.trim_end_matches('0');
            return match trimmed.split_once('.') {
                Some((_, d)) if d.len() >= 2 => trimmed.to_owned(),
                _ => format!("{value:.2}"),
            };
        }
        decimals += 1;
    }
}

/// This filter's share of a statement target split `shares` ways: an equal
/// part of the allowed loss, and of the allowed error.
pub fn share(recall: f64, confidence: Option<f64>, shares: usize) -> (f64, Option<f64>) {
    let n = shares as f64;
    (
        1.0 - (1.0 - recall) / n,
        confidence.map(|c| 1.0 - (1.0 - c) / n),
    )
}

fn as_filter(plan: &LogicalPlan) -> Option<&SemFilterNode> {
    match plan {
        LogicalPlan::Extension(Extension { node }) => node.as_any().downcast_ref::<SemFilterNode>(),
        _ => None,
    }
}

impl OptimizerRule for RecallBudgetRule {
    fn name(&self) -> &str {
        "semcast_recall_budget"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        // The whole plan at once: the split needs every filter counted first.
        None
    }

    fn supports_rewrite(&self) -> bool {
        true
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        // The statement's target: every filter that has one has the same.
        let mut target: Option<(f64, Option<f64>)> = None;
        let mut lossy = 0usize;
        // Each lossy filter's curve key, in visiting order.
        let mut keys: Vec<u128> = Vec::new();
        plan.apply_with_subqueries(|node| {
            if let Some(filter) = as_filter(node) {
                if let Some(recall) = filter.statement_recall {
                    target.get_or_insert((recall, filter.statement_confidence));
                }
                if self.lossy(filter) {
                    lossy += 1;
                    keys.push(self.curve_key(filter));
                }
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        // No WITH RECALL: every filter stays best-effort, as asked.
        let Some((statement_recall, statement_confidence)) = target else {
            return Ok(Transformed::no(plan));
        };
        if lossy == 0 {
            return Ok(Transformed::no(plan));
        }
        let (equal_recall, confidence) = share(statement_recall, statement_confidence, lossy);
        // With every filter's curve, the loss goes where it prunes most;
        // otherwise equal shares. Curves are point estimates, so a certified
        // target splits equally: a certified floor at a high level is far
        // stricter than its curve says, and recall 1 can't be certified.
        let curves: Option<Vec<crate::cache::PruneCurve>> =
            if lossy > 1 && statement_confidence.is_none() {
                keys.iter()
                    .map(|key| self.runtime.cache().curve(*key).filter(|c| !c.is_empty()))
                    .collect()
            } else {
                None
            };
        let recalls: Vec<f64> = match curves {
            Some(curves) => allocate(1.0 - statement_recall, &curves),
            None => vec![equal_recall; lossy],
        };
        let mut next = 0usize;
        plan.transform_down_with_subqueries(|node| {
            let Some(filter) = as_filter(&node) else {
                return Ok(Transformed::no(node));
            };
            if !self.lossy(filter) {
                return Ok(Transformed::no(node));
            }
            let recall = recalls.get(next).copied().unwrap_or(equal_recall);
            next += 1;
            if filter.recall == Some(recall)
                && filter.confidence == confidence
                && filter.shares == lossy
                && filter.statement_recall == Some(statement_recall)
            {
                return Ok(Transformed::no(node));
            }
            let mut filter = filter.clone();
            filter.statement_recall = Some(statement_recall);
            filter.statement_confidence = statement_confidence;
            filter.recall = Some(recall);
            filter.confidence = confidence;
            filter.shares = lossy;
            Ok(Transformed::yes(LogicalPlan::Extension(Extension {
                node: Arc::new(filter),
            })))
        })
    }
}

/// The recall levels a pruning curve is measured at.
pub const CURVE_LEVELS: [f64; 7] = [0.8, 0.85, 0.9, 0.95, 0.97, 0.99, 1.0];

/// A filter's pruning curve from one calibration: for each level, the share
/// of `population` (each row's score, `None` for one that passes at any
/// floor) kept by the floor that keeps that share of the sampled matches.
/// `scored` are the matches' scores, `always` the matches that pass at any
/// floor, `lost` those lost at any floor. A level the sample can't reach,
/// or reaches with the always-passing matches alone (no floor is set then,
/// as calibration sets none), keeps everything. `None` when there is nothing
/// to measure: no known match, or no rows.
pub fn curve_from(
    scored: &[f32],
    always: usize,
    lost: usize,
    population: &[Option<f32>],
) -> Option<crate::cache::PruneCurve> {
    let mut scored = scored.to_vec();
    scored.sort_by(|a, b| b.total_cmp(a));
    let matches = scored.len() + always + lost;
    if matches == 0 || population.is_empty() {
        return None;
    }
    let rows = population.len() as f64;
    let curve = CURVE_LEVELS
        .iter()
        .map(|&level| {
            let needed = (level * matches as f64).ceil() as usize;
            let kept = match needed.checked_sub(always) {
                Some(n) if n > 0 && n <= scored.len() => {
                    let floor = scored[n - 1];
                    population
                        .iter()
                        .filter(|score| score.is_none_or(|s| s >= floor))
                        .count() as f64
                        / rows
                }
                _ => 1.0,
            };
            (level, kept)
        })
        .collect();
    Some(monotone(curve))
}

/// A curve where asking for more recall never keeps fewer rows: each level
/// keeps at least what the level below it does.
pub fn monotone(mut curve: crate::cache::PruneCurve) -> crate::cache::PruneCurve {
    let mut floor = 0.0f64;
    for (_, kept) in &mut curve {
        *kept = kept.max(floor).clamp(0.0, 1.0);
        floor = *kept;
    }
    curve
}

/// Units the allowed loss is divided into when it is split by curves.
const LOSS_UNITS: usize = 20;

/// Split the allowed `loss` across filters to keep the fewest rows in all:
/// each filter's recall is `1 - (its units) * loss / LOSS_UNITS`, the units
/// adding up to at most `LOSS_UNITS`, so the losses add up to at most `loss`
/// and the union bound holds whatever the split. The curves come from
/// earlier calibrations, not this query's sample, so choosing by them costs
/// the certificate nothing.
pub fn allocate(loss: f64, curves: &[crate::cache::PruneCurve]) -> Vec<f64> {
    let recall = |units: usize| 1.0 - units as f64 * loss / LOSS_UNITS as f64;
    // best[u] = (rows kept, units per filter) over the filters so far,
    // using exactly u units.
    let mut best: Vec<Option<(f64, Vec<usize>)>> = vec![None; LOSS_UNITS + 1];
    best[0] = Some((0.0, Vec::new()));
    for curve in curves {
        let mut next: Vec<Option<(f64, Vec<usize>)>> = vec![None; LOSS_UNITS + 1];
        for (used, entry) in best.iter().enumerate() {
            let Some((kept, units)) = entry else { continue };
            for mine in 0..=LOSS_UNITS - used {
                let total = kept + kept_at(curve, recall(mine));
                let slot = &mut next[used + mine];
                if slot.as_ref().is_none_or(|(k, _)| total < *k - 1e-12) {
                    let mut units = units.clone();
                    units.push(mine);
                    *slot = Some((total, units));
                }
            }
        }
        best = next;
    }
    let (_, units) = best
        .into_iter()
        .flatten()
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .expect("zero units is always possible");
    units.into_iter().map(recall).collect()
}

/// The share of rows `curve` keeps at `recall`, interpolated between its
/// levels; above its last level, everything.
pub fn kept_at(curve: &[(f64, f64)], recall: f64) -> f64 {
    let mut previous: Option<(f64, f64)> = None;
    for &(level, kept) in curve {
        if recall <= level {
            return match previous {
                Some((lower, kept_lower)) if recall > lower => {
                    kept_lower + (kept - kept_lower) * (recall - lower) / (level - lower)
                }
                _ => kept,
            };
        }
        previous = Some((level, kept));
    }
    1.0
}

#[cfg(test)]
mod tests {
    use super::{allocate, curve_from, fraction, kept_at, share};

    #[test]
    fn the_loss_goes_where_it_prunes() {
        // Filter A keeps 90% of rows until recall 0.8, a flat curve: loss
        // there buys nothing. Filter B halves its rows for each step down.
        let a: Vec<(f64, f64)> = vec![(0.8, 0.9), (0.9, 0.9), (1.0, 0.9)];
        let b: Vec<(f64, f64)> = vec![(0.8, 0.1), (0.9, 0.3), (1.0, 1.0)];
        let recalls = allocate(0.1, &[a, b]);
        assert!((recalls[0] - 1.0).abs() < 1e-9, "{recalls:?}");
        assert!((recalls[1] - 0.9).abs() < 1e-9, "{recalls:?}");
        let loss: f64 = recalls.iter().map(|r| 1.0 - r).sum();
        assert!(loss <= 0.1 + 1e-9);
    }

    #[test]
    fn a_curve_keeps_what_each_level_needs() {
        // Ten matches scoring 0.1..1.0; population of 100 rows spread evenly.
        let scored: Vec<f32> = (1..=10).map(|i| i as f32 / 10.0).collect();
        let population: Vec<Option<f32>> = (0..100).map(|i| Some(i as f32 / 100.0)).collect();
        let curve = curve_from(&scored, 0, 0, &population).unwrap();
        // 0.8 needs 8 matches: floor 0.3, keeping 70 rows.
        assert_eq!(curve[0], (0.8, 0.7));
        // 1.0 needs all ten: floor 0.1, keeping 90.
        assert_eq!(curve.last().copied(), Some((1.0, 0.9)));
        // A lost match makes 1.0 unreachable: everything is kept.
        let lossy = curve_from(&scored, 0, 1, &population).unwrap();
        assert_eq!(lossy.last().copied(), Some((1.0, 1.0)));
        assert!((kept_at(&curve, 0.825) - 0.75).abs() < 1e-9);
        // Matches that pass any floor meeting a level alone set no floor.
        let unindexed = curve_from(&[0.9], 9, 0, &population).unwrap();
        assert_eq!(unindexed[0], (0.8, 1.0));
        assert!(curve_from(&[], 0, 0, &population).is_none());
        assert!(curve_from(&scored, 0, 0, &[]).is_none());
    }

    #[test]
    fn a_share_keeps_the_digits_it_needs() {
        assert_eq!(fraction(0.9), "0.90");
        assert_eq!(fraction(0.95), "0.95");
        assert_eq!(fraction(0.975), "0.975");
        assert_eq!(fraction(1.0 - 0.01 / 3.0), "0.9967");
        assert_eq!(fraction(1.0 - 0.01 / 200.0), "0.99995");
        assert_eq!(fraction(1.0), "1.00");
    }

    #[test]
    fn shares_add_back_to_the_statement_target() {
        let (r, c) = share(0.9, Some(0.95), 3);
        assert!((3.0 * (1.0 - r) - 0.1).abs() < 1e-12);
        assert!((3.0 * (1.0 - c.unwrap()) - 0.05).abs() < 1e-12);
        assert_eq!(share(0.9, None, 1), (0.9, None));
    }
}
