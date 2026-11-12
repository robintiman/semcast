//! What a `MEANS` filter's funnel is expected to cost, before it runs
//! (phase 4 step 4c).
//!
//! The physical planner knows which stages it built (an index floor, a
//! scorer, verify) and the plan's row estimate; the store knows what earlier
//! runs saw of this question on this column: its input size (`AskStats`)
//! and how much each recall level kept (`PruneCurve`). [`derive_funnel`]
//! combines them into expected calls per stage and a list-price cost, which
//! `EXPLAIN` prints beside the worst-case ceiling it already showed.
//!
//! These are estimates and say so: pruning with no history is priced at its
//! ceiling (every row reaches verify), a model with no listed price costs
//! "unknown", and nothing here decides a row.

use crate::cache::{AskStats, PruneCurve};
use crate::model::ModelId;

/// Characters of prompt every yes/no request carries besides its input.
const PROMPT_CHARS: f64 = 250.0;

/// Output tokens a yes/no answer is billed for (`VERIFY_MAX_TOKENS`).
const ANSWER_TOKENS: f64 = 8.0;

/// What the planner built, and what the store has seen.
#[derive(Debug, Clone)]
pub struct FunnelFacts {
    /// The plan's estimate of rows reaching the filter, if it has one.
    pub input_rows: Option<usize>,
    /// An index floor prunes before verify.
    pub index: bool,
    /// A scorer prunes before verify (alone, or after the index).
    pub scorer: bool,
    /// This filter's recall share, under `WITH RECALL`.
    pub recall: Option<f64>,
    /// Whether the recall is certified (`WITH CONFIDENCE`).
    pub certified: bool,
    /// Labels a calibration draws per tranche, and at most in all.
    pub sample: usize,
    pub label_budget: usize,
    /// The filter's history on this column.
    pub stats: Option<AskStats>,
    pub curve: Option<PruneCurve>,
    /// The session model, for its price.
    pub model: ModelId,
}

/// Expected cost of one filter's funnel.
#[derive(Debug, Clone, PartialEq)]
pub struct FunnelEstimate {
    /// Labels calibration buys: exactly this many when estimating, up to it
    /// when certifying.
    pub labels: Option<usize>,
    pub labels_up_to: bool,
    /// Rows the scorer reads, when a scorer runs alone (after an index, the
    /// rows reaching it are part of what the curve prices).
    pub scorer_calls: Option<f64>,
    /// Share of rows expected to reach verify, from the pruning curve; `None`
    /// when no history prices the pruning (every row is the ceiling).
    pub reach_share: Option<f64>,
    /// Verify calls expected, when the row count is known.
    pub verify_calls: Option<f64>,
    /// Session-model cost of labels and verify, at list price.
    pub cost_usd: Option<f64>,
}

/// Combine what the planner built with what the store has seen.
pub fn derive_funnel(facts: &FunnelFacts) -> FunnelEstimate {
    let rows = facts.input_rows.map(|rows| rows as f64);
    let lossy = facts.index || facts.scorer;
    let labels = facts.recall.filter(|_| lossy).map(|_| {
        let wanted = if facts.certified {
            facts.label_budget
        } else {
            facts.sample
        };
        facts.input_rows.map_or(wanted, |rows| wanted.min(rows))
    });
    // Without a recall target an index prunes at a fixed floor no curve
    // describes, and no scorer is built.
    let reach_share = match (lossy, facts.recall, &facts.curve) {
        (false, _, _) => Some(1.0),
        (true, Some(recall), Some(curve)) => Some(crate::optimizer::budget::kept_at(curve, recall)),
        _ => None,
    };
    let verify_calls = rows.map(|rows| rows * reach_share.unwrap_or(1.0));
    let scorer_calls = if facts.scorer && !facts.index && facts.recall.is_some() {
        rows
    } else {
        None
    };
    let cost_usd = (|| {
        let (input_price, output_price) = crate::model::pricing::price_per_mtok(&facts.model)?;
        let chars = facts.stats.and_then(|s| s.chars_per_ask())?;
        let calls = verify_calls? + labels.unwrap_or(0) as f64;
        let input_tokens = calls * (chars + PROMPT_CHARS) / 4.0;
        let output_tokens = calls * ANSWER_TOKENS;
        Some((input_tokens * input_price + output_tokens * output_price) / 1e6)
    })();
    FunnelEstimate {
        labels,
        labels_up_to: facts.certified,
        scorer_calls,
        reach_share,
        verify_calls,
        cost_usd,
    }
}

impl std::fmt::Display for FunnelEstimate {
    /// `expect ~200 verify calls (20% reach verify, from history) + 64
    /// labels, ~$0.0700 at list price`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "expect ")?;
        match self.verify_calls {
            Some(calls) => write!(f, "~{calls:.0} verify calls")?,
            None => write!(f, "verify calls unknown")?,
        }
        match self.reach_share {
            Some(share) if share < 1.0 => {
                write!(f, " ({:.0}% reach verify, from history)", share * 100.0)?
            }
            None => write!(f, " (no pruning history: ceiling)")?,
            _ => {}
        }
        if let Some(scorer) = self.scorer_calls {
            write!(f, " + ~{scorer:.0} scorer calls")?;
        }
        if let Some(labels) = self.labels {
            let up_to = if self.labels_up_to { "≤" } else { "" };
            write!(f, " + {up_to}{labels} labels")?;
        }
        match self.cost_usd {
            Some(cost) => write!(f, ", ~${cost:.4} at list price"),
            None => write!(f, ", cost unknown"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts() -> FunnelFacts {
        FunnelFacts {
            input_rows: Some(1000),
            index: true,
            scorer: false,
            recall: Some(0.9),
            certified: false,
            sample: 64,
            label_budget: 256,
            stats: Some(AskStats {
                asked: 100,
                matched: 10,
                input_chars: 100 * 750,
            }),
            curve: Some(vec![(0.8, 0.1), (0.9, 0.2), (1.0, 0.6)]),
            model: ModelId("anthropic/claude-haiku-4-5".to_owned()),
        }
    }

    #[test]
    fn history_prices_the_pruning() {
        let estimate = derive_funnel(&facts());
        assert_eq!(estimate.reach_share, Some(0.2));
        assert_eq!(estimate.verify_calls, Some(200.0));
        assert_eq!(estimate.labels, Some(64));
        // 264 calls of (750 + 250) / 4 = 250 input tokens and 8 output, at
        // $1 / $5 per million.
        let expected = (264.0 * 250.0 * 1.0 + 264.0 * 8.0 * 5.0) / 1e6;
        assert!(
            (estimate.cost_usd.unwrap() - expected).abs() < 1e-12,
            "{estimate:?}"
        );
        let shown = estimate.to_string();
        assert!(
            shown.starts_with("expect ~200 verify calls (20% reach verify"),
            "{shown}"
        );
    }

    #[test]
    fn no_history_prices_the_ceiling() {
        let estimate = derive_funnel(&FunnelFacts {
            curve: None,
            stats: None,
            ..facts()
        });
        assert_eq!(estimate.reach_share, None);
        assert_eq!(estimate.verify_calls, Some(1000.0));
        assert_eq!(estimate.cost_usd, None);
        assert!(estimate.to_string().contains("no pruning history: ceiling"));
    }

    #[test]
    fn verify_alone_reads_every_row_and_buys_no_labels() {
        let estimate = derive_funnel(&FunnelFacts {
            index: false,
            recall: None,
            ..facts()
        });
        assert_eq!(estimate.reach_share, Some(1.0));
        assert_eq!(estimate.labels, None);
        assert_eq!(estimate.verify_calls, Some(1000.0));
    }

    #[test]
    fn a_certified_calibration_buys_up_to_its_budget() {
        let estimate = derive_funnel(&FunnelFacts {
            certified: true,
            input_rows: Some(100),
            ..facts()
        });
        assert_eq!(estimate.labels, Some(100));
        assert!(estimate.to_string().contains("≤100 labels"));
    }
}
