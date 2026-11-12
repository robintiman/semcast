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
//! "unknown", and nothing here decides a row. They are priced before stored
//! answers: which rows the store already holds is known only once the input
//! is read, so a repeat query can cost far less than its estimate. A `LIMIT`
//! that stops verify early is noted, not priced.
//!
//! On a joint index-and-scorer funnel the history is cost per row in
//! verify calls, scorer calls weighted in (what the joint choice minimizes),
//! so the estimate reads "verify-equivalent calls" there rather than a share
//! reaching verify.

use crate::cache::{AskStats, PruneCurve};
use crate::model::ModelId;

/// Characters of one chunk under the default chunking (512 tokens, about
/// four characters each): the input a chunked verify reads per chunk.
const CHUNK_CHARS: f64 = 2048.0;

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
    /// Verify reads this many chunks per row (an index without a full-text
    /// fallback); `None` when it reads the full text.
    pub chunks_per_row: Option<usize>,
    /// Characters of the synthesized prompt around each input.
    pub prompt_chars: usize,
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
    /// Session-model cost of labels and verify, at list price (the scorer
    /// is not priced).
    pub cost_usd: Option<f64>,
    /// The share is a joint funnel's cost per row, scorer weighted in.
    pub joint: bool,
    /// No recall target: the index prunes at its fixed floor, which no
    /// history prices.
    pub uncalibrated: bool,
    /// A `LIMIT` verify stops at, set when the plan pushes one down.
    pub fetch: Option<usize>,
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
    let joint = facts.index && facts.scorer && facts.recall.is_some();
    let scorer_calls = if facts.scorer && !facts.index && facts.recall.is_some() {
        rows
    } else {
        None
    };
    let cost_usd = (|| {
        let (input_price, output_price) = crate::model::pricing::price_per_mtok(&facts.model)?;
        // Full text, as the history measured it: what labels read, and what
        // verify reads unless it reads a few chunks.
        let full = facts.stats.and_then(|s| s.chars_per_ask())?;
        let read = match facts.chunks_per_row {
            Some(chunks) => full.min(chunks as f64 * CHUNK_CHARS),
            None => full,
        };
        let prompt = facts.prompt_chars as f64;
        let labels = labels.unwrap_or(0) as f64;
        // A joint funnel's count mixes in scorer work: only its labels are
        // the session model's alone.
        let verify = if joint { 0.0 } else { verify_calls? };
        let input_tokens = (verify * (read + prompt) + labels * (full + prompt)) / 4.0;
        let output_tokens = (verify + labels) * ANSWER_TOKENS;
        Some((input_tokens * input_price + output_tokens * output_price) / 1e6)
    })();
    FunnelEstimate {
        labels,
        labels_up_to: facts.certified,
        scorer_calls,
        reach_share,
        verify_calls,
        cost_usd,
        joint,
        uncalibrated: facts.index && facts.recall.is_none(),
        fetch: None,
    }
}

/// A share as a percentage, with a decimal where whole percents would round
/// it to 0 or 100.
fn percent(share: f64) -> String {
    let p = share * 100.0;
    // Decimals until a share that is neither nothing nor everything stops
    // reading as one.
    for decimals in 0..=3 {
        let text = format!("{p:.decimals$}");
        let shown: f64 = text.parse().unwrap_or(p);
        let misleads = (shown == 0.0 && p > 0.0) || (shown == 100.0 && p < 100.0);
        if !misleads {
            return format!("{text}%");
        }
    }
    if p < 50.0 {
        "<0.001%".to_owned()
    } else {
        ">99.999%".to_owned()
    }
}

/// Dollars to two significant figures, so a small cost isn't shown as zero.
fn dollars(cost: f64) -> String {
    if cost <= 0.0 {
        return "$0".to_owned();
    }
    if cost >= 10.0 {
        return format!("${cost:.0}");
    }
    let magnitude = cost.log10().floor() as i32;
    let decimals = (1 - magnitude).max(0) as usize;
    let text = format!("{cost:.decimals$}");
    // Rounding up to the next power of ten adds a digit: drop it.
    let rounded: f64 = text.parse().unwrap_or(cost);
    if rounded >= 10f64.powi(magnitude + 1) && decimals > 0 {
        format!("${rounded:.prec$}", prec = decimals - 1)
    } else {
        format!("${text}")
    }
}

impl std::fmt::Display for FunnelEstimate {
    /// `expect, before stored answers, ~200 verify calls (20% reach verify,
    /// from history) + 64 labels, session model ~$0.07 at list price`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "expect, before stored answers, ")?;
        let unit = if self.joint {
            "verify-equivalent calls"
        } else {
            "verify calls"
        };
        match self.verify_calls {
            Some(calls) => write!(f, "~{calls:.0} {unit}")?,
            None => write!(f, "{unit} unknown")?,
        }
        match self.reach_share {
            Some(share) if self.joint => write!(
                f,
                " ({} of rows, verify + scorer, from history)",
                percent(share)
            )?,
            Some(share) if share < 1.0 || self.labels.is_some() => {
                write!(f, " ({} reach verify, from history)", percent(share))?
            }
            None if self.uncalibrated => write!(f, " (uncalibrated floor: ceiling)")?,
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
        if let Some(fetch) = self.fetch {
            write!(f, ", fewer under LIMIT {fetch}")?;
        }
        match self.cost_usd {
            Some(cost) if self.joint => write!(
                f,
                ", labels ~{} at list price (verify and scorer not priced)",
                dollars(cost)
            ),
            Some(cost) => write!(f, ", session model ~{} at list price", dollars(cost)),
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
            chunks_per_row: None,
            prompt_chars: 250,
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
            shown.starts_with("expect, before stored answers, ~200 verify calls (20% reach verify"),
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
    fn chunked_verify_reads_less_than_the_full_text() {
        let full = derive_funnel(&FunnelFacts {
            stats: Some(AskStats {
                asked: 1,
                matched: 0,
                input_chars: 40_000,
            }),
            ..facts()
        });
        let chunked = derive_funnel(&FunnelFacts {
            chunks_per_row: Some(3),
            stats: Some(AskStats {
                asked: 1,
                matched: 0,
                input_chars: 40_000,
            }),
            ..facts()
        });
        assert!(chunked.cost_usd.unwrap() < full.cost_usd.unwrap() / 2.0);
    }

    #[test]
    fn a_joint_funnel_reads_as_cost_not_a_share() {
        let estimate = derive_funnel(&FunnelFacts {
            scorer: true,
            ..facts()
        });
        let shown = estimate.to_string();
        assert!(
            shown.contains("~200 verify-equivalent calls (20% of rows, verify + scorer"),
            "{shown}"
        );
        assert!(shown.contains("verify and scorer not priced"), "{shown}");
    }

    #[test]
    fn an_uncalibrated_index_says_so() {
        let shown = derive_funnel(&FunnelFacts {
            recall: None,
            ..facts()
        })
        .to_string();
        assert!(shown.contains("uncalibrated floor: ceiling"), "{shown}");
    }

    #[test]
    fn small_shares_and_costs_keep_their_digits() {
        assert_eq!(percent(0.996), "99.6%");
        assert_eq!(percent(0.004), "0.4%");
        assert_eq!(percent(0.2), "20%");
        assert_eq!(dollars(0.00001234), "$0.000012");
        assert_eq!(dollars(1.5), "$1.5");
        assert_eq!(dollars(0.0123), "$0.012");
        assert_eq!(dollars(0.0099999), "$0.010");
        assert_eq!(dollars(0.0), "$0");
        assert_eq!(percent(0.9996), "99.96%");
        assert_eq!(percent(0.0004), "0.04%");
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
