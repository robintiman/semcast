//! Paying ahead for a view's semantic questions (redesign phase 6).
//!
//! ```sql
//! CREATE VIEW angry AS SELECT id FROM tickets WHERE body MEANS 'the customer is angry';
//! EXPLAIN REFRESH SEMANTIC VIEW angry;   -- what paying ahead would cost
//! REFRESH SEMANTIC VIEW angry;           -- pay it now
//! ```
//!
//! Any view works: a view is already saved in the catalog, and reading one
//! already asks only what the store hasn't answered. What a refresh adds is
//! *when* that is paid. `REFRESH SEMANTIC VIEW` reads the view to the end,
//! so every answer its next read needs is stored, and that read is free.
//! `EXPLAIN REFRESH SEMANTIC VIEW` quotes it first, asking nothing: one row
//! per semantic question the view asks, as `EXPLAIN REFRESH SEMANTIC COLUMN`
//! quotes one column.
//!
//! The quote is an upper bound. It counts every distinct text of the column
//! a question reads, over the whole table, as if no filter came before it;
//! the view's own read may need fewer (a `WHERE` before the `MEANS`, an
//! index pruning under `WITH RECALL`). A question over a computed
//! expression has no column to count, and its counts are NULL.
//!
//! Questions are asked one per call, as the view's own reads ask them, so
//! the answers a refresh stores are exactly the ones those reads look up.

use std::collections::HashSet;
use std::sync::Arc;

use datafusion::arrow::array::{ArrayRef, Float64Array, StringArray, UInt64Array};
use datafusion::arrow::compute::cast;
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::dataframe::DataFrame;
use datafusion::datasource::TableType;
use datafusion::error::DataFusionError;
use datafusion::execution::context::SessionContext;
use datafusion::logical_expr::LogicalPlan;

use crate::cache::{CacheKey, CachedValue};
use crate::index::registry::{column_behind_casts, qualified_name};
use crate::logical::sem_classify::{ClassifyOutput, SemClassifyNode};
use crate::logical::sem_filter::SemFilterNode;
use crate::model::ModelId;
use crate::physical::classify::{
    CLASSIFY_MAX_TOKENS, CLASSIFY_PROMPT_VERSION, synthesize_classify_prompt,
};
use crate::physical::verify::{
    MEANS_PROMPT_VERSION, VERIFY_MAX_TOKENS, means_cache_key, synthesize_means_prompt,
};

/// One semantic question a plan asks: its conditions over one text, and how
/// it is asked.
#[derive(Debug, Clone)]
pub(crate) struct Question {
    /// `MEANS` (a filter), `CASE`/label (one or several conditions, fused
    /// into one call when several), or `CONFIDENCE` (a probability).
    pub kind: QuestionKind,
    pub conditions: Vec<String>,
    /// The table and column the text comes from; `None` for a computed
    /// expression, which can't be quoted.
    pub source: Option<(String, String)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QuestionKind {
    Filter,
    Label,
    Confidence,
}

impl QuestionKind {
    fn name(self) -> &'static str {
        match self {
            QuestionKind::Filter => "MEANS",
            QuestionKind::Label => "label",
            QuestionKind::Confidence => "CONFIDENCE",
        }
    }
}

/// What answering a question over a column would cost: rows, distinct
/// texts, texts with every answer stored, texts to ask (one call each), and
/// the estimated tokens and list-price cost of asking them.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Quote {
    pub rows: u64,
    pub texts: u64,
    pub stored: u64,
    pub to_ask: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cost: Option<f64>,
}

/// Every semantic question the optimized `plan` asks.
pub(crate) fn questions(plan: &LogicalPlan) -> crate::Result<Vec<Question>> {
    let mut found = Vec::new();
    plan.apply(|node| {
        if let LogicalPlan::Extension(extension) = node {
            let any = extension.node.as_any();
            if let Some(filter) = any.downcast_ref::<SemFilterNode>() {
                found.push(Question {
                    kind: QuestionKind::Filter,
                    conditions: vec![filter.condition.clone()],
                    source: source(&filter.text, filter.input.schema()),
                });
            } else if let Some(classify) = any.downcast_ref::<SemClassifyNode>() {
                found.push(Question {
                    kind: match classify.output {
                        ClassifyOutput::Verdict => QuestionKind::Label,
                        ClassifyOutput::YesProbability => QuestionKind::Confidence,
                    },
                    conditions: classify.conditions.clone(),
                    source: source(&classify.text, classify.input.schema()),
                });
            }
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    Ok(found)
}

/// The table and column `text` reads, when it is a column of a table.
fn source(
    text: &datafusion::logical_expr::Expr,
    schema: &datafusion::common::DFSchema,
) -> Option<(String, String)> {
    let column = column_behind_casts(text)?;
    qualified_name(column, schema).ok()
}

/// Quote `question` over its source column, asking nothing; `None` when it
/// has no column, or the column can't be read as a table's.
pub(crate) async fn quote(
    ctx: &SessionContext,
    question: &Question,
    model: &ModelId,
) -> crate::Result<Option<Quote>> {
    let Some((table, column)) = &question.source else {
        return Ok(None);
    };
    // A qualifier the view gave a subquery names no table: nothing to count.
    let Ok(frame) = ctx.table(table.as_str()).await else {
        return Ok(None);
    };
    // The text column only: no semantic column is read, so nothing is asked.
    let batches = frame.select_columns(&[column.as_str()])?.collect().await?;
    let mut quote = Quote::default();
    let mut seen = HashSet::new();
    let mut texts: Vec<String> = Vec::new();
    for batch in &batches {
        quote.rows += batch.num_rows() as u64;
        let strings = cast(batch.column(0), &DataType::Utf8).map_err(DataFusionError::from)?;
        let strings = strings
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("cast to Utf8");
        for text in strings.iter().flatten() {
            if seen.insert(crate::cache::stable_digest(&[text.as_bytes()])) {
                texts.push(text.to_owned());
            }
        }
    }
    quote.texts = texts.len() as u64;

    // Per text and condition, the keys any of which answers it — the ones
    // the question's own read would accept.
    let fused = question.conditions.len() > 1;
    let keys_for = |condition: &str, text: &str| -> Vec<CacheKey> {
        let verify = means_cache_key(condition, text, model, MEANS_PROMPT_VERSION);
        match question.kind {
            QuestionKind::Confidence => vec![crate::semir::probability_key(&verify)],
            _ if fused => vec![
                means_cache_key(condition, text, model, CLASSIFY_PROMPT_VERSION),
                verify,
            ],
            _ => vec![verify],
        }
    };
    let per_condition = keys_for("", "").len();
    let keys: Vec<CacheKey> = texts
        .iter()
        .flat_map(|text| {
            question
                .conditions
                .iter()
                .flat_map(move |condition| keys_for(condition, text))
        })
        .collect();
    let cache = Arc::clone(crate::index::runtime(ctx)?.cache());
    let answers = tokio::task::spawn_blocking(move || cache.get_many(&keys))
        .await
        .map_err(|error| DataFusionError::External(Box::new(error)))?;

    let conditions: Vec<&str> = question.conditions.iter().map(String::as_str).collect();
    let (prompt_chars, output_cap) = if fused {
        (
            synthesize_classify_prompt(&conditions).chars().count() as u64,
            CLASSIFY_MAX_TOKENS as u64,
        )
    } else {
        (
            synthesize_means_prompt(&question.conditions[0])
                .chars()
                .count() as u64,
            VERIFY_MAX_TOKENS as u64,
        )
    };
    let per_text = question.conditions.len() * per_condition;
    for (text, answers) in texts.iter().zip(answers.chunks(per_text.max(1))) {
        let answered = answers.chunks(per_condition).all(|keys| {
            keys.iter()
                .any(|a| matches!(a, Some(CachedValue::Value(_))))
        });
        if answered {
            quote.stored += 1;
        } else {
            quote.to_ask += 1;
            quote.input_tokens += (prompt_chars + text.chars().count() as u64).div_ceil(4);
        }
    }
    quote.output_tokens = quote.to_ask * output_cap;
    quote.cost = crate::model::pricing::price_per_mtok(model).map(|(input, output)| {
        (quote.input_tokens as f64 * input + quote.output_tokens as f64 * output) / 1_000_000.0
    });
    Ok(Some(quote))
}

/// The optimized plan of the view `name`, refusing anything else and a view
/// that asks no semantic question.
async fn view_questions(
    ctx: &SessionContext,
    name: &str,
) -> crate::Result<(DataFrame, Vec<Question>)> {
    let provider = ctx.table_provider(name).await?;
    if provider.table_type() != TableType::View {
        return Err(DataFusionError::Plan(format!("{name} is not a view")).into());
    }
    let frame = ctx.table(name).await?;
    let plan = frame.clone().into_optimized_plan()?;
    let questions = questions(&plan)?;
    if questions.is_empty() {
        return Err(DataFusionError::Plan(format!(
            "the view {name} asks no semantic question; there is nothing to refresh"
        ))
        .into());
    }
    Ok((frame, questions))
}

/// `REFRESH SEMANTIC VIEW name`: read the view to the end, so every answer
/// its next read needs is asked now and stored. Returns the rows read.
pub async fn refresh_view(ctx: &SessionContext, name: &str) -> crate::Result<usize> {
    let (frame, _) = view_questions(ctx, name).await?;
    // The analyzer and optimizer rules apply to DataFrame plans too, so this
    // reads the view exactly as a query would.
    let batches = frame.collect().await?;
    Ok(batches.iter().map(|batch| batch.num_rows()).sum())
}

/// `EXPLAIN REFRESH SEMANTIC VIEW name`: one row per semantic question the
/// view asks, quoted over its column (see the module docs), asking nothing.
pub async fn explain_refresh(ctx: &SessionContext, name: &str) -> crate::Result<DataFrame> {
    let (_, questions) = view_questions(ctx, name).await?;
    let model = crate::index::runtime(ctx)?.model.id();
    let mut quotes = Vec::with_capacity(questions.len());
    for question in &questions {
        quotes.push(quote(ctx, question, &model).await?);
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("question", DataType::Utf8, false),
        Field::new("conditions", DataType::Utf8, false),
        Field::new("reads", DataType::Utf8, true),
        Field::new("model", DataType::Utf8, false),
        Field::new("rows", DataType::UInt64, true),
        Field::new("texts", DataType::UInt64, true),
        Field::new("stored", DataType::UInt64, true),
        Field::new("to_ask", DataType::UInt64, true),
        Field::new("est_input_tokens", DataType::UInt64, true),
        Field::new("est_output_tokens", DataType::UInt64, true),
        Field::new("est_cost_usd", DataType::Float64, true),
        Field::new("prices_as_of", DataType::Utf8, false),
    ]));
    let field = |f: fn(&Quote) -> u64| -> ArrayRef {
        Arc::new(UInt64Array::from(
            quotes.iter().map(|q| q.as_ref().map(f)).collect::<Vec<_>>(),
        ))
    };
    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(
            questions.iter().map(|q| q.kind.name()).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            questions
                .iter()
                .map(|q| q.conditions.join("; "))
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            questions
                .iter()
                .map(|q| q.source.as_ref().map(|(t, c)| format!("{t}.{c}")))
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(vec![model.0.clone(); questions.len()])),
        field(|q| q.rows),
        field(|q| q.texts),
        field(|q| q.stored),
        field(|q| q.to_ask),
        field(|q| q.input_tokens),
        field(|q| q.output_tokens),
        Arc::new(Float64Array::from(
            quotes
                .iter()
                .map(|q| q.as_ref().and_then(|q| q.cost))
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(vec![
            crate::model::pricing::PRICES_AS_OF;
            questions.len()
        ])),
    ];
    let batch = RecordBatch::try_new(schema, columns).map_err(DataFusionError::from)?;
    Ok(ctx.read_batch(batch)?)
}
