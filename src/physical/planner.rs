//! Maps semcast logical extension nodes to physical operators.
//!
//! DataFusion hook: a `QueryPlanner` that wraps the default planner with a
//! semcast `ExtensionPlanner`, installed by [`semcast_context`].
//!
//! [`semcast_context`]: crate::semcast_context

use std::sync::Arc;

use async_trait::async_trait;
use datafusion::common::DFSchema;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::context::QueryPlanner;
use datafusion::execution::session_state::SessionState;
use datafusion::logical_expr::{LogicalPlan, UserDefinedLogicalNode};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_planner::{DefaultPhysicalPlanner, ExtensionPlanner, PhysicalPlanner};

use crate::cache::SemanticCache;
use crate::index::SemanticIndex;
use crate::index::registry::{SemcastRuntime, column_behind_casts, qualified_name};
use crate::logical::{
    SemClassifyNode, SemClusterNode, SemDistinctNode, SemExtractNode, SemFilterNode, SemRankNode,
};
use crate::model::ModelProvider;
use crate::optimizer::calibrate::DEFAULT_CALIBRATION_SAMPLE;
use crate::physical::index_scan::{CalibrationConfig, ChunkEvidence, IndexScanExec};
use crate::physical::score_scan::ScorerHealth;
use crate::physical::store_lookup::{StoreEvidence, StoreLookupExec};
use crate::physical::{
    ScoreScanExec, SemClassifyExec, SemClusterExec, SemDistinctExec, SemExtractExec, SemRankExec,
    VerifyExec,
};

/// The default DataFusion planner plus semcast extension planning.
#[derive(Debug)]
pub struct SemcastQueryPlanner {
    model: Arc<dyn ModelProvider>,
    cache: Arc<dyn SemanticCache>,
    scorer: Option<(Arc<dyn ModelProvider>, Arc<ScorerHealth>)>,
}

impl SemcastQueryPlanner {
    pub fn new(model: Arc<dyn ModelProvider>, cache: Arc<dyn SemanticCache>) -> Self {
        Self {
            model,
            cache,
            scorer: None,
        }
    }

    /// A cheap model whose P("yes") pre-filters an unindexed `MEANS` under
    /// `WITH RECALL`; see [`ScoreScanExec`].
    pub fn with_scorer(mut self, scorer: Option<Arc<dyn ModelProvider>>) -> Self {
        self.scorer = scorer.map(|scorer| (scorer, Arc::new(ScorerHealth::default())));
        self
    }
}

#[async_trait]
impl QueryPlanner for SemcastQueryPlanner {
    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
        session_state: &SessionState,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let planner = DefaultPhysicalPlanner::with_extension_planners(vec![Arc::new(
            SemcastExtensionPlanner {
                model: Arc::clone(&self.model),
                cache: Arc::clone(&self.cache),
                scorer: self.scorer.clone(),
            },
        )]);
        let plan = planner
            .create_physical_plan(logical_plan, session_state)
            .await?;
        tracing::info!(
            target: "semcast::plan",
            "optimised plan:\n{}",
            datafusion::physical_plan::displayable(plan.as_ref()).indent(true)
        );
        Ok(plan)
    }
}

/// Plans `SemFilter` (and eventually `SemExtract`) extension nodes. Today
/// every `SemFilter` becomes a verify-only [`VerifyExec`] — correct first,
/// cheap later; funnel stages slot in with roadmap step 2.
struct SemcastExtensionPlanner {
    model: Arc<dyn ModelProvider>,
    cache: Arc<dyn SemanticCache>,
    scorer: Option<(Arc<dyn ModelProvider>, Arc<ScorerHealth>)>,
}

#[async_trait]
impl ExtensionPlanner for SemcastExtensionPlanner {
    async fn plan_extension(
        &self,
        planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        logical_inputs: &[&LogicalPlan],
        physical_inputs: &[Arc<dyn ExecutionPlan>],
        session_state: &SessionState,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        if let Some(filter) = node.as_any().downcast_ref::<SemFilterNode>() {
            let text = planner.create_physical_expr(
                &filter.text,
                logical_inputs[0].schema(),
                session_state,
            )?;
            // The store answers first: every row whose question it has
            // already answered is settled before any pre-filter or call, and
            // the later stages honour its "yes" (see `StoreLookupExec`).
            let store = Arc::new(StoreEvidence::default());
            // Where this condition's answers on this column are counted, by
            // the stages that see an unbiased view of it.
            let stats_key = crate::physical::verify::stats_key(
                &filter.condition,
                &self.model.id(),
                &filter.text.to_string(),
            );
            let looked_up: Arc<dyn ExecutionPlan> = Arc::new(StoreLookupExec::new(
                Arc::clone(&physical_inputs[0]),
                Arc::clone(&text),
                filter.condition.clone(),
                self.model.id(),
                Arc::clone(&self.cache),
                Arc::clone(&store),
            ));
            // Funnel when an index covers the filtered column: index scan
            // prunes, verify reads the surviving rows' top chunks. No index
            // (or an unresolvable text expr) → verify-only, correct but full
            // price.
            if let Some(index) = resolve_index(filter, logical_inputs[0].schema(), session_state) {
                let params = index.search_params();
                // WITH RECALL: the scan calibrates its own floor at execution
                // time by labeling a sample — same model and cache as verify.
                let calibration = filter.recall.map(|target_recall| CalibrationConfig {
                    target_recall,
                    confidence: filter.confidence,
                    sample_size: DEFAULT_CALIBRATION_SAMPLE,
                    model: Arc::clone(&self.model),
                    cache: Arc::clone(&self.cache),
                    stats_key: Some(stats_key),
                });
                let evidence = Arc::new(ChunkEvidence::default());
                let scan = Arc::new(
                    IndexScanExec::new(
                        looked_up,
                        Arc::clone(&text),
                        filter.condition.clone(),
                        index,
                        params,
                        calibration,
                        Arc::clone(&evidence),
                    )
                    .with_store_evidence(Arc::clone(&store)),
                );
                return Ok(Some(Arc::new(
                    VerifyExec::new_with_evidence(
                        scan,
                        text,
                        filter.condition.clone(),
                        Arc::clone(&self.model),
                        Arc::clone(&self.cache),
                        evidence,
                        params.chunks_per_doc,
                    )
                    .with_store_evidence(store),
                )));
            }
            // No index, but a scorer and a recall target: the scorer reads
            // every row, verify reads those above a calibrated floor. Without
            // a target there is nothing to calibrate against, and the scorer
            // stays out of the way.
            let mut input = looked_up;
            let mut prefiltered = false;
            // A scorer that is the session model would save nothing, and
            // under one id its answers are the session model's verdicts.
            let scorer = self
                .scorer
                .as_ref()
                .filter(|(scorer, _)| scorer.id() != self.model.id());
            if let (Some((scorer, health)), Some(target_recall)) = (scorer, filter.recall) {
                input = Arc::new(
                    ScoreScanExec::new(
                        input,
                        Arc::clone(&text),
                        filter.condition.clone(),
                        Arc::clone(scorer),
                        Arc::clone(health),
                        CalibrationConfig {
                            target_recall,
                            confidence: filter.confidence,
                            sample_size: DEFAULT_CALIBRATION_SAMPLE,
                            model: Arc::clone(&self.model),
                            cache: Arc::clone(&self.cache),
                            stats_key: Some(stats_key),
                        },
                    )
                    .with_store_evidence(Arc::clone(&store)),
                );
                prefiltered = true;
            }
            let mut verify = VerifyExec::new(
                input,
                text,
                filter.condition.clone(),
                Arc::clone(&self.model),
                Arc::clone(&self.cache),
            )
            .with_store_evidence(store);
            // Behind a scorer, verify sees mostly matches: its rate would
            // mislead the optimizer, and calibration's labels count instead.
            if !prefiltered {
                verify = verify.with_stats_key(stats_key);
            }
            return Ok(Some(Arc::new(verify)));
        }
        if let Some(distinct) = node.as_any().downcast_ref::<SemDistinctNode>() {
            let schema = logical_inputs[0].schema();
            let text = planner.create_physical_expr(&distinct.text, schema, session_state)?;
            // Dedupe is the index's vectors or nothing — and unlike the other
            // operators there is no model fallback, because no model is used.
            let column = column_behind_casts(&distinct.text).ok_or_else(|| {
                DataFusionError::Plan(format!(
                    "SEMANTIC DISTINCT ON requires a plain indexed column, not a \
                     computed expression (got `{}`)",
                    distinct.text
                ))
            })?;
            let (table, field) = qualified_name(column, schema)?;
            let index = index_for(&table, &field, session_state).ok_or_else(|| {
                DataFusionError::Plan(format!(
                    "SEMANTIC DISTINCT ON requires a semantic index on {table}({field}); \
                     create one with: CREATE SEMANTIC INDEX ON {table}({field});"
                ))
            })?;
            return Ok(Some(Arc::new(SemDistinctExec::new(
                Arc::clone(&physical_inputs[0]),
                text,
                distinct.similarity,
                crate::logical::sem_distinct::key_column_name(distinct.id),
                index,
            )?)));
        }
        if let Some(cluster) = node.as_any().downcast_ref::<SemClusterNode>() {
            let schema = logical_inputs[0].schema();
            let text = planner.create_physical_expr(&cluster.text, schema, session_state)?;
            // Clustering is the index's vectors or nothing: without them the
            // only alternative is embedding the corpus mid-query.
            let column = column_behind_casts(&cluster.text).ok_or_else(|| {
                DataFusionError::Plan(format!(
                    "MEANING OF requires a plain indexed column, not a computed \
                     expression (got `{}`)",
                    cluster.text
                ))
            })?;
            let (table, field) = qualified_name(column, schema)?;
            let index = index_for(&table, &field, session_state).ok_or_else(|| {
                DataFusionError::Plan(format!(
                    "MEANING OF requires a semantic index on {table}({field}); \
                     create one with: CREATE SEMANTIC INDEX ON {table}({field});"
                ))
            })?;
            return Ok(Some(Arc::new(SemClusterExec::new(
                Arc::clone(&physical_inputs[0]),
                text,
                cluster.k,
                crate::logical::sem_cluster::label_column_name(cluster.id),
                index,
                Arc::clone(&self.model),
                Arc::clone(&self.cache),
            )?)));
        }
        if let Some(classify) = node.as_any().downcast_ref::<SemClassifyNode>() {
            let schema = logical_inputs[0].schema();
            let text = planner.create_physical_expr(&classify.text, schema, session_state)?;
            let guard = classify
                .guard
                .as_ref()
                .map(|guard| planner.create_physical_expr(guard, schema, session_state))
                .transpose()?;
            let unguarded = guard.is_none();
            let mut exec = SemClassifyExec::new(
                Arc::clone(&physical_inputs[0]),
                text,
                classify.conditions.clone(),
                guard,
                classify.output,
                classify.column_names(),
                Arc::clone(&self.model),
                Arc::clone(&self.cache),
            )?;
            // Every row reaches an unguarded label, so its answers are the
            // condition's own rate on this column.
            if unguarded {
                let text = classify.text.to_string();
                exec = exec.with_stats_keys(
                    classify
                        .conditions
                        .iter()
                        .map(|c| crate::physical::verify::stats_key(c, &self.model.id(), &text))
                        .collect(),
                );
            }
            return Ok(Some(Arc::new(exec)));
        }
        if let Some(rank) = node.as_any().downcast_ref::<SemRankNode>() {
            let text = planner.create_physical_expr(
                &rank.text,
                logical_inputs[0].schema(),
                session_state,
            )?;
            // Ranking has no verify-only fallback: without an index there are
            // no candidates, so "rank" would mean a model call per row of the
            // whole table. Say so, and name the statement that fixes it.
            let column = column_behind_casts(&rank.text).ok_or_else(|| {
                DataFusionError::Plan(format!(
                    "RELEVANCE TO requires a plain indexed column, not a computed \
                     expression (got `{}`)",
                    rank.text
                ))
            })?;
            let (table, field) = qualified_name(column, logical_inputs[0].schema())?;
            let index = index_for(&table, &field, session_state).ok_or_else(|| {
                DataFusionError::Plan(format!(
                    "RELEVANCE TO requires a semantic index on {table}({field}); \
                     create one with: CREATE SEMANTIC INDEX ON {table}({field});"
                ))
            })?;
            let chunks_per_doc = index.search_params().chunks_per_doc;
            return Ok(Some(Arc::new(SemRankExec::new(
                Arc::clone(&physical_inputs[0]),
                text,
                rank.query.clone(),
                rank.candidates(),
                crate::logical::sem_rank::score_column_name(rank.id),
                index,
                chunks_per_doc,
                Arc::clone(&self.model),
                Arc::clone(&self.cache),
            )?)));
        }
        if let Some(extract) = node.as_any().downcast_ref::<SemExtractNode>() {
            let source = planner.create_physical_expr(
                &extract.source,
                logical_inputs[0].schema(),
                session_state,
            )?;
            return Ok(Some(Arc::new(SemExtractExec::new(
                Arc::clone(&physical_inputs[0]),
                source,
                extract.target.clone(),
                extract.id,
                Arc::clone(&self.model),
                Arc::clone(&self.cache),
            )?)));
        }
        Ok(None)
    }
}

/// The registered index for the filtered column, if the text expression
/// resolves to a plain qualified column. Resolution is deliberately exact:
/// computed expressions or alias-erased qualifiers plan verify-only rather
/// than borrow a possibly-wrong index.
fn resolve_index(
    filter: &SemFilterNode,
    schema: &DFSchema,
    session_state: &SessionState,
) -> Option<Arc<dyn SemanticIndex>> {
    session_state
        .config()
        .get_extension::<SemcastRuntime>()?
        .index_for_text(&filter.text, schema)
}

fn index_for(
    table: &str,
    column: &str,
    session_state: &SessionState,
) -> Option<Arc<dyn SemanticIndex>> {
    session_state
        .config()
        .get_extension::<SemcastRuntime>()?
        .index_for(table, column)
}
