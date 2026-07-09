//! Rewrites the typed-extraction markers (`sem_extract`, `sem_extract_field`,
//! `sem_extract_inline`) in a `Projection` into [`SemExtractNode`]s — the
//! second extension operator (roadmap step 4).
//!
//! This *is* field pushdown: the rule builds each node from the fields the
//! projection actually references (plus their `TOGETHER` closure), so unused
//! fields never enter the plan. Markers are only legal in the SELECT list;
//! anywhere else is a plan-time error, with a subquery as the escape hatch.
//!
//! [`SemExtractNode`]: crate::logical::SemExtractNode

use std::collections::BTreeSet;
use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Result, ScalarValue, plan_err};
use datafusion::logical_expr::expr::ScalarFunction;
use datafusion::logical_expr::expr_rewriter::NamePreserver;
use datafusion::logical_expr::{Expr, Extension, LogicalPlan, Projection, lit};
use datafusion::optimizer::optimizer::ApplyOrder;
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};

use crate::logical::SemExtractNode;
use crate::sql::ddl::parse_field_type_str;
use crate::sql::extract_udf::{
    SEM_EXTRACT_FIELD_UDF_NAME, SEM_EXTRACT_INLINE_UDF_NAME, SEM_EXTRACT_UDF_NAME,
};
use crate::types::registry::TypeRegistry;
use crate::types::{FieldSpec, SemanticType, unit_hash};

/// Turns typed-extraction markers into `SemExtract` nodes below the projection.
#[derive(Debug)]
pub struct ExtractRewriteRule {
    registry: Arc<TypeRegistry>,
}

impl ExtractRewriteRule {
    pub fn new(registry: Arc<TypeRegistry>) -> Self {
        Self { registry }
    }
}

const NOT_IN_SELECT: &str = "typed extraction is only supported in the SELECT list; \
     wrap it in a subquery to filter or group on an extracted field";

impl OptimizerRule for ExtractRewriteRule {
    fn name(&self) -> &str {
        "semcast_extract_rewrite"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::BottomUp)
    }

    fn supports_rewrite(&self) -> bool {
        true
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        let LogicalPlan::Projection(projection) = plan else {
            // Markers anywhere but a projection are a plan-time error.
            for expr in plan.expressions() {
                if contains_marker(&expr)? {
                    return plan_err!("{NOT_IN_SELECT}");
                }
            }
            return Ok(Transformed::no(plan));
        };

        let mut markers = Vec::new();
        for expr in &projection.expr {
            collect_markers(expr, &mut markers)?;
        }
        if markers.is_empty() {
            return Ok(Transformed::no(LogicalPlan::Projection(projection)));
        }

        // Build one extraction node per (source, type) group, stacking them
        // above the projection's input.
        let mut builder = NodeBuilder::new(&self.registry);
        for marker in &markers {
            builder.observe(marker)?;
        }
        let (stacked_input, lookup) = builder.build((*projection.input).clone())?;

        // Replace every marker with a column (or a struct of columns) into the
        // node schema, preserving the projection's output names.
        let preserver = NamePreserver::new_for_projection();
        let new_exprs = projection
            .expr
            .into_iter()
            .map(|expr| {
                let saved = preserver.save(&expr);
                let rewritten = expr
                    .transform(|e| replace_marker(e, &lookup))
                    .map(|t| t.data)?;
                Ok(saved.restore(rewritten))
            })
            .collect::<Result<Vec<_>>>()?;

        let projection = Projection::try_new(new_exprs, Arc::new(stacked_input))?;
        Ok(Transformed::yes(LogicalPlan::Projection(projection)))
    }
}

/// A parsed marker call found in a projection expression.
enum Marker {
    /// `sem_extract(source, 'Type')` — the whole struct.
    Struct { source: Expr, type_name: String },
    /// `sem_extract_field(source, 'Type', 'field')`.
    Field {
        source: Expr,
        type_name: String,
        field: String,
    },
    /// `sem_extract_inline(source, 'field', 'spec', 'doc')`.
    Inline {
        source: Expr,
        field: String,
        spec: String,
        doc: String,
    },
}

fn is_marker(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::ScalarFunction(f) if matches!(
            f.func.name(),
            SEM_EXTRACT_UDF_NAME | SEM_EXTRACT_FIELD_UDF_NAME | SEM_EXTRACT_INLINE_UDF_NAME
        )
    )
}

fn contains_marker(expr: &Expr) -> Result<bool> {
    expr.exists(|e| Ok(is_marker(e)))
}

fn collect_markers(expr: &Expr, out: &mut Vec<Marker>) -> Result<()> {
    expr.apply(|e| {
        if let Some(marker) = parse_marker(e)? {
            out.push(marker);
        }
        Ok(TreeNodeRecursion::Continue)
    })
    .map(|_| ())
}

fn parse_marker(expr: &Expr) -> Result<Option<Marker>> {
    let Expr::ScalarFunction(ScalarFunction { func, args }) = expr else {
        return Ok(None);
    };
    let marker = match func.name() {
        SEM_EXTRACT_UDF_NAME => {
            let (source, type_name) = (arg(args, 0)?, literal(args, 1)?);
            Marker::Struct { source, type_name }
        }
        SEM_EXTRACT_FIELD_UDF_NAME => Marker::Field {
            source: arg(args, 0)?,
            type_name: literal(args, 1)?,
            field: literal(args, 2)?,
        },
        SEM_EXTRACT_INLINE_UDF_NAME => Marker::Inline {
            source: arg(args, 0)?,
            field: literal(args, 1)?,
            spec: literal(args, 2)?,
            doc: literal(args, 3)?,
        },
        _ => return Ok(None),
    };
    Ok(Some(marker))
}

fn arg(args: &[Expr], idx: usize) -> Result<Expr> {
    args.get(idx).cloned().ok_or_else(|| {
        datafusion::error::DataFusionError::Plan(format!("marker missing argument {idx}"))
    })
}

fn literal(args: &[Expr], idx: usize) -> Result<String> {
    match args.get(idx) {
        Some(Expr::Literal(ScalarValue::Utf8(Some(s)), _))
        | Some(Expr::Literal(ScalarValue::LargeUtf8(Some(s)), _))
        | Some(Expr::Literal(ScalarValue::Utf8View(Some(s)), _)) => Ok(s.clone()),
        _ => plan_err!("a typed-extraction marker argument must be a string literal"),
    }
}

/// One extraction node's identity and pruned spec, used to build the
/// replacement column references after the nodes are constructed.
struct BuiltNode {
    source: Expr,
    /// The registered type name (lowercased) for named groups, or the
    /// inline group's field name — the key markers are matched back against.
    key: NodeKey,
    id: usize,
    target: SemanticType,
}

#[derive(PartialEq)]
enum NodeKey {
    Named(String),
    Inline { field: String, spec: String },
}

/// Accumulates the distinct (source, type) groups a projection references,
/// then materializes them into stacked [`SemExtractNode`]s.
struct NodeBuilder<'a> {
    registry: &'a TypeRegistry,
    named: Vec<NamedGroup>,
    inline: Vec<InlineGroup>,
}

struct NamedGroup {
    source: Expr,
    type_name: String,
    full: Arc<SemanticType>,
    whole: bool,
    fields: BTreeSet<String>,
}

struct InlineGroup {
    source: Expr,
    field: String,
    spec: String,
    doc: String,
}

impl<'a> NodeBuilder<'a> {
    fn new(registry: &'a TypeRegistry) -> Self {
        Self {
            registry,
            named: Vec::new(),
            inline: Vec::new(),
        }
    }

    fn observe(&mut self, marker: &Marker) -> Result<()> {
        match marker {
            Marker::Struct { source, type_name } => {
                self.named_group(source, type_name)?.whole = true;
            }
            Marker::Field {
                source,
                type_name,
                field,
            } => {
                self.named_group(source, type_name)?
                    .fields
                    .insert(field.clone());
            }
            Marker::Inline {
                source,
                field,
                spec,
                doc,
            } => {
                if !self
                    .inline
                    .iter()
                    .any(|g| &g.source == source && &g.field == field && &g.spec == spec)
                {
                    self.inline.push(InlineGroup {
                        source: source.clone(),
                        field: field.clone(),
                        spec: spec.clone(),
                        doc: doc.clone(),
                    });
                }
            }
        }
        Ok(())
    }

    fn named_group(&mut self, source: &Expr, type_name: &str) -> Result<&mut NamedGroup> {
        let key = type_name.to_lowercase();
        if let Some(pos) = self
            .named
            .iter()
            .position(|g| &g.source == source && g.type_name == key)
        {
            return Ok(&mut self.named[pos]);
        }
        let full = self.registry.get(type_name).ok_or_else(|| {
            datafusion::error::DataFusionError::Plan(format!(
                "unknown semantic type {type_name}; define it with CREATE SEMANTIC TYPE first"
            ))
        })?;
        self.named.push(NamedGroup {
            source: source.clone(),
            type_name: key,
            full,
            whole: false,
            fields: BTreeSet::new(),
        });
        Ok(self.named.last_mut().expect("just pushed"))
    }

    /// Build the stacked node plan and the lookup used to rewrite markers.
    fn build(self, input: LogicalPlan) -> Result<(LogicalPlan, Vec<BuiltNode>)> {
        let mut plan = input;
        let mut lookup = Vec::new();
        let mut id = 0usize;

        for group in self.named {
            let target = if group.whole {
                (*group.full).clone()
            } else {
                let fields: Vec<&str> = group.fields.iter().map(String::as_str).collect();
                group.full.pruned(&fields)?
            };
            let node = SemExtractNode::try_new(plan, group.source.clone(), target.clone(), id)?;
            plan = LogicalPlan::Extension(Extension {
                node: Arc::new(node),
            });
            lookup.push(BuiltNode {
                source: group.source,
                key: NodeKey::Named(group.type_name),
                id,
                target,
            });
            id += 1;
        }

        for group in self.inline {
            let ty = parse_field_type_str(&group.spec)?;
            let spec = FieldSpec {
                name: group.field.clone(),
                ty,
                doc: group.doc,
            };
            let target = SemanticType {
                name: format!("__extract_{}", group.field),
                version: format!("{:x}", unit_hash(&[&spec])),
                fields: vec![spec],
                together: vec![],
            };
            let node = SemExtractNode::try_new(plan, group.source.clone(), target.clone(), id)?;
            plan = LogicalPlan::Extension(Extension {
                node: Arc::new(node),
            });
            lookup.push(BuiltNode {
                source: group.source,
                key: NodeKey::Inline {
                    field: group.field,
                    spec: group.spec,
                },
                id,
                target,
            });
            id += 1;
        }

        Ok((plan, lookup))
    }
}

fn replace_marker(expr: Expr, lookup: &[BuiltNode]) -> Result<Transformed<Expr>> {
    let Some(marker) = parse_marker(&expr)? else {
        return Ok(Transformed::no(expr));
    };
    let replacement = match marker {
        Marker::Struct { source, type_name } => {
            let node = named_node(lookup, &source, &type_name)?;
            // A struct of every column the node produced.
            let mut args = Vec::with_capacity(node.target.fields.len() * 2);
            for spec in &node.target.fields {
                args.push(lit(spec.name.clone()));
                args.push(Expr::Column(node_column(node, &spec.name)));
            }
            named_struct(args)
        }
        Marker::Field {
            source,
            type_name,
            field,
        } => {
            let node = named_node(lookup, &source, &type_name)?;
            Expr::Column(node_column(node, &field))
        }
        Marker::Inline {
            source,
            field,
            spec,
            ..
        } => {
            let node = lookup
                .iter()
                .find(|n| {
                    n.source == source
                        && n.key
                            == NodeKey::Inline {
                                field: field.clone(),
                                spec: spec.clone(),
                            }
                })
                .expect("every marker has a built node");
            Expr::Column(node_column(node, &field))
        }
    };
    Ok(Transformed::yes(replacement))
}

fn named_node<'a>(
    lookup: &'a [BuiltNode],
    source: &Expr,
    type_name: &str,
) -> Result<&'a BuiltNode> {
    let key = NodeKey::Named(type_name.to_lowercase());
    lookup
        .iter()
        .find(|n| &n.source == source && n.key == key)
        .ok_or_else(|| {
            datafusion::error::DataFusionError::Internal(
                "typed-extraction node missing for a collected marker".to_owned(),
            )
        })
}

fn node_column(node: &BuiltNode, field: &str) -> datafusion::common::Column {
    datafusion::common::Column::new_unqualified(crate::logical::sem_extract::output_column_name(
        node.id, field,
    ))
}

fn named_struct(args: Vec<Expr>) -> Expr {
    Expr::ScalarFunction(ScalarFunction::new_udf(
        datafusion::functions::core::named_struct(),
        args,
    ))
}
