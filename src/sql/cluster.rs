//! Binding the `AS <label>` of a `GROUP BY MEANING OF`.
//!
//! `GROUP BY MEANING OF transcript AS topic` names a column that does not
//! exist yet: `SELECT topic` has nothing to resolve against, because the
//! column only appears once the optimizer inserts a `SemCluster` node — long
//! after the SQL planner would have rejected the query.
//!
//! So the label is bound on the AST, before planning. Every `topic` in the
//! same `SELECT` becomes the `meaning_of(...)` call the `GROUP BY` holds, and
//! a bare `SELECT topic` keeps its name through an explicit alias. What
//! DataFusion then plans is an ordinary "group by an expression, select the
//! same expression" query.

use std::ops::ControlFlow;

use datafusion::sql::parser::Statement;
use datafusion::sql::sqlparser::ast::{
    Expr, Function, FunctionArg, FunctionArgExpr, FunctionArguments, GroupByExpr, Ident,
    ObjectName, Query, Select, SelectItem, Value, VisitMut, VisitorMut,
};

use crate::sql::cluster_udf::MEANING_OF_UDF_NAME;

/// Replace every reference to a `GROUP BY MEANING OF ... AS <label>` label
/// with the call it names.
pub fn bind_meaning_labels(statement: &mut Statement) {
    let Statement::Statement(inner) = statement else {
        return;
    };
    let _: ControlFlow<()> = inner.as_mut().visit(&mut BindLabels);
}

struct BindLabels;

impl VisitorMut for BindLabels {
    type Break = ();

    /// `pre_visit_query` reaches every `SELECT` in the statement, so derived
    /// tables and CTEs are bound on the same terms as the top level. Each
    /// binding is scoped to the `SELECT` that declared it.
    fn pre_visit_query(&mut self, query: &mut Query) -> ControlFlow<Self::Break> {
        if let datafusion::sql::sqlparser::ast::SetExpr::Select(select) = query.body.as_mut() {
            bind_select(select);
        }
        ControlFlow::Continue(())
    }
}

fn bind_select(select: &mut Select) {
    let GroupByExpr::Expressions(exprs, _) = &mut select.group_by else {
        return;
    };
    // Strip the label argument off each grouping call, keeping what it named.
    let mut labels: Vec<(String, Expr)> = Vec::new();
    for expr in exprs.iter_mut() {
        if let Some(label) = take_label(expr) {
            labels.push((label, expr.clone()));
        }
    }
    if labels.is_empty() {
        return;
    }

    for item in &mut select.projection {
        bind_select_item(item, &labels);
    }
    for expr in select
        .having
        .iter_mut()
        .chain(select.selection.iter_mut())
        .chain(select.qualify.iter_mut())
    {
        bind_expr(expr, &labels);
    }
}

/// A bare `SELECT topic` keeps the name the query gave it; anywhere deeper,
/// the call simply stands in for the identifier.
fn bind_select_item(item: &mut SelectItem, labels: &[(String, Expr)]) {
    if let SelectItem::UnnamedExpr(Expr::Identifier(ident)) = item
        && let Some((label, call)) = labels
            .iter()
            .find(|(label, _)| ident.value.eq_ignore_ascii_case(label))
    {
        *item = SelectItem::ExprWithAlias {
            expr: call.clone(),
            alias: Ident::new(label.clone()),
        };
        return;
    }
    match item {
        SelectItem::UnnamedExpr(expr) | SelectItem::ExprWithAlias { expr, .. } => {
            bind_expr(expr, labels);
        }
        _ => {}
    }
}

fn bind_expr(expr: &mut Expr, labels: &[(String, Expr)]) {
    let _: ControlFlow<()> = datafusion::sql::sqlparser::ast::visit_expressions_mut(expr, |expr| {
        if let Expr::Identifier(ident) = expr
            && let Some((_, call)) = labels
                .iter()
                .find(|(label, _)| ident.value.eq_ignore_ascii_case(label))
        {
            *expr = call.clone();
        }
        ControlFlow::Continue(())
    });
}

/// Pull the trailing label literal off a `meaning_of(source, k, 'label')`
/// call, leaving `meaning_of(source, k)` behind.
fn take_label(expr: &mut Expr) -> Option<String> {
    let Expr::Function(Function {
        name: ObjectName(parts),
        args: FunctionArguments::List(list),
        ..
    }) = expr
    else {
        return None;
    };
    if parts.len() != 1
        || !parts[0]
            .as_ident()
            .is_some_and(|i| i.value.eq_ignore_ascii_case(MEANING_OF_UDF_NAME))
    {
        return None;
    }
    if list.args.len() != 3 {
        return None;
    }
    let FunctionArg::Unnamed(FunctionArgExpr::Expr(Expr::Value(value))) = list.args.last()? else {
        return None;
    };
    let Value::SingleQuotedString(label) = &value.value else {
        return None;
    };
    let label = label.clone();
    list.args.pop();
    Some(label)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sql::statement::parse_statement_with_recall;

    fn rewrite(sql: &str) -> String {
        let (mut statement, _) = parse_statement_with_recall(sql).unwrap();
        bind_meaning_labels(&mut statement);
        statement.to_string()
    }

    #[test]
    fn parses_meaning_of_into_a_marker_call() {
        assert_eq!(
            rewrite("SELECT count(*) FROM m GROUP BY MEANING OF transcript"),
            "SELECT count(*) FROM m GROUP BY meaning_of(transcript)"
        );
    }

    #[test]
    fn into_becomes_the_second_argument() {
        assert_eq!(
            rewrite("SELECT count(*) FROM m GROUP BY MEANING OF transcript INTO 8"),
            "SELECT count(*) FROM m GROUP BY meaning_of(transcript, 8)"
        );
    }

    #[test]
    fn the_label_binds_the_select_list() {
        assert_eq!(
            rewrite("SELECT topic, count(*) FROM m GROUP BY MEANING OF transcript INTO 8 AS topic"),
            "SELECT meaning_of(transcript, 8) AS topic, count(*) FROM m \
             GROUP BY meaning_of(transcript, 8)"
        );
    }

    #[test]
    fn a_label_without_into_keeps_the_auto_k_sentinel() {
        assert_eq!(
            rewrite("SELECT topic, count(*) FROM m GROUP BY MEANING OF transcript AS topic"),
            "SELECT meaning_of(transcript, -1) AS topic, count(*) FROM m \
             GROUP BY meaning_of(transcript, -1)"
        );
    }

    #[test]
    fn the_label_binds_nested_references_too() {
        assert_eq!(
            rewrite(
                "SELECT upper(topic), count(*) FROM m GROUP BY MEANING OF t AS topic \
                 HAVING count(*) > 1"
            ),
            "SELECT upper(meaning_of(t, -1)), count(*) FROM m GROUP BY meaning_of(t, -1) \
             HAVING count(*) > 1"
        );
    }

    #[test]
    fn an_unrelated_identifier_is_untouched() {
        assert_eq!(
            rewrite("SELECT title, count(*) FROM m GROUP BY title, MEANING OF t AS topic"),
            "SELECT title, count(*) FROM m GROUP BY title, meaning_of(t, -1)"
        );
    }

    #[test]
    fn meaning_without_of_stays_an_identifier() {
        assert_eq!(
            rewrite("SELECT meaning FROM m GROUP BY meaning"),
            "SELECT meaning FROM m GROUP BY meaning"
        );
    }
}
