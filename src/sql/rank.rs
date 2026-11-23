//! Sort-direction defaulting for semantic ranking.
//!
//! `relevance()` returns a score where higher is better, so
//! `ORDER BY transcript RELEVANCE TO 'x'` has to mean *best first*. SQL's
//! default is ascending, and by the time a logical plan exists that default
//! has already been baked in — so the fix belongs on the AST, before
//! planning, next to [`rewrite_semantic_casts`].
//!
//! The rule applies to the desugared call too: a bare
//! `ORDER BY relevance(t, 'x')` is best-first however it was written. An
//! explicit `ASC` or `DESC` is always honored.
//!
//! A NULL score (NULL text, or a row the model failed to score) is no
//! score, so it sorts last unless the query says `NULLS FIRST` — SQL's
//! default under `DESC` would put it on top. That holds for an `ORDER BY`
//! on a select-list alias of `relevance()` too, whose direction is left as
//! written.
//!
//! [`rewrite_semantic_casts`]: crate::sql::typed::rewrite_semantic_casts

use std::ops::ControlFlow;

use datafusion::sql::parser::Statement;
use datafusion::sql::sqlparser::ast::{
    Expr, Function, FunctionArguments, Ident, ObjectName, OrderByKind, Query, SelectItem, SetExpr,
    VisitMut, VisitorMut,
};

use crate::sql::rank_udf::RELEVANCE_UDF_NAME;

/// Give every direction-less `ORDER BY relevance(..)` item a `DESC`, and
/// every one without a `NULLS` clause `NULLS LAST`.
///
/// Visits every `Query` in the statement, so derived tables, CTEs, and
/// scalar subqueries are covered on the same terms as the top level. Only the
/// plain-SQL statement variant carries a sqlparser AST; DataFusion's own
/// extensions (`CREATE EXTERNAL TABLE`, `COPY TO`, ...) are skipped.
pub fn default_relevance_desc(statement: &mut Statement) {
    let Statement::Statement(inner) = statement else {
        return;
    };
    let _: ControlFlow<()> = inner.as_mut().visit(&mut RelevanceDesc);
}

struct RelevanceDesc;

impl VisitorMut for RelevanceDesc {
    type Break = ();

    fn pre_visit_query(&mut self, query: &mut Query) -> ControlFlow<Self::Break> {
        let aliases = relevance_aliases(query);
        if let Some(order_by) = query.order_by.as_mut()
            && let OrderByKind::Expressions(items) = &mut order_by.kind
        {
            for item in items {
                let call = is_relevance_call(&item.expr);
                let alias = matches!(&item.expr, Expr::Identifier(ident)
                    if aliases.contains(&resolved(ident)));
                if call && item.options.asc.is_none() {
                    item.options.asc = Some(false);
                }
                if (call || alias) && item.options.nulls_first.is_none() {
                    item.options.nulls_first = Some(false);
                }
            }
        }
        ControlFlow::Continue(())
    }
}

/// The select-list aliases of `relevance()` calls in `query`'s own
/// `SELECT`: what an `ORDER BY score` can mean.
fn relevance_aliases(query: &Query) -> Vec<String> {
    let SetExpr::Select(select) = query.body.as_ref() else {
        return Vec::new();
    };
    select
        .projection
        .iter()
        .filter_map(|item| match item {
            SelectItem::ExprWithAlias { expr, alias } if is_relevance_call(expr) => {
                Some(resolved(alias))
            }
            _ => None,
        })
        .collect()
}

/// The name an identifier resolves to: as written when quoted, folded to
/// lower case when not, as DataFusion resolves it.
fn resolved(ident: &Ident) -> String {
    if ident.quote_style.is_some() {
        ident.value.clone()
    } else {
        ident.value.to_lowercase()
    }
}

fn is_relevance_call(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::Function(Function {
            name: ObjectName(parts),
            args: FunctionArguments::List(_),
            ..
        }) if parts.len() == 1
            && parts[0]
                .as_ident()
                .is_some_and(|ident| ident.value.eq_ignore_ascii_case(RELEVANCE_UDF_NAME))
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sql::statement::parse_statement_with_recall;

    fn rewrite(sql: &str) -> String {
        let (mut statement, _) = parse_statement_with_recall(sql).unwrap();
        default_relevance_desc(&mut statement);
        statement.to_string()
    }

    #[test]
    fn bare_relevance_order_by_becomes_desc() {
        assert_eq!(
            rewrite("SELECT id FROM t ORDER BY body RELEVANCE TO 'x' LIMIT 5"),
            "SELECT id FROM t ORDER BY relevance(body, 'x') DESC NULLS LAST LIMIT 5"
        );
    }

    #[test]
    fn handwritten_relevance_call_gets_the_same_default() {
        assert_eq!(
            rewrite("SELECT id FROM t ORDER BY relevance(body, 'x') LIMIT 5"),
            "SELECT id FROM t ORDER BY relevance(body, 'x') DESC NULLS LAST LIMIT 5"
        );
    }

    #[test]
    fn explicit_direction_is_honored() {
        assert_eq!(
            rewrite("SELECT id FROM t ORDER BY body RELEVANCE TO 'x' ASC LIMIT 5"),
            "SELECT id FROM t ORDER BY relevance(body, 'x') ASC NULLS LAST LIMIT 5"
        );
        assert_eq!(
            rewrite("SELECT id FROM t ORDER BY body RELEVANCE TO 'x' DESC LIMIT 5"),
            "SELECT id FROM t ORDER BY relevance(body, 'x') DESC NULLS LAST LIMIT 5"
        );
    }

    #[test]
    fn an_alias_of_relevance_sorts_nulls_last_in_its_own_direction() {
        assert_eq!(
            rewrite("SELECT relevance(body, 'x') AS score FROM t ORDER BY score DESC LIMIT 5"),
            "SELECT relevance(body, 'x') AS score FROM t ORDER BY score DESC NULLS LAST LIMIT 5"
        );
        assert_eq!(
            rewrite("SELECT relevance(body, 'x') AS score FROM t ORDER BY score LIMIT 5"),
            "SELECT relevance(body, 'x') AS score FROM t ORDER BY score NULLS LAST LIMIT 5"
        );
        assert_eq!(
            rewrite("SELECT body AS score FROM t ORDER BY score DESC LIMIT 5"),
            "SELECT body AS score FROM t ORDER BY score DESC LIMIT 5"
        );
        assert_eq!(
            rewrite("SELECT relevance(body, 'x') AS Score FROM t ORDER BY score DESC LIMIT 5"),
            "SELECT relevance(body, 'x') AS Score FROM t ORDER BY score DESC NULLS LAST LIMIT 5",
            "unquoted names fold alike"
        );
        assert_eq!(
            rewrite(
                "SELECT relevance(body, 'x') AS \"Score\", score FROM t ORDER BY score DESC LIMIT 5"
            ),
            "SELECT relevance(body, 'x') AS \"Score\", score FROM t ORDER BY score DESC LIMIT 5",
            "a quoted alias names only itself"
        );
    }

    #[test]
    fn an_explicit_nulls_clause_is_honored() {
        assert_eq!(
            rewrite("SELECT id FROM t ORDER BY body RELEVANCE TO 'x' DESC NULLS FIRST LIMIT 5"),
            "SELECT id FROM t ORDER BY relevance(body, 'x') DESC NULLS FIRST LIMIT 5"
        );
    }

    #[test]
    fn other_order_by_items_are_untouched() {
        assert_eq!(
            rewrite("SELECT id FROM t ORDER BY a, relevance(b, 'x'), c DESC LIMIT 5"),
            "SELECT id FROM t ORDER BY a, relevance(b, 'x') DESC NULLS LAST, c DESC LIMIT 5"
        );
    }

    #[test]
    fn derived_table_order_by_is_reached() {
        assert_eq!(
            rewrite("SELECT * FROM (SELECT id FROM t ORDER BY body RELEVANCE TO 'x' LIMIT 5) AS s"),
            "SELECT * FROM (SELECT id FROM t ORDER BY relevance(body, 'x') DESC NULLS LAST LIMIT 5) AS s"
        );
    }
}
