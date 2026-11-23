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
    Cte, Expr, Function, FunctionArguments, Ident, ObjectName, OrderByKind, Query,
    RenameSelectItem, SelectItem, SelectItemQualifiedWildcardKind, SetExpr, TableAliasColumnDef,
    TableFactor, Visit, VisitMut, Visitor, VisitorMut, WildcardAdditionalOptions,
};

use crate::sql::rank_udf::RELEVANCE_UDF_NAME;

/// Give every direction-less `ORDER BY relevance(..)` item a `DESC`, and
/// every one without a `NULLS` clause `NULLS LAST`.
///
/// Visits every `Query` in the statement, so derived tables, CTEs, and
/// scalar subqueries are covered on the same terms as the top level. Only the
/// plain-SQL statement variant carries a sqlparser AST, and `EXPLAIN` wraps
/// one; DataFusion's other extensions (`CREATE EXTERNAL TABLE`, `COPY TO`,
/// ...) are skipped.
pub fn default_relevance_desc(statement: &mut Statement) {
    match statement {
        Statement::Statement(inner) => {
            let _: ControlFlow<()> = inner.as_mut().visit(&mut RelevanceDesc);
        }
        // `EXPLAIN` shows the plan of the statement it wraps, rewritten alike.
        Statement::Explain(explain) => default_relevance_desc(&mut explain.statement),
        _ => {}
    }
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
                let alias = aliases.names(&item.expr);
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

/// The names an `ORDER BY` in `query` can use for a relevance score: the
/// query's own select-list aliases of an expression over `relevance()`, and
/// those a derived table or CTE in its `FROM` exposes, plain or qualified by
/// the table's name. An own select item of the same name that isn't a score
/// shadows a table's.
#[derive(Default)]
struct RelevanceAliases {
    /// Own select-list aliases of a score.
    own: Vec<String>,
    /// Own select-list aliases of anything else.
    shadowed: Vec<String>,
    /// (table name, column) a `FROM` item exposes a score as.
    tables: Vec<(String, String)>,
}

impl RelevanceAliases {
    /// Whether `expr` names a relevance score.
    fn names(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Identifier(ident) => {
                let name = resolved(ident);
                self.own.contains(&name)
                    || (!self.shadowed.contains(&name)
                        && self.tables.iter().any(|(_, column)| *column == name))
            }
            Expr::CompoundIdentifier(parts) if parts.len() == 2 => {
                let (table, column) = (resolved(&parts[0]), resolved(&parts[1]));
                self.tables.iter().any(|t| t.0 == table && t.1 == column)
            }
            _ => false,
        }
    }
}

fn relevance_aliases(query: &Query) -> RelevanceAliases {
    let mut aliases = RelevanceAliases::default();
    let SetExpr::Select(select) = query.body.as_ref() else {
        return aliases;
    };
    let ctes: Vec<(String, &Cte)> = query
        .with
        .iter()
        .flat_map(|with| &with.cte_tables)
        .map(|cte| (resolved(&cte.alias.name), cte))
        .collect();
    let factors = select.from.iter().flat_map(|from| {
        std::iter::once(&from.relation).chain(from.joins.iter().map(|j| &j.relation))
    });
    for factor in factors {
        // The table's name, its query, and the column names it is renamed to.
        let (table, body, renamed) = match factor {
            TableFactor::Derived {
                subquery,
                alias: Some(alias),
                ..
            } => (resolved(&alias.name), subquery.as_ref(), &alias.columns),
            TableFactor::Table { name, alias, .. } => {
                let [part] = name.0.as_slice() else { continue };
                let Some(ident) = part.as_ident() else {
                    continue;
                };
                let Some((_, cte)) = ctes.iter().find(|(cte, _)| *cte == resolved(ident)) else {
                    continue;
                };
                match alias {
                    Some(alias) if !alias.columns.is_empty() => {
                        (resolved(&alias.name), cte.query.as_ref(), &alias.columns)
                    }
                    Some(alias) => (
                        resolved(&alias.name),
                        cte.query.as_ref(),
                        &cte.alias.columns,
                    ),
                    None => (resolved(ident), cte.query.as_ref(), &cte.alias.columns),
                }
            }
            _ => continue,
        };
        for column in exposed_scores(body, renamed) {
            aliases.tables.push((table.clone(), column));
        }
    }
    for item in &select.projection {
        match item {
            SelectItem::ExprWithAlias { expr, alias } if contains_relevance_call(expr) => {
                aliases.own.push(resolved(alias));
            }
            SelectItem::ExprWithAlias { alias, .. } => aliases.shadowed.push(resolved(alias)),
            // A plain column in the select list is an output of that name:
            // the table's score passed through, or something that hides it.
            SelectItem::UnnamedExpr(Expr::Identifier(ident)) => {
                let name = resolved(ident);
                if aliases.tables.iter().any(|t| t.1 == name) {
                    aliases.own.push(name);
                } else {
                    aliases.shadowed.push(name);
                }
            }
            SelectItem::UnnamedExpr(Expr::CompoundIdentifier(parts)) => {
                let [.., table, column] = parts.as_slice() else {
                    continue;
                };
                let (table, column) = (resolved(table), resolved(column));
                if aliases.tables.iter().any(|t| t.0 == table && t.1 == column) {
                    aliases.own.push(column);
                } else {
                    aliases.shadowed.push(column);
                }
            }
            _ => {}
        }
    }
    aliases
}

/// The names `body`'s scores are known by outside it. Renamed by a column
/// list, a score is found by its position, which a `*` hides; otherwise by
/// its alias, plus what the body's own `FROM` exposes through a `*` (all of
/// it) or an `x.*` (`x`'s).
fn exposed_scores(body: &Query, renamed: &[TableAliasColumnDef]) -> Vec<String> {
    let SetExpr::Select(select) = body.body.as_ref() else {
        return Vec::new();
    };
    if !renamed.is_empty() {
        let mut names = Vec::new();
        for (position, item) in select.projection.iter().enumerate() {
            let expr = match item {
                SelectItem::UnnamedExpr(expr) | SelectItem::ExprWithAlias { expr, .. } => expr,
                // Positions past a wildcard can't be counted.
                _ => return Vec::new(),
            };
            if contains_relevance_call(expr)
                && let Some(column) = renamed.get(position)
            {
                names.push(resolved(&column.name));
            }
        }
        return names;
    }
    let inner = relevance_aliases(body);
    let mut names = inner.own;
    for item in &select.projection {
        match item {
            SelectItem::Wildcard(options) => {
                let hidden = replaced(options);
                names.extend(
                    inner
                        .tables
                        .iter()
                        .map(|t| t.1.clone())
                        .filter(|name| !hidden.contains(name)),
                );
            }
            SelectItem::QualifiedWildcard(
                SelectItemQualifiedWildcardKind::ObjectName(name),
                options,
            ) => {
                let hidden = replaced(options);
                let Some(table) = name.0.last().and_then(|part| part.as_ident()) else {
                    continue;
                };
                let table = resolved(table);
                names.extend(
                    inner
                        .tables
                        .iter()
                        .filter(|t| t.0 == table && !hidden.contains(&t.1))
                        .map(|t| t.1.clone()),
                );
            }
            _ => {}
        }
    }
    names
}

/// The columns a `*` replaces or renames, which no longer carry what the
/// table exposed under that name.
fn replaced(options: &WildcardAdditionalOptions) -> Vec<String> {
    let mut names: Vec<String> = options
        .opt_replace
        .iter()
        .flat_map(|replace| &replace.items)
        .map(|item| resolved(&item.column_name))
        .collect();
    match &options.opt_rename {
        Some(RenameSelectItem::Single(rename)) => names.push(resolved(&rename.ident)),
        Some(RenameSelectItem::Multiple(renames)) => {
            names.extend(renames.iter().map(|rename| resolved(&rename.ident)));
        }
        None => {}
    }
    names
}

/// Whether `expr` reads a `relevance()` score, not counting inside a
/// subquery (whose score is aggregated or filtered away, not passed up).
fn contains_relevance_call(expr: &Expr) -> bool {
    struct Finder {
        depth: usize,
    }
    impl Visitor for Finder {
        type Break = ();
        fn pre_visit_query(&mut self, _query: &Query) -> ControlFlow<()> {
            self.depth += 1;
            ControlFlow::Continue(())
        }
        fn post_visit_query(&mut self, _query: &Query) -> ControlFlow<()> {
            self.depth -= 1;
            ControlFlow::Continue(())
        }
        fn pre_visit_expr(&mut self, expr: &Expr) -> ControlFlow<()> {
            if self.depth == 0 && is_relevance_call(expr) {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        }
    }
    expr.visit(&mut Finder { depth: 0 }).is_break()
}

/// The name an identifier resolves to: as written when quoted, folded to
/// ASCII lower case when not, as DataFusion resolves it.
fn resolved(ident: &Ident) -> String {
    if ident.quote_style.is_some() {
        ident.value.clone()
    } else {
        ident.value.to_ascii_lowercase()
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
    fn a_score_from_a_derived_table_or_cte_sorts_nulls_last() {
        let cases = [
            (
                "SELECT * FROM (SELECT id, relevance(b, 'x') AS s FROM t) AS q ORDER BY s DESC LIMIT 5",
                "SELECT * FROM (SELECT id, relevance(b, 'x') AS s FROM t) AS q ORDER BY s DESC NULLS LAST LIMIT 5",
            ),
            (
                "SELECT * FROM (SELECT id, relevance(b, 'x') AS s FROM t) AS q ORDER BY q.s DESC LIMIT 5",
                "SELECT * FROM (SELECT id, relevance(b, 'x') AS s FROM t) AS q ORDER BY q.s DESC NULLS LAST LIMIT 5",
            ),
            (
                "WITH r AS (SELECT id, ROUND(relevance(b, 'x'), 2) AS s FROM t) SELECT * FROM r ORDER BY s DESC LIMIT 5",
                "WITH r AS (SELECT id, ROUND(relevance(b, 'x'), 2) AS s FROM t) SELECT * FROM r ORDER BY s DESC NULLS LAST LIMIT 5",
            ),
            (
                "SELECT * FROM (SELECT * FROM (SELECT relevance(b, 'x') AS s FROM t) AS a) AS q ORDER BY s DESC LIMIT 5",
                "SELECT * FROM (SELECT * FROM (SELECT relevance(b, 'x') AS s FROM t) AS a) AS q ORDER BY s DESC NULLS LAST LIMIT 5",
            ),
        ];
        for (sql, expected) in cases {
            assert_eq!(rewrite(sql), expected);
        }
    }

    #[test]
    fn a_name_that_is_not_the_score_is_untouched() {
        // The outer select's own `s` shadows the table's score.
        let sql = "SELECT b AS s FROM (SELECT b, relevance(b, 'x') AS s FROM t) AS q ORDER BY s DESC LIMIT 5";
        assert_eq!(rewrite(sql), sql);
        // The middle query doesn't pass the inner score through.
        let sql = "SELECT * FROM (SELECT id FROM (SELECT id, relevance(b, 'x') AS s FROM t) AS a) AS q, u ORDER BY s DESC LIMIT 5";
        assert_eq!(rewrite(sql), sql);
    }

    #[test]
    fn a_column_alias_list_renames_by_position() {
        // `s` is now the id; the score is `k`.
        let sql = "SELECT * FROM (SELECT relevance(b, 'x') AS s, id AS k FROM t) AS q (k, s) ORDER BY s DESC LIMIT 5";
        assert_eq!(rewrite(sql), sql);
        assert_eq!(
            rewrite(
                "SELECT * FROM (SELECT id, relevance(b, 'x') FROM t) AS q (k, score) ORDER BY score DESC LIMIT 5"
            ),
            "SELECT * FROM (SELECT id, relevance(b, 'x') FROM t) AS q (k, score) ORDER BY score DESC NULLS LAST LIMIT 5"
        );
        let sql = "WITH r (k, s) AS (SELECT relevance(b, 'x'), id FROM t) SELECT * FROM r ORDER BY s DESC LIMIT 5";
        assert_eq!(rewrite(sql), sql);
    }

    #[test]
    fn a_selected_score_column_is_still_the_score() {
        for (sql, expected) in [
            (
                "SELECT id, s FROM (SELECT id, relevance(b, 'x') AS s FROM t) AS q ORDER BY s DESC LIMIT 5",
                "SELECT id, s FROM (SELECT id, relevance(b, 'x') AS s FROM t) AS q ORDER BY s DESC NULLS LAST LIMIT 5",
            ),
            (
                "SELECT q.id, q.s FROM (SELECT id, relevance(b, 'x') AS s FROM t) AS q ORDER BY s DESC LIMIT 5",
                "SELECT q.id, q.s FROM (SELECT id, relevance(b, 'x') AS s FROM t) AS q ORDER BY s DESC NULLS LAST LIMIT 5",
            ),
            (
                "WITH r AS (SELECT id, relevance(b, 'x') AS s FROM t) SELECT id, s FROM r ORDER BY s DESC LIMIT 5",
                "WITH r AS (SELECT id, relevance(b, 'x') AS s FROM t) SELECT id, s FROM r ORDER BY s DESC NULLS LAST LIMIT 5",
            ),
        ] {
            assert_eq!(rewrite(sql), expected);
        }
    }

    #[test]
    fn a_replaced_column_is_not_the_score() {
        let sql = "SELECT * FROM (SELECT * REPLACE (0 AS s) FROM (SELECT relevance(b, 'x') AS s FROM t) AS a) AS q ORDER BY s DESC LIMIT 5";
        assert_eq!(rewrite(sql), sql);
    }

    #[test]
    fn a_plain_select_column_shadows_a_tables_score() {
        let sql = "SELECT u.s FROM (SELECT id, relevance(b, 'x') AS s FROM t) AS q JOIN u ON q.id = u.id ORDER BY s DESC LIMIT 5";
        assert_eq!(rewrite(sql), sql);
    }

    #[test]
    fn a_qualified_star_passes_only_its_tables_scores() {
        let sql = "SELECT * FROM (SELECT a.* FROM (SELECT id AS s FROM t) AS a, (SELECT relevance(b, 'x') AS s FROM t) AS b) AS q ORDER BY s DESC LIMIT 5";
        assert_eq!(rewrite(sql), sql);
        assert_eq!(
            rewrite(
                "SELECT * FROM (SELECT b.* FROM (SELECT id FROM t) AS a, (SELECT relevance(b, 'x') AS s FROM t) AS b) AS q ORDER BY s DESC LIMIT 5"
            ),
            "SELECT * FROM (SELECT b.* FROM (SELECT id FROM t) AS a, (SELECT relevance(b, 'x') AS s FROM t) AS b) AS q ORDER BY s DESC NULLS LAST LIMIT 5"
        );
    }

    #[test]
    fn a_score_inside_a_scalar_subquery_is_not_one() {
        let sql = "SELECT (SELECT count(*) FROM (SELECT id FROM t ORDER BY relevance(b, 'x') DESC NULLS LAST LIMIT 1)) AS n FROM u ORDER BY n DESC LIMIT 5";
        assert_eq!(rewrite(sql), sql);
    }

    #[test]
    fn explain_is_rewritten_like_its_statement() {
        assert_eq!(
            rewrite("EXPLAIN SELECT id FROM t ORDER BY body RELEVANCE TO 'x' LIMIT 5"),
            "EXPLAIN SELECT id FROM t ORDER BY relevance(body, 'x') DESC NULLS LAST LIMIT 5"
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
