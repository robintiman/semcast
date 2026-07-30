//! AST-level accommodations for Postgres clients that emit SQL DataFusion
//! doesn't quite accept.

use std::ops::ControlFlow;

use datafusion::sql::parser::Statement;
use datafusion::sql::sqlparser::ast::{Expr, Statement as SQLStatement, visit_expressions};

use crate::error::SemcastError;

/// Treat `CREATE TEMPORARY TABLE` as a plain `CREATE TABLE`.
///
/// DataFusion rejects the keyword outright, and dbt's incremental
/// materialization leans on it: it stages the new rows in `<model>__dbt_tmp`,
/// merges, then drops it. A regular table gets the same result — the staging
/// relation is created and dropped inside one materialization.
///
/// What's lost is the isolation: a temporary table is private to its session
/// and vanishes when the session ends. A semcast server shares one context
/// across connections, so the staging table is briefly visible to everyone,
/// and a crash mid-materialization leaves it behind.
pub fn demote_temporary_tables(statement: &mut Statement) {
    if let Statement::Statement(inner) = statement
        && let SQLStatement::CreateTable(create) = inner.as_mut()
    {
        create.temporary = false;
    }
}

/// Reject `DELETE`/`UPDATE` whose `WHERE` contains a subquery.
///
/// DataFusion 54 plans these with the predicate dropped: the statement hits
/// every row in the table and reports it as a success. `DELETE FROM t WHERE
/// id IN (SELECT id FROM staging)` empties `t`. dbt's `delete+insert`
/// incremental strategy emits exactly that shape, so an incremental model
/// would silently destroy its own table on the second run.
///
/// Refusing the statement is the conservative read: losing a table quietly is
/// far worse than an error the user can work around. Drop this once upstream
/// plans the predicate.
pub fn reject_dml_with_subquery(statement: &Statement) -> crate::Result<()> {
    let Statement::Statement(inner) = statement else {
        return Ok(());
    };
    let (kind, selection) = match inner.as_ref() {
        SQLStatement::Delete(delete) => ("DELETE", delete.selection.as_ref()),
        SQLStatement::Update(update) => ("UPDATE", update.selection.as_ref()),
        _ => return Ok(()),
    };
    let Some(selection) = selection else {
        return Ok(());
    };
    if !contains_subquery(selection) {
        return Ok(());
    }
    Err(SemcastError::DataFusion(
        datafusion::error::DataFusionError::NotImplemented(format!(
            "{kind} with a subquery in WHERE — DataFusion drops the predicate and would \
             affect every row. Materialize the subquery into a table and join, or use a \
             literal predicate."
        )),
    ))
}

fn contains_subquery(expr: &Expr) -> bool {
    visit_expressions(expr, |node| {
        if matches!(
            node,
            Expr::Subquery(_) | Expr::InSubquery { .. } | Expr::Exists { .. },
        ) {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    })
    .is_break()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sql::recall::parse_statement_with_recall;

    fn is_temporary(sql: &str) -> bool {
        let (statement, _) = parse_statement_with_recall(sql).unwrap();
        matches!(
            &statement,
            Statement::Statement(inner)
                if matches!(inner.as_ref(), SQLStatement::CreateTable(c) if c.temporary),
        )
    }

    #[test]
    fn temporary_is_dropped_from_create_table() {
        assert!(is_temporary("CREATE TEMPORARY TABLE t AS SELECT 1"));

        let (mut statement, _) =
            parse_statement_with_recall("CREATE TEMPORARY TABLE t AS SELECT 1").unwrap();
        demote_temporary_tables(&mut statement);
        assert!(matches!(
            &statement,
            Statement::Statement(inner)
                if matches!(inner.as_ref(), SQLStatement::CreateTable(c) if !c.temporary),
        ));
    }

    fn rejects(sql: &str) -> bool {
        let (statement, _) = parse_statement_with_recall(sql).unwrap();
        reject_dml_with_subquery(&statement).is_err()
    }

    #[test]
    fn subquery_predicates_are_refused_not_silently_widened() {
        assert!(rejects("DELETE FROM t WHERE id IN (SELECT id FROM s)"));
        assert!(rejects(
            "DELETE FROM t WHERE (id) IN (SELECT DISTINCT id FROM s)"
        ));
        assert!(rejects(
            "DELETE FROM t WHERE EXISTS (SELECT 1 FROM s WHERE s.id = t.id)"
        ));
        assert!(rejects("UPDATE t SET a = 1 WHERE id IN (SELECT id FROM s)"));
        assert!(rejects(
            "DELETE FROM t WHERE flag AND id IN (SELECT id FROM s)",
        ));
    }

    #[test]
    fn ordinary_dml_still_passes() {
        assert!(!rejects("DELETE FROM t WHERE id = 1"));
        assert!(!rejects("DELETE FROM t"));
        assert!(!rejects("UPDATE t SET a = 1 WHERE id > 3"));
        assert!(!rejects("SELECT id FROM t WHERE id IN (SELECT id FROM s)"));
        assert!(!rejects("INSERT INTO t SELECT * FROM s"));
    }

    #[tokio::test]
    async fn a_delete_that_would_empty_the_table_errors_instead() {
        use std::sync::Arc;

        use crate::model::MockModel;
        use crate::semcast_context;

        let ctx = semcast_context(Arc::new(MockModel::default()));
        for setup in [
            "CREATE TABLE t AS SELECT * FROM (VALUES (1),(2),(3)) AS v(id)",
            "CREATE TABLE s AS SELECT * FROM (VALUES (1)) AS v(id)",
        ] {
            crate::sql(&ctx, setup)
                .await
                .unwrap()
                .collect()
                .await
                .unwrap();
        }

        assert!(
            crate::sql(&ctx, "DELETE FROM t WHERE id IN (SELECT id FROM s)")
                .await
                .is_err(),
        );

        let rows = crate::sql(&ctx, "SELECT id FROM t")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(
            rows.iter().map(|b| b.num_rows()).sum::<usize>(),
            3,
            "the table survives the refused delete",
        );
    }

    #[tokio::test]
    async fn temporary_tables_are_queryable() {
        use std::sync::Arc;

        use crate::model::MockModel;
        use crate::semcast_context;

        let ctx = semcast_context(Arc::new(MockModel::default()));
        crate::sql(&ctx, "CREATE TEMPORARY TABLE staging AS SELECT 1 AS id")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();

        let rows = crate::sql(&ctx, "SELECT id FROM staging")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(rows.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
    }
}
