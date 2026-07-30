//! `ALTER TABLE <name> RENAME TO <name>` — DataFusion has no rename, but a
//! registered table is just a name bound to a `TableProvider`, so a rename is
//! a deregister/register pair.
//!
//! dbt needs it: the table materialization builds `<model>__dbt_tmp`, renames
//! the live relation to `<model>__dbt_backup`, renames the temp into place,
//! then drops the backup. Views rename through the same statement.

use datafusion::execution::context::SessionContext;
use datafusion::sql::TableReference;
use datafusion::sql::sqlparser::ast::{
    AlterTableOperation, Ident, ObjectName, ObjectNamePart, RenameTableNameKind, Statement,
};
use datafusion::sql::sqlparser::parser::Parser;

use crate::Result;
use crate::sql::SemcastDialect;

/// A parsed rename. Postgres keeps a renamed relation in its original schema
/// — the new name is a bare identifier — so `to` inherits `from`'s qualifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenameTable {
    pub from: TableReference,
    pub to: TableReference,
}

/// `Some` only for a lone `ALTER TABLE .. RENAME TO ..`; anything else (a
/// parse error, a multi-operation `ALTER`, `RENAME COLUMN`) is `None` and
/// falls through to DataFusion, which reports its own error.
pub fn parse_rename(query: &str) -> Option<RenameTable> {
    let statements = Parser::parse_sql(&SemcastDialect::default(), query).ok()?;
    let [Statement::AlterTable(alter)] = statements.as_slice() else {
        return None;
    };
    let [AlterTableOperation::RenameTable { table_name }] = alter.operations.as_slice() else {
        return None;
    };
    let (RenameTableNameKind::To(new) | RenameTableNameKind::As(new)) = table_name;

    let from = table_reference(&alter.name)?;
    let new_identifier = idents(new).last().map(normalize)?;
    let to = match &from {
        TableReference::Full {
            catalog, schema, ..
        } => TableReference::full(catalog.to_string(), schema.to_string(), new_identifier),
        TableReference::Partial { schema, .. } => {
            TableReference::partial(schema.to_string(), new_identifier)
        }
        TableReference::Bare { .. } => TableReference::bare(new_identifier),
    };
    Some(RenameTable { from, to })
}

/// Swap the provider from one name to the other. Errors if `from` is unknown
/// or `to` is taken, matching Postgres.
pub async fn apply_rename(ctx: &SessionContext, rename: &RenameTable) -> Result<()> {
    let provider = ctx.table_provider(rename.from.clone()).await?;
    ctx.deregister_table(rename.from.clone())?;
    ctx.register_table(rename.to.clone(), provider)?;
    Ok(())
}

fn table_reference(name: &ObjectName) -> Option<TableReference> {
    let parts: Vec<String> = idents(name).map(normalize).collect();
    match parts.as_slice() {
        [table] => Some(TableReference::bare(table.clone())),
        [schema, table] => Some(TableReference::partial(schema.clone(), table.clone())),
        [catalog, schema, table] => Some(TableReference::full(
            catalog.clone(),
            schema.clone(),
            table.clone(),
        )),
        _ => None,
    }
}

fn idents(name: &ObjectName) -> impl Iterator<Item = &Ident> {
    name.0.iter().filter_map(|part| match part {
        ObjectNamePart::Identifier(ident) => Some(ident),
        _ => None,
    })
}

/// Unquoted identifiers fold to lowercase, as everywhere else in DataFusion.
fn normalize(ident: &Ident) -> String {
    match ident.quote_style {
        Some(_) => ident.value.clone(),
        None => ident.value.to_ascii_lowercase(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rename_keeps_the_source_qualifier() {
        let rename = parse_rename(r#"ALTER TABLE "db"."analytics"."hello" RENAME TO "hello__bak""#)
            .expect("a rename");
        assert_eq!(
            rename.from,
            TableReference::full("db", "analytics", "hello")
        );
        assert_eq!(
            rename.to,
            TableReference::full("db", "analytics", "hello__bak"),
        );
    }

    #[test]
    fn unquoted_names_fold_to_lowercase() {
        let rename =
            parse_rename("ALTER TABLE Public.Hello RENAME TO Hello_Bak").expect("a rename");
        assert_eq!(rename.from, TableReference::partial("public", "hello"));
        assert_eq!(rename.to, TableReference::partial("public", "hello_bak"));
    }

    #[test]
    fn other_statements_fall_through() {
        assert_eq!(parse_rename("SELECT 1"), None);
        assert_eq!(parse_rename("ALTER TABLE t ADD COLUMN c INT"), None);
        assert_eq!(parse_rename("ALTER TABLE t RENAME COLUMN a TO b"), None);
        assert_eq!(parse_rename("not sql at all"), None);
    }

    #[tokio::test]
    async fn renaming_moves_the_rows_to_the_new_name() {
        use std::sync::Arc;

        use crate::model::MockModel;
        use crate::semcast_context;

        let ctx = semcast_context(Arc::new(MockModel::default()));
        crate::sql(&ctx, "CREATE TABLE t AS SELECT 1 AS id")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();

        crate::sql(&ctx, "ALTER TABLE t RENAME TO t2")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();

        let rows = crate::sql(&ctx, "SELECT id FROM t2")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(rows.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
        assert!(crate::sql(&ctx, "SELECT id FROM t").await.is_err());
    }
}
