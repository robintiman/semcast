//! A `pg_catalog` shim wide enough for dbt.
//!
//! dbt-postgres introspects through `pg_namespace`, `pg_tables`, `pg_views`,
//! `pg_matviews` and `pg_proc` — unqualified, so they resolve in the default
//! schema. Each is a view over `information_schema`, which DataFusion keeps
//! live, so `CREATE TABLE` shows up in the next `pg_tables` scan.
//!
//! The shim's own views are filtered out of `pg_views`: they live in the
//! default schema, which is usually a dbt target schema, and dbt would
//! otherwise cache them as models it owns.

use datafusion::execution::context::SessionContext;

use crate::Result;

/// Views the shim installs — and the names it hides from `pg_views`.
pub const SHIM_VIEWS: [&str; 5] = [
    "pg_namespace",
    "pg_tables",
    "pg_views",
    "pg_matviews",
    "pg_proc",
];

/// Column list dbt's `postgres__get_relations` expects back. The shim answers
/// that query with zero rows — see [`is_relation_dependency_query`].
pub const RELATION_DEPENDENCY_COLUMNS: [&str; 4] = [
    "dependent_schema",
    "dependent_name",
    "referenced_schema",
    "referenced_name",
];

/// Install the shim into `ctx`. Idempotent — every view is `OR REPLACE`.
pub async fn install(ctx: &SessionContext) -> Result<()> {
    let hidden = SHIM_VIEWS.map(|name| format!("'{name}'")).join(", ");
    let statements = [
        // `oid` exists only to satisfy the `pg_proc` join below; nothing reads
        // it, and `pg_proc` is always empty.
        "CREATE OR REPLACE VIEW pg_namespace AS \
         SELECT schema_name AS nspname, 0 AS oid FROM information_schema.schemata"
            .to_owned(),
        "CREATE OR REPLACE VIEW pg_tables AS \
         SELECT table_schema AS schemaname, table_name AS tablename, '' AS tableowner \
         FROM information_schema.tables \
         WHERE table_type = 'BASE TABLE' AND table_schema <> 'information_schema'"
            .to_owned(),
        // `information_schema.views` lists base tables too, with a null
        // definition — only a real view has one.
        format!(
            "CREATE OR REPLACE VIEW pg_views AS \
             SELECT table_schema AS schemaname, table_name AS viewname, '' AS viewowner, \
                    definition \
             FROM information_schema.views \
             WHERE definition IS NOT NULL \
               AND table_schema <> 'information_schema' \
               AND table_name NOT IN ({hidden})"
        ),
        // No materialized views and no user functions, but dbt unions both in,
        // so they have to exist with the right shape.
        "CREATE OR REPLACE VIEW pg_matviews AS \
         SELECT table_schema AS schemaname, table_name AS matviewname, '' AS matviewowner \
         FROM information_schema.tables WHERE false"
            .to_owned(),
        "CREATE OR REPLACE VIEW pg_proc AS \
         SELECT table_name AS proname, 0 AS pronamespace \
         FROM information_schema.tables WHERE false"
            .to_owned(),
    ];
    for statement in statements {
        ctx.sql(&statement).await?.collect().await?;
    }
    Ok(())
}

/// Does this statement look like dbt's `postgres__get_relations` — the
/// `pg_depend`/`pg_rewrite` walk that links a cached view to the relations it
/// reads?
///
/// It casts to `regclass`, which DataFusion has no type for, so the shim
/// answers it directly with zero rows. Cost of lying: dbt's relation cache
/// doesn't know view→table edges, so a `DROP ... CASCADE` on a table leaves
/// dependent views in the cache. Rebuilding a dropped view still works — dbt
/// re-reads `pg_views` per run — but a cascade within one run can go stale.
pub fn is_relation_dependency_query(statement: &str) -> bool {
    let lower = statement.to_ascii_lowercase();
    lower.contains("pg_depend") && lower.contains("pg_rewrite")
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::model::MockModel;

    async fn shimmed() -> SessionContext {
        let ctx = crate::SemcastContextBuilder::new(Arc::new(MockModel::default()))
            .with_information_schema(true)
            .build();
        install(&ctx).await.unwrap();
        ctx
    }

    async fn rows(ctx: &SessionContext, sql: &str) -> Vec<String> {
        let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
        let mut out = Vec::new();
        for batch in &batches {
            let column = batch.column(0);
            let formatter = datafusion::arrow::util::display::ArrayFormatter::try_new(
                column.as_ref(),
                &datafusion::arrow::util::display::FormatOptions::default(),
            )
            .unwrap();
            for row in 0..batch.num_rows() {
                out.push(formatter.value(row).to_string());
            }
        }
        out.sort();
        out
    }

    #[tokio::test]
    async fn pg_tables_tracks_tables_as_they_are_created() {
        let ctx = shimmed().await;
        assert!(
            rows(&ctx, "SELECT tablename FROM pg_tables")
                .await
                .is_empty()
        );

        ctx.sql("CREATE TABLE hello AS SELECT 1 AS id")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(
            rows(&ctx, "SELECT tablename FROM pg_tables").await,
            vec!["hello"],
        );
    }

    #[tokio::test]
    async fn pg_views_holds_views_only_and_hides_the_shim_itself() {
        let ctx = shimmed().await;
        assert!(rows(&ctx, "SELECT viewname FROM pg_views").await.is_empty());

        ctx.sql("CREATE TABLE t AS SELECT 1 AS id")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        ctx.sql("CREATE VIEW v AS SELECT 1 AS id")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();

        // A table must not show up as a view: dbt would then hold two cache
        // entries for it and reach for `DROP VIEW` on a table.
        assert_eq!(rows(&ctx, "SELECT viewname FROM pg_views").await, vec!["v"]);
        assert_eq!(
            rows(&ctx, "SELECT tablename FROM pg_tables").await,
            vec!["t"]
        );
    }

    #[tokio::test]
    async fn pg_namespace_lists_schemas() {
        let ctx = shimmed().await;
        let schemas = rows(&ctx, "SELECT nspname FROM pg_namespace").await;
        assert!(schemas.contains(&"public".to_owned()), "got {schemas:?}");
    }

    #[tokio::test]
    async fn empty_catalogs_still_have_the_shape_dbt_unions() {
        let ctx = shimmed().await;
        assert!(
            rows(&ctx, "SELECT matviewname FROM pg_matviews")
                .await
                .is_empty()
        );
        assert!(
            rows(
                &ctx,
                "SELECT proname FROM pg_proc JOIN pg_namespace AS ns ON pronamespace = ns.oid",
            )
            .await
            .is_empty()
        );
    }

    #[test]
    fn the_dependency_walk_is_recognized() {
        assert!(is_relation_dependency_query(
            "select distinct x from pg_class \
             join pg_depend as d on d.classid = 'pg_rewrite'::regclass",
        ));
        assert!(!is_relation_dependency_query("SELECT * FROM pg_tables"));
    }
}
