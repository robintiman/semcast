//! semcast's statement grammar — the one place a statement is parsed.
//!
//! A semcast statement is either the custom `CREATE SEMANTIC ...` DDL
//! ([`crate::sql::ddl`]) or a SQL statement carrying trailing `WITH` knobs
//! ([`crate::sql::recall`]). Both come out of a single [`DFParser`]: the
//! parser is built once, then a two-token lookahead on the live token stream
//! decides which branch runs. That is ordinary recursive descent, and it is
//! what keeps comments invisible — `Parser::peek_nth_token_ref` skips
//! `Token::Whitespace`, and sqlparser models `--` and `/* */` as whitespace.
//!
//! Statements go through DataFusion's [`DFParser`] rather than a raw sqlparser
//! [`Parser`] so DataFusion-only syntax — `CREATE EXTERNAL TABLE`,
//! `COPY ... TO` — parses too; everything else delegates to sqlparser under
//! [`SemcastDialect`].
//!
//! # Why the DDL is not a dialect hook
//!
//! sqlparser has a `Dialect::parse_statement` hook, and it cannot carry this:
//!
//! 1. `DFParser::parse_statement` matches `Keyword::CREATE` itself, consumes
//!    the token, and calls `parser.parse_create()` directly. sqlparser only
//!    consults the hook from `Parser::parse_statement`, which that path never
//!    reaches — so the hook would never fire for `CREATE SEMANTIC`.
//! 2. The hook must return a `sqlparser::ast::Statement`, and no variant can
//!    carry a [`SemanticDdl`] out to [`crate::sql()`], which has to *act* on it
//!    (build an index, register a type) rather than plan it.
//!
//! Dispatching inside the parser we already own is what's left, and it is
//! enough.

use datafusion::sql::parser::{DFParserBuilder, Statement};

use crate::sql::SemcastDialect;
use crate::sql::ddl::{self, SemanticDdl};
use crate::sql::head::peek_nth_is_word;
use crate::sql::recall::{TrailingClauses, trailing_clauses};

/// One parsed semcast statement.
#[derive(Debug)]
pub enum SemcastStatement {
    /// `CREATE SEMANTIC INDEX / TYPE / PREDICATE` — executed, not planned.
    Ddl(SemanticDdl),
    /// Everything else, plus whatever trailing knobs it carried.
    Sql(Statement, TrailingClauses),
}

/// Parse exactly one semcast statement.
pub fn parse_semcast_statement(query: &str) -> crate::Result<SemcastStatement> {
    let dialect = SemcastDialect::default();
    let mut df_parser = DFParserBuilder::new(query).with_dialect(&dialect).build()?;

    // The whole custom-DDL dispatch: two tokens of lookahead, no text
    // inspection. Requiring *unquoted* words keeps `"CREATE" "SEMANTIC"` —
    // quoted identifiers — out of the branch.
    if peek_nth_is_word(&df_parser.parser, 0, "CREATE")
        && peek_nth_is_word(&df_parser.parser, 1, "SEMANTIC")
    {
        let parser = &mut df_parser.parser;
        parser.next_token(); // CREATE
        parser.next_token(); // SEMANTIC
        return ddl::parse_semantic_ddl(parser).map(SemcastStatement::Ddl);
    }

    let statement = df_parser.parse_statement()?;
    let clauses = trailing_clauses(&mut df_parser.parser)?;
    Ok(SemcastStatement::Sql(statement, clauses))
}

/// [`parse_semcast_statement`] for callers that only handle SQL — the AST
/// passes, which never see DDL.
pub fn parse_statement_with_recall(query: &str) -> crate::Result<(Statement, TrailingClauses)> {
    match parse_semcast_statement(query)? {
        SemcastStatement::Sql(statement, clauses) => Ok((statement, clauses)),
        SemcastStatement::Ddl(ddl) => Err(datafusion::error::DataFusionError::Plan(format!(
            "expected a SQL statement, got semantic DDL: {ddl:?}"
        ))
        .into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ddl_of(sql: &str) -> SemanticDdl {
        match parse_semcast_statement(sql).unwrap() {
            SemcastStatement::Ddl(ddl) => ddl,
            SemcastStatement::Sql(statement, _) => {
                panic!("expected DDL, parsed as SQL: {statement}")
            }
        }
    }

    fn is_sql(sql: &str) -> bool {
        matches!(
            parse_semcast_statement(sql).unwrap(),
            SemcastStatement::Sql(..)
        )
    }

    fn index_on(table: &str, column: &str) -> SemanticDdl {
        SemanticDdl::CreateIndex {
            table: table.to_owned(),
            column: column.to_owned(),
        }
    }

    #[test]
    fn dispatches_create_semantic_to_the_ddl_branch() {
        assert_eq!(
            ddl_of("CREATE SEMANTIC INDEX ON meetings(transcript)"),
            index_on("meetings", "transcript"),
        );
    }

    /// Issue #17: a leading comment used to hide the statement, because the
    /// dispatch read raw text instead of tokens.
    #[test]
    fn a_comment_never_hides_the_statement() {
        for sql in [
            "-- any comment at all\nCREATE SEMANTIC INDEX ON meetings(transcript)",
            "/* hi */ CREATE SEMANTIC INDEX ON meetings(transcript)",
            "/* nested /* comment */ */\nCREATE SEMANTIC INDEX ON meetings(transcript)",
            "CREATE /* between */ SEMANTIC -- and here\n INDEX ON meetings(transcript)",
            "\n\n  -- indented\n  CREATE SEMANTIC INDEX ON meetings(transcript);",
        ] {
            assert_eq!(
                ddl_of(sql),
                index_on("meetings", "transcript"),
                "for `{sql}`"
            );
        }
    }

    #[test]
    fn a_comment_never_hides_create_semantic_type() {
        let ddl = ddl_of("/* hi */\nCREATE SEMANTIC TYPE Foo AS (claim TEXT 'the main claim')");
        match ddl {
            SemanticDdl::CreateType(ty) => {
                assert_eq!(ty.name, "Foo");
                assert_eq!(ty.fields.len(), 1);
            }
            other => panic!("expected CreateType, got {other:?}"),
        }
    }

    #[test]
    fn ordinary_statements_take_the_sql_branch() {
        assert!(is_sql("SELECT 1"));
        assert!(is_sql("CREATE TABLE t AS SELECT 1"));
        assert!(is_sql(
            "CREATE EXTERNAL TABLE t STORED AS CSV LOCATION '/data/t.csv'"
        ));
        assert!(is_sql("EXPLAIN SELECT 1"));
    }

    #[test]
    fn the_words_only_count_as_tokens() {
        // In a string literal, and as quoted identifiers — neither is the DDL.
        assert!(is_sql("SELECT 'CREATE SEMANTIC INDEX ON t(c)'"));
        assert!(is_sql(r#"SELECT "create", "semantic" FROM t"#));
    }

    #[test]
    fn trailing_knobs_still_attach() {
        let (_, clauses) =
            parse_statement_with_recall("SELECT * FROM t WHERE x MEANS 'c' WITH RECALL 0.9")
                .unwrap();
        assert_eq!(clauses.recall, Some(0.9));
    }

    #[test]
    fn a_malformed_semantic_statement_still_says_what_it_wanted() {
        // The branch is taken on CREATE SEMANTIC, so the error comes from the
        // DDL parser rather than from DataFusion complaining about `SEMANTIC`.
        let err = parse_semcast_statement("-- c\nCREATE SEMANTIC INDEX meetings(transcript)")
            .unwrap_err()
            .to_string();
        assert!(err.contains("expected ON"), "got: {err}");
        assert!(
            err.contains("CREATE SEMANTIC INDEX ON table(column)"),
            "shows the expected shape: {err}",
        );
    }
}
