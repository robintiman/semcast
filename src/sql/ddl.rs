//! `CREATE SEMANTIC INDEX / TYPE / PREDICATE` statements.
//!
//! DataFusion hook: a custom statement parser (sqlparser dialect extension)
//! feeding a planner extension. Roadmap: `CREATE SEMANTIC INDEX` arrives with
//! step 2; `TYPE` and `PREDICATE` with typed extraction.

use datafusion::error::DataFusionError;
use datafusion::sql::sqlparser::dialect::GenericDialect;
use datafusion::sql::sqlparser::parser::Parser;
use datafusion::sql::sqlparser::tokenizer::Token;

use crate::types::SemanticType;

#[derive(Debug, Clone, PartialEq)]
pub enum SemanticDdl {
    /// `CREATE SEMANTIC INDEX ON table(column)`
    CreateIndex { table: String, column: String },
    /// `CREATE SEMANTIC TYPE Name AS (...)`
    CreateType(SemanticType),
    /// `CREATE SEMANTIC PREDICATE name(params) AS template [CHEAP USING ...]`
    CreatePredicate {
        name: String,
        params: Vec<String>,
        /// `MEANS` template with `{param}` placeholders.
        template: String,
        /// Optional `CHEAP USING` override — a user-pinned pre-filter.
        cheap_using: Option<String>,
    },
}

/// Recognize a semantic DDL statement; `Ok(None)` means "not ours, let
/// DataFusion parse it".
pub fn parse_semantic_ddl(sql: &str) -> crate::Result<Option<SemanticDdl>> {
    // Cheap gate so ordinary statements never pay for a second parse: the
    // first two words must be CREATE SEMANTIC.
    let mut words = sql.split_whitespace();
    let prefix_matches = words
        .next()
        .is_some_and(|w| w.eq_ignore_ascii_case("create"))
        && words
            .next()
            .is_some_and(|w| w.eq_ignore_ascii_case("semantic"));
    if !prefix_matches {
        return Ok(None);
    }

    let mut parser = Parser::new(&GenericDialect {})
        .try_with_sql(sql)
        .map_err(|e| DataFusionError::Plan(format!("invalid CREATE SEMANTIC statement: {e}")))?;
    let error = |message: String| {
        crate::SemcastError::DataFusion(DataFusionError::Plan(format!(
            "invalid CREATE SEMANTIC INDEX statement: {message}; \
             expected CREATE SEMANTIC INDEX ON table(column)"
        )))
    };

    expect_word(&mut parser, "CREATE").map_err(&error)?;
    expect_word(&mut parser, "SEMANTIC").map_err(&error)?;
    match parser.next_token().token {
        Token::Word(word) if word.value.eq_ignore_ascii_case("index") => {}
        Token::Word(word)
            if word.value.eq_ignore_ascii_case("type")
                || word.value.eq_ignore_ascii_case("predicate") =>
        {
            return Err(crate::SemcastError::DataFusion(DataFusionError::Plan(
                format!(
                    "CREATE SEMANTIC {} is not implemented yet (typed extraction roadmap step)",
                    word.value.to_uppercase()
                ),
            )));
        }
        other => {
            return Err(error(format!(
                "expected INDEX, TYPE, or PREDICATE, got {other}"
            )));
        }
    }
    expect_word(&mut parser, "ON").map_err(&error)?;
    let table = parser
        .parse_identifier()
        .map_err(|e| error(e.to_string()))?;
    parser
        .expect_token(&Token::LParen)
        .map_err(|e| error(e.to_string()))?;
    let column = parser
        .parse_identifier()
        .map_err(|e| error(e.to_string()))?;
    parser
        .expect_token(&Token::RParen)
        .map_err(|e| error(e.to_string()))?;
    let _ = parser.consume_token(&Token::SemiColon);
    parser
        .expect_token(&Token::EOF)
        .map_err(|e| error(e.to_string()))?;

    Ok(Some(SemanticDdl::CreateIndex {
        table: table.value,
        column: column.value,
    }))
}

/// `SEMANTIC`, `INDEX`, and `ON` are plain words to the tokenizer (or
/// keywords we don't want keyword semantics for) — match them textually.
fn expect_word(parser: &mut Parser, expected: &str) -> Result<(), String> {
    match parser.next_token().token {
        Token::Word(word) if word.value.eq_ignore_ascii_case(expected) => Ok(()),
        other => Err(format!("expected {expected}, got {other}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_index(sql: &str) -> SemanticDdl {
        parse_semantic_ddl(sql).unwrap().expect("recognized as DDL")
    }

    #[test]
    fn parses_create_semantic_index() {
        assert_eq!(
            create_index("CREATE SEMANTIC INDEX ON meetings(transcript)"),
            SemanticDdl::CreateIndex {
                table: "meetings".to_owned(),
                column: "transcript".to_owned(),
            },
        );
    }

    #[test]
    fn is_case_insensitive_and_tolerates_whitespace_and_semicolon() {
        assert_eq!(
            create_index("  create Semantic INDEX on  Meetings ( Transcript ) ;"),
            SemanticDdl::CreateIndex {
                table: "Meetings".to_owned(),
                column: "Transcript".to_owned(),
            },
        );
    }

    #[test]
    fn respects_quoted_identifiers() {
        assert_eq!(
            create_index(r#"CREATE SEMANTIC INDEX ON "my table"("weird column")"#),
            SemanticDdl::CreateIndex {
                table: "my table".to_owned(),
                column: "weird column".to_owned(),
            },
        );
    }

    #[test]
    fn ordinary_statements_are_not_ours() {
        assert_eq!(parse_semantic_ddl("SELECT 1").unwrap(), None);
        assert_eq!(
            parse_semantic_ddl("CREATE TABLE t AS SELECT 1").unwrap(),
            None,
        );
        assert_eq!(parse_semantic_ddl("").unwrap(), None);
        assert_eq!(parse_semantic_ddl("CREATE").unwrap(), None);
    }

    #[test]
    fn malformed_index_statement_is_a_clear_error() {
        let err = parse_semantic_ddl("CREATE SEMANTIC INDEX meetings(transcript)").unwrap_err();
        let message = err.to_string();
        assert!(message.contains("expected ON"), "got: {message}");
        assert!(
            message.contains("CREATE SEMANTIC INDEX ON table(column)"),
            "shows the expected shape: {message}",
        );

        let err = parse_semantic_ddl("CREATE SEMANTIC INDEX ON meetings(transcript) garbage")
            .unwrap_err();
        assert!(err.to_string().contains("EOF"), "got: {err}");
    }

    #[test]
    fn type_and_predicate_are_explicit_not_implemented() {
        let err = parse_semantic_ddl("CREATE SEMANTIC TYPE Facts AS (x TEXT 'doc')").unwrap_err();
        assert!(err.to_string().contains("not implemented"), "got: {err}");
        let err = parse_semantic_ddl("CREATE SEMANTIC PREDICATE p(t) AS t").unwrap_err();
        assert!(err.to_string().contains("not implemented"), "got: {err}");
    }
}
