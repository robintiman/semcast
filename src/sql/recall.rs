//! Trailing `WITH RECALL <target>` — statement-level syntax.
//!
//! The dialect can't carry it: infix `MEANS` desugars to a function call,
//! and the recall target rides the whole statement, not one expression. So
//! this is a wrapped-parser extension — parse the statement, then consume
//! the clause if one trails it. [`crate::sql`] threads the target back into
//! every `means()` call in the plan.

use datafusion::error::DataFusionError;
use datafusion::sql::sqlparser::ast::Statement;
use datafusion::sql::sqlparser::keywords::Keyword;
use datafusion::sql::sqlparser::parser::Parser;
use datafusion::sql::sqlparser::tokenizer::Token;

use super::SemcastDialect;

/// Parse exactly one statement plus an optional trailing `WITH RECALL <f>`.
pub fn parse_statement_with_recall(query: &str) -> crate::Result<(Statement, Option<f64>)> {
    let dialect = SemcastDialect::default();
    let mut parser = Parser::new(&dialect)
        .try_with_sql(query)
        .map_err(DataFusionError::from)?;
    let statement = parser.parse_statement().map_err(DataFusionError::from)?;

    // The statement parse stops before a trailing WITH — only the
    // multi-statement loop of `Parser::parse_sql` would reject it.
    let recall = if parser.parse_keyword(Keyword::WITH) {
        match parser.next_token().token {
            Token::Word(word) if word.value.eq_ignore_ascii_case("recall") => {}
            other => {
                return Err(plan_error(format!(
                    "expected RECALL after trailing WITH, got {other}"
                )));
            }
        }
        Some(parse_recall_target(&mut parser)?)
    } else {
        None
    };

    let _ = parser.consume_token(&Token::SemiColon);
    parser
        .expect_token(&Token::EOF)
        .map_err(|e| plan_error(format!("unexpected trailing input: {e}")))?;
    Ok((statement, recall))
}

fn parse_recall_target(parser: &mut Parser) -> crate::Result<f64> {
    let target = match parser.next_token().token {
        Token::Number(number, _) => number
            .parse::<f64>()
            .map_err(|e| plan_error(format!("WITH RECALL expects a number, got {number}: {e}")))?,
        other => {
            return Err(plan_error(format!(
                "WITH RECALL expects a number in (0, 1], got {other}"
            )));
        }
    };
    if !(target > 0.0 && target <= 1.0) {
        return Err(plan_error(format!(
            "WITH RECALL must be in (0, 1], got {target}"
        )));
    }
    Ok(target)
}

fn plan_error(message: String) -> crate::SemcastError {
    crate::SemcastError::DataFusion(DataFusionError::Plan(message))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recall_of(query: &str) -> Option<f64> {
        parse_statement_with_recall(query).unwrap().1
    }

    fn error_of(query: &str) -> String {
        parse_statement_with_recall(query).unwrap_err().to_string()
    }

    #[test]
    fn statement_without_clause_parses_as_before() {
        assert_eq!(recall_of("SELECT 1"), None);
        assert_eq!(recall_of("SELECT 1;"), None);
    }

    #[test]
    fn parses_trailing_with_recall() {
        assert_eq!(
            recall_of("SELECT * FROM t WHERE x MEANS 'c' WITH RECALL 0.9"),
            Some(0.9),
        );
    }

    #[test]
    fn is_case_insensitive_and_tolerates_whitespace_and_semicolon() {
        assert_eq!(recall_of("SELECT 1  with   Recall 0.75 ;"), Some(0.75));
    }

    #[test]
    fn recall_one_is_allowed() {
        assert_eq!(recall_of("SELECT 1 WITH RECALL 1"), Some(1.0));
    }

    #[test]
    fn works_under_explain() {
        let (statement, recall) =
            parse_statement_with_recall("EXPLAIN SELECT 1 WITH RECALL 0.9").unwrap();
        assert!(matches!(statement, Statement::Explain { .. }));
        assert_eq!(recall, Some(0.9));
    }

    #[test]
    fn clause_inside_a_string_literal_is_not_the_clause() {
        assert_eq!(
            recall_of("SELECT * FROM t WHERE x MEANS 'says WITH RECALL 0.9'"),
            None,
        );
    }

    #[test]
    fn leading_with_cte_is_still_a_cte() {
        assert_eq!(
            recall_of("WITH c AS (SELECT 1 AS x) SELECT x FROM c WITH RECALL 0.5"),
            Some(0.5),
        );
    }

    #[test]
    fn out_of_range_targets_are_rejected() {
        for query in [
            "SELECT 1 WITH RECALL 0",
            "SELECT 1 WITH RECALL 0.0",
            "SELECT 1 WITH RECALL 1.5",
        ] {
            let message = error_of(query);
            assert!(message.contains("(0, 1]"), "got: {message}");
        }
    }

    #[test]
    fn negative_and_non_numeric_targets_are_clear_errors() {
        let message = error_of("SELECT 1 WITH RECALL -0.9");
        assert!(message.contains("expects a number"), "got: {message}");
        let message = error_of("SELECT 1 WITH RECALL high");
        assert!(message.contains("expects a number"), "got: {message}");
        let message = error_of("SELECT 1 WITH RECALL");
        assert!(message.contains("expects a number"), "got: {message}");
    }

    #[test]
    fn trailing_garbage_is_rejected() {
        let message = error_of("SELECT 1 WITH RECALL 0.9 garbage");
        assert!(message.contains("trailing input"), "got: {message}");
        let message = error_of("SELECT 1; SELECT 2");
        assert!(message.contains("trailing input"), "got: {message}");
    }

    #[test]
    fn trailing_with_that_is_not_recall_is_rejected() {
        let message = error_of("SELECT 1 WITH options");
        assert!(message.contains("expected RECALL"), "got: {message}");
    }
}
