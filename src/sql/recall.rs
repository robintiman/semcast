//! Trailing `WITH <knob> <value>` clauses — statement-level syntax.
//!
//! The dialect can't carry these: infix `MEANS` desugars to a function call,
//! and a recall target rides the whole statement, not one expression. So this
//! is a wrapped-parser extension — parse the statement, then consume whatever
//! clauses trail it. [`crate::sql`] threads each value back into the calls it
//! governs.
//!
//! Two knobs today, both fractions in `(0, 1]`: `WITH RECALL` calibrates the
//! index pre-filter under a `MEANS`, and `WITH SIMILARITY` sets how alike two
//! documents must be to count as duplicates under a `SEMANTIC DISTINCT ON`.
//! `BUDGET` is meant to land here too.

use datafusion::error::DataFusionError;
use datafusion::sql::parser::{DFParserBuilder, Statement};
use datafusion::sql::sqlparser::keywords::Keyword;
use datafusion::sql::sqlparser::parser::Parser;
use datafusion::sql::sqlparser::tokenizer::Token;

use super::SemcastDialect;

/// The trailing knobs a statement carried.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TrailingClauses {
    /// `WITH RECALL <f>` — calibration target for the index pre-filter.
    pub recall: Option<f64>,
    /// `WITH SIMILARITY <f>` — the duplicate threshold for a semantic dedupe.
    pub similarity: Option<f64>,
}

/// Parse exactly one statement plus any trailing `WITH <knob> <value>`.
///
/// Statements go through DataFusion's [`DFParserBuilder`] rather than a raw
/// sqlparser `Parser` so DataFusion-only syntax — `CREATE EXTERNAL TABLE`,
/// `COPY ... TO` — parses too; everything else delegates to sqlparser under
/// [`SemcastDialect`].
pub fn parse_statement_with_recall(query: &str) -> crate::Result<(Statement, TrailingClauses)> {
    let dialect = SemcastDialect::default();
    let mut df_parser = DFParserBuilder::new(query).with_dialect(&dialect).build()?;
    let statement = df_parser.parse_statement()?;
    let parser = &mut df_parser.parser;

    // The statement parse stops before a trailing WITH — only the
    // multi-statement loop of `Parser::parse_sql` would reject it. Loop, so
    // a statement can carry more than one knob.
    let mut clauses = TrailingClauses::default();
    while parser.parse_keyword(Keyword::WITH) {
        let knob = match parser.next_token().token {
            Token::Word(word) => word.value.to_lowercase(),
            other => {
                return Err(plan_error(format!(
                    "expected RECALL or SIMILARITY after trailing WITH, got {other}"
                )));
            }
        };
        let slot = match knob.as_str() {
            "recall" => &mut clauses.recall,
            "similarity" => &mut clauses.similarity,
            other => {
                return Err(plan_error(format!(
                    "expected RECALL or SIMILARITY after trailing WITH, got {other}"
                )));
            }
        };
        if slot.is_some() {
            return Err(plan_error(format!(
                "WITH {} given twice",
                knob.to_uppercase()
            )));
        }
        *slot = Some(parse_fraction(parser, &knob.to_uppercase())?);
    }

    let _ = parser.consume_token(&Token::SemiColon);
    parser
        .expect_token(&Token::EOF)
        .map_err(|e| plan_error(format!("unexpected trailing input: {e}")))?;
    Ok((statement, clauses))
}

/// Both knobs are fractions in `(0, 1]`; neither zero nor "more than all"
/// means anything for a recall target or a similarity threshold.
fn parse_fraction(parser: &mut Parser, knob: &str) -> crate::Result<f64> {
    let target = match parser.next_token().token {
        Token::Number(number, _) => number
            .parse::<f64>()
            .map_err(|e| plan_error(format!("WITH {knob} expects a number, got {number}: {e}")))?,
        other => {
            return Err(plan_error(format!(
                "WITH {knob} expects a number in (0, 1], got {other}"
            )));
        }
    };
    if !(target > 0.0 && target <= 1.0) {
        return Err(plan_error(format!(
            "WITH {knob} must be in (0, 1], got {target}"
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
        parse_statement_with_recall(query).unwrap().1.recall
    }

    fn similarity_of(query: &str) -> Option<f64> {
        parse_statement_with_recall(query).unwrap().1.similarity
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
        let (statement, clauses) =
            parse_statement_with_recall("EXPLAIN SELECT 1 WITH RECALL 0.9").unwrap();
        assert!(matches!(statement, Statement::Explain(_)));
        assert_eq!(clauses.recall, Some(0.9));
    }

    #[test]
    fn parses_create_external_table() {
        let (statement, clauses) = parse_statement_with_recall(
            "CREATE EXTERNAL TABLE t STORED AS CSV LOCATION '/data/t.csv'",
        )
        .unwrap();
        assert!(matches!(statement, Statement::CreateExternalTable(_)));
        assert_eq!(clauses, TrailingClauses::default());
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
    fn trailing_with_that_is_not_a_known_knob_is_rejected() {
        let message = error_of("SELECT 1 WITH options");
        assert!(
            message.contains("expected RECALL or SIMILARITY"),
            "got: {message}"
        );
    }

    #[test]
    fn parses_trailing_with_similarity() {
        assert_eq!(similarity_of("SELECT 1 WITH SIMILARITY 0.85"), Some(0.85));
        assert_eq!(similarity_of("select 1 with similarity 0.5;"), Some(0.5));
        assert_eq!(similarity_of("SELECT 1"), None);
    }

    #[test]
    fn similarity_shares_the_range_check() {
        assert!(error_of("SELECT 1 WITH SIMILARITY 0").contains("must be in (0, 1]"));
        assert!(error_of("SELECT 1 WITH SIMILARITY 1.5").contains("must be in (0, 1]"));
        assert!(error_of("SELECT 1 WITH SIMILARITY high").contains("expects a number"));
    }

    /// Both knobs can ride one statement — a dedupe over a filtered scan
    /// wants a threshold *and* a recall target.
    #[test]
    fn knobs_compose() {
        let clauses = parse_statement_with_recall("SELECT 1 WITH RECALL 0.9 WITH SIMILARITY 0.8")
            .unwrap()
            .1;
        assert_eq!(clauses.recall, Some(0.9));
        assert_eq!(clauses.similarity, Some(0.8));
    }

    #[test]
    fn the_same_knob_twice_is_rejected() {
        let message = error_of("SELECT 1 WITH RECALL 0.9 WITH RECALL 0.8");
        assert!(message.contains("given twice"), "got: {message}");
    }
}
