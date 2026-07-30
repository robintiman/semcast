//! `SUBMIT <statement>` and `CANCEL JOB '<id>'`.
//!
//! `SUBMIT` deliberately does not re-parse or re-serialize its payload: the
//! remainder of the string is handed to the engine byte-for-byte, so `MEANS
//! '...'` literals and a trailing `WITH RECALL` behave exactly as they do on
//! the interactive path.

use datafusion::sql::sqlparser::dialect::GenericDialect;
use datafusion::sql::sqlparser::parser::Parser;
use datafusion::sql::sqlparser::tokenizer::Token;

/// The statement behind a leading `SUBMIT`, or `None` if this is not one.
pub fn parse_submit(statement: &str) -> Option<&str> {
    let rest = strip_keyword(statement, "submit")?;
    (!rest.is_empty()).then_some(rest)
}

/// The job id in `CANCEL JOB '<id>'`. `Ok(None)` means "not ours".
pub fn parse_cancel_job(statement: &str) -> Result<Option<String>, String> {
    // Cheap gate first so ordinary statements never pay for a parse.
    let Some(rest) = strip_keyword(statement, "cancel") else {
        return Ok(None);
    };
    let Some(rest) = strip_keyword(rest, "job") else {
        return Ok(None);
    };

    let mut parser = Parser::new(&GenericDialect {})
        .try_with_sql(rest)
        .map_err(|e| cancel_error(e.to_string()))?;
    let id = match parser.next_token().token {
        Token::SingleQuotedString(id) => id,
        // An unquoted id is the obvious typo; accept it rather than nitpick.
        Token::Word(word) => word.value,
        other => return Err(cancel_error(format!("expected a job id, got {other}"))),
    };
    let _ = parser.consume_token(&Token::SemiColon);
    parser
        .expect_token(&Token::EOF)
        .map_err(|e| cancel_error(e.to_string()))?;
    Ok(Some(id))
}

/// The remainder after a leading `keyword`, if `statement` starts with it as a
/// whole word.
fn strip_keyword<'a>(statement: &'a str, keyword: &str) -> Option<&'a str> {
    let trimmed = statement.trim_start();
    let head = trimmed.get(..keyword.len())?;
    if !head.eq_ignore_ascii_case(keyword) {
        return None;
    }
    let rest = &trimmed[keyword.len()..];
    // `SUBMITTED` is not `SUBMIT`.
    if !rest.is_empty() && !rest.starts_with(|c: char| c.is_whitespace()) {
        return None;
    }
    Some(rest.trim())
}

fn cancel_error(message: impl Into<String>) -> String {
    format!(
        "invalid CANCEL JOB statement: {}; expected CANCEL JOB '<job_id>'",
        message.into()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn submit_hands_the_statement_through_untouched() {
        assert_eq!(
            parse_submit("SUBMIT SELECT * FROM t WHERE body MEANS 'a; b'"),
            Some("SELECT * FROM t WHERE body MEANS 'a; b'"),
        );
        assert_eq!(
            parse_submit("submit  SELECT 1 WITH RECALL 0.9"),
            Some("SELECT 1 WITH RECALL 0.9"),
        );
        assert_eq!(
            parse_submit("SUBMIT\nCREATE SEMANTIC INDEX ON t(c)"),
            Some("CREATE SEMANTIC INDEX ON t(c)"),
        );
    }

    #[test]
    fn submit_needs_a_statement_and_a_word_boundary() {
        assert_eq!(parse_submit("SUBMIT"), None);
        assert_eq!(parse_submit("SUBMIT   "), None);
        assert_eq!(parse_submit("SUBMITTED SELECT 1"), None);
        assert_eq!(parse_submit("SELECT * FROM submissions"), None);
    }

    #[test]
    fn cancel_job_takes_a_quoted_or_bare_id() {
        assert_eq!(
            parse_cancel_job("CANCEL JOB 'job_17_0001'").unwrap(),
            Some("job_17_0001".to_owned()),
        );
        assert_eq!(
            parse_cancel_job("cancel job job_17_0001;").unwrap(),
            Some("job_17_0001".to_owned()),
        );
    }

    #[test]
    fn other_statements_are_not_cancel_job() {
        assert_eq!(parse_cancel_job("SELECT 1").unwrap(), None);
        assert_eq!(parse_cancel_job("CANCEL").unwrap(), None);
        assert_eq!(parse_cancel_job("CANCEL QUERY 'x'").unwrap(), None);
    }

    #[test]
    fn malformed_cancel_job_says_what_it_wanted() {
        let err = parse_cancel_job("CANCEL JOB 'a' extra").unwrap_err();
        assert!(err.contains("CANCEL JOB '<job_id>'"), "got: {err}");
        let err = parse_cancel_job("CANCEL JOB").unwrap_err();
        assert!(err.contains("expected a job id"), "got: {err}");
    }
}
