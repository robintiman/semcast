//! Parsing `SELECT SEMANTIC DISTINCT ON (...)`.
//!
//! `DISTINCT` is a select modifier, and sqlparser offers no dialect hook for
//! that position — `parse_prefix` and `parse_infix` are expression hooks, and
//! by the time either could fire the modifier has already been read. So the
//! word `SEMANTIC` is removed from the text *before* parsing, leaving an
//! ordinary Postgres `DISTINCT ON` that sqlparser and DataFusion both already
//! understand.
//!
//! Removal is done over the token stream rather than the raw string: a
//! `WHERE body MEANS 'semantic distinct'` must not be touched, and only the
//! tokenizer knows where the string literals are.
//!
//! What survives into the AST is then marked structurally — each `DISTINCT
//! ON` expression is wrapped in [`semantic_key`] — so nothing downstream has
//! to be told out of band which `DISTINCT` was the semantic one.
//!
//! [`semantic_key`]: crate::sql::distinct_udf

use std::ops::ControlFlow;

use datafusion::sql::parser::Statement;
use datafusion::sql::sqlparser::ast::{
    Distinct, Expr, Function, FunctionArg, FunctionArgExpr, FunctionArgumentList,
    FunctionArguments, Ident, ObjectName, Query, SetExpr, VisitMut, VisitorMut,
};
use datafusion::sql::sqlparser::keywords::Keyword;
use datafusion::sql::sqlparser::tokenizer::{Token, Tokenizer};

use crate::sql::SemcastDialect;
use crate::sql::distinct_udf::SEMANTIC_KEY_UDF_NAME;

/// Remove the word `SEMANTIC` where it qualifies a `DISTINCT`, returning the
/// rewritten SQL and whether anything was removed.
///
/// Tokenizing costs one extra pass over the statement, and buys the guarantee
/// that `SEMANTIC` inside a string literal, a comment, or a quoted identifier
/// is left alone.
pub fn strip_semantic_distinct(sql: &str) -> (String, bool) {
    let Ok(tokens) = Tokenizer::new(&SemcastDialect::default(), sql).tokenize_with_location()
    else {
        // Not tokenizable — let the real parser produce the error.
        return (sql.to_owned(), false);
    };

    let offsets = line_offsets(sql);
    let mut cuts = Vec::new();
    // Comments are `Whitespace` variants in sqlparser, so one filter covers
    // both — `SEMANTIC /* why */ DISTINCT` still qualifies.
    let mut significant = tokens
        .iter()
        .filter(|t| !matches!(t.token, Token::Whitespace(_)));
    let mut previous = significant.next();
    for token in significant {
        if let Some(prev) = previous
            && matches!(&prev.token, Token::Word(w)
                if w.quote_style.is_none() && w.value.eq_ignore_ascii_case("SEMANTIC"))
            && matches!(&token.token, Token::Word(w) if w.keyword == Keyword::DISTINCT)
            && let (Some(start), Some(end)) = (
                byte_offset(sql, &offsets, prev.span.start.line, prev.span.start.column),
                byte_offset(sql, &offsets, prev.span.end.line, prev.span.end.column),
            )
        {
            cuts.push((start, end));
        }
        previous = Some(token);
    }
    if cuts.is_empty() {
        return (sql.to_owned(), false);
    }

    let mut stripped = String::with_capacity(sql.len());
    let mut cursor = 0;
    for (start, end) in cuts {
        stripped.push_str(&sql[cursor..start]);
        cursor = end;
    }
    stripped.push_str(&sql[cursor..]);
    (stripped, true)
}

/// Byte offset of the start of each 1-based line.
fn line_offsets(sql: &str) -> Vec<usize> {
    let mut offsets = vec![0usize];
    for (i, byte) in sql.bytes().enumerate() {
        if byte == b'\n' {
            offsets.push(i + 1);
        }
    }
    offsets
}

/// sqlparser reports 1-based line and column in *characters*; the cut needs
/// bytes. Walking the line rather than adding the column keeps a multibyte
/// literal earlier on the same line from shifting the cut.
fn byte_offset(sql: &str, offsets: &[usize], line: u64, column: u64) -> Option<usize> {
    let line_start = *offsets.get(line.checked_sub(1)? as usize)?;
    let column = column.checked_sub(1)? as usize;
    let rest = sql.get(line_start..)?;
    Some(match rest.char_indices().nth(column) {
        Some((offset, _)) => line_start + offset,
        // The location is the end of the text.
        None => sql.len(),
    })
}

/// Wrap every `DISTINCT ON` expression in `semantic_key(...)`, so the marker
/// travels in the AST rather than beside it.
pub fn mark_semantic_distinct(statement: &mut Statement) -> bool {
    let Statement::Statement(inner) = statement else {
        return false;
    };
    let mut marker = MarkDistinct { marked: false };
    let _: ControlFlow<()> = inner.as_mut().visit(&mut marker);
    marker.marked
}

struct MarkDistinct {
    marked: bool,
}

impl VisitorMut for MarkDistinct {
    type Break = ();

    fn pre_visit_query(&mut self, query: &mut Query) -> ControlFlow<Self::Break> {
        if let SetExpr::Select(select) = query.body.as_mut()
            && let Some(Distinct::On(exprs)) = select.distinct.as_mut()
        {
            for expr in exprs.iter_mut() {
                *expr = key_call(expr.clone());
            }
            self.marked = true;
        }
        ControlFlow::Continue(())
    }
}

fn key_call(arg: Expr) -> Expr {
    Expr::Function(Function {
        name: ObjectName::from(vec![Ident::new(SEMANTIC_KEY_UDF_NAME)]),
        uses_odbc_syntax: false,
        parameters: FunctionArguments::None,
        args: FunctionArguments::List(FunctionArgumentList {
            duplicate_treatment: None,
            args: vec![FunctionArg::Unnamed(FunctionArgExpr::Expr(arg))],
            clauses: vec![],
        }),
        filter: None,
        null_treatment: None,
        over: None,
        within_group: vec![],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sql::recall::parse_statement_with_recall;

    fn strip(sql: &str) -> (String, bool) {
        strip_semantic_distinct(sql)
    }

    #[test]
    fn removes_the_qualifier_before_distinct() {
        let (sql, found) = strip("SELECT SEMANTIC DISTINCT ON (body) * FROM t");
        assert!(found);
        assert_eq!(sql, "SELECT  DISTINCT ON (body) * FROM t");
    }

    #[test]
    fn is_case_insensitive_and_tolerates_newlines() {
        let (sql, found) = strip("SELECT\n  semantic distinct ON (body) *\nFROM t");
        assert!(found);
        assert_eq!(sql, "SELECT\n   distinct ON (body) *\nFROM t");
    }

    #[test]
    fn leaves_a_plain_distinct_alone() {
        let (sql, found) = strip("SELECT DISTINCT body FROM t");
        assert!(!found);
        assert_eq!(sql, "SELECT DISTINCT body FROM t");
    }

    #[test]
    fn a_string_literal_is_never_cut() {
        // The tokenizer is what makes this safe: a naive text scan would
        // mangle the predicate.
        let sql = "SELECT * FROM t WHERE body MEANS 'semantic distinct things'";
        let (out, found) = strip(sql);
        assert!(!found);
        assert_eq!(out, sql);
    }

    #[test]
    fn a_quoted_identifier_is_never_cut() {
        let sql = r#"SELECT "semantic" FROM t"#;
        let (out, found) = strip(sql);
        assert!(!found);
        assert_eq!(out, sql);
    }

    #[test]
    fn semantic_stays_usable_as_a_bare_column_name() {
        let sql = "SELECT semantic FROM t WHERE semantic > 1";
        let (out, found) = strip(sql);
        assert!(!found, "only a SEMANTIC that qualifies DISTINCT is removed");
        assert_eq!(out, sql);
    }

    #[test]
    fn marks_the_distinct_on_expressions() {
        let (sql, _) = strip("SELECT SEMANTIC DISTINCT ON (body) * FROM t");
        let (mut statement, _) = parse_statement_with_recall(&sql).unwrap();
        assert!(mark_semantic_distinct(&mut statement));
        assert_eq!(
            statement.to_string(),
            "SELECT DISTINCT ON (semantic_key(body)) * FROM t"
        );
    }

    #[test]
    fn marks_every_expression_in_the_list() {
        let (sql, _) = strip("SELECT SEMANTIC DISTINCT ON (title, body) * FROM t");
        let (mut statement, _) = parse_statement_with_recall(&sql).unwrap();
        assert!(mark_semantic_distinct(&mut statement));
        assert_eq!(
            statement.to_string(),
            "SELECT DISTINCT ON (semantic_key(title), semantic_key(body)) * FROM t"
        );
    }
}
