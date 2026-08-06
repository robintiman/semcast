//! [`SemcastDialect`] — sqlparser dialect adding the infix `MEANS` and
//! `RELEVANCE TO` operators.
//!
//! `text MEANS 'condition'` desugars at parse time into a call to the
//! [`means`] marker UDF, and `text RELEVANCE TO 'query'` into [`relevance`],
//! so the dialect is pure surface syntax: the rewrite rules, extension nodes,
//! and execution operators see exactly what they see today.
//!
//! The dialect wraps [`GenericDialect`] (DataFusion's default) and forwards
//! its feature flags, so everything Generic accepts still parses. One
//! deliberate cost: `means` is effectively a reserved word — an unquoted
//! trailing alias like `SELECT x means FROM t` no longer parses. Quote it
//! (`"means"`) if you need it as an identifier. `relevance` costs nothing,
//! because it only binds when followed by `TO`.
//!
//! [`means`]: crate::sql::means_udf
//! [`relevance`]: crate::sql::rank_udf

use datafusion::sql::sqlparser::ast::{
    Expr, Function, FunctionArg, FunctionArgExpr, FunctionArgumentList, FunctionArguments, Ident,
    ObjectName, Value,
};
use datafusion::sql::sqlparser::dialect::{Dialect, GenericDialect, Precedence};
use datafusion::sql::sqlparser::keywords::Keyword;
use datafusion::sql::sqlparser::parser::{Parser, ParserError};
use datafusion::sql::sqlparser::tokenizer::Token;

use crate::sql::cluster_udf::MEANING_OF_UDF_NAME;
use crate::sql::ddl::parse_field_type;
use crate::sql::extract_udf::SEM_EXTRACT_INLINE_UDF_NAME;
use crate::sql::head::{peek_is_word, peek_nth_is_word};
use crate::sql::means_udf::MEANS_UDF_NAME;
use crate::sql::rank_udf::RELEVANCE_UDF_NAME;

/// Forward `fn(&self) -> bool` feature flags to the wrapped [`GenericDialect`]
/// so we track its behavior instead of the trait's conservative defaults.
macro_rules! delegate_flags {
    ($($method:ident),* $(,)?) => {
        $(fn $method(&self) -> bool { self.generic.$method() })*
    };
}

#[derive(Debug, Default)]
pub struct SemcastDialect {
    generic: GenericDialect,
}

/// Is the next token the (case-insensitive, unquoted) word `MEANS`?
///
/// `MEANS` is not a sqlparser keyword, so it tokenizes as a plain word with
/// `Keyword::NoKeyword`; a quoted `"means"` keeps its quote style and stays
/// an ordinary identifier.
fn peek_is_means(parser: &Parser) -> bool {
    match &parser.peek_token_ref().token {
        Token::Word(w) => {
            w.keyword == Keyword::NoKeyword
                && w.quote_style.is_none()
                && w.value.eq_ignore_ascii_case("MEANS")
        }
        _ => false,
    }
}

/// Is the next token pair the (case-insensitive, unquoted) words
/// `RELEVANCE TO`?
///
/// The two-token guard is what keeps `relevance` an ordinary identifier:
/// `SELECT score relevance FROM t` still parses as a trailing alias, because
/// nothing follows it with `TO`. `MEANS` cannot afford the same trick — it
/// takes a single operand — which is why that word really is reserved.
fn peek_is_relevance_to(parser: &Parser) -> bool {
    peek_is_word(parser, "RELEVANCE") && peek_nth_is_word(parser, 1, "TO")
}

/// Parse `EXTRACT ( field TYPE 'doc' FROM source )` into a
/// `sem_extract_inline(source, 'field', '<typespec>', 'doc')` call. Errors
/// (including a plain `EXTRACT(part FROM ts)`) make `maybe_parse` backtrack.
fn parse_inline_extract(parser: &mut Parser) -> Result<Expr, ParserError> {
    parser.next_token(); // EXTRACT
    parser.expect_token(&Token::LParen)?;
    let field = parser.parse_identifier()?;
    // The field type has no registry here, so serialize it to its canonical
    // string; the optimizer parses it back with the shared DDL parser.
    let field_type =
        parse_field_type(parser).map_err(|e| ParserError::ParserError(e.to_string()))?;
    let doc = match parser.next_token().token {
        Token::SingleQuotedString(s) => s,
        other => {
            return Err(ParserError::ParserError(format!(
                "expected a single-quoted doc string in EXTRACT, got {other}"
            )));
        }
    };
    if !parser.parse_keyword(Keyword::FROM) {
        return Err(ParserError::ParserError(
            "expected FROM in EXTRACT(field TYPE 'doc' FROM source)".to_owned(),
        ));
    }
    let source = parser.parse_expr()?;
    parser.expect_token(&Token::RParen)?;

    Ok(marker_call(
        SEM_EXTRACT_INLINE_UDF_NAME,
        vec![
            source,
            string_literal(field.value),
            string_literal(field_type.to_string()),
            string_literal(doc),
        ],
    ))
}

/// A plain `udf(args..)` call — how every piece of semcast surface syntax
/// leaves the parser.
fn marker_call(udf: &str, args: Vec<Expr>) -> Expr {
    Expr::Function(Function {
        name: ObjectName::from(vec![Ident::new(udf)]),
        uses_odbc_syntax: false,
        parameters: FunctionArguments::None,
        args: FunctionArguments::List(FunctionArgumentList {
            duplicate_treatment: None,
            args: args
                .into_iter()
                .map(|arg| FunctionArg::Unnamed(FunctionArgExpr::Expr(arg)))
                .collect(),
            clauses: vec![],
        }),
        filter: None,
        null_treatment: None,
        over: None,
        within_group: vec![],
    })
}

/// Parse `MEANING OF <expr> [INTO <k>] [AS <label>]` into a
/// `meaning_of(source[, k][, 'label'])` call.
///
/// `INTO` and `AS` are consumed here rather than left to the surrounding
/// clause because neither is legal where this appears: a `GROUP BY` list
/// takes bare expressions, so nothing downstream would accept them.
/// Errors make `maybe_parse` backtrack, leaving `meaning` an ordinary
/// identifier.
fn parse_meaning_of(parser: &mut Parser) -> Result<Expr, ParserError> {
    parser.next_token(); // MEANING
    parser.next_token(); // OF
    let source = parser.parse_subexpr(prec_value_like())?;

    let mut args = vec![source];
    if parser.parse_keyword(Keyword::INTO) {
        // `INTO 0` parses here and is rejected by the optimizer: an error
        // raised inside `maybe_parse` is swallowed as a backtrack, which
        // would surface as a baffling complaint about the word `OF`.
        let k = parser.parse_literal_uint()?;
        args.push(Expr::Value(
            Value::Number(k.to_string(), false).with_empty_span(),
        ));
    }
    // The label rides along as a literal so the AST pass can bind it in the
    // SELECT list, where a bare `MEANING OF ... AS topic` alias cannot reach.
    if parser.parse_keyword(Keyword::AS) {
        let label = parser.parse_identifier()?;
        if args.len() == 1 {
            args.push(Expr::Value(
                Value::Number(AUTO_K_SENTINEL.to_string(), false).with_empty_span(),
            ));
        }
        args.push(string_literal(label.value));
    }
    Ok(marker_call(MEANING_OF_UDF_NAME, args))
}

/// Stands in for an absent `INTO` when a label forces the argument to exist.
/// Negative because zero is a `k` a user can write, and writing it should be
/// an error rather than silently meaning "choose for me".
pub const AUTO_K_SENTINEL: i64 = -1;

fn prec_value_like() -> u8 {
    // `Dialect::prec_value` needs a dialect; the free functions above only
    // need the numeric value, which is fixed for `Precedence::Like`.
    SemcastDialect::default().prec_value(Precedence::Like)
}

fn string_literal(s: String) -> Expr {
    Expr::Value(Value::SingleQuotedString(s).with_empty_span())
}

impl Dialect for SemcastDialect {
    fn is_delimited_identifier_start(&self, ch: char) -> bool {
        self.generic.is_delimited_identifier_start(ch)
    }

    fn is_identifier_start(&self, ch: char) -> bool {
        self.generic.is_identifier_start(ch)
    }

    fn is_identifier_part(&self, ch: char) -> bool {
        self.generic.is_identifier_part(ch)
    }

    /// `MEANS` and `RELEVANCE TO` bind like `LIKE`: tighter than `NOT` /
    /// `AND` / `OR`, looser than comparisons. `NOT a MEANS 'x' AND b > 1`
    /// groups as `(NOT (a MEANS 'x')) AND (b > 1)`.
    fn get_next_precedence(&self, parser: &Parser) -> Option<Result<u8, ParserError>> {
        if peek_is_means(parser) || peek_is_relevance_to(parser) {
            return Some(Ok(self.prec_value(Precedence::Like)));
        }
        None
    }

    /// Inline `EXTRACT(field TYPE 'doc' FROM source)` — desugared to the
    /// `sem_extract_inline` marker. Consulted before the standard prefix
    /// parser, so it must not shadow SQL's `EXTRACT(YEAR FROM ts)`: the semcast
    /// grammar is attempted under `maybe_parse` (which backtracks), and on any
    /// mismatch we return `None` to let the default `EXTRACT` path run.
    fn parse_prefix(&self, parser: &mut Parser) -> Option<Result<Expr, ParserError>> {
        if peek_is_word(parser, "MEANING") && peek_nth_is_word(parser, 1, "OF") {
            return match parser.maybe_parse(parse_meaning_of) {
                Ok(Some(expr)) => Some(Ok(expr)),
                Ok(None) => None,
                Err(e) => Some(Err(e)),
            };
        }
        if !peek_is_word(parser, "EXTRACT") {
            return None;
        }
        match parser.maybe_parse(parse_inline_extract) {
            Ok(Some(expr)) => Some(Ok(expr)),
            // Not the semcast grammar — fall back to standard EXTRACT parsing.
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }

    fn parse_infix(
        &self,
        parser: &mut Parser,
        expr: &Expr,
        _precedence: u8,
    ) -> Option<Result<Expr, ParserError>> {
        // Both operators desugar to a two-argument marker UDF the optimizer
        // rewrites; only the keyword shape and the target name differ.
        let udf = if peek_is_means(parser) {
            parser.next_token(); // MEANS
            MEANS_UDF_NAME
        } else if peek_is_relevance_to(parser) {
            parser.next_token(); // RELEVANCE
            parser.next_token(); // TO
            RELEVANCE_UDF_NAME
        } else {
            return None;
        };

        // Parse the right-hand side at the operator's own precedence.
        let rhs = match parser.parse_subexpr(self.prec_value(Precedence::Like)) {
            Ok(rhs) => rhs,
            Err(e) => return Some(Err(e)),
        };
        Some(Ok(marker_call(udf, vec![expr.clone(), rhs])))
    }

    delegate_flags!(
        supports_unicode_string_literal,
        supports_partition_by_after_order_by,
        supports_array_join_syntax,
        supports_group_by_expr,
        supports_group_by_with_modifier,
        supports_left_associative_joins_without_parens,
        supports_connect_by,
        supports_match_recognize,
        supports_pipe_operator,
        supports_start_transaction_modifier,
        supports_window_function_null_treatment_arg,
        supports_dictionary_syntax,
        supports_window_clause_named_window_reference,
        supports_parenthesized_set_variables,
        supports_select_wildcard_except,
        support_map_literal_syntax,
        allow_extract_custom,
        allow_extract_single_quotes,
        supports_extract_comma_syntax,
        supports_create_view_comment_syntax,
        supports_parens_around_table_factor,
        supports_values_as_table_factor,
        supports_create_index_with_clause,
        supports_explain_with_utility_options,
        supports_limit_comma,
        supports_update_order_by,
        supports_from_first_select,
        supports_projection_trailing_commas,
        supports_asc_desc_in_column_definition,
        supports_try_convert,
        supports_bitwise_shift_operators,
        supports_comment_on,
        supports_load_extension,
        supports_named_fn_args_with_assignment_operator,
        supports_struct_literal,
        supports_empty_projections,
        supports_nested_comments,
        supports_multiline_comment_hints,
        supports_user_host_grantee,
        supports_string_escape_constant,
        supports_array_typedef_with_brackets,
        supports_match_against,
        supports_set_names,
        supports_comma_separated_set_assignments,
        supports_filter_during_aggregation,
        supports_select_wildcard_exclude,
        supports_data_type_signed_suffix,
        supports_interval_options,
        supports_quote_delimited_string,
        supports_select_wildcard_replace,
        supports_select_wildcard_ilike,
        supports_select_wildcard_rename,
        supports_optimize_table,
        supports_install,
        supports_detach,
        supports_prewhere,
        supports_with_fill,
        supports_limit_by,
        supports_interpolate,
        supports_settings,
        supports_select_format,
        supports_comment_optimizer_hint,
        supports_constraint_keyword_without_name,
        supports_key_column_option,
        supports_comma_separated_trim,
        supports_cte_without_as,
        supports_select_item_multi_column_alias,
        supports_xml_expressions,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(sql: &str) -> String {
        let stmts = Parser::parse_sql(&SemcastDialect::default(), sql).unwrap();
        assert_eq!(stmts.len(), 1);
        stmts[0].to_string()
    }

    #[test]
    fn means_desugars_to_udf_call() {
        assert_eq!(
            parse("SELECT * FROM meetings WHERE notes MEANS 'action items were assigned'"),
            "SELECT * FROM meetings WHERE means(notes, 'action items were assigned')"
        );
    }

    #[test]
    fn means_is_case_insensitive() {
        assert_eq!(
            parse("SELECT * FROM t WHERE body means 'urgent'"),
            "SELECT * FROM t WHERE means(body, 'urgent')"
        );
    }

    #[test]
    fn means_binds_tighter_than_not_and_or() {
        assert_eq!(
            parse("SELECT * FROM t WHERE NOT a MEANS 'x' AND b MEANS 'y' OR c = 1"),
            "SELECT * FROM t WHERE NOT means(a, 'x') AND means(b, 'y') OR c = 1"
        );
    }

    #[test]
    fn means_condition_can_be_any_expression() {
        assert_eq!(
            parse("SELECT * FROM t WHERE body MEANS 'about ' || topic"),
            "SELECT * FROM t WHERE means(body, 'about ' || topic)"
        );
    }

    #[test]
    fn quoted_means_stays_an_identifier() {
        assert_eq!(
            parse(r#"SELECT x "means" FROM t"#),
            r#"SELECT x AS "means" FROM t"#
        );
    }

    #[test]
    fn plain_sql_still_parses() {
        assert_eq!(
            parse("SELECT a, count(*) FROM t WHERE a LIKE 'x%' GROUP BY 1 + 1"),
            "SELECT a, count(*) FROM t WHERE a LIKE 'x%' GROUP BY 1 + 1"
        );
    }

    #[test]
    fn inline_extract_desugars_to_sem_extract_inline() {
        assert_eq!(
            parse("SELECT EXTRACT(products TEXT[] 'product names' FROM transcript) FROM meetings"),
            "SELECT sem_extract_inline(transcript, 'products', 'TEXT[]', 'product names') \
             FROM meetings"
        );
    }

    #[test]
    fn inline_extract_carries_oneof_typespec() {
        assert_eq!(
            parse("SELECT EXTRACT(stage ONEOF(none, shipped) 'the stage' FROM t) FROM m"),
            "SELECT sem_extract_inline(t, 'stage', 'ONEOF(none,shipped)', 'the stage') FROM m"
        );
    }

    #[test]
    fn relevance_to_desugars_to_udf_call() {
        assert_eq!(
            parse("SELECT id FROM meetings ORDER BY transcript RELEVANCE TO 'offline sync'"),
            "SELECT id FROM meetings ORDER BY relevance(transcript, 'offline sync')"
        );
    }

    #[test]
    fn relevance_to_is_case_insensitive() {
        assert_eq!(
            parse("SELECT id FROM t ORDER BY body relevance to 'urgent' LIMIT 5"),
            "SELECT id FROM t ORDER BY relevance(body, 'urgent') LIMIT 5"
        );
    }

    #[test]
    fn relevance_without_to_stays_an_identifier() {
        // The two-token guard: `relevance` is only an operator before `TO`,
        // so it still works as a trailing alias and as a column name.
        assert_eq!(
            parse("SELECT score relevance FROM t"),
            "SELECT score AS relevance FROM t"
        );
        assert_eq!(
            parse("SELECT relevance FROM t ORDER BY relevance"),
            "SELECT relevance FROM t ORDER BY relevance"
        );
    }

    #[test]
    fn relevance_to_binds_like_means() {
        assert_eq!(
            parse("SELECT * FROM t WHERE a MEANS 'x' ORDER BY b RELEVANCE TO 'y' DESC LIMIT 3"),
            "SELECT * FROM t WHERE means(a, 'x') ORDER BY relevance(b, 'y') DESC LIMIT 3"
        );
    }

    #[test]
    fn standard_extract_still_parses() {
        assert_eq!(
            parse("SELECT EXTRACT(YEAR FROM ts) FROM t"),
            "SELECT EXTRACT(YEAR FROM ts) FROM t"
        );
    }

    #[tokio::test]
    async fn means_executes_end_to_end() {
        use std::sync::Arc;

        use datafusion::arrow::array::AsArray;

        use crate::model::MockModel;
        use crate::semcast_context;

        let ctx = semcast_context(Arc::new(MockModel::answering_yes_to(["offline sync"])));
        ctx.sql(
            "CREATE TABLE notes AS SELECT * FROM (VALUES
                 (1, 'we shipped offline sync'),
                 (2, 'nothing notable happened')
             ) AS t(id, body)",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

        let batches = crate::sql(
            &ctx,
            "SELECT id FROM notes WHERE body MEANS 'syncs offline'",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

        let ids: Vec<i64> = batches
            .iter()
            .flat_map(|b| {
                b.column(0)
                    .as_primitive::<datafusion::arrow::datatypes::Int64Type>()
                    .values()
                    .to_vec()
            })
            .collect();
        assert_eq!(ids, vec![1]);
    }
}
